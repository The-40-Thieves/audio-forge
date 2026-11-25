//! Bounded in-memory SEP-2663 task registry for asynchronous `forge-mcp` tool execution.

use crate::handlers;
use crate::sandbox::PathSandbox;
use rmcp::ErrorData as McpError;
use rmcp::model::{
    CallToolResult, CreateTaskResult, DetailedTask, GetTaskResult, JsonObject, Task, TaskPayload,
    TaskStatus,
};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tokio::task::AbortHandle;

const MAX_RETAINED_TASKS: usize = 64;
const TASK_TTL_MS: u64 = 600_000;
const TASK_POLL_INTERVAL_MS: u64 = 500;

fn utc_now_iso8601() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64);
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    let hour = sod / 3600;
    let min = (sod % 3600) / 60;
    let sec = sod % 60;

    // Howard Hinnant civil-from-days algorithm in UTC.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec:02}Z")
}

struct TaskRecord {
    detailed: DetailedTask,
    created_instant: Instant,
    abort: Option<AbortHandle>,
}

/// Thread-safe bounded registry tracking active and completed SEP-2663 MCP tasks.
#[derive(Clone, Default)]
pub(crate) struct TaskRegistry {
    next_id: Arc<AtomicU64>,
    records: Arc<RwLock<HashMap<String, TaskRecord>>>,
}

impl TaskRegistry {
    async fn insert_bounded(&self, task_id: String, record: TaskRecord) {
        let mut map = self.records.write().await;
        if map.len() >= MAX_RETAINED_TASKS
            && let Some(evict_id) = map
                .iter()
                .filter(|(_, r)| r.detailed.status().is_terminal())
                .min_by_key(|(_, r)| r.created_instant)
                .map(|(id, _)| id.clone())
        {
            map.remove(&evict_id);
        }
        map.insert(task_id, record);
    }

    pub(crate) async fn spawn_tool_task(
        &self,
        name: String,
        args: Value,
        sandbox: Arc<PathSandbox>,
    ) -> CreateTaskResult {
        let seq = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        let task_id = format!("forge-task-{seq}");
        let now = utc_now_iso8601();
        let seed_task = Task::new(&task_id, TaskStatus::Working, &now, &now)
            .with_status_message(format!("Running {name}..."))
            .with_ttl_ms(TASK_TTL_MS)
            .with_poll_interval_ms(TASK_POLL_INTERVAL_MS);

        let detailed = DetailedTask::new(seed_task.clone(), TaskPayload::Working);
        self.insert_bounded(
            task_id.clone(),
            TaskRecord {
                detailed,
                created_instant: Instant::now(),
                abort: None,
            },
        )
        .await;

        let registry = self.clone();
        let task_id_bg = task_id.clone();
        let tool_label = name.clone();
        let handle = tokio::spawn(async move {
            let outcome = tokio::task::spawn_blocking(move || {
                handlers::call_tool_result(&name, &args, &sandbox)
            })
            .await;
            registry
                .finish_task(&task_id_bg, &tool_label, outcome)
                .await;
        });

        let mut map = self.records.write().await;
        if let Some(rec) = map.get_mut(&task_id)
            && !rec.detailed.status().is_terminal()
        {
            rec.abort = Some(handle.abort_handle());
        }
        CreateTaskResult::new(seed_task)
    }

    async fn finish_task(
        &self,
        task_id: &str,
        tool_name: &str,
        outcome: Result<CallToolResult, tokio::task::JoinError>,
    ) {
        let mut map = self.records.write().await;
        let Some(rec) = map.get_mut(task_id) else {
            return;
        };
        if matches!(rec.detailed.status(), TaskStatus::Cancelled) {
            return;
        }
        let now = utc_now_iso8601();
        let mut task = rec.detailed.task.clone();
        task.last_updated_at = now;
        rec.abort = None;
        match outcome {
            Ok(res) => {
                let is_err = res.is_error.unwrap_or(false);
                task.status_message = Some(if is_err {
                    format!("{tool_name} finished with error")
                } else {
                    format!("{tool_name} completed")
                });
                let obj: JsonObject = match serde_json::to_value(&res) {
                    Ok(Value::Object(m)) => m,
                    _ => JsonObject::new(),
                };
                rec.detailed = DetailedTask::new(task, TaskPayload::Completed { result: obj });
            }
            Err(join_err) => {
                task.status_message = Some(format!("{tool_name} failed: {join_err}"));
                let err_obj: JsonObject = match json!({
                    "code": -32603,
                    "message": format!("Tool worker task failed: {join_err}"),
                }) {
                    Value::Object(m) => m,
                    _ => JsonObject::new(),
                };
                rec.detailed = DetailedTask::new(task, TaskPayload::Failed { error: err_obj });
            }
        }
    }

    pub(crate) async fn get_task(&self, task_id: &str) -> Result<GetTaskResult, McpError> {
        let map = self.records.read().await;
        let rec = map.get(task_id).ok_or_else(|| {
            McpError::invalid_params(format!("Unknown task_id: '{task_id}'"), None)
        })?;
        Ok(GetTaskResult::new(rec.detailed.clone()))
    }

    pub(crate) async fn cancel_task(&self, task_id: &str) -> Result<(), McpError> {
        let mut map = self.records.write().await;
        let rec = map.get_mut(task_id).ok_or_else(|| {
            McpError::invalid_params(format!("Unknown task_id: '{task_id}'"), None)
        })?;
        if !rec.detailed.status().is_terminal() {
            if let Some(abort) = rec.abort.take() {
                abort.abort();
            }
            let mut task = rec.detailed.task.clone();
            task.last_updated_at = utc_now_iso8601();
            task.status_message = Some("Cancelled by client".to_string());
            rec.detailed = DetailedTask::new(task, TaskPayload::Cancelled);
        }
        Ok(())
    }
}
