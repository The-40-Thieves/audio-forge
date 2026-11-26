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
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tokio::task::AbortHandle;

const MAX_RETAINED_TASKS: usize = 64;
const MAX_ACTIVE_TASKS: usize = 16;
const TASK_TTL_MS: u64 = 600_000;
const TASK_POLL_INTERVAL_MS: u64 = 500;

const EXEC_QUEUED: u8 = 0;
const EXEC_RUNNING: u8 = 1;
const EXEC_CANCELLED: u8 = 2;

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
    exec_state: Arc<AtomicU8>,
}

fn build_seed_task(task_id: &str, name: &str) -> Task {
    let now = utc_now_iso8601();
    Task::new(task_id, TaskStatus::Working, &now, &now)
        .with_status_message(format!("Running {name}..."))
        .with_ttl_ms(TASK_TTL_MS)
        .with_poll_interval_ms(TASK_POLL_INTERVAL_MS)
}

/// Thread-safe bounded registry tracking active and completed SEP-2663 MCP tasks.
#[derive(Clone, Default)]
pub(crate) struct TaskRegistry {
    next_id: Arc<AtomicU64>,
    records: Arc<RwLock<HashMap<String, TaskRecord>>>,
}

impl TaskRegistry {
    async fn reserve_slot(&self, task_id: String, record: TaskRecord) -> Result<(), McpError> {
        let mut map = self.records.write().await;
        let active = map
            .values()
            .filter(|r| !r.detailed.status().is_terminal())
            .count();
        if active >= MAX_ACTIVE_TASKS {
            return Err(McpError::internal_error(
                format!(
                    "Too many active background tasks (limit {MAX_ACTIVE_TASKS}); wait for running tasks to complete"
                ),
                None,
            ));
        }
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
        Ok(())
    }

    pub(crate) async fn spawn_tool_task(
        &self,
        name: String,
        args: Value,
        sandbox: Arc<PathSandbox>,
    ) -> Result<CreateTaskResult, McpError> {
        let seq = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        let task_id = format!("forge-task-{seq}");
        let seed_task = build_seed_task(&task_id, &name);
        let exec_state = Arc::new(AtomicU8::new(EXEC_QUEUED));
        let record = TaskRecord {
            detailed: DetailedTask::new(seed_task.clone(), TaskPayload::Working),
            created_instant: Instant::now(),
            abort: None,
            exec_state: Arc::clone(&exec_state),
        };
        self.reserve_slot(task_id.clone(), record).await?;

        let registry = self.clone();
        let task_id_bg = task_id.clone();
        let tool_label = name.clone();
        let handle = tokio::spawn(async move {
            let outcome = tokio::task::spawn_blocking(move || {
                if exec_state
                    .compare_exchange(
                        EXEC_QUEUED,
                        EXEC_RUNNING,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_err()
                {
                    return None;
                }
                Some(handlers::call_tool_result(&name, &args, &sandbox))
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
        Ok(CreateTaskResult::new(seed_task))
    }

    async fn finish_task(
        &self,
        task_id: &str,
        tool_name: &str,
        outcome: Result<Option<CallToolResult>, tokio::task::JoinError>,
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
            Ok(Some(res)) => {
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
            Ok(None) => {
                task.status_message = Some("Cancelled by client".to_string());
                rec.detailed = DetailedTask::new(task, TaskPayload::Cancelled);
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
        if rec.detailed.status().is_terminal() {
            return Err(McpError::invalid_params(
                format!(
                    "Task '{task_id}' is already in terminal state {:?}",
                    rec.detailed.status()
                ),
                None,
            ));
        }
        if rec
            .exec_state
            .compare_exchange(
                EXEC_QUEUED,
                EXEC_CANCELLED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return Err(McpError::invalid_params(
                "Task is already executing a non-interruptible blocking tool and cannot be safely cancelled; wait for completion",
                None,
            ));
        }
        if let Some(abort) = rec.abort.take() {
            abort.abort();
        }
        let mut task = rec.detailed.task.clone();
        task.last_updated_at = utc_now_iso8601();
        task.status_message = Some("Cancelled by client".to_string());
        rec.detailed = DetailedTask::new(task, TaskPayload::Cancelled);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn seed_record(registry: &TaskRegistry, task_id: &str, state: u8) -> Arc<AtomicU8> {
        let exec_state = Arc::new(AtomicU8::new(state));
        let seed_task = build_seed_task(task_id, "test_tool");
        registry
            .reserve_slot(
                task_id.to_string(),
                TaskRecord {
                    detailed: DetailedTask::new(seed_task, TaskPayload::Working),
                    created_instant: Instant::now(),
                    abort: None,
                    exec_state: Arc::clone(&exec_state),
                },
            )
            .await
            .expect("reserve test slot");
        exec_state
    }

    #[tokio::test]
    async fn active_task_cap_rejects_excess_tasks() {
        let registry = TaskRegistry::default();
        for idx in 0..MAX_ACTIVE_TASKS {
            seed_record(&registry, &format!("seed-{idx}"), EXEC_RUNNING).await;
        }
        let tmp = tempfile::tempdir().expect("tempdir");
        let sb = Arc::new(PathSandbox::new([tmp.path().to_path_buf()]).expect("sandbox"));
        let Err(err) = registry
            .spawn_tool_task("forge_status".to_string(), json!({}), sb)
            .await
        else {
            panic!("expected spawn_tool_task to reject once MAX_ACTIVE_TASKS are active");
        };
        assert!(err.message.contains("Too many active background tasks"));
    }

    #[tokio::test]
    async fn cancel_task_handles_queued_running_and_terminal_states() {
        let registry = TaskRegistry::default();
        let q_state = seed_record(&registry, "queued-task", EXEC_QUEUED).await;
        registry
            .cancel_task("queued-task")
            .await
            .expect("queued task cancels cleanly");
        assert_eq!(q_state.load(Ordering::Acquire), EXEC_CANCELLED);
        assert_eq!(
            registry
                .get_task("queued-task")
                .await
                .unwrap()
                .task
                .status(),
            TaskStatus::Cancelled
        );
        assert!(registry.cancel_task("queued-task").await.is_err());

        seed_record(&registry, "running-task", EXEC_RUNNING).await;
        let Err(running_err) = registry.cancel_task("running-task").await else {
            panic!("running non-interruptible blocking task must refuse cancellation");
        };
        assert!(running_err.message.contains("cannot be safely cancelled"));
        assert_eq!(
            registry
                .get_task("running-task")
                .await
                .unwrap()
                .task
                .status(),
            TaskStatus::Working
        );
    }
}
