//! Ephemeral local GPU worker (`uv run python-worker/forge_gpu_worker.py separate ...`).
//!
//! The worker runs with a wall-clock timeout. On timeout the whole process tree is killed and
//! the pipes are abandoned, so a hung model download or CUDA stall cannot block the caller.

use std::ffi::OsString;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use forge_core::{ForgeError, Result, StemBundle};

/// Location of the worker script inside the repository.
const SCRIPT_REL: &str = "python-worker/forge_gpu_worker.py";
/// Longest stderr excerpt carried into an error message.
const STDERR_EXCERPT: usize = 600;

/// The program and leading arguments that launch the local GPU worker. The task arguments
/// (`separate --input ... --output-dir ...`) are appended to `args`.
#[derive(Debug, Clone)]
pub struct LocalWorkerCommand {
    pub program: OsString,
    pub args: Vec<OsString>,
}

impl LocalWorkerCommand {
    /// `uv run --no-sync --project <python-worker> <script>`. `--no-sync` stops `uv` from
    /// installing the multi-gigabyte model environment on a machine that never set it up.
    pub fn uv(script: &Path) -> Self {
        let project = script.parent().unwrap_or(Path::new("."));
        Self {
            program: "uv".into(),
            args: vec![
                "run".into(),
                "--no-sync".into(),
                "--project".into(),
                project.into(),
                script.into(),
            ],
        }
    }
}

/// Finds the worker script: `override_path` (from `FORGE_GPU_WORKER_SCRIPT`) when given,
/// otherwise the first ancestor of the executable's directory that contains
/// `python-worker/forge_gpu_worker.py`. The process working directory is never consulted.
pub fn resolve_worker_script(
    override_path: Option<PathBuf>,
    exe: Option<&Path>,
) -> Option<PathBuf> {
    if let Some(path) = override_path {
        return path.is_file().then_some(path);
    }
    exe?.parent()?
        .ancestors()
        .map(|dir| dir.join(SCRIPT_REL))
        .find(|candidate| candidate.is_file())
}

/// Runs `separate` on the local worker and parses the `StemBundle` it prints.
pub(crate) fn separate(
    worker: &LocalWorkerCommand,
    input_path: &Path,
    output_dir: &Path,
    timeout: Duration,
) -> Result<StemBundle> {
    let mut cmd = Command::new(&worker.program);
    cmd.args(&worker.args)
        .arg("separate")
        .arg("--input")
        .arg(input_path)
        .arg("--output-dir")
        .arg(output_dir);
    let out = run_with_timeout(cmd, timeout)?;
    if !out.status.success() {
        return Err(ForgeError::WorkerError(format!(
            "GPU worker exited with {}: {}",
            out.status,
            stderr_excerpt(&out.stderr)
        )));
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let last_line = stdout
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("");
    serde_json::from_str(last_line).map_err(|e| {
        ForgeError::WorkerError(format!(
            "GPU worker printed no stem JSON ({e}): {last_line}"
        ))
    })
}

/// Exit status and captured output of a finished worker process.
pub(crate) struct WorkerOutput {
    pub status: ExitStatus,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

/// Spawns `cmd` and waits at most `timeout`. On timeout the process tree is killed and the
/// reader threads are left to finish on their own (a surviving grandchild may hold the pipes).
pub(crate) fn run_with_timeout(mut cmd: Command, timeout: Duration) -> Result<WorkerOutput> {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    own_process_group(&mut cmd);
    let program = cmd.get_program().to_string_lossy().into_owned();
    let mut child = cmd.spawn().map_err(|e| {
        ForgeError::WorkerError(format!("failed to launch GPU worker '{program}': {e}"))
    })?;
    let stdout = drain(child.stdout.take());
    let stderr = drain(child.stderr.take());
    let deadline = Instant::now() + timeout;

    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(WorkerOutput {
                status,
                stdout: stdout.join().unwrap_or_default(),
                stderr: stderr.join().unwrap_or_default(),
            });
        }
        if Instant::now() >= deadline {
            kill_tree(&mut child);
            return Err(ForgeError::WorkerError(format!(
                "GPU worker timed out after {:.1} s and was killed",
                timeout.as_secs_f64()
            )));
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn drain<R: Read + Send + 'static>(pipe: Option<R>) -> JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut pipe) = pipe {
            let _ = pipe.read_to_end(&mut buf);
        }
        buf
    })
}

fn stderr_excerpt(stderr: &[u8]) -> String {
    let text = String::from_utf8_lossy(stderr);
    let text = text.trim();
    let start = text
        .char_indices()
        .rev()
        .nth(STDERR_EXCERPT)
        .map_or(0, |(i, _)| i);
    if text.is_empty() {
        "(no stderr)".to_string()
    } else {
        text[start..].to_string()
    }
}

/// Puts the worker in its own process group so a timeout can kill `uv` and its Python child.
#[cfg(unix)]
fn own_process_group(cmd: &mut Command) {
    use std::os::unix::process::CommandExt as _;
    cmd.process_group(0);
}

#[cfg(not(unix))]
fn own_process_group(_cmd: &mut Command) {}

#[cfg(unix)]
fn kill_tree(child: &mut Child) {
    let _ = Command::new("kill")
        .args(["-KILL", &format!("-{}", child.id())])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    let _ = child.kill();
    let _ = child.wait();
}

#[cfg(windows)]
fn kill_tree(child: &mut Child) {
    let _ = Command::new("taskkill")
        .args(["/F", "/T", "/PID", &child.id().to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    let _ = child.kill();
    let _ = child.wait();
}

#[cfg(not(any(unix, windows)))]
fn kill_tree(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}
