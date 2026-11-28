//! Ephemeral local GPU worker (`uv run python-worker/forge_gpu_worker.py separate ...`).
//!
//! The worker runs with a wall-clock timeout. On timeout the whole process tree is killed and
//! the pipes are abandoned, so a hung model download or CUDA stall cannot block the caller.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use forge_core::{ForgeError, Result, StemBundle};
use forge_dsp::BeatGridReport;
use serde::{Deserialize, Serialize};

/// Location of the worker script inside the repository.
const SCRIPT_REL: &str = "python-worker/forge_gpu_worker.py";
/// Longest stderr excerpt carried into an error message.
const STDERR_EXCERPT: usize = 600;
/// Maximum allowed wall-clock timeout for a local GPU worker subprocess (24 hours,
/// accommodating long multi-track separation jobs while preventing `Instant` overflow).
pub const MAX_WORKER_TIMEOUT: Duration = Duration::from_secs(86_400);

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

/// Repository root of the source tree this binary was built from, fixed at compile time.
pub const BUILD_SOURCE_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

/// Finds the worker script: `override_path` (from `FORGE_GPU_WORKER_SCRIPT`) when given,
/// otherwise `python-worker/forge_gpu_worker.py` under `source_root` (normally
/// [`BUILD_SOURCE_ROOT`]). Neither the working directory nor the executable's location, which a
/// symlink or copy can change, is consulted: the script either comes from explicit
/// configuration or from the tree the binary was compiled from.
pub fn resolve_worker_script(
    override_path: Option<PathBuf>,
    source_root: Option<&Path>,
) -> Option<PathBuf> {
    if let Some(path) = override_path {
        return path.is_file().then_some(path);
    }
    let candidate = source_root?.join(SCRIPT_REL);
    candidate.is_file().then_some(candidate)
}

/// Resolves the packaged `python-worker/forge_gpu_worker.py` sibling directory shipped alongside
/// release binaries (`bin/forge*` + `python-worker/{forge_gpu_worker.py,pyproject.toml,uv.lock}`).
pub fn resolve_packaged_worker_script(exe_path: &Path) -> Option<PathBuf> {
    let resolved_exe = std::fs::canonicalize(exe_path).ok()?;
    let bin_dir = resolved_exe.parent()?;
    let root_dir = bin_dir.parent()?;
    let worker_dir = root_dir.join("python-worker");
    let script = worker_dir.join("forge_gpu_worker.py");
    let manifest = worker_dir.join("pyproject.toml");
    let lock = worker_dir.join("uv.lock");
    (script.is_file() && manifest.is_file() && lock.is_file()).then_some(script)
}

fn worker_io_cmd(
    worker: &LocalWorkerCommand,
    subcmd: &str,
    input_path: &Path,
    output_dir: &Path,
) -> Command {
    let mut cmd = Command::new(&worker.program);
    cmd.args(&worker.args)
        .arg(subcmd)
        .arg("--input")
        .arg(input_path)
        .arg("--output-dir")
        .arg(output_dir);
    cmd
}

#[derive(Debug, Deserialize)]
struct SeparateWorkerResponse {
    #[serde(flatten)]
    bundle: StemBundle,
    #[serde(default)]
    warnings: Vec<String>,
}

/// Runs `separate` on the local worker and parses the `StemBundle` + warnings it prints.
pub(crate) fn separate(
    worker: &LocalWorkerCommand,
    input_path: &Path,
    output_dir: &Path,
    stems: u8,
    config: &crate::routing::SeparationConfig,
) -> Result<(StemBundle, Vec<String>)> {
    let mut cmd = worker_io_cmd(worker, "separate", input_path, output_dir);
    cmd.arg("--stems")
        .arg(stems.to_string())
        .arg("--quality")
        .arg(config.quality.as_str());
    if !config.dereverb {
        cmd.arg("--no-dereverb");
    }
    for &(enabled, flag) in &[
        (config.drum_sep, "--drum-sep"),
        (config.split_lead_backing, "--split-lead-backing"),
        (config.split_male_female, "--split-male-female"),
        (config.split_guitar, "--split-guitar"),
        (config.split_orchestral, "--split-orchestral"),
        (config.split_crowd, "--split-crowd"),
        (config.restore, "--restore"),
        (config.debleed, "--debleed"),
        (config.de_aspiration, "--de-aspiration"),
        (config.dereverb_echo, "--dereverb-echo"),
    ] {
        if enabled {
            cmd.arg(flag);
        }
    }
    let parsed: SeparateWorkerResponse =
        run_worker_json(cmd, config.local_timeout, "GPU worker", "stem")?;
    Ok((parsed.bundle, parsed.warnings))
}

fn run_worker_json<T: serde::de::DeserializeOwned>(
    cmd: Command,
    timeout: Duration,
    exit_prefix: &str,
    json_kind: &str,
) -> Result<T> {
    let out = run_with_timeout(cmd, timeout)?;
    if !out.status.success() {
        return Err(ForgeError::WorkerError(format!(
            "{exit_prefix} exited with {}: {}",
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
            "GPU worker printed no {json_kind} JSON ({e}): {last_line}"
        ))
    })
}

/// Output of `forge_gpu_worker.py denoise`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NeuralDenoiseResult {
    pub denoised: PathBuf,
    pub backend: String,
}

/// Output of `forge_gpu_worker.py transcribe`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MidiTranscribeResult {
    pub midi: PathBuf,
}

/// Output of `forge_gpu_worker.py restore` / Colab `restore`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NeuralRestoreResult {
    pub restored: PathBuf,
    pub backend: String,
    #[serde(default)]
    pub warnings: Vec<String>,
}

/// Output of `forge_gpu_worker.py vocal-rescue` / Colab `vocal_rescue`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VocalRescueResult {
    pub rescued: PathBuf,
    pub backend: String,
    #[serde(default)]
    pub stages: Vec<String>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

/// Perceptual & acoustic quality scores produced by `forge_gpu_worker.py evaluate` or acoustic fallback.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QualityEvalMetrics {
    pub production_quality: Option<f64>,
    pub production_complexity: Option<f64>,
    pub content_enjoyment: Option<f64>,
    pub content_usefulness: Option<f64>,
    pub singmos: Option<f64>,
    pub crest_factor_db: f64,
    pub hf_ratio_db: f64,
    pub clipping_ratio: f64,
    pub composite_score: f64,
}

/// Before/after perceptual quality evaluation report (`Audiobox Aesthetics`, `SingMOS`, or acoustic proxy).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QualityEvalResult {
    pub backend: String,
    pub metrics: QualityEvalMetrics,
    #[serde(default)]
    pub reference_metrics: Option<QualityEvalMetrics>,
    #[serde(default)]
    pub delta: Option<BTreeMap<String, f64>>,
    #[serde(default)]
    pub improved: Option<bool>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

/// Single frame emitted by `forge_gpu_worker.py f0-track`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct F0TrackFrame {
    pub time_secs: f64,
    pub f0_hz: f64,
    pub voiced: bool,
    pub confidence: f64,
}

/// Neural/autocorrelation fundamental-frequency (`f0-track`) report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct F0TrackResult {
    pub backend: String,
    pub sample_rate: u32,
    pub hop_secs: f64,
    pub voiced_ratio: f64,
    pub median_f0_hz: Option<f64>,
    pub frames: Vec<F0TrackFrame>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

/// Resolves the worker script path from `FORGE_GPU_WORKER_SCRIPT`, `BUILD_SOURCE_ROOT`, or the
/// packaged sibling `python-worker/` layout next to `exe_path`.
pub(crate) fn resolve_default_worker_script_with(
    override_path: Option<PathBuf>,
    source_root: Option<&Path>,
    exe_path: Option<&Path>,
) -> Option<PathBuf> {
    resolve_worker_script(override_path, source_root)
        .or_else(|| exe_path.and_then(resolve_packaged_worker_script))
}

/// Resolves the default local GPU worker command from `FORGE_GPU_WORKER_SCRIPT`,
/// `BUILD_SOURCE_ROOT`, or the packaged sibling `python-worker/` layout next to the binary.
pub fn resolve_default_local_worker() -> Result<LocalWorkerCommand> {
    let override_path = std::env::var_os("FORGE_GPU_WORKER_SCRIPT").map(PathBuf::from);
    let exe_path = std::env::current_exe().ok();
    let script = resolve_default_worker_script_with(
        override_path,
        Some(Path::new(BUILD_SOURCE_ROOT)),
        exe_path.as_deref(),
    )
    .ok_or_else(|| {
        ForgeError::WorkerError(
            "Could not locate python-worker/forge_gpu_worker.py; set FORGE_GPU_WORKER_SCRIPT"
                .to_string(),
        )
    })?;
    Ok(LocalWorkerCommand::uv(&script))
}

fn worker_in_only_cmd(worker: &LocalWorkerCommand, subcmd: &str, input_path: &Path) -> Command {
    let mut cmd = Command::new(&worker.program);
    cmd.args(&worker.args)
        .arg(subcmd)
        .arg("--input")
        .arg(input_path);
    cmd
}

/// Runs neural denoising (`DeepFilterNet3` / `Mel-Band-RoFormer`) via the local Python GPU worker.
pub fn run_local_neural_denoise(
    worker: &LocalWorkerCommand,
    input_path: &Path,
    output_dir: &Path,
    backend: &str,
    attenuation_limit_db: Option<f64>,
    timeout: Duration,
) -> Result<NeuralDenoiseResult> {
    let mut cmd = worker_io_cmd(worker, "denoise", input_path, output_dir);
    cmd.arg("--backend").arg(backend);
    if let Some(lim) = attenuation_limit_db {
        if !lim.is_finite() || !(0.0..=100.0).contains(&lim) {
            return Err(ForgeError::InvalidParameter(format!(
                "attenuation_limit_db ({lim}) must be finite and in 0..=100"
            )));
        }
        cmd.arg("--attenuation-limit-db").arg(format!("{lim}"));
    }
    run_worker_json(cmd, timeout, "GPU worker denoise", "denoise")
}

/// Runs neural bandwidth/codec restoration (`Apollo` / `FlowHigh` / `AudioSR` / `SonicMaster` / `exciter`).
pub fn run_local_neural_restore(
    worker: &LocalWorkerCommand,
    input_path: &Path,
    output_dir: &Path,
    backend: &str,
    prompt: Option<&str>,
    chunk_secs: Option<f64>,
    timeout: Duration,
) -> Result<NeuralRestoreResult> {
    let mut cmd = worker_io_cmd(worker, "restore", input_path, output_dir);
    cmd.arg("--backend").arg(backend);
    if let Some(p) = prompt.map(str::trim).filter(|s| !s.is_empty()) {
        cmd.arg("--prompt").arg(p);
    }
    if let Some(cs) = chunk_secs {
        if !cs.is_finite() || !(0.5..=120.0).contains(&cs) {
            return Err(ForgeError::InvalidParameter(format!(
                "chunk_secs ({cs}) must be finite and in 0.5..=120.0"
            )));
        }
        cmd.arg("--chunk-secs").arg(format!("{cs}"));
    }
    run_worker_json(cmd, timeout, "GPU worker restore", "restore")
}

/// Runs neural/autocorrelation fundamental-frequency (`F0`) tracking via the local GPU worker.
pub fn run_local_f0_track(
    worker: &LocalWorkerCommand,
    input_path: &Path,
    backend: &str,
    hop_secs: f64,
    timeout: Duration,
) -> Result<F0TrackResult> {
    if !hop_secs.is_finite() || !(0.002..=0.25).contains(&hop_secs) {
        return Err(ForgeError::InvalidParameter(format!(
            "hop_secs ({hop_secs}) must be finite and in 0.002..=0.25"
        )));
    }
    let mut cmd = worker_in_only_cmd(worker, "f0-track", input_path);
    cmd.arg("--backend")
        .arg(backend)
        .arg("--hop-secs")
        .arg(format!("{hop_secs}"));
    run_worker_json(cmd, timeout, "GPU worker f0-track", "f0-track")
}

/// Runs neural vocal rescue / timbre resynthesis (`Seed-VC` / `Resemble Enhance` / `RoFormer`).
#[allow(clippy::too_many_arguments)]
pub fn run_local_vocal_rescue(
    worker: &LocalWorkerCommand,
    input_path: &Path,
    output_dir: &Path,
    backend: &str,
    reference_voice: Option<&Path>,
    dereverb: bool,
    de_aspiration: bool,
    timeout: Duration,
) -> Result<VocalRescueResult> {
    let mut cmd = worker_io_cmd(worker, "vocal-rescue", input_path, output_dir);
    cmd.arg("--backend").arg(backend);
    if let Some(ref_path) = reference_voice {
        cmd.arg("--reference-voice").arg(ref_path);
    }
    if !dereverb {
        cmd.arg("--no-dereverb");
    }
    if !de_aspiration {
        cmd.arg("--no-de-aspiration");
    }
    run_worker_json(cmd, timeout, "GPU worker vocal-rescue", "vocal-rescue")
}

/// Runs neural beat & downbeat tracking (`beat_this` / `librosa` / `energy`) via local Python worker.
pub fn run_local_beat_track(
    worker: &LocalWorkerCommand,
    input_path: &Path,
    backend: &str,
    timeout: Duration,
) -> Result<BeatGridReport> {
    let mut cmd = worker_in_only_cmd(worker, "beat-track", input_path);
    cmd.arg("--backend").arg(backend);
    run_worker_json(cmd, timeout, "GPU worker beat-track", "beat-track")
}

/// Runs perceptual audio quality scoring (`Audiobox Aesthetics` / `SingMOS` / `acoustic-proxy`).
pub fn run_local_evaluate_quality(
    worker: &LocalWorkerCommand,
    input_path: &Path,
    reference_path: Option<&Path>,
    vocal: bool,
    backend: &str,
    timeout: Duration,
) -> Result<QualityEvalResult> {
    let mut cmd = worker_in_only_cmd(worker, "evaluate", input_path);
    cmd.arg("--backend").arg(backend);
    if let Some(ref_path) = reference_path {
        cmd.arg("--reference").arg(ref_path);
    }
    if vocal {
        cmd.arg("--vocal");
    }
    run_worker_json(cmd, timeout, "GPU worker evaluate", "evaluate")
}

/// Runs polyphonic audio-to-MIDI transcription (`basic-pitch`) via the local Python GPU worker.
pub fn run_local_transcribe_midi(
    worker: &LocalWorkerCommand,
    input_path: &Path,
    output_dir: &Path,
    timeout: Duration,
) -> Result<MidiTranscribeResult> {
    let cmd = worker_io_cmd(worker, "transcribe", input_path, output_dir);
    run_worker_json(cmd, timeout, "GPU worker transcribe", "transcribe")
}

/// Exit status and captured output of a finished worker process.
pub(crate) struct WorkerOutput {
    pub status: ExitStatus,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

/// Spawns `cmd` and waits at most `timeout`, including the time spent draining its output. On
/// timeout the process tree is killed and the reader threads are abandoned (a surviving
/// grandchild may hold the pipes open after the worker itself exits).
pub(crate) fn run_with_timeout(mut cmd: Command, timeout: Duration) -> Result<WorkerOutput> {
    if timeout.is_zero() || timeout > MAX_WORKER_TIMEOUT {
        return Err(ForgeError::InvalidParameter(format!(
            "worker timeout ({:.3} s) must be > 0 and <= {} s",
            timeout.as_secs_f64(),
            MAX_WORKER_TIMEOUT.as_secs()
        )));
    }
    let deadline = Instant::now().checked_add(timeout).ok_or_else(|| {
        ForgeError::InvalidParameter("worker timeout overflows Instant".to_string())
    })?;
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
    let timed_out = |child: &mut Child| {
        kill_tree(child);
        ForgeError::WorkerError(format!(
            "GPU worker timed out after {:.1} s and was killed",
            timeout.as_secs_f64()
        ))
    };

    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if Instant::now() >= deadline {
            return Err(timed_out(&mut child));
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    match (collect(&stdout, deadline), collect(&stderr, deadline)) {
        (Some(stdout), Some(stderr)) => Ok(WorkerOutput {
            status,
            stdout,
            stderr,
        }),
        _ => Err(timed_out(&mut child)),
    }
}

const MAX_WORKER_PIPE_BYTES: u64 = 16 * 1024 * 1024;

/// Reads a pipe to its end (bounded to 16 MiB) on a helper thread and sends the bytes back.
fn drain<R: Read + Send + 'static>(pipe: Option<R>) -> Receiver<Vec<u8>> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(pipe) = pipe {
            let _ = pipe.take(MAX_WORKER_PIPE_BYTES).read_to_end(&mut buf);
        }
        let _ = tx.send(buf);
    });
    rx
}

/// Waits for a drained pipe until `deadline`; `None` when it is still held open then.
fn collect(rx: &Receiver<Vec<u8>>, deadline: Instant) -> Option<Vec<u8>> {
    let left = deadline.saturating_duration_since(Instant::now());
    match rx.recv_timeout(left) {
        Ok(buf) => Some(buf),
        Err(RecvTimeoutError::Disconnected) => Some(Vec::new()),
        Err(RecvTimeoutError::Timeout) => None,
    }
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
    // `--` ends option parsing: procps-ng `kill` otherwise reads `-<pgid>` as an option,
    // exits 0 and leaves the group alive.
    let _ = Command::new("kill")
        .args(["-s", "KILL", "--", &format!("-{}", child.id())])
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

#[cfg(test)]
mod client_tests {
    use super::*;

    fn mock_worker_echo(
        stdout_lines: &[&str],
        exit_code: i32,
    ) -> (tempfile::TempDir, LocalWorkerCommand) {
        let dir = tempfile::tempdir().expect("tempdir");
        let out_file = dir.path().join("stdout.txt");
        let mut content = stdout_lines.join("\n");
        if !content.is_empty() {
            content.push('\n');
        }
        std::fs::write(&out_file, content).expect("write mock stdout");
        #[cfg(unix)]
        let cmd = LocalWorkerCommand {
            program: "sh".into(),
            args: vec![
                "-c".into(),
                format!("cat \"$1\"; exit {exit_code}").into(),
                "mock-worker".into(),
                out_file.into_os_string(),
            ],
        };
        #[cfg(windows)]
        let cmd = {
            let bat = dir.path().join("mock.cmd");
            std::fs::write(
                &bat,
                format!(
                    "@echo off\r\ntype \"{}\"\r\nexit /b {exit_code}\r\n",
                    out_file.display()
                ),
            )
            .expect("write mock.cmd");
            LocalWorkerCommand {
                program: "cmd".into(),
                args: vec!["/C".into(), bat.into_os_string()],
            }
        };
        (dir, cmd)
    }

    #[test]
    fn f0_track_client_parses_json_and_validates_hop() {
        let (_dir, worker) = mock_worker_echo(
            &[
                "Tracking F0 with autocorr fallback...",
                r#"{"backend":"autocorr","sample_rate":48000,"hop_secs":0.02,"voiced_ratio":1.0,"median_f0_hz":220.0,"frames":[{"time_secs":0.02,"f0_hz":220.0,"voiced":true,"confidence":0.95}],"warnings":[]}"#,
            ],
            0,
        );
        let err = run_local_f0_track(
            &worker,
            Path::new("vocal.wav"),
            "auto",
            0.0,
            Duration::from_secs(10),
        )
        .expect_err("zero hop must be rejected");
        assert!(err.to_string().contains("hop_secs"), "{err}");

        let res = run_local_f0_track(
            &worker,
            Path::new("vocal.wav"),
            "autocorr",
            0.02,
            Duration::from_secs(10),
        )
        .expect("f0-track json");
        assert_eq!(res.backend, "autocorr");
        assert_eq!(res.sample_rate, 48_000);
        assert_eq!(res.frames.len(), 1);
        assert!((res.frames[0].f0_hz - 220.0).abs() < 1e-6);
    }

    #[test]
    fn denoise_client_parses_json_after_log_lines() {
        let (_dir, worker) = mock_worker_echo(
            &[
                "Loading DeepFilterNet3...",
                r#"{"denoised":"/tmp/out/vocals_denoised.wav","backend":"deepfilternet3"}"#,
            ],
            0,
        );
        let res = run_local_neural_denoise(
            &worker,
            Path::new("vocals.wav"),
            Path::new("/tmp/out"),
            "auto",
            Some(18.0),
            Duration::from_secs(10),
        )
        .expect("denoise json");
        assert_eq!(res.backend, "deepfilternet3");
        assert_eq!(res.denoised, PathBuf::from("/tmp/out/vocals_denoised.wav"));
    }

    #[test]
    fn denoise_client_rejects_invalid_attenuation_and_timeout() {
        let (_dir, worker) = mock_worker_echo(&[], 0);
        for bad in [-1.0, 101.0, f64::NAN, f64::INFINITY] {
            let err = run_local_neural_denoise(
                &worker,
                Path::new("in.wav"),
                Path::new("out"),
                "auto",
                Some(bad),
                Duration::from_secs(10),
            )
            .expect_err("invalid attenuation");
            assert!(err.to_string().contains("attenuation_limit_db"), "{err}");
        }
        for bad_timeout in [Duration::ZERO, Duration::MAX] {
            let err = run_local_neural_denoise(
                &worker,
                Path::new("in.wav"),
                Path::new("out"),
                "auto",
                None,
                bad_timeout,
            )
            .expect_err("invalid timeout");
            assert!(err.to_string().contains("timeout"), "{err}");
        }
    }

    #[test]
    fn denoise_client_reports_nonzero_exit_and_malformed_json() {
        let (_dir1, fail_worker) = mock_worker_echo(&["boom"], 2);
        assert!(
            run_local_neural_denoise(
                &fail_worker,
                Path::new("in.wav"),
                Path::new("out"),
                "auto",
                None,
                Duration::from_secs(10),
            )
            .is_err()
        );

        let (_dir2, bad_json_worker) = mock_worker_echo(&["not-json"], 0);
        let err = run_local_neural_denoise(
            &bad_json_worker,
            Path::new("in.wav"),
            Path::new("out"),
            "auto",
            None,
            Duration::from_secs(10),
        )
        .expect_err("malformed json");
        assert!(err.to_string().contains("denoise JSON"), "{err}");
    }

    #[test]
    fn transcribe_client_parses_json_and_handles_failures() {
        let (_dir1, worker) = mock_worker_echo(
            &[
                "Running basic-pitch...",
                r#"{"midi":"/tmp/out/piano_basic_pitch.mid"}"#,
            ],
            0,
        );
        let res = run_local_transcribe_midi(
            &worker,
            Path::new("piano.wav"),
            Path::new("/tmp/out"),
            Duration::from_secs(10),
        )
        .expect("transcribe json");
        assert_eq!(res.midi, PathBuf::from("/tmp/out/piano_basic_pitch.mid"));

        let (_dir2, bad_worker) = mock_worker_echo(&["not-json"], 0);
        assert!(
            run_local_transcribe_midi(
                &bad_worker,
                Path::new("piano.wav"),
                Path::new("/tmp/out"),
                Duration::from_secs(10),
            )
            .is_err()
        );

        assert!(
            run_local_transcribe_midi(
                &worker,
                Path::new("piano.wav"),
                Path::new("/tmp/out"),
                Duration::MAX,
            )
            .is_err()
        );
    }

    #[test]
    fn resolve_packaged_worker_script_requires_complete_release_layout() {
        let dir = tempfile::tempdir().expect("tempdir");
        let exe_dir = dir.path().join("bin");
        let pw_dir = dir.path().join("python-worker");
        std::fs::create_dir_all(&exe_dir).unwrap();
        std::fs::create_dir_all(&pw_dir).unwrap();
        let fake_exe = exe_dir.join("forge");
        std::fs::write(&fake_exe, b"bin").unwrap();

        let script = pw_dir.join("forge_gpu_worker.py");
        std::fs::write(&script, b"# worker").unwrap();
        assert!(resolve_packaged_worker_script(&fake_exe).is_none());

        std::fs::write(pw_dir.join("pyproject.toml"), b"[project]").unwrap();
        assert!(resolve_packaged_worker_script(&fake_exe).is_none());

        std::fs::write(pw_dir.join("uv.lock"), b"version = 1").unwrap();
        let found = resolve_packaged_worker_script(&fake_exe).expect("complete packaged layout");
        assert_eq!(found, std::fs::canonicalize(&script).unwrap());
    }

    #[test]
    fn resolve_default_worker_falls_back_to_packaged_sibling_layout() {
        let dir = tempfile::tempdir().expect("tempdir");
        let missing_src = dir.path().join("no_source_root");
        let exe_dir = dir.path().join("bin");
        let pw_dir = dir.path().join("python-worker");
        std::fs::create_dir_all(&exe_dir).unwrap();
        std::fs::create_dir_all(&pw_dir).unwrap();
        let fake_exe = exe_dir.join("forge-mcp");
        std::fs::write(&fake_exe, b"bin").unwrap();
        let script = pw_dir.join("forge_gpu_worker.py");
        std::fs::write(&script, b"# worker").unwrap();
        std::fs::write(pw_dir.join("pyproject.toml"), b"[project]").unwrap();
        std::fs::write(pw_dir.join("uv.lock"), b"version = 1").unwrap();

        let found = resolve_default_worker_script_with(None, Some(&missing_src), Some(&fake_exe))
            .expect("falls back to packaged sibling layout");
        assert_eq!(found, std::fs::canonicalize(&script).unwrap());
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    fn alive(pid: &str) -> bool {
        Command::new("kill")
            .args(["-0", pid])
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    }

    #[test]
    fn timeout_kills_the_workers_grandchildren() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pid_file = dir.path().join("grandchild.pid");
        let mut cmd = Command::new("sh");
        cmd.arg("-c")
            .arg(r#"sleep 60 & echo $! > "$1"; wait"#)
            .arg("sh")
            .arg(&pid_file);
        let Err(err) = run_with_timeout(cmd, Duration::from_millis(500)) else {
            panic!("worker finished before the timeout");
        };
        assert!(err.to_string().contains("timed out"), "{err}");

        let pid = std::fs::read_to_string(&pid_file).expect("pid file");
        let pid = pid.trim();
        let deadline = Instant::now() + Duration::from_secs(5);
        while alive(pid) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(!alive(pid), "grandchild {pid} survived the timeout");
    }
}

#[cfg(all(test, windows))]
mod windows_tests {
    use super::*;

    fn pid_alive(pid: &str) -> bool {
        let needle = format!("\"{pid}\"");
        Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
            .output()
            .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).contains(&needle))
    }

    #[test]
    fn timeout_kills_windows_grandchild_process_tree() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pid_file = dir.path().join("grandchild.pid");
        let mut cmd = Command::new("powershell");
        cmd.env("FORGE_TEST_PID_FILE", &pid_file).args([
            "-NoLogo",
            "-NonInteractive",
            "-NoProfile",
            "-Command",
            "$p = Start-Process -FilePath 'ping.exe' -ArgumentList '-n','60','127.0.0.1' -NoNewWindow -PassThru; [IO.File]::WriteAllText($env:FORGE_TEST_PID_FILE, [string]$p.Id); Wait-Process -Id $p.Id",
        ]);
        let Err(err) = run_with_timeout(cmd, Duration::from_secs(10)) else {
            panic!("worker finished before the timeout");
        };
        assert!(err.to_string().contains("timed out"), "{err}");

        let pid = std::fs::read_to_string(&pid_file).expect("pid file written by worker");
        let pid = pid.trim();
        let deadline = Instant::now() + Duration::from_secs(5);
        while pid_alive(pid) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(100));
        }
        assert!(
            !pid_alive(pid),
            "Windows grandchild PID {pid} survived timeout kill_tree"
        );
    }
}
