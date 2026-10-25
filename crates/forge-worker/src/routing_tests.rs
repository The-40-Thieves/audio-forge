//! Routing, fallback and timeout tests. No GPU, Colab or network: the local worker is a stub
//! shell command and the Colab worker is a one-shot HTTP server on 127.0.0.1.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use base64::Engine as _;
use forge_core::AudioBuffer;

use super::*;
use crate::local_gpu::{self, LocalWorkerCommand, resolve_worker_script};

const SHORT: Duration = Duration::from_millis(300);

fn config(backend: WorkerBackend) -> SeparationConfig {
    SeparationConfig {
        backend,
        local_worker: None,
        colab_url: None,
        local_timeout: Duration::from_secs(10),
        http_timeout: Duration::from_secs(10),
    }
}

fn write_mix(dir: &Path) -> PathBuf {
    let path = dir.join("mix.wav");
    let tone: Vec<f64> = (0..512)
        .map(|i| (f64::from(i) * 0.05).sin() * 0.3)
        .collect();
    let buf = AudioBuffer::from_channels(vec![tone.clone(), tone], 48_000).unwrap();
    buf.save_wav(&path, 24).unwrap();
    path
}

/// A worker that runs `unix` under `sh -c` or `windows` as a `.bat` file under `cmd /C`.
/// The appended `separate --input ...` arguments are ignored by both.
fn stub_worker(dir: &Path, unix: &str, windows: &str) -> LocalWorkerCommand {
    if cfg!(windows) {
        let bat = dir.join("stub_worker.bat");
        std::fs::write(&bat, format!("@echo off\r\n{windows}\r\n")).unwrap();
        LocalWorkerCommand {
            program: "cmd".into(),
            args: vec!["/C".into(), bat.into()],
        }
    } else {
        LocalWorkerCommand {
            program: "sh".into(),
            args: vec!["-c".into(), unix.into(), "stub".into()],
        }
    }
}

/// A worker that prints a stem JSON naming `vocals` and `other` files it "wrote".
fn succeeding_worker(dir: &Path) -> (LocalWorkerCommand, PathBuf) {
    let vocals = dir.join("gpu_vocals.wav");
    let other = dir.join("gpu_other.wav");
    std::fs::write(&vocals, b"v").unwrap();
    std::fs::write(&other, b"o").unwrap();
    let json_path = dir.join("stems.json");
    let json = serde_json::json!({ "vocals": vocals, "other": other });
    std::fs::write(&json_path, json.to_string()).unwrap();
    let p = json_path.display();
    (
        stub_worker(dir, &format!("cat '{p}'"), &format!("type \"{p}\"")),
        vocals,
    )
}

fn sleeping_worker(dir: &Path) -> LocalWorkerCommand {
    stub_worker(
        dir,
        "exec sleep 5",
        "powershell -NoProfile -Command Start-Sleep -Seconds 5",
    )
}

/// Serves one request on 127.0.0.1. `None` accepts the connection and never answers.
fn serve_once(response: Option<(u16, String)>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let Some((code, body)) = response else {
            std::thread::sleep(Duration::from_secs(5));
            return;
        };
        read_request(&mut stream);
        let reply = format!(
            "HTTP/1.1 {code} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\
             Connection: close\r\n\r\n{body}",
            body.len()
        );
        let _ = stream.write_all(reply.as_bytes());
    });
    format!("http://{addr}")
}

fn read_request(stream: &mut impl Read) {
    let mut data = Vec::new();
    let mut chunk = [0_u8; 8192];
    loop {
        let n = stream.read(&mut chunk).unwrap_or(0);
        data.extend_from_slice(&chunk[..n]);
        let text = String::from_utf8_lossy(&data);
        if let Some(end) = text.find("\r\n\r\n") {
            let length = text[..end]
                .lines()
                .find_map(|l| {
                    l.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                })
                .unwrap_or(0);
            if data.len() >= end + 4 + length {
                return;
            }
        }
        if n == 0 {
            return;
        }
    }
}

fn colab_reply(vocal_bytes: &[u8]) -> String {
    let b64 = base64::engine::general_purpose::STANDARD.encode(vocal_bytes);
    serde_json::json!({ "ok": true, "data": { "stems": { "vocals": b64, "other": b64 } } })
        .to_string()
}

#[test]
fn auto_uses_local_gpu_worker_when_it_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    let (worker, vocals) = succeeding_worker(dir.path());
    let mut cfg = config(WorkerBackend::Auto);
    cfg.local_worker = Some(worker);
    let report = separate_stems(&write_mix(dir.path()), &dir.path().join("stems"), &cfg).unwrap();
    assert_eq!(
        report.backend,
        BackendUsed::LocalGpu,
        "{:?}",
        report.fallbacks
    );
    assert_eq!(report.bundle.vocals, Some(vocals));
}

#[test]
fn auto_reports_local_gpu_failure_and_falls_back_to_dsp() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = config(WorkerBackend::Auto);
    cfg.local_worker = Some(LocalWorkerCommand {
        program: "forge-no-such-gpu-worker".into(),
        args: Vec::new(),
    });
    let report = separate_stems(&write_mix(dir.path()), &dir.path().join("stems"), &cfg).unwrap();
    assert_eq!(report.backend, BackendUsed::PureRustDsp);
    assert!(
        report
            .fallbacks
            .iter()
            .any(|r| r.starts_with("local_gpu failed:") && r.contains("failed to launch")),
        "{:?}",
        report.fallbacks
    );
    assert!(
        report
            .fallbacks
            .iter()
            .any(|r| r.starts_with("remote_colab skipped"))
    );
    assert!(report.bundle.vocals.as_ref().is_some_and(|p| p.exists()));
}

#[test]
fn auto_tries_colab_after_a_missing_local_worker() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = config(WorkerBackend::Auto);
    cfg.colab_url = Some(serve_once(Some((200, colab_reply(b"RIFFcolab")))));
    let stems = dir.path().join("stems");
    let report = separate_stems(&write_mix(dir.path()), &stems, &cfg).unwrap();
    assert_eq!(
        report.backend,
        BackendUsed::RemoteColab,
        "{:?}",
        report.fallbacks
    );
    assert!(report.fallbacks[0].starts_with("local_gpu skipped"));
    let vocals = report.bundle.vocals.unwrap();
    assert_eq!(vocals, stems.join("colab_vocals.wav"));
    assert_eq!(std::fs::read(vocals).unwrap(), b"RIFFcolab");
}

#[test]
fn remote_colab_error_is_reported_before_dsp_fallback() {
    let dir = tempfile::tempdir().unwrap();
    let url = serve_once(Some((
        500,
        "{\"detail\":\"CUDA out of memory\"}".to_string(),
    )));
    let cfg = config(WorkerBackend::RemoteColab { base_url: url });
    let report = separate_stems(&write_mix(dir.path()), &dir.path().join("stems"), &cfg).unwrap();
    assert_eq!(report.backend, BackendUsed::PureRustDsp);
    assert_eq!(report.fallbacks.len(), 1, "{:?}", report.fallbacks);
    assert!(
        report.fallbacks[0].contains("HTTP 500"),
        "{}",
        report.fallbacks[0]
    );
    assert!(report.fallbacks[0].contains("CUDA out of memory"));
}

#[test]
fn pure_rust_dsp_never_starts_a_worker() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = config(WorkerBackend::PureRustDsp);
    cfg.local_worker = Some(sleeping_worker(dir.path()));
    cfg.colab_url = Some("http://127.0.0.1:9".to_string());
    let started = Instant::now();
    let report = separate_stems(&write_mix(dir.path()), &dir.path().join("stems"), &cfg).unwrap();
    assert_eq!(report.backend, BackendUsed::PureRustDsp);
    assert!(report.fallbacks.is_empty(), "{:?}", report.fallbacks);
    assert!(started.elapsed() < Duration::from_secs(2));
}

#[test]
fn hung_local_worker_is_killed_at_the_timeout() {
    let dir = tempfile::tempdir().unwrap();
    let worker = sleeping_worker(dir.path());
    let started = Instant::now();
    let err = local_gpu::separate(&worker, Path::new("in.wav"), dir.path(), SHORT).unwrap_err();
    let elapsed = started.elapsed();
    assert!(err.to_string().contains("timed out after 0.3 s"), "{err}");
    assert!(elapsed < Duration::from_secs(2), "took {elapsed:?}");
}

#[test]
fn hung_colab_worker_times_out() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = config(WorkerBackend::RemoteColab {
        base_url: serve_once(None),
    });
    cfg.http_timeout = SHORT;
    let started = Instant::now();
    let report = separate_stems(&write_mix(dir.path()), &dir.path().join("stems"), &cfg).unwrap();
    let elapsed = started.elapsed();
    assert_eq!(report.backend, BackendUsed::PureRustDsp);
    assert!(
        report.fallbacks[0].contains("timed out"),
        "{:?}",
        report.fallbacks
    );
    assert!(elapsed < Duration::from_secs(2), "took {elapsed:?}");
}

#[test]
fn worker_script_is_found_from_the_executable_not_the_cwd() {
    let root = tempfile::tempdir().unwrap();
    let script = root
        .path()
        .join("python-worker")
        .join("forge_gpu_worker.py");
    std::fs::create_dir_all(script.parent().unwrap()).unwrap();
    std::fs::write(&script, "").unwrap();
    let exe = root
        .path()
        .join("target")
        .join("release")
        .join("forge-mcp.exe");
    assert_eq!(
        resolve_worker_script(None, Some(&exe)),
        Some(script.clone())
    );
    let elsewhere = tempfile::tempdir().unwrap();
    let stray_exe = elsewhere.path().join("forge-mcp.exe");
    assert_eq!(resolve_worker_script(None, Some(&stray_exe)), None);
    assert_eq!(
        resolve_worker_script(Some(script.clone()), None),
        Some(script)
    );
    assert_eq!(
        resolve_worker_script(Some(root.path().join("missing.py")), Some(&exe)),
        None
    );
}

#[test]
fn backend_names_map_to_the_requested_backend() {
    let url = Some("https://colab.example");
    assert_eq!(parse_backend("auto", url).unwrap(), WorkerBackend::Auto);
    assert_eq!(
        parse_backend("LOCAL_GPU", None).unwrap(),
        WorkerBackend::LocalGpu
    );
    assert_eq!(
        parse_backend("dsp", None).unwrap(),
        WorkerBackend::PureRustDsp
    );
    assert_eq!(
        parse_backend("colab", url).unwrap(),
        WorkerBackend::RemoteColab {
            base_url: "https://colab.example".to_string()
        }
    );
    assert!(parse_backend("colab", None).is_err());
    assert!(parse_backend("cuda", url).is_err());
}

#[test]
fn forced_local_gpu_without_script_falls_back_with_reason() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = config(WorkerBackend::LocalGpu);
    cfg.colab_url = Some("http://127.0.0.1:9".to_string());
    let report = separate_stems(&write_mix(dir.path()), &dir.path().join("stems"), &cfg).unwrap();
    assert_eq!(report.backend, BackendUsed::PureRustDsp);
    assert_eq!(report.fallbacks.len(), 1, "{:?}", report.fallbacks);
    assert!(report.fallbacks[0].contains("FORGE_GPU_WORKER_SCRIPT"));
}
