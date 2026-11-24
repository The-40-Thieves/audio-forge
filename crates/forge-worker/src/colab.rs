//! Remote Google Colab A100/L4 worker client (`notebooks/ForgeAudio_Colab_A100_Worker.ipynb`).
//!
//! Protocol: `POST {base_url}/api/v1/execute` with
//! `{"task": "separate_stems", "payload": {"filename": ..., "audio_b64": ...}}`. The worker
//! answers `{"ok": true, "data": {"stems": {"vocals": "<base64 wav>", "other": ...}}}`.
//! The audio travels in the request because the Colab machine cannot read local paths.

use std::path::Path;
use std::time::Duration;

use base64::Engine as _;
use forge_core::{ForgeError, Result, StemBundle};
use serde_json::{Value, json};

/// Stem keys the worker may return, in `StemBundle` field order.
const STEM_KEYS: [&str; 6] = ["vocals", "drums", "bass", "guitar", "keys", "other"];

const MAX_COLAB_UPLOAD_BYTES: u64 = 256 * 1024 * 1024;
const MAX_COLAB_RESPONSE_BYTES: u64 = 512 * 1024 * 1024;

fn validate_colab_url(base_url: &str) -> Result<reqwest::Url> {
    let trimmed = base_url.trim();
    let url = reqwest::Url::parse(trimmed).map_err(|e| {
        ForgeError::WorkerError(format!(
            "FORGE_COLAB_WORKER_URL is not a valid URL ('{trimmed}'): {e}"
        ))
    })?;
    if !url.username().is_empty() || url.password().is_some() {
        return Err(ForgeError::WorkerError(
            "FORGE_COLAB_WORKER_URL must not embed userinfo credentials".to_string(),
        ));
    }
    let loopback = matches!(url.host_str(), Some("localhost"))
        || url
            .host_str()
            .and_then(|h| {
                h.trim_matches(|c| c == '[' || c == ']')
                    .parse::<std::net::IpAddr>()
                    .ok()
            })
            .is_some_and(|ip| ip.is_loopback());
    if url.scheme() == "https" || (url.scheme() == "http" && loopback) {
        return Ok(url);
    }
    Err(ForgeError::WorkerError(format!(
        "FORGE_COLAB_WORKER_URL must use https:// (or localhost http:// for local testing), got '{trimmed}'"
    )))
}

/// Uploads `input_path` to the Colab worker and writes the returned stems into `output_dir`.
pub(crate) fn separate(
    base_url: &str,
    input_path: &Path,
    output_dir: &Path,
    timeout: Duration,
) -> Result<StemBundle> {
    validate_colab_url(base_url)?;
    let meta = std::fs::metadata(input_path)?;
    if meta.len() > MAX_COLAB_UPLOAD_BYTES {
        return Err(ForgeError::WorkerError(format!(
            "Input audio ({} bytes) exceeds Colab upload cap ({MAX_COLAB_UPLOAD_BYTES} bytes)",
            meta.len()
        )));
    }
    let audio = std::fs::read(input_path)?;
    let filename = input_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "input.wav".to_string());
    let body = json!({
        "task": "separate_stems",
        "payload": {
            "filename": filename,
            "audio_b64": base64::engine::general_purpose::STANDARD.encode(audio),
        }
    });
    let url = format!("{}/api/v1/execute", base_url.trim_end_matches('/'));
    let reply = post_json_blocking(&url, &body, timeout)?;
    write_stems(&reply, output_dir)
}

/// Sends the request on a private current-thread runtime in its own thread, so this works
/// both from plain threads (forge-mcp) and from inside a Tokio runtime (forge-cli).
fn post_json_blocking(url: &str, body: &Value, timeout: Duration) -> Result<Value> {
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| ForgeError::WorkerError(format!("Tokio runtime failed: {e}")))?;
                rt.block_on(post_json(url, body, timeout))
            })
            .join()
            .unwrap_or_else(|_| Err(ForgeError::WorkerError("Colab request panicked".into())))
    })
}

async fn post_json(url: &str, body: &Value, timeout: Duration) -> Result<Value> {
    let client = reqwest::Client::builder()
        .timeout(timeout)
        .connect_timeout(timeout.min(Duration::from_secs(30)))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| ForgeError::WorkerError(format!("HTTP client init failed: {e}")))?;
    let mut req = client.post(url).json(body);
    if let Ok(token) = std::env::var("FORGE_COLAB_API_TOKEN") {
        let t = token.trim();
        if !t.is_empty() {
            req = req.bearer_auth(t);
        }
    }
    let mut resp = req.send().await.map_err(|e| {
        let kind = if e.is_timeout() {
            "timed out"
        } else {
            "failed"
        };
        ForgeError::WorkerError(format!("Colab request to {url} {kind}: {e}"))
    })?;
    let status = resp.status();
    if let Some(len) = resp.content_length()
        && len > MAX_COLAB_RESPONSE_BYTES
    {
        return Err(ForgeError::WorkerError(format!(
            "Colab response ({len} bytes) exceeds safety limit ({MAX_COLAB_RESPONSE_BYTES} bytes)"
        )));
    }
    let mut raw = Vec::new();
    while let Some(chunk) = resp.chunk().await.map_err(|e| {
        ForgeError::WorkerError(format!("Colab response from {url} was cut off: {e}"))
    })? {
        if raw.len().saturating_add(chunk.len()) > MAX_COLAB_RESPONSE_BYTES as usize {
            return Err(ForgeError::WorkerError(format!(
                "Colab response exceeded safety limit ({MAX_COLAB_RESPONSE_BYTES} bytes)"
            )));
        }
        raw.extend_from_slice(&chunk);
    }
    if !status.is_success() {
        let preview = String::from_utf8_lossy(&raw);
        return Err(ForgeError::WorkerError(format!(
            "Colab worker returned HTTP {status}: {}",
            preview.chars().take(400).collect::<String>()
        )));
    }
    serde_json::from_slice(&raw)
        .map_err(|e| ForgeError::WorkerError(format!("Colab worker sent invalid JSON: {e}")))
}

/// Decodes `data.stems` from a worker reply into `output_dir/colab_<stem>.wav` files.
fn write_stems(reply: &Value, output_dir: &Path) -> Result<StemBundle> {
    if reply.get("ok").and_then(Value::as_bool) != Some(true) {
        let detail = reply.get("error").or_else(|| reply.get("detail"));
        return Err(ForgeError::WorkerError(format!(
            "Colab worker reported failure: {}",
            detail.map_or_else(|| reply.to_string(), Value::to_string)
        )));
    }
    let stems = reply.pointer("/data/stems").and_then(Value::as_object);
    let mut paths: [Option<std::path::PathBuf>; 6] = Default::default();
    for (slot, key) in paths.iter_mut().zip(STEM_KEYS) {
        let Some(b64) = stems.and_then(|s| s.get(key)).and_then(Value::as_str) else {
            continue;
        };
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .map_err(|e| {
                ForgeError::WorkerError(format!("Colab stem '{key}' is not base64: {e}"))
            })?;
        let path = output_dir.join(format!("colab_{key}.wav"));
        std::fs::write(&path, bytes)?;
        *slot = Some(path);
    }
    let [vocals, drums, bass, guitar, keys, other] = paths;
    Ok(StemBundle {
        vocals,
        drums,
        bass,
        guitar,
        keys,
        other,
        ..StemBundle::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_colab_url_rejects_host_confusion_and_allows_https_or_loopback() {
        for bad in [
            "http://localhost.evil.com",
            "http://127.0.0.1.nip.io",
            "http://localhost@169.254.169.254",
            "https://user:pass@example.com",
            "http://example.com",
            "ftp://example.com",
        ] {
            assert!(
                validate_colab_url(bad).is_err(),
                "must reject unsafe URL '{bad}'"
            );
        }
        for good in [
            "https://abc123.ngrok-free.app",
            "http://localhost:8000",
            "http://127.0.0.1:8000/",
            "http://[::1]:8000",
        ] {
            assert!(
                validate_colab_url(good).is_ok(),
                "must allow valid URL '{good}'"
            );
        }
    }
}
