//! Remote Google Colab A100/L4 worker client (`notebooks/ForgeAudio_Colab_A100_Worker.ipynb`).
//!
//! Protocol: `POST {base_url}/api/v1/execute` with `{"task": ..., "payload": {...}}`.
//! Audio inputs are base64-encoded so the Colab GPU runtime needs no access to local paths.

use std::path::{Path, PathBuf};
use std::time::Duration;

use base64::Engine as _;
use forge_core::{ForgeError, Result, StemBundle};
use forge_dsp::BeatGridReport;
use serde_json::{Map, Value, json};

use crate::local_gpu::{
    MidiTranscribeResult, NeuralDenoiseResult, NeuralRestoreResult, QualityEvalResult,
    VocalRescueResult,
};
use crate::routing::SeparationConfig;

const MAX_COLAB_UPLOAD_BYTES: u64 = 256 * 1024 * 1024;
const MAX_COLAB_RESPONSE_BYTES: u64 = 512 * 1024 * 1024;

fn is_blocked_private_or_metadata_ip(ip: std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => {
            let octets = v4.octets();
            let cgnat = octets[0] == 100 && (64..=127).contains(&octets[1]);
            v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || v4.is_multicast()
                || cgnat
        }
        std::net::IpAddr::V6(v6) => {
            if let Some(mapped) = v6.to_ipv4_mapped() {
                return !mapped.is_loopback()
                    && is_blocked_private_or_metadata_ip(std::net::IpAddr::V4(mapped));
            }
            let seg0 = v6.segments()[0];
            let ula = (seg0 & 0xfe00) == 0xfc00;
            let link_local = (seg0 & 0xffc0) == 0xfe80;
            v6.is_unspecified() || v6.is_multicast() || ula || link_local
        }
    }
}

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
    let parsed_ip = url.host_str().and_then(|h| {
        h.trim_matches(|c| c == '[' || c == ']')
            .parse::<std::net::IpAddr>()
            .ok()
    });
    if let Some(ip) = parsed_ip
        && !ip.is_loopback()
        && is_blocked_private_or_metadata_ip(ip)
    {
        return Err(ForgeError::WorkerError(
            "FORGE_COLAB_WORKER_URL must not target private, link-local, or metadata IP addresses"
                .to_string(),
        ));
    }
    let loopback =
        matches!(url.host_str(), Some("localhost")) || parsed_ip.is_some_and(|ip| ip.is_loopback());
    if url.scheme() == "https" || (url.scheme() == "http" && loopback) {
        return Ok(url);
    }
    Err(ForgeError::WorkerError(format!(
        "FORGE_COLAB_WORKER_URL must use https:// (or localhost http:// for local testing), got '{trimmed}'"
    )))
}

fn encode_audio_file(path: &Path) -> Result<(String, String)> {
    let meta = std::fs::metadata(path)?;
    if meta.len() > MAX_COLAB_UPLOAD_BYTES {
        return Err(ForgeError::WorkerError(format!(
            "Input audio ({} bytes) exceeds Colab upload cap ({MAX_COLAB_UPLOAD_BYTES} bytes)",
            meta.len()
        )));
    }
    let bytes = std::fs::read(path)?;
    let filename = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "input.wav".to_string());
    Ok((
        filename,
        base64::engine::general_purpose::STANDARD.encode(bytes),
    ))
}

fn execute_task_with_audio(
    base_url: &str,
    task: &str,
    input_path: &Path,
    mut payload: Map<String, Value>,
    timeout: Duration,
) -> Result<Value> {
    validate_colab_url(base_url)?;
    let (filename, audio_b64) = encode_audio_file(input_path)?;
    payload.insert("filename".to_string(), json!(filename));
    payload.insert("audio_b64".to_string(), json!(audio_b64));
    let body = json!({
        "task": task,
        "payload": Value::Object(payload),
    });
    let url = format!("{}/api/v1/execute", base_url.trim_end_matches('/'));
    let reply = post_json_blocking(&url, &body, timeout)?;
    if reply.get("ok").and_then(Value::as_bool) != Some(true) {
        let detail = reply.get("error").or_else(|| reply.get("detail"));
        return Err(ForgeError::WorkerError(format!(
            "Colab worker reported failure: {}",
            detail.map_or_else(|| reply.to_string(), Value::to_string)
        )));
    }
    Ok(reply)
}

fn validate_decoded_artifact_header(bytes: &[u8], out_path: &Path, label: &str) -> Result<()> {
    let ext = out_path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    match ext.as_deref() {
        Some("wav") if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" => {
            Err(ForgeError::WorkerError(format!(
                "Colab {label} failed RIFF/WAVE header validation"
            )))
        }
        Some("mid" | "midi") if bytes.len() < 4 || &bytes[0..4] != b"MThd" => Err(
            ForgeError::WorkerError(format!("Colab {label} failed MThd MIDI header validation")),
        ),
        _ => Ok(()),
    }
}

fn decode_b64_to_path(b64: &str, out_path: &Path, label: &str) -> Result<PathBuf> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|e| ForgeError::WorkerError(format!("Colab {label} is not base64: {e}")))?;
    validate_decoded_artifact_header(&bytes, out_path, label)?;
    if let Some(parent) = out_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(out_path, bytes)?;
    Ok(out_path.to_path_buf())
}

fn extract_warnings(reply: &Value) -> Vec<String> {
    reply
        .pointer("/data/warnings")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(ToString::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn file_stem_safe(path: &Path) -> String {
    let cleaned: String = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "audio".to_string())
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let trimmed = cleaned.trim_matches(['.', '_']);
    if trimmed.is_empty() {
        "audio".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Uploads `input_path` to the Colab worker with full `SeparationConfig` flags and writes returned stems.
pub(crate) fn separate(
    base_url: &str,
    input_path: &Path,
    output_dir: &Path,
    config: &SeparationConfig,
) -> Result<(StemBundle, Vec<String>)> {
    let mut payload = Map::new();
    for (k, v) in [
        ("stems", json!(config.stems)),
        ("quality", json!(config.quality.as_str())),
        ("dereverb", json!(config.dereverb)),
        ("dereverb_echo", json!(config.dereverb_echo)),
        ("debleed", json!(config.debleed)),
        ("de_aspiration", json!(config.de_aspiration)),
        ("drum_sep", json!(config.drum_sep)),
        ("split_lead_backing", json!(config.split_lead_backing)),
        ("split_male_female", json!(config.split_male_female)),
        ("split_guitar", json!(config.split_guitar)),
        ("split_orchestral", json!(config.split_orchestral)),
        ("split_crowd", json!(config.split_crowd)),
        ("restore", json!(config.restore)),
    ] {
        payload.insert(k.to_string(), v);
    }
    let reply = execute_task_with_audio(
        base_url,
        "separate_stems",
        input_path,
        payload,
        config.http_timeout,
    )?;
    let warnings = extract_warnings(&reply);
    let bundle = write_stems(&reply, output_dir)?;
    Ok((bundle, warnings))
}

fn decode_audio_task_reply(
    reply: &Value,
    default_backend: &str,
    out_path: PathBuf,
    label: &str,
    task_name: &str,
) -> Result<(PathBuf, String)> {
    let b64 = reply
        .pointer("/data/audio_b64")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ForgeError::WorkerError(format!("Colab {task_name} reply missing data.audio_b64"))
        })?;
    let path = decode_b64_to_path(b64, &out_path, label)?;
    let backend = reply
        .pointer("/data/backend")
        .and_then(Value::as_str)
        .unwrap_or(default_backend)
        .to_string();
    Ok((path, backend))
}

#[allow(dead_code)]
pub(crate) fn denoise(
    base_url: &str,
    input_path: &Path,
    output_dir: &Path,
    backend: &str,
    attenuation_limit_db: Option<f64>,
    timeout: Duration,
) -> Result<NeuralDenoiseResult> {
    let mut payload = Map::new();
    payload.insert("backend".to_string(), json!(backend));
    if let Some(lim) = attenuation_limit_db {
        payload.insert("attenuation_limit_db".to_string(), json!(lim));
    }
    let reply = execute_task_with_audio(base_url, "denoise", input_path, payload, timeout)?;
    let out_path = output_dir.join(format!("{}_denoised.wav", file_stem_safe(input_path)));
    let (denoised, used_backend) =
        decode_audio_task_reply(&reply, backend, out_path, "denoised audio", "denoise")?;
    Ok(NeuralDenoiseResult {
        denoised,
        backend: used_backend,
    })
}

#[allow(dead_code)]
pub(crate) fn transcribe(
    base_url: &str,
    input_path: &Path,
    output_dir: &Path,
    timeout: Duration,
) -> Result<MidiTranscribeResult> {
    let reply = execute_task_with_audio(base_url, "transcribe", input_path, Map::new(), timeout)?;
    let b64 = reply
        .pointer("/data/midi_b64")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ForgeError::WorkerError("Colab transcribe reply missing data.midi_b64".into())
        })?;
    let out_path = output_dir.join(format!("{}_basic_pitch.mid", file_stem_safe(input_path)));
    Ok(MidiTranscribeResult {
        midi: decode_b64_to_path(b64, &out_path, "MIDI")?,
    })
}

pub(crate) fn restore(
    base_url: &str,
    input_path: &Path,
    output_dir: &Path,
    backend: &str,
    prompt: Option<&str>,
    chunk_secs: Option<f64>,
    timeout: Duration,
) -> Result<NeuralRestoreResult> {
    let mut payload = Map::new();
    payload.insert("backend".to_string(), json!(backend));
    if let Some(p) = prompt {
        payload.insert("prompt".to_string(), json!(p));
    }
    if let Some(cs) = chunk_secs {
        payload.insert("chunk_secs".to_string(), json!(cs));
    }
    let reply = execute_task_with_audio(base_url, "restore", input_path, payload, timeout)?;
    let out_path = output_dir.join(format!("{}_restored.wav", file_stem_safe(input_path)));
    let (restored, used_backend) =
        decode_audio_task_reply(&reply, backend, out_path, "restored audio", "restore")?;
    Ok(NeuralRestoreResult {
        restored,
        backend: used_backend,
        warnings: extract_warnings(&reply),
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn vocal_rescue(
    base_url: &str,
    input_path: &Path,
    output_dir: &Path,
    backend: &str,
    reference_voice: Option<&Path>,
    dereverb: bool,
    de_aspiration: bool,
    timeout: Duration,
) -> Result<VocalRescueResult> {
    let mut payload = Map::new();
    payload.insert("backend".to_string(), json!(backend));
    payload.insert("dereverb".to_string(), json!(dereverb));
    payload.insert("de_aspiration".to_string(), json!(de_aspiration));
    if let Some(ref_path) = reference_voice {
        let (ref_name, ref_b64) = encode_audio_file(ref_path)?;
        payload.insert("reference_filename".to_string(), json!(ref_name));
        payload.insert("reference_voice_b64".to_string(), json!(ref_b64));
    }
    let reply = execute_task_with_audio(base_url, "vocal_rescue", input_path, payload, timeout)?;
    let out_path = output_dir.join(format!("{}_vocal_rescued.wav", file_stem_safe(input_path)));
    let (rescued, used_backend) =
        decode_audio_task_reply(&reply, backend, out_path, "rescued vocal", "vocal_rescue")?;
    let stages = reply
        .pointer("/data/stages")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(ToString::to_string))
                .collect()
        })
        .unwrap_or_default();
    Ok(VocalRescueResult {
        rescued,
        backend: used_backend,
        stages,
        warnings: extract_warnings(&reply),
    })
}

pub(crate) fn beat_track(
    base_url: &str,
    input_path: &Path,
    backend: &str,
    timeout: Duration,
) -> Result<BeatGridReport> {
    let mut payload = Map::new();
    payload.insert("backend".to_string(), json!(backend));
    let reply = execute_task_with_audio(base_url, "beat_track", input_path, payload, timeout)?;
    let data = reply
        .get("data")
        .cloned()
        .ok_or_else(|| ForgeError::WorkerError("Colab beat_track reply missing data".into()))?;
    serde_json::from_value(data)
        .map_err(|e| ForgeError::WorkerError(format!("Invalid Colab beat_track data: {e}")))
}

pub(crate) fn evaluate_quality(
    base_url: &str,
    input_path: &Path,
    reference_path: Option<&Path>,
    vocal: bool,
    backend: &str,
    timeout: Duration,
) -> Result<QualityEvalResult> {
    let mut payload = Map::new();
    payload.insert("backend".to_string(), json!(backend));
    payload.insert("vocal".to_string(), json!(vocal));
    if let Some(ref_path) = reference_path {
        let (ref_name, ref_b64) = encode_audio_file(ref_path)?;
        payload.insert("reference_filename".to_string(), json!(ref_name));
        payload.insert("reference_b64".to_string(), json!(ref_b64));
    }
    let reply =
        execute_task_with_audio(base_url, "evaluate_quality", input_path, payload, timeout)?;
    let data = reply.get("data").cloned().ok_or_else(|| {
        ForgeError::WorkerError("Colab evaluate_quality reply missing data".into())
    })?;
    serde_json::from_value(data)
        .map_err(|e| ForgeError::WorkerError(format!("Invalid Colab evaluate_quality data: {e}")))
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

/// Decodes up to 23 `data.stems` entries from a worker reply into `output_dir/colab_<stem>.wav`.
fn write_stems(reply: &Value, output_dir: &Path) -> Result<StemBundle> {
    let stems = reply.pointer("/data/stems").and_then(Value::as_object);
    let mut bundle = StemBundle::default();
    let slots: [(&str, &mut Option<PathBuf>); 23] = [
        ("vocals", &mut bundle.vocals),
        ("lead_vocals", &mut bundle.lead_vocals),
        ("backing_vocals", &mut bundle.backing_vocals),
        ("male_vocals", &mut bundle.male_vocals),
        ("female_vocals", &mut bundle.female_vocals),
        ("vocal_reverb", &mut bundle.vocal_reverb),
        ("drums", &mut bundle.drums),
        ("bass", &mut bundle.bass),
        ("guitar", &mut bundle.guitar),
        ("electric_guitar", &mut bundle.electric_guitar),
        ("acoustic_guitar", &mut bundle.acoustic_guitar),
        ("keys", &mut bundle.keys),
        ("strings", &mut bundle.strings),
        ("brass", &mut bundle.brass),
        ("wind", &mut bundle.wind),
        ("synth", &mut bundle.synth),
        ("crowd", &mut bundle.crowd),
        ("other", &mut bundle.other),
        ("kick", &mut bundle.kick),
        ("snare", &mut bundle.snare),
        ("toms", &mut bundle.toms),
        ("hihat", &mut bundle.hihat),
        ("cymbals", &mut bundle.cymbals),
    ];
    for (key, slot) in slots {
        let Some(b64) = stems.and_then(|s| s.get(key)).and_then(Value::as_str) else {
            continue;
        };
        let path = output_dir.join(format!("colab_{key}.wav"));
        *slot = Some(decode_b64_to_path(b64, &path, &format!("stem '{key}'"))?);
    }
    Ok(bundle)
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

    #[test]
    fn separate_forwards_flags_and_decodes_23_stem_slots() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let dir = tempfile::tempdir().unwrap();
        let inp = dir.path().join("song.wav");
        let out_dir = dir.path().join("stems");
        std::fs::write(&inp, b"RIFFdemo").unwrap();

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let captured = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let captured_srv = std::sync::Arc::clone(&captured);
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = Vec::new();
            let mut chunk = [0_u8; 4096];
            loop {
                let n = stream.read(&mut chunk).unwrap_or(0);
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&chunk[..n]);
                let txt = String::from_utf8_lossy(&buf);
                if let Some(hdr_end) = txt.find("\r\n\r\n") {
                    let body_start = hdr_end + 4;
                    let content_len = txt[..hdr_end]
                        .lines()
                        .find_map(|l| {
                            let (k, v) = l.split_once(':')?;
                            k.trim()
                                .eq_ignore_ascii_case("content-length")
                                .then(|| v.trim().parse::<usize>().ok())
                                .flatten()
                        })
                        .unwrap_or(0);
                    if buf.len() >= body_start + content_len {
                        *captured_srv.lock().unwrap() =
                            String::from_utf8_lossy(&buf[body_start..body_start + content_len])
                                .into_owned();
                        break;
                    }
                }
            }
            let stem_b64 =
                base64::engine::general_purpose::STANDARD.encode(b"RIFF\x24\x00\x00\x00WAVEfmt ");
            let resp_body = json!({
                "ok": true,
                "data": {
                    "warnings": ["sub-split notice"],
                    "stems": {
                        "vocals": stem_b64,
                        "lead_vocals": stem_b64,
                        "strings": stem_b64,
                        "cymbals": stem_b64,
                        "other": stem_b64
                    }
                }
            })
            .to_string();
            let reply = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                resp_body.len(),
                resp_body
            );
            stream.write_all(reply.as_bytes()).unwrap();
        });

        let cfg = SeparationConfig {
            debleed: true,
            de_aspiration: true,
            dereverb_echo: true,
            http_timeout: Duration::from_secs(5),
            ..SeparationConfig::default()
        };
        let url = format!("http://127.0.0.1:{}", addr.port());
        let (bundle, warnings) = separate(&url, &inp, &out_dir, &cfg).expect("colab separate");
        handle.join().unwrap();

        let sent: Value = serde_json::from_str(&captured.lock().unwrap()).unwrap();
        assert_eq!(sent["task"], "separate_stems");
        assert_eq!(sent["payload"]["debleed"], true);
        assert_eq!(sent["payload"]["de_aspiration"], true);
        assert_eq!(sent["payload"]["dereverb_echo"], true);
        assert_eq!(warnings, vec!["sub-split notice".to_string()]);
        assert!(bundle.vocals.as_ref().unwrap().is_file());
        assert!(bundle.lead_vocals.as_ref().unwrap().is_file());
        assert!(bundle.strings.as_ref().unwrap().is_file());
        assert!(bundle.cymbals.as_ref().unwrap().is_file());
        assert!(bundle.other.as_ref().unwrap().is_file());
    }

    #[test]
    fn validate_colab_url_blocks_private_link_local_and_metadata_ips() {
        for blocked in [
            "https://169.254.169.254/latest/meta-data",
            "https://10.0.0.1:8000",
            "https://192.168.1.5:8000",
            "https://172.16.0.1:8000",
            "https://100.64.0.1:8000",
            "https://[fe80::1]:8000",
            "https://[fc00::1]:8000",
            "https://[::ffff:169.254.169.254]:8000",
        ] {
            assert!(
                validate_colab_url(blocked).is_err(),
                "must reject SSRF target '{blocked}'"
            );
        }
    }

    #[test]
    fn decode_b64_to_path_validates_wav_and_midi_magic_headers() {
        let dir = tempfile::tempdir().unwrap();
        let bad_wav_b64 = base64::engine::general_purpose::STANDARD.encode(b"NOT_A_WAV_FILE");
        assert!(
            decode_b64_to_path(&bad_wav_b64, &dir.path().join("bad.wav"), "stem").is_err(),
            "corrupt .wav payload must be rejected"
        );

        let bad_mid_b64 = base64::engine::general_purpose::STANDARD.encode(b"NO_MIDI_HEADER");
        assert!(
            decode_b64_to_path(&bad_mid_b64, &dir.path().join("bad.mid"), "midi").is_err(),
            "corrupt .mid payload must be rejected"
        );

        let good_mid_b64 =
            base64::engine::general_purpose::STANDARD.encode(b"MThd\x00\x00\x00\x06");
        let saved_mid = decode_b64_to_path(&good_mid_b64, &dir.path().join("ok.mid"), "midi")
            .expect("valid MThd header should pass");
        assert!(saved_mid.is_file());
    }
}
