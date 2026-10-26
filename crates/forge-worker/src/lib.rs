//! `forge-worker` — Hybrid AI Model Router & Cloud/GPU Bridge:
//! - Ephemeral Local GPU Worker (`uv run python-worker/forge_gpu_worker.py` on RTX 3080)
//! - Remote Google Colab A100 / L4 Worker (`FORGE_COLAB_WORKER_URL`)
//! - Native Rust Client for Google DeepMind `Lyria 3.5` (`lyria-3.5`) & `Lyria 3 Clip` via Gemini Interactions API
//! - Instant Pure-Rust Mid/Side & Spectral Stem Splitter Fallback (0 VRAM, runs in <100ms)
//! - XO / Atlas-style Local Drum Sample Spectral Indexer & Matcher

use std::path::{Path, PathBuf};
use std::time::Duration;

use base64::Engine as _;
use forge_core::{AudioBuffer, ForgeError, Result};
use forge_dsp::{Biquad, sanitize_sample_f64};
use serde::{Deserialize, Serialize};

mod colab;
mod dsp_split;
mod local_gpu;
mod routing;

pub use local_gpu::{BUILD_SOURCE_ROOT, LocalWorkerCommand, resolve_worker_script};
pub use routing::{
    BackendUsed, SeparationConfig, SeparationReport, WorkerBackend, parse_backend, separate_stems,
};

/// Calls Google DeepMind `Lyria 3.5` (`lyria-3.5`) or `Lyria 3 Clip` (`lyria-3-clip-preview`)
/// via the official Gemini `v1beta/interactions` endpoint to generate 44.1 kHz stereo WAV/MP3 audio.
pub async fn generate_music_lyria(
    prompt: &str,
    model_id: &str,
    output_audio_path: &Path,
) -> Result<(PathBuf, Option<String>)> {
    let api_key = std::env::var("GEMINI_API_KEY")
        .or_else(|_| std::env::var("GOOGLE_API_KEY"))
        .map_err(|_| {
            ForgeError::WorkerError(
                "GEMINI_API_KEY environment variable is required for Google Lyria generation"
                    .to_string(),
            )
        })?;

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(180))
        .build()
        .map_err(|e| ForgeError::WorkerError(format!("HTTP client init failed: {e}")))?;

    let payload = serde_json::json!({
        "model": model_id,
        "input": prompt,
        "response_format": { "type": "audio" }
    });

    let resp = client
        .post("https://generativelanguage.googleapis.com/v1beta/interactions")
        .header("x-goog-api-key", api_key)
        .json(&payload)
        .send()
        .await
        .map_err(|e| ForgeError::WorkerError(format!("Lyria API request failed: {e}")))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(ForgeError::WorkerError(format!(
            "Lyria API returned HTTP {status}: {body}"
        )));
    }

    let body_json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| ForgeError::WorkerError(format!("Invalid JSON from Lyria API: {e}")))?;

    let b64_audio = body_json
        .pointer("/output_audio/data")
        .or_else(|| body_json.pointer("/interaction/output_audio/data"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            ForgeError::WorkerError(
                "Lyria response did not contain output_audio.data payload".to_string(),
            )
        })?;

    let audio_bytes = base64::engine::general_purpose::STANDARD
        .decode(b64_audio)
        .map_err(|e| ForgeError::WorkerError(format!("Base64 audio decode failed: {e}")))?;

    if let Some(parent) = output_audio_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(output_audio_path, &audio_bytes)?;

    let lyrics = body_json
        .pointer("/output_text")
        .or_else(|| body_json.pointer("/interaction/output_text"))
        .and_then(|v| v.as_str())
        .map(ToString::to_string);

    Ok((output_audio_path.to_path_buf(), lyrics))
}

/// Spectral fingerprint of a local one-shot drum sample (XO / Atlas 2 style sample browser).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrumSampleDescriptor {
    pub path: PathBuf,
    pub category: String, // "Kick", "Snare", "HiHat", "Perc"
    pub spectral_centroid_hz: f64,
    pub sub_120hz_ratio: f64,
    pub crest_factor_db: f64,
    pub duration_ms: f64,
}

/// Scans a directory of `.wav` drum one-shots and classifies/indexes them by spectral centroid
/// and sub-bass energy ratio (Open-Source XLN Audio XO / Algonaut Atlas equivalent).
pub fn index_drum_sample_library(folder: &Path) -> Result<Vec<DrumSampleDescriptor>> {
    if !folder.exists() {
        return Err(ForgeError::InvalidParameter(format!(
            "Sample folder '{}' does not exist",
            folder.display()
        )));
    }

    let mut samples = Vec::new();
    for entry in std::fs::read_dir(folder)? {
        let entry = entry?;
        let path = entry.path();
        if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("wav"))
            && let Ok(buf) = AudioBuffer::load_wav(&path)
        {
            let sr = f64::from(buf.sample_rate.max(1));
            let mono = buf.to_mono();
            if mono.is_empty() || buf.duration_secs() > 4.0 {
                continue;
            }
            let mut lp = Biquad::lowpass(sr, 120.0, std::f64::consts::FRAC_1_SQRT_2);
            let mut total_e = 1e-12_f64;
            let mut sub_e = 0.0_f64;
            let mut zcr = 0_usize;
            let mut peak = 1e-12_f64;

            for i in 0..mono.len() {
                let x = sanitize_sample_f64(mono[i]);
                total_e += x * x;
                let lo = lp.process_sample(x);
                sub_e += lo * lo;
                if x.abs() > peak {
                    peak = x.abs();
                }
                if i > 0 && (x >= 0.0) != (mono[i - 1] >= 0.0) {
                    zcr += 1;
                }
            }

            let rms = (total_e / (mono.len() as f64)).sqrt().max(1e-12);
            let crest_factor_db = 20.0 * (peak / rms).log10();
            let sub_120hz_ratio = (sub_e / total_e).clamp(0.0, 1.0);
            let spectral_centroid_hz = ((zcr as f64) / (mono.len() as f64)) * (sr * 0.5);

            let category = if sub_120hz_ratio > 0.55 && spectral_centroid_hz < 900.0 {
                "Kick"
            } else if spectral_centroid_hz > 3800.0 && sub_120hz_ratio < 0.15 {
                "HiHat"
            } else {
                "Snare"
            }
            .to_string();

            samples.push(DrumSampleDescriptor {
                path,
                category,
                spectral_centroid_hz,
                sub_120hz_ratio,
                crest_factor_db,
                duration_ms: buf.duration_secs() * 1000.0,
            });
        }
    }

    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(16))]

        #[test]
        fn prop_pure_rust_stem_separator_and_drum_indexer_totality(
            frames in 256usize..2048,
            sub_amp in 0.0f64..0.8,
            voc_amp in 0.0f64..0.8,
            hat_amp in 0.0f64..0.8,
        ) {
            let dir = tempfile::tempdir().unwrap();
            let input_wav = dir.path().join("mix.wav");
            let stems_dir = dir.path().join("stems");

            let sr = 48_000;
            let mut l = vec![0.0_f64; frames];
            let mut r = vec![0.0_f64; frames];
            for i in 0..frames {
                let t = (i as f64) / (sr as f64);
                let sub = sub_amp * (2.0 * std::f64::consts::PI * 55.0 * t).sin();
                let voc = voc_amp * (2.0 * std::f64::consts::PI * 440.0 * t).sin();
                let hat = hat_amp * (2.0 * std::f64::consts::PI * 8000.0 * t).sin();
                l[i] = sub + voc + hat;
                r[i] = sub + voc - hat;
            }
            let buf = AudioBuffer::from_channels(vec![l, r], sr).unwrap();
            buf.save_wav(&input_wav, 24).unwrap();

            let config = SeparationConfig {
                backend: WorkerBackend::PureRustDsp,
                local_worker: None,
                colab_url: None,
                local_timeout: Duration::from_secs(1),
                http_timeout: Duration::from_secs(1),
                require_neural: false,
            };
            let report = separate_stems(&input_wav, &stems_dir, &config).unwrap();
            prop_assert_eq!(report.backend, BackendUsed::PureRustDsp);
            let bundle = report.bundle;
            prop_assert!(bundle.vocals.as_ref().unwrap().exists());
            prop_assert!(bundle.drums.as_ref().unwrap().exists());
            prop_assert!(bundle.bass.as_ref().unwrap().exists());
            prop_assert!(bundle.other.as_ref().unwrap().exists());

            let indexed = index_drum_sample_library(&stems_dir).unwrap();
            prop_assert_eq!(indexed.len(), 4);
            for desc in &indexed {
                prop_assert!(desc.spectral_centroid_hz.is_finite() && desc.spectral_centroid_hz >= 0.0);
                prop_assert!(desc.sub_120hz_ratio.is_finite() && (0.0..=1.0).contains(&desc.sub_120hz_ratio));
                prop_assert!(desc.crest_factor_db.is_finite());
                prop_assert!(["Kick", "Snare", "HiHat"].contains(&desc.category.as_str()));
            }
        }
    }
}
