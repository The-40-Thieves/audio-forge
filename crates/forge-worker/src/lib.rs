//! `forge-worker` — Hybrid AI Model Router & Cloud/GPU Bridge:
//! - Ephemeral Local GPU Worker (`uv run python-worker/forge_gpu_worker.py` on RTX 3080)
//! - Remote Google Colab A100 / L4 Worker (`FORGE_COLAB_WORKER_URL`)
//! - Native Rust Client for Google DeepMind `Lyria 3.5` (`lyria-3.5`) & `Lyria 3 Clip` via Gemini Interactions API
//! - Instant Pure-Rust Mid/Side & Spectral Stem Splitter Fallback (0 VRAM, runs in <100ms)
//! - XO / Atlas-style Local Drum Sample Spectral Indexer & Matcher

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::Duration;

use base64::Engine as _;
use forge_core::{AudioBuffer, ForgeError, Result};
use forge_dsp::{BeatGridReport, Biquad, analyze_beat_grid, audit_audio, sanitize_sample_f64};
use serde::{Deserialize, Serialize};

mod colab;
mod dsp_split;
mod local_gpu;
mod routing;

pub use local_gpu::{
    BUILD_SOURCE_ROOT, F0TrackFrame, F0TrackResult, LocalWorkerCommand, MidiTranscribeResult,
    NeuralDenoiseResult, NeuralRestoreResult, QualityEvalMetrics, QualityEvalResult,
    VocalRescueResult, resolve_default_local_worker, run_local_beat_track,
    run_local_evaluate_quality, run_local_f0_track, run_local_neural_denoise,
    run_local_neural_restore, run_local_transcribe_midi, run_local_vocal_rescue,
};
pub use routing::{
    BackendUsed, SeparationConfig, SeparationQuality, SeparationReport, WorkerBackend,
    parse_backend, parse_quality, separate_stems,
};

fn env_colab_url() -> Option<String> {
    std::env::var("FORGE_COLAB_WORKER_URL")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn try_local_worker() -> Option<LocalWorkerCommand> {
    resolve_default_local_worker().ok()
}

/// Restores bandwidth / codecs (`Apollo` / `FlowHigh` / `AudioSR` / `SonicMaster` / `exciter`)
/// trying local GPU first, then `FORGE_COLAB_WORKER_URL` if configured.
fn try_local_then_colab<T>(
    task: &str,
    run_local: impl FnOnce(&LocalWorkerCommand) -> Result<T>,
    run_colab: impl FnOnce(&str) -> Result<T>,
) -> Result<T> {
    let mut errs = Vec::new();
    if let Some(worker) = try_local_worker() {
        match run_local(&worker) {
            Ok(res) => return Ok(res),
            Err(e) => errs.push(format!("local_gpu: {e}")),
        }
    }
    if let Some(url) = env_colab_url() {
        match run_colab(&url) {
            Ok(res) => return Ok(res),
            Err(e) => errs.push(format!("remote_colab: {e}")),
        }
    }
    Err(ForgeError::WorkerError(if errs.is_empty() {
        format!("No local GPU worker or FORGE_COLAB_WORKER_URL available for {task}")
    } else {
        errs.join("; ")
    }))
}

/// Restores bandwidth / codecs (`Apollo` / `FlowHigh` / `AudioSR` / `SonicMaster` / `exciter`)
/// trying local GPU first, then `FORGE_COLAB_WORKER_URL` if configured.
pub fn neural_restore_hybrid(
    input_path: &Path,
    output_dir: &Path,
    backend: &str,
    prompt: Option<&str>,
    chunk_secs: Option<f64>,
    timeout: Duration,
) -> Result<NeuralRestoreResult> {
    std::fs::create_dir_all(output_dir)?;
    try_local_then_colab(
        "neural restore",
        |worker| {
            run_local_neural_restore(
                worker, input_path, output_dir, backend, prompt, chunk_secs, timeout,
            )
        },
        |url| {
            colab::restore(
                url, input_path, output_dir, backend, prompt, chunk_secs, timeout,
            )
        },
    )
}

/// Runs neural speech/vocal denoising (`DeepFilterNet3` / `Mel-Band-RoFormer`),
/// trying local GPU first, then `FORGE_COLAB_WORKER_URL` if configured.
pub fn denoise_hybrid(
    input_path: &Path,
    output_dir: &Path,
    backend: &str,
    attenuation_limit_db: Option<f64>,
    timeout: Duration,
) -> Result<NeuralDenoiseResult> {
    std::fs::create_dir_all(output_dir)?;
    try_local_then_colab(
        "neural denoise",
        |worker| {
            run_local_neural_denoise(
                worker,
                input_path,
                output_dir,
                backend,
                attenuation_limit_db,
                timeout,
            )
        },
        |url| {
            colab::denoise(
                url,
                input_path,
                output_dir,
                backend,
                attenuation_limit_db,
                timeout,
            )
        },
    )
}

/// Transcribes polyphonic audio to MIDI via `basic-pitch`,
/// trying local GPU first, then `FORGE_COLAB_WORKER_URL` if configured.
pub fn transcribe_midi_hybrid(
    input_path: &Path,
    output_dir: &Path,
    timeout: Duration,
) -> Result<MidiTranscribeResult> {
    std::fs::create_dir_all(output_dir)?;
    try_local_then_colab(
        "MIDI transcription",
        |worker| run_local_transcribe_midi(worker, input_path, output_dir, timeout),
        |url| colab::transcribe(url, input_path, output_dir, timeout),
    )
}

/// Rescues robotic / phasey AI vocals via `Seed-VC`, `Resemble Enhance`, or verified `RoFormer` chain.
#[allow(clippy::too_many_arguments)]
pub fn vocal_rescue_hybrid(
    input_path: &Path,
    output_dir: &Path,
    backend: &str,
    reference_voice: Option<&Path>,
    dereverb: bool,
    de_aspiration: bool,
    timeout: Duration,
) -> Result<VocalRescueResult> {
    std::fs::create_dir_all(output_dir)?;
    try_local_then_colab(
        "vocal rescue",
        |worker| {
            run_local_vocal_rescue(
                worker,
                input_path,
                output_dir,
                backend,
                reference_voice,
                dereverb,
                de_aspiration,
                timeout,
            )
        },
        |url| {
            colab::vocal_rescue(
                url,
                input_path,
                output_dir,
                backend,
                reference_voice,
                dereverb,
                de_aspiration,
                timeout,
            )
        },
    )
}

/// Tracks beats, downbeats, and tempo drift (`beat_this` / `librosa` via GPU worker, with pure-Rust fallback).
pub fn track_beats_hybrid(
    input_path: &Path,
    hint_bpm: Option<f64>,
    backend: &str,
    timeout: Duration,
) -> Result<BeatGridReport> {
    let mut warnings = Vec::new();
    let allow_worker = !matches!(backend, "rust" | "pure_rust_onset");
    if allow_worker {
        if let Some(worker) = try_local_worker() {
            match run_local_beat_track(&worker, input_path, backend, timeout) {
                Ok(rep) => return Ok(rep),
                Err(e) => warnings.push(format!("local_gpu skipped: {e}")),
            }
        }
        if let Some(url) = env_colab_url() {
            match colab::beat_track(&url, input_path, backend, timeout) {
                Ok(rep) => return Ok(rep),
                Err(e) => warnings.push(format!("remote_colab skipped: {e}")),
            }
        }
    }
    let buf = AudioBuffer::load(input_path)?;
    let mut rep = analyze_beat_grid(&buf, hint_bpm);
    rep.warnings.extend(warnings);
    Ok(rep)
}

fn rust_acoustic_metrics(path: &Path) -> Result<QualityEvalMetrics> {
    let buf = AudioBuffer::load(path)?;
    let audit = audit_audio(&buf);
    let crest_score = ((audit.crest_factor_db - 3.0) / 9.0).clamp(0.0, 1.0);
    let total_frames = buf.num_frames().max(1) as f64;
    let clip_ratio = (audit.clipped_samples as f64 / total_frames).clamp(0.0, 1.0);
    let clip_penalty = (clip_ratio * 25.0).clamp(0.0, 3.0);
    let fizz_penalty = (audit.ai_fizz_index / 100.0) * 2.5;
    let composite = (5.5 + 3.5 * crest_score - clip_penalty - fizz_penalty).clamp(1.0, 10.0);
    Ok(QualityEvalMetrics {
        production_quality: Some((composite * 1000.0).round() / 1000.0),
        production_complexity: None,
        content_enjoyment: None,
        content_usefulness: None,
        singmos: None,
        crest_factor_db: (audit.crest_factor_db * 1000.0).round() / 1000.0,
        hf_ratio_db: (audit.sibilance_5_14khz_db * 1000.0).round() / 1000.0,
        clipping_ratio: clip_ratio,
        composite_score: (composite * 1000.0).round() / 1000.0,
    })
}

/// Evaluates perceptual audio quality (`Audiobox Aesthetics` / `SingMOS`, falling back to Rust acoustic proxy).
pub fn evaluate_quality_hybrid(
    input_path: &Path,
    reference_path: Option<&Path>,
    vocal: bool,
    backend: &str,
    timeout: Duration,
) -> Result<QualityEvalResult> {
    let mut warnings = Vec::new();
    let allow_worker = !matches!(backend, "rust" | "acoustic-proxy");
    if allow_worker {
        if let Some(worker) = try_local_worker() {
            match run_local_evaluate_quality(
                &worker,
                input_path,
                reference_path,
                vocal,
                backend,
                timeout,
            ) {
                Ok(res) => return Ok(res),
                Err(e) => warnings.push(format!("local_gpu skipped: {e}")),
            }
        }
        if let Some(url) = env_colab_url() {
            match colab::evaluate_quality(&url, input_path, reference_path, vocal, backend, timeout)
            {
                Ok(res) => return Ok(res),
                Err(e) => warnings.push(format!("remote_colab skipped: {e}")),
            }
        }
    }
    let metrics = rust_acoustic_metrics(input_path)?;
    let ref_metrics = reference_path.map(rust_acoustic_metrics).transpose()?;
    let (delta, improved) = match &ref_metrics {
        Some(rm) => {
            let mut d = BTreeMap::new();
            let comp_d =
                ((metrics.composite_score - rm.composite_score) * 10_000.0).round() / 10_000.0;
            d.insert("composite_score".to_string(), comp_d);
            d.insert(
                "crest_factor_db".to_string(),
                ((metrics.crest_factor_db - rm.crest_factor_db) * 10_000.0).round() / 10_000.0,
            );
            (Some(d), Some(comp_d >= 0.0))
        }
        None => (None, None),
    };
    Ok(QualityEvalResult {
        backend: "rust-acoustic-proxy".to_string(),
        metrics,
        reference_metrics: ref_metrics,
        delta,
        improved,
        warnings,
    })
}

/// One detected drum trigger event ready to be rendered against an indexed one-shot library.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DrumHit {
    pub frame: usize,
    pub category: &'static str,
    pub velocity: f64,
}

impl DrumHit {
    /// Converts MIDI note `(start_tick, velocity)` pairs at `bpm`/`ppq` into sample-aligned hits.
    pub fn from_ticks(
        category: &'static str,
        ticks_and_vel: impl IntoIterator<Item = (u32, u8)>,
        bpm: f64,
        ppq: u16,
        sample_rate: u32,
    ) -> Vec<Self> {
        if !bpm.is_finite() || bpm <= 0.0 || ppq == 0 || sample_rate == 0 {
            return Vec::new();
        }
        let secs_per_tick = 60.0 / (bpm * f64::from(ppq));
        let sr = f64::from(sample_rate);
        ticks_and_vel
            .into_iter()
            .filter_map(|(tick, vel)| {
                let frame_f = (f64::from(tick) * secs_per_tick * sr).round();
                if !frame_f.is_finite() || frame_f < 0.0 {
                    return None;
                }
                Some(Self {
                    frame: frame_f.min((usize::MAX / 2) as f64) as usize,
                    category,
                    velocity: (f64::from(vel.min(127)) / 127.0).clamp(0.0, 1.0),
                })
            })
            .collect()
    }

    /// Batch-converts `(category, [(start_tick, velocity)])` trigger groups into a single hit list.
    pub fn from_note_groups(
        groups: [(&'static str, Vec<(u32, u8)>); 3],
        bpm: f64,
        ppq: u16,
        sample_rate: u32,
    ) -> Vec<Self> {
        let mut out = Vec::new();
        for (cat, items) in groups {
            out.extend(Self::from_ticks(cat, items, bpm, ppq, sample_rate));
        }
        out
    }
}

fn resample_stereo_oneshot(buf: &AudioBuffer, target_sr: u32) -> [Vec<f64>; 2] {
    let stereo = buf.to_stereo();
    let in_len = stereo.num_frames();
    if in_len == 0 || buf.sample_rate == target_sr {
        return [stereo.channels[0].clone(), stereo.channels[1].clone()];
    }
    let ratio = f64::from(target_sr) / f64::from(buf.sample_rate.max(1));
    let out_len = ((in_len as f64) * ratio).round().max(1.0) as usize;
    let inv_ratio = 1.0 / ratio;
    let mut out = [vec![0.0_f64; out_len], vec![0.0_f64; out_len]];
    for (dst_ch, src_ch) in out.iter_mut().zip(stereo.channels.iter()) {
        for (i, dst) in dst_ch.iter_mut().enumerate() {
            let pos = (i as f64) * inv_ratio;
            let idx = (pos.floor() as usize).min(in_len - 1);
            let frac = pos - (idx as f64);
            let s0 = sanitize_sample_f64(src_ch[idx]);
            let s1 = sanitize_sample_f64(src_ch[(idx + 1).min(in_len - 1)]);
            *dst = s0 + frac * (s1 - s0);
        }
    }
    out
}

/// Renders phase-locked drum sample hits from `library` and blends them with optional `dry_drums`.
/// Returns `(rendered_buffer, hits_rendered)`.
pub fn render_drum_sample_layer(
    frames: usize,
    sample_rate: u32,
    hits: &[DrumHit],
    library: &[DrumSampleDescriptor],
    wet_mix: f64,
    dry_drums: Option<&AudioBuffer>,
) -> Result<(AudioBuffer, usize)> {
    if frames == 0 || sample_rate == 0 {
        return Err(ForgeError::InvalidParameter(
            "render_drum_sample_layer requires non-zero frames and sample_rate".to_string(),
        ));
    }
    if !wet_mix.is_finite() || !(0.0..=1.0).contains(&wet_mix) {
        return Err(ForgeError::InvalidParameter(format!(
            "wet_mix ({wet_mix}) must be finite and in 0.0..=1.0"
        )));
    }
    let mut cache: HashMap<&'static str, [Vec<f64>; 2]> = HashMap::new();
    for cat in ["Kick", "Snare", "HiHat"] {
        if let Some(desc) = library
            .iter()
            .filter(|d| d.category.eq_ignore_ascii_case(cat))
            .max_by(|a, b| {
                a.crest_factor_db
                    .partial_cmp(&b.crest_factor_db)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            && let Ok(sample_buf) = AudioBuffer::load(&desc.path)
        {
            cache.insert(cat, resample_stereo_oneshot(&sample_buf, sample_rate));
        }
    }

    let mut wet = [vec![0.0_f64; frames], vec![0.0_f64; frames]];
    let mut rendered_hits = 0_usize;
    for hit in hits {
        let Some(oneshot) = cache.get(hit.category) else {
            continue;
        };
        if hit.frame >= frames {
            continue;
        }
        let vel = hit.velocity.clamp(0.05, 1.0);
        let copy_len = oneshot[0].len().min(frames - hit.frame);
        for c in 0..2 {
            let dst = &mut wet[c][hit.frame..hit.frame + copy_len];
            let src = &oneshot[c][..copy_len];
            for (d, &s) in dst.iter_mut().zip(src.iter()) {
                *d += s * vel;
            }
        }
        rendered_hits += 1;
    }

    if let Some(dry) = dry_drums {
        if dry.sample_rate != sample_rate {
            return Err(ForgeError::InvalidParameter(format!(
                "dry_drums sample rate ({}) must match target ({sample_rate})",
                dry.sample_rate
            )));
        }
        let dry_stereo = dry.to_stereo();
        let n = frames.min(dry_stereo.num_frames());
        let dry_w = 1.0 - wet_mix;
        for (wet_ch, dry_ch) in wet.iter_mut().zip(dry_stereo.channels.iter()) {
            for (w, &d) in wet_ch[..n].iter_mut().zip(dry_ch[..n].iter()) {
                *w = *w * wet_mix + sanitize_sample_f64(d) * dry_w;
            }
        }
    }

    Ok((
        AudioBuffer::from_channels(vec![wet[0].clone(), wet[1].clone()], sample_rate)?,
        rendered_hits,
    ))
}

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
        .connect_timeout(Duration::from_secs(15))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| ForgeError::WorkerError(format!("HTTP client init failed: {e}")))?;

    let payload = serde_json::json!({
        "model": model_id,
        "input": prompt,
        "response_format": { "type": "audio" }
    });

    let mut resp = client
        .post("https://generativelanguage.googleapis.com/v1beta/interactions")
        .header("x-goog-api-key", api_key)
        .json(&payload)
        .send()
        .await
        .map_err(|e| ForgeError::WorkerError(format!("Lyria API request failed: {e}")))?;

    let status = resp.status();
    if !status.is_success() {
        const MAX_ERROR_BODY_BYTES: usize = 64 * 1024;
        let mut err_raw = Vec::new();
        while let Some(chunk) = resp
            .chunk()
            .await
            .map_err(|e| ForgeError::WorkerError(format!("Lyria response stream failed: {e}")))?
        {
            let remaining = MAX_ERROR_BODY_BYTES.saturating_sub(err_raw.len());
            if remaining == 0 {
                break;
            }
            let take = chunk.len().min(remaining);
            err_raw.extend_from_slice(&chunk[..take]);
            if err_raw.len() >= MAX_ERROR_BODY_BYTES {
                break;
            }
        }
        let body = String::from_utf8_lossy(&err_raw);
        return Err(ForgeError::WorkerError(format!(
            "Lyria API returned HTTP {status}: {}",
            body.chars().take(1024).collect::<String>()
        )));
    }

    const MAX_LYRIA_RESPONSE_BYTES: usize = 256 * 1024 * 1024;
    if let Some(len) = resp.content_length()
        && len > MAX_LYRIA_RESPONSE_BYTES as u64
    {
        return Err(ForgeError::WorkerError(format!(
            "Lyria response ({len} bytes) exceeds safety limit ({MAX_LYRIA_RESPONSE_BYTES} bytes)"
        )));
    }
    let mut raw = Vec::new();
    while let Some(chunk) = resp
        .chunk()
        .await
        .map_err(|e| ForgeError::WorkerError(format!("Lyria response stream failed: {e}")))?
    {
        if raw.len().saturating_add(chunk.len()) > MAX_LYRIA_RESPONSE_BYTES {
            return Err(ForgeError::WorkerError(format!(
                "Lyria response exceeded safety limit ({MAX_LYRIA_RESPONSE_BYTES} bytes)"
            )));
        }
        raw.extend_from_slice(&chunk);
    }

    let body_json: serde_json::Value = serde_json::from_slice(&raw)
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
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::write(output_audio_path, &audio_bytes).await?;

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
                ..SeparationConfig::default()
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

    #[test]
    fn drum_hits_from_ticks_render_layer_and_rust_hybrid_fallbacks() {
        let dir = tempfile::tempdir().unwrap();
        let samples_dir = dir.path().join("samples");
        std::fs::create_dir_all(&samples_dir).unwrap();

        let sr = 48_000_u32;
        // Write Kick, Snare, and HiHat one-shots so indexer classifies each category
        let write_shot = |name: &str, freq: f64| {
            let ch: Vec<f64> = (0..2400)
                .map(|i| {
                    0.8 * (-(i as f64) / 300.0).exp()
                        * (2.0 * std::f64::consts::PI * freq * (i as f64) / f64::from(sr)).sin()
                })
                .collect();
            AudioBuffer::from_channels(vec![ch.clone(), ch], sr)
                .unwrap()
                .save_wav(samples_dir.join(name), 24)
                .unwrap();
        };
        write_shot("kick.wav", 65.0);
        write_shot("snare.wav", 950.0);
        write_shot("hat.wav", 8500.0);

        let lib = index_drum_sample_library(&samples_dir).unwrap();
        let hits = DrumHit::from_note_groups(
            [
                ("Kick", vec![(0, 127)]),
                ("Snare", vec![(480, 100)]),
                ("HiHat", vec![(240, 80)]),
            ],
            120.0,
            480,
            sr,
        );
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].frame, 0);
        assert_eq!(hits[1].frame, 24_000);

        let dry = AudioBuffer::from_channels(vec![vec![0.1; 48_000]; 2], sr).unwrap();
        let (layered, count) =
            render_drum_sample_layer(48_000, sr, &hits, &lib, 0.5, Some(&dry)).unwrap();
        assert!(count >= 1);
        assert_eq!(layered.num_frames(), 48_000);

        // Beat click WAV for pure-Rust hybrid fallbacks
        let beat_wav = dir.path().join("beats.wav");
        let frames = (sr as usize) * 4;
        let mut beat_ch = vec![0.0_f64; frames];
        for n in 0..8 {
            let start = n * 24_000;
            for k in 0..400 {
                if start + k < frames {
                    beat_ch[start + k] = 0.8 * (-(k as f64) / 80.0).exp();
                }
            }
        }
        AudioBuffer::from_channels(vec![beat_ch.clone(), beat_ch], sr)
            .unwrap()
            .save_wav(&beat_wav, 24)
            .unwrap();

        let grid = track_beats_hybrid(&beat_wav, Some(120.0), "rust", Duration::from_secs(5))
            .expect("rust beat tracking fallback");
        assert_eq!(grid.backend, "pure_rust_onset");
        assert!(!grid.beats_secs.is_empty());

        let eval = evaluate_quality_hybrid(
            &beat_wav,
            Some(&beat_wav),
            true,
            "rust",
            Duration::from_secs(5),
        )
        .expect("rust acoustic proxy fallback");
        assert_eq!(eval.backend, "rust-acoustic-proxy");
        assert!(eval.metrics.production_quality.is_some_and(f64::is_finite));
        assert!(eval.reference_metrics.is_some());
    }
}
