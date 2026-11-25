//! Onset/DP beat-grid tracker, tempo-drift analyzer, and pitch-preserving stereo-locked WSOLA
//! time-warper for quantized AI tracks and variable REAPER `<TEMPOENVEX>` tempo maps.

use std::f64::consts::{FRAC_1_SQRT_2, PI};

use forge_core::{AudioBuffer, ForgeError, Result};
use serde::{Deserialize, Serialize};

use crate::dtw_align::StretchMarker;
use crate::loudness_limiter::{Biquad, sanitize_param_f64, sanitize_sample_f64};

/// A REAPER `<TEMPOENVEX>` / beat-grid tempo node in beats per minute at `time_secs`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TempoPoint {
    pub time_secs: f64,
    pub bpm: f64,
    pub time_sig_num: u8,
    pub time_sig_den: u8,
}

/// Complete beat, downbeat, and tempo-drift telemetry report for a track or drum stem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeatGridReport {
    pub beats_secs: Vec<f64>,
    pub downbeats_secs: Vec<f64>,
    pub median_bpm: f64,
    pub max_bpm_drift: f64,
    pub tempo_map: Vec<TempoPoint>,
    #[serde(default)]
    pub backend: String,
    #[serde(default)]
    pub warnings: Vec<String>,
}

fn empty_beat_grid(fallback_bpm: f64) -> BeatGridReport {
    BeatGridReport {
        beats_secs: Vec::new(),
        downbeats_secs: Vec::new(),
        median_bpm: fallback_bpm,
        max_bpm_drift: 0.0,
        tempo_map: Vec::new(),
        backend: "pure_rust_onset".to_string(),
        warnings: Vec::new(),
    }
}

/// Estimates beat positions, downbeats, and per-bar tempo drift using a 10 ms sub+broadband
/// onset-flux envelope and Ellis-style dynamic programming beat tracking.
pub fn analyze_beat_grid(buffer: &AudioBuffer, hint_bpm: Option<f64>) -> BeatGridReport {
    let sr = f64::from(buffer.sample_rate.max(1));
    let frames = buffer.num_frames();
    let fallback_bpm = hint_bpm
        .filter(|b| b.is_finite() && (30.0..=300.0).contains(b))
        .unwrap_or(120.0);

    let hop = ((sr * 0.010).round() as usize).clamp(32, 2048);
    if frames < hop * 16 || buffer.num_channels() == 0 {
        return empty_beat_grid(fallback_bpm);
    }

    let mono = buffer.to_mono();
    let mut sub_lp = Biquad::lowpass(sr, 240.0_f64.min(sr * 0.45), FRAC_1_SQRT_2);
    let num_hops = frames / hop;
    let mut onset = vec![0.0_f64; num_hops];
    let mut prev_broad = 0.0_f64;
    let mut prev_sub = 0.0_f64;

    for (h, slot) in onset.iter_mut().enumerate() {
        let slice = &mono[h * hop..(h + 1) * hop];
        let mut broad_e = 0.0_f64;
        let mut sub_e = 0.0_f64;
        for &raw in slice {
            let x = sanitize_sample_f64(raw);
            let s = sub_lp.process_sample(x);
            broad_e += x * x;
            sub_e += s * s;
        }
        let b_rms = (broad_e / (hop as f64)).sqrt();
        let s_rms = (sub_e / (hop as f64)).sqrt();
        *slot = (b_rms - prev_broad).max(0.0) + 1.5 * (s_rms - prev_sub).max(0.0);
        prev_broad = b_rms;
        prev_sub = s_rms;
    }

    let max_onset = onset.iter().copied().fold(0.0_f64, f64::max);
    if max_onset <= 1e-9 {
        return empty_beat_grid(fallback_bpm);
    }
    for v in &mut onset {
        *v /= max_onset;
    }

    let hop_secs = (hop as f64) / sr;
    let min_lag = (((60.0 / 200.0) / hop_secs).round() as usize).max(4);
    let max_lag = (((60.0 / 55.0) / hop_secs).round() as usize).min(num_hops / 2);
    let hint_lag = hint_bpm
        .filter(|b| b.is_finite() && (40.0..=260.0).contains(b))
        .map(|b| (60.0 / b) / hop_secs);

    let mut best_lag = min_lag.clamp(4, (num_hops / 2).max(5));
    let mut best_score = -1.0_f64;
    for lag in min_lag..=max_lag {
        let mut corr = 0.0_f64;
        for i in 0..(num_hops - lag) {
            corr += onset[i] * onset[i + lag];
        }
        if let Some(hl) = hint_lag {
            let log_err = ((lag as f64) / hl).ln();
            corr *= 1.0 + 0.35 * (-0.5 * (log_err / 0.18).powi(2)).exp();
        }
        if corr > best_score {
            best_score = corr;
            best_lag = lag;
        }
    }

    // Ellis DP beat tracking (handles moderate tempo drift gracefully).
    let period = best_lag as f64;
    let d_lo = ((period * 0.65).floor() as usize).max(3);
    let d_hi = ((period * 1.45).ceil() as usize).max(d_lo + 1);
    let alpha = 3.5_f64;
    let mut dp = vec![0.0_f64; num_hops];
    let mut back = vec![usize::MAX; num_hops];

    for t in 0..num_hops {
        let mut best_prev = 0.0_f64;
        let mut best_idx = usize::MAX;
        if t >= d_lo {
            let max_d = d_hi.min(t);
            for d in d_lo..=max_d {
                let p_idx = t - d;
                let dev = ((d as f64) / period).ln();
                let cand = dp[p_idx] - alpha * dev * dev;
                if cand > best_prev || best_idx == usize::MAX {
                    best_prev = cand;
                    best_idx = p_idx;
                }
            }
        }
        dp[t] = onset[t] + best_prev;
        back[t] = best_idx;
    }

    let tail_start = num_hops.saturating_sub(d_hi.max(1));
    let mut cur = (tail_start..num_hops)
        .max_by(|&a, &b| {
            dp[a]
                .partial_cmp(&dp[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(num_hops - 1);

    let mut beat_hops = Vec::new();
    while cur < num_hops {
        beat_hops.push(cur);
        cur = back[cur];
    }
    beat_hops.reverse();

    // Trim weak lead-in before the first real onset so beats align cleanly.
    while beat_hops.len() > 2 && onset[beat_hops[0]] < 0.02 {
        beat_hops.remove(0);
    }

    let beats_secs: Vec<f64> = beat_hops.iter().map(|&h| h as f64 * hop_secs).collect();
    if beats_secs.len() < 2 {
        return BeatGridReport {
            beats_secs,
            downbeats_secs: Vec::new(),
            median_bpm: (60.0 / (period * hop_secs)).clamp(40.0, 260.0),
            max_bpm_drift: 0.0,
            tempo_map: Vec::new(),
            backend: "pure_rust_onset".to_string(),
            warnings: Vec::new(),
        };
    }

    // Pick downbeat phase 0..4 maximizing summed onset strength.
    let best_phase = (0..4)
        .max_by(|&p1, &p2| {
            let sum_phase = |p: usize| -> f64 {
                beat_hops[p.min(beat_hops.len() - 1)..]
                    .iter()
                    .step_by(4)
                    .map(|&h| onset[h])
                    .sum()
            };
            sum_phase(p1)
                .partial_cmp(&sum_phase(p2))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    let downbeats_secs: Vec<f64> = beats_secs[best_phase.min(beats_secs.len() - 1)..]
        .iter()
        .step_by(4)
        .copied()
        .collect();

    let mut bpms: Vec<f64> = beats_secs
        .windows(2)
        .filter_map(|w| {
            let dt = w[1] - w[0];
            (dt > 0.1).then(|| (60.0 / dt).clamp(40.0, 260.0))
        })
        .collect();
    if bpms.is_empty() {
        bpms.push(fallback_bpm);
    }
    let mut sorted = bpms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_bpm = sorted[sorted.len() / 2];
    let max_bpm_drift = bpms
        .iter()
        .map(|&b| (b - median_bpm).abs())
        .fold(0.0_f64, f64::max);

    let anchors = if downbeats_secs.len() >= 2 {
        &downbeats_secs[..]
    } else {
        &beats_secs[..]
    };
    let beats_per_anchor = if downbeats_secs.len() >= 2 { 4.0 } else { 1.0 };
    let tempo_map: Vec<TempoPoint> = anchors
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            let bpm = if i + 1 < anchors.len() {
                let dt = (anchors[i + 1] - t).max(0.15);
                ((60.0 * beats_per_anchor) / dt).clamp(40.0, 260.0)
            } else {
                median_bpm
            };
            TempoPoint {
                time_secs: t,
                bpm,
                time_sig_num: 4,
                time_sig_den: 4,
            }
        })
        .collect();

    BeatGridReport {
        beats_secs,
        downbeats_secs,
        median_bpm,
        max_bpm_drift,
        tempo_map,
        backend: "pure_rust_onset".to_string(),
        warnings: Vec::new(),
    }
}

/// Renders a pitch-preserving, stereo-phase-locked Waveform Similarity Overlap-Add (WSOLA)
/// time-warped `AudioBuffer` mapping output frame `f` to `timeline_to_source(f)`.
pub fn warp_buffer_wsola(
    buffer: &AudioBuffer,
    timeline_to_source: impl Fn(usize) -> f64,
    out_frames: usize,
) -> AudioBuffer {
    let in_frames = buffer.num_frames();
    let num_ch = buffer.num_channels();
    if in_frames == 0 || num_ch == 0 || out_frames == 0 {
        return AudioBuffer {
            channels: vec![vec![0.0_f64; out_frames]; num_ch.max(1)],
            sample_rate: buffer.sample_rate,
        };
    }

    let sr = f64::from(buffer.sample_rate.max(1));
    let win_len = ((((sr * 0.032).round() as usize).clamp(128, 2048)) & !1).min(in_frames & !1);
    if win_len < 16 {
        return AudioBuffer {
            channels: buffer
                .channels
                .iter()
                .map(|ch| {
                    (0..out_frames)
                        .map(|f| {
                            let s = timeline_to_source(f).round() as usize;
                            ch.get(s).copied().unwrap_or(0.0)
                        })
                        .collect()
                })
                .collect(),
            sample_rate: buffer.sample_rate,
        };
    }

    let hop = win_len / 2;
    let tol = (win_len / 4).clamp(8, 256);
    let window: Vec<f64> = (0..win_len)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * (i as f64) / (win_len as f64)).cos()))
        .collect();

    let mono = buffer.to_mono();
    let mut out_ch = vec![vec![0.0_f64; out_frames]; num_ch];
    let mut norm = vec![0.0_f64; out_frames];

    let max_src_start = in_frames.saturating_sub(win_len);
    let mut prev_natural_next = 0_usize;
    let mut out_pos = 0_usize;

    while out_pos < out_frames {
        let src_center = timeline_to_source(out_pos);
        if !src_center.is_finite() || src_center < -(hop as f64) || src_center >= in_frames as f64 {
            out_pos += hop;
            continue;
        }
        let nominal = src_center.round().max(0.0) as usize;
        if nominal >= in_frames {
            out_pos += hop;
            continue;
        }
        let src_start =
            if out_pos == 0 || nominal >= max_src_start || prev_natural_next >= max_src_start {
                nominal
            } else {
                // Find candidate grain start within nominal ± tol maximizing waveform correlation
                // against the natural continuation `mono[prev_natural_next .. prev_natural_next + hop]`.
                let ref_slice = &mono[prev_natural_next..(prev_natural_next + hop).min(in_frames)];
                let ref_len = ref_slice.len();
                let lo = nominal.saturating_sub(tol).min(max_src_start);
                let hi = (nominal + tol).min(max_src_start);
                let mut best_pos = nominal;
                let mut best_sim = f64::NEG_INFINITY;
                let ref_e: f64 = ref_slice.iter().map(|&x| x * x).sum();

                for cand in lo..=hi {
                    let cand_slice = &mono[cand..cand + ref_len];
                    let mut dot = 0.0_f64;
                    let mut cand_e = 0.0_f64;
                    for (&a, &b) in ref_slice.iter().zip(cand_slice.iter()) {
                        dot += a * b;
                        cand_e += b * b;
                    }
                    let denom = (ref_e * cand_e).sqrt();
                    let sim = if denom > 1e-12 {
                        dot / denom
                    } else if cand == nominal {
                        1.0
                    } else {
                        0.0
                    };
                    if sim > best_sim {
                        best_sim = sim;
                        best_pos = cand;
                    }
                }
                best_pos
            };

        let copy_len = win_len.min(out_frames - out_pos).min(in_frames - src_start);
        for (dst, src) in out_ch.iter_mut().zip(buffer.channels.iter()) {
            for i in 0..copy_len {
                dst[out_pos + i] += sanitize_sample_f64(src[src_start + i]) * window[i];
            }
        }
        for (i, &w) in window[..copy_len].iter().enumerate() {
            norm[out_pos + i] += w;
        }

        prev_natural_next = (src_start + hop).min(in_frames);
        out_pos += hop;
    }

    for dst in &mut out_ch {
        for (s, &n) in dst.iter_mut().zip(norm.iter()) {
            if n > 1e-6 {
                *s /= n;
            }
        }
    }

    AudioBuffer {
        channels: out_ch,
        sample_rate: buffer.sample_rate,
    }
}

/// Time-warps `buffer` cleanly to a constant `target_bpm` using tracked beat timestamps and
/// pitch-preserving WSOLA, returning both the quantized audio and REAPER [`StretchMarker`]s.
pub fn quantize_to_constant_bpm(
    buffer: &AudioBuffer,
    beats_secs: &[f64],
    target_bpm: f64,
) -> Result<(AudioBuffer, Vec<StretchMarker>)> {
    if !target_bpm.is_finite() || !(30.0..=300.0).contains(&target_bpm) {
        return Err(ForgeError::InvalidParameter(format!(
            "target_bpm ({target_bpm}) must be finite and in 30.0..=300.0"
        )));
    }
    if beats_secs.len() < 2 {
        return Err(ForgeError::InvalidParameter(
            "quantize_to_constant_bpm requires at least 2 beat timestamps".to_string(),
        ));
    }
    for w in beats_secs.windows(2) {
        if !w[0].is_finite() || !w[1].is_finite() || w[1] <= w[0] {
            return Err(ForgeError::InvalidParameter(
                "beat timestamps must be finite and strictly increasing".to_string(),
            ));
        }
    }

    let sr = f64::from(buffer.sample_rate.max(1));
    let beat_period = 60.0 / sanitize_param_f64(target_bpm, 120.0, 30.0, 300.0);
    let anchor0 = beats_secs[0];
    let markers: Vec<StretchMarker> = beats_secs
        .iter()
        .enumerate()
        .map(|(k, &src_t)| StretchMarker {
            timeline_pos_secs: anchor0 + (k as f64) * beat_period,
            source_pos_secs: src_t,
        })
        .collect();

    let total_src_secs = buffer.duration_secs();
    let last_m = markers[markers.len() - 1];
    let out_secs = (last_m.timeline_pos_secs + (total_src_secs - last_m.source_pos_secs)).max(0.0);
    let out_frames = ((out_secs * sr).round() as usize).max(1);

    let warped = warp_buffer_wsola(
        buffer,
        |f| {
            let t = (f as f64) / sr;
            if t <= markers[0].timeline_pos_secs {
                return (t + (markers[0].source_pos_secs - markers[0].timeline_pos_secs)) * sr;
            }
            if t >= last_m.timeline_pos_secs {
                return (last_m.source_pos_secs + (t - last_m.timeline_pos_secs)) * sr;
            }
            let seg = ((t - markers[0].timeline_pos_secs) / beat_period).floor() as usize;
            let idx = seg.min(markers.len() - 2);
            let m0 = markers[idx];
            let m1 = markers[idx + 1];
            let alpha = (t - m0.timeline_pos_secs) / (m1.timeline_pos_secs - m0.timeline_pos_secs);
            (m0.source_pos_secs + alpha * (m1.source_pos_secs - m0.source_pos_secs)) * sr
        },
        out_frames,
    );

    Ok((warped, markers))
}
