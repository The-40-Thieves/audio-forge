//! Multi-Mic Phase & Polarity Aligner (GCC-PHAT Cross-Correlation) and
//! FastDTW Vocal Double / Harmony Time-Aligner (VocAlign equivalent).

use forge_core::{AudioBuffer, ForgeError, Result};
use rustfft::{num_complex::Complex, FftPlanner};
use serde::{Deserialize, Serialize};

/// Result of GCC-PHAT phase and delay analysis between a reference and target track.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PhaseAlignmentResult {
    /// Detected lag of target relative to reference in samples (`> 0` means target is delayed).
    pub delay_samples: isize,
    /// Detected lag in milliseconds.
    pub delay_ms: f64,
    /// Optimal polarity (`false` = normal `+1`, `true` = invert phase `-1`).
    pub invert_polarity: bool,
    /// Cross-correlation peak confidence (`0.0..=1.0`).
    pub correlation_peak: f64,
}

/// Estimates sample delay and optimal polarity (`+1` vs `-1`) via Generalized Cross-Correlation
/// with Phase Transform (GCC-PHAT) and aligns `target` to `reference` (Sound Radix Auto-Align equivalent).
pub fn align_phase_gcc_phat(
    reference: &AudioBuffer,
    target: &AudioBuffer,
    max_search_ms: f64,
) -> Result<(AudioBuffer, PhaseAlignmentResult)> {
    if reference.sample_rate != target.sample_rate {
        return Err(ForgeError::InvalidParameter(
            "Reference and target must have identical sample rates for GCC-PHAT alignment"
                .to_string(),
        ));
    }
    let sr = f64::from(reference.sample_rate.max(1));
    let ref_mono = reference.to_mono();
    let tgt_mono = target.to_mono();
    let n = ref_mono.len().min(tgt_mono.len());
    if n < 64 {
        return Ok((
            target.clone(),
            PhaseAlignmentResult {
                delay_samples: 0,
                delay_ms: 0.0,
                invert_polarity: false,
                correlation_peak: 1.0,
            },
        ));
    }

    // Analyze up to 65536 samples around the highest-energy window for speed and precision
    let fft_len = n.min(65536).next_power_of_two();
    let slice_len = n.min(fft_len / 2);

    let mut planner = FftPlanner::<f64>::new();
    let fft_fwd = planner.plan_fft_forward(fft_len);
    let fft_inv = planner.plan_fft_inverse(fft_len);

    let mut r_spec = vec![Complex::new(0.0, 0.0); fft_len];
    let mut t_spec = vec![Complex::new(0.0, 0.0); fft_len];
    for i in 0..slice_len {
        r_spec[i] = Complex::new(ref_mono[i], 0.0);
        t_spec[i] = Complex::new(tgt_mono[i], 0.0);
    }

    fft_fwd.process(&mut r_spec);
    fft_fwd.process(&mut t_spec);

    // Standard Cross-Correlation + mild PHAT weighting to handle both transients and bass phase
    let mut cross = vec![Complex::new(0.0, 0.0); fft_len];
    for i in 0..fft_len {
        let prod = t_spec[i] * r_spec[i].conj();
        let mag = prod.norm().max(1e-12);
        cross[i] = prod / mag.powf(0.5);
    }
    fft_inv.process(&mut cross);

    let max_lag = (((sr * max_search_ms / 1000.0).round() as isize).max(1) as usize)
        .min(fft_len / 2 - 1);

    let mut best_lag: isize = 0;
    let mut best_val: f64 = 0.0;

    for lag in -(max_lag as isize)..=(max_lag as isize) {
        let idx = if lag >= 0 {
            lag as usize
        } else {
            (fft_len as isize + lag) as usize
        };
        let val = cross[idx].re / (fft_len as f64);
        if val.abs() > best_val.abs() {
            best_val = val;
            best_lag = lag;
        }
    }

    let invert_polarity = best_val < 0.0;
    let sign = if invert_polarity { -1.0 } else { 1.0 };

    // Shift and polarity-correct all channels of target
    let frames = target.num_frames();
    let mut aligned_channels = Vec::with_capacity(target.num_channels());
    for ch in &target.channels {
        let mut out_ch = vec![0.0_f64; frames];
        for (i, sample_out) in out_ch.iter_mut().enumerate() {
            let src_idx = (i as isize) + best_lag;
            if src_idx >= 0 && (src_idx as usize) < frames {
                *sample_out = ch[src_idx as usize] * sign;
            }
        }
        aligned_channels.push(out_ch);
    }

    let result = PhaseAlignmentResult {
        delay_samples: best_lag,
        delay_ms: (best_lag as f64) * 1000.0 / sr,
        invert_polarity,
        correlation_peak: best_val.abs().clamp(0.0, 1.0),
    };

    Ok((
        AudioBuffer {
            channels: aligned_channels,
            sample_rate: target.sample_rate,
        },
        result,
    ))
}

/// A REAPER-compatible stretch marker mapping `source_pos_secs` to `timeline_pos_secs`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StretchMarker {
    pub timeline_pos_secs: f64,
    pub source_pos_secs: f64,
}

/// Time-aligns a vocal double/harmony (`dub`) to a lead vocal (`guide`) using banded Dynamic Time Warping
/// (Synchro Arts VocAlign equivalent) and emits native REAPER `SM` stretch markers.
pub fn align_vocal_double_dtw(
    guide: &AudioBuffer,
    dub: &AudioBuffer,
    max_warp_ms: f64,
) -> Result<(AudioBuffer, Vec<StretchMarker>)> {
    if guide.sample_rate != dub.sample_rate {
        return Err(ForgeError::InvalidParameter(
            "Guide and dub vocals must share the same sample rate".to_string(),
        ));
    }
    let sr = f64::from(guide.sample_rate.max(1));
    let g_mono = guide.to_mono();
    let d_mono = dub.to_mono();
    let frames = g_mono.len().min(d_mono.len());
    if frames < 1024 {
        return Ok((dub.clone(), Vec::new()));
    }

    // Extract 10ms energy + zero-crossing (sibilance/consonant) feature vectors
    let hop = ((sr * 0.010).round() as usize).max(64);
    let num_steps = frames / hop;
    if num_steps < 4 {
        return Ok((dub.clone(), Vec::new()));
    }

    let extract_features = |sig: &[f64]| -> Vec<(f64, f64)> {
        let mut feats = Vec::with_capacity(num_steps);
        for s in 0..num_steps {
            let slice = &sig[s * hop..(s + 1) * hop];
            let mut sum_sq = 0.0;
            let mut zcr = 0.0;
            for i in 0..slice.len() {
                sum_sq += slice[i] * slice[i];
                if i > 0 && (slice[i] >= 0.0) != (slice[i - 1] >= 0.0) {
                    zcr += 1.0;
                }
            }
            let log_e = 10.0 * (sum_sq / (slice.len() as f64) + 1e-9).log10();
            let norm_zcr = zcr / (slice.len() as f64);
            feats.push((log_e, norm_zcr * 40.0));
        }
        feats
    };

    let g_feats = extract_features(&g_mono);
    let d_feats = extract_features(&d_mono);

    let band = (((max_warp_ms / 10.0).round() as usize).max(2)).min(num_steps / 2);
    let mut cost = vec![vec![f64::INFINITY; num_steps]; num_steps];
    let mut back = vec![vec![(0_usize, 0_usize); num_steps]; num_steps];
    cost[0][0] = 0.0;

    for i in 0..num_steps {
        let j_min = i.saturating_sub(band);
        let j_max = (i + band).min(num_steps - 1);
        for j in j_min..=j_max {
            let dist = (g_feats[i].0 - d_feats[j].0).abs() + (g_feats[i].1 - d_feats[j].1).abs();
            if i == 0 && j == 0 {
                cost[0][0] = dist;
                continue;
            }
            let mut best_prev = f64::INFINITY;
            let mut best_idx = (0, 0);
            if i > 0 && j > 0 && cost[i - 1][j - 1] < best_prev {
                best_prev = cost[i - 1][j - 1];
                best_idx = (i - 1, j - 1);
            }
            if i > 0 && cost[i - 1][j] + 1.5 < best_prev {
                best_prev = cost[i - 1][j] + 1.5;
                best_idx = (i - 1, j);
            }
            if j > 0 && cost[i][j - 1] + 1.5 < best_prev {
                best_prev = cost[i][j - 1] + 1.5;
                best_idx = (i, j - 1);
            }
            cost[i][j] = best_prev + dist;
            back[i][j] = best_idx;
        }
    }

    // Backtrack optimal warping path
    let mut path = Vec::with_capacity(num_steps * 2);
    let (mut ci, mut cj) = (num_steps - 1, num_steps - 1);
    while ci > 0 || cj > 0 {
        path.push((ci, cj));
        let prev = back[ci][cj];
        if prev == (ci, cj) {
            break;
        }
        ci = prev.0;
        cj = prev.1;
    }
    path.push((0, 0));
    path.reverse();

    // Map each guide step `i` to dub step `j`
    let mut step_map = vec![0.0_f64; num_steps];
    for &(i, j) in &path {
        step_map[i] = j as f64;
    }

    // Emit decimated REAPER StretchMarkers every ~250ms where timing diverges
    let mut markers = Vec::new();
    for i in (0..num_steps).step_by(25) {
        let t_sec = (i * hop) as f64 / sr;
        let s_sec = (step_map[i] * (hop as f64)) / sr;
        markers.push(StretchMarker {
            timeline_pos_secs: t_sec,
            source_pos_secs: s_sec,
        });
    }

    // Resample dub channels smoothly along the warped contour
    let out_frames = guide.num_frames();
    let dub_frames = dub.num_frames();
    let mut warped_channels = Vec::with_capacity(dub.num_channels());

    for ch in &dub.channels {
        let mut out_ch = vec![0.0_f64; out_frames];
        for (f, sample_out) in out_ch.iter_mut().enumerate() {
            let step_f = (f as f64) / (hop as f64);
            let s0 = (step_f.floor() as usize).min(num_steps - 1);
            let s1 = (s0 + 1).min(num_steps - 1);
            let frac = step_f - (s0 as f64);
            let mapped_step = step_map[s0] * (1.0 - frac) + step_map[s1] * frac;
            let src_f = (mapped_step * (hop as f64)).clamp(0.0, (dub_frames.saturating_sub(2)) as f64);
            let idx0 = src_f.floor() as usize;
            let idx_frac = src_f - (idx0 as f64);
            *sample_out = ch[idx0] * (1.0 - idx_frac) + ch[(idx0 + 1).min(dub_frames - 1)] * idx_frac;
        }
        warped_channels.push(out_ch);
    }

    Ok((
        AudioBuffer {
            channels: warped_channels,
            sample_rate: dub.sample_rate,
        },
        markers,
    ))
}
