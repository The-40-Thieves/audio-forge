//! Multi-Mic Phase & Polarity Aligner (GCC-PHAT Cross-Correlation).
//!
//! The banded DTW vocal aligner lives in [`crate::dtw_align`]; its public items are re-exported
//! here so existing `phase_align::` paths keep working.

use forge_core::{AudioBuffer, ForgeError, Result};
use rustfft::{FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};

pub use crate::dtw_align::{StretchMarker, align_vocal_double_dtw};

/// Result of GCC-PHAT phase and delay analysis between a reference and target track.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PhaseAlignmentResult {
    /// Detected lag of target relative to reference in samples (`> 0` means target is delayed).
    pub delay_samples: isize,
    /// Detected lag in milliseconds.
    pub delay_ms: f64,
    /// Optimal polarity (`false` = normal `+1`, `true` = invert phase `-1`).
    pub invert_polarity: bool,
    /// Energy-normalized correlation of reference and shifted target over the analysis window
    /// (`0.0..=1.0`): near 1 for matching signals, near 0 for unrelated ones. Inputs shorter than
    /// 64 samples are not analyzed and report 0.
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
        // Too short to analyze: pass the target through and report no confidence in the result.
        return Ok((
            target.clone(),
            PhaseAlignmentResult {
                delay_samples: 0,
                delay_ms: 0.0,
                invert_polarity: false,
                correlation_peak: 0.0,
            },
        ));
    }

    // Analyze the highest-energy window of up to 32768 samples, zero-padded to twice its length
    // so the circular cross-correlation equals the linear one at every searched lag.
    let fft_len = n.min(65536).next_power_of_two();
    let slice_len = n.min(fft_len / 2);
    let start = highest_energy_window(&ref_mono[..n], &tgt_mono[..n], slice_len);
    let ref_w = &ref_mono[start..start + slice_len];
    let tgt_w = &tgt_mono[start..start + slice_len];

    let max_lag = max_lag_samples(sr, max_search_ms, fft_len);
    let (best_lag, frac_lag, best_val) = gcc_phat_peak(ref_w, tgt_w, fft_len, max_lag);
    let invert_polarity = best_val < 0.0;
    let sign = if invert_polarity { -1.0 } else { 1.0 };

    let result = PhaseAlignmentResult {
        delay_samples: best_lag,
        delay_ms: frac_lag * 1000.0 / sr,
        invert_polarity,
        correlation_peak: normalized_correlation(ref_w, tgt_w, best_lag),
    };
    Ok((shift_target_fractional(target, frac_lag, sign), result))
}

/// Largest searched lag in samples for a `max_search_ms` window.
fn max_lag_samples(sr: f64, max_search_ms: f64, fft_len: usize) -> usize {
    let clean_search_ms =
        crate::loudness_limiter::sanitize_param_f64(max_search_ms, 20.0, 0.5, 500.0);
    // `n >= 64` in the caller already keeps `fft_len / 2 >= 32`; saturate anyway so the bound
    // stays safe if that guard ever changes.
    (((sr * clean_search_ms / 1000.0).round() as isize).max(1) as usize)
        .min((fft_len / 2).saturating_sub(1))
}

/// Start of the `len`-sample window with the most combined reference and target energy.
/// Ties keep the earliest window, so an all-silent input analyzes from the start.
fn highest_energy_window(reference: &[f64], target: &[f64], len: usize) -> usize {
    let power = |i: usize| reference[i] * reference[i] + target[i] * target[i];
    let mut energy: f64 = (0..len).map(power).sum();
    let (mut best_start, mut best_energy) = (0, energy);
    for start in 1..=(reference.len() - len) {
        energy += power(start + len - 1) - power(start - 1);
        if energy > best_energy {
            best_energy = energy;
            best_start = start;
        }
    }
    best_start
}

/// Integer lag, parabolic sub-sample lag, and signed value of the mildly PHAT-weighted
/// cross-correlation peak within `±max_lag`.
fn gcc_phat_peak(
    ref_w: &[f64],
    tgt_w: &[f64],
    fft_len: usize,
    max_lag: usize,
) -> (isize, f64, f64) {
    let mut planner = FftPlanner::<f64>::new();
    let fft_fwd = planner.plan_fft_forward(fft_len);
    let fft_inv = planner.plan_fft_inverse(fft_len);
    let scratch_len = fft_fwd
        .get_inplace_scratch_len()
        .max(fft_inv.get_inplace_scratch_len());
    let mut scratch = vec![Complex::new(0.0, 0.0); scratch_len];

    let mut spectrum = |window: &[f64]| {
        let mut spec = vec![Complex::new(0.0, 0.0); fft_len];
        for (bin, &x) in spec.iter_mut().zip(window) {
            *bin = Complex::new(x, 0.0);
        }
        fft_fwd.process_with_scratch(&mut spec, &mut scratch);
        spec
    };
    let r_spec = spectrum(ref_w);
    let t_spec = spectrum(tgt_w);

    // Standard Cross-Correlation + mild PHAT weighting to handle both transients and bass phase
    let mut cross: Vec<Complex<f64>> = t_spec
        .iter()
        .zip(&r_spec)
        .map(|(t, r)| {
            let prod = t * r.conj();
            prod / prod.norm().max(1e-12).powf(0.5)
        })
        .collect();
    fft_inv.process_with_scratch(&mut cross, &mut scratch);

    let lag_val = |lag: isize| -> f64 {
        let idx = if lag >= 0 {
            lag as usize
        } else {
            (fft_len as isize + lag) as usize
        };
        cross[idx].re / (fft_len as f64)
    };

    let mut best = (0_isize, 0.0_f64);
    for lag in -(max_lag as isize)..=(max_lag as isize) {
        let val = lag_val(lag);
        if val.abs() > best.1.abs() {
            best = (lag, val);
        }
    }

    // Parabolic sub-sample refinement around the integer peak when not at the search boundary.
    let mut frac_lag = best.0 as f64;
    if best.0.unsigned_abs() < max_lag && best.1.abs() > 1e-12 {
        let y_m1 = lag_val(best.0 - 1).abs();
        let y_0 = best.1.abs();
        let y_p1 = lag_val(best.0 + 1).abs();
        let denom = y_m1 - 2.0 * y_0 + y_p1;
        if denom < -1e-12 {
            let delta = (0.5 * (y_m1 - y_p1) / denom).clamp(-0.5, 0.5);
            frac_lag += delta;
        }
    }

    (best.0, frac_lag, best.1)
}

/// `|Σ ref[i]·tgt[i + lag]| / √(E_ref·E_tgt)` over the analysis window. Cauchy-Schwarz keeps it
/// in `0..=1`; silent input scores 0.
fn normalized_correlation(ref_w: &[f64], tgt_w: &[f64], lag: isize) -> f64 {
    let energy = |w: &[f64]| w.iter().map(|x| x * x).sum::<f64>();
    let denom = (energy(ref_w) * energy(tgt_w)).sqrt();
    if denom <= f64::MIN_POSITIVE {
        return 0.0;
    }
    let shift = lag.unsigned_abs().min(ref_w.len());
    let (r, t) = if lag >= 0 {
        (ref_w, &tgt_w[shift..])
    } else {
        (&ref_w[shift..], tgt_w)
    };
    let dot: f64 = r.iter().zip(t).map(|(a, b)| a * b).sum();
    (dot.abs() / denom).clamp(0.0, 1.0)
}

/// 4-point Catmull-Rom cubic interpolation at `pos` inside `ch` (zero outside bounds).
#[inline]
fn catmull_rom_sample(ch: &[f64], pos: f64) -> f64 {
    let frames = ch.len() as isize;
    let i0 = pos.floor() as isize;
    let t = pos - (i0 as f64);
    let get = |idx: isize| -> f64 {
        if idx >= 0 && idx < frames {
            ch[idx as usize]
        } else {
            0.0
        }
    };
    let p0 = get(i0 - 1);
    let p1 = get(i0);
    let p2 = get(i0 + 1);
    let p3 = get(i0 + 2);
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

/// Shifts every channel of `target` earlier by `frac_lag` samples (using exact integer copy
/// when `frac_lag` is within `1e-4` of an integer, and 4-point Catmull-Rom cubic interpolation
/// for sub-sample offsets) and applies `sign`.
fn shift_target_fractional(target: &AudioBuffer, frac_lag: f64, sign: f64) -> AudioBuffer {
    let frames = target.num_frames();
    let nearest = frac_lag.round();
    let is_integer = (frac_lag - nearest).abs() <= 1e-4;
    let int_lag = nearest as isize;

    let channels = target
        .channels
        .iter()
        .map(|ch| {
            let mut out_ch = vec![0.0_f64; frames];
            if is_integer {
                for (i, sample_out) in out_ch.iter_mut().enumerate() {
                    let src_idx = (i as isize) + int_lag;
                    if src_idx >= 0 && (src_idx as usize) < frames {
                        *sample_out = ch[src_idx as usize] * sign;
                    }
                }
            } else {
                for (i, sample_out) in out_ch.iter_mut().enumerate() {
                    let src_pos = (i as f64) + frac_lag;
                    if src_pos >= 0.0 && src_pos <= (frames.saturating_sub(1) as f64) {
                        *sample_out = catmull_rom_sample(ch, src_pos) * sign;
                    }
                }
            }
            out_ch
        })
        .collect();
    AudioBuffer {
        channels,
        sample_rate: target.sample_rate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{mono, noise};

    /// `signal` delayed by `delay` samples with its polarity flipped.
    fn delayed_inverted(signal: &[f64], delay: usize) -> Vec<f64> {
        let mut out = vec![0.0; signal.len()];
        for (o, s) in out[delay..].iter_mut().zip(signal) {
            *o = -s;
        }
        out
    }

    #[test]
    fn gcc_phat_survives_two_second_silent_intro() {
        let sr = 48_000;
        let mut reference = vec![0.0; 2 * 48_000];
        reference.extend(noise(72_000, 7).iter().map(|x| 0.5 * x));
        let target = delayed_inverted(&reference, 37);
        let (_, res) =
            align_phase_gcc_phat(&mono(reference, sr), &mono(target, sr), 20.0).expect("aligned");
        assert_eq!(res.delay_samples, 37, "{res:?}");
        assert!(res.invert_polarity, "{res:?}");
    }

    #[test]
    fn gcc_phat_confidence_separates_matched_from_uncorrelated_audio() {
        let sr = 48_000;
        let a = noise(48_000, 11);
        let b = noise(48_000, 12_345);
        let (_, unrelated) =
            align_phase_gcc_phat(&mono(a.clone(), sr), &mono(b, sr), 20.0).expect("aligned");
        assert!(unrelated.correlation_peak < 0.2, "{unrelated:?}");
        let target = delayed_inverted(&a, 23);
        let (_, matched) =
            align_phase_gcc_phat(&mono(a, sr), &mono(target, sr), 20.0).expect("aligned");
        assert!(matched.correlation_peak > 0.9, "{matched:?}");
    }

    #[test]
    fn gcc_phat_reports_no_confidence_for_inputs_too_short_to_analyze() {
        let a = noise(32, 7);
        let (_, res) =
            align_phase_gcc_phat(&mono(a.clone(), 48_000), &mono(a, 48_000), 20.0).expect("ok");
        assert_eq!(res.correlation_peak, 0.0, "{res:?}");
    }
}
