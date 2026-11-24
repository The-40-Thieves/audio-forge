//! "OpenSpacer" 32-Band / STFT Dynamic Sidechain Frequency Unmasker.
//!
//! Carves a real-time mirror pocket in the target buffer (e.g., Instrument Bus or Sub-Bass)
//! strictly at the frequencies and timestamps where the sidechain buffer (e.g., Lead Vocal or Kick)
//! is active—replacing Wavesfactory Trackspacer / iZotope Neutron Unmasking.

use std::f64::consts::PI;
use std::sync::Arc;

use forge_core::{AudioBuffer, ForgeError, Result};
use rayon::prelude::*;
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};

use crate::loudness_limiter::{sanitize_param_f64, sanitize_sample_f64};

use crate::spectral_defizz::{BAND_SKIRT_OCTAVES, FFT_SIZE, HALF_BINS, HOP};

/// Attack time constant in seconds for sidechain unmasking.
const ATTACK_TAU_SECS: f64 = 0.009;
/// Release time constant in seconds for sidechain unmasking.
const RELEASE_TAU_SECS: f64 = 0.065;

/// Configuration for OpenSpacer Sidechain Spectral Unmasking.
///
/// Every field is sanitized before use: a non-finite value falls back to the
/// [`UnmaskConfig::vocal_over_instruments`] value, and a finite one is clamped to its range.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct UnmaskConfig {
    /// Lowest frequency in Hz where sidechain unmasking is active (e.g., `180.0` for Vocal, `30.0` for Kick).
    pub low_cut_hz: f64,
    /// Highest frequency in Hz where sidechain unmasking is active (e.g., `8000.0` for Vocal, `140.0` for Kick).
    pub high_cut_hz: f64,
    /// Maximum spectral gain reduction in dB, clamped to `0.5..=20.0`.
    pub max_reduction_db: f64,
    /// Sensitivity, clamped to `0.1..=2.0` (default `1.0`).
    pub sensitivity: f64,
    /// When `true`, outputs *only* the carved frequencies removed by the sidechain (Delta audition).
    #[serde(default)]
    pub delta_listen: bool,
    /// Mid/Side carving focus for stereo targets (`-1.0` = carve Mid only, preserving stereo sides;
    /// `0.0` = balanced L/R; `+1.0` = carve Side only). Ignored for non-stereo channel counts.
    #[serde(default)]
    pub mid_side_balance: f64,
}

impl Default for UnmaskConfig {
    fn default() -> Self {
        Self::vocal_over_instruments()
    }
}

impl UnmaskConfig {
    /// Preset for Lead Vocal → Instrument/Synth/Guitar Bus unmasking (`200 Hz – 7.5 kHz`).
    pub const fn vocal_over_instruments() -> Self {
        Self {
            low_cut_hz: 200.0,
            high_cut_hz: 7500.0,
            max_reduction_db: 4.5,
            sensitivity: 1.0,
            delta_listen: false,
            mid_side_balance: 0.0,
        }
    }

    /// Preset for Kick Drum → Sub-Bass / 808 Bus unmasking (`30 Hz – 130 Hz`).
    pub const fn kick_over_bass() -> Self {
        Self {
            low_cut_hz: 30.0,
            high_cut_hz: 130.0,
            max_reduction_db: 7.5,
            sensitivity: 1.25,
            delta_listen: false,
            mid_side_balance: 0.0,
        }
    }
}

/// Sanitized analysis parameters resolved from an [`UnmaskConfig`] at one sample rate.
#[derive(Debug, Clone)]
struct UnmaskBands {
    b_start: usize,
    b_end: usize,
    band_weights: Vec<f64>,
    min_gain: f64,
    sensitivity: f64,
    attack_alpha: f64,
    release_alpha: f64,
}

/// Smooth raised-cosine bandpass weight for a bin at `freq_hz` over `[low_hz, high_hz]`.
fn smooth_unmask_weight(freq_hz: f64, low_hz: f64, high_hz: f64, nyquist: f64) -> f64 {
    if freq_hz <= 0.0 || low_hz >= high_hz {
        return 0.0;
    }
    let log_f = freq_hz.log2();
    let lo_mask = if low_hz <= 1e-6 {
        1.0
    } else {
        let log_lo = low_hz.log2();
        if log_f <= log_lo - BAND_SKIRT_OCTAVES {
            0.0
        } else if log_f < log_lo {
            let t = (log_f - (log_lo - BAND_SKIRT_OCTAVES)) / BAND_SKIRT_OCTAVES;
            0.5 * (1.0 - (PI * t).cos())
        } else {
            1.0
        }
    };

    let hi_mask = if high_hz >= nyquist * (1.0 - 1e-6) {
        1.0
    } else {
        let log_hi = high_hz.log2();
        if log_f >= log_hi + BAND_SKIRT_OCTAVES {
            0.0
        } else if log_f > log_hi {
            let t = (log_f - log_hi) / BAND_SKIRT_OCTAVES;
            0.5 * (1.0 + (PI * t).cos())
        } else {
            1.0
        }
    };

    lo_mask * hi_mask
}

impl UnmaskBands {
    /// Routes every config value through [`sanitize_param_f64`] so `NaN` / `±Inf` cannot reach the
    /// gain curve, then builds smooth raised-cosine frequency weights and sample-rate-independent
    /// attack/release coefficients.
    fn resolve(config: &UnmaskConfig, sample_rate: f64) -> Self {
        let defaults = UnmaskConfig::vocal_over_instruments();
        let nyquist = sample_rate / 2.0;
        let bin_hz = sample_rate / (FFT_SIZE as f64);
        let low_hz = sanitize_param_f64(config.low_cut_hz, defaults.low_cut_hz, 0.0, nyquist)
            .min((nyquist - bin_hz).max(0.0));
        let high_hz = sanitize_param_f64(config.high_cut_hz, defaults.high_cut_hz, 0.0, nyquist)
            .max((low_hz + bin_hz).min(nyquist));
        let max_reduction_db = sanitize_param_f64(
            config.max_reduction_db,
            defaults.max_reduction_db,
            0.5,
            20.0,
        );

        let mut band_weights = vec![0.0_f64; HALF_BINS];
        let mut b_start = HALF_BINS - 2;
        let mut b_end = 1_usize;
        for (b, weight_slot) in band_weights
            .iter_mut()
            .enumerate()
            .take(HALF_BINS - 1)
            .skip(1)
        {
            let w = smooth_unmask_weight((b as f64) * bin_hz, low_hz, high_hz, nyquist);
            *weight_slot = w;
            if w > 1e-4 {
                b_start = b_start.min(b);
                b_end = b_end.max(b);
            }
        }
        if b_start > b_end {
            b_start = 1;
            b_end = 0;
        }

        let hop_secs = (HOP as f64) / sample_rate;
        let attack_alpha = (-hop_secs / ATTACK_TAU_SECS).exp();
        let release_alpha = (-hop_secs / RELEASE_TAU_SECS).exp();

        Self {
            b_start,
            b_end,
            band_weights,
            min_gain: 10.0_f64.powf(-max_reduction_db / 20.0),
            sensitivity: sanitize_param_f64(config.sensitivity, defaults.sensitivity, 0.1, 2.0),
            attack_alpha,
            release_alpha,
        }
    }

    /// Instantaneous ducking gain for one sidechain bin at `db` dBFS (active above `-48 dBFS`),
    /// weighted by the smooth band skirt `weight`.
    fn raw_gain(&self, db: f64, weight: f64) -> f64 {
        if db > -48.0 && weight > 1e-4 {
            let norm = ((db + 48.0) / 36.0 * self.sensitivity).clamp(0.0, 1.0);
            1.0 - weight * norm * (1.0 - self.min_gain)
        } else {
            1.0
        }
    }

    /// Sample-rate-independent asymmetric gain envelope: fast attack toward deeper ducking,
    /// slow release back to unity.
    fn smooth_gain(&self, env: f64, raw: f64) -> f64 {
        let alpha = if raw < env {
            self.attack_alpha
        } else {
            self.release_alpha
        };
        alpha * env + (1.0 - alpha) * raw
    }
}

/// Forward and inverse plans for the `FFT_SIZE` transform.
struct StftPlans {
    fwd: Arc<dyn Fft<f64>>,
    inv: Arc<dyn Fft<f64>>,
}

/// Periodic Hann window of length `n`.
fn hann_window(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * (i as f64) / (n as f64)).cos()))
        .collect()
}

/// Copies up to `frames` samples of `src` into a zero buffer padded by `FFT_SIZE` on both sides,
/// sanitizing each one so a single `NaN` / `±Inf` cannot spread across every frame it touches.
fn padded_sanitized(src: &[f64], frames: usize) -> Vec<f64> {
    let mut padded = vec![0.0_f64; frames + FFT_SIZE * 2];
    for (dst, &x) in padded[FFT_SIZE..FFT_SIZE + frames].iter_mut().zip(src) {
        *dst = sanitize_sample_f64(x);
    }
    padded
}

/// Fills `spec` with the windowed frame starting at `src[0]`.
fn load_windowed(spec: &mut [Complex<f64>], src: &[f64], window: &[f64]) {
    for ((s, &x), &w) in spec.iter_mut().zip(src).zip(window) {
        *s = Complex::new(x * w, 0.0);
    }
}

/// Per-frame, per-bin smoothed ducking gains measured from the padded sidechain.
fn sidechain_gain_frames(
    sc_padded: &[f64],
    window: &[f64],
    fft_fwd: &dyn Fft<f64>,
    bands: &UnmaskBands,
) -> Vec<Vec<f64>> {
    let num_stft_frames = sc_padded.len().saturating_sub(FFT_SIZE) / HOP + 1;
    let mut gain_frames = Vec::with_capacity(num_stft_frames);
    let mut spec = vec![Complex::new(0.0, 0.0); FFT_SIZE];
    let mut scratch = vec![Complex::new(0.0, 0.0); fft_fwd.get_inplace_scratch_len()];
    let mut env_gains = vec![1.0_f64; HALF_BINS];

    for f_idx in 0..num_stft_frames {
        load_windowed(&mut spec, &sc_padded[f_idx * HOP..], window);
        fft_fwd.process_with_scratch(&mut spec, &mut scratch);

        let mut frame_gains = vec![1.0_f64; HALF_BINS];
        if bands.b_start <= bands.b_end {
            for b in bands.b_start..=bands.b_end {
                let mag = spec[b].norm() / (FFT_SIZE as f64);
                let raw_g = bands.raw_gain(20.0 * mag.max(1e-9).log10(), bands.band_weights[b]);
                env_gains[b] = bands.smooth_gain(env_gains[b], raw_g);
                frame_gains[b] = env_gains[b];
            }
        }
        gain_frames.push(frame_gains);
    }
    gain_frames
}

/// Applies the 3-tap smoothed gain curve to the positive bins and mirrors the negative ones.
///
/// `depth_scale` scales the carve depth (`0.0` = bypass, `1.0` = full carve); when `delta_listen`
/// is `true`, outputs only the carved energy (`1.0 - g`) and zeroes DC/Nyquist.
fn apply_smoothed_gains(
    spec: &mut [Complex<f64>],
    gains: &[f64],
    depth_scale: f64,
    delta_listen: bool,
) {
    for b in 1..(HALF_BINS - 1) {
        let g_raw = 0.25 * gains[b - 1] + 0.50 * gains[b] + 0.25 * gains[b + 1];
        let g = (1.0 - depth_scale * (1.0 - g_raw)).clamp(0.0, 1.0);
        let eg = if delta_listen { 1.0 - g } else { g };
        spec[b] *= eg;
        spec[FFT_SIZE - b] = spec[b].conj();
    }
    if delta_listen {
        spec[0] = Complex::new(0.0, 0.0);
        spec[HALF_BINS - 1] = Complex::new(0.0, 0.0);
    }
}

/// Runs weighted overlap-add resynthesis of one padded target channel under `gain_frames`.
fn unmask_channel(
    padded: &[f64],
    gain_frames: &[Vec<f64>],
    window: &[f64],
    plans: &StftPlans,
    frames: usize,
    depth_scale: f64,
    delta_listen: bool,
) -> Vec<f64> {
    let mut out_accum = vec![0.0_f64; padded.len()];
    let mut norm_accum = vec![0.0_f64; padded.len()];
    let mut spec = vec![Complex::new(0.0, 0.0); FFT_SIZE];
    let scratch_len = plans
        .fwd
        .get_inplace_scratch_len()
        .max(plans.inv.get_inplace_scratch_len());
    let mut scratch = vec![Complex::new(0.0, 0.0); scratch_len];
    let inv_n = 1.0 / (FFT_SIZE as f64);

    for (f_idx, frame_gains) in gain_frames.iter().enumerate() {
        let pos = f_idx * HOP;
        load_windowed(&mut spec, &padded[pos..], window);
        plans.fwd.process_with_scratch(&mut spec, &mut scratch);
        apply_smoothed_gains(&mut spec, frame_gains, depth_scale, delta_listen);
        plans.inv.process_with_scratch(&mut spec, &mut scratch);

        let out = &mut out_accum[pos..pos + FFT_SIZE];
        let norm = &mut norm_accum[pos..pos + FFT_SIZE];
        for (((o, n), s), &w) in out.iter_mut().zip(norm.iter_mut()).zip(&spec).zip(window) {
            *o += s.re * inv_n * w;
            *n += w * w;
        }
    }

    (FFT_SIZE..FFT_SIZE + frames)
        .map(|idx| out_accum[idx] / norm_accum[idx].max(1e-9))
        .collect()
}

/// Rejects a ragged `target` whose channels do not all hold `frames` samples.
fn check_equal_channel_lengths(target: &AudioBuffer, frames: usize) -> Result<()> {
    match target.channels.iter().position(|ch| ch.len() != frames) {
        Some(idx) => Err(ForgeError::InvalidParameter(format!(
            "Ragged target buffer: channel 0 has {frames} frames, but channel {idx} has {}",
            target.channels[idx].len()
        ))),
        None => Ok(()),
    }
}

/// Carves a dynamic spectral pocket in `target` driven by `sidechain`.
///
/// Non-finite samples in either buffer are treated as silence, and every [`UnmaskConfig`] field is
/// sanitized, so the output is always finite. Returns [`ForgeError::InvalidParameter`] when the
/// sample rates differ or when `target` has channels of different lengths.
pub fn apply_spectral_unmask(
    target: &AudioBuffer,
    sidechain: &AudioBuffer,
    config: &UnmaskConfig,
) -> Result<AudioBuffer> {
    if target.sample_rate != sidechain.sample_rate {
        return Err(ForgeError::InvalidParameter(format!(
            "Target sample rate ({}) must match sidechain sample rate ({})",
            target.sample_rate, sidechain.sample_rate
        )));
    }
    let frames = target.num_frames();
    check_equal_channel_lengths(target, frames)?;
    if frames == 0 {
        return Ok(target.clone());
    }

    let bands = UnmaskBands::resolve(config, f64::from(target.sample_rate.max(1)));
    let window = hann_window(FFT_SIZE);
    let mut planner = FftPlanner::<f64>::new();
    let plans = StftPlans {
        fwd: planner.plan_fft_forward(FFT_SIZE),
        inv: planner.plan_fft_inverse(FFT_SIZE),
    };

    let sc_padded = padded_sanitized(&sidechain.to_mono(), frames);
    let gain_frames = sidechain_gain_frames(&sc_padded, &window, plans.fwd.as_ref(), &bands);
    let clean_ms = sanitize_param_f64(config.mid_side_balance, 0.0, -1.0, 1.0);
    let use_ms = target.channels.len() == 2 && clean_ms.abs() > 1e-4;

    let channels = if use_ms {
        let mid_scale = (1.0 - clean_ms.max(0.0)).clamp(0.0, 1.0);
        let side_scale = (1.0 + clean_ms.min(0.0)).clamp(0.0, 1.0);
        let mut mid = vec![0.0_f64; frames];
        let mut side = vec![0.0_f64; frames];
        for i in 0..frames {
            let l = sanitize_sample_f64(target.channels[0][i]);
            let r = sanitize_sample_f64(target.channels[1][i]);
            mid[i] = 0.5 * (l + r);
            side[i] = 0.5 * (l - r);
        }
        let (mid_out, side_out) = rayon::join(
            || {
                unmask_channel(
                    &padded_sanitized(&mid, frames),
                    &gain_frames,
                    &window,
                    &plans,
                    frames,
                    mid_scale,
                    config.delta_listen,
                )
            },
            || {
                unmask_channel(
                    &padded_sanitized(&side, frames),
                    &gain_frames,
                    &window,
                    &plans,
                    frames,
                    side_scale,
                    config.delta_listen,
                )
            },
        );
        let mut left = vec![0.0_f64; frames];
        let mut right = vec![0.0_f64; frames];
        for i in 0..frames {
            left[i] = mid_out[i] + side_out[i];
            right[i] = mid_out[i] - side_out[i];
        }
        vec![left, right]
    } else {
        target
            .channels
            .par_iter()
            .map(|ch| {
                unmask_channel(
                    &padded_sanitized(ch, frames),
                    &gain_frames,
                    &window,
                    &plans,
                    frames,
                    1.0,
                    config.delta_listen,
                )
            })
            .collect()
    };

    Ok(AudioBuffer {
        channels,
        sample_rate: target.sample_rate,
    })
}
