//! "OpenSpacer" 32-Band / STFT Dynamic Sidechain Frequency Unmasker.
//!
//! Carves a real-time mirror pocket in the target buffer (e.g., Instrument Bus or Sub-Bass)
//! strictly at the frequencies and timestamps where the sidechain buffer (e.g., Lead Vocal or Kick)
//! is active—replacing Wavesfactory Trackspacer / iZotope Neutron Unmasking.

use std::f64::consts::PI;
use std::sync::Arc;

use forge_core::{AudioBuffer, ForgeError, Result};
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};

use crate::loudness_limiter::{sanitize_param_f64, sanitize_sample_f64};

/// STFT frame length in samples.
const FFT_SIZE: usize = 2048;
/// STFT hop between frames in samples.
const HOP: usize = 512;
/// Number of non-redundant bins of a real `FFT_SIZE` transform.
const HALF_BINS: usize = FFT_SIZE / 2 + 1;

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
}

impl UnmaskConfig {
    /// Preset for Lead Vocal → Instrument/Synth/Guitar Bus unmasking (`200 Hz – 7.5 kHz`).
    pub const fn vocal_over_instruments() -> Self {
        Self {
            low_cut_hz: 200.0,
            high_cut_hz: 7500.0,
            max_reduction_db: 4.5,
            sensitivity: 1.0,
        }
    }

    /// Preset for Kick Drum → Sub-Bass / 808 Bus unmasking (`30 Hz – 130 Hz`).
    pub const fn kick_over_bass() -> Self {
        Self {
            low_cut_hz: 30.0,
            high_cut_hz: 130.0,
            max_reduction_db: 7.5,
            sensitivity: 1.25,
        }
    }
}

/// Sanitized analysis parameters resolved from an [`UnmaskConfig`] at one sample rate.
#[derive(Debug, Clone, Copy)]
struct UnmaskBands {
    b_low: usize,
    b_high: usize,
    min_gain: f64,
    sensitivity: f64,
}

impl UnmaskBands {
    /// Routes every config value through [`sanitize_param_f64`] so `NaN` / `±Inf` cannot reach the
    /// gain curve, then maps the cutoffs onto bins.
    fn resolve(config: &UnmaskConfig, sample_rate: f64) -> Self {
        let defaults = UnmaskConfig::vocal_over_instruments();
        let nyquist = sample_rate / 2.0;
        let bin_hz = sample_rate / (FFT_SIZE as f64);
        let low_hz = sanitize_param_f64(config.low_cut_hz, defaults.low_cut_hz, 0.0, nyquist);
        let high_hz = sanitize_param_f64(config.high_cut_hz, defaults.high_cut_hz, 0.0, nyquist);
        let max_reduction_db = sanitize_param_f64(
            config.max_reduction_db,
            defaults.max_reduction_db,
            0.5,
            20.0,
        );
        let b_low = ((low_hz / bin_hz).floor() as usize).clamp(1, HALF_BINS - 3);
        let b_high = ((high_hz / bin_hz).ceil() as usize).clamp(b_low + 1, HALF_BINS - 2);
        Self {
            b_low,
            b_high,
            min_gain: 10.0_f64.powf(-max_reduction_db / 20.0),
            sensitivity: sanitize_param_f64(config.sensitivity, defaults.sensitivity, 0.1, 2.0),
        }
    }

    /// Instantaneous ducking gain for one sidechain bin at `db` dBFS (active above `-48 dBFS`).
    fn raw_gain(&self, db: f64) -> f64 {
        if db > -48.0 {
            let norm = ((db + 48.0) / 36.0 * self.sensitivity).clamp(0.0, 1.0);
            1.0 - norm * (1.0 - self.min_gain)
        } else {
            1.0
        }
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

/// Asymmetric gain envelope: fast attack toward deeper ducking, slow release back to unity.
fn smooth_gain(env: f64, raw: f64) -> f64 {
    if raw < env {
        0.30 * env + 0.70 * raw
    } else {
        0.85 * env + 0.15 * raw
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
    let mut env_gains = vec![1.0_f64; HALF_BINS];

    for f_idx in 0..num_stft_frames {
        load_windowed(&mut spec, &sc_padded[f_idx * HOP..], window);
        fft_fwd.process(&mut spec);

        let mut frame_gains = vec![1.0_f64; HALF_BINS];
        for b in bands.b_low..=bands.b_high {
            let mag = spec[b].norm() / (FFT_SIZE as f64);
            let raw_g = bands.raw_gain(20.0 * mag.max(1e-9).log10());
            env_gains[b] = smooth_gain(env_gains[b], raw_g);
            frame_gains[b] = env_gains[b];
        }
        gain_frames.push(frame_gains);
    }
    gain_frames
}

/// Applies the 3-tap smoothed gain curve to the positive bins and mirrors the negative ones.
fn apply_smoothed_gains(spec: &mut [Complex<f64>], gains: &[f64]) {
    for b in 1..(HALF_BINS - 1) {
        let g = 0.25 * gains[b - 1] + 0.50 * gains[b] + 0.25 * gains[b + 1];
        spec[b] *= g;
        spec[FFT_SIZE - b] = spec[b].conj();
    }
}

/// Runs weighted overlap-add resynthesis of one padded target channel under `gain_frames`.
fn unmask_channel(
    padded: &[f64],
    gain_frames: &[Vec<f64>],
    window: &[f64],
    plans: &StftPlans,
    frames: usize,
) -> Vec<f64> {
    let mut out_accum = vec![0.0_f64; padded.len()];
    let mut norm_accum = vec![0.0_f64; padded.len()];
    let mut spec = vec![Complex::new(0.0, 0.0); FFT_SIZE];
    let inv_n = 1.0 / (FFT_SIZE as f64);

    for (f_idx, frame_gains) in gain_frames.iter().enumerate() {
        let pos = f_idx * HOP;
        load_windowed(&mut spec, &padded[pos..], window);
        plans.fwd.process(&mut spec);
        apply_smoothed_gains(&mut spec, frame_gains);
        plans.inv.process(&mut spec);

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
    if frames == 0 {
        return Ok(target.clone());
    }
    check_equal_channel_lengths(target, frames)?;

    let bands = UnmaskBands::resolve(config, f64::from(target.sample_rate.max(1)));
    let window = hann_window(FFT_SIZE);
    let mut planner = FftPlanner::<f64>::new();
    let plans = StftPlans {
        fwd: planner.plan_fft_forward(FFT_SIZE),
        inv: planner.plan_fft_inverse(FFT_SIZE),
    };

    let sc_padded = padded_sanitized(&sidechain.to_mono(), frames);
    let gain_frames = sidechain_gain_frames(&sc_padded, &window, plans.fwd.as_ref(), &bands);
    let channels = target
        .channels
        .iter()
        .map(|ch| {
            unmask_channel(
                &padded_sanitized(ch, frames),
                &gain_frames,
                &window,
                &plans,
                frames,
            )
        })
        .collect();

    Ok(AudioBuffer {
        channels,
        sample_rate: target.sample_rate,
    })
}
