//! "OpenSoothe" Dynamic Spectral Resonance Suppressor.
//!
//! Targets the erratic 2.0–6.5 kHz metallic whistling/phase-fizz typical of neural audio codecs
//! (Suno, Udio, YuE, ACE-Step) as well as harsh acoustic pick/cymbal resonances and 5.5–14 kHz
//! vocal sibilance sprays. Supports `delta_listen` to solo the removed harshness.

use std::f64::consts::PI;

use forge_core::{AudioBuffer, Result};
use rustfft::{FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};

use crate::loudness_limiter::{sanitize_param_f64, sanitize_sample_f64};

/// STFT frame length in samples.
const FFT_SIZE: usize = 2048;
/// STFT hop between frames in samples (75% overlap for COLA reconstruction with Hann^2 window).
const HOP: usize = 512;
/// Number of non-redundant positive-frequency bins.
const HALF_BINS: usize = FFT_SIZE / 2 + 1;
/// Attack time constant in seconds for spectral gain smoothing.
const ATTACK_TAU_SECS: f64 = 0.010;
/// Release time constant in seconds for spectral gain smoothing.
const RELEASE_TAU_SECS: f64 = 0.055;
/// Raised-cosine band transition width in octaves.
const BAND_SKIRT_OCTAVES: f64 = 1.0 / 3.0;

/// Configuration for OpenSoothe Dynamic Spectral Resonance Suppression.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DeFizzConfig {
    /// Threshold in dB above the local spectral neighborhood before attenuation begins.
    pub sensitivity_db: f64,
    /// Maximum attenuation depth in dB (`0.0..=18.0`).
    pub max_depth_db: f64,
    /// Sharpness (relative perceptual neighborhood width, `2..=24`, default `6`).
    pub sharpness_bins: usize,
    /// Focus weight on 2.0–6.5 kHz metallic/plastic zone (`0.0..=2.0`).
    pub fizz_band_weight: f64,
    /// Focus weight on 5.5–14.0 kHz sibilance zone (`0.0..=2.0`).
    pub sibilance_band_weight: f64,
    /// When `true`, outputs *only* the suppressed harsh resonances (Delta monitor mode).
    pub delta_listen: bool,
}

impl Default for DeFizzConfig {
    fn default() -> Self {
        Self {
            sensitivity_db: 3.5,
            max_depth_db: 9.0,
            sharpness_bins: 6,
            fizz_band_weight: 1.25,
            sibilance_band_weight: 1.0,
            delta_listen: false,
        }
    }
}

/// Smooth raised-cosine bandpass mask in log-frequency space for `[low_hz, high_hz]`.
fn raised_cosine_band_mask(freq_hz: f64, low_hz: f64, high_hz: f64, skirt_octaves: f64) -> f64 {
    if freq_hz <= 0.0 || low_hz >= high_hz {
        return 0.0;
    }
    let skirt = skirt_octaves.max(1e-4);
    let log_f = freq_hz.log2();
    let log_lo = low_hz.log2();
    let log_hi = high_hz.log2();

    if log_f <= log_lo - skirt || log_f >= log_hi + skirt {
        0.0
    } else if log_f < log_lo {
        let t = (log_f - (log_lo - skirt)) / skirt;
        0.5 * (1.0 - (PI * t).cos())
    } else if log_f <= log_hi {
        1.0
    } else {
        let t = (log_f - log_hi) / skirt;
        0.5 * (1.0 + (PI * t).cos())
    }
}

/// Glasberg & Moore Equivalent Rectangular Bandwidth (Hz) at `freq_hz`.
fn erb_hz(freq_hz: f64) -> f64 {
    24.7 * (4.37 * (freq_hz / 1000.0) + 1.0)
}

/// Precomputes the `[b_start, b_end]` ERB-scaled neighborhood bounds for each positive bin.
fn compute_erb_bounds(bin_hz: f64, sharpness_bins: usize) -> Vec<(usize, usize)> {
    let scale = (sharpness_bins as f64) / 6.0;
    (0..HALF_BINS)
        .map(|b| {
            let f = (b as f64) * bin_hz;
            let half_w = ((erb_hz(f) / bin_hz) * scale).round() as usize;
            let half_w = half_w.clamp(2, 64);
            let b_start = b.saturating_sub(half_w).max(1);
            let b_end = (b + half_w).min(HALF_BINS - 2);
            (b_start, b_end)
        })
        .collect()
}

/// Applies stereo-linked Overlap-Add (OLA) STFT dynamic resonance suppression across all channels.
///
/// Uses ERB-scaled log-magnitude (geometric mean) local baseline estimation, smooth raised-cosine
/// frequency-band transitions, and sample-rate-independent attack/release temporal smoothing.
pub fn apply_spectral_defizz(buffer: &AudioBuffer, config: &DeFizzConfig) -> Result<AudioBuffer> {
    let sr = f64::from(buffer.sample_rate.max(1));
    let frames = buffer.num_frames();
    let num_channels = buffer.channels.len();
    if frames == 0 || num_channels == 0 {
        return Ok(buffer.clone());
    }

    let bin_hz = sr / (FFT_SIZE as f64);

    // Precompute periodic Hann window and smooth frequency-band sensitivity weights
    let window: Vec<f64> = (0..FFT_SIZE)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * (i as f64) / (FFT_SIZE as f64)).cos()))
        .collect();

    let clean_sens_db = sanitize_param_f64(config.sensitivity_db, 3.5, 0.5, 24.0);
    let clean_depth_db = sanitize_param_f64(config.max_depth_db, 9.0, 0.5, 24.0);
    let clean_fizz_w = sanitize_param_f64(config.fizz_band_weight, 1.25, 0.0, 2.0);
    let clean_sib_w = sanitize_param_f64(config.sibilance_band_weight, 1.0, 0.0, 2.0);
    let sharpness = config.sharpness_bins.clamp(2, 24);

    let band_weights: Vec<f64> = (0..HALF_BINS)
        .map(|b| {
            let f = (b as f64) * bin_hz;
            let w_mud = 0.35 * raised_cosine_band_mask(f, 250.0, 500.0, BAND_SKIRT_OCTAVES);
            let w_fizz =
                clean_fizz_w * raised_cosine_band_mask(f, 1800.0, 6800.0, BAND_SKIRT_OCTAVES);
            let w_sib =
                clean_sib_w * raised_cosine_band_mask(f, 5500.0, 14500.0, BAND_SKIRT_OCTAVES);
            w_mud.max(w_fizz).max(w_sib)
        })
        .collect();

    let erb_bounds = compute_erb_bounds(bin_hz, sharpness);

    let thresh_ratio = 10.0_f64.powf(clean_sens_db / 20.0);
    let min_gain = 10.0_f64.powf(-clean_depth_db / 20.0);

    // Sample-rate-independent temporal smoothing coefficients
    let hop_secs = (HOP as f64) / sr;
    let attack_alpha = (-hop_secs / ATTACK_TAU_SECS).exp();
    let release_alpha = (-hop_secs / RELEASE_TAU_SECS).exp();

    let padded_len = frames + FFT_SIZE * 2;
    let padded_channels: Vec<Vec<f64>> = buffer
        .channels
        .iter()
        .map(|ch| {
            let mut padded = vec![0.0_f64; padded_len];
            for (dst, &src) in padded[FFT_SIZE..FFT_SIZE + frames]
                .iter_mut()
                .zip(ch.iter())
            {
                *dst = sanitize_sample_f64(src);
            }
            padded
        })
        .collect();

    let mut out_accums: Vec<Vec<f64>> = vec![vec![0.0_f64; padded_len]; num_channels];
    let mut norm_accum = vec![0.0_f64; padded_len];

    let mut planner = FftPlanner::<f64>::new();
    let fft_fwd = planner.plan_fft_forward(FFT_SIZE);
    let fft_inv = planner.plan_fft_inverse(FFT_SIZE);
    let scratch_len = fft_fwd
        .get_inplace_scratch_len()
        .max(fft_inv.get_inplace_scratch_len());
    let mut scratch = vec![Complex::new(0.0, 0.0); scratch_len];

    let mut specs: Vec<Vec<Complex<f64>>> =
        vec![vec![Complex::new(0.0, 0.0); FFT_SIZE]; num_channels];
    let mut log_mags = vec![0.0_f64; HALF_BINS];
    let mut prefix_log = vec![0.0_f64; HALF_BINS + 1];
    let mut smoothed_gains = vec![1.0_f64; HALF_BINS];
    let mut effective_gains = vec![1.0_f64; HALF_BINS];

    let inv_n = 1.0 / (FFT_SIZE as f64);
    let inv_channels = 1.0 / (num_channels as f64);

    let mut pos = 0_usize;
    while pos + FFT_SIZE <= padded_len {
        // Forward STFT for each channel
        for (ch_idx, padded) in padded_channels.iter().enumerate() {
            let spec = &mut specs[ch_idx];
            for i in 0..FFT_SIZE {
                spec[i] = Complex::new(padded[pos + i] * window[i], 0.0);
            }
            fft_fwd.process_with_scratch(spec, &mut scratch);
        }

        // Stereo-linked RMS magnitude per bin in log domain + prefix sum for O(1) ERB baseline
        prefix_log[0] = 0.0;
        for b in 0..HALF_BINS {
            let mut power_sum = 0.0_f64;
            for spec in &specs {
                power_sum += spec[b].norm_sqr();
            }
            let linked_mag = (power_sum * inv_channels).sqrt().max(1e-12);
            let lm = linked_mag.ln();
            log_mags[b] = lm;
            prefix_log[b + 1] = prefix_log[b] + lm;
        }

        // Compute per-bin dynamic resonance gain against ERB geometric-mean neighborhood
        for b in 1..(HALF_BINS - 1) {
            let weight = band_weights[b];
            if weight <= 1e-4 {
                smoothed_gains[b] = 1.0;
                continue;
            }
            let (b_start, b_end) = erb_bounds[b];
            let total_sum = prefix_log[b_end + 1] - prefix_log[b_start];
            let total_count = b_end + 1 - b_start;

            // Exclude the 3-bin Hann window main lobe (b-1, b, b+1)
            let excl_lo = b.saturating_sub(1).max(b_start);
            let excl_hi = (b + 1).min(b_end);
            let excl_sum = prefix_log[excl_hi + 1] - prefix_log[excl_lo];
            let excl_count = excl_hi + 1 - excl_lo;

            let count = total_count.saturating_sub(excl_count);
            let local_log_env = if count > 0 {
                (total_sum - excl_sum) / (count as f64)
            } else {
                log_mags[b]
            };

            let ratio = (log_mags[b] - local_log_env).exp();
            let excess = ratio / thresh_ratio;
            let target_gain = if excess > 1.0 {
                (1.0 / excess.powf(0.65 * weight)).clamp(min_gain, 1.0)
            } else {
                1.0
            };

            let alpha = if target_gain < smoothed_gains[b] {
                attack_alpha
            } else {
                release_alpha
            };
            smoothed_gains[b] = alpha * smoothed_gains[b] + (1.0 - alpha) * target_gain;
        }

        // Smooth gains across the 3-bin Hann main lobe to prevent musical noise
        for b in 1..(HALF_BINS - 1) {
            let min_lobe = smoothed_gains[b - 1]
                .min(smoothed_gains[b])
                .min(smoothed_gains[b + 1]);
            let avg_lobe = 0.25 * smoothed_gains[b - 1]
                + 0.50 * smoothed_gains[b]
                + 0.25 * smoothed_gains[b + 1];
            let g = 0.65 * min_lobe + 0.35 * avg_lobe;
            effective_gains[b] = if config.delta_listen { 1.0 - g } else { g };
        }

        // Apply identical linked gains to every channel and overlap-add resynthesize
        for (ch_idx, spec) in specs.iter_mut().enumerate() {
            for (b, &eg) in effective_gains
                .iter()
                .enumerate()
                .take(HALF_BINS - 1)
                .skip(1)
            {
                spec[b] *= eg;
                spec[FFT_SIZE - b] = spec[b].conj();
            }
            if config.delta_listen {
                spec[0] = Complex::new(0.0, 0.0);
                spec[HALF_BINS - 1] = Complex::new(0.0, 0.0);
            }

            fft_inv.process_with_scratch(spec, &mut scratch);
            let out_accum = &mut out_accums[ch_idx];
            for i in 0..FFT_SIZE {
                out_accum[pos + i] += spec[i].re * inv_n * window[i];
            }
        }
        for i in 0..FFT_SIZE {
            let w = window[i];
            norm_accum[pos + i] += w * w;
        }

        pos += HOP;
    }

    let processed_channels: Vec<Vec<f64>> = out_accums
        .into_iter()
        .map(|out_accum| {
            let mut result = vec![0.0_f64; frames];
            for (i, res_val) in result.iter_mut().enumerate() {
                let idx = FFT_SIZE + i;
                let denom = norm_accum[idx].max(1e-9);
                *res_val = out_accum[idx] / denom;
            }
            result
        })
        .collect();

    Ok(AudioBuffer {
        channels: processed_channels,
        sample_rate: buffer.sample_rate,
    })
}
