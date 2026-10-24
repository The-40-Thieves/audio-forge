//! "OpenSoothe" Dynamic Spectral Resonance Suppressor.
//!
//! Targets the erratic 2.0–6.5 kHz metallic whistling/phase-fizz typical of neural audio codecs
//! (Suno, Udio, YuE, ACE-Step) as well as harsh acoustic pick/cymbal resonances and 5.5–14 kHz
//! vocal sibilance sprays. Supports `delta_listen` to solo the removed harshness.

use std::f64::consts::PI;

use forge_core::{AudioBuffer, Result};
use rayon::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};
use serde::{Deserialize, Serialize};

/// Configuration for OpenSoothe Dynamic Spectral Resonance Suppression.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DeFizzConfig {
    /// Threshold in dB above the local spectral neighborhood before attenuation begins.
    pub sensitivity_db: f64,
    /// Maximum attenuation depth in dB (`0.0..=18.0`).
    pub max_depth_db: f64,
    /// Sharpness (neighborhood window half-width in FFT bins, `3..=16`).
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

/// Applies real-time-capable Overlap-Add (OLA) STFT dynamic resonance suppression across all channels.
pub fn apply_spectral_defizz(buffer: &AudioBuffer, config: &DeFizzConfig) -> Result<AudioBuffer> {
    let sr = f64::from(buffer.sample_rate.max(1));
    let frames = buffer.num_frames();
    if frames == 0 {
        return Ok(buffer.clone());
    }

    let fft_size = 2048_usize;
    let hop = 512_usize; // 75% overlap for artifact-free COLA reconstruction with Hann^2 window
    let bin_hz = sr / (fft_size as f64);
    let half_bins = fft_size / 2 + 1;

    // Precompute Hann window and frequency-band sensitivity weights
    let window: Vec<f64> = (0..fft_size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * (i as f64) / (fft_size as f64)).cos()))
        .collect();

    let band_weights: Vec<f64> = (0..half_bins)
        .map(|b| {
            let f = (b as f64) * bin_hz;
            let mut w = 0.0_f64;
            if (1800.0..=6800.0).contains(&f) {
                w = w.max(config.fizz_band_weight);
            }
            if (5500.0..=14500.0).contains(&f) {
                w = w.max(config.sibilance_band_weight);
            }
            if (250.0..=500.0).contains(&f) {
                // Gentle dynamic mud taming in 250-500 Hz
                w = w.max(0.35);
            }
            w
        })
        .collect();

    let thresh_ratio = 10.0_f64.powf(config.sensitivity_db.max(0.5) / 20.0);
    let min_gain = 10.0_f64.powf(-config.max_depth_db.clamp(0.5, 24.0) / 20.0);
    let radius = config.sharpness_bins.clamp(2, 24);

    let processed_channels: Vec<Vec<f64>> = buffer
        .channels
        .par_iter()
        .map(|ch| {
            let padded_len = frames + fft_size * 2;
            let mut padded = vec![0.0_f64; padded_len];
            padded[fft_size..fft_size + frames].copy_from_slice(ch);

            let mut out_accum = vec![0.0_f64; padded_len];
            let mut norm_accum = vec![0.0_f64; padded_len];

            let mut planner = FftPlanner::<f64>::new();
            let fft_fwd = planner.plan_fft_forward(fft_size);
            let fft_inv = planner.plan_fft_inverse(fft_size);

            let mut spec = vec![Complex::new(0.0, 0.0); fft_size];
            let mut mags = vec![0.0_f64; half_bins];
            let mut smoothed_gains = vec![1.0_f64; half_bins];

            let mut pos = 0_usize;
            while pos + fft_size <= padded_len {
                for i in 0..fft_size {
                    spec[i] = Complex::new(padded[pos + i] * window[i], 0.0);
                }
                fft_fwd.process(&mut spec);

                for b in 0..half_bins {
                    mags[b] = spec[b].norm().max(1e-12);
                }

                // Compute per-bin dynamic resonance gain
                for b in 1..(half_bins - 1) {
                    let weight = band_weights[b];
                    if weight <= 0.01 {
                        smoothed_gains[b] = 1.0;
                        continue;
                    }
                    let b_start = b.saturating_sub(radius.max(8)).max(1);
                    let b_end = (b + radius.max(8)).min(half_bins - 1);
                    let mut sum_neigh = 0.0_f64;
                    let mut count = 0_usize;
                    for (nb, &mag_val) in mags.iter().enumerate().take(b_end + 1).skip(b_start) {
                        // Exclude the 3-bin Hann window main lobe (b-1, b, b+1)
                        if nb.abs_diff(b) > 1 {
                            sum_neigh += mag_val;
                            count += 1;
                        }
                    }
                    let local_env = if count > 0 {
                        (sum_neigh / (count as f64)).max(1e-12)
                    } else {
                        mags[b]
                    };

                    let excess = (mags[b] / local_env) / thresh_ratio;
                    let target_gain = if excess > 1.0 {
                        (1.0 / excess.powf(0.65 * weight)).clamp(min_gain, 1.0)
                    } else {
                        1.0
                    };

                    // Temporal attack/release smoothing across STFT frames
                    if target_gain < smoothed_gains[b] {
                        smoothed_gains[b] = 0.35 * smoothed_gains[b] + 0.65 * target_gain;
                    } else {
                        smoothed_gains[b] = 0.82 * smoothed_gains[b] + 0.18 * target_gain;
                    }
                }

                // Apply gains across the 3-bin Hann main lobe and smooth to prevent musical noise
                for b in 1..(half_bins - 1) {
                    let min_lobe = smoothed_gains[b - 1]
                        .min(smoothed_gains[b])
                        .min(smoothed_gains[b + 1]);
                    let avg_lobe = 0.25 * smoothed_gains[b - 1]
                        + 0.50 * smoothed_gains[b]
                        + 0.25 * smoothed_gains[b + 1];
                    let g = 0.65 * min_lobe + 0.35 * avg_lobe;
                    let effective_g = if config.delta_listen { 1.0 - g } else { g };
                    spec[b] *= effective_g;
                    spec[fft_size - b] = spec[b].conj();
                }
                if config.delta_listen {
                    spec[0] = Complex::new(0.0, 0.0);
                    spec[half_bins - 1] = Complex::new(0.0, 0.0);
                }

                fft_inv.process(&mut spec);
                let inv_n = 1.0 / (fft_size as f64);
                for i in 0..fft_size {
                    let w = window[i];
                    out_accum[pos + i] += spec[i].re * inv_n * w;
                    norm_accum[pos + i] += w * w;
                }

                pos += hop;
            }

            let mut result = vec![0.0_f64; frames];
            for (i, res_val) in result.iter_mut().enumerate() {
                let idx = fft_size + i;
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
