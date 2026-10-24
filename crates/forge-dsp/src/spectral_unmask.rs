//! "OpenSpacer" 32-Band / STFT Dynamic Sidechain Frequency Unmasker.
//!
//! Carves a real-time mirror pocket in the target buffer (e.g., Instrument Bus or Sub-Bass)
//! strictly at the frequencies and timestamps where the sidechain buffer (e.g., Lead Vocal or Kick)
//! is active—replacing Wavesfactory Trackspacer / iZotope Neutron Unmasking.

use std::f64::consts::PI;

use forge_core::{AudioBuffer, ForgeError, Result};
use rustfft::{num_complex::Complex, FftPlanner};
use serde::{Deserialize, Serialize};

/// Configuration for OpenSpacer Sidechain Spectral Unmasking.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct UnmaskConfig {
    /// Lowest frequency in Hz where sidechain unmasking is active (e.g., `180.0` for Vocal, `30.0` for Kick).
    pub low_cut_hz: f64,
    /// Highest frequency in Hz where sidechain unmasking is active (e.g., `8000.0` for Vocal, `140.0` for Kick).
    pub high_cut_hz: f64,
    /// Maximum spectral gain reduction in dB (`1.0..=15.0`).
    pub max_reduction_db: f64,
    /// Sensitivity (`0.1..=2.0`, default `1.0`).
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

/// Carves a dynamic spectral pocket in `target` driven by `sidechain`.
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

    let sr = f64::from(target.sample_rate.max(1));
    let sc_mono = sidechain.to_mono();
    let fft_size = 2048_usize;
    let hop = 512_usize;
    let half_bins = fft_size / 2 + 1;
    let bin_hz = sr / (fft_size as f64);

    let b_low = ((config.low_cut_hz / bin_hz).floor() as usize).clamp(1, half_bins - 2);
    let b_high = ((config.high_cut_hz / bin_hz).ceil() as usize).clamp(b_low + 1, half_bins - 2);
    let min_gain = 10.0_f64.powf(-config.max_reduction_db.clamp(0.5, 20.0) / 20.0);

    let window: Vec<f64> = (0..fft_size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * (i as f64) / (fft_size as f64)).cos()))
        .collect();

    // Pre-compute sidechain STFT magnitude frames
    let padded_len = frames + fft_size * 2;
    let mut sc_padded = vec![0.0_f64; padded_len];
    let copy_len = sc_mono.len().min(frames);
    sc_padded[fft_size..fft_size + copy_len].copy_from_slice(&sc_mono[..copy_len]);

    let mut planner = FftPlanner::<f64>::new();
    let fft_fwd = planner.plan_fft_forward(fft_size);
    let fft_inv = planner.plan_fft_inverse(fft_size);

    let num_stft_frames = (padded_len.saturating_sub(fft_size)) / hop + 1;
    let mut sc_gain_frames = Vec::with_capacity(num_stft_frames);
    let mut sc_spec = vec![Complex::new(0.0, 0.0); fft_size];
    let mut env_gains = vec![1.0_f64; half_bins];

    for f_idx in 0..num_stft_frames {
        let pos = f_idx * hop;
        for i in 0..fft_size {
            sc_spec[i] = Complex::new(sc_padded[pos + i] * window[i], 0.0);
        }
        fft_fwd.process(&mut sc_spec);

        let mut frame_gains = vec![1.0_f64; half_bins];
        for b in b_low..=b_high {
            let mag = sc_spec[b].norm() / (fft_size as f64);
            let db = 20.0 * mag.max(1e-9).log10();
            // Activate ducking when sidechain bin exceeds -48 dBFS
            let raw_g = if db > -48.0 {
                let norm = ((db + 48.0) / 36.0 * config.sensitivity).clamp(0.0, 1.0);
                1.0 - norm * (1.0 - min_gain)
            } else {
                1.0
            };
            if raw_g < env_gains[b] {
                env_gains[b] = 0.30 * env_gains[b] + 0.70 * raw_g;
            } else {
                env_gains[b] = 0.85 * env_gains[b] + 0.15 * raw_g;
            }
            frame_gains[b] = env_gains[b];
        }
        sc_gain_frames.push(frame_gains);
    }

    // Apply smoothed sidechain spectral curve to each channel of target
    let mut out_channels = Vec::with_capacity(target.num_channels());
    for ch in &target.channels {
        let mut t_padded = vec![0.0_f64; padded_len];
        t_padded[fft_size..fft_size + frames].copy_from_slice(ch);

        let mut out_accum = vec![0.0_f64; padded_len];
        let mut norm_accum = vec![0.0_f64; padded_len];
        let mut t_spec = vec![Complex::new(0.0, 0.0); fft_size];

        for (f_idx, frame_gains) in sc_gain_frames.iter().enumerate() {
            let pos = f_idx * hop;
            for i in 0..fft_size {
                t_spec[i] = Complex::new(t_padded[pos + i] * window[i], 0.0);
            }
            fft_fwd.process(&mut t_spec);

            for b in 1..(half_bins - 1) {
                let g = 0.25 * frame_gains[b - 1]
                    + 0.50 * frame_gains[b]
                    + 0.25 * frame_gains[b + 1];
                t_spec[b] *= g;
                t_spec[fft_size - b] = t_spec[b].conj();
            }

            fft_inv.process(&mut t_spec);
            let inv_n = 1.0 / (fft_size as f64);
            for i in 0..fft_size {
                let w = window[i];
                out_accum[pos + i] += t_spec[i].re * inv_n * w;
                norm_accum[pos + i] += w * w;
            }
        }

        let mut ch_out = vec![0.0_f64; frames];
        for (i, sample_out) in ch_out.iter_mut().enumerate() {
            let idx = fft_size + i;
            *sample_out = out_accum[idx] / norm_accum[idx].max(1e-9);
        }
        out_channels.push(ch_out);
    }

    Ok(AudioBuffer {
        channels: out_channels,
        sample_rate: target.sample_rate,
    })
}
