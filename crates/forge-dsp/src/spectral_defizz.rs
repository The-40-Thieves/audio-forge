//! "OpenSoothe" Dynamic Spectral Resonance Suppressor.
//!
//! Targets the erratic 2.0–6.5 kHz metallic whistling/phase-fizz typical of neural audio codecs
//! (Suno, Udio, YuE, ACE-Step) as well as harsh acoustic pick/cymbal resonances and 5.5–14 kHz
//! vocal sibilance sprays. Supports `delta_listen` to solo the removed harshness.

use std::f64::consts::PI;

use forge_core::{AudioBuffer, AuditReport, Result};
use rustfft::{FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};

use crate::loudness_limiter::{Biquad, sanitize_param_f64, sanitize_sample_f64};

/// STFT frame length in samples.
pub(crate) const FFT_SIZE: usize = 2048;
/// STFT hop between frames in samples (75% overlap for COLA reconstruction with Hann^2 window).
pub(crate) const HOP: usize = 512;
/// Number of non-redundant positive-frequency bins.
pub(crate) const HALF_BINS: usize = FFT_SIZE / 2 + 1;
/// Default attack time constant in milliseconds for spectral gain smoothing.
const DEFAULT_ATTACK_MS: f64 = 10.0;
/// Default release time constant in milliseconds for spectral gain smoothing.
const DEFAULT_RELEASE_MS: f64 = 55.0;
/// Raised-cosine band transition width in octaves.
pub(crate) const BAND_SKIRT_OCTAVES: f64 = 1.0 / 3.0;

const fn default_stereo_link() -> f64 {
    1.0
}

const fn default_attack_ms() -> f64 {
    DEFAULT_ATTACK_MS
}

const fn default_release_ms() -> f64 {
    DEFAULT_RELEASE_MS
}

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
    /// Spectral sensitivity tilt in dB/octave pivoted at 1 kHz (`-6.0..=+6.0`, default `0.0`).
    /// Positive tilt focuses suppression toward higher frequencies; negative toward lows/mids.
    #[serde(default)]
    pub tilt_db_per_oct: f64,
    /// Stereo gain-linking amount (`0.0` = independent dual-mono, `1.0` = fully linked stereo).
    #[serde(default = "default_stereo_link")]
    pub stereo_link: f64,
    /// Mid/Side suppression focus for stereo buffers (`-1.0` = Mid only, `0.0` = balanced,
    /// `+1.0` = Side only). Ignored for non-stereo channel counts.
    #[serde(default)]
    pub mid_side_balance: f64,
    /// Temporal smoothing attack constant in milliseconds (`0.5..=200.0`, default `10.0`).
    #[serde(default = "default_attack_ms")]
    pub attack_ms: f64,
    /// Temporal smoothing release constant in milliseconds (`5.0..=500.0`, default `55.0`).
    #[serde(default = "default_release_ms")]
    pub release_ms: f64,
    /// Enables steeper resonance suppression slope (`0.95` vs `0.65`) for stubborn AI codecs.
    #[serde(default)]
    pub hard_mode: bool,
    /// 3-band focus weighting `[low_db (<400Hz), mid_db (400..4000Hz), high_db (>4000Hz)]`.
    #[serde(default)]
    pub focus_eq_db: [f64; 3],
    /// HPSS frame-erratic shimmer suppression boost (`0.0..=1.0`, default `0.0`).
    #[serde(default)]
    pub hpss_shimmer_suppress: f64,
    /// Optional gentle ultrasonic LPF cutoff Hz (e.g. `Some(16_000.0)` for AI vocal stems).
    #[serde(default)]
    pub ultrasonic_lpf_hz: Option<f64>,
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
            tilt_db_per_oct: 0.0,
            stereo_link: 1.0,
            mid_side_balance: 0.0,
            attack_ms: DEFAULT_ATTACK_MS,
            release_ms: DEFAULT_RELEASE_MS,
            hard_mode: false,
            focus_eq_db: [0.0; 3],
            hpss_shimmer_suppress: 0.0,
            ultrasonic_lpf_hz: None,
        }
    }
}

impl DeFizzConfig {
    /// Derives an AI-artifact-aware preset from an [`AuditReport`], shifting focus toward the
    /// Side channel when Stereo-Side shimmer dominates and engaging HPSS shimmer suppression.
    pub fn ai_mix_from_audit(audit: &AuditReport) -> Self {
        let severity = (audit.ai_fizz_index / 100.0).clamp(0.0, 1.0);
        let side_bias =
            if audit.channels >= 2 && audit.side_fizz_p95_db > audit.mid_fizz_p95_db + 2.0 {
                ((audit.side_fizz_p95_db - audit.mid_fizz_p95_db) * 0.08).clamp(0.15, 0.55)
            } else {
                0.0
            };
        Self {
            sensitivity_db: (3.8 - 1.1 * severity).clamp(2.2, 4.5),
            max_depth_db: (8.5 + 4.5 * severity).clamp(8.5, 13.5),
            fizz_band_weight: (1.2 + 0.35 * severity).clamp(1.1, 1.65),
            sibilance_band_weight: (1.0 + 0.25 * severity).clamp(1.0, 1.35),
            mid_side_balance: side_bias,
            hard_mode: audit.ai_fizz_index >= 55.0,
            hpss_shimmer_suppress: (0.25 + 0.45 * severity).clamp(0.25, 0.75),
            ..Self::default()
        }
    }

    /// Vocal-stem preset with HPSS breath-shimmer suppression and optional 16 kHz neural buzz LPF.
    pub fn ai_vocal() -> Self {
        Self {
            sensitivity_db: 3.2,
            max_depth_db: 10.5,
            fizz_band_weight: 1.25,
            sibilance_band_weight: 1.35,
            attack_ms: 6.0,
            release_ms: 45.0,
            hpss_shimmer_suppress: 0.40,
            ultrasonic_lpf_hz: Some(16_000.0),
            ..Self::default()
        }
    }
}

/// Smooth raised-cosine bandpass mask in log-frequency space for `[low_hz, high_hz]`.
#[inline]
pub(crate) fn raised_cosine_band_mask(
    freq_hz: f64,
    low_hz: f64,
    high_hz: f64,
    skirt_octaves: f64,
) -> f64 {
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

/// Smooth log-frequency focus EQ weight across Low (`<400 Hz`), Mid (`400..4000 Hz`), High (`>4000 Hz`).
#[inline]
pub(crate) fn focus_eq_multiplier(freq_hz: f64, focus_db: [f64; 3]) -> f64 {
    if focus_db[0].abs() < 1e-4 && focus_db[1].abs() < 1e-4 && focus_db[2].abs() < 1e-4 {
        return 1.0;
    }
    let f = freq_hz.max(20.0);
    let lf = f.log2();
    let l_lo = 400.0_f64.log2();
    let l_hi = 4000.0_f64.log2();
    let w_low = ((l_lo + 0.5 - lf) / 1.0).clamp(0.0, 1.0);
    let w_high = ((lf - (l_hi - 0.5)) / 1.0).clamp(0.0, 1.0);
    let w_mid = (1.0 - w_low - w_high).max(0.0);
    let db = w_low * focus_db[0] + w_mid * focus_db[1] + w_high * focus_db[2];
    10.0_f64.powf(db / 20.0).clamp(0.15, 4.0)
}

/// Spectral sensitivity tilt multiplier pivoted around 1 kHz (`dB/octave`).
#[inline]
pub(crate) fn tilt_multiplier(freq_hz: f64, tilt_db_per_oct: f64) -> f64 {
    if tilt_db_per_oct.abs() <= 1e-6 || freq_hz <= 20.0 {
        return 1.0;
    }
    let octaves_from_1k = (freq_hz / 1000.0).log2();
    10.0_f64
        .powf((tilt_db_per_oct * octaves_from_1k) / 20.0)
        .clamp(0.15, 3.0)
}

/// 5-element branch-free median on stack array for frame-wise HPSS harmonic baseline.
#[inline]
fn median5(mut v: [f64; 5]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[2]
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

/// Shared parameters for per-bin ERB dynamic resonance gain estimation.
struct ErbGainParams<'a> {
    band_weights: &'a [f64],
    erb_bounds: &'a [(usize, usize)],
    thresh_ratio: f64,
    min_gain: f64,
    attack_alpha: f64,
    release_alpha: f64,
    slope_exp: f64,
    hpss_shimmer_suppress: f64,
    b_hpss_lo: usize,
    b_hpss_hi: usize,
}

/// Updates `smoothed_gains` in place from `log_mags` + `prefix_log` using O(1) ERB baselines
/// and optional HPSS frame-erratic shimmer detection.
fn update_erb_gains(
    log_mags: &[f64],
    prefix_log: &[f64],
    mag_hist: &mut [[f64; 5]],
    hist_slot: usize,
    smoothed_gains: &mut [f64],
    focus: f64,
    params: &ErbGainParams<'_>,
) {
    for b in 1..(HALF_BINS - 1) {
        let mag = log_mags[b].exp();
        mag_hist[b][hist_slot] = mag;

        let mut weight = params.band_weights[b] * focus;
        if weight <= 1e-4 {
            smoothed_gains[b] = 1.0;
            continue;
        }
        let (b_start, b_end) = params.erb_bounds[b];
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
        if params.hpss_shimmer_suppress > 1e-4
            && b >= params.b_hpss_lo
            && b <= params.b_hpss_hi
            && ratio > params.thresh_ratio
        {
            let harmonic_est = median5(mag_hist[b]).max(1e-12);
            let burst_ratio = mag / harmonic_est;
            if burst_ratio > 1.35 {
                let boost =
                    1.0 + params.hpss_shimmer_suppress * ((burst_ratio - 1.35).clamp(0.0, 1.5));
                weight = (weight * boost).clamp(0.0, 3.5);
            }
        }

        let excess = ratio / params.thresh_ratio;
        let target_gain = if excess > 1.0 {
            (1.0 / excess.powf(params.slope_exp * weight)).clamp(params.min_gain, 1.0)
        } else {
            1.0
        };

        let alpha = if target_gain < smoothed_gains[b] {
            params.attack_alpha
        } else {
            params.release_alpha
        };
        smoothed_gains[b] = alpha * smoothed_gains[b] + (1.0 - alpha) * target_gain;
    }
}

/// Smooths a bin's gain across its 3-bin Hann main lobe to suppress musical noise.
#[inline]
fn smooth_lobe_gain(gains: &[f64], b: usize) -> f64 {
    let min_lobe = gains[b - 1].min(gains[b]).min(gains[b + 1]);
    let avg_lobe = 0.25 * gains[b - 1] + 0.50 * gains[b] + 0.25 * gains[b + 1];
    0.65 * min_lobe + 0.35 * avg_lobe
}

/// Applies stereo-linked Overlap-Add (OLA) STFT dynamic resonance suppression across all channels.
///
/// Uses ERB-scaled log-magnitude (geometric mean) local baseline estimation, smooth raised-cosine
/// frequency-band transitions, optional HPSS shimmer suppression, and sample-rate-independent
/// attack/release temporal smoothing.
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
    let clean_tilt = sanitize_param_f64(config.tilt_db_per_oct, 0.0, -6.0, 6.0);
    let clean_link = sanitize_param_f64(config.stereo_link, 1.0, 0.0, 1.0);
    let clean_ms = sanitize_param_f64(config.mid_side_balance, 0.0, -1.0, 1.0);
    let clean_attack_ms = sanitize_param_f64(config.attack_ms, DEFAULT_ATTACK_MS, 0.5, 200.0);
    let clean_release_ms = sanitize_param_f64(config.release_ms, DEFAULT_RELEASE_MS, 5.0, 500.0);
    let clean_hpss = sanitize_param_f64(config.hpss_shimmer_suppress, 0.0, 0.0, 1.0);
    let clean_focus = [
        sanitize_param_f64(config.focus_eq_db[0], 0.0, -12.0, 12.0),
        sanitize_param_f64(config.focus_eq_db[1], 0.0, -12.0, 12.0),
        sanitize_param_f64(config.focus_eq_db[2], 0.0, -12.0, 12.0),
    ];
    let sharpness = config.sharpness_bins.clamp(2, 24);
    let use_ms = num_channels == 2 && clean_ms.abs() > 1e-4;
    let use_per_channel = num_channels == 2 && (clean_link < 0.999 || use_ms);

    let band_weights: Vec<f64> = (0..HALF_BINS)
        .map(|b| {
            let f = (b as f64) * bin_hz;
            let w_mud = 0.35 * raised_cosine_band_mask(f, 250.0, 500.0, BAND_SKIRT_OCTAVES);
            let w_fizz =
                clean_fizz_w * raised_cosine_band_mask(f, 1800.0, 6800.0, BAND_SKIRT_OCTAVES);
            let w_sib =
                clean_sib_w * raised_cosine_band_mask(f, 5500.0, 14500.0, BAND_SKIRT_OCTAVES);
            let base = w_mud.max(w_fizz).max(w_sib) * focus_eq_multiplier(f, clean_focus);
            (base * tilt_multiplier(f, clean_tilt)).clamp(0.0, 3.5)
        })
        .collect();

    let erb_bounds = compute_erb_bounds(bin_hz, sharpness);

    let thresh_ratio = 10.0_f64.powf(clean_sens_db / 20.0);
    let min_gain = 10.0_f64.powf(-clean_depth_db / 20.0);

    // Sample-rate-independent temporal smoothing coefficients
    let hop_secs = (HOP as f64) / sr;
    let attack_alpha = (-hop_secs / (clean_attack_ms * 0.001)).exp();
    let release_alpha = (-hop_secs / (clean_release_ms * 0.001)).exp();
    let slope_exp = if config.hard_mode { 0.95 } else { 0.65 };
    let b_hpss_lo = ((2_500.0 / bin_hz).round() as usize).clamp(1, HALF_BINS - 2);
    let b_hpss_hi = ((14_500.0 / bin_hz).round() as usize).clamp(b_hpss_lo, HALF_BINS - 2);

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
    let mut ch_smoothed_gains: Vec<Vec<f64>> = vec![vec![1.0_f64; HALF_BINS]; num_channels];
    let mut linked_mag_hist = vec![[0.0_f64; 5]; HALF_BINS];
    let mut ch_mag_hist: Vec<Vec<[f64; 5]>> = vec![vec![[0.0_f64; 5]; HALF_BINS]; num_channels];
    let mut effective_gains = vec![1.0_f64; HALF_BINS];
    let mut hist_slot = 0_usize;

    // Mid/Side focus scale per channel when `use_ms` is active:
    // channel 0 = Mid (`1.0` when `clean_ms <= 0`, fades to `0.0` at `+1.0`),
    // channel 1 = Side (`1.0` when `clean_ms >= 0`, fades to `0.0` at `-1.0`).
    let ms_channel_scale = [
        (1.0 - clean_ms.max(0.0)).clamp(0.0, 1.0),
        (1.0 + clean_ms.min(0.0)).clamp(0.0, 1.0),
    ];

    let inv_n = 1.0 / (FFT_SIZE as f64);
    let inv_channels = 1.0 / (num_channels as f64);

    let mut pos = 0_usize;
    while pos + FFT_SIZE <= padded_len {
        // Forward STFT for each channel (or Mid/Side matrix when M/S balance is active)
        if use_ms {
            for i in 0..FFT_SIZE {
                let l = padded_channels[0][pos + i] * window[i];
                let r = padded_channels[1][pos + i] * window[i];
                specs[0][i] = Complex::new(0.5 * (l + r), 0.0);
                specs[1][i] = Complex::new(0.5 * (l - r), 0.0);
            }
            for spec in &mut specs {
                fft_fwd.process_with_scratch(spec, &mut scratch);
            }
        } else {
            for (ch_idx, padded) in padded_channels.iter().enumerate() {
                let spec = &mut specs[ch_idx];
                for i in 0..FFT_SIZE {
                    spec[i] = Complex::new(padded[pos + i] * window[i], 0.0);
                }
                fft_fwd.process_with_scratch(spec, &mut scratch);
            }
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

        let erb_params = ErbGainParams {
            band_weights: &band_weights,
            erb_bounds: &erb_bounds,
            thresh_ratio,
            min_gain,
            attack_alpha,
            release_alpha,
            slope_exp,
            hpss_shimmer_suppress: clean_hpss,
            b_hpss_lo,
            b_hpss_hi,
        };
        update_erb_gains(
            &log_mags,
            &prefix_log,
            &mut linked_mag_hist,
            hist_slot,
            &mut smoothed_gains,
            1.0,
            &erb_params,
        );

        if use_per_channel {
            for (ch_idx, spec) in specs.iter_mut().enumerate() {
                let focus = if use_ms {
                    ms_channel_scale[ch_idx]
                } else {
                    1.0
                };
                prefix_log[0] = 0.0;
                for b in 0..HALF_BINS {
                    let lm = spec[b].norm().max(1e-12).ln();
                    log_mags[b] = lm;
                    prefix_log[b + 1] = prefix_log[b] + lm;
                }
                let ch_gains = &mut ch_smoothed_gains[ch_idx];
                update_erb_gains(
                    &log_mags,
                    &prefix_log,
                    &mut ch_mag_hist[ch_idx],
                    hist_slot,
                    ch_gains,
                    focus,
                    &erb_params,
                );

                for b in 1..(HALF_BINS - 1) {
                    let g_ch = smooth_lobe_gain(ch_gains, b);
                    let g_link = 1.0 - focus * (1.0 - smooth_lobe_gain(&smoothed_gains, b));
                    let g = (clean_link * g_link + (1.0 - clean_link) * g_ch).clamp(min_gain, 1.0);
                    let eg = if config.delta_listen { 1.0 - g } else { g };
                    spec[b] *= eg;
                    spec[FFT_SIZE - b] = spec[b].conj();
                }
                if config.delta_listen {
                    spec[0] = Complex::new(0.0, 0.0);
                    spec[HALF_BINS - 1] = Complex::new(0.0, 0.0);
                }
                fft_inv.process_with_scratch(spec, &mut scratch);
            }

            if use_ms {
                for i in 0..FFT_SIZE {
                    let m = specs[0][i].re * inv_n * window[i];
                    let s = specs[1][i].re * inv_n * window[i];
                    out_accums[0][pos + i] += m + s;
                    out_accums[1][pos + i] += m - s;
                }
            } else {
                for (ch_idx, spec) in specs.iter().enumerate() {
                    let out_accum = &mut out_accums[ch_idx];
                    for i in 0..FFT_SIZE {
                        out_accum[pos + i] += spec[i].re * inv_n * window[i];
                    }
                }
            }
        } else {
            for (b, eg) in effective_gains
                .iter_mut()
                .enumerate()
                .take(HALF_BINS - 1)
                .skip(1)
            {
                let g = smooth_lobe_gain(&smoothed_gains, b);
                *eg = if config.delta_listen { 1.0 - g } else { g };
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
        }
        for i in 0..FFT_SIZE {
            let w = window[i];
            norm_accum[pos + i] += w * w;
        }

        hist_slot = (hist_slot + 1) % 5;
        pos += HOP;
    }

    let mut processed_channels: Vec<Vec<f64>> = out_accums
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

    if !config.delta_listen
        && let Some(raw_lpf) = config.ultrasonic_lpf_hz
        && raw_lpf.is_finite()
        && raw_lpf < sr * 0.48
    {
        let cutoff = raw_lpf.clamp(4_000.0, (sr * 0.47).min(22_000.0));
        for ch in &mut processed_channels {
            let mut lp = Biquad::lowpass(sr, cutoff, std::f64::consts::FRAC_1_SQRT_2);
            for s in ch.iter_mut() {
                *s = lp.process_sample(*s);
            }
        }
    }

    Ok(AudioBuffer {
        channels: processed_channels,
        sample_rate: buffer.sample_rate,
    })
}
