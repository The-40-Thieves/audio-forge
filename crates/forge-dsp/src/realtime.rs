//! Real-Time Zero-Allocation Streaming DSP Kernels for VST3, CLAP, and Live DAW Processing.
//!
//! Every processor in this module pre-allocates its state during `new()` / `prepare(sample_rate)`
//! and performs **zero heap allocations** inside `process_frame()` or `process_block()`,
//! guaranteeing glitch-free real-time audio-thread execution at any buffer size (32–4096 samples).

use std::f64::consts::{FRAC_1_SQRT_2, PI};

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

use crate::elite_dsp::{AllpassDisperser, JilesAthertonHysteresis};
use crate::loudness_limiter::{
    Biquad, CytomicSvf, DitherMode, LIMITER_LOOKAHEAD_MS, LimiterStyle, TRUE_PEAK_PHASES,
    flush_denormal_f64, sanitize_param_f64, sanitize_sample_f32,
};
use crate::restoration::{LowBandSplit, MASTERING_18_BANDS_HZ};
use crate::spectrum::{
    SPECTRUM_BANDS, SPECTRUM_FLOOR_DB, accumulate_stft_band_power, band_power_to_db,
    stft_band_bins, stft_band_power_norm,
};

/// Exponential one-pole parameter smoother to eliminate zipper noise during real-time knob automation.
#[derive(Debug, Clone, Copy)]
pub struct ParamSmoother {
    current: f64,
    target: f64,
    coeff: f64,
}

impl ParamSmoother {
    pub fn new(initial: f64, sample_rate: f64, smoothing_ms: f64) -> Self {
        let init = if initial.is_finite() { initial } else { 0.0 };
        let sr = sanitize_param_f64(sample_rate, 48_000.0, 8_000.0, 768_000.0);
        let tau = sanitize_param_f64(smoothing_ms * 0.001, 0.010, 0.0005, 10.0);
        let coeff = (-1.0 / (tau * sr)).exp();
        Self {
            current: init,
            target: init,
            coeff,
        }
    }

    #[inline]
    pub fn set_sample_rate(&mut self, sample_rate: f64, smoothing_ms: f64) {
        let sr = sanitize_param_f64(sample_rate, 48_000.0, 8_000.0, 768_000.0);
        let tau = sanitize_param_f64(smoothing_ms * 0.001, 0.010, 0.0005, 10.0);
        self.coeff = (-1.0 / (tau * sr)).exp();
    }

    #[inline]
    pub fn set_target(&mut self, target: f64) {
        if target.is_finite() {
            self.target = target;
        }
    }

    #[inline]
    pub fn next_val(&mut self) -> f64 {
        self.current =
            flush_denormal_f64(self.coeff * self.current + (1.0 - self.coeff) * self.target);
        self.current
    }
}

/// Real-Time Zero-Allocation **OpenSoothe** OLA-STFT Dynamic Spectral Resonance & AI De-Fizz Processor.
///
/// Uses a 2048-sample Hann-windowed ring buffer with 512-sample hop (`75%` overlap, exact COLA).
/// Each input sample reaches the output exactly `fft_size` (2048) samples later, which is the
/// latency reported to the DAW for sample-accurate PDC.
pub struct RealtimeOpenSoothe {
    sample_rate: f64,
    fft_size: usize,
    hop_size: usize,
    half_bins: usize,
    window: Vec<f64>,
    /// `stft_band_power_norm(window)`: turns summed bin powers into sine-referenced band power.
    spectrum_power_norm: f64,
    band_weights: Vec<f64>,
    fft_fwd: Arc<dyn Fft<f64>>,
    fft_inv: Arc<dyn Fft<f64>>,
    in_ring_l: Vec<f64>,
    in_ring_r: Vec<f64>,
    out_ring_l: Vec<f64>,
    out_ring_r: Vec<f64>,
    norm_ring: Vec<f64>,
    ring_pos: usize,
    hop_counter: usize,
    spec_scratch: Vec<Complex<f64>>,
    spec_scratch_r: Vec<Complex<f64>>,
    /// Pre-allocated rustfft scratch, so `process_with_scratch` never allocates per hop.
    fft_scratch: Vec<Complex<f64>>,
    mags_scratch: Vec<f64>,
    smoothed_gains_l: Vec<f64>,
    smoothed_gains_r: Vec<f64>,
    tilt_db_per_oct: f64,
    stereo_link: f64,
    mid_side_balance: f64,
    last_gr_db: f32,
    last_spectrum_db: [f32; 32],
    last_band_gr_db: [f32; 32],
}

impl RealtimeOpenSoothe {
    pub fn new(sample_rate: f64) -> Self {
        let fft_size = 2048;
        let hop_size = 512;
        let half_bins = fft_size / 2 + 1;
        let mut planner = FftPlanner::<f64>::new();
        let fft_fwd = planner.plan_fft_forward(fft_size);
        let fft_inv = planner.plan_fft_inverse(fft_size);
        let scratch_len = fft_fwd
            .get_inplace_scratch_len()
            .max(fft_inv.get_inplace_scratch_len());

        let window: Vec<f64> = (0..fft_size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * (i as f64) / (fft_size as f64)).cos()))
            .collect();
        let spectrum_power_norm = stft_band_power_norm(&window);

        let mut proc = Self {
            sample_rate: sample_rate.max(8000.0),
            fft_size,
            hop_size,
            half_bins,
            window,
            spectrum_power_norm,
            band_weights: vec![0.0; half_bins],
            fft_fwd,
            fft_inv,
            in_ring_l: vec![0.0; fft_size],
            in_ring_r: vec![0.0; fft_size],
            out_ring_l: vec![0.0; fft_size * 2],
            out_ring_r: vec![0.0; fft_size * 2],
            norm_ring: vec![0.0; fft_size * 2],
            ring_pos: 0,
            hop_counter: 0,
            spec_scratch: vec![Complex::new(0.0, 0.0); fft_size],
            spec_scratch_r: vec![Complex::new(0.0, 0.0); fft_size],
            fft_scratch: vec![Complex::new(0.0, 0.0); scratch_len],
            mags_scratch: vec![0.0; half_bins],
            smoothed_gains_l: vec![1.0; half_bins],
            smoothed_gains_r: vec![1.0; half_bins],
            tilt_db_per_oct: 0.0,
            stereo_link: 1.0,
            mid_side_balance: 0.0,
            last_gr_db: 0.0,
            last_spectrum_db: [SPECTRUM_FLOOR_DB; SPECTRUM_BANDS],
            last_band_gr_db: [0.0; 32],
        };
        proc.update_band_weights(1.35, 1.15);
        proc
    }

    /// Delay from input to output: a sample written at ring position `k` is overlap-added
    /// into the output ring at `k + fft_size`, so the delay is the full frame, not `fft_size - hop`.
    pub fn latency_samples(&self) -> u32 {
        self.fft_size as u32
    }

    pub fn reset(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate.max(8000.0);
        self.in_ring_l.fill(0.0);
        self.in_ring_r.fill(0.0);
        self.out_ring_l.fill(0.0);
        self.out_ring_r.fill(0.0);
        self.norm_ring.fill(0.0);
        self.smoothed_gains_l.fill(1.0);
        self.smoothed_gains_r.fill(1.0);
        self.ring_pos = 0;
        self.hop_counter = 0;
        self.last_gr_db = 0.0;
        self.last_spectrum_db = [SPECTRUM_FLOOR_DB; SPECTRUM_BANDS];
        self.last_band_gr_db = [0.0; 32];
        self.update_band_weights(1.35, 1.15);
    }

    pub fn update_band_weights(&mut self, fizz_weight: f64, sibilance_weight: f64) {
        let bin_hz = self.sample_rate / (self.fft_size as f64);
        let clean_fizz = sanitize_param_f64(fizz_weight, 1.25, 0.0, 2.0);
        let clean_sib = sanitize_param_f64(sibilance_weight, 1.0, 0.0, 2.0);
        let skirt_mask = |f: f64, lo: f64, hi: f64| -> f64 {
            const SKIRT_OCT: f64 = 1.0 / 3.0;
            if f <= 0.0 || lo >= hi {
                return 0.0;
            }
            let lf = f.log2();
            let llo = lo.log2();
            let lhi = hi.log2();
            if lf <= llo - SKIRT_OCT || lf >= lhi + SKIRT_OCT {
                0.0
            } else if lf < llo {
                let t = (lf - (llo - SKIRT_OCT)) / SKIRT_OCT;
                0.5 * (1.0 - (PI * t).cos())
            } else if lf <= lhi {
                1.0
            } else {
                let t = (lf - lhi) / SKIRT_OCT;
                0.5 * (1.0 + (PI * t).cos())
            }
        };
        for (b, w_out) in self.band_weights.iter_mut().enumerate() {
            let f = (b as f64) * bin_hz;
            let w_mud = 0.35 * skirt_mask(f, 250.0, 500.0);
            let w_fizz = clean_fizz * skirt_mask(f, 1800.0, 6800.0);
            let w_sib = clean_sib * skirt_mask(f, 5500.0, 14500.0);
            *w_out = w_mud.max(w_fizz).max(w_sib);
        }
    }

    pub fn last_gain_reduction_db(&self) -> f32 {
        self.last_gr_db
    }

    pub fn last_spectrum_32_db(&self) -> [f32; 32] {
        self.last_spectrum_db
    }

    pub fn last_band_gr_32_db(&self) -> [f32; 32] {
        self.last_band_gr_db
    }

    /// Sets spectral sensitivity tilt in dB/octave pivoted at 1 kHz (`-6.0..=+6.0`). Zero-alloc.
    #[inline]
    pub fn set_tilt_db_per_oct(&mut self, tilt_db_per_oct: f64) {
        self.tilt_db_per_oct = sanitize_param_f64(tilt_db_per_oct, 0.0, -6.0, 6.0);
    }

    /// Sets stereo gain linking (`0.0` = dual-mono, `1.0` = linked stereo). Zero-alloc.
    #[inline]
    pub fn set_stereo_link(&mut self, stereo_link: f64) {
        self.stereo_link = sanitize_param_f64(stereo_link, 1.0, 0.0, 1.0);
    }

    /// Sets Mid/Side focus (`-1.0` = Mid only, `0.0` = balanced, `+1.0` = Side only). Zero-alloc.
    #[inline]
    pub fn set_mid_side_balance(&mut self, mid_side_balance: f64) {
        self.mid_side_balance = sanitize_param_f64(mid_side_balance, 0.0, -1.0, 1.0);
    }

    #[inline]
    pub fn process_stereo_sample(
        &mut self,
        in_l: f32,
        in_r: f32,
        sensitivity_db: f64,
        max_depth_db: f64,
        sharpness_bins: usize,
        delta_listen: bool,
    ) -> (f32, f32) {
        let mask_in = self.fft_size - 1;
        let mask_out = self.fft_size * 2 - 1;

        self.in_ring_l[self.ring_pos & mask_in] = sanitize_sample_f32(in_l);
        self.in_ring_r[self.ring_pos & mask_in] = sanitize_sample_f32(in_r);

        let read_idx = self.ring_pos & mask_out;
        let denom = self.norm_ring[read_idx].max(1e-9);
        let out_l = flush_denormal_f64(self.out_ring_l[read_idx] / denom) as f32;
        let out_r = flush_denormal_f64(self.out_ring_r[read_idx] / denom) as f32;

        // Clear consumed output slot for future overlap-add
        self.out_ring_l[read_idx] = 0.0;
        self.out_ring_r[read_idx] = 0.0;
        self.norm_ring[read_idx] = 0.0;

        self.ring_pos = self.ring_pos.wrapping_add(1);
        self.hop_counter += 1;

        if self.hop_counter >= self.hop_size {
            self.hop_counter = 0;
            let clean_sens = sanitize_param_f64(sensitivity_db, 3.5, 0.5, 24.0);
            let clean_depth = sanitize_param_f64(max_depth_db, 9.0, 0.5, 24.0);
            self.process_stft_hop(clean_sens, clean_depth, sharpness_bins, delta_listen);
        }

        (out_l, out_r)
    }

    fn process_stft_hop(
        &mut self,
        sensitivity_db: f64,
        max_depth_db: f64,
        sharpness_bins: usize,
        delta_listen: bool,
    ) {
        let thresh_ratio = 10.0_f64.powf(sensitivity_db.max(0.5) / 20.0);
        let min_gain = 10.0_f64.powf(-max_depth_db.clamp(0.5, 24.0) / 20.0);
        let radius = sharpness_bins.clamp(4, 24);
        let mask_in = self.fft_size - 1;
        let mask_out = self.fft_size * 2 - 1;
        let frame_start = self.ring_pos.wrapping_sub(self.fft_size);
        let inv_n = 1.0 / (self.fft_size as f64);
        let bin_hz = self.sample_rate / (self.fft_size as f64);
        let hop_secs = (self.hop_size as f64) / self.sample_rate.max(1.0);
        let attack_alpha = (-hop_secs / 0.010).exp();
        let release_alpha = (-hop_secs / 0.055).exp();
        let mut band_pow = [0.0_f64; SPECTRUM_BANDS];

        let mut min_g_seen = 1.0_f64;
        let use_ms = self.mid_side_balance.abs() > 1e-4;
        let ms_focus = [
            (1.0 - self.mid_side_balance.max(0.0)).clamp(0.0, 1.0),
            (1.0 + self.mid_side_balance.min(0.0)).clamp(0.0, 1.0),
        ];
        // M = 0.5*(L+R) and S = 0.5*(L-R) satisfy M^2 + S^2 = 0.5*(L^2 + R^2), so double the
        // per-channel spectrum norm in Mid/Side mode to keep displayed levels invariant.
        let spec_norm = if use_ms {
            self.spectrum_power_norm
        } else {
            0.5 * self.spectrum_power_norm
        };

        // Pass 1: FFT both channels and update both current-hop smoothed gain curves before
        // applying stereo linking so L and R always share current-hop gains symmetrically.
        for (ch_idx, &ms_f) in ms_focus.iter().enumerate() {
            let focus = if use_ms { ms_f } else { 1.0 };
            let spec = if ch_idx == 0 {
                &mut self.spec_scratch
            } else {
                &mut self.spec_scratch_r
            };
            for (i, slot) in spec.iter_mut().enumerate().take(self.fft_size) {
                let idx = frame_start.wrapping_add(i) & mask_in;
                let sample = if use_ms {
                    let l = self.in_ring_l[idx];
                    let r = self.in_ring_r[idx];
                    if ch_idx == 0 {
                        0.5 * (l + r)
                    } else {
                        0.5 * (l - r)
                    }
                } else if ch_idx == 0 {
                    self.in_ring_l[idx]
                } else {
                    self.in_ring_r[idx]
                };
                *slot = Complex::new(sample * self.window[i], 0.0);
            }

            self.fft_fwd
                .process_with_scratch(spec, &mut self.fft_scratch);

            for (mag, c) in self
                .mags_scratch
                .iter_mut()
                .zip(spec.iter())
                .take(self.half_bins)
            {
                *mag = c.norm().max(1e-12);
            }
            // Input spectrum for the display, summed over both channels before any gain.
            accumulate_stft_band_power(&self.mags_scratch, bin_hz, spec_norm, &mut band_pow);

            let smoothed = if ch_idx == 0 {
                &mut self.smoothed_gains_l
            } else {
                &mut self.smoothed_gains_r
            };

            for (b, smoothed_val) in smoothed
                .iter_mut()
                .enumerate()
                .take(self.half_bins - 1)
                .skip(1)
            {
                let base_w = self.band_weights[b] * focus;
                let weight = if self.tilt_db_per_oct.abs() > 1e-6 && base_w > 0.01 {
                    let f = (b as f64) * bin_hz;
                    let tilt_mul = if f > 20.0 {
                        10.0_f64
                            .powf((self.tilt_db_per_oct * (f / 1000.0).log2()) / 20.0)
                            .clamp(0.15, 3.0)
                    } else {
                        1.0
                    };
                    (base_w * tilt_mul).clamp(0.0, 3.0)
                } else {
                    base_w
                };
                if weight <= 0.01 {
                    *smoothed_val = 1.0;
                    continue;
                }
                let b_start = b.saturating_sub(radius).max(1);
                let b_end = (b + radius).min(self.half_bins - 1);
                let mut sum_neigh = 0.0_f64;
                let mut count = 0_usize;
                for (nb, &mag_val) in self
                    .mags_scratch
                    .iter()
                    .enumerate()
                    .take(b_end + 1)
                    .skip(b_start)
                {
                    if nb.abs_diff(b) > 1 {
                        sum_neigh += mag_val;
                        count += 1;
                    }
                }
                let local_env = if count > 0 {
                    (sum_neigh / (count as f64)).max(1e-12)
                } else {
                    self.mags_scratch[b]
                };

                let excess = (self.mags_scratch[b] / local_env) / thresh_ratio;
                let target_gain = if excess > 1.0 {
                    (1.0 / excess.powf(0.65 * weight)).clamp(min_gain, 1.0)
                } else {
                    1.0
                };

                let alpha = if target_gain < *smoothed_val {
                    attack_alpha
                } else {
                    release_alpha
                };
                *smoothed_val = alpha * (*smoothed_val) + (1.0 - alpha) * target_gain;
            }
        }

        // Pass 2: Apply symmetric stereo-linked lobe-smoothed gains, IFFT, and overlap-add.
        for b in 1..(self.half_bins - 1) {
            let lobe_l = {
                let min_l = self.smoothed_gains_l[b - 1]
                    .min(self.smoothed_gains_l[b])
                    .min(self.smoothed_gains_l[b + 1]);
                let avg_l = 0.25 * self.smoothed_gains_l[b - 1]
                    + 0.50 * self.smoothed_gains_l[b]
                    + 0.25 * self.smoothed_gains_l[b + 1];
                0.65 * min_l + 0.35 * avg_l
            };
            let lobe_r = {
                let min_r = self.smoothed_gains_r[b - 1]
                    .min(self.smoothed_gains_r[b])
                    .min(self.smoothed_gains_r[b + 1]);
                let avg_r = 0.25 * self.smoothed_gains_r[b - 1]
                    + 0.50 * self.smoothed_gains_r[b]
                    + 0.25 * self.smoothed_gains_r[b + 1];
                0.65 * min_r + 0.35 * avg_r
            };
            let (gl, gr) = if !use_ms && self.stereo_link > 1e-4 {
                let g_link = lobe_l.min(lobe_r);
                (
                    self.stereo_link * g_link + (1.0 - self.stereo_link) * lobe_l,
                    self.stereo_link * g_link + (1.0 - self.stereo_link) * lobe_r,
                )
            } else {
                (lobe_l, lobe_r)
            };
            let g_min = gl.min(gr);
            if g_min < min_g_seen {
                min_g_seen = g_min;
            }
            let eff_l = if delta_listen { 1.0 - gl } else { gl };
            let eff_r = if delta_listen { 1.0 - gr } else { gr };
            self.spec_scratch[b] *= eff_l;
            self.spec_scratch[self.fft_size - b] = self.spec_scratch[b].conj();
            self.spec_scratch_r[b] *= eff_r;
            self.spec_scratch_r[self.fft_size - b] = self.spec_scratch_r[b].conj();
        }

        if delta_listen {
            self.spec_scratch[0] = Complex::new(0.0, 0.0);
            self.spec_scratch[self.half_bins - 1] = Complex::new(0.0, 0.0);
            self.spec_scratch_r[0] = Complex::new(0.0, 0.0);
            self.spec_scratch_r[self.half_bins - 1] = Complex::new(0.0, 0.0);
        }

        self.fft_inv
            .process_with_scratch(&mut self.spec_scratch, &mut self.fft_scratch);
        self.fft_inv
            .process_with_scratch(&mut self.spec_scratch_r, &mut self.fft_scratch);

        let out_base = self.ring_pos;
        for i in 0..self.fft_size {
            let w = self.window[i];
            let val_l = self.spec_scratch[i].re * inv_n * w;
            let val_r = self.spec_scratch_r[i].re * inv_n * w;
            let idx = out_base.wrapping_add(i) & mask_out;
            if use_ms {
                self.out_ring_l[idx] += val_l + val_r;
                self.out_ring_r[idx] += val_l - val_r;
            } else {
                self.out_ring_l[idx] += val_l;
                self.out_ring_r[idx] += val_r;
            }
            self.norm_ring[idx] += w * w;
        }

        self.last_gr_db = (-20.0 * min_g_seen.max(1e-6).log10()) as f32;

        // 32-band input power spectrum (L/R power average) and dynamic gain-reduction curve
        self.last_spectrum_db = band_power_to_db(&band_pow);
        for band_idx in 0..SPECTRUM_BANDS {
            let (b0, b1) = stft_band_bins(band_idx, bin_hz, self.half_bins);
            let mut min_band_g = 1.0_f64;
            for b in b0..=b1 {
                let g_lr = 0.5 * (self.smoothed_gains_l[b] + self.smoothed_gains_r[b]);
                if g_lr < min_band_g {
                    min_band_g = g_lr;
                }
            }
            self.last_band_gr_db[band_idx] =
                (20.0 * min_band_g.max(1.0e-4).log10() as f32).clamp(-24.0, 0.0);
        }
    }
}

/// Real-Time Zero-Allocation **OpenSpacer** 4-Channel Sidechain Spectral Unmasker.
///
/// Carves the exact spectral footprint of the sidechain input (Vocal or Kick) out of the
/// target track (Instruments or Bass) in real time with zero heap allocations.
pub struct RealtimeOpenSpacer {
    sample_rate: f64,
    fft_size: usize,
    hop_size: usize,
    half_bins: usize,
    window: Vec<f64>,
    fft_fwd: Arc<dyn Fft<f64>>,
    fft_inv: Arc<dyn Fft<f64>>,
    tgt_ring_l: Vec<f64>,
    tgt_ring_r: Vec<f64>,
    sc_ring: Vec<f64>,
    out_ring_l: Vec<f64>,
    out_ring_r: Vec<f64>,
    norm_ring: Vec<f64>,
    ring_pos: usize,
    hop_counter: usize,
    tgt_spec_l: Vec<Complex<f64>>,
    tgt_spec_r: Vec<Complex<f64>>,
    sc_spec: Vec<Complex<f64>>,
    /// Pre-allocated rustfft scratch, so `process_with_scratch` never allocates per hop.
    fft_scratch: Vec<Complex<f64>>,
    spectrum_power_norm: f64,
    mags_scratch: Vec<f64>,
    smoothed_gains: Vec<f64>,
    delta_listen: bool,
    mid_side_balance: f64,
    last_carve_db: f32,
    last_spectrum_db: [f32; SPECTRUM_BANDS],
    last_carve_bands_db: [f32; SPECTRUM_BANDS],
}

impl RealtimeOpenSpacer {
    pub fn new(sample_rate: f64) -> Self {
        let fft_size = 2048;
        let hop_size = 512;
        let half_bins = fft_size / 2 + 1;
        let mut planner = FftPlanner::<f64>::new();
        let fft_fwd = planner.plan_fft_forward(fft_size);
        let fft_inv = planner.plan_fft_inverse(fft_size);
        let scratch_len = fft_fwd
            .get_inplace_scratch_len()
            .max(fft_inv.get_inplace_scratch_len());
        let window: Vec<f64> = (0..fft_size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * (i as f64) / (fft_size as f64)).cos()))
            .collect();
        let spectrum_power_norm = stft_band_power_norm(&window);

        Self {
            sample_rate: sample_rate.max(8000.0),
            fft_size,
            hop_size,
            half_bins,
            window,
            fft_fwd,
            fft_inv,
            tgt_ring_l: vec![0.0; fft_size],
            tgt_ring_r: vec![0.0; fft_size],
            sc_ring: vec![0.0; fft_size],
            out_ring_l: vec![0.0; fft_size * 2],
            out_ring_r: vec![0.0; fft_size * 2],
            norm_ring: vec![0.0; fft_size * 2],
            ring_pos: 0,
            hop_counter: 0,
            tgt_spec_l: vec![Complex::new(0.0, 0.0); fft_size],
            tgt_spec_r: vec![Complex::new(0.0, 0.0); fft_size],
            sc_spec: vec![Complex::new(0.0, 0.0); fft_size],
            fft_scratch: vec![Complex::new(0.0, 0.0); scratch_len],
            spectrum_power_norm,
            mags_scratch: vec![0.0; half_bins],
            smoothed_gains: vec![1.0; half_bins],
            delta_listen: false,
            mid_side_balance: 0.0,
            last_carve_db: 0.0,
            last_spectrum_db: [SPECTRUM_FLOOR_DB; SPECTRUM_BANDS],
            last_carve_bands_db: [0.0; SPECTRUM_BANDS],
        }
    }

    /// Delay from input to output: a sample written at ring position `k` is overlap-added
    /// into the output ring at `k + fft_size`, so the delay is the full frame, not `fft_size - hop`.
    pub fn latency_samples(&self) -> u32 {
        self.fft_size as u32
    }

    pub fn reset(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate.max(8000.0);
        self.tgt_ring_l.fill(0.0);
        self.tgt_ring_r.fill(0.0);
        self.sc_ring.fill(0.0);
        self.out_ring_l.fill(0.0);
        self.out_ring_r.fill(0.0);
        self.norm_ring.fill(0.0);
        self.smoothed_gains.fill(1.0);
        self.ring_pos = 0;
        self.hop_counter = 0;
        self.last_carve_db = 0.0;
        self.last_spectrum_db = [SPECTRUM_FLOOR_DB; SPECTRUM_BANDS];
        self.last_carve_bands_db = [0.0; SPECTRUM_BANDS];
    }

    pub fn last_carve_db(&self) -> f32 {
        self.last_carve_db
    }

    /// Returns the latest 32-band target input spectrum in dBFS (updated per STFT hop).
    pub fn last_spectrum_db(&self) -> [f32; SPECTRUM_BANDS] {
        self.last_spectrum_db
    }

    /// Returns the latest 32-band dynamic carve gain reduction in dB (0..=24 dB, updated per hop).
    pub fn last_carve_bands_db(&self) -> [f32; SPECTRUM_BANDS] {
        self.last_carve_bands_db
    }

    /// Enables Delta-Listen mode to solo *only* the carved frequencies. Zero-alloc.
    #[inline]
    pub fn set_delta_listen(&mut self, delta_listen: bool) {
        self.delta_listen = delta_listen;
    }

    /// Sets Mid/Side carving focus (`-1.0` = Mid only, `0.0` = balanced, `+1.0` = Side only). Zero-alloc.
    #[inline]
    pub fn set_mid_side_balance(&mut self, mid_side_balance: f64) {
        self.mid_side_balance = sanitize_param_f64(mid_side_balance, 0.0, -1.0, 1.0);
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn process_sample(
        &mut self,
        tgt_l: f32,
        tgt_r: f32,
        sc_l: f32,
        sc_r: f32,
        low_hz: f64,
        high_hz: f64,
        amount: f64,
        max_cut_db: f64,
    ) -> (f32, f32) {
        let mask_in = self.fft_size - 1;
        let mask_out = self.fft_size * 2 - 1;

        self.tgt_ring_l[self.ring_pos & mask_in] = sanitize_sample_f32(tgt_l);
        self.tgt_ring_r[self.ring_pos & mask_in] = sanitize_sample_f32(tgt_r);
        self.sc_ring[self.ring_pos & mask_in] =
            0.5 * (sanitize_sample_f32(sc_l) + sanitize_sample_f32(sc_r));

        let read_idx = self.ring_pos & mask_out;
        let denom = self.norm_ring[read_idx].max(1e-9);
        let out_l = flush_denormal_f64(self.out_ring_l[read_idx] / denom) as f32;
        let out_r = flush_denormal_f64(self.out_ring_r[read_idx] / denom) as f32;

        self.out_ring_l[read_idx] = 0.0;
        self.out_ring_r[read_idx] = 0.0;
        self.norm_ring[read_idx] = 0.0;

        self.ring_pos = self.ring_pos.wrapping_add(1);
        self.hop_counter += 1;

        if self.hop_counter >= self.hop_size {
            self.hop_counter = 0;
            let clean_low = sanitize_param_f64(low_hz, 180.0, 20.0, 20_000.0);
            let clean_high = sanitize_param_f64(high_hz, 8_000.0, clean_low + 10.0, 22_000.0);
            let clean_amt = sanitize_param_f64(amount, 0.5, 0.0, 1.0);
            let clean_cut = sanitize_param_f64(max_cut_db, 6.0, 0.5, 24.0);
            self.process_unmask_hop(clean_low, clean_high, clean_amt, clean_cut);
        }

        (out_l, out_r)
    }

    fn process_unmask_hop(&mut self, low_hz: f64, high_hz: f64, amount: f64, max_cut_db: f64) {
        let mask_in = self.fft_size - 1;
        let mask_out = self.fft_size * 2 - 1;
        let frame_start = self.ring_pos.wrapping_sub(self.fft_size);
        let bin_hz = self.sample_rate / (self.fft_size as f64);
        let inv_n = 1.0 / (self.fft_size as f64);
        let min_gain = 10.0_f64.powf(-max_cut_db.clamp(0.5, 24.0) / 20.0);

        let use_ms = self.mid_side_balance.abs() > 1e-4;
        let mid_scale = (1.0 - self.mid_side_balance.max(0.0)).clamp(0.0, 1.0);
        let side_scale = (1.0 + self.mid_side_balance.min(0.0)).clamp(0.0, 1.0);

        for i in 0..self.fft_size {
            let idx = frame_start.wrapping_add(i) & mask_in;
            let w = self.window[i];
            let l = self.tgt_ring_l[idx] * w;
            let r = self.tgt_ring_r[idx] * w;
            if use_ms {
                self.tgt_spec_l[i] = Complex::new(0.5 * (l + r), 0.0);
                self.tgt_spec_r[i] = Complex::new(0.5 * (l - r), 0.0);
            } else {
                self.tgt_spec_l[i] = Complex::new(l, 0.0);
                self.tgt_spec_r[i] = Complex::new(r, 0.0);
            }
            self.sc_spec[i] = Complex::new(self.sc_ring[idx] * w, 0.0);
        }

        self.fft_fwd
            .process_with_scratch(&mut self.tgt_spec_l, &mut self.fft_scratch);
        self.fft_fwd
            .process_with_scratch(&mut self.tgt_spec_r, &mut self.fft_scratch);
        self.fft_fwd
            .process_with_scratch(&mut self.sc_spec, &mut self.fft_scratch);

        let mut band_pow = [0.0_f64; SPECTRUM_BANDS];
        let spec_norm = if use_ms {
            self.spectrum_power_norm
        } else {
            0.5 * self.spectrum_power_norm
        };
        for b in 0..self.half_bins {
            self.mags_scratch[b] = self.tgt_spec_l[b].norm().max(1e-12);
        }
        accumulate_stft_band_power(&self.mags_scratch, bin_hz, spec_norm, &mut band_pow);
        for b in 0..self.half_bins {
            self.mags_scratch[b] = self.tgt_spec_r[b].norm().max(1e-12);
        }
        accumulate_stft_band_power(&self.mags_scratch, bin_hz, spec_norm, &mut band_pow);

        let mut min_g_seen = 1.0_f64;

        for b in 1..(self.half_bins - 1) {
            let f = (b as f64) * bin_hz;
            if f < low_hz || f > high_hz {
                self.smoothed_gains[b] = 1.0;
                continue;
            }
            let sc_mag = self.sc_spec[b].norm() / (self.fft_size as f64);
            let sc_db = 20.0 * sc_mag.max(1e-9).log10();
            let over_db = (sc_db + 48.0).max(0.0);
            let cut_db = (over_db * 0.35 * amount.clamp(0.0, 1.0)).min(max_cut_db);
            let target_g = 10.0_f64.powf(-cut_db / 20.0).clamp(min_gain, 1.0);

            if target_g < self.smoothed_gains[b] {
                self.smoothed_gains[b] = 0.30 * self.smoothed_gains[b] + 0.70 * target_g;
            } else {
                self.smoothed_gains[b] = 0.85 * self.smoothed_gains[b] + 0.15 * target_g;
            }
        }

        for b in 1..(self.half_bins - 1) {
            let g_raw = 0.25 * self.smoothed_gains[b - 1]
                + 0.50 * self.smoothed_gains[b]
                + 0.25 * self.smoothed_gains[b + 1];
            if g_raw < min_g_seen {
                min_g_seen = g_raw;
            }
            let (gl, gr) = if use_ms {
                let gm = (1.0 - mid_scale * (1.0 - g_raw)).clamp(0.0, 1.0);
                let gs = (1.0 - side_scale * (1.0 - g_raw)).clamp(0.0, 1.0);
                (gm, gs)
            } else {
                (g_raw, g_raw)
            };
            let (egl, egr) = if self.delta_listen {
                (1.0 - gl, 1.0 - gr)
            } else {
                (gl, gr)
            };
            self.tgt_spec_l[b] *= egl;
            self.tgt_spec_l[self.fft_size - b] = self.tgt_spec_l[b].conj();
            self.tgt_spec_r[b] *= egr;
            self.tgt_spec_r[self.fft_size - b] = self.tgt_spec_r[b].conj();
        }
        if self.delta_listen {
            self.tgt_spec_l[0] = Complex::new(0.0, 0.0);
            self.tgt_spec_l[self.half_bins - 1] = Complex::new(0.0, 0.0);
            self.tgt_spec_r[0] = Complex::new(0.0, 0.0);
            self.tgt_spec_r[self.half_bins - 1] = Complex::new(0.0, 0.0);
        }

        self.fft_inv
            .process_with_scratch(&mut self.tgt_spec_l, &mut self.fft_scratch);
        self.fft_inv
            .process_with_scratch(&mut self.tgt_spec_r, &mut self.fft_scratch);

        let out_base = self.ring_pos;
        for i in 0..self.fft_size {
            let w = self.window[i];
            let idx = out_base.wrapping_add(i) & mask_out;
            let a = self.tgt_spec_l[i].re * inv_n * w;
            let b = self.tgt_spec_r[i].re * inv_n * w;
            if use_ms {
                self.out_ring_l[idx] += a + b;
                self.out_ring_r[idx] += a - b;
            } else {
                self.out_ring_l[idx] += a;
                self.out_ring_r[idx] += b;
            }
            self.norm_ring[idx] += w * w;
        }

        self.last_carve_db = (-20.0 * min_g_seen.max(1e-6).log10()) as f32;
        for (band_idx, &p) in band_pow.iter().enumerate() {
            let db = if p > 0.0 {
                (10.0 * p.log10()) as f32
            } else {
                SPECTRUM_FLOOR_DB
            };
            self.last_spectrum_db[band_idx] = db.clamp(SPECTRUM_FLOOR_DB, 6.0);
        }
        for band_idx in 0..SPECTRUM_BANDS {
            let (b_lo, b_hi) = stft_band_bins(band_idx, bin_hz, self.half_bins);
            let mut min_band_g = 1.0_f64;
            for &g in &self.smoothed_gains[b_lo..=b_hi] {
                if g < min_band_g {
                    min_band_g = g;
                }
            }
            self.last_carve_bands_db[band_idx] =
                (-20.0 * min_band_g.max(1e-6).log10()).clamp(0.0, 24.0) as f32;
        }
    }
}

/// Real-Time Zero-Latency **Vocal Surgical Channel Strip**
/// (Slew-Rate Mouth De-Click + `<115 Hz` Dynamic Plosive Suppressor + Breath Gate + Split-Band De-Esser).
///
/// The plosive bands are complementary and the de-esser subtracts part of a unit-peak band-pass,
/// so with nothing being ducked the strip passes its input through with a flat response.
pub struct RealtimeVocalSurgical {
    prev_l: f64,
    prev2_l: f64,
    prev_r: f64,
    prev2_r: f64,
    slew_env_l: f64,
    slew_env_r: f64,
    plosive_split_l: LowBandSplit,
    plosive_split_r: LowBandSplit,
    low_env: f64,
    mid_env: f64,
    sib_bp_l: Biquad,
    sib_bp_r: Biquad,
    sib_env: f64,
    gate_env: f64,
}

impl RealtimeVocalSurgical {
    pub fn new(sample_rate: f64) -> Self {
        let sr = sample_rate.max(8000.0);
        Self {
            prev_l: 0.0,
            prev2_l: 0.0,
            prev_r: 0.0,
            prev2_r: 0.0,
            slew_env_l: 1e-4,
            slew_env_r: 1e-4,
            plosive_split_l: LowBandSplit::new(sr, 115.0),
            plosive_split_r: LowBandSplit::new(sr, 115.0),
            low_env: 0.0,
            mid_env: 0.0,
            sib_bp_l: Biquad::bandpass(sr, 7200.0, 1.4),
            sib_bp_r: Biquad::bandpass(sr, 7200.0, 1.4),
            sib_env: 0.0,
            gate_env: 0.0,
        }
    }

    pub fn reset(&mut self, sample_rate: f64) {
        *self = Self::new(sample_rate);
    }

    #[inline]
    pub fn process_stereo_sample(
        &mut self,
        in_l: f32,
        in_r: f32,
        declick_sensitivity: f64,
        deplosive_strength: f64,
        deess_threshold_db: f64,
        breath_gate_db: f64,
    ) -> (f32, f32) {
        let l = sanitize_sample_f32(in_l);
        let r = sanitize_sample_f32(in_r);
        let clean_declick = sanitize_param_f64(declick_sensitivity, 0.5, 0.0, 1.0);
        let clean_deplosive = sanitize_param_f64(deplosive_strength, 0.5, 0.0, 1.0);
        let clean_deess_db = sanitize_param_f64(deess_threshold_db, -20.0, -60.0, 0.0);
        let clean_gate_db = sanitize_param_f64(breath_gate_db, -55.0, -90.0, -10.0);

        let (l, r) = self.declick(l, r, clean_declick);
        let (l, r) = self.duck_plosives(l, r, clean_deplosive);
        let (l, r) = self.deess(l, r, clean_deess_db);
        let (l, r) = self.gate_breaths(l, r, clean_gate_db);

        (flush_denormal_f64(l) as f32, flush_denormal_f64(r) as f32)
    }

    /// 1. Real-Time Slew-Rate Mouth De-Clicker
    #[inline]
    fn declick(&mut self, mut l: f64, mut r: f64, sensitivity: f64) -> (f64, f64) {
        if sensitivity > 0.01 {
            let slew_l = (l - self.prev_l).abs();
            let slew_r = (r - self.prev_r).abs();
            let thresh_mult = 12.0 - 7.5 * sensitivity;
            if slew_l > self.slew_env_l * thresh_mult && slew_l > 0.015 {
                l = 0.5 * (self.prev_l + self.prev2_l);
            } else {
                self.slew_env_l =
                    flush_denormal_f64(0.992 * self.slew_env_l + 0.008 * slew_l.max(1e-5))
                        .max(1e-5);
            }
            if slew_r > self.slew_env_r * thresh_mult && slew_r > 0.015 {
                r = 0.5 * (self.prev_r + self.prev2_r);
            } else {
                self.slew_env_r =
                    flush_denormal_f64(0.992 * self.slew_env_r + 0.008 * slew_r.max(1e-5))
                        .max(1e-5);
            }
        }
        self.prev2_l = flush_denormal_f64(self.prev_l);
        self.prev_l = flush_denormal_f64(l);
        self.prev2_r = flush_denormal_f64(self.prev_r);
        self.prev_r = flush_denormal_f64(r);
        (l, r)
    }

    /// 2. Dynamic <115 Hz Plosive Suppressor
    #[inline]
    fn duck_plosives(&mut self, l: f64, r: f64, strength: f64) -> (f64, f64) {
        let bands_l = self.plosive_split_l.process(l);
        let bands_r = self.plosive_split_r.process(r);

        let lo_mag = 0.5 * (bands_l.low.abs() + bands_r.low.abs());
        let hi_mag = 0.5 * (bands_l.detect_high.abs() + bands_r.detect_high.abs());
        self.low_env = flush_denormal_f64(0.985 * self.low_env + 0.015 * lo_mag);
        self.mid_env = flush_denormal_f64(0.985 * self.mid_env + 0.015 * hi_mag) + 1e-7;

        let plosive_ratio = self.low_env / self.mid_env;
        let low_gain = if plosive_ratio > 2.2 {
            let raw_g = (2.2 / plosive_ratio).clamp(0.12, 1.0);
            1.0 - strength * (1.0 - raw_g)
        } else {
            1.0
        };
        (bands_l.ducked(low_gain), bands_r.ducked(low_gain))
    }

    /// 3. Split-Band High-Frequency De-Esser around 7.2 kHz.
    ///
    /// The band-pass peaks at unity, so removing `1 - cut` of it leaves the band at a gain
    /// between `cut` and 1, never inverted.
    #[inline]
    fn deess(&mut self, mut l: f64, mut r: f64, threshold_db: f64) -> (f64, f64) {
        let sib_l = self.sib_bp_l.process_sample(l);
        let sib_r = self.sib_bp_r.process_sample(r);
        let sib_lvl = 0.5 * (sib_l.abs() + sib_r.abs());
        self.sib_env = flush_denormal_f64(0.95 * self.sib_env + 0.05 * sib_lvl);
        let sib_thresh = 10.0_f64.powf(threshold_db / 20.0);
        if self.sib_env > sib_thresh {
            let cut = (sib_thresh / self.sib_env).clamp(0.25, 1.0);
            l -= sib_l * (1.0 - cut);
            r -= sib_r * (1.0 - cut);
        }
        (l, r)
    }

    /// 4. Soft-Knee Breath & Room Gate
    #[inline]
    fn gate_breaths(&mut self, mut l: f64, mut r: f64, threshold_db: f64) -> (f64, f64) {
        let abs_lvl = 0.5 * (l.abs() + r.abs());
        self.gate_env = flush_denormal_f64(0.995 * self.gate_env + 0.005 * abs_lvl);
        let gate_thresh = 10.0_f64.powf(threshold_db / 20.0);
        if self.gate_env < gate_thresh {
            let ratio = (self.gate_env / gate_thresh.max(1e-9)).clamp(0.0, 1.0);
            let g = ratio * ratio;
            l *= g;
            r *= g;
        }
        (l, r)
    }
}

/// Time constant of the mono-cutoff glide, in milliseconds.
const MONO_CUTOFF_SMOOTHING_MS: f64 = 20.0;
/// Samples between side high-pass coefficient updates while the cutoff glides.
const MONO_CUTOFF_CONTROL_SAMPLES: u32 = 16;
/// Once the glide is this close to the target (Hz), the target is applied exactly.
const MONO_CUTOFF_SNAP_HZ: f64 = 0.01;
/// Length of the crossfade into and out of the tape branch, in milliseconds.
const TAPE_CROSSFADE_MS: f64 = 5.0;
/// Tape drive at or below which the tape branch fades out and stops running.
const TAPE_BYPASS_DRIVE: f64 = 0.001;

/// Real-Time Zero-Latency **Sub-Bass Mono-Maker, Allpass Phase Disperser & ChowTape Jiles-Atherton Bus**.
///
/// The mono cutoff glides to each new target and the side high-passes are retuned in place every
/// `MONO_CUTOFF_CONTROL_SAMPLES` samples while it moves, so their state survives a cutoff change.
/// The tape branch crossfades in and out over `TAPE_CROSSFADE_MS`, and starts from cleared state
/// each time it re-engages instead of from whatever it held when it was bypassed.
pub struct RealtimeSubAndTapeBus {
    sample_rate: f64,
    mono_cutoff_hz: f64,
    applied_cutoff_hz: f64,
    cutoff_smoother: ParamSmoother,
    cutoff_control_count: u32,
    side_hp1: Biquad,
    side_hp2: Biquad,
    sub_lp: Biquad,
    head_bump_l: CytomicSvf,
    head_bump_r: CytomicSvf,
    disperser_l: AllpassDisperser,
    disperser_r: AllpassDisperser,
    ja_hyst_l: JilesAthertonHysteresis,
    ja_hyst_r: JilesAthertonHysteresis,
    flutter_buf_l: [f64; 64],
    flutter_buf_r: [f64; 64],
    flutter_pos: usize,
    flutter_phase: f64,
    tape_mix: f64,
    tape_mix_step: f64,
    /// False until the first sample sets `tape_mix` from the drive, so a bus that starts with
    /// tape off is dry from its first sample and one that starts with tape on is fully wet.
    tape_mix_primed: bool,
    sub_phase_flip: f64,
    prev_sub_sample: f64,
}

impl RealtimeSubAndTapeBus {
    pub fn new(sample_rate: f64, mono_cutoff_hz: f64) -> Self {
        let sr = sample_rate.max(8000.0);
        let cutoff = mono_cutoff_hz.clamp(40.0, 350.0);
        Self {
            sample_rate: sr,
            mono_cutoff_hz: cutoff,
            applied_cutoff_hz: cutoff,
            cutoff_smoother: ParamSmoother::new(cutoff, sr, MONO_CUTOFF_SMOOTHING_MS),
            cutoff_control_count: 0,
            side_hp1: Biquad::highpass(sr, cutoff, FRAC_1_SQRT_2),
            side_hp2: Biquad::highpass(sr, cutoff, FRAC_1_SQRT_2),
            sub_lp: Biquad::lowpass(sr, 95.0, FRAC_1_SQRT_2),
            head_bump_l: CytomicSvf::bell(sr, 72.0, 1.15, 1.4),
            head_bump_r: CytomicSvf::bell(sr, 72.0, 1.15, 1.4),
            disperser_l: AllpassDisperser::default(),
            disperser_r: AllpassDisperser::default(),
            ja_hyst_l: JilesAthertonHysteresis::default(),
            ja_hyst_r: JilesAthertonHysteresis::default(),
            flutter_buf_l: [0.0; 64],
            flutter_buf_r: [0.0; 64],
            flutter_pos: 0,
            flutter_phase: 0.0,
            tape_mix: 0.0,
            tape_mix_step: 1.0 / (TAPE_CROSSFADE_MS * 0.001 * sr).max(1.0),
            tape_mix_primed: false,
            sub_phase_flip: 1.0,
            prev_sub_sample: 0.0,
        }
    }

    /// Clears all state in place for `sample_rate`, keeping the current mono cutoff. Holds only
    /// fixed-size state, so this never allocates.
    pub fn reset(&mut self, sample_rate: f64) {
        *self = Self::new(sample_rate, self.mono_cutoff_hz);
    }

    /// Sets the mono cutoff target. The side high-passes glide to it while processing, keeping
    /// their state, so calling this every block with an unchanged value costs nothing.
    pub fn set_mono_cutoff(&mut self, mono_cutoff_hz: f64) {
        let cutoff = if mono_cutoff_hz.is_finite() {
            mono_cutoff_hz.clamp(40.0, 350.0)
        } else {
            self.mono_cutoff_hz
        };
        self.mono_cutoff_hz = cutoff;
        self.cutoff_smoother.set_target(cutoff);
    }

    /// Advances the cutoff glide by one sample and retunes the side high-passes in place every
    /// `MONO_CUTOFF_CONTROL_SAMPLES` samples, only when the applied cutoff would change.
    #[inline]
    fn advance_mono_cutoff(&mut self) {
        let smoothed = self.cutoff_smoother.next_val();
        self.cutoff_control_count += 1;
        if self.cutoff_control_count < MONO_CUTOFF_CONTROL_SAMPLES {
            return;
        }
        self.cutoff_control_count = 0;
        let target = self.mono_cutoff_hz;
        let next = if (smoothed - target).abs() < MONO_CUTOFF_SNAP_HZ {
            target
        } else {
            smoothed
        };
        if (next - self.applied_cutoff_hz).abs() > f64::EPSILON {
            self.applied_cutoff_hz = next;
            let design = Biquad::highpass(self.sample_rate, next, FRAC_1_SQRT_2);
            self.side_hp1.set_coeffs(&design);
            self.side_hp2.set_coeffs(&design);
        }
    }

    #[inline]
    pub fn process_stereo_sample(
        &mut self,
        in_l: f32,
        in_r: f32,
        sub_reinforce: f64,
        stereo_width: f64,
        tape_drive: f64,
    ) -> (f32, f32) {
        let l = sanitize_sample_f32(in_l);
        let r = sanitize_sample_f32(in_r);
        let clean_sub = sanitize_param_f64(sub_reinforce, 0.25, 0.0, 1.0);
        let clean_width = sanitize_param_f64(stereo_width, 1.0, 0.0, 2.0);
        let amt = sanitize_param_f64(tape_drive, 0.2, 0.0, 1.0);
        self.advance_mono_cutoff();

        // 1. Mid/Side decomposition & 4th-order Linkwitz-Riley <cutoff Side high-pass (Sub Mono-Maker)
        let mid = 0.5 * (l + r);
        let side = 0.5 * (l - r);
        let clean_side = self
            .side_hp2
            .process_sample(self.side_hp1.process_sample(side))
            * clean_width;

        // 2. Sub-harmonic & 2nd-harmonic bass reinforcement + Allpass Phase Disperser (diopser punch)
        let reinforced_mid = mid + self.sub_octave(mid) * clean_sub * 0.35;
        let disp_amt = clean_sub * 0.45;
        let out_l = self.disperser_l.process_sample(
            reinforced_mid + clean_side,
            self.sample_rate,
            135.0,
            disp_amt,
        );
        let out_r = self.disperser_r.process_sample(
            reinforced_mid - clean_side,
            self.sample_rate,
            135.0,
            disp_amt,
        );

        // 3. ChowTape branch, crossfaded in and out around the bypass threshold
        let (out_l, out_r) = self.tape_crossfade(out_l, out_r, amt);
        (
            flush_denormal_f64(out_l) as f32,
            flush_denormal_f64(out_r) as f32,
        )
    }

    /// Sub-octave plus 2nd-harmonic reinforcement signal derived from the mid channel.
    #[inline]
    fn sub_octave(&mut self, mid: f64) -> f64 {
        let sub_sig = self.sub_lp.process_sample(mid);
        if self.prev_sub_sample <= 0.0 && sub_sig > 0.0 {
            self.sub_phase_flip = -self.sub_phase_flip;
        }
        self.prev_sub_sample = flush_denormal_f64(sub_sig);
        sub_sig.abs() * self.sub_phase_flip * 0.45 + (sub_sig * 1.6).tanh() * 0.55
    }

    /// Moves the tape wet mix one step toward on or off and blends the tape branch with the dry
    /// signal. The branch only runs while the mix is above zero, and its state is cleared when it
    /// re-engages from fully bypassed, so stale buffers never reach the output.
    #[inline]
    fn tape_crossfade(&mut self, dry_l: f64, dry_r: f64, amt: f64) -> (f64, f64) {
        let engaged = amt > TAPE_BYPASS_DRIVE;
        if !self.tape_mix_primed {
            // The first sample picks the starting side; fades apply only to later transitions.
            self.tape_mix_primed = true;
            self.tape_mix = if engaged { 1.0 } else { 0.0 };
        }
        if !engaged && self.tape_mix <= 0.0 {
            return (dry_l, dry_r);
        }
        if engaged && self.tape_mix <= 0.0 {
            self.clear_tape_state();
        }
        let step = if engaged {
            self.tape_mix_step
        } else {
            -self.tape_mix_step
        };
        self.tape_mix = (self.tape_mix + step).clamp(0.0, 1.0);
        let (wet_l, wet_r) = self.process_tape(dry_l, dry_r, amt);
        (
            dry_l + self.tape_mix * (wet_l - dry_l),
            dry_r + self.tape_mix * (wet_r - dry_r),
        )
    }

    /// Clears the head bump, hysteresis and flutter state of the tape branch.
    fn clear_tape_state(&mut self) {
        self.head_bump_l.reset();
        self.head_bump_r.reset();
        self.ja_hyst_l = JilesAthertonHysteresis::default();
        self.ja_hyst_r = JilesAthertonHysteresis::default();
        self.flutter_buf_l = [0.0; 64];
        self.flutter_buf_r = [0.0; 64];
        self.flutter_pos = 0;
        self.flutter_phase = 0.0;
    }

    /// ChowTapeModel Jiles-Atherton RK4 Ferromagnetic Hysteresis + 72Hz TPT Head Bump + Tape Flutter.
    #[inline]
    fn process_tape(&mut self, in_l: f64, in_r: f64, amt: f64) -> (f64, f64) {
        let xl = self.head_bump_l.process_sample(in_l);
        let xr = self.head_bump_r.process_sample(in_r);

        let hyst_l = self.ja_hyst_l.process(xl, amt, 0.45 + 0.45 * amt);
        let hyst_r = self.ja_hyst_r.process(xr, amt, 0.45 + 0.45 * amt);

        // Subtle 5.8 Hz tape transport flutter fractional delay line
        let w_pos = self.flutter_pos & 63;
        self.flutter_buf_l[w_pos] = hyst_l;
        self.flutter_buf_r[w_pos] = hyst_r;
        self.flutter_phase = (self.flutter_phase + 5.8 / self.sample_rate).fract();
        let delay_samps = 2.0 + 1.2 * amt * (2.0 * PI * self.flutter_phase).sin();
        let d_int = delay_samps.floor() as usize;
        let d_frac = delay_samps - (d_int as f64);
        let r0 = (w_pos + 64 - d_int) & 63;
        let r1 = (w_pos + 64 - (d_int + 1)) & 63;
        let fl_l = self.flutter_buf_l[r0] * (1.0 - d_frac) + self.flutter_buf_l[r1] * d_frac;
        let fl_r = self.flutter_buf_r[r0] * (1.0 - d_frac) + self.flutter_buf_r[r1] * d_frac;
        self.flutter_pos = self.flutter_pos.wrapping_add(1);

        (
            (1.0 - amt * 0.60) * xl + (amt * 0.60) * fl_l,
            (1.0 - amt * 0.60) * xr + (amt * 0.60) * fl_r,
        )
    }
}

/// Real-Time **Lookahead True-Peak Brickwall Limiter & ITU-R BS.1770-4 Loudness Meter**.
///
/// The gain path has three stages, all on pre-allocated state:
///
/// 1. A 4x polyphase true-peak detector (the ITU-R BS.1770-4 Annex 2 FIR that
///    `compute_true_peak` uses) turns each input frame into a required gain.
/// 2. A sliding-window minimum holds that gain for `lookahead + HOLD_AFTER` frames, followed by
///    a release one-pole that only acts upward.
/// 3. A moving average of `lookahead - RAMP_MARGIN` frames turns the held gain into a linear
///    attack ramp.
///
/// The ramp finishes before a detected peak leaves the `lookahead` delay line, and the hold keeps
/// the gain flat across the interpolation filter's neighbourhood, so the output's true peak stays
/// under the ceiling without the final clamp engaging.
/// Inter-sample true-peak detector oversampling mode for [`RealtimeMasterLimiter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IspOversampling {
    /// Standard ITU-R BS.1770-4 Annex 2 4x 48-tap polyphase FIR.
    #[default]
    X4,
    /// Ultra-precision 8x oversampling (4x BS.1770-4 polyphase + cubic half-phase midpoints).
    X8,
}

pub struct RealtimeMasterLimiter {
    sample_rate: f64,
    lookahead_samples: usize,
    delay_l: Vec<f32>,
    delay_r: Vec<f32>,
    write_pos: usize,
    envelope_gain: f64,
    k_shelf_l: Biquad,
    k_r1b_l: Biquad,
    k_shelf_r: Biquad,
    k_r1b_r: Biquad,
    momentary_energy: f64,
    last_lufs: f32,
    last_true_peak_dbtp: f32,
    last_gr_db: f32,
    tp_hist_l: [f64; TP_TAPS],
    tp_hist_r: [f64; TP_TAPS],
    isp_oversampling: IspOversampling,
    hold_l: SlidingMin,
    hold_r: SlidingMin,
    released_gain_l: f64,
    released_gain_r: f64,
    ramp_l: Vec<f64>,
    ramp_r: Vec<f64>,
    ramp_len: usize,
    ramp_pos: usize,
    ramp_sum_l: f64,
    ramp_sum_r: f64,
    last_target_lin: f64,
    last_release_coeff: f64,
    style: LimiterStyle,
    channel_link: f64,
    dither_mode: DitherMode,
    dither_bits: u16,
    dither_rng_l: u64,
    dither_rng_r: u64,
    dither_e1_l: f64,
    dither_e2_l: f64,
    dither_e1_r: f64,
    dither_e2_r: f64,
}

/// Taps per phase of the true-peak interpolation filter.
const TP_TAPS: usize = 12;

/// Sliding-window minimum over the last `window` values, as a monotonic deque in fixed-size
/// ring buffers. `push` is amortized O(1) and never allocates.
struct SlidingMin {
    vals: Vec<f64>,
    idx: Vec<u64>,
    head: usize,
    len: usize,
    window: u64,
    next: u64,
}

impl SlidingMin {
    fn new(capacity: usize) -> Self {
        Self {
            vals: vec![1.0; capacity],
            idx: vec![0; capacity],
            head: 0,
            len: 0,
            window: 1,
            next: 0,
        }
    }

    fn reset(&mut self, window: usize) {
        self.head = 0;
        self.len = 0;
        self.window = window.clamp(1, self.vals.len()) as u64;
        self.next = 0;
    }

    /// Adds `v` and returns the minimum of the last `window` values pushed.
    fn push(&mut self, v: f64) -> f64 {
        let cap = self.vals.len();
        // Expire before inserting: at most `window - 1` entries survive, so the insert below
        // can never overwrite the head even when `window == cap`.
        while self.len > 0 && self.idx[self.head] + self.window <= self.next {
            self.head = (self.head + 1) % cap;
            self.len -= 1;
        }
        while self.len > 0 && self.vals[(self.head + self.len - 1) % cap] >= v {
            self.len -= 1;
        }
        let slot = (self.head + self.len) % cap;
        self.vals[slot] = v;
        self.idx[slot] = self.next;
        self.len += 1;
        self.next += 1;
        self.vals[self.head]
    }
}

impl RealtimeMasterLimiter {
    /// Upper bound of the lookahead delay. The delay lines are allocated at this size once, so
    /// `reset` can change sample rate without allocating.
    const MAX_LOOKAHEAD_SAMPLES: usize = 512;

    /// The detector sees a peak up to `TP_TAPS` frames after the earliest sample whose true peak
    /// it affects, so the attack ramp must end that many frames before the delay line's exit.
    const RAMP_MARGIN: usize = TP_TAPS;

    /// Frames past the delay line's exit that the gain stays held, covering the later half of the
    /// interpolation filter around the peak.
    const HOLD_AFTER: usize = TP_TAPS / 2 + 1;

    /// Gain is computed against a ceiling this many dB below the requested one, so a fully
    /// limited sample lands just under the final clamp rather than on it.
    const CEILING_MARGIN_DB: f64 = 0.05;

    fn lookahead_for(sample_rate: f64) -> usize {
        ((sample_rate * LIMITER_LOOKAHEAD_MS / 1000.0).round() as usize)
            .clamp(32, Self::MAX_LOOKAHEAD_SAMPLES)
    }

    pub fn new(sample_rate: f64) -> Self {
        let sr = sample_rate.max(8000.0);
        let mut limiter = Self {
            sample_rate: sr,
            lookahead_samples: Self::lookahead_for(sr),
            delay_l: vec![0.0; Self::MAX_LOOKAHEAD_SAMPLES],
            delay_r: vec![0.0; Self::MAX_LOOKAHEAD_SAMPLES],
            write_pos: 0,
            envelope_gain: 1.0,
            k_shelf_l: Biquad::k_weight_pre_shelf(sr),
            k_r1b_l: Biquad::k_weight_rlb_highpass(sr),
            k_shelf_r: Biquad::k_weight_pre_shelf(sr),
            k_r1b_r: Biquad::k_weight_rlb_highpass(sr),
            momentary_energy: 1e-9,
            last_lufs: -70.0,
            last_true_peak_dbtp: -120.0,
            last_gr_db: 0.0,
            tp_hist_l: [0.0; TP_TAPS],
            tp_hist_r: [0.0; TP_TAPS],
            isp_oversampling: IspOversampling::X4,
            hold_l: SlidingMin::new(Self::MAX_LOOKAHEAD_SAMPLES + Self::HOLD_AFTER),
            hold_r: SlidingMin::new(Self::MAX_LOOKAHEAD_SAMPLES + Self::HOLD_AFTER),
            released_gain_l: 1.0,
            released_gain_r: 1.0,
            ramp_l: vec![1.0; Self::MAX_LOOKAHEAD_SAMPLES],
            ramp_r: vec![1.0; Self::MAX_LOOKAHEAD_SAMPLES],
            ramp_len: 1,
            ramp_pos: 0,
            ramp_sum_l: 1.0,
            ramp_sum_r: 1.0,
            last_target_lin: 1.0,
            last_release_coeff: 0.0,
            style: LimiterStyle::Transparent,
            channel_link: 1.0,
            dither_mode: DitherMode::None,
            dither_bits: 24,
            dither_rng_l: 0x9E37_79B9_7F4A_7C15,
            dither_rng_r: 0x2545_F491_4F6C_DD1D,
            dither_e1_l: 0.0,
            dither_e2_l: 0.0,
            dither_e1_r: 0.0,
            dither_e2_r: 0.0,
        };
        limiter.reset(sr);
        limiter
    }

    pub fn latency_samples(&self) -> u32 {
        self.lookahead_samples as u32
    }

    /// Clears all state in place for `sample_rate`. Safe to call on the audio thread: it never
    /// allocates.
    pub fn reset(&mut self, sample_rate: f64) {
        let sr = sample_rate.max(8000.0);
        self.sample_rate = sr;
        self.lookahead_samples = Self::lookahead_for(sr);
        self.delay_l.fill(0.0);
        self.delay_r.fill(0.0);
        self.write_pos = 0;
        self.envelope_gain = 1.0;
        self.k_shelf_l = Biquad::k_weight_pre_shelf(sr);
        self.k_r1b_l = Biquad::k_weight_rlb_highpass(sr);
        self.k_shelf_r = Biquad::k_weight_pre_shelf(sr);
        self.k_r1b_r = Biquad::k_weight_rlb_highpass(sr);
        self.momentary_energy = 1e-9;
        self.last_lufs = -70.0;
        self.last_true_peak_dbtp = -120.0;
        self.last_gr_db = 0.0;
        self.tp_hist_l = [0.0; TP_TAPS];
        self.tp_hist_r = [0.0; TP_TAPS];
        self.hold_l.reset(self.lookahead_samples + Self::HOLD_AFTER);
        self.hold_r.reset(self.lookahead_samples + Self::HOLD_AFTER);
        self.released_gain_l = 1.0;
        self.released_gain_r = 1.0;
        self.ramp_len = self.lookahead_samples - Self::RAMP_MARGIN;
        self.ramp_l.fill(1.0);
        self.ramp_r.fill(1.0);
        self.ramp_pos = 0;
        self.ramp_sum_l = self.ramp_len as f64;
        self.ramp_sum_r = self.ramp_len as f64;
        self.last_target_lin = 10.0_f64.powf((-1.0 - Self::CEILING_MARGIN_DB) / 20.0);
        self.last_release_coeff = Self::release_coeff(sr, 85.0);
        self.dither_rng_l = 0x9E37_79B9_7F4A_7C15;
        self.dither_rng_r = 0x2545_F491_4F6C_DD1D;
        self.dither_e1_l = 0.0;
        self.dither_e2_l = 0.0;
        self.dither_e1_r = 0.0;
        self.dither_e2_r = 0.0;
    }

    /// Passes a frame through the lookahead delay without limiting, so a bypassed limiter keeps
    /// the same latency as an active one and the host's delay compensation stays correct.
    ///
    /// The gain path still analyses the frame, with the last active ceiling and release, so the
    /// frames already in the delay line are covered when limiting resumes.
    #[inline]
    pub fn process_delay_only(&mut self, in_l: f32, in_r: f32) -> (f32, f32) {
        let xl = sanitize_sample_f32(in_l);
        let xr = sanitize_sample_f32(in_r);
        self.advance_gain(xl, xr, self.last_target_lin, self.last_release_coeff);
        self.envelope_gain = 1.0;
        self.last_gr_db = 0.0;
        self.swap_delay(xl as f32, xr as f32)
    }

    pub fn telemetry(&self) -> (f32, f32, f32) {
        (self.last_lufs, self.last_true_peak_dbtp, self.last_gr_db)
    }

    /// Sets the mastering release topology (`Transparent`, `Punchy`, `Dynamic`, `Aggressive`). Zero-alloc.
    #[inline]
    pub fn set_style(&mut self, style: LimiterStyle) {
        self.style = style;
    }

    /// Sets stereo channel linking (`0.0..=1.0`). Zero-alloc.
    #[inline]
    pub fn set_channel_link(&mut self, channel_link: f64) {
        self.channel_link = sanitize_param_f64(channel_link, 1.0, 0.0, 1.0);
    }

    /// Sets real-time output dithering / noise shaping (`8..=24` bits). Zero-alloc.
    #[inline]
    pub fn set_dither(&mut self, mode: DitherMode, bits: u16) {
        self.dither_mode = mode;
        self.dither_bits = bits.clamp(8, 24);
    }

    /// Selects 4x (ITU-R BS.1770-4) or 8x (ultra-precision) inter-sample peak oversampling.
    /// Zero-alloc.
    #[inline]
    pub fn set_isp_oversampling(&mut self, mode: IspOversampling) {
        self.isp_oversampling = mode;
    }

    #[inline]
    pub const fn isp_oversampling(&self) -> IspOversampling {
        self.isp_oversampling
    }

    #[inline]
    fn next_dither_bipolar(state: &mut u64) -> f64 {
        let mut x = *state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        *state = x;
        let bits = x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        (bits as f64) * (1.0 / ((1_u64 << 53) as f64)) - 0.5
    }

    #[inline]
    fn apply_rt_dither(
        mode: DitherMode,
        bits: u16,
        x: f64,
        rng: &mut u64,
        e1: &mut f64,
        e2: &mut f64,
    ) -> f64 {
        if mode == DitherMode::None {
            return x;
        }
        let full_scale = (1_u64 << (bits - 1)) as f64;
        let max_code = full_scale - 1.0;
        let min_code = -full_scale;
        let scaled = x.clamp(-1.0, 1.0) * full_scale;
        let tpdf = Self::next_dither_bipolar(rng) + Self::next_dither_bipolar(rng);
        let shaped = match mode {
            DitherMode::None | DitherMode::Tpdf => scaled,
            DitherMode::NoiseShapedHighPass => scaled + 1.623 * (*e1) - 0.692 * (*e2),
        };
        let q = (shaped + tpdf).round().clamp(min_code, max_code);
        let err = flush_denormal_f64(shaped - q);
        *e2 = *e1;
        *e1 = err;
        q / full_scale
    }

    fn release_coeff(sample_rate: f64, release_ms: f64) -> f64 {
        (-1.0 / ((release_ms * 0.001) * sample_rate)).exp()
    }

    /// Pushes `x` into the newest-first history and returns the largest of `|x|` and the
    /// interpolated sub-samples (4x BS.1770-4 polyphase FIR, or 8x with Catmull-Rom midpoints).
    #[inline]
    fn estimate_intersample_peak(hist: &mut [f64; TP_TAPS], x: f64, mode: IspOversampling) -> f64 {
        hist.copy_within(0..TP_TAPS - 1, 1);
        hist[0] = x;
        let mut peak = x.abs();
        let mut phases = [0.0_f64; 4];
        for (slot, coeffs) in phases.iter_mut().zip(&TRUE_PEAK_PHASES) {
            let interp: f64 = coeffs.iter().zip(hist.iter()).map(|(c, h)| c * h).sum();
            *slot = interp;
            peak = peak.max(interp.abs());
        }
        if mode == IspOversampling::X8 {
            // Evaluate strictly contiguous half-phase midpoints within the 4x polyphase window
            // (all four `phases` share the exact same group delay).
            let m01 = 0.5 * (phases[0] + phases[1]);
            let m12 = 0.5625 * (phases[1] + phases[2]) - 0.0625 * (phases[0] + phases[3]);
            let m23 = 0.5 * (phases[2] + phases[3]);
            peak = peak.max(m01.abs()).max(m12.abs()).max(m23.abs());
        }
        peak
    }

    /// Runs one frame through the per-channel detector, hold, and attack ramp honoring
    /// `channel_link` (`0.0` = dual-mono, `1.0` = linked stereo). Returns `(max_tp, gain_l, gain_r)`.
    #[inline]
    fn advance_gain(
        &mut self,
        xl: f64,
        xr: f64,
        target_lin: f64,
        release_coeff: f64,
    ) -> (f64, f64, f64) {
        let mode = self.isp_oversampling;
        let tp_l = Self::estimate_intersample_peak(&mut self.tp_hist_l, xl, mode).max(1e-12);
        let tp_r = Self::estimate_intersample_peak(&mut self.tp_hist_r, xr, mode).max(1e-12);
        let tp_max = tp_l.max(tp_r);
        let link = self.channel_link;
        let eff_tp_l = link * tp_max + (1.0 - link) * tp_l;
        let eff_tp_r = link * tp_max + (1.0 - link) * tp_r;

        let req_l = if eff_tp_l > target_lin {
            target_lin / eff_tp_l
        } else {
            1.0
        };
        let req_r = if eff_tp_r > target_lin {
            target_lin / eff_tp_r
        } else {
            1.0
        };

        let held_l = self.hold_l.push(req_l);
        let held_r = self.hold_r.push(req_r);
        self.released_gain_l = if held_l < self.released_gain_l {
            held_l
        } else {
            flush_denormal_f64(
                release_coeff * self.released_gain_l + (1.0 - release_coeff) * held_l,
            )
            .min(held_l)
        };
        self.released_gain_r = if held_r < self.released_gain_r {
            held_r
        } else {
            flush_denormal_f64(
                release_coeff * self.released_gain_r + (1.0 - release_coeff) * held_r,
            )
            .min(held_r)
        };

        self.ramp_sum_l += self.released_gain_l - self.ramp_l[self.ramp_pos];
        self.ramp_sum_r += self.released_gain_r - self.ramp_r[self.ramp_pos];
        self.ramp_l[self.ramp_pos] = self.released_gain_l;
        self.ramp_r[self.ramp_pos] = self.released_gain_r;
        self.ramp_pos += 1;
        if self.ramp_pos == self.ramp_len {
            self.ramp_pos = 0;
            self.ramp_sum_l = self.ramp_l[..self.ramp_len].iter().sum();
            self.ramp_sum_r = self.ramp_r[..self.ramp_len].iter().sum();
        }
        let inv_ramp = 1.0 / (self.ramp_len as f64);
        let gain_l = (self.ramp_sum_l * inv_ramp).clamp(1e-6, 1.0);
        let gain_r = (self.ramp_sum_r * inv_ramp).clamp(1e-6, 1.0);
        ((tp_l * gain_l).max(tp_r * gain_r), gain_l, gain_r)
    }

    #[inline]
    fn swap_delay(&mut self, xl: f32, xr: f32) -> (f32, f32) {
        let delayed_l = self.delay_l[self.write_pos];
        let delayed_r = self.delay_r[self.write_pos];
        self.delay_l[self.write_pos] = xl;
        self.delay_r[self.write_pos] = xr;
        self.write_pos = (self.write_pos + 1) % self.lookahead_samples;
        (delayed_l, delayed_r)
    }

    /// ITU-R BS.1770-4 K-weighted momentary loudness tracker (400 ms time constant), plus the
    /// telemetry the editor reads.
    #[inline]
    fn update_meter(&mut self, out_l: f64, out_r: f64, out_tp_est: f64, ceiling_lin: f64) {
        let kw_l = self
            .k_r1b_l
            .process_sample(self.k_shelf_l.process_sample(out_l));
        let kw_r = self
            .k_r1b_r
            .process_sample(self.k_shelf_r.process_sample(out_r));
        let sq_sum = kw_l * kw_l + kw_r * kw_r;
        let alpha_400ms = (-1.0 / (0.400 * self.sample_rate)).exp();
        self.momentary_energy =
            flush_denormal_f64(alpha_400ms * self.momentary_energy + (1.0 - alpha_400ms) * sq_sum)
                .max(1e-12);

        self.last_lufs = (-0.691 + 10.0 * self.momentary_energy.log10()) as f32;
        self.last_true_peak_dbtp = (20.0 * out_tp_est.min(ceiling_lin).max(1e-9).log10()) as f32;
        self.last_gr_db = (-20.0 * self.envelope_gain.max(1e-6).log10()) as f32;
    }

    #[inline]
    pub fn process_stereo_sample(
        &mut self,
        in_l: f32,
        in_r: f32,
        input_drive_db: f64,
        ceiling_dbtp: f64,
        release_ms: f64,
    ) -> (f32, f32) {
        let clean_drive_db = sanitize_param_f64(input_drive_db, 0.0, -24.0, 24.0);
        let clean_ceil_db = sanitize_param_f64(ceiling_dbtp, -1.0, -12.0, -0.05);
        let clean_rel_ms = sanitize_param_f64(release_ms, 85.0, 10.0, 1000.0);
        let drive = 10.0_f64.powf(clean_drive_db / 20.0);
        let ceiling_lin = 10.0_f64.powf(clean_ceil_db / 20.0);
        let base_target_lin = 10.0_f64.powf((clean_ceil_db - Self::CEILING_MARGIN_DB) / 20.0);
        // Reserve headroom for TPDF/noise-shaped dither peak excursion (including 4x intersample
        // overshoot at low bit depths such as 8-bit) so dithered output stays under `ceiling_lin`.
        let target_lin = if self.dither_mode != DitherMode::None {
            let lsb = 1.0 / ((1_u64 << (self.dither_bits - 1)) as f64);
            let reserve = match self.dither_mode {
                DitherMode::None => 0.0,
                DitherMode::Tpdf => 3.5 * lsb,
                DitherMode::NoiseShapedHighPass => 6.0 * lsb,
            };
            (base_target_lin - reserve).max(1e-6)
        } else {
            base_target_lin
        };
        let style_rel_mul = match self.style {
            LimiterStyle::Transparent => 1.0,
            LimiterStyle::Punchy => 0.65,
            LimiterStyle::Dynamic => 1.45,
            LimiterStyle::Aggressive => 0.45,
        };
        let eff_rel_ms = (clean_rel_ms * style_rel_mul).clamp(10.0, 1000.0);
        let release_coeff = Self::release_coeff(self.sample_rate, eff_rel_ms);
        self.last_target_lin = target_lin;
        self.last_release_coeff = release_coeff;

        let dl = sanitize_sample_f32(in_l) * drive;
        let dr = sanitize_sample_f32(in_r) * drive;
        let (tp, gain_l, gain_r) = self.advance_gain(dl, dr, target_lin, release_coeff);
        self.envelope_gain = gain_l.min(gain_r);
        let (delayed_l, delayed_r) = self.swap_delay(dl as f32, dr as f32);

        // Safety net only: the ramp has already brought every detected peak under `target_lin`.
        let mut out_l = (f64::from(delayed_l) * gain_l).clamp(-ceiling_lin, ceiling_lin);
        let mut out_r = (f64::from(delayed_r) * gain_r).clamp(-ceiling_lin, ceiling_lin);
        if self.dither_mode != DitherMode::None {
            out_l = Self::apply_rt_dither(
                self.dither_mode,
                self.dither_bits,
                out_l,
                &mut self.dither_rng_l,
                &mut self.dither_e1_l,
                &mut self.dither_e2_l,
            )
            .clamp(-ceiling_lin, ceiling_lin);
            out_r = Self::apply_rt_dither(
                self.dither_mode,
                self.dither_bits,
                out_r,
                &mut self.dither_rng_r,
                &mut self.dither_e1_r,
                &mut self.dither_e2_r,
            )
            .clamp(-ceiling_lin, ceiling_lin);
        }
        self.update_meter(out_l, out_r, tp, ceiling_lin);

        (
            flush_denormal_f64(out_l) as f32,
            flush_denormal_f64(out_r) as f32,
        )
    }
}

/// Time constant of the 18-band gain glide, in milliseconds.
const EQ18_GAIN_SMOOTHING_MS: f64 = 20.0;
/// Samples between band coefficient updates while the gains glide.
const EQ18_CONTROL_SAMPLES: u32 = 32;
/// Once a band's glide is this close to its target (dB), the target is applied exactly.
const EQ18_GAIN_SNAP_DB: f64 = 0.001;

/// Real-Time **18-Band Mastering & Reference Match EQ** Bank.
///
/// `set_gains` only sets targets. While processing, each band glides toward its target and is
/// retuned in place every `EQ18_CONTROL_SAMPLES` samples, keeping its filter state, so automating
/// the amount or switching genre does not click. Bands already at their target are not recomputed.
pub struct RealtimeMasteringEq18 {
    sample_rate: f64,
    bands_l: [Biquad; 18],
    bands_r: [Biquad; 18],
    current_gains_db: [f64; 18],
    target_gains_db: [f64; 18],
    glide_coeff: f64,
    control_count: u32,
    gliding: bool,
}

impl RealtimeMasteringEq18 {
    pub fn new(sample_rate: f64) -> Self {
        let sr = sanitize_param_f64(sample_rate, 48_000.0, 8_000.0, 768_000.0);
        let bands_l =
            std::array::from_fn(|i| Biquad::peaking_eq(sr, MASTERING_18_BANDS_HZ[i], 1.4, 0.0));
        let bands_r =
            std::array::from_fn(|i| Biquad::peaking_eq(sr, MASTERING_18_BANDS_HZ[i], 1.4, 0.0));
        let tick_s = f64::from(EQ18_CONTROL_SAMPLES) / sr;
        Self {
            sample_rate: sr,
            bands_l,
            bands_r,
            current_gains_db: [0.0; 18],
            target_gains_db: [0.0; 18],
            glide_coeff: 1.0 - (-tick_s / (EQ18_GAIN_SMOOTHING_MS * 0.001)).exp(),
            control_count: 0,
            gliding: false,
        }
    }

    /// Sets each band's target gain to `gains_db[i] * amount`, clamped to +/-15 dB.
    pub fn set_gains(&mut self, gains_db: &[f64; 18], amount: f64) {
        let amt = sanitize_param_f64(amount, 1.0, 0.0, 2.0);
        for (target, &g_in) in self.target_gains_db.iter_mut().zip(gains_db) {
            let g = if g_in.is_finite() { g_in } else { 0.0 };
            *target = (g * amt).clamp(-15.0, 15.0);
        }
        self.gliding = self.target_gains_db != self.current_gains_db;
    }

    /// Moves every band one control step toward its target and retunes the bands that moved.
    fn glide_gains(&mut self) {
        let mut still_gliding = false;
        for (i, &band_hz) in MASTERING_18_BANDS_HZ.iter().enumerate() {
            let diff = self.target_gains_db[i] - self.current_gains_db[i];
            if diff == 0.0 {
                continue;
            }
            let g = if diff.abs() < EQ18_GAIN_SNAP_DB {
                self.target_gains_db[i]
            } else {
                still_gliding = true;
                self.current_gains_db[i] + self.glide_coeff * diff
            };
            self.current_gains_db[i] = g;
            let design = Biquad::peaking_eq(self.sample_rate, band_hz, 1.4, g);
            self.bands_l[i].set_coeffs(&design);
            self.bands_r[i].set_coeffs(&design);
        }
        self.gliding = still_gliding;
    }

    #[inline]
    pub fn process_stereo_sample(&mut self, in_l: f32, in_r: f32) -> (f32, f32) {
        if self.gliding {
            self.control_count += 1;
            if self.control_count >= EQ18_CONTROL_SAMPLES {
                self.control_count = 0;
                self.glide_gains();
            }
        }
        let mut l = sanitize_sample_f32(in_l);
        let mut r = sanitize_sample_f32(in_r);
        for i in 0..18 {
            l = self.bands_l[i].process_sample(l);
            r = self.bands_r[i].process_sample(r);
        }
        (flush_denormal_f64(l) as f32, flush_denormal_f64(r) as f32)
    }
}

/// Real-Time **Sub-Sample Fractional Delay & Polarity Aligner** (Sound Radix Auto-Align equivalent).
///
/// Captures a 16384-sample window from Main + Sidechain Reference on demand, computes the
/// GCC-PHAT delay and polarity inversion, and applies sample-accurate delay compensation in real time.
pub struct RealtimePhaseAligner {
    max_delay_samples: usize,
    delay_line_l: Vec<f32>,
    delay_line_r: Vec<f32>,
    write_pos: usize,
    current_delay_samples: usize,
    invert_polarity: bool,
}

impl RealtimePhaseAligner {
    pub fn new(sample_rate: f64) -> Self {
        let sr = sanitize_param_f64(sample_rate, 48_000.0, 8_000.0, 768_000.0);
        let max_delay_samples = ((sr * 0.050).round() as usize).clamp(512, 8192);
        Self {
            max_delay_samples,
            delay_line_l: vec![0.0; max_delay_samples],
            delay_line_r: vec![0.0; max_delay_samples],
            write_pos: 0,
            current_delay_samples: 0,
            invert_polarity: false,
        }
    }

    pub fn set_alignment(&mut self, delay_samples: usize, invert_polarity: bool) {
        self.current_delay_samples = delay_samples.min(self.max_delay_samples - 1);
        self.invert_polarity = invert_polarity;
    }

    #[inline]
    pub fn process_stereo_sample(&mut self, in_l: f32, in_r: f32) -> (f32, f32) {
        let clean_l = sanitize_sample_f32(in_l) as f32;
        let clean_r = sanitize_sample_f32(in_r) as f32;
        self.delay_line_l[self.write_pos] = clean_l;
        self.delay_line_r[self.write_pos] = clean_r;

        let read_pos = (self.write_pos + self.max_delay_samples - self.current_delay_samples)
            % self.max_delay_samples;
        self.write_pos = (self.write_pos + 1) % self.max_delay_samples;

        let sign = if self.invert_polarity {
            -1.0_f32
        } else {
            1.0_f32
        };
        (
            self.delay_line_l[read_pos] * sign,
            self.delay_line_r[read_pos] * sign,
        )
    }
}

/// Fourth-order 180 Hz to 2.5 kHz snare band: two Butterworth high-passes then two low-passes.
///
/// A true band-pass with unity passband, like the kick low-pass and hat high-pass. The old
/// `peaking_eq - 0.5 * input` left half the broadband input in the band, so kick and hat hits
/// fired the snare too.
fn snare_band_pass(sr: f64) -> [Biquad; 4] {
    [
        Biquad::highpass(sr, 180.0, FRAC_1_SQRT_2),
        Biquad::highpass(sr, 180.0, FRAC_1_SQRT_2),
        Biquad::lowpass(sr, 2500.0, FRAC_1_SQRT_2),
        Biquad::lowpass(sr, 2500.0, FRAC_1_SQRT_2),
    ]
}

/// Real-Time **3-Band Drum Transient-to-MIDI Trigger Detector** (Slate Trigger 2 equivalent).
///
/// Separates incoming audio into Kick (`<115 Hz`), Snare (`180 Hz–2.5 kHz`), and Hi-Hat (`>5.5 kHz`)
/// bands and emits `(midi_note, velocity_0_to_1)` triggers with refractory holdoff to prevent double-triggering.
pub struct RealtimeDrumTrigger {
    kick_lp: Biquad,
    snare_band: [Biquad; 4],
    hat_hp: Biquad,
    kick: BandEnvelope,
    snare: BandEnvelope,
    hat: BandEnvelope,
    min_holdoff_samples: usize,
}

/// Envelope smoothing, trigger thresholds and output of one drum band.
struct BandTuning {
    /// `(keep, add)` weights of the fast and slow envelope followers.
    fast: (f64, f64),
    slow: (f64, f64),
    /// Scale on the threshold amplitude the fast envelope must exceed.
    amp_scale: f64,
    /// How far the fast envelope must rise above the slow one.
    ratio: f64,
    /// `(scale, floor)` mapping the fast envelope to a velocity.
    velocity: (f64, f64),
    /// Divisor on the refractory holdoff.
    holdoff_div: usize,
    note: u8,
}

/// C1 kick.
const KICK_TUNING: BandTuning = BandTuning {
    fast: (0.85, 0.15),
    slow: (0.996, 0.004),
    amp_scale: 1.0,
    ratio: 2.8,
    velocity: (2.2, 0.15),
    holdoff_div: 1,
    note: 36,
};
/// D1 snare.
const SNARE_TUNING: BandTuning = BandTuning {
    fast: (0.85, 0.15),
    slow: (0.996, 0.004),
    amp_scale: 1.0,
    ratio: 3.0,
    velocity: (2.4, 0.15),
    holdoff_div: 1,
    note: 38,
};
/// F#1 closed hat: half the threshold and half the holdoff.
const HAT_TUNING: BandTuning = BandTuning {
    fast: (0.80, 0.20),
    slow: (0.995, 0.005),
    amp_scale: 0.5,
    ratio: 2.9,
    velocity: (3.0, 0.12),
    holdoff_div: 2,
    note: 42,
};

/// One band's fast and slow envelopes and refractory countdown.
#[derive(Clone, Copy)]
struct BandEnvelope {
    fast: f64,
    slow: f64,
    cooldown: usize,
}

impl BandEnvelope {
    const fn new() -> Self {
        Self {
            fast: 0.0,
            slow: 1e-5,
            cooldown: 0,
        }
    }

    /// Advances the band by one sample of rectified `level` and returns its trigger, if any.
    #[inline]
    fn process(
        &mut self,
        level: f64,
        min_amp: f64,
        holdoff: usize,
        tuning: &BandTuning,
    ) -> Option<(u8, f32)> {
        let (fast_keep, fast_add) = tuning.fast;
        let (slow_keep, slow_add) = tuning.slow;
        self.fast = flush_denormal_f64(fast_keep * self.fast + fast_add * level);
        self.slow = flush_denormal_f64(slow_keep * self.slow + slow_add * level) + 1e-6;
        self.cooldown = self.cooldown.saturating_sub(1);
        let fires = self.cooldown == 0
            && self.fast > min_amp * tuning.amp_scale
            && self.fast > self.slow * tuning.ratio;
        if !fires {
            return None;
        }
        self.cooldown = holdoff / tuning.holdoff_div;
        let (scale, floor) = tuning.velocity;
        Some((tuning.note, (self.fast * scale).clamp(floor, 1.0) as f32))
    }
}

impl RealtimeDrumTrigger {
    pub fn new(sample_rate: f64) -> Self {
        let sr = sanitize_param_f64(sample_rate, 48_000.0, 8_000.0, 768_000.0);
        Self {
            kick_lp: Biquad::lowpass(sr, 115.0, FRAC_1_SQRT_2),
            snare_band: snare_band_pass(sr),
            hat_hp: Biquad::highpass(sr, 5500.0, FRAC_1_SQRT_2),
            kick: BandEnvelope::new(),
            snare: BandEnvelope::new(),
            hat: BandEnvelope::new(),
            min_holdoff_samples: ((sr * 0.045).round() as usize).max(256), // 45ms refractory window
        }
    }

    pub fn reset(&mut self, sample_rate: f64) {
        *self = Self::new(sample_rate);
    }

    /// Processes a stereo frame and returns up to 3 triggered MIDI notes `(midi_note, velocity_0_to_1)`.
    #[inline]
    pub fn process_stereo_sample(
        &mut self,
        in_l: f32,
        in_r: f32,
        threshold_db: f64,
    ) -> [Option<(u8, f32)>; 3] {
        let mono = 0.5 * (sanitize_sample_f32(in_l) + sanitize_sample_f32(in_r));
        let clean_thresh_db = sanitize_param_f64(threshold_db, -24.0, -70.0, 0.0);
        let min_amp = 10.0_f64.powf(clean_thresh_db / 20.0);

        let k_abs = self.kick_lp.process_sample(mono).abs();
        let s_abs = self
            .snare_band
            .iter_mut()
            .fold(mono, |x, stage| stage.process_sample(x))
            .abs();
        let h_abs = self.hat_hp.process_sample(mono).abs();

        // The bands are independent, so each runs its envelopes, countdown and trigger in turn.
        let holdoff = self.min_holdoff_samples;
        [
            self.kick.process(k_abs, min_amp, holdoff, &KICK_TUNING),
            self.snare.process(s_abs, min_amp, holdoff, &SNARE_TUNING),
            self.hat.process(h_abs, min_amp, holdoff, &HAT_TUNING),
        ]
    }
}

#[cfg(test)]
mod sliding_min_tests {
    use super::SlidingMin;

    /// Checks `SlidingMin` against a brute-force minimum over the same window.
    fn check(capacity: usize, window: usize, values: &[f64]) {
        let mut sm = SlidingMin::new(capacity);
        sm.reset(window);
        let window = window.clamp(1, capacity);
        for (i, &v) in values.iter().enumerate() {
            let got = sm.push(v);
            let start = (i + 1).saturating_sub(window);
            let want = values[start..=i]
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            assert_eq!(got, want, "cap {capacity} window {window} at {i}");
        }
    }

    #[test]
    fn rising_run_at_full_capacity_keeps_the_window_minimum() {
        // The 192 kHz case: the hold window equals the deque capacity and the required gain
        // rises for longer than the window.
        let values: Vec<f64> = (0..1_200).map(|i| 0.2 + 0.001 * i as f64).collect();
        check(519, 519, &values);
    }

    #[test]
    fn matches_brute_force_on_mixed_input() {
        let values: Vec<f64> = (0..2_000_u32)
            .map(|i| f64::from(i.wrapping_mul(2_654_435_761) % 1_000) / 1_000.0)
            .collect();
        for (cap, window) in [
            (1, 1),
            (8, 1),
            (8, 3),
            (8, 8),
            (64, 64),
            (519, 151),
            (519, 519),
        ] {
            check(cap, window, &values);
        }
    }
}

#[cfg(test)]
mod tape_mix_tests {
    use super::RealtimeSubAndTapeBus;

    #[test]
    fn tape_mix_starts_from_the_first_drive_value() {
        for drive in [0.0, 0.5] {
            let mut bus = RealtimeSubAndTapeBus::new(48_000.0, 160.0);
            for _ in 0..2 {
                bus.process_stereo_sample(0.5, 0.5, 0.0, 1.0, drive);
                let want = if drive > 0.0 { 1.0 } else { 0.0 };
                assert_eq!(
                    bus.tape_mix, want,
                    "drive {drive}: tape mix after first samples"
                );
                // A reset re-primes from the next drive value.
                bus.reset(48_000.0);
            }
        }
    }
}
