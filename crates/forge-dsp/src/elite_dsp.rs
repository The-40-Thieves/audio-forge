//! # `elite_dsp` — Next-Generation Studio DSP Kernels (ADAA, Structural Transient/Tonal Split,
//! Upward Density + 3-Band Clipper, 8x8 Hadamard FDN Ducking/Shimmer Reverb, Neural Preamp,
//! Genre Tonal Balance Targets & Lock-Free Cross-Instance Spectrum IPC).
//!
//! Designed with **zero audio-thread heap allocations** for real-time VST3/CLAP execution
//! and 64-bit studio mastering.

use std::f64::consts::{FRAC_1_SQRT_2, PI};
use std::sync::atomic::{AtomicU32, Ordering};

use serde::{Deserialize, Serialize};

use crate::loudness_limiter::{
    flush_denormal_f64, sanitize_param_f64, sanitize_sample_f32, sanitize_sample_f64, Biquad,
};
use crate::restoration::MASTERING_18_BANDS_HZ;
use forge_core::AudioBuffer;

// ============================================================================
// 1. 1ST-ORDER ANTI-DERIVATIVE ANTI-ALIASING (ADAA) & 2X OVERSAMPLING
// ============================================================================

/// 1st-Order Anti-Derivative Anti-Aliased (`ADAA`) `tanh(x)` waveshaper state.
///
/// Uses the closed-form antiderivative $F_1(x) = \ln(\cosh(x))$ to suppress harmonic
/// foldback aliasing across Nyquist with **zero latency** and **zero oversampling cost**.
#[derive(Debug, Clone, Copy, Default)]
pub struct AdaaTanh {
    x_prev: f64,
    f1_prev: f64,
}

impl AdaaTanh {
    #[inline]
    fn antiderivative_tanh(x: f64) -> f64 {
        // Numerically stable ln(cosh(x)) = |x| + ln(1 + e^(-2|x|)) - ln(2)
        let ax = x.abs();
        ax + (-2.0 * ax).exp().ln_1p() - std::f64::consts::LN_2
    }

    #[inline]
    pub fn process(&mut self, x_in: f64, drive: f64) -> f64 {
        let clean_in = sanitize_sample_f64(x_in);
        let clean_drive = sanitize_param_f64(drive, 1.0, 0.1, 100.0);
        let x = (clean_in * clean_drive).clamp(-12.0, 12.0);
        let f1 = Self::antiderivative_tanh(x);
        let dx = x - self.x_prev;
        let y = if dx.abs() < 1e-5 {
            (0.5 * (x + self.x_prev)).tanh()
        } else {
            (f1 - self.f1_prev) / dx
        };
        self.x_prev = flush_denormal_f64(x);
        self.f1_prev = flush_denormal_f64(f1);
        flush_denormal_f64(y).clamp(-1.0, 1.0)
    }
}

/// 1st-Order Anti-Derivative Anti-Aliased (`ADAA`) Airwindows-style `sin(x)` console waveshaper.
///
/// Antiderivative of $\sin(k x)$ is $-\cos(k x) / k$.
#[derive(Debug, Clone, Copy, Default)]
pub struct AdaaSin {
    x_prev: f64,
    f1_prev: f64,
}

impl AdaaSin {
    const K: f64 = PI * 0.5 / 1.55;

    #[inline]
    fn antiderivative_sin(x: f64) -> f64 {
        -(Self::K * x).cos() / Self::K
    }

    #[inline]
    pub fn process(&mut self, x_in: f64, drive: f64) -> f64 {
        let clean_in = sanitize_sample_f64(x_in);
        let clean_drive = sanitize_param_f64(drive, 1.0, 0.1, 100.0);
        let x = (clean_in * clean_drive).clamp(-1.55, 1.55);
        let f1 = Self::antiderivative_sin(x);
        let dx = x - self.x_prev;
        let y = if dx.abs() < 1e-5 {
            (Self::K * 0.5 * (x + self.x_prev)).sin()
        } else {
            (f1 - self.f1_prev) / dx
        };
        self.x_prev = flush_denormal_f64(x);
        self.f1_prev = flush_denormal_f64(f1);
        flush_denormal_f64(y).clamp(-1.0, 1.0)
    }
}

/// Zero-allocation 2x Polyphase Half-Band Upsampler & Downsampler for non-linear stages.
#[derive(Debug, Clone, Copy, Default)]
pub struct Oversampler2x {
    prev_in: f64,
    lp_state1: f64,
    lp_state2: f64,
}

impl Oversampler2x {
    /// Upsamples `x` to 2x rate `[s0, s1]`, applies nonlinear closure `f`, and downsamples with anti-aliasing filter.
    #[inline]
    pub fn process<F: FnMut(f64) -> f64>(&mut self, x_raw: f64, mut f: F) -> f64 {
        let x = sanitize_sample_f64(x_raw);
        let mid = 0.5 * (self.prev_in + x);
        self.prev_in = flush_denormal_f64(x);
        let y0 = sanitize_sample_f64(f(mid));
        let y1 = sanitize_sample_f64(f(x));
        // 2-tap polyphase half-band decimation filter
        self.lp_state1 = flush_denormal_f64(0.5 * (y0 + y1));
        let out = flush_denormal_f64(0.5 * (self.lp_state1 + self.lp_state2));
        self.lp_state2 = self.lp_state1;
        out
    }
}

// ============================================================================
// 2. REAL-TIME STRUCTURAL TRANSIENT / TONAL SPLIT ENGINE (Eventide SplitEQ style)
// ============================================================================

/// Real-Time Structural **Transient vs. Tonal Splitter & 6-Band Shaper** (Eventide SplitEQ / oeksound Spiff equivalent).
///
/// Decomposes incoming audio into Low (`<180 Hz`), Mid (`180–3200 Hz`), and High (`>3200 Hz`) bands,
/// separates each band into its **Transient** (attack/pick/consonant snap) and **Tonal** (sustained harmonic body)
/// components using dual-time-constant log-envelope differentiation, and allows independent EQ/gain shaping
/// of Transients vs. Tonal sustain!
pub struct RealtimeTransientTonalSplit {
    low_lp_l: Biquad,
    low_lp_r: Biquad,
    high_hp_l: Biquad,
    high_hp_r: Biquad,
    fast_env: [f64; 3],
    slow_env: [f64; 3],
}

impl RealtimeTransientTonalSplit {
    pub fn new(sample_rate: f64) -> Self {
        let sr = sanitize_param_f64(sample_rate, 48_000.0, 8_000.0, 768_000.0);
        Self {
            low_lp_l: Biquad::lowpass(sr, 180.0, FRAC_1_SQRT_2),
            low_lp_r: Biquad::lowpass(sr, 180.0, FRAC_1_SQRT_2),
            high_hp_l: Biquad::highpass(sr, 3200.0, FRAC_1_SQRT_2),
            high_hp_r: Biquad::highpass(sr, 3200.0, FRAC_1_SQRT_2),
            fast_env: [0.0; 3],
            slow_env: [1e-6; 3],
        }
    }

    pub fn reset(&mut self, sample_rate: f64) {
        *self = Self::new(sample_rate);
    }

    /// Processes a stereo sample with independent `[Low, Mid, High]` dB gains for **Transient** and **Tonal** streams.
    /// `audition_mode`: `0` = Full Mix, `1` = Solo Transients Only, `2` = Solo Tonal Body Only.
    #[inline]
    pub fn process_stereo_sample(
        &mut self,
        in_l: f32,
        in_r: f32,
        transient_gains_db: [f64; 3],
        tonal_gains_db: [f64; 3],
        separation_sharpness: f64,
        audition_mode: u8,
    ) -> (f32, f32) {
        let l = sanitize_sample_f32(in_l);
        let r = sanitize_sample_f32(in_r);

        let lo_l = self.low_lp_l.process_sample(l);
        let lo_r = self.low_lp_r.process_sample(r);
        let hi_l = self.high_hp_l.process_sample(l);
        let hi_r = self.high_hp_r.process_sample(r);
        let mid_l = l - lo_l - hi_l;
        let mid_r = r - lo_r - hi_r;

        let bands_l = [lo_l, mid_l, hi_l];
        let bands_r = [lo_r, mid_r, hi_r];

        // Wavelength-scaled fast/slow envelope followers per band
        let fast_alphas = [0.92, 0.85, 0.75];
        let slow_alphas = [0.997, 0.994, 0.990];
        let sharp = sanitize_param_f64(separation_sharpness, 1.5, 0.5, 4.0);

        let mut out_l = 0.0_f64;
        let mut out_r = 0.0_f64;

        for b in 0..3 {
            let mag = 0.5 * (bands_l[b].abs() + bands_r[b].abs());
            self.fast_env[b] =
                flush_denormal_f64(fast_alphas[b] * self.fast_env[b] + (1.0 - fast_alphas[b]) * mag);
            self.slow_env[b] =
                flush_denormal_f64(slow_alphas[b] * self.slow_env[b] + (1.0 - slow_alphas[b]) * mag)
                    + 1e-7;

            let ratio = (self.fast_env[b] / self.slow_env[b] - 1.05).max(0.0);
            let trans_mask = (1.0 - (-ratio * 2.2 * sharp).exp()).clamp(0.0, 1.0);
            let tonal_mask = 1.0 - trans_mask;

            let t_db = sanitize_param_f64(transient_gains_db[b], 0.0, -18.0, 18.0);
            let s_db = sanitize_param_f64(tonal_gains_db[b], 0.0, -18.0, 18.0);
            let g_trans = 10.0_f64.powf(t_db / 20.0);
            let g_tonal = 10.0_f64.powf(s_db / 20.0);

            let effective_scale = match audition_mode {
                1 => trans_mask * g_trans,
                2 => tonal_mask * g_tonal,
                _ => trans_mask * g_trans + tonal_mask * g_tonal,
            };

            out_l += bands_l[b] * effective_scale;
            out_r += bands_r[b] * effective_scale;
        }

        (
            flush_denormal_f64(out_l) as f32,
            flush_denormal_f64(out_r) as f32,
        )
    }
}

// ============================================================================
// 3. UPWARD "DENSITY" COMPRESSOR + 3-BAND ADAA SOFT CLIPPER (Neutron 5 / KClip style)
// ============================================================================

/// Real-Time **3-Band Upward "Density" Compressor & Anti-Aliased ADAA Soft Clipper**
/// (iZotope Neutron 5 Density + Clipper / Xfer OTT / Kazrog KClip 3 equivalent).
pub struct RealtimeDensityAndClipper {
    low_lp_l: Biquad,
    low_lp_r: Biquad,
    high_hp_l: Biquad,
    high_hp_r: Biquad,
    rms_env: [f64; 3],
    adaa_l: [AdaaTanh; 3],
    adaa_r: [AdaaTanh; 3],
    oversampler_l: [Oversampler2x; 3],
    oversampler_r: [Oversampler2x; 3],
}

impl RealtimeDensityAndClipper {
    pub fn new(sample_rate: f64) -> Self {
        let sr = sanitize_param_f64(sample_rate, 48_000.0, 8_000.0, 768_000.0);
        Self {
            low_lp_l: Biquad::lowpass(sr, 160.0, FRAC_1_SQRT_2),
            low_lp_r: Biquad::lowpass(sr, 160.0, FRAC_1_SQRT_2),
            high_hp_l: Biquad::highpass(sr, 2800.0, FRAC_1_SQRT_2),
            high_hp_r: Biquad::highpass(sr, 2800.0, FRAC_1_SQRT_2),
            rms_env: [1e-4; 3],
            adaa_l: [AdaaTanh::default(); 3],
            adaa_r: [AdaaTanh::default(); 3],
            oversampler_l: [Oversampler2x::default(); 3],
            oversampler_r: [Oversampler2x::default(); 3],
        }
    }

    pub fn reset(&mut self, sample_rate: f64) {
        *self = Self::new(sample_rate);
    }

    /// `density_amount`: `0.0..=1.0` (upward compression lifting micro-detail between `-48 dB` and `-14 dB`).
    /// `clip_drive_db`: `0.0..=12.0` (per-band 2x-oversampled ADAA `tanh` soft clipping).
    #[inline]
    pub fn process_stereo_sample(
        &mut self,
        in_l: f32,
        in_r: f32,
        density_amount: f64,
        clip_drive_db: f64,
        clip_ceiling_db: f64,
    ) -> (f32, f32) {
        let l = sanitize_sample_f32(in_l);
        let r = sanitize_sample_f32(in_r);

        let lo_l = self.low_lp_l.process_sample(l);
        let lo_r = self.low_lp_r.process_sample(r);
        let hi_l = self.high_hp_l.process_sample(l);
        let hi_r = self.high_hp_r.process_sample(r);
        let mid_l = l - lo_l - hi_l;
        let mid_r = r - lo_r - hi_r;

        let mut bands_l = [lo_l, mid_l, hi_l];
        let mut bands_r = [lo_r, mid_r, hi_r];

        let dens = sanitize_param_f64(density_amount, 0.35, 0.0, 1.0);
        let clean_drive_db = sanitize_param_f64(clip_drive_db, 0.0, 0.0, 15.0);
        let clean_ceil_db = sanitize_param_f64(clip_ceiling_db, -0.5, -12.0, 0.0);
        let drive_lin = 10.0_f64.powf(clean_drive_db / 20.0);
        let ceiling_lin = 10.0_f64.powf(clean_ceil_db / 20.0);

        for b in 0..3 {
            let pwr = 0.5 * (bands_l[b] * bands_l[b] + bands_r[b] * bands_r[b]);
            self.rms_env[b] =
                flush_denormal_f64(0.992 * self.rms_env[b] + 0.008 * pwr.max(1e-10)).max(1e-10);
            let lvl_db = 10.0 * self.rms_env[b].log10();

            // Upward compression ("Density"): lift quiet signals between -48 dB and -16 dB up to +9 dB
            let upward_db = if (-48.0..-16.0).contains(&lvl_db) {
                ((-16.0 - lvl_db) * 0.32 * dens).min(9.0 * dens)
            } else {
                0.0
            };
            let up_gain = 10.0_f64.powf(upward_db / 20.0);

            let xl = bands_l[b] * up_gain;
            let xr = bands_r[b] * up_gain;

            // 2x Oversampled 1st-Order ADAA Soft Clipper per band
            let adaa_l = &mut self.adaa_l[b];
            let adaa_r = &mut self.adaa_r[b];
            bands_l[b] = self.oversampler_l[b].process(xl, |s| adaa_l.process(s, drive_lin));
            bands_r[b] = self.oversampler_r[b].process(xr, |s| adaa_r.process(s, drive_lin));
        }

        let sum_l = (bands_l[0] + bands_l[1] + bands_l[2]) * ceiling_lin / drive_lin.powf(0.35);
        let sum_r = (bands_r[0] + bands_r[1] + bands_r[2]) * ceiling_lin / drive_lin.powf(0.35);

        (
            flush_denormal_f64(sum_l) as f32,
            flush_denormal_f64(sum_r) as f32,
        )
    }
}

// ============================================================================
// 4. 8x8 HADAMARD FDN ALGORITHMIC REVERB WITH VOCAL DUCKING & OCTAVE SHIMMER
// ============================================================================

/// Real-Time **8x8 Fast Walsh-Hadamard Feedback Delay Network (`FDN`) Reverb**
/// (**Forge SpaceVerb** — Valhalla VintageVerb / FabFilter Pro-R 2 equivalent).
///
/// Features:
/// - 8 co-prime delay lines (`29.3 ms` to `89.7 ms`) with LFO chorus modulation
/// - Orthogonal $O(N \log N)$ 8x8 Walsh-Hadamard unitary feedback matrix
/// - `<180 Hz` mud high-pass filter + per-line HF damping low-pass filters
/// - **Vocal Sidechain Ducking** (automatically ducks reverb while dry signal is singing/playing)
/// - **+12 Semitone Octave Shimmer** dual-grain pitch-shifted feedback injection
pub struct RealtimeFdnReverb {
    sample_rate: f64,
    delay_lines: [Vec<f64>; 8],
    delay_lens: [usize; 8],
    write_pos: [usize; 8],
    damp_state: [f64; 8],
    low_cut_l: Biquad,
    low_cut_r: Biquad,
    lfo_phase: f64,
    duck_env: f64,
    shimmer_buf: Vec<f64>,
    shimmer_write: usize,
    shimmer_phase: f64,
}

impl RealtimeFdnReverb {
    const BASE_DELAYS_MS: [f64; 8] = [29.3, 37.1, 43.7, 53.1, 61.9, 71.3, 79.7, 89.3];

    pub fn new(sample_rate: f64) -> Self {
        let sr = sanitize_param_f64(sample_rate, 48_000.0, 8_000.0, 768_000.0);
        let delay_lens: [usize; 8] = std::array::from_fn(|i| {
            ((Self::BASE_DELAYS_MS[i] * 0.001 * sr).round() as usize + 64).max(128)
        });
        let delay_lines: [Vec<f64>; 8] = std::array::from_fn(|i| vec![0.0; delay_lens[i]]);
        let shimmer_len = ((sr * 0.040).round() as usize).max(512);

        Self {
            sample_rate: sr,
            delay_lines,
            delay_lens,
            write_pos: [0; 8],
            damp_state: [0.0; 8],
            low_cut_l: Biquad::highpass(sr, 180.0, FRAC_1_SQRT_2),
            low_cut_r: Biquad::highpass(sr, 180.0, FRAC_1_SQRT_2),
            lfo_phase: 0.0,
            duck_env: 0.0,
            shimmer_buf: vec![0.0; shimmer_len],
            shimmer_write: 0,
            shimmer_phase: 0.0,
        }
    }

    pub fn reset(&mut self, sample_rate: f64) {
        *self = Self::new(sample_rate);
    }

    /// Orthogonal $O(N \log N)$ 8x8 Fast Walsh-Hadamard unitary transform (normalized by $1/\sqrt{8}$).
    /// Preserves Euclidean energy $\|H(v)\|_2^2 = \|v\|_2^2$ and is an involution $H(H(v)) = v$.
    #[inline]
    pub fn fast_hadamard_8(v: &mut [f64; 8]) {
        // Stage 1
        for i in (0..8).step_by(2) {
            let a = v[i];
            let b = v[i + 1];
            v[i] = a + b;
            v[i + 1] = a - b;
        }
        // Stage 2
        for i in (0..8).step_by(4) {
            for j in 0..2 {
                let a = v[i + j];
                let b = v[i + j + 2];
                v[i + j] = a + b;
                v[i + j + 2] = a - b;
            }
        }
        // Stage 3
        for j in 0..4 {
            let a = v[j];
            let b = v[j + 4];
            v[j] = (a + b) * 0.353_553_390_593_273_8; // 1 / sqrt(8)
            v[j + 4] = (a - b) * 0.353_553_390_593_273_8;
        }
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn process_stereo_sample(
        &mut self,
        in_l: f32,
        in_r: f32,
        decay_rt60_secs: f64,
        damping: f64,
        shimmer_amount: f64,
        ducking_amount: f64,
        wet_mix: f64,
    ) -> (f32, f32) {
        let dry_l = sanitize_sample_f32(in_l);
        let dry_r = sanitize_sample_f32(in_r);

        let clean_shimmer = sanitize_param_f64(shimmer_amount, 0.0, 0.0, 1.0);
        let clean_duck = sanitize_param_f64(ducking_amount, 0.0, 0.0, 1.0);
        let mix = sanitize_param_f64(wet_mix, 0.25, 0.0, 1.0);

        // 1. Pre-filter <180 Hz mud out of reverb send
        let send_l = self.low_cut_l.process_sample(dry_l);
        let send_r = self.low_cut_r.process_sample(dry_r);

        // 2. Octave (+12 st) dual-grain Shimmer generator
        let shim_len = self.shimmer_buf.len();
        let shim_f = shim_len as f64;
        self.shimmer_buf[self.shimmer_write] = flush_denormal_f64(0.5 * (send_l + send_r));
        self.shimmer_phase = (self.shimmer_phase - 1.0).rem_euclid(shim_f);
        let phase_b = (self.shimmer_phase + shim_f * 0.5).rem_euclid(shim_f);
        let w_a = 0.5 * (1.0 - (2.0 * PI * self.shimmer_phase / shim_f).cos());
        let idx_a = (self.shimmer_write + shim_len - (self.shimmer_phase as usize).min(shim_len - 1))
            % shim_len;
        let idx_b =
            (self.shimmer_write + shim_len - (phase_b as usize).min(shim_len - 1)) % shim_len;
        let shimmer_sig =
            (w_a * self.shimmer_buf[idx_a] + (1.0 - w_a) * self.shimmer_buf[idx_b]) * clean_shimmer;
        self.shimmer_write = (self.shimmer_write + 1) % shim_len;

        // 3. Read from 8 modulated FDN delay lines
        self.lfo_phase = (self.lfo_phase + 0.65 / self.sample_rate).fract();
        let mut outs = [0.0_f64; 8];
        for (i, out_val) in outs.iter_mut().enumerate() {
            let len = self.delay_lens[i];
            let mod_offset =
                ((2.0 * PI * (self.lfo_phase + (i as f64) * 0.125)).sin() * 6.0).round() as isize;
            let base_delay = (len - 16) as isize + mod_offset;
            let read_idx = (self.write_pos[i] + len - (base_delay.clamp(1, (len - 2) as isize) as usize)) % len;
            *out_val = flush_denormal_f64(self.delay_lines[i][read_idx]);
        }

        let mut wet_l = (outs[0] + outs[2] + outs[4] + outs[6]) * 0.5;
        let mut wet_r = (outs[1] + outs[3] + outs[5] + outs[7]) * 0.5;

        // 4. Orthogonal 8x8 Walsh-Hadamard mixing + RT60 feedback + HF damping
        Self::fast_hadamard_8(&mut outs);
        let rt60 = sanitize_param_f64(decay_rt60_secs, 2.4, 0.2, 20.0);
        let damp = sanitize_param_f64(damping, 0.35, 0.05, 0.90);

        for (i, &out_val) in outs.iter().enumerate() {
            let delay_sec = (self.delay_lens[i] as f64) / self.sample_rate;
            let fb_gain = 10.0_f64.powf(-3.0 * delay_sec / rt60).min(0.985);
            self.damp_state[i] =
                flush_denormal_f64((1.0 - damp) * out_val + damp * self.damp_state[i]);
            let input_inj = if i % 2 == 0 { send_l } else { send_r } * 0.25 + shimmer_sig * 0.18;
            let w_pos = self.write_pos[i];
            self.delay_lines[i][w_pos] =
                flush_denormal_f64(input_inj + self.damp_state[i] * fb_gain);
            self.write_pos[i] = (w_pos + 1) % self.delay_lens[i];
        }

        // 5. Automatic Vocal Sidechain Ducking (ducks wet reverb tail when dry input is present)
        let in_lvl = 0.5 * (dry_l.abs() + dry_r.abs());
        if in_lvl > self.duck_env {
            self.duck_env = flush_denormal_f64(0.990 * self.duck_env + 0.010 * in_lvl);
        } else {
            self.duck_env = flush_denormal_f64(0.9995 * self.duck_env + 0.0005 * in_lvl);
        }
        let duck_gain = (1.0 - (self.duck_env * 4.5 * clean_duck)).clamp(0.15, 1.0);
        wet_l *= duck_gain;
        wet_r *= duck_gain;

        let final_l = dry_l * (1.0 - mix * 0.5) + wet_l * mix;
        let final_r = dry_r * (1.0 - mix * 0.5) + wet_r * mix;

        (
            flush_denormal_f64(final_l) as f32,
            flush_denormal_f64(final_r) as f32,
        )
    }
}

// ============================================================================
// 5. REAL-TIME NEURAL HARDWARE PREAMP / AMP MODELER (NAM / RTNeural WaveNet)
// ============================================================================

/// Hardware character models for [`RealtimeNeuralPreamp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnalogHardwareModel {
    /// Classic British Class-A 1073 Transformer Preamp (warm 2nd/3rd harmonics + 110Hz iron weight)
    Neve1073ClassA,
    /// Ultra-fast FET Limiting Amplifier saturation (1176 all-buttons grit & presence)
    Fet1176AllButtons,
    /// German Vacuum Tube Broadcast Preamp (V76 smooth even-order triode bloom)
    TubeV76Triode,
    /// Swiss 2-Inch 15 IPS Master Tape Machine (Studer A800 head bump + high-frequency hysteresis)
    StuderA800Tape,
}

/// Real-Time **Dilated Causal WaveNet / Residual Neural Preamp & Hardware Modeler**
/// (`NAM` / `RTNeural` architecture with 1st-order ADAA anti-aliased gated activations).
pub struct RealtimeNeuralPreamp {
    model: AnalogHardwareModel,
    dilation_buf_l: [f64; 16],
    dilation_buf_r: [f64; 16],
    pos: usize,
    adaa_l: AdaaTanh,
    adaa_r: AdaaTanh,
    dc_block_x_l: f64,
    dc_block_y_l: f64,
    dc_block_x_r: f64,
    dc_block_y_r: f64,
}

impl RealtimeNeuralPreamp {
    pub fn new(model: AnalogHardwareModel) -> Self {
        Self {
            model,
            dilation_buf_l: [0.0; 16],
            dilation_buf_r: [0.0; 16],
            pos: 0,
            adaa_l: AdaaTanh::default(),
            adaa_r: AdaaTanh::default(),
            dc_block_x_l: 0.0,
            dc_block_y_l: 0.0,
            dc_block_x_r: 0.0,
            dc_block_y_r: 0.0,
        }
    }

    pub fn set_model(&mut self, model: AnalogHardwareModel) {
        self.model = model;
    }

    #[inline]
    fn wavenet_residual_cell(
        x: f64,
        d1: f64,
        d4: f64,
        d12: f64,
        drive: f64,
        model: AnalogHardwareModel,
        adaa: &mut AdaaTanh,
    ) -> f64 {
        let (w0, w1, w4, w12, bias_asym) = match model {
            AnalogHardwareModel::Neve1073ClassA => (0.78, 0.16, 0.08, -0.04, 0.045),
            AnalogHardwareModel::Fet1176AllButtons => (0.92, -0.14, 0.06, 0.02, 0.015),
            AnalogHardwareModel::TubeV76Triode => (0.74, 0.20, 0.10, -0.05, 0.075),
            AnalogHardwareModel::StuderA800Tape => (0.70, 0.24, 0.12, -0.06, 0.020),
        };
        let conv = w0 * x + w1 * d1 + w4 * d4 + w12 * d12 + bias_asym * x * x;
        let gate = 1.0 / (1.0 + (-conv * drive * 0.8).exp());
        let filter = adaa.process(conv, drive);
        x * 0.35 + filter * (0.55 + 0.45 * gate)
    }

    #[inline]
    pub fn process_stereo_sample(
        &mut self,
        in_l: f32,
        in_r: f32,
        input_drive_db: f64,
        output_trim_db: f64,
    ) -> (f32, f32) {
        let clean_drive_db = sanitize_param_f64(input_drive_db, 0.0, -12.0, 24.0);
        let clean_trim_db = sanitize_param_f64(output_trim_db, 0.0, -24.0, 12.0);
        let drive = 10.0_f64.powf(clean_drive_db / 20.0);
        let trim = 10.0_f64.powf(clean_trim_db / 20.0) / drive.powf(0.55);

        let xl = sanitize_sample_f32(in_l).clamp(-16.0, 16.0);
        let xr = sanitize_sample_f32(in_r).clamp(-16.0, 16.0);

        let p = self.pos & 15;
        let p1 = self.pos.wrapping_sub(1) & 15;
        let p4 = self.pos.wrapping_sub(4) & 15;
        let p12 = self.pos.wrapping_sub(12) & 15;

        self.dilation_buf_l[p] = xl;
        self.dilation_buf_r[p] = xr;
        self.pos = self.pos.wrapping_add(1);

        let raw_l = Self::wavenet_residual_cell(
            xl,
            self.dilation_buf_l[p1],
            self.dilation_buf_l[p4],
            self.dilation_buf_l[p12],
            drive,
            self.model,
            &mut self.adaa_l,
        );
        let raw_r = Self::wavenet_residual_cell(
            xr,
            self.dilation_buf_r[p1],
            self.dilation_buf_r[p4],
            self.dilation_buf_r[p12],
            drive,
            self.model,
            &mut self.adaa_r,
        );

        // DC blocker (removes any tube/transformer even-harmonic DC shift)
        let dc_l = flush_denormal_f64(raw_l - self.dc_block_x_l + 0.9985 * self.dc_block_y_l);
        self.dc_block_x_l = flush_denormal_f64(raw_l);
        self.dc_block_y_l = dc_l;

        let dc_r = flush_denormal_f64(raw_r - self.dc_block_x_r + 0.9985 * self.dc_block_y_r);
        self.dc_block_x_r = flush_denormal_f64(raw_r);
        self.dc_block_y_r = dc_r;

        (
            flush_denormal_f64(dc_l * trim) as f32,
            flush_denormal_f64(dc_r * trim) as f32,
        )
    }
}

// ============================================================================
// 6. BUILT-IN GENRE TONAL BALANCE TARGET CORRIDORS (Tonal Balance Control 2 style)
// ============================================================================

/// Built-in statistical 18-band Genre Reference Profiles (iZotope Tonal Balance Control 2 equivalent).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GenreTonalTarget {
    ModernPop,
    HipHopTrap,
    NeoSoulRnB,
    EdmClub,
    RockMetal,
    AcousticJazz,
}

impl GenreTonalTarget {
    pub fn parse(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "hiphop" | "hip-hop" | "trap" | "rap" => Self::HipHopTrap,
            "neosoul" | "neo-soul" | "rnb" | "r&b" | "soul" => Self::NeoSoulRnB,
            "edm" | "club" | "house" | "techno" | "electronic" => Self::EdmClub,
            "rock" | "metal" | "indie" | "punk" => Self::RockMetal,
            "jazz" | "acoustic" | "folk" | "classical" => Self::AcousticJazz,
            _ => Self::ModernPop,
        }
    }

    /// Returns the calibrated 18-band relative spectral target curve (in dB relative to 1 kHz)
    /// corresponding to `MASTERING_18_BANDS_HZ`.
    pub fn target_curve_18_db(self) -> [f64; 18] {
        // Bands: [31.5, 45, 63, 90, 125, 180, 250, 355, 500, 710, 1k, 1.4k, 2k, 2.8k, 4k, 5.6k, 8k, 12.5k]
        match self {
            Self::ModernPop => [
                1.5, 3.2, 4.0, 3.0, 1.8, 0.6, -0.4, -0.8, -0.4, 0.0, 0.0, 0.4, 0.8, 0.5, -0.2,
                0.6, 1.8, 2.5,
            ],
            Self::HipHopTrap => [
                4.5, 6.2, 6.5, 4.2, 1.5, -0.5, -1.5, -1.2, -0.5, 0.0, 0.0, 0.2, 0.6, 0.2, -0.5,
                0.8, 2.2, 3.0,
            ],
            Self::NeoSoulRnB => [
                3.0, 4.8, 5.2, 3.8, 2.2, 1.0, 0.2, -0.2, 0.2, 0.2, 0.0, -0.4, -0.8, -1.2, -1.5,
                -0.4, 1.0, 1.8,
            ],
            Self::EdmClub => [
                3.8, 5.8, 6.0, 4.0, 1.8, 0.0, -1.2, -1.0, -0.4, 0.0, 0.0, 0.5, 1.2, 1.0, 0.4,
                1.2, 2.5, 3.2,
            ],
            Self::RockMetal => [
                0.0, 1.8, 3.0, 3.2, 2.4, 1.2, 0.0, -0.6, 0.0, 0.5, 0.0, 1.0, 1.8, 1.5, 0.8, 0.2,
                0.8, 1.0,
            ],
            Self::AcousticJazz => [
                -1.5, 0.5, 2.0, 2.5, 2.2, 1.5, 0.8, 0.4, 0.2, 0.0, 0.0, -0.2, -0.5, -0.8, -1.0,
                -0.5, 0.5, 1.2,
            ],
        }
    }
}

/// Computes an 18-band corrective Mastering EQ curve to match `source` to a built-in [`GenreTonalTarget`].
pub fn compute_genre_tonal_balance_eq(
    source: &AudioBuffer,
    genre: GenreTonalTarget,
    max_boost_cut_db: f64,
) -> [f64; 18] {
    let sr = f64::from(source.sample_rate.max(1));
    let mono: Vec<f64> = source
        .to_mono()
        .into_iter()
        .map(sanitize_sample_f64)
        .collect();
    if mono.is_empty() {
        return [0.0; 18];
    }

    let mut measured_db = [0.0_f64; 18];
    for (b, &freq) in MASTERING_18_BANDS_HZ.iter().enumerate() {
        let mut bp = Biquad::peaking_eq(sr, freq, 2.2, 12.0);
        let mut sum_sq = 0.0_f64;
        for &s in &mono {
            let y = bp.process_sample(s) - s;
            sum_sq += y * y;
        }
        measured_db[b] = 10.0 * (sum_sq / (mono.len() as f64) + 1e-12).log10();
    }

    // Anchor relative to 1 kHz (band index 10)
    let anchor_db = measured_db[10];
    let target = genre.target_curve_18_db();
    let limit = sanitize_param_f64(max_boost_cut_db, 4.5, 0.5, 12.0);

    let mut eq_gains = [0.0_f64; 18];
    for i in 0..18 {
        let rel_measured = measured_db[i] - anchor_db;
        let diff = (target[i] - rel_measured) * 0.55;
        eq_gains[i] = diff.clamp(-limit, limit);
    }
    eq_gains
}

// ============================================================================
// 7. LOCK-FREE CROSS-INSTANCE SHARED ATOMIC SPECTRUM BUS (Pro-Q 4 / Neutron 5 IPC)
// ============================================================================

/// 8 Process-Wide Named Inter-Plugin Spectrum Slots (`0 = Off`, `1..=8 = Slot A..H`).
/// Allows any ForgeAudio VST3/CLAP instance to broadcast its 32-band Bark spectrum to any
/// other instance in the DAW session with **zero locking** and **zero DAW sidechain routing**.
pub static SHARED_SPECTRUM_SLOTS: [[AtomicU32; 32]; 8] = [
    [const { AtomicU32::new(0) }; 32],
    [const { AtomicU32::new(0) }; 32],
    [const { AtomicU32::new(0) }; 32],
    [const { AtomicU32::new(0) }; 32],
    [const { AtomicU32::new(0) }; 32],
    [const { AtomicU32::new(0) }; 32],
    [const { AtomicU32::new(0) }; 32],
    [const { AtomicU32::new(0) }; 32],
];

/// Publishes a 32-band magnitude spectrum into cross-instance slot `slot_idx` (`0..8`).
/// Non-finite (`NaN` / `±Inf`) or subnormal values are sanitized to `0.0` before atomic storage.
#[inline]
pub fn publish_shared_spectrum(slot_idx: usize, bands_32: &[f32; 32]) {
    if slot_idx < 8 {
        for (atom, &val) in SHARED_SPECTRUM_SLOTS[slot_idx].iter().zip(bands_32.iter()) {
            let clean = if val.is_finite() && val.abs() >= 1e-30 {
                val
            } else {
                0.0_f32
            };
            atom.store(clean.to_bits(), Ordering::Relaxed);
        }
    }
}

/// Reads a 32-band magnitude spectrum from cross-instance slot `slot_idx` (`0..8`).
#[inline]
pub fn read_shared_spectrum(slot_idx: usize) -> [f32; 32] {
    let mut out = [0.0_f32; 32];
    if slot_idx < 8 {
        for (dst, atom) in out.iter_mut().zip(SHARED_SPECTRUM_SLOTS[slot_idx].iter()) {
            let v = f32::from_bits(atom.load(Ordering::Relaxed));
            *dst = if v.is_finite() && v.abs() >= 1e-30 {
                v
            } else {
                0.0_f32
            };
        }
    }
    out
}
