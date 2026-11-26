//! Biquad filter primitives, ITU-R BS.1770-4 K-weighting, 4x oversampled True-Peak,
//! AI Fizz telemetry auditor, and lookahead brickwall mastering limiter.

use std::f64::consts::PI;
use std::sync::{Arc, OnceLock};

use forge_core::{AudioBuffer, AuditReport, ForgeError, Result};
use rustfft::{Fft, FftPlanner, num_complex::Complex};

/// Sanitizes an arbitrary `f64` audio sample: flushes `NaN`, `±Inf`, and subnormals (`|x| < 1e-30`)
/// to `0.0`, and bounds extreme finite amplitudes to `[-1e6, 1e6]` (`+120 dBFS`) so squaring (`x * x`)
/// can never overflow `f64::MAX`.
#[inline]
pub fn sanitize_sample_f64(x: f64) -> f64 {
    if !x.is_finite() || x.abs() < 1e-30 {
        0.0
    } else {
        x.clamp(-1.0e6, 1.0e6)
    }
}

/// Promotes an `f32` audio sample to `f64` and sanitizes `NaN`, `±Inf`, and subnormals.
#[inline]
pub fn sanitize_sample_f32(x: f32) -> f64 {
    sanitize_sample_f64(f64::from(x))
}

/// Flushes subnormal (`|x| < 1e-30`) or non-finite recursive filter state to `0.0`
/// to prevent x86/ARM denormal CPU traps and permanent `NaN` state poisoning.
#[inline]
pub fn flush_denormal_f64(x: f64) -> f64 {
    if !x.is_finite() || x.abs() < 1e-30 {
        0.0
    } else {
        x
    }
}

/// Safely clamps a parameter `val` into `[min, max]`, substituting `default` if `val` is `NaN`
/// and ordering `(min, max)` so `min > max` can never panic.
#[inline]
pub fn sanitize_param_f64(val: f64, default: f64, min: f64, max: f64) -> f64 {
    let lo = if min.is_finite() { min } else { default };
    let hi = if max.is_finite() && max >= lo {
        max
    } else {
        lo
    };
    if !val.is_finite() {
        default.clamp(lo, hi)
    } else {
        val.clamp(lo, hi)
    }
}

/// Second-order Direct Form II Transposed IIR Biquad filter.
#[derive(Debug, Clone, Copy)]
pub struct Biquad {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
    z1: f64,
    z2: f64,
}

impl Biquad {
    #[inline]
    pub const fn new(b0: f64, b1: f64, b2: f64, a1: f64, a2: f64) -> Self {
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            z1: 0.0,
            z2: 0.0,
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    /// Copies the coefficients of `design` while keeping this filter's `z1`/`z2` state, so a
    /// running filter can be retuned without the click that rebuilding it from scratch causes.
    #[inline]
    pub fn set_coeffs(&mut self, design: &Self) {
        self.b0 = design.b0;
        self.b1 = design.b1;
        self.b2 = design.b2;
        self.a1 = design.a1;
        self.a2 = design.a2;
    }

    #[inline(always)]
    fn step_raw(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }

    #[inline]
    fn flush_state(&mut self) {
        self.z1 = flush_denormal_f64(self.z1);
        self.z2 = flush_denormal_f64(self.z2);
    }

    #[inline]
    fn process_sanitized(&mut self, x: f64) -> f64 {
        let y_raw = self.b0 * x + self.z1;
        if !y_raw.is_finite() {
            self.z1 = 0.0;
            self.z2 = 0.0;
            return 0.0;
        }
        let y = flush_denormal_f64(y_raw);
        self.z1 = flush_denormal_f64(self.b1 * x - self.a1 * y + self.z2);
        self.z2 = flush_denormal_f64(self.b2 * x - self.a2 * y);
        y
    }

    #[inline]
    pub fn process_sample(&mut self, x_raw: f64) -> f64 {
        self.process_sanitized(sanitize_sample_f64(x_raw))
    }

    /// Audio EQ Cookbook peaking bell filter with Orfanidis / ZLEqualizer Nyquist Q De-Cramping.
    pub fn peaking_eq(sample_rate: f64, freq_hz: f64, q: f64, gain_db: f64) -> Self {
        let sr = sanitize_param_f64(sample_rate, 48_000.0, 100.0, 768_000.0);
        let f0 = sanitize_param_f64(freq_hz, 1000.0, 10.0, (sr * 0.48).max(10.0));
        let clean_db = sanitize_param_f64(gain_db, 0.0, -36.0, 36.0);
        let clean_q = sanitize_param_f64(q, 1.0, 0.05, 100.0);
        let a = 10.0_f64.powf(clean_db / 40.0);
        let w0 = 2.0 * PI * f0 / sr;
        let cos_w0 = w0.cos();
        // Orfanidis / Matched-Z Nyquist De-Cramping: prevents high-frequency bell narrowing near fs/2
        let decramped_q = (clean_q * (w0 * 0.5).cos().clamp(0.35, 1.0)).max(0.05);
        let alpha = w0.sin() / (2.0 * decramped_q);

        let b0 = 1.0 + alpha * a;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0 - alpha * a;
        let a0 = 1.0 + alpha / a;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha / a;

        Self::new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    /// Evaluates the exact analytical magnitude response $|H(e^{j\omega})|$ in dB at `freq_hz`
    /// for interactive Bode frequency-response curve plotting (`ZLEqualizer` style).
    #[inline]
    pub fn magnitude_db_at(&self, freq_hz: f64, sample_rate: f64) -> f64 {
        let sr = sanitize_param_f64(sample_rate, 48_000.0, 100.0, 768_000.0);
        let f = sanitize_param_f64(freq_hz, 1000.0, 10.0, sr * 0.495);
        let w = 2.0 * PI * f / sr;
        let c1 = w.cos();
        let s1 = w.sin();
        let c2 = (2.0 * w).cos();
        let s2 = (2.0 * w).sin();

        let num_re = self.b0 + self.b1 * c1 + self.b2 * c2;
        let num_im = -(self.b1 * s1 + self.b2 * s2);
        let den_re = 1.0 + self.a1 * c1 + self.a2 * c2;
        let den_im = -(self.a1 * s1 + self.a2 * s2);

        let num_mag_sq = (num_re * num_re + num_im * num_im).max(1.0e-18);
        let den_mag_sq = (den_re * den_re + den_im * den_im).max(1.0e-18);
        (10.0 * (num_mag_sq / den_mag_sq).log10()).clamp(-48.0, 48.0)
    }

    /// Audio EQ Cookbook narrow notch filter (ideal for 50/60 Hz hum harmonics).
    pub fn notch(sample_rate: f64, freq_hz: f64, q: f64) -> Self {
        let sr = sanitize_param_f64(sample_rate, 48_000.0, 100.0, 768_000.0);
        if !freq_hz.is_finite() || freq_hz <= 5.0 || freq_hz >= sr * 0.49 {
            return Self::new(1.0, 0.0, 0.0, 0.0, 0.0);
        }
        let clean_q = sanitize_param_f64(q, 10.0, 0.1, 200.0);
        let w0 = 2.0 * PI * freq_hz / sr;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * clean_q);

        let b0 = 1.0;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        Self::new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    /// Second-order Butterworth High-Pass Filter.
    pub fn highpass(sample_rate: f64, freq_hz: f64, q: f64) -> Self {
        let sr = sanitize_param_f64(sample_rate, 48_000.0, 100.0, 768_000.0);
        let f0 = sanitize_param_f64(freq_hz, 80.0, 5.0, (sr * 0.48).max(5.0));
        let clean_q = sanitize_param_f64(q, 0.707, 0.1, 50.0);
        let w0 = 2.0 * PI * f0 / sr;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * clean_q);

        let b0 = (1.0 + cos_w0) * 0.5;
        let b1 = -(1.0 + cos_w0);
        let b2 = (1.0 + cos_w0) * 0.5;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        Self::new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    /// Audio EQ Cookbook band-pass with a constant 0 dB peak, so its gain is at most 1 everywhere.
    pub fn bandpass(sample_rate: f64, freq_hz: f64, q: f64) -> Self {
        let sr = sanitize_param_f64(sample_rate, 48_000.0, 100.0, 768_000.0);
        let f0 = sanitize_param_f64(freq_hz, 1000.0, 20.0, (sr * 0.48).max(20.0));
        let clean_q = sanitize_param_f64(q, 0.707, 0.1, 50.0);
        let w0 = 2.0 * PI * f0 / sr;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * clean_q);

        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        Self::new(alpha / a0, 0.0, -alpha / a0, a1 / a0, a2 / a0)
    }

    /// Second-order Butterworth Low-Pass Filter.
    pub fn lowpass(sample_rate: f64, freq_hz: f64, q: f64) -> Self {
        let sr = sanitize_param_f64(sample_rate, 48_000.0, 100.0, 768_000.0);
        let f0 = sanitize_param_f64(freq_hz, 1000.0, 20.0, (sr * 0.48).max(20.0));
        let clean_q = sanitize_param_f64(q, 0.707, 0.1, 50.0);
        let w0 = 2.0 * PI * f0 / sr;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * clean_q);

        let b0 = (1.0 - cos_w0) * 0.5;
        let b1 = 1.0 - cos_w0;
        let b2 = (1.0 - cos_w0) * 0.5;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        Self::new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    /// ITU-R BS.1770-4 Stage 1 K-weighting high-shelf pre-filter (+4 dB at ~1681.97 Hz).
    pub fn k_weight_pre_shelf(sample_rate: f64) -> Self {
        let sr = sanitize_param_f64(sample_rate, 48_000.0, 8_000.0, 768_000.0);
        let db = 3.999_843_853_973_347_f64;
        let f0 = 1_681.974_450_955_533_f64.min(sr * 0.45);
        let q = 0.707_175_236_955_419_6_f64;
        let k = (PI * f0 / sr).tan();
        let vh = 10.0_f64.powf(db / 20.0);
        let vb = vh.powf(0.499_666_774_154_541_6);
        let a0 = 1.0 + k / q + k * k;
        let b0 = (vh + vb * k / q + k * k) / a0;
        let b1 = 2.0 * (k * k - vh) / a0;
        let b2 = (vh - vb * k / q + k * k) / a0;
        let a1 = 2.0 * (k * k - 1.0) / a0;
        let a2 = (1.0 - k / q + k * k) / a0;
        Self::new(b0, b1, b2, a1, a2)
    }

    /// ITU-R BS.1770-4 Stage 2 RLB high-pass filter (~38.135 Hz).
    pub fn k_weight_rlb_highpass(sample_rate: f64) -> Self {
        let sr = sanitize_param_f64(sample_rate, 48_000.0, 8_000.0, 768_000.0);
        let f0 = 38.135_470_876_024_44_f64.min(sr * 0.45);
        let q = 0.500_327_037_323_877_3_f64;
        let k = (PI * f0 / sr).tan();
        let a0 = 1.0 + k / q + k * k;
        let b0 = 1.0 / a0;
        let b1 = -2.0 / a0;
        let b2 = 1.0 / a0;
        let a1 = 2.0 * (k * k - 1.0) / a0;
        let a2 = (1.0 - k / q + k * k) / a0;
        Self::new(b0, b1, b2, a1, a2)
    }
}

/// Andy Simper (Cytomic) Topology-Preserving Transform (TPT) Trapezoidal State-Variable Filter (SVF)
/// (`Surge XT / sst-filters` & `ZLEqualizer` standard).
///
/// Unconditionally stable under fast audio-rate cutoff ($f_0$) and resonance ($Q$) modulation
/// without Direct-Form II Biquad state transients ("zipper thumps") or low-frequency float noise.
#[derive(Debug, Clone, Copy, Default)]
pub struct CytomicSvf {
    pub a1: f64,
    pub a2: f64,
    pub a3: f64,
    pub m0: f64,
    pub m1: f64,
    pub m2: f64,
    ic1eq: f64,
    ic2eq: f64,
}

impl CytomicSvf {
    #[inline]
    pub fn reset(&mut self) {
        self.ic1eq = 0.0;
        self.ic2eq = 0.0;
    }

    #[inline]
    fn set_tpt_common(&mut self, sample_rate: f64, freq_hz: f64, q: f64) -> f64 {
        let sr = sanitize_param_f64(sample_rate, 48_000.0, 100.0, 768_000.0);
        let f0 = sanitize_param_f64(freq_hz, 1000.0, 5.0, (sr * 0.48).max(5.0));
        let clean_q = sanitize_param_f64(q, std::f64::consts::FRAC_1_SQRT_2, 0.1, 50.0);
        let g = (PI * f0 / sr).tan();
        let k = 1.0 / clean_q;
        self.a1 = 1.0 / (1.0 + g * (g + k));
        self.a2 = g * self.a1;
        self.a3 = g * self.a2;
        k
    }

    /// Configures the TPT SVF as a Low-Pass filter while preserving integrator state (`ic1eq`, `ic2eq`).
    #[inline]
    pub fn set_lowpass(&mut self, sample_rate: f64, freq_hz: f64, q: f64) {
        self.set_tpt_common(sample_rate, freq_hz, q);
        self.m0 = 0.0;
        self.m1 = 0.0;
        self.m2 = 1.0;
    }

    /// Configures the TPT SVF as a High-Pass filter while preserving integrator state.
    #[inline]
    pub fn set_highpass(&mut self, sample_rate: f64, freq_hz: f64, q: f64) {
        let k = self.set_tpt_common(sample_rate, freq_hz, q);
        self.m0 = 1.0;
        self.m1 = -k;
        self.m2 = -1.0;
    }

    /// Configures the TPT SVF as a Peaking Bell EQ while preserving integrator state (ideal for Dynamic EQ).
    #[inline]
    pub fn set_bell(&mut self, sample_rate: f64, freq_hz: f64, q: f64, gain_db: f64) {
        let sr = sanitize_param_f64(sample_rate, 48_000.0, 100.0, 768_000.0);
        let f0 = sanitize_param_f64(freq_hz, 1000.0, 5.0, (sr * 0.48).max(5.0));
        let clean_q = sanitize_param_f64(q, 1.0, 0.1, 50.0);
        let clean_db = sanitize_param_f64(gain_db, 0.0, -36.0, 36.0);
        let a = 10.0_f64.powf(clean_db / 40.0);
        let g = (PI * f0 / sr).tan();
        let k = 1.0 / (clean_q * a);
        self.a1 = 1.0 / (1.0 + g * (g + k));
        self.a2 = g * self.a1;
        self.a3 = g * self.a2;
        self.m0 = 1.0;
        self.m1 = k * (a * a - 1.0);
        self.m2 = 0.0;
    }

    pub fn lowpass(sample_rate: f64, freq_hz: f64, q: f64) -> Self {
        let mut s = Self::default();
        s.set_lowpass(sample_rate, freq_hz, q);
        s
    }

    pub fn highpass(sample_rate: f64, freq_hz: f64, q: f64) -> Self {
        let mut s = Self::default();
        s.set_highpass(sample_rate, freq_hz, q);
        s
    }

    pub fn bell(sample_rate: f64, freq_hz: f64, q: f64, gain_db: f64) -> Self {
        let mut s = Self::default();
        s.set_bell(sample_rate, freq_hz, q, gain_db);
        s
    }

    /// Processes a single sample and returns `(lowpass, bandpass, highpass)` simultaneously
    /// from the same trapezoidal integrator state.
    #[inline]
    pub fn process_multimode(&mut self, x_raw: f64) -> (f64, f64, f64) {
        let v0 = sanitize_sample_f64(x_raw);
        let v3 = v0 - self.ic2eq;
        let v1 = self.a1 * self.ic1eq + self.a2 * v3;
        let v2 = self.ic2eq + self.a2 * self.ic1eq + self.a3 * v3;
        if !v1.is_finite() || !v2.is_finite() {
            self.ic1eq = 0.0;
            self.ic2eq = 0.0;
            return (0.0, 0.0, 0.0);
        }
        self.ic1eq = flush_denormal_f64(2.0 * v1 - self.ic1eq);
        self.ic2eq = flush_denormal_f64(2.0 * v2 - self.ic2eq);
        let lp = flush_denormal_f64(v2);
        let bp = flush_denormal_f64(v1);
        let hp = flush_denormal_f64(v0 - v1 - v2);
        (lp, bp, hp)
    }

    /// Processes a single sample using the configured `m0, m1, m2` topology output weights.
    #[inline]
    pub fn process_sample(&mut self, x_raw: f64) -> f64 {
        let v0 = sanitize_sample_f64(x_raw);
        let v3 = v0 - self.ic2eq;
        let v1 = self.a1 * self.ic1eq + self.a2 * v3;
        let v2 = self.ic2eq + self.a2 * self.ic1eq + self.a3 * v3;
        if !v1.is_finite() || !v2.is_finite() {
            self.ic1eq = 0.0;
            self.ic2eq = 0.0;
            return 0.0;
        }
        self.ic1eq = flush_denormal_f64(2.0 * v1 - self.ic1eq);
        self.ic2eq = flush_denormal_f64(2.0 * v2 - self.ic2eq);
        flush_denormal_f64(self.m0 * v0 + self.m1 * v1 + self.m2 * v2)
    }
}

/// ITU-R BS.1770-4 Annex 2 4x oversampling interpolation filter: 48-tap FIR split into
/// 4 polyphase sub-filters of 12 taps each.
#[rustfmt::skip]
pub const TRUE_PEAK_PHASES: [[f64; 12]; 4] = [
    [
        0.001_708_984_375_0, 0.010_986_328_125_0, -0.019_653_320_312_5, 0.033_203_125_000_0,
        -0.059_448_242_187_5, 0.137_329_101_562_5, 0.972_167_968_750_0, -0.102_294_921_875_0,
        0.047_607_421_875_0, -0.026_611_328_125_0, 0.014_892_578_125_0, -0.008_300_781_250_0,
    ],
    [
        -0.029_174_804_687_5, 0.029_296_875_000_0, -0.051_757_812_500_0, 0.089_111_328_125_0,
        -0.166_503_906_250_0, 0.465_087_890_625_0, 0.779_785_156_250_0, -0.200_317_382_812_5,
        0.101_562_500_000_0, -0.058_227_539_062_5, 0.033_081_054_687_5, -0.018_920_898_437_5,
    ],
    [
        -0.018_920_898_437_5, 0.033_081_054_687_5, -0.058_227_539_062_5, 0.101_562_500_000_0,
        -0.200_317_382_812_5, 0.779_785_156_250_0, 0.465_087_890_625_0, -0.166_503_906_250_0,
        0.089_111_328_125_0, -0.051_757_812_500_0, 0.029_296_875_000_0, -0.029_174_804_687_5,
    ],
    [
        -0.008_300_781_250_0, 0.014_892_578_125_0, -0.026_611_328_125_0, 0.047_607_421_875_0,
        -0.102_294_921_875_0, 0.972_167_968_750_0, 0.137_329_101_562_5, -0.059_448_242_187_5,
        0.033_203_125_000_0, -0.019_653_320_312_5, 0.010_986_328_125_0, 0.001_708_984_375_0,
    ],
];

/// Group delay of the Annex 2 interpolation filter, in input samples: the linear-phase
/// 48-tap prototype is centered on upsampled tap 23.5, i.e. 23.5 / 4 input samples.
const TRUE_PEAK_FILTER_DELAY: f64 = 23.5 / 4.0;

/// Computes ITU-R BS.1770-4 Annex 2 4x oversampled Inter-Sample True Peak (`dBTP`)
/// and exact timestamp in seconds using a zero-allocation stack chunk buffer.
pub fn compute_true_peak(buffer: &AudioBuffer) -> (f64, f64, f64) {
    const TAPS: usize = 12;
    const CHUNK: usize = 512;
    // Tap-major, reversed coefficients: `taps[k][p]` multiplies the k-th sample of the
    // oldest-first window `[x[i - 11], ..., x[i]]` for output phase `p`.
    let mut taps = [[0.0_f64; 4]; TAPS];
    for (p, coeffs) in TRUE_PEAK_PHASES.iter().enumerate() {
        for (k, &c) in coeffs.iter().enumerate() {
            taps[TAPS - 1 - k][p] = c;
        }
    }

    let mut max_sample_abs = 1e-12_f64;
    let mut max_true_abs = 1e-12_f64;
    let mut peak_pos = 0.0_f64;
    let mut chunk = [0.0_f64; CHUNK + TAPS - 1];

    for ch in &buffer.channels {
        let n = ch.len();
        let last = n.saturating_sub(1) as f64;
        chunk[..TAPS - 1].fill(0.0);
        let total_windows = n + TAPS - 1;
        let mut base = 0_usize;
        while base < total_windows {
            let len = (total_windows - base).min(CHUNK);
            for (k, dst) in chunk[TAPS - 1..TAPS - 1 + len].iter_mut().enumerate() {
                let src_idx = base + k;
                *dst = if src_idx < n {
                    sanitize_sample_f64(ch[src_idx])
                } else {
                    0.0
                };
            }
            for k in 0..len {
                let i = base + k;
                let window = &chunk[k..k + TAPS];
                if i < n {
                    let s_abs = window[TAPS - 1].abs();
                    if s_abs > max_sample_abs {
                        max_sample_abs = s_abs;
                    }
                    if s_abs > max_true_abs {
                        max_true_abs = s_abs;
                        peak_pos = i as f64;
                    }
                }
                let mut acc = [0.0_f64; 4];
                for (tap, &x) in taps.iter().zip(window) {
                    for (a, &c) in acc.iter_mut().zip(tap) {
                        *a += c * x;
                    }
                }
                if acc.iter().all(|a| a.abs() <= max_true_abs) {
                    continue;
                }
                for (p, a) in acc.iter().enumerate() {
                    let interp = a.abs();
                    if interp > max_true_abs {
                        max_true_abs = interp;
                        // Output phase `p` of input step `i` sits at input time `i + p/4 - delay`.
                        peak_pos =
                            (i as f64 + p as f64 * 0.25 - TRUE_PEAK_FILTER_DELAY).clamp(0.0, last);
                    }
                }
            }
            chunk.copy_within(len..len + TAPS - 1, 0);
            base += len;
        }
    }

    let sample_peak_dbfs = 20.0 * max_sample_abs.log10();
    let true_peak_dbtp = 20.0 * max_true_abs.log10();
    let timestamp = peak_pos / f64::from(buffer.sample_rate.max(1));
    (sample_peak_dbfs, true_peak_dbtp, timestamp)
}

/// Band levels reported for audio whose RMS is at or below [`SILENCE_RMS_DBFS`].
pub const SILENT_BAND_DB: f64 = -120.0;

/// RMS level (dBFS, DC removed) at or below which `audit_audio` treats the input as silence.
pub const SILENCE_RMS_DBFS: f64 = -100.0;

/// Multi-channel speaker layout for ITU-R BS.1770-5 loudness weighting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ChannelLayout {
    Mono,
    Stereo,
    Surround5_1,
    Surround7_1,
    Atmos7_1_4,
    Custom(usize),
}

/// Returns the ITU-R BS.1770-5 Annex 3 speaker-position linear power weight `g_i`
/// from spherical speaker coordinates (`azimuth_deg`, `elevation_deg`, `is_lfe`).
///
/// - LFE channels return `0.0` (excluded).
/// - Horizontal/mid-plane surrounds (`|el| < 30 deg` and `60 <= |az| <= 120 deg`) return `1.41`.
/// - Front (`|az| < 60 deg`), rear (`|az| > 120 deg`), and elevated (`|el| >= 30 deg`) speakers return `1.0`.
pub fn bs1770_5_position_weight(azimuth_deg: f64, elevation_deg: f64, is_lfe: bool) -> f64 {
    if is_lfe || !azimuth_deg.is_finite() || !elevation_deg.is_finite() {
        return 0.0;
    }
    if elevation_deg.abs() < 30.0 && (60.0..=120.0).contains(&azimuth_deg.abs()) {
        1.41
    } else {
        1.0
    }
}

impl ChannelLayout {
    /// Infers the standard SMPTE/Atmos layout from the interleaved channel count.
    pub fn infer(num_channels: usize) -> Self {
        match num_channels {
            1 => Self::Mono,
            2 => Self::Stereo,
            6 => Self::Surround5_1,
            8 => Self::Surround7_1,
            12 => Self::Atmos7_1_4,
            n => Self::Custom(n),
        }
    }

    /// Returns ITU-R BS.1770-5 Annex 3 linear power channel weights `g_i` for `num_channels`.
    ///
    /// Note: For 7.1 and 7.1.4 (SMPTE L, R, C, LFE, Lss, Rss, Lrs, Rrs[, Ltf, Rtf, Ltr, Rtr]),
    /// side surrounds (`|az| = 90 deg`, `|el| < 30 deg`) receive `1.41`, rear surrounds
    /// (`|az| = 135 deg`) and heights (`|el| >= 30 deg`) receive `1.0`, and LFE receives `0.0`.
    pub fn weights(self, num_channels: usize) -> Vec<f64> {
        match self {
            Self::Mono => vec![1.0],
            Self::Stereo => vec![1.0, 1.0],
            Self::Surround5_1 => vec![1.0, 1.0, 1.0, 0.0, 1.41, 1.41],
            Self::Surround7_1 => vec![1.0, 1.0, 1.0, 0.0, 1.41, 1.41, 1.0, 1.0],
            Self::Atmos7_1_4 => vec![1.0, 1.0, 1.0, 0.0, 1.41, 1.41, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            Self::Custom(_) => vec![1.0; num_channels],
        }
    }
}

/// Returns ITU-R BS.1770-5 linear power channel weights (`LFE = 0.0`, `surround = 1.41`).
pub fn bs1770_5_channel_weights(num_channels: usize) -> Vec<f64> {
    ChannelLayout::infer(num_channels).weights(num_channels)
}

/// Computes 100 ms K-weighted sub-block energies in a single streaming pass per channel
/// using explicit BS.1770-5 linear channel weights `g_i`.
fn compute_k_weighted_sub_energies_with_weights(
    channels: &[Vec<f64>],
    sample_rate: u32,
    weights: &[f64],
) -> Vec<f64> {
    let sr = f64::from(sample_rate.max(1));
    let frames = channels.first().map_or(0, Vec::len);
    let sub_len = (sr * 0.1).round().max(1.0) as usize;
    let num_subs = frames / sub_len;
    if num_subs == 0 {
        return Vec::new();
    }

    let inv_sub_len = 1.0 / (sub_len as f64);
    let mut sub_energies = vec![0.0_f64; num_subs];
    for (ch_idx, ch) in channels.iter().enumerate() {
        let g = weights.get(ch_idx).copied().unwrap_or(1.0);
        if !g.is_finite() || g <= 0.0 {
            continue;
        }
        let mut shelf = Biquad::k_weight_pre_shelf(sr);
        let mut rlb = Biquad::k_weight_rlb_highpass(sr);
        let usable = ch.len().min(num_subs * sub_len);
        for (s, slot) in sub_energies.iter_mut().enumerate() {
            let start = s * sub_len;
            let end = (start + sub_len).min(usable);
            if start >= end {
                break;
            }
            let mut ch_sq = 0.0;
            for blk in ch[start..end].chunks(32) {
                for &x in blk {
                    let v = rlb.step_raw(shelf.step_raw(sanitize_sample_f64(x)));
                    ch_sq += v * v;
                }
                shelf.flush_state();
                rlb.flush_state();
            }
            *slot += g * ch_sq * inv_sub_len;
        }
    }
    sub_energies
}

/// Computes 100 ms K-weighted sub-block energies using inferred ITU-R BS.1770-5 channel weights.
fn compute_k_weighted_sub_energies(channels: &[Vec<f64>], sample_rate: u32) -> Vec<f64> {
    let weights = bs1770_5_channel_weights(channels.len());
    compute_k_weighted_sub_energies_with_weights(channels, sample_rate, &weights)
}

/// Computes `(integrated_lkfs, short_term_max_lkfs, momentary_max_lkfs, lra_lu)` using ITU-R BS.1770-5 weights.
pub fn compute_bs1770_5_loudness(
    buffer: &AudioBuffer,
    layout: Option<ChannelLayout>,
) -> (f64, f64, f64, f64) {
    let resolved = layout.unwrap_or_else(|| ChannelLayout::infer(buffer.num_channels()));
    let weights = resolved.weights(buffer.num_channels());
    compute_bs1770_5_loudness_with_weights(buffer, &weights)
}

/// Computes `(integrated_lkfs, short_term_max_lkfs, momentary_max_lkfs, lra_lu)` using explicit per-channel BS.1770-5 weights `g_i`.
pub fn compute_bs1770_5_loudness_with_weights(
    buffer: &AudioBuffer,
    weights: &[f64],
) -> (f64, f64, f64, f64) {
    let sub_energies =
        compute_k_weighted_sub_energies_with_weights(&buffer.channels, buffer.sample_rate, weights);
    compute_loudness_metrics_from_subs(&sub_energies)
}

/// Computes ITU-R BS.1770-4 two-stage gated integrated loudness (LUFS) directly from channels.
fn compute_integrated_lufs(channels: &[Vec<f64>], sample_rate: u32) -> f64 {
    let sub_energies = compute_k_weighted_sub_energies(channels, sample_rate);
    let mut momentary_energies = Vec::new();
    if sub_energies.len() >= 4 {
        momentary_energies.reserve(sub_energies.len() - 3);
        for w in sub_energies.windows(4) {
            momentary_energies.push(w.iter().sum::<f64>() * 0.25);
        }
    } else if !sub_energies.is_empty() {
        momentary_energies.push(sub_energies.iter().sum::<f64>() / (sub_energies.len() as f64));
    }

    let abs_gate_e = 10.0_f64.powf((-70.0 + 0.691) / 10.0);
    let gated_abs: Vec<f64> = momentary_energies
        .into_iter()
        .filter(|&e| e > abs_gate_e)
        .collect();

    compute_relative_gated_lufs(&gated_abs)
}

fn compute_relative_gated_lufs(gated_abs: &[f64]) -> f64 {
    if gated_abs.is_empty() {
        return -70.0;
    }
    let mean_abs = gated_abs.iter().sum::<f64>() / (gated_abs.len() as f64);
    let rel_gate_lufs = -0.691 + 10.0 * mean_abs.log10() - 10.0;
    let rel_gate_e = 10.0_f64.powf((rel_gate_lufs + 0.691) / 10.0);
    let mut sum_rel = 0.0_f64;
    let mut count_rel = 0_usize;
    for &e in gated_abs {
        if e > rel_gate_e {
            sum_rel += e;
            count_rel += 1;
        }
    }
    if count_rel == 0 {
        -70.0
    } else {
        let mean_rel = sum_rel / (count_rel as f64);
        -0.691 + 10.0 * mean_rel.log10()
    }
}

fn compute_short_term_and_lra(
    sub_energies: &[f64],
    integrated_lufs: f64,
    momentary_max_lufs: f64,
) -> (f64, f64) {
    if sub_energies.len() < 30 {
        return (momentary_max_lufs, 0.0);
    }

    let mut short_term_lufs = Vec::with_capacity(sub_energies.len() - 29);
    let mut short_term_max_lufs = -70.0_f64;
    for w in sub_energies.windows(30) {
        let mean_e = w.iter().sum::<f64>() / 30.0;
        if mean_e > 1e-12 {
            let lufs = -0.691 + 10.0 * mean_e.log10();
            short_term_lufs.push(lufs);
            if lufs > short_term_max_lufs {
                short_term_max_lufs = lufs;
            }
        }
    }

    let loudness_range_lu = if short_term_lufs.len() >= 4 {
        let mut st: Vec<f64> = short_term_lufs
            .into_iter()
            .filter(|&l| l > -70.0 && l >= integrated_lufs - 20.0)
            .collect();
        st.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if st.len() >= 2 {
            let p10 = st[(st.len() - 1) * 10 / 100];
            let p95 = st[(st.len() - 1) * 95 / 100];
            (p95 - p10).max(0.0)
        } else {
            0.0
        }
    } else {
        0.0
    };

    (short_term_max_lufs, loudness_range_lu)
}

/// Computes Momentary, Short-Term, Integrated LUFS and LRA from 100 ms K-weighted sub-block energies.
fn compute_loudness_metrics_from_subs(sub_energies: &[f64]) -> (f64, f64, f64, f64) {
    let abs_gate_e = 10.0_f64.powf((-70.0 + 0.691) / 10.0);
    let mut gated_abs = Vec::new();
    let mut momentary_max_lufs = -70.0_f64;
    if sub_energies.len() >= 4 {
        gated_abs.reserve(sub_energies.len() - 3);
        for w in sub_energies.windows(4) {
            let mean_e = w.iter().sum::<f64>() * 0.25;
            if mean_e > abs_gate_e {
                gated_abs.push(mean_e);
            }
            if mean_e > 1e-12 {
                momentary_max_lufs = momentary_max_lufs.max(-0.691 + 10.0 * mean_e.log10());
            }
        }
    } else if !sub_energies.is_empty() {
        let mean_e = sub_energies.iter().sum::<f64>() / (sub_energies.len() as f64);
        if mean_e > abs_gate_e {
            gated_abs.push(mean_e);
        }
        if mean_e > 1e-12 {
            momentary_max_lufs = -0.691 + 10.0 * mean_e.log10();
        }
    }

    let integrated_lufs = compute_relative_gated_lufs(&gated_abs);
    let (short_term_max_lufs, loudness_range_lu) =
        compute_short_term_and_lra(sub_energies, integrated_lufs, momentary_max_lufs);
    (
        integrated_lufs,
        short_term_max_lufs,
        momentary_max_lufs,
        loudness_range_lu,
    )
}

/// Computes Momentary, Short-Term, Integrated LUFS and LRA from 100 ms K-weighted blocks.
fn compute_loudness_metrics(channels: &[Vec<f64>], sample_rate: u32) -> (f64, f64, f64, f64) {
    let sub_energies = compute_k_weighted_sub_energies(channels, sample_rate);
    compute_loudness_metrics_from_subs(&sub_energies)
}

/// Maximum bin-over-neighborhood prominence (dB) across `b_lo..=b_hi` (excluding 3-bin main lobe).
fn max_local_excess_db(mag_sq: &[f64], b_lo: usize, b_hi: usize, ref_floor: f64) -> f64 {
    if b_hi <= b_lo + 8 || b_hi >= mag_sq.len() {
        return 0.0;
    }
    let mut max_ratio_sq = 1.0_f64;
    for b in (b_lo + 4)..=(b_hi.saturating_sub(4)) {
        let outer_mean = (mag_sq[b - 4]
            + mag_sq[b - 3]
            + mag_sq[b - 2]
            + mag_sq[b + 2]
            + mag_sq[b + 3]
            + mag_sq[b + 4])
            / 6.0;
        let ref_env = outer_mean.max(ref_floor).max(1e-18);
        let ratio_sq = mag_sq[b] / ref_env;
        if ratio_sq > max_ratio_sq {
            max_ratio_sq = ratio_sq;
        }
    }
    10.0 * max_ratio_sq.log10()
}

fn percentile_sorted(vals: &mut [f64], pct: f64) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = (((vals.len() - 1) as f64) * pct.clamp(0.0, 1.0)).round() as usize;
    vals[idx.min(vals.len() - 1)]
}

#[inline]
fn multi_frame_energy_and_peak(channels: &[Vec<f64>], i: usize, inv_ch: f64) -> (f64, f64) {
    let mut sq_sum = 0.0_f64;
    let mut peak = 0.0_f64;
    for ch in channels {
        let x = sanitize_sample_f64(ch[i]);
        sq_sum += x * x;
        peak = peak.max(x.abs());
    }
    (sq_sum * inv_ch, peak)
}

fn collect_stereo_half_blocks(
    l: &[f64],
    r: &[f64],
    frames: usize,
    half_blk: usize,
    num_halves: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut half_peak = Vec::with_capacity(num_halves);
    let mut half_prefix = Vec::with_capacity(num_halves + 1);
    half_prefix.push(0.0_f64);
    let mut acc_sq = 0.0_f64;
    for h in 0..num_halves {
        let s = h * half_blk;
        let e = (s + half_blk).min(frames);
        let mut sq = 0.0_f64;
        let mut pk = 0.0_f64;
        for (&lv_raw, &rv_raw) in l[s..e].iter().zip(&r[s..e]) {
            let lx = sanitize_sample_f64(lv_raw);
            let rx = sanitize_sample_f64(rv_raw);
            sq += 0.5 * (lx * lx + rx * rx);
            pk = pk.max(lx.abs().max(rx.abs()));
        }
        acc_sq += sq;
        half_peak.push(pk);
        half_prefix.push(acc_sq);
    }
    (half_peak, half_prefix)
}

fn collect_half_block_stats(
    channels: &[Vec<f64>],
    frames: usize,
    half_blk: usize,
) -> (Vec<f64>, Vec<f64>) {
    let num_halves = frames.div_ceil(half_blk);
    if channels.len() == 2 {
        return collect_stereo_half_blocks(
            &channels[0],
            &channels[1],
            frames,
            half_blk,
            num_halves,
        );
    }
    let inv_ch = 1.0 / (channels.len() as f64);
    let mut half_peak = Vec::with_capacity(num_halves);
    let mut half_prefix = Vec::with_capacity(num_halves + 1);
    half_prefix.push(0.0_f64);
    let mut acc_sq = 0.0_f64;
    for h in 0..num_halves {
        let s = h * half_blk;
        let e = (s + half_blk).min(frames);
        let mut sq = 0.0_f64;
        let mut pk = 0.0_f64;
        for i in s..e {
            let (fe, fp) = multi_frame_energy_and_peak(channels, i, inv_ch);
            sq += fe;
            pk = pk.max(fp);
        }
        acc_sq += sq;
        half_peak.push(pk);
        half_prefix.push(acc_sq);
    }
    (half_peak, half_prefix)
}

/// 95th-percentile 5 ms block peak vs 50 ms local RMS (dB) over active blocks (`> -55 dBFS`).
/// Uses O(frames / 2.5 ms) block summaries so each sample is read and sanitized only once.
fn compute_transient_sharpness_db(channels: &[Vec<f64>], frames: usize, sr: f64) -> f64 {
    if channels.is_empty() || frames < 32 {
        return 0.0;
    }
    let short_blk = ((sr * 0.005).round() as usize).clamp(4, frames.max(4));
    let win_half = ((sr * 0.025).round() as usize).max(short_blk);
    let half_blk = (short_blk / 2).max(1);
    let num_halves = frames.div_ceil(half_blk);
    let (half_peak, half_prefix) = collect_half_block_stats(channels, frames, half_blk);

    let win_halves = ((win_half + half_blk / 2) / half_blk).max(1);
    let mut ratios_db = Vec::with_capacity(num_halves.div_ceil(2));
    for b in 0..num_halves.div_ceil(2) {
        let h0 = 2 * b;
        let h1 = (h0 + 1).min(num_halves - 1);
        let blk_peak = half_peak[h0].max(half_peak[h1]);
        let w_lo_h = (h0 + 1).saturating_sub(win_halves);
        let w_hi_h = (h0 + 1 + win_halves).min(num_halves);
        let w_samples = (w_hi_h * half_blk)
            .min(frames)
            .saturating_sub(w_lo_h * half_blk)
            .max(1);
        let rms_50 = ((half_prefix[w_hi_h] - half_prefix[w_lo_h]) / (w_samples as f64))
            .sqrt()
            .max(1e-12);
        if rms_50 > 0.00178 {
            ratios_db.push((20.0 * (blk_peak / rms_50).log10()).clamp(0.0, 40.0));
        }
    }
    percentile_sorted(&mut ratios_db, 0.95)
}

#[derive(Debug, Clone, Copy)]
struct ElectricalCorrelationStats {
    dc_offset: f64,
    rms_db: f64,
    max_ch_rms_db: f64,
    clipped_samples: usize,
    stereo_correlation: f64,
    sub_160hz_correlation: f64,
    high_mid_stereo_correlation: f64,
}

#[inline]
fn norm_correlation(lr: f64, l2: f64, r2: f64) -> f64 {
    if l2 * r2 > 1e-18 {
        (lr / (l2 * r2).sqrt()).clamp(-1.0, 1.0)
    } else {
        1.0
    }
}

#[inline]
fn variance_from_moments(count: usize, sum: f64, sq_sum: f64) -> Option<f64> {
    if count == 0 {
        return Some(0.0);
    }
    let inv = 1.0 / (count as f64);
    let mean = sum * inv;
    let mean_sq = sq_sum * inv;
    if mean * mean <= 0.125 * mean_sq {
        Some((mean_sq - mean * mean).max(0.0))
    } else {
        None
    }
}

fn channel_rms_centered(ch: &[f64], mean: f64) -> f64 {
    if ch.is_empty() {
        return 1e-12;
    }
    (ch.iter()
        .map(|&x| (sanitize_sample_f64(x) - mean).powi(2))
        .sum::<f64>()
        / (ch.len() as f64))
        .sqrt()
        .max(1e-12)
}

#[inline]
fn update_clip_run(run: &mut usize, abs_x: f64) -> usize {
    if abs_x >= 0.999 {
        *run += 1;
        usize::from(*run >= 3)
    } else {
        *run = 0;
        0
    }
}

struct StereoCorrAccum {
    mono_sum: f64,
    mono_sq: f64,
    l_sum: f64,
    r_sum: f64,
    l_sq: f64,
    r_sq: f64,
    sum_lr: f64,
    run_l: usize,
    run_r: usize,
    clipped_samples: usize,
    lp_l: Biquad,
    lp_r: Biquad,
    sub_lr: f64,
    sub_l2: f64,
    sub_r2: f64,
    use_high_mid: bool,
    hp2_l: Biquad,
    hp2_r: Biquad,
    lp12_l: Biquad,
    lp12_r: Biquad,
    hm_lr: f64,
    hm_l2: f64,
    hm_r2: f64,
}

impl StereoCorrAccum {
    fn new(sr: f64) -> Self {
        let hm_hi = 12_000.0_f64.min(sr * 0.47).max(2_200.0);
        Self {
            mono_sum: 0.0,
            mono_sq: 0.0,
            l_sum: 0.0,
            r_sum: 0.0,
            l_sq: 0.0,
            r_sq: 0.0,
            sum_lr: 0.0,
            run_l: 0,
            run_r: 0,
            clipped_samples: 0,
            lp_l: Biquad::lowpass(sr, 160.0, 0.707),
            lp_r: Biquad::lowpass(sr, 160.0, 0.707),
            sub_lr: 0.0,
            sub_l2: 0.0,
            sub_r2: 0.0,
            use_high_mid: sr > 5_000.0,
            hp2_l: Biquad::highpass(sr, 2_000.0, 0.707),
            hp2_r: Biquad::highpass(sr, 2_000.0, 0.707),
            lp12_l: Biquad::lowpass(sr, hm_hi, 0.707),
            lp12_r: Biquad::lowpass(sr, hm_hi, 0.707),
            hm_lr: 0.0,
            hm_l2: 0.0,
            hm_r2: 0.0,
        }
    }

    #[inline(always)]
    fn step_pair(&mut self, lv_raw: f64, rv_raw: f64) {
        let lv = sanitize_sample_f64(lv_raw);
        let rv = sanitize_sample_f64(rv_raw);
        let m = 0.5 * (lv + rv);
        self.mono_sum += m;
        self.mono_sq += m * m;
        self.l_sum += lv;
        self.r_sum += rv;
        self.l_sq += lv * lv;
        self.r_sq += rv * rv;
        self.sum_lr += lv * rv;
        self.clipped_samples += update_clip_run(&mut self.run_l, lv.abs());
        self.clipped_samples += update_clip_run(&mut self.run_r, rv.abs());

        let sl = self.lp_l.step_raw(lv);
        let sr_v = self.lp_r.step_raw(rv);
        self.sub_lr += sl * sr_v;
        self.sub_l2 += sl * sl;
        self.sub_r2 += sr_v * sr_v;

        if self.use_high_mid {
            let hml = self.lp12_l.step_raw(self.hp2_l.step_raw(lv));
            let hmr = self.lp12_r.step_raw(self.hp2_r.step_raw(rv));
            self.hm_lr += hml * hmr;
            self.hm_l2 += hml * hml;
            self.hm_r2 += hmr * hmr;
        }
    }

    #[inline(always)]
    fn flush_filters(&mut self) {
        self.lp_l.flush_state();
        self.lp_r.flush_state();
        if self.use_high_mid {
            self.hp2_l.flush_state();
            self.hp2_r.flush_state();
            self.lp12_l.flush_state();
            self.lp12_r.flush_state();
        }
    }

    fn process_chunks(&mut self, l: &[f64], r: &[f64], frames: usize) {
        for (lc, rc) in l[..frames].chunks(32).zip(r[..frames].chunks(32)) {
            for (&lv_raw, &rv_raw) in lc.iter().zip(rc.iter()) {
                self.step_pair(lv_raw, rv_raw);
            }
            self.flush_filters();
        }
    }

    fn finish(self, l: &[f64], r: &[f64], frames: usize) -> ElectricalCorrelationStats {
        let dc_offset = self.mono_sum / (frames as f64);
        let rms = match variance_from_moments(frames, self.mono_sum, self.mono_sq) {
            Some(v) => v.sqrt().max(1e-12),
            None => stereo_mono_centered_rms(&l[..frames], &r[..frames], dc_offset),
        };
        let l_rms = match variance_from_moments(frames, self.l_sum, self.l_sq) {
            Some(v) => v.sqrt().max(1e-12),
            None => channel_rms_centered(&l[..frames], self.l_sum / (frames as f64)),
        };
        let r_rms = match variance_from_moments(frames, self.r_sum, self.r_sq) {
            Some(v) => v.sqrt().max(1e-12),
            None => channel_rms_centered(&r[..frames], self.r_sum / (frames as f64)),
        };
        ElectricalCorrelationStats {
            dc_offset,
            rms_db: 20.0 * rms.log10(),
            max_ch_rms_db: 20.0 * l_rms.max(r_rms).max(1e-12).log10(),
            clipped_samples: self.clipped_samples,
            stereo_correlation: norm_correlation(self.sum_lr, self.l_sq, self.r_sq),
            sub_160hz_correlation: norm_correlation(self.sub_lr, self.sub_l2, self.sub_r2),
            high_mid_stereo_correlation: if self.use_high_mid {
                norm_correlation(self.hm_lr, self.hm_l2, self.hm_r2)
            } else {
                1.0
            },
        }
    }
}

fn stereo_mono_centered_rms(l: &[f64], r: &[f64], dc_offset: f64) -> f64 {
    if l.is_empty() {
        return 1e-12;
    }
    let centered_sq: f64 = l
        .iter()
        .zip(r.iter())
        .map(|(&lv_raw, &rv_raw)| {
            let m = sanitize_sample_f64(
                0.5 * (sanitize_sample_f64(lv_raw) + sanitize_sample_f64(rv_raw)),
            );
            (m - dc_offset).powi(2)
        })
        .sum();
    (centered_sq / (l.len() as f64)).sqrt().max(1e-12)
}

fn compute_stereo_electrical_and_correlation(
    l: &[f64],
    r: &[f64],
    sr: f64,
) -> ElectricalCorrelationStats {
    let frames = l.len().min(r.len());
    let mut accum = StereoCorrAccum::new(sr);
    accum.process_chunks(l, r, frames);
    accum.finish(l, r, frames)
}

fn compute_multichannel_mono_dc_rms(
    channels: &[Vec<f64>],
    frames: usize,
    inv_ch: f64,
) -> (f64, f64) {
    let mut mono_sum = 0.0_f64;
    let mut mono_sq = 0.0_f64;
    for i in 0..frames {
        let mut s = 0.0_f64;
        for ch in channels {
            s += sanitize_sample_f64(ch[i]) * inv_ch;
        }
        let m = sanitize_sample_f64(s);
        mono_sum += m;
        mono_sq += m * m;
    }
    let dc_offset = mono_sum / (frames as f64);
    let rms = match variance_from_moments(frames, mono_sum, mono_sq) {
        Some(v) => v.sqrt().max(1e-12),
        None => {
            let mut c_sq = 0.0_f64;
            for i in 0..frames {
                let mut s = 0.0_f64;
                for ch in channels {
                    s += sanitize_sample_f64(ch[i]) * inv_ch;
                }
                c_sq += (sanitize_sample_f64(s) - dc_offset).powi(2);
            }
            (c_sq / (frames as f64)).sqrt().max(1e-12)
        }
    };
    (dc_offset, rms)
}

fn compute_multichannel_clips_and_max_rms(channels: &[Vec<f64>]) -> (usize, f64) {
    let mut clipped_samples = 0_usize;
    let mut max_ch_rms = 1e-12_f64;
    for ch in channels {
        let mut run = 0_usize;
        let mut sum = 0.0_f64;
        let mut sq = 0.0_f64;
        for &x_raw in ch {
            let x = sanitize_sample_f64(x_raw);
            sum += x;
            sq += x * x;
            clipped_samples += update_clip_run(&mut run, x.abs());
        }
        let ch_rms = match variance_from_moments(ch.len(), sum, sq) {
            Some(v) => v.sqrt().max(1e-12),
            None => channel_rms_centered(ch, sum / (ch.len() as f64)),
        };
        max_ch_rms = max_ch_rms.max(ch_rms);
    }
    (clipped_samples, max_ch_rms)
}

fn compute_electrical_and_correlation(buffer: &AudioBuffer, sr: f64) -> ElectricalCorrelationStats {
    let frames = buffer.num_frames();
    let ch_count = buffer.num_channels();
    if frames == 0 || ch_count == 0 {
        let floor_db = 20.0 * 1e-12_f64.log10();
        return ElectricalCorrelationStats {
            dc_offset: 0.0,
            rms_db: floor_db,
            max_ch_rms_db: floor_db,
            clipped_samples: 0,
            stereo_correlation: 1.0,
            sub_160hz_correlation: 1.0,
            high_mid_stereo_correlation: 1.0,
        };
    }
    if ch_count == 2 {
        return compute_stereo_electrical_and_correlation(
            &buffer.channels[0],
            &buffer.channels[1],
            sr,
        );
    }

    let inv_ch = 1.0 / (ch_count as f64);
    let (dc_offset, rms) = compute_multichannel_mono_dc_rms(&buffer.channels, frames, inv_ch);
    let (clipped_samples, max_ch_rms) = compute_multichannel_clips_and_max_rms(&buffer.channels);
    let (stereo_correlation, sub_160hz_correlation, high_mid_stereo_correlation) = if ch_count >= 2
    {
        let stats =
            compute_stereo_electrical_and_correlation(&buffer.channels[0], &buffer.channels[1], sr);
        (
            stats.stereo_correlation,
            stats.sub_160hz_correlation,
            stats.high_mid_stereo_correlation,
        )
    } else {
        (1.0, 1.0, 1.0)
    };

    ElectricalCorrelationStats {
        dc_offset,
        rms_db: 20.0 * rms.log10(),
        max_ch_rms_db: 20.0 * max_ch_rms.log10(),
        clipped_samples,
        stereo_correlation,
        sub_160hz_correlation,
        high_mid_stereo_correlation,
    }
}

const AUDIT_FFT_SIZE: usize = 2048;
const AUDIT_HALF_BINS: usize = AUDIT_FFT_SIZE / 2;

struct AuditFft2048 {
    fft: Arc<dyn Fft<f64>>,
    window: Box<[f64; AUDIT_FFT_SIZE]>,
}

static AUDIT_FFT_2048: OnceLock<AuditFft2048> = OnceLock::new();

fn audit_fft_and_window() -> &'static AuditFft2048 {
    AUDIT_FFT_2048.get_or_init(|| {
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(AUDIT_FFT_SIZE);
        let mut win = Box::new([0.0_f64; AUDIT_FFT_SIZE]);
        for (i, w) in win.iter_mut().enumerate() {
            *w = 0.5 * (1.0 - (2.0 * PI * (i as f64) / (AUDIT_FFT_SIZE as f64)).cos());
        }
        AuditFft2048 { fft, window: win }
    })
}

/// Unpacks Hermitian-symmetric power spectra `|M[0..=N/2]|^2` and `|S[0..=N/2]|^2` from a
/// single complex forward FFT of `m[n] + j * s[n]`.
pub(crate) fn unpack_packed_ms_power(
    spectrum: &[Complex<f64>],
    inv_n_sq: f64,
    mid_pow: &mut [f64],
    side_pow: &mut [f64],
) -> (f64, f64) {
    let n = spectrum.len();
    let half = n / 2;

    let z0 = spectrum[0];
    let zh = spectrum[half];
    mid_pow[0] = z0.re * z0.re * inv_n_sq;
    side_pow[0] = z0.im * z0.im * inv_n_sq;

    let mut mid_sum = zh.re * zh.re * inv_n_sq;
    let mut side_sum = zh.im * zh.im * inv_n_sq;
    mid_pow[half] = mid_sum;
    side_pow[half] = side_sum;

    for b in 1..half {
        let zp = spectrum[b];
        let zn = spectrum[n - b];
        let m_re = 0.5 * (zp.re + zn.re);
        let m_im = 0.5 * (zp.im - zn.im);
        let s_re = 0.5 * (zp.im + zn.im);
        let s_im = 0.5 * (zn.re - zp.re);
        let mp = (m_re * m_re + m_im * m_im) * inv_n_sq;
        let sp = (s_re * s_re + s_im * s_im) * inv_n_sq;
        mid_pow[b] = mp;
        side_pow[b] = sp;
        mid_sum += mp;
        side_sum += sp;
    }

    (mid_sum, side_sum)
}

struct SpectralAuditStats {
    avg_mag_sq: Vec<f64>,
    avg_side_sq: Vec<f64>,
    frame_count: usize,
    whistle_db: f64,
    mid_fizz_p95_db: f64,
    side_fizz_p95_db: f64,
}

fn pack_audit_fft_frame(
    channels: &[Vec<f64>],
    pos: usize,
    dc_offset: f64,
    inv_ch: f64,
    window: &[f64],
    buf: &mut [Complex<f64>],
) {
    let ch_count = channels.len();
    if ch_count == 2 {
        let l = &channels[0][pos..pos + AUDIT_FFT_SIZE];
        let r = &channels[1][pos..pos + AUDIT_FFT_SIZE];
        for i in 0..AUDIT_FFT_SIZE {
            let lv = sanitize_sample_f64(l[i]);
            let rv = sanitize_sample_f64(r[i]);
            let w = window[i];
            buf[i] = Complex::new((0.5 * (lv + rv) - dc_offset) * w, (0.5 * (lv - rv)) * w);
        }
    } else if ch_count >= 2 {
        let l = &channels[0][pos..pos + AUDIT_FFT_SIZE];
        let r = &channels[1][pos..pos + AUDIT_FFT_SIZE];
        for i in 0..AUDIT_FFT_SIZE {
            let m_acc: f64 = channels
                .iter()
                .map(|ch| sanitize_sample_f64(ch[pos + i]) * inv_ch)
                .sum();
            let m = sanitize_sample_f64(m_acc) - dc_offset;
            let s = 0.5 * (sanitize_sample_f64(l[i]) - sanitize_sample_f64(r[i]));
            buf[i] = Complex::new(m * window[i], s * window[i]);
        }
    } else {
        let ch0 = &channels[0][pos..pos + AUDIT_FFT_SIZE];
        for i in 0..AUDIT_FFT_SIZE {
            let m = sanitize_sample_f64(ch0[i]) - dc_offset;
            buf[i] = Complex::new(m * window[i], 0.0);
        }
    }
}

struct SpectralAuditAccum {
    avg_mag_sq: Vec<f64>,
    avg_side_sq: Vec<f64>,
    mid_pow: Vec<f64>,
    side_pow: Vec<f64>,
    mid_frame_excess_db: Vec<f64>,
    side_frame_excess_db: Vec<f64>,
    frame_count: usize,
    b_2k: usize,
    b_10k: usize,
    active_frame_floor_sq: f64,
}

impl SpectralAuditAccum {
    fn new(b_2k: usize, b_10k: usize) -> Self {
        Self {
            avg_mag_sq: vec![0.0_f64; AUDIT_HALF_BINS + 1],
            avg_side_sq: vec![0.0_f64; AUDIT_HALF_BINS + 1],
            mid_pow: vec![0.0_f64; AUDIT_HALF_BINS + 1],
            side_pow: vec![0.0_f64; AUDIT_HALF_BINS + 1],
            mid_frame_excess_db: Vec::new(),
            side_frame_excess_db: Vec::new(),
            frame_count: 0,
            b_2k,
            b_10k,
            active_frame_floor_sq: 10.0_f64.powf(-55.0 / 10.0) / (AUDIT_HALF_BINS as f64),
        }
    }

    fn accumulate_mono_power(&mut self, buf: &[Complex<f64>], inv_n_sq: f64) -> f64 {
        let mut m_sum = 0.0_f64;
        for (b, c) in buf[..=AUDIT_HALF_BINS].iter().enumerate() {
            let p = (c.re * c.re + c.im * c.im) * inv_n_sq;
            self.mid_pow[b] = p;
            self.avg_mag_sq[b] += p;
            if b > 0 {
                m_sum += p;
            }
        }
        m_sum
    }

    fn accumulate_frame(&mut self, buf: &[Complex<f64>], ch_count: usize, inv_n_sq: f64) {
        let (mid_sum, side_sum) = if ch_count >= 2 {
            let sums = unpack_packed_ms_power(buf, inv_n_sq, &mut self.mid_pow, &mut self.side_pow);
            for ((am, as_), (&mp, &sp)) in self
                .avg_mag_sq
                .iter_mut()
                .zip(self.avg_side_sq.iter_mut())
                .zip(self.mid_pow.iter().zip(self.side_pow.iter()))
            {
                *am += mp;
                *as_ += sp;
            }
            sums
        } else {
            (self.accumulate_mono_power(buf, inv_n_sq), 0.0)
        };

        let mid_mean = (mid_sum / (AUDIT_HALF_BINS as f64)).max(1e-18);
        if mid_mean > self.active_frame_floor_sq {
            let mid_lm_peak = self.mid_pow[1..self.b_2k]
                .iter()
                .copied()
                .fold(1e-18_f64, f64::max);
            let floor = (mid_mean * 0.5).max(mid_lm_peak * 0.10);
            self.mid_frame_excess_db.push(max_local_excess_db(
                &self.mid_pow,
                self.b_2k,
                self.b_10k,
                floor,
            ));
        }
        if ch_count >= 2 {
            let side_mean = (side_sum / (AUDIT_HALF_BINS as f64)).max(1e-18);
            if side_mean > self.active_frame_floor_sq {
                self.side_frame_excess_db.push(max_local_excess_db(
                    &self.side_pow,
                    self.b_2k,
                    self.b_10k,
                    side_mean * 0.4,
                ));
            }
        }
        self.frame_count += 1;
    }

    fn finalize(mut self, b_6k: usize) -> SpectralAuditStats {
        if self.frame_count > 0 {
            let inv_frames = 1.0 / (self.frame_count as f64);
            for v in &mut self.avg_mag_sq {
                *v *= inv_frames;
            }
            for v in &mut self.avg_side_sq {
                *v *= inv_frames;
            }
        }
        let broadband_bin_mean =
            (self.avg_mag_sq[1..].iter().sum::<f64>() / (AUDIT_HALF_BINS as f64)).max(1e-12);
        let low_mid_peak = self.avg_mag_sq[1..self.b_2k]
            .iter()
            .copied()
            .fold(1e-12_f64, f64::max);
        let whistle_db = max_local_excess_db(
            &self.avg_mag_sq,
            self.b_2k,
            b_6k,
            (broadband_bin_mean * 0.5).max(low_mid_peak * 0.12),
        );
        let mid_fizz_p95_db = percentile_sorted(&mut self.mid_frame_excess_db, 0.95);
        let side_fizz_p95_db = percentile_sorted(&mut self.side_frame_excess_db, 0.95);
        SpectralAuditStats {
            avg_mag_sq: self.avg_mag_sq,
            avg_side_sq: self.avg_side_sq,
            frame_count: self.frame_count,
            whistle_db,
            mid_fizz_p95_db,
            side_fizz_p95_db,
        }
    }
}

fn compute_spectral_audit(buffer: &AudioBuffer, dc_offset: f64, sr: f64) -> SpectralAuditStats {
    let frames = buffer.num_frames();
    let ch_count = buffer.num_channels();
    let bin_hz = sr / (AUDIT_FFT_SIZE as f64);
    let b_2k = ((2000.0 / bin_hz).round() as usize).clamp(2, AUDIT_HALF_BINS - 5);
    let b_6k = ((6000.0 / bin_hz).round() as usize).clamp(b_2k + 1, AUDIT_HALF_BINS - 4);
    let b_10k = ((10_000.0 / bin_hz).round() as usize).clamp(b_6k + 1, AUDIT_HALF_BINS - 2);
    let mut accum = SpectralAuditAccum::new(b_2k, b_10k);

    if frames >= AUDIT_FFT_SIZE && ch_count > 0 {
        let cache = audit_fft_and_window();
        let fft = &cache.fft;
        let window = &cache.window;
        let mut buf = vec![Complex::new(0.0, 0.0); AUDIT_FFT_SIZE];
        let mut scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
        let inv_n_sq = 1.0 / ((AUDIT_FFT_SIZE * AUDIT_FFT_SIZE) as f64);
        let inv_ch = 1.0 / (ch_count as f64);

        let mut pos = 0_usize;
        while pos + AUDIT_FFT_SIZE <= frames {
            pack_audit_fft_frame(
                &buffer.channels,
                pos,
                dc_offset,
                inv_ch,
                window.as_slice(),
                &mut buf,
            );
            fft.process_with_scratch(&mut buf, &mut scratch);
            accum.accumulate_frame(&buf, ch_count, inv_n_sq);
            pos += 1024;
        }
    }

    accum.finalize(b_6k)
}

fn compute_effective_bandwidth_hz(
    avg_mag_sq: &[f64],
    avg_side_sq: &[f64],
    frame_count: usize,
    channel_silent: bool,
    sr: f64,
) -> f64 {
    if channel_silent || frame_count == 0 {
        return 0.0;
    }
    let bin_hz = sr / (AUDIT_FFT_SIZE as f64);
    let b_200 = ((200.0 / bin_hz).round() as usize).clamp(1, AUDIT_HALF_BINS - 4);
    let b_2k = ((2000.0 / bin_hz).round() as usize).clamp(2, AUDIT_HALF_BINS - 5);
    let b_4k = ((4000.0 / bin_hz).round() as usize).clamp(b_2k + 1, AUDIT_HALF_BINS - 3);

    let total_at = |b: usize| avg_mag_sq[b] + avg_side_sq[b];
    let ref_span = (b_4k - b_200 + 1).max(1) as f64;
    let mid_ref = (b_200..=b_4k).map(total_at).sum::<f64>() / ref_span;
    let all_mean = (1..=AUDIT_HALF_BINS).map(total_at).sum::<f64>() / (AUDIT_HALF_BINS as f64);
    let cutoff_thresh = mid_ref.max(all_mean).max(1e-18) * 1e-4; // -40 dB drop
    let mut cutoff_bin = 0_usize;
    for b in (3..AUDIT_HALF_BINS).rev() {
        let local3 = (total_at(b - 1) + total_at(b) + total_at(b + 1)) / 3.0;
        if local3 >= cutoff_thresh {
            cutoff_bin = b + 1;
            break;
        }
    }
    ((cutoff_bin as f64) * bin_hz).clamp(0.0, sr * 0.5)
}

fn push_loudness_and_spectral_recs(
    recommendations: &mut Vec<String>,
    report: &AuditReport,
    whistle_db: f64,
    frames: usize,
    channel_silent: bool,
) {
    let sr = f64::from(report.sample_rate.max(1));
    if report.true_peak_dbtp > -1.0 {
        recommendations.push(format!(
            "True Peak is {:.2} dBTP at {:.2}s (exceeds DistroKid -1.0 dBTP ceiling). Run `forge master` brickwall limiter.",
            report.true_peak_dbtp, report.true_peak_timestamp_secs
        ));
    }
    if report.ai_fizz_index >= 25.0 {
        recommendations.push(format!(
            "Elevated AI/Metallic Resonance detected (Fizz Index {:.1}/100, peak whistle +{:.1} dB in 2-6 kHz). Apply `forge restore` OpenSoothe De-Fizz.",
            report.ai_fizz_index, whistle_db
        ));
    }
    if report.side_fizz_p95_db >= 9.0 && report.channels >= 2 && !channel_silent {
        recommendations.push(format!(
            "High Stereo-Side shimmer detected (Side p95 excess {:.1} dB). Apply `forge restore --mid-side 0.4` to tame Side harshness.",
            report.side_fizz_p95_db
        ));
    }
    if !channel_silent
        && report.channels >= 2
        && report.high_mid_stereo_correlation < 0.25
        && frames >= (sr * 0.25) as usize
    {
        recommendations.push(format!(
            "2-12 kHz stereo correlation is low ({:.2}, diffusion phase wash). Narrow high-band stereo width and de-fizz Side channel.",
            report.high_mid_stereo_correlation
        ));
    }
    if !channel_silent
        && sr >= 44_100.0
        && frames >= (sr as usize)
        && report.effective_bandwidth_hz > 0.0
        && report.effective_bandwidth_hz < 16_500.0
    {
        recommendations.push(format!(
            "Narrow effective bandwidth cutoff at {:.0} Hz (AI codec brickwalling). Run `forge neural-restore` bandwidth extension.",
            report.effective_bandwidth_hz
        ));
    }
}

fn push_electrical_recs(
    recommendations: &mut Vec<String>,
    report: &AuditReport,
    channel_silent: bool,
) {
    if report.sub_160hz_correlation < 0.85 {
        recommendations.push(format!(
            "Sub-160 Hz stereo correlation is {:.2} (phasey low-end). Enable Mid/Side Sub-160 Hz Mono-Maker.",
            report.sub_160hz_correlation
        ));
    }
    if report.hum_50_60hz_db > -12.0 {
        recommendations.push(
            "Elevated 50/60 Hz ground-loop hum energy detected. Enable 10-harmonic hum notch filter."
                .to_string(),
        );
    }
    if report.clipped_samples > 0 {
        recommendations.push(format!(
            "{} clipped sample runs detected. Run cubic-spline de-clipper before mastering.",
            report.clipped_samples
        ));
    }
    if !channel_silent && report.dc_offset.abs() > 0.01 {
        recommendations.push(format!(
            "DC offset is {:+.4} (reduces headroom and causes clicky edits). Enable DC removal in `forge restore`.",
            report.dc_offset
        ));
    }
    if !channel_silent && report.channels >= 2 && report.stereo_correlation < 0.0 {
        recommendations.push(format!(
            "Negative broadband stereo correlation ({:.2}) risks phase cancellation in mono playback. Check stem polarity with `forge phase-align`.",
            report.stereo_correlation
        ));
    }
}

fn push_dynamics_recs(
    recommendations: &mut Vec<String>,
    report: &AuditReport,
    frames: usize,
    mono_silent: bool,
) {
    let sr = f64::from(report.sample_rate.max(1));
    if mono_silent || frames < (sr * 0.5) as usize {
        return;
    }
    if report.crest_factor_db < 4.5 {
        recommendations.push(format!(
            "Low crest factor ({:.1} dB) indicates heavy dynamic squashing. Consider backing off limiter drive or lowering target loudness.",
            report.crest_factor_db
        ));
    }
    if report.transient_sharpness_db > 0.0 && report.transient_sharpness_db < 4.5 {
        recommendations.push(format!(
            "Softened transient punch ({:.1} dB 5ms/50ms ratio). Consider transient restoration or drum layer reinforcement.",
            report.transient_sharpness_db
        ));
    }
}

fn build_audit_recommendations(
    report: &AuditReport,
    whistle_db: f64,
    frames: usize,
    channel_silent: bool,
    mono_silent: bool,
) -> Vec<String> {
    let mut recommendations = Vec::new();
    push_loudness_and_spectral_recs(
        &mut recommendations,
        report,
        whistle_db,
        frames,
        channel_silent,
    );
    push_electrical_recs(&mut recommendations, report, channel_silent);
    push_dynamics_recs(&mut recommendations, report, frames, mono_silent);
    recommendations
}

struct DerivedAuditMetrics {
    hum_50_60hz_db: f64,
    box_mud_200_500hz_db: f64,
    harshness_2_6khz_db: f64,
    sibilance_5_14khz_db: f64,
    effective_bandwidth_hz: f64,
    transient_sharpness_db: f64,
    ai_fizz_index: f64,
    mono_silent: bool,
    channel_silent: bool,
}

fn compute_relative_band_db(
    avg_mag_sq: &[f64],
    bin_hz: f64,
    rms_db: f64,
    mono_silent: bool,
    f_low: f64,
    f_high: f64,
) -> f64 {
    if mono_silent {
        return SILENT_BAND_DB;
    }
    let i0 = ((f_low / bin_hz).floor() as usize).clamp(1, AUDIT_HALF_BINS);
    let i1 = ((f_high / bin_hz).ceil() as usize).clamp(i0, AUDIT_HALF_BINS);
    let sum_e: f64 = avg_mag_sq[i0..=i1].iter().sum();
    10.0 * sum_e.max(1e-15).log10() - rms_db
}

fn compute_ai_fizz_index(
    spec: &SpectralAuditStats,
    elec: &ElectricalCorrelationStats,
    harshness_2_6khz_db: f64,
    ch_count: usize,
    channel_silent: bool,
    mono_silent: bool,
) -> f64 {
    if channel_silent {
        return 0.0;
    }
    let whistle_score = ((spec.whistle_db - 6.0) * 4.5).clamp(0.0, 50.0);
    let harsh_score = ((harshness_2_6khz_db + 8.0) * 3.0).clamp(0.0, 20.0);
    let sub_smear_score = ((0.85 - elec.sub_160hz_correlation) * 30.0).clamp(0.0, 15.0);
    let mid_p95_score = ((spec.mid_fizz_p95_db - 8.5) * 1.5).clamp(0.0, 15.0);
    let side_p95_score = ((spec.side_fizz_p95_db - 7.5) * 2.5).clamp(0.0, 25.0);
    let phase_wash_score = if ch_count >= 2 {
        ((0.25 - elec.high_mid_stereo_correlation) * 25.0).clamp(0.0, 12.0)
    } else {
        0.0
    };
    let mid_component = if mono_silent {
        0.0
    } else {
        whistle_score + harsh_score + sub_smear_score + mid_p95_score
    };
    (mid_component + side_p95_score + phase_wash_score).clamp(0.0, 100.0)
}

fn compute_derived_audit_metrics(
    buffer: &AudioBuffer,
    elec: &ElectricalCorrelationStats,
    spec: &SpectralAuditStats,
    sr: f64,
) -> DerivedAuditMetrics {
    let frames = buffer.num_frames();
    let ch_count = buffer.num_channels();
    let bin_hz = sr / (AUDIT_FFT_SIZE as f64);
    let mono_silent = elec.rms_db <= SILENCE_RMS_DBFS;
    let channel_silent = elec.max_ch_rms_db <= SILENCE_RMS_DBFS;
    let band = |lo, hi| {
        compute_relative_band_db(&spec.avg_mag_sq, bin_hz, elec.rms_db, mono_silent, lo, hi)
    };
    let harshness_2_6khz_db = band(2000.0, 6000.0);
    let effective_bandwidth_hz = compute_effective_bandwidth_hz(
        &spec.avg_mag_sq,
        &spec.avg_side_sq,
        spec.frame_count,
        channel_silent,
        sr,
    );
    let transient_sharpness_db = if channel_silent {
        0.0
    } else {
        compute_transient_sharpness_db(&buffer.channels, frames, sr)
    };
    let ai_fizz_index = compute_ai_fizz_index(
        spec,
        elec,
        harshness_2_6khz_db,
        ch_count,
        channel_silent,
        mono_silent,
    );
    DerivedAuditMetrics {
        hum_50_60hz_db: band(45.0, 65.0),
        box_mud_200_500hz_db: band(200.0, 500.0),
        harshness_2_6khz_db,
        sibilance_5_14khz_db: band(5000.0, 14000.0),
        effective_bandwidth_hz,
        transient_sharpness_db,
        ai_fizz_index,
        mono_silent,
        channel_silent,
    }
}

struct AuditInputs<'a> {
    loudness: (f64, f64, f64, f64),
    peaks: (f64, f64, f64),
    elec: &'a ElectricalCorrelationStats,
    spec: &'a SpectralAuditStats,
    derived: &'a DerivedAuditMetrics,
}

fn assemble_audit_report(buffer: &AudioBuffer, inputs: &AuditInputs<'_>) -> AuditReport {
    let (integrated_lufs, short_term_max_lufs, momentary_max_lufs, loudness_range_lu) =
        inputs.loudness;
    let (sample_peak_dbfs, true_peak_dbtp, true_peak_timestamp_secs) = inputs.peaks;
    let elec = inputs.elec;
    let spec = inputs.spec;
    let derived = inputs.derived;
    let distrokid_ready = (-16.5..=-7.5).contains(&integrated_lufs)
        && true_peak_dbtp <= -0.95
        && elec.clipped_samples == 0;
    AuditReport {
        sample_rate: buffer.sample_rate,
        channels: buffer.num_channels(),
        duration_secs: buffer.duration_secs(),
        integrated_lufs,
        short_term_max_lufs,
        momentary_max_lufs,
        loudness_range_lu,
        sample_peak_dbfs,
        true_peak_dbtp,
        true_peak_timestamp_secs,
        crest_factor_db: (sample_peak_dbfs - elec.rms_db).max(0.0),
        dc_offset: elec.dc_offset,
        clipped_samples: elec.clipped_samples,
        stereo_correlation: elec.stereo_correlation,
        sub_160hz_correlation: elec.sub_160hz_correlation,
        hum_50_60hz_db: derived.hum_50_60hz_db,
        box_mud_200_500hz_db: derived.box_mud_200_500hz_db,
        harshness_2_6khz_db: derived.harshness_2_6khz_db,
        sibilance_5_14khz_db: derived.sibilance_5_14khz_db,
        ai_fizz_index: derived.ai_fizz_index,
        mid_fizz_p95_db: spec.mid_fizz_p95_db,
        side_fizz_p95_db: spec.side_fizz_p95_db,
        effective_bandwidth_hz: derived.effective_bandwidth_hz,
        high_mid_stereo_correlation: elec.high_mid_stereo_correlation,
        transient_sharpness_db: derived.transient_sharpness_db,
        distrokid_ready,
        recommendations: Vec::new(),
    }
}

/// Performs a full ITU-R BS.1770-4 loudness, phase, electrical, and AI-artifact audit.
pub fn audit_audio(buffer: &AudioBuffer) -> AuditReport {
    let sr = f64::from(buffer.sample_rate.max(1));
    let loudness = compute_loudness_metrics(&buffer.channels, buffer.sample_rate);
    let peaks = compute_true_peak(buffer);
    let elec = compute_electrical_and_correlation(buffer, sr);
    let spec = compute_spectral_audit(buffer, elec.dc_offset, sr);
    let derived = compute_derived_audit_metrics(buffer, &elec, &spec, sr);
    let inputs = AuditInputs {
        loudness,
        peaks,
        elec: &elec,
        spec: &spec,
        derived: &derived,
    };
    let mut report = assemble_audit_report(buffer, &inputs);
    report.recommendations = build_audit_recommendations(
        &report,
        spec.whistle_db,
        buffer.num_frames(),
        derived.channel_silent,
        derived.mono_silent,
    );
    report
}

/// Lookahead window of the mastering limiters, in milliseconds. The README and the
/// architecture guide quote this value; `tests/docs_consistency.rs` keeps them in sync.
pub const LIMITER_LOOKAHEAD_MS: f64 = 3.0;

/// Release time of the `normalize_and_limit` gain envelope, in milliseconds.
pub const LIMITER_RELEASE_MS: f64 = 50.0;

/// Frames the offline limiter holds its gain flat on each side of a peak, so the 12-tap
/// interpolation filter around the peak sees a constant gain.
const LIMITER_HOLD_FRAMES: usize = 8;

/// Per-frame true peak across all channels: the larger of `|x|` and the four interpolated phases
/// of the ITU-R BS.1770-4 Annex 2 filter. An interpolated value is charged to both frames it
/// falls between.
fn true_peak_per_frame(channels: &[Vec<f64>], frames: usize) -> Vec<f64> {
    const TAPS: usize = 12;
    // Phase outputs at input step `i` sit at times `i - 5.875 .. i - 5.125`.
    const LAG: usize = 6;
    let mut peaks = vec![0.0_f64; frames];
    for ch in channels {
        let mut hist = [0.0_f64; TAPS];
        for i in 0..frames + TAPS - 1 {
            let x = ch.get(i).copied().unwrap_or(0.0);
            hist.copy_within(0..TAPS - 1, 1);
            hist[0] = x;
            if i < frames {
                peaks[i] = peaks[i].max(x.abs());
            }
            let interp = TRUE_PEAK_PHASES
                .iter()
                .map(|c| c.iter().zip(&hist).map(|(c, h)| c * h).sum::<f64>().abs())
                .fold(0.0_f64, f64::max);
            for frame in [i.checked_sub(LAG), i.checked_sub(LAG - 1)]
                .into_iter()
                .flatten()
            {
                if frame < frames {
                    peaks[frame] = peaks[frame].max(interp);
                }
            }
        }
    }
    peaks
}

/// `out[i] = min(values[i .. i + window])`, via a monotonic deque in O(n).
fn forward_window_min(values: &[f64], window: usize) -> Vec<f64> {
    let mut out = vec![1.0_f64; values.len()];
    let mut deque = std::collections::VecDeque::with_capacity(window + 1);
    for j in (0..values.len()).rev() {
        while deque
            .back()
            .is_some_and(|&k: &usize| values[k] >= values[j])
        {
            deque.pop_back();
        }
        deque.push_back(j);
        while deque.front().is_some_and(|&k| k >= j + window) {
            deque.pop_front();
        }
        out[j] = values[deque[0]];
    }
    out
}

/// Slow program-dependent release engagement attack time constant (seconds).
/// Long enough that isolated transients (<10 ms) recover via the fast 50 ms stage,
/// while sustained loud passages smoothly engage the slower release.
const LIMITER_SLOW_ATTACK_SECS: f64 = 0.080;

/// Slow program-dependent release time constant (seconds) for sustained limiting.
const LIMITER_SLOW_RELEASE_SECS: f64 = 0.250;

/// Maximum closed-loop LUFS convergence passes in `normalize_and_limit`.
const LIMITER_MAX_LUFS_ITERS: usize = 4;

/// Stop iterating once measured post-limiter loudness is within this margin of `target_lufs`.
const LIMITER_LUFS_CONVERGENCE_LU: f64 = 0.15;

/// Mastering limiter release & lookahead topology (inspired by FabFilter Pro-L 2 / Ozone IRC).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum LimiterStyle {
    /// Balanced transparent mastering (default; matches `normalize_and_limit`).
    #[default]
    Transparent,
    /// Shorter lookahead and faster transient recovery for drums, hip-hop, and pop punch.
    Punchy,
    /// Longer lookahead and slower release preserving macro-dynamics for acoustic/jazz/orchestral.
    Dynamic,
    /// Tight lookahead and fast release for club/EDM loudness maximization.
    Aggressive,
}

impl LimiterStyle {
    const fn timing_params(self) -> (f64, f64, f64, f64) {
        // (lookahead_ms, fast_release_ms, slow_attack_s, slow_release_s)
        match self {
            Self::Transparent => (
                LIMITER_LOOKAHEAD_MS,
                LIMITER_RELEASE_MS,
                LIMITER_SLOW_ATTACK_SECS,
                LIMITER_SLOW_RELEASE_SECS,
            ),
            Self::Punchy => (0.8, 22.0, 0.060, 0.140),
            Self::Dynamic => (2.0, 65.0, 0.110, 0.320),
            Self::Aggressive => (0.5, 14.0, 0.045, 0.090),
        }
    }
}

/// Mastering-grade output dithering and psychoacoustic noise shaping mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum DitherMode {
    /// No dithering (keep full 64-bit floating-point precision).
    #[default]
    None,
    /// Flat Triangular Probability Density Function (TPDF) dither at ±1 LSB.
    Tpdf,
    /// TPDF dither with 2nd-order psychoacoustic high-pass error-feedback noise shaping,
    /// pushing quantization noise out of the 1–5 kHz ear-sensitive band toward Nyquist.
    NoiseShapedHighPass,
}

/// Full configuration for [`normalize_and_limit_with_config`].
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct MasteringLimiterConfig {
    /// Target Integrated Loudness in LUFS (`-40.0..=-4.0`).
    pub target_lufs: f64,
    /// Strict 4x oversampled True-Peak ceiling in dBTP (`-24.0..=0.0`).
    pub ceiling_dbtp: f64,
    /// Limiter lookahead and dual-stage release topology.
    pub style: LimiterStyle,
    /// Stereo channel linking (`0.0` = independent dual-mono, `1.0` = fully linked stereo).
    pub channel_link: f64,
    /// Optional mastering dither / noise shaping applied after final true-peak limiting.
    pub dither: DitherMode,
    /// Target integer bit depth for dithering (`8..=24`, typically `16` or `24`).
    pub dither_bits: u16,
}

impl Default for MasteringLimiterConfig {
    fn default() -> Self {
        Self {
            target_lufs: -14.0,
            ceiling_dbtp: -1.0,
            style: LimiterStyle::Transparent,
            channel_link: 1.0,
            dither: DitherMode::None,
            dither_bits: 24,
        }
    }
}

/// Gain envelope for `peaks`: the minimum required gain over a window that reaches
/// `lookahead + LIMITER_HOLD_FRAMES` frames ahead and `LIMITER_HOLD_FRAMES` behind, a
/// program-dependent dual-stage release parameterized by `style`, then a drift-free
/// `lookahead`-frame moving average.
fn limiter_gain_envelope_styled(
    peaks: &[f64],
    ceiling_lin: f64,
    lookahead: usize,
    sample_rate: f64,
    style: LimiterStyle,
) -> Vec<f64> {
    let (_, fast_release_ms, slow_attack_s, slow_release_s) = style.timing_params();
    let required: Vec<f64> = peaks
        .iter()
        .map(|&p| {
            if p > ceiling_lin {
                ceiling_lin / p
            } else {
                1.0
            }
        })
        .collect();
    let ahead_min = forward_window_min(&required, lookahead + 2 * LIMITER_HOLD_FRAMES + 1);

    let fast_release_coeff = (-1.0 / (sample_rate * fast_release_ms / 1000.0)).exp();
    let slow_attack_coeff = (-1.0 / (sample_rate * slow_attack_s)).exp();
    let slow_release_coeff = (-1.0 / (sample_rate * slow_release_s)).exp();

    // Prime the fast release and the moving average as if the first window's gain had been
    // held for a full lookahead before the file starts, so an over in the first frames is fully
    // reduced where it occurs instead of leaving the whole-file trim to catch it. Keep the slow
    // program-dependent stage at unity so a single-sample start spike does not engage 250 ms release.
    let start = ahead_min.first().copied().unwrap_or(1.0);
    let mut fast_released = start;
    let mut slow_env = 1.0_f64;
    let mut ring = vec![start; lookahead];
    let mut sum = start * lookahead as f64;
    let mut gain_env = vec![1.0_f64; peaks.len()];
    for (i, g) in gain_env.iter_mut().enumerate() {
        let held = ahead_min[i.saturating_sub(LIMITER_HOLD_FRAMES)];
        fast_released = if held < fast_released {
            held
        } else {
            (fast_release_coeff * fast_released + (1.0 - fast_release_coeff) * held).min(held)
        };
        slow_env = if held < slow_env {
            slow_attack_coeff * slow_env + (1.0 - slow_attack_coeff) * held
        } else {
            (slow_release_coeff * slow_env + (1.0 - slow_release_coeff) * held).min(held)
        };
        let released = fast_released.min(slow_env);
        let slot = i % lookahead;
        sum += released - ring[slot];
        ring[slot] = released;
        // Periodically resync the running sum to prevent floating-point drift on long tracks.
        if i & 4095 == 4095 {
            sum = ring.iter().sum();
        }
        *g = (sum / lookahead as f64).min(1.0);
    }
    gain_env
}

/// Masters an `AudioBuffer` to a target Integrated LUFS and strict True-Peak ceiling (`-1.0 dBTP`).
///
/// Uses a closed-loop iterative loudness solver around the lookahead true-peak limiter so
/// dynamic or transient-heavy mixes converge onto `target_lufs` instead of falling short after
/// peak limiting, while computing the 4x polyphase true-peak scan only once.
pub fn normalize_and_limit(
    buffer: &AudioBuffer,
    target_lufs: f64,
    ceiling_dbtp: f64,
) -> Result<AudioBuffer> {
    normalize_and_limit_with_config(
        buffer,
        &MasteringLimiterConfig {
            target_lufs,
            ceiling_dbtp,
            ..MasteringLimiterConfig::default()
        },
    )
}

pub(crate) fn validate_lufs_and_ceiling(
    lufs: f64,
    ceiling_dbtp: f64,
    lufs_label: &str,
    ceiling_label: &str,
) -> Result<()> {
    if !lufs.is_finite() || !(-40.0..=-4.0).contains(&lufs) {
        return Err(ForgeError::InvalidParameter(format!(
            "{lufs_label} ({lufs}) must be between -40.0 and -4.0 LUFS"
        )));
    }
    if !ceiling_dbtp.is_finite() || !(-24.0..=0.0).contains(&ceiling_dbtp) {
        return Err(ForgeError::InvalidParameter(format!(
            "{ceiling_label} ({ceiling_dbtp}) must be between -24.0 and 0.0 dBTP"
        )));
    }
    Ok(())
}

/// Masters an `AudioBuffer` with full control over limiter release topology ([`LimiterStyle`]),
/// stereo channel linking (`0.0..=1.0`), and output dithering / psychoacoustic noise shaping
/// ([`DitherMode`]).
pub fn normalize_and_limit_with_config(
    buffer: &AudioBuffer,
    config: &MasteringLimiterConfig,
) -> Result<AudioBuffer> {
    let target_lufs = config.target_lufs;
    let ceiling_dbtp = config.ceiling_dbtp;
    validate_lufs_and_ceiling(target_lufs, ceiling_dbtp, "target_lufs", "ceiling_dbtp")?;
    if config.dither != DitherMode::None && !(8..=24).contains(&config.dither_bits) {
        return Err(ForgeError::InvalidParameter(format!(
            "dither_bits ({}) must be between 8 and 24",
            config.dither_bits
        )));
    }
    // Reserve headroom for TPDF / noise-shaped quantization error so dithering cannot push
    // inter-sample true peaks above the requested `ceiling_dbtp`.
    let dither_headroom_db = match config.dither {
        DitherMode::None => 0.0,
        DitherMode::Tpdf | DitherMode::NoiseShapedHighPass => {
            let full_scale = (1_u64 << (config.dither_bits - 1)) as f64;
            let margin_lsb = if config.dither == DitherMode::NoiseShapedHighPass {
                6.0
            } else {
                2.5
            };
            let ceil_lin = 10.0_f64.powf(ceiling_dbtp / 20.0);
            let safe_lin = (ceil_lin - margin_lsb / full_scale).max(ceil_lin * 0.5);
            (ceiling_dbtp - 20.0 * safe_lin.log10()).max(0.0)
        }
    };
    let clean_ceiling_dbtp = ceiling_dbtp - dither_headroom_db;
    let clean_link = sanitize_param_f64(config.channel_link, 1.0, 0.0, 1.0);
    let frames = buffer.num_frames();
    if frames == 0 {
        return Ok(buffer.clone());
    }

    // Sanitize once so all iterations share clean inputs and one 4x polyphase true-peak pass.
    let sanitized_channels: Vec<Vec<f64>> = buffer
        .channels
        .iter()
        .map(|ch| ch.iter().map(|&s| sanitize_sample_f64(s)).collect())
        .collect();

    let initial_lufs = compute_integrated_lufs(&sanitized_channels, buffer.sample_rate);
    let silent_input = initial_lufs <= -65.0;
    let mut makeup_db = if silent_input {
        0.0
    } else {
        (target_lufs - initial_lufs).clamp(-24.0, 24.0)
    };

    let ceiling_lin = 10.0_f64.powf((clean_ceiling_dbtp - 0.04) / 20.0);
    let sr = f64::from(buffer.sample_rate.max(1));
    let (lookahead_ms, _, _, _) = config.style.timing_params();
    let lookahead = ((sr * lookahead_ms / 1000.0).round() as usize).max(4);
    let base_peaks = true_peak_per_frame(&sanitized_channels, frames);
    let use_unlinked = sanitized_channels.len() > 1 && clean_link < 0.999;
    let per_ch_peaks: Vec<Vec<f64>> = if use_unlinked {
        sanitized_channels
            .iter()
            .map(|ch| true_peak_per_frame(std::slice::from_ref(ch), frames))
            .collect()
    } else {
        Vec::new()
    };

    let mut out = AudioBuffer {
        channels: vec![vec![0.0_f64; frames]; sanitized_channels.len()],
        sample_rate: buffer.sample_rate,
    };

    let unity_makeup_only = (target_lufs - initial_lufs).abs() < 0.05;
    let max_iters = if silent_input || unity_makeup_only {
        1
    } else {
        LIMITER_MAX_LUFS_ITERS
    };
    let mut prev_abs_err = f64::INFINITY;

    for iter in 0..max_iters {
        let makeup_gain = 10.0_f64.powf(makeup_db / 20.0);
        // True-peak FIR is linear, so scaling the signal by `makeup_gain` is equivalent to
        // comparing `base_peaks` against `ceiling_lin / makeup_gain`.
        let scaled_ceiling = ceiling_lin / makeup_gain;
        let linked_env =
            limiter_gain_envelope_styled(&base_peaks, scaled_ceiling, lookahead, sr, config.style);

        if use_unlinked {
            for (ch_idx, (dst_ch, src_ch)) in
                out.channels.iter_mut().zip(&sanitized_channels).enumerate()
            {
                let ch_env = limiter_gain_envelope_styled(
                    &per_ch_peaks[ch_idx],
                    scaled_ceiling,
                    lookahead,
                    sr,
                    config.style,
                );
                for (((dst, &src), &g_link), &g_ch) in dst_ch
                    .iter_mut()
                    .zip(src_ch.iter())
                    .zip(linked_env.iter())
                    .zip(ch_env.iter())
                {
                    let g = (clean_link * g_link + (1.0 - clean_link) * g_ch).min(g_ch);
                    *dst = src * makeup_gain * g;
                }
            }
        } else {
            for (dst_ch, src_ch) in out.channels.iter_mut().zip(&sanitized_channels) {
                for ((dst, &src), &g) in dst_ch.iter_mut().zip(src_ch.iter()).zip(linked_env.iter())
                {
                    *dst = src * makeup_gain * g;
                }
            }
        }

        if iter + 1 >= max_iters {
            break;
        }
        let post_lufs = compute_integrated_lufs(&out.channels, out.sample_rate);
        if post_lufs <= -65.0 {
            break;
        }
        let err = target_lufs - post_lufs;
        if err.abs() <= LIMITER_LUFS_CONVERGENCE_LU {
            break;
        }
        // Stop if heavy limiting has saturated loudness recovery.
        if iter > 0 && err.abs() >= prev_abs_err - 0.05 {
            break;
        }
        prev_abs_err = err.abs();
        let next_makeup_db = (makeup_db + err).clamp(-24.0, 24.0);
        if (next_makeup_db - makeup_db).abs() < 1e-3 {
            break;
        }
        makeup_db = next_makeup_db;
    }

    // Single final 4x true-peak safety trim after loudness convergence.
    apply_true_peak_trim(&mut out, clean_ceiling_dbtp);

    if config.dither != DitherMode::None {
        out = apply_dither(&out, config.dither_bits, config.dither)?;
    }

    Ok(out)
}

/// Deterministic xorshift64* PRNG producing uniform `[-0.5, +0.5)` floats for audio dithering.
#[inline]
pub(crate) fn next_uniform_bipolar(state: &mut u64) -> f64 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    let bits = x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
    (bits as f64) * (1.0 / ((1_u64 << 53) as f64)) - 0.5
}

/// Applies mastering-grade TPDF dithering and optional 2nd-order psychoacoustic high-pass
/// noise shaping (`H(z) = -1.623 z^-1 + 0.692 z^-2`) quantized to `bit_depth` (`8..=24` bits).
pub fn apply_dither(buffer: &AudioBuffer, bit_depth: u16, mode: DitherMode) -> Result<AudioBuffer> {
    if !(8..=24).contains(&bit_depth) {
        return Err(ForgeError::InvalidParameter(format!(
            "dither bit_depth ({bit_depth}) must be between 8 and 24"
        )));
    }
    if mode == DitherMode::None || buffer.num_frames() == 0 {
        return Ok(buffer.clone());
    }

    let full_scale = (1_u64 << (bit_depth - 1)) as f64;
    let inv_scale = 1.0 / full_scale;
    let max_code = full_scale - 1.0;
    let min_code = -full_scale;

    let mut out_channels = Vec::with_capacity(buffer.channels.len());
    for (ch_idx, ch) in buffer.channels.iter().enumerate() {
        // Decorrelated per-channel deterministic seed so stereo dither noise is uncorrelated.
        let mut rng: u64 = 0x9E37_79B9_7F4A_7C15_u64
            ^ ((ch_idx as u64)
                .wrapping_add(1)
                .wrapping_mul(0xBF58_476D_1CE4_E5B9));
        let mut e1 = 0.0_f64;
        let mut e2 = 0.0_f64;
        let mut dst = Vec::with_capacity(ch.len());

        for &raw in ch {
            let x = sanitize_sample_f64(raw).clamp(-1.0, 1.0) * full_scale;
            // Triangular PDF dither in [-1.0, +1.0) LSB
            let tpdf = next_uniform_bipolar(&mut rng) + next_uniform_bipolar(&mut rng);
            let shaped = match mode {
                DitherMode::None => x,
                DitherMode::Tpdf => x,
                DitherMode::NoiseShapedHighPass => x + 1.623 * e1 - 0.692 * e2,
            };
            let q = (shaped + tpdf).round().clamp(min_code, max_code);
            let err = flush_denormal_f64(shaped - q);
            e2 = e1;
            e1 = err;
            dst.push(q * inv_scale);
        }
        out_channels.push(dst);
    }

    Ok(AudioBuffer {
        channels: out_channels,
        sample_rate: buffer.sample_rate,
    })
}

/// Final safety trim if the 4x true peak still slightly exceeds `ceiling_dbtp`.
fn apply_true_peak_trim(out: &mut AudioBuffer, ceiling_dbtp: f64) {
    let (_, post_tp_db, _) = compute_true_peak(out);
    if post_tp_db > ceiling_dbtp {
        let trim = 10.0_f64.powf((ceiling_dbtp - post_tp_db - 0.02) / 20.0);
        for ch in &mut out.channels {
            for s in ch.iter_mut() {
                *s *= trim;
            }
        }
    }
}
