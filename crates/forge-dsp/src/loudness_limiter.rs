//! Biquad filter primitives, ITU-R BS.1770-4 K-weighting, 4x oversampled True-Peak,
//! AI Fizz telemetry auditor, and lookahead brickwall mastering limiter.

use std::f64::consts::PI;

use forge_core::{AudioBuffer, AuditReport, ForgeError, Result};
use rustfft::{FftPlanner, num_complex::Complex};

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

    #[inline]
    pub fn process_sample(&mut self, x_raw: f64) -> f64 {
        let x = sanitize_sample_f64(x_raw);
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

    /// Configures the TPT SVF as a Low-Pass filter while preserving integrator state (`ic1eq`, `ic2eq`).
    #[inline]
    pub fn set_lowpass(&mut self, sample_rate: f64, freq_hz: f64, q: f64) {
        let sr = sanitize_param_f64(sample_rate, 48_000.0, 100.0, 768_000.0);
        let f0 = sanitize_param_f64(freq_hz, 1000.0, 5.0, (sr * 0.48).max(5.0));
        let clean_q = sanitize_param_f64(q, std::f64::consts::FRAC_1_SQRT_2, 0.1, 50.0);
        let g = (PI * f0 / sr).tan();
        let k = 1.0 / clean_q;
        self.a1 = 1.0 / (1.0 + g * (g + k));
        self.a2 = g * self.a1;
        self.a3 = g * self.a2;
        self.m0 = 0.0;
        self.m1 = 0.0;
        self.m2 = 1.0;
    }

    /// Configures the TPT SVF as a High-Pass filter while preserving integrator state.
    #[inline]
    pub fn set_highpass(&mut self, sample_rate: f64, freq_hz: f64, q: f64) {
        let sr = sanitize_param_f64(sample_rate, 48_000.0, 100.0, 768_000.0);
        let f0 = sanitize_param_f64(freq_hz, 1000.0, 5.0, (sr * 0.48).max(5.0));
        let clean_q = sanitize_param_f64(q, std::f64::consts::FRAC_1_SQRT_2, 0.1, 50.0);
        let g = (PI * f0 / sr).tan();
        let k = 1.0 / clean_q;
        self.a1 = 1.0 / (1.0 + g * (g + k));
        self.a2 = g * self.a1;
        self.a3 = g * self.a2;
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
pub(crate) const TRUE_PEAK_PHASES: [[f64; 12]; 4] = [
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
/// and exact timestamp in seconds.
pub fn compute_true_peak(buffer: &AudioBuffer) -> (f64, f64, f64) {
    const TAPS: usize = 12;
    // Tap-major, reversed coefficients: `taps[k][p]` multiplies the k-th sample of the
    // oldest-first window `[x[i - 11], ..., x[i]]` for output phase `p`. Accumulating all four
    // phases per tap gives four independent add chains the compiler can vectorize, with the
    // same per-phase summation order (and therefore bit-identical results) as a dot product.
    let mut taps = [[0.0_f64; 4]; TAPS];
    for (p, coeffs) in TRUE_PEAK_PHASES.iter().enumerate() {
        for (k, &c) in coeffs.iter().enumerate() {
            taps[TAPS - 1 - k][p] = c;
        }
    }

    let mut max_sample_abs = 1e-12_f64;
    let mut max_true_abs = 1e-12_f64;
    let mut peak_pos = 0.0_f64;
    let mut padded = Vec::new();

    for ch in &buffer.channels {
        let n = ch.len();
        let last = n.saturating_sub(1) as f64;
        // `TAPS - 1` zeros before the start and after the end, so window `i` is
        // `[x[i - 11], ..., x[i]]` and the filter tail around the final samples is evaluated.
        padded.clear();
        padded.extend(std::iter::repeat_n(0.0, TAPS - 1));
        padded.extend(ch.iter().map(|&x| sanitize_sample_f64(x)));
        padded.extend(std::iter::repeat_n(0.0, TAPS - 1));
        for (i, window) in padded.windows(TAPS).enumerate() {
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

/// Computes 100 ms K-weighted sub-block energies in a single streaming pass per channel
/// (O(num_subs) memory instead of O(frames)), matching ITU-R BS.1770-4 channel summation.
fn compute_k_weighted_sub_energies(channels: &[Vec<f64>], sample_rate: u32) -> Vec<f64> {
    let sr = f64::from(sample_rate.max(1));
    let frames = channels.first().map_or(0, Vec::len);
    let sub_len = (sr * 0.1).round().max(1.0) as usize;
    let num_subs = frames / sub_len;
    if num_subs == 0 {
        return Vec::new();
    }

    let inv_sub_len = 1.0 / (sub_len as f64);
    let mut sub_energies = vec![0.0_f64; num_subs];
    for ch in channels {
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
            for &x in &ch[start..end] {
                let v = rlb.process_sample(shelf.process_sample(x));
                ch_sq += v * v;
            }
            *slot += ch_sq * inv_sub_len;
        }
    }
    sub_energies
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

    if gated_abs.is_empty() {
        return -70.0;
    }
    let mean_abs = gated_abs.iter().sum::<f64>() / (gated_abs.len() as f64);
    let rel_gate_lufs = -0.691 + 10.0 * mean_abs.log10() - 10.0;
    let rel_gate_e = 10.0_f64.powf((rel_gate_lufs + 0.691) / 10.0);
    let gated_rel: Vec<f64> = gated_abs.into_iter().filter(|&e| e > rel_gate_e).collect();
    if gated_rel.is_empty() {
        -70.0
    } else {
        let mean_rel = gated_rel.iter().sum::<f64>() / (gated_rel.len() as f64);
        -0.691 + 10.0 * mean_rel.log10()
    }
}

/// Performs a full ITU-R BS.1770-4 loudness, phase, electrical, and AI-artifact audit.
pub fn audit_audio(buffer: &AudioBuffer) -> AuditReport {
    let sr = f64::from(buffer.sample_rate.max(1));
    let frames = buffer.num_frames();
    let ch_count = buffer.num_channels();

    // 1 & 2. Streaming K-weighted 100ms sub-blocks for BS.1770-4 gating
    // (400ms Momentary = 4 sub-blocks, 3s Short-Term = 30 sub-blocks)
    let sub_energies = compute_k_weighted_sub_energies(&buffer.channels, buffer.sample_rate);

    // Momentary (400ms = 4 blocks) and Short-Term (3000ms = 30 blocks)
    let mut momentary_energies = Vec::new();
    let mut momentary_max_lufs = -70.0_f64;
    if sub_energies.len() >= 4 {
        for w in sub_energies.windows(4) {
            let mean_e = w.iter().sum::<f64>() * 0.25;
            momentary_energies.push(mean_e);
            if mean_e > 1e-12 {
                let lufs = -0.691 + 10.0 * mean_e.log10();
                if lufs > momentary_max_lufs {
                    momentary_max_lufs = lufs;
                }
            }
        }
    } else if !sub_energies.is_empty() {
        let mean_e = sub_energies.iter().sum::<f64>() / (sub_energies.len() as f64);
        momentary_energies.push(mean_e);
        if mean_e > 1e-12 {
            momentary_max_lufs = -0.691 + 10.0 * mean_e.log10();
        }
    }

    let mut short_term_lufs_vec = Vec::new();
    // The loudest 3 s window. A file shorter than one window has no short-term reading and
    // falls back to the momentary max; otherwise a single 400 ms burst must not set it.
    let mut short_term_max_lufs = if sub_energies.len() >= 30 {
        -70.0
    } else {
        momentary_max_lufs
    };
    if sub_energies.len() >= 30 {
        for w in sub_energies.windows(30) {
            let mean_e = w.iter().sum::<f64>() / 30.0;
            if mean_e > 1e-12 {
                let lufs = -0.691 + 10.0 * mean_e.log10();
                short_term_lufs_vec.push(lufs);
                if lufs > short_term_max_lufs {
                    short_term_max_lufs = lufs;
                }
            }
        }
    }

    // BS.1770-4 Two-Stage Gating for Integrated LUFS
    let abs_gate_e = 10.0_f64.powf((-70.0 + 0.691) / 10.0);
    let gated_abs: Vec<f64> = momentary_energies
        .iter()
        .copied()
        .filter(|&e| e > abs_gate_e)
        .collect();

    let integrated_lufs = if gated_abs.is_empty() {
        -70.0
    } else {
        let mean_abs = gated_abs.iter().sum::<f64>() / (gated_abs.len() as f64);
        let rel_gate_lufs = -0.691 + 10.0 * mean_abs.log10() - 10.0;
        let rel_gate_e = 10.0_f64.powf((rel_gate_lufs + 0.691) / 10.0);
        let gated_rel: Vec<f64> = gated_abs.into_iter().filter(|&e| e > rel_gate_e).collect();
        if gated_rel.is_empty() {
            -70.0
        } else {
            let mean_rel = gated_rel.iter().sum::<f64>() / (gated_rel.len() as f64);
            -0.691 + 10.0 * mean_rel.log10()
        }
    };

    // Loudness Range (LRA = P95 - P10 of gated short-term blocks)
    let loudness_range_lu = if short_term_lufs_vec.len() >= 4 {
        let mut st: Vec<f64> = short_term_lufs_vec
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

    // 3. True Peak, Sample Peak, DC Offset, Clipping & Crest Factor
    let (sample_peak_dbfs, true_peak_dbtp, true_peak_timestamp_secs) = compute_true_peak(buffer);
    let mono: Vec<f64> = buffer
        .to_mono()
        .into_iter()
        .map(sanitize_sample_f64)
        .collect();
    let dc_offset = if mono.is_empty() {
        0.0
    } else {
        mono.iter().sum::<f64>() / (mono.len() as f64)
    };
    let rms = if mono.is_empty() {
        1e-12
    } else {
        (mono.iter().map(|&x| (x - dc_offset).powi(2)).sum::<f64>() / (mono.len() as f64))
            .sqrt()
            .max(1e-12)
    };
    let rms_db = 20.0 * rms.log10();
    let crest_factor_db = (sample_peak_dbfs - rms_db).max(0.0);

    let mut clipped_samples = 0_usize;
    for ch in &buffer.channels {
        for w in ch.windows(3) {
            if sanitize_sample_f64(w[0]).abs() >= 0.999
                && sanitize_sample_f64(w[1]).abs() >= 0.999
                && sanitize_sample_f64(w[2]).abs() >= 0.999
            {
                clipped_samples += 1;
            }
        }
    }

    // 4. Stereo & Sub-160Hz Phase Correlation
    let (stereo_correlation, sub_160hz_correlation) = if ch_count >= 2 && frames > 0 {
        let l = &buffer.channels[0];
        let r = &buffer.channels[1];
        let mut sum_lr = 0.0;
        let mut sum_l2 = 0.0;
        let mut sum_r2 = 0.0;

        let mut lp_l = Biquad::lowpass(sr, 160.0, 0.707);
        let mut lp_r = Biquad::lowpass(sr, 160.0, 0.707);
        let mut sub_lr = 0.0;
        let mut sub_l2 = 0.0;
        let mut sub_r2 = 0.0;

        for (&lv_raw, &rv_raw) in l.iter().zip(r.iter()) {
            let lv = sanitize_sample_f64(lv_raw);
            let rv = sanitize_sample_f64(rv_raw);
            sum_lr += lv * rv;
            sum_l2 += lv * lv;
            sum_r2 += rv * rv;

            let sl = lp_l.process_sample(lv);
            let sr_v = lp_r.process_sample(rv);
            sub_lr += sl * sr_v;
            sub_l2 += sl * sl;
            sub_r2 += sr_v * sr_v;
        }
        let corr = if sum_l2 * sum_r2 > 1e-18 {
            (sum_lr / (sum_l2 * sum_r2).sqrt()).clamp(-1.0, 1.0)
        } else {
            1.0
        };
        let sub_corr = if sub_l2 * sub_r2 > 1e-18 {
            (sub_lr / (sub_l2 * sub_r2).sqrt()).clamp(-1.0, 1.0)
        } else {
            1.0
        };
        (corr, sub_corr)
    } else {
        (1.0, 1.0)
    };

    // 5. Spectral Band & AI Fizz Index Analysis (2048-point FFT)
    let fft_size = 2048_usize;
    let hop = 1024_usize;
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(fft_size);
    let mut avg_mag_sq = vec![0.0_f64; fft_size / 2 + 1];
    let mut max_peak_over_median_2_6k = 1.0_f64;
    let mut frame_count = 0_usize;

    let bin_hz = sr / (fft_size as f64);
    let b_2k = ((2000.0 / bin_hz).round() as usize).clamp(2, fft_size / 2 - 5);
    let b_6k = ((6000.0 / bin_hz).round() as usize).clamp(b_2k + 1, fft_size / 2 - 4);

    if mono.len() >= fft_size {
        let window: Vec<f64> = (0..fft_size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * (i as f64) / (fft_size as f64)).cos()))
            .collect();
        let mut buf = vec![Complex::new(0.0, 0.0); fft_size];
        let mut scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
        let inv_n_sq = 1.0 / ((fft_size * fft_size) as f64);

        let mut pos = 0;
        while pos + fft_size <= mono.len() {
            for i in 0..fft_size {
                buf[i] = Complex::new((mono[pos + i] - dc_offset) * window[i], 0.0);
            }
            fft.process_with_scratch(&mut buf, &mut scratch);
            for (dst, c) in avg_mag_sq.iter_mut().zip(&buf[..=fft_size / 2]) {
                *dst += (c.re * c.re + c.im * c.im) * inv_n_sq;
            }

            frame_count += 1;
            pos += hop;
        }
    }

    if frame_count > 0 {
        for v in &mut avg_mag_sq {
            *v /= frame_count as f64;
        }
    }

    let broadband_bin_mean =
        (avg_mag_sq[1..].iter().sum::<f64>() / ((fft_size / 2) as f64)).max(1e-12);
    let low_mid_peak = avg_mag_sq[1..b_2k]
        .iter()
        .copied()
        .fold(1e-12_f64, f64::max);
    for b in (b_2k + 4)..=(b_6k.saturating_sub(4)) {
        let outer_mean = (avg_mag_sq[b - 4]
            + avg_mag_sq[b - 3]
            + avg_mag_sq[b - 2]
            + avg_mag_sq[b + 2]
            + avg_mag_sq[b + 3]
            + avg_mag_sq[b + 4])
            / 6.0;
        let ref_env = outer_mean
            .max(broadband_bin_mean * 0.5)
            .max(low_mid_peak * 0.12);
        let ratio = (avg_mag_sq[b] / ref_env).sqrt();
        if ratio > max_peak_over_median_2_6k {
            max_peak_over_median_2_6k = ratio;
        }
    }

    let band_db = |f_low: f64, f_high: f64| -> f64 {
        let i0 = ((f_low / bin_hz).floor() as usize).clamp(1, fft_size / 2);
        let i1 = ((f_high / bin_hz).ceil() as usize).clamp(i0, fft_size / 2);
        let sum_e: f64 = avg_mag_sq[i0..=i1].iter().sum();
        10.0 * sum_e.max(1e-15).log10()
    };

    // Near digital silence there is no programme to judge: band energy and RMS both sit at
    // their numeric floors, and their difference would read as a loud band.
    // - `mono_silent` gates the mono-FFT band levels & fizz index (so a 180-degree out-of-phase
    //   stereo signal whose mono sum cancels to zero does not report bogus relative band levels).
    // - `channel_silent` checks the loudest individual channel's RMS so electrical/phase warnings
    //   (DC offset, negative stereo correlation, low crest factor) still fire on out-of-phase audio.
    let mono_silent = rms_db <= SILENCE_RMS_DBFS;
    let max_ch_rms = buffer
        .channels
        .iter()
        .map(|ch| {
            if ch.is_empty() {
                1e-12
            } else {
                let mean =
                    ch.iter().map(|&x| sanitize_sample_f64(x)).sum::<f64>() / (ch.len() as f64);
                (ch.iter()
                    .map(|&x| (sanitize_sample_f64(x) - mean).powi(2))
                    .sum::<f64>()
                    / (ch.len() as f64))
                    .sqrt()
                    .max(1e-12)
            }
        })
        .fold(1e-12_f64, f64::max);
    let max_ch_rms_db = 20.0 * max_ch_rms.log10();
    let channel_silent = max_ch_rms_db <= SILENCE_RMS_DBFS;
    let relative_band_db = |f_low: f64, f_high: f64| {
        if mono_silent {
            SILENT_BAND_DB
        } else {
            band_db(f_low, f_high) - rms_db
        }
    };
    let hum_50_60hz_db = relative_band_db(45.0, 65.0);
    let box_mud_200_500hz_db = relative_band_db(200.0, 500.0);
    let harshness_2_6khz_db = relative_band_db(2000.0, 6000.0);
    let sibilance_5_14khz_db = relative_band_db(5000.0, 14000.0);

    // Composite AI Fizz Index (0..100): measures 2-6 kHz spectral whistle spike + upper-mid energy excess + sub phase smear
    let whistle_db = 20.0 * max_peak_over_median_2_6k.log10();
    let whistle_score = ((whistle_db - 6.0) * 4.5).clamp(0.0, 60.0);
    let harsh_score = ((harshness_2_6khz_db + 8.0) * 3.0).clamp(0.0, 25.0);
    let sub_smear_score = ((0.85 - sub_160hz_correlation) * 30.0).clamp(0.0, 15.0);
    let ai_fizz_index = if mono_silent {
        0.0
    } else {
        (whistle_score + harsh_score + sub_smear_score).clamp(0.0, 100.0)
    };

    let distrokid_ready = (-16.5..=-7.5).contains(&integrated_lufs)
        && true_peak_dbtp <= -0.95
        && clipped_samples == 0;

    let mut recommendations = Vec::new();
    if true_peak_dbtp > -1.0 {
        recommendations.push(format!(
            "True Peak is {:.2} dBTP at {:.2}s (exceeds DistroKid -1.0 dBTP ceiling). Run `forge master` brickwall limiter.",
            true_peak_dbtp, true_peak_timestamp_secs
        ));
    }
    if ai_fizz_index >= 25.0 {
        recommendations.push(format!(
            "Elevated AI/Metallic Resonance detected (Fizz Index {:.1}/100, peak whistle +{:.1} dB in 2-6 kHz). Apply `forge restore` OpenSoothe De-Fizz.",
            ai_fizz_index, whistle_db
        ));
    }
    if sub_160hz_correlation < 0.85 {
        recommendations.push(format!(
            "Sub-160 Hz stereo correlation is {:.2} (phasey low-end). Enable Mid/Side Sub-160 Hz Mono-Maker.",
            sub_160hz_correlation
        ));
    }
    if hum_50_60hz_db > -12.0 {
        recommendations.push(
            "Elevated 50/60 Hz ground-loop hum energy detected. Enable 10-harmonic hum notch filter."
                .to_string(),
        );
    }
    if clipped_samples > 0 {
        recommendations.push(format!(
            "{clipped_samples} clipped sample runs detected. Run cubic-spline de-clipper before mastering."
        ));
    }
    if !channel_silent && dc_offset.abs() > 0.01 {
        recommendations.push(format!(
            "DC offset is {:+.4} (reduces headroom and causes clicky edits). Enable DC removal in `forge restore`.",
            dc_offset
        ));
    }
    if !channel_silent && ch_count >= 2 && stereo_correlation < 0.0 {
        recommendations.push(format!(
            "Negative broadband stereo correlation ({:.2}) risks phase cancellation in mono playback. Check stem polarity with `forge phase-align`.",
            stereo_correlation
        ));
    }
    if !mono_silent && frames >= (sr * 0.5) as usize && crest_factor_db < 4.5 {
        recommendations.push(format!(
            "Low crest factor ({:.1} dB) indicates heavy dynamic squashing. Consider backing off limiter drive or lowering target loudness.",
            crest_factor_db
        ));
    }

    AuditReport {
        sample_rate: buffer.sample_rate,
        channels: ch_count,
        duration_secs: buffer.duration_secs(),
        integrated_lufs,
        short_term_max_lufs,
        momentary_max_lufs,
        loudness_range_lu,
        sample_peak_dbfs,
        true_peak_dbtp,
        true_peak_timestamp_secs,
        crest_factor_db,
        dc_offset,
        clipped_samples,
        stereo_correlation,
        sub_160hz_correlation,
        hum_50_60hz_db,
        box_mud_200_500hz_db,
        harshness_2_6khz_db,
        sibilance_5_14khz_db,
        ai_fizz_index,
        distrokid_ready,
        recommendations,
    }
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

/// Masters an `AudioBuffer` with full control over limiter release topology ([`LimiterStyle`]),
/// stereo channel linking (`0.0..=1.0`), and output dithering / psychoacoustic noise shaping
/// ([`DitherMode`]).
pub fn normalize_and_limit_with_config(
    buffer: &AudioBuffer,
    config: &MasteringLimiterConfig,
) -> Result<AudioBuffer> {
    let target_lufs = config.target_lufs;
    let ceiling_dbtp = config.ceiling_dbtp;
    if !target_lufs.is_finite() || !(-40.0..=-4.0).contains(&target_lufs) {
        return Err(ForgeError::InvalidParameter(format!(
            "target_lufs ({target_lufs}) must be between -40.0 and -4.0 LUFS"
        )));
    }
    if !ceiling_dbtp.is_finite() || !(-24.0..=0.0).contains(&ceiling_dbtp) {
        return Err(ForgeError::InvalidParameter(format!(
            "ceiling_dbtp ({ceiling_dbtp}) must be between -24.0 and 0.0 dBTP"
        )));
    }
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
fn next_uniform_bipolar(state: &mut u64) -> f64 {
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
