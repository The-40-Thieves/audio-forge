//! Biquad filter primitives, ITU-R BS.1770-4 K-weighting, 4x oversampled True-Peak,
//! AI Fizz telemetry auditor, and lookahead brickwall mastering limiter.

use std::f64::consts::PI;

use forge_core::{AuditReport, AudioBuffer, ForgeError, Result};
use rustfft::{num_complex::Complex, FftPlanner};

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
    let hi = if max.is_finite() && max >= lo { max } else { lo };
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
const TRUE_PEAK_PHASES: [[f64; 12]; 4] = [
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

/// Group delay of the Annex 2 interpolation filter, in input samples.
const TRUE_PEAK_FILTER_DELAY: usize = 6;

/// Computes ITU-R BS.1770-4 Annex 2 4x oversampled Inter-Sample True Peak (`dBTP`)
/// and exact timestamp in seconds.
pub fn compute_true_peak(buffer: &AudioBuffer) -> (f64, f64, f64) {
    let taps = TRUE_PEAK_PHASES[0].len();
    let mut max_sample_abs = 1e-12_f64;
    let mut max_true_abs = 1e-12_f64;
    let mut peak_frame: usize = 0;

    for ch in &buffer.channels {
        let n = ch.len();
        // Circular history of the last `taps` sanitized samples (zero-padded before the start),
        // fed `taps - 1` trailing zeros so the filter tail around the final samples is evaluated.
        let mut history = [0.0_f64; 12];
        let mut head = 0_usize;
        let padded = ch.iter().copied().chain(std::iter::repeat_n(0.0, taps - 1));
        for (i, raw) in padded.enumerate() {
            let s_curr = sanitize_sample_f64(raw);
            if i < n {
                let s_abs = s_curr.abs();
                if s_abs > max_sample_abs {
                    max_sample_abs = s_abs;
                }
                if s_abs > max_true_abs {
                    max_true_abs = s_abs;
                    peak_frame = i;
                }
            }
            history[head] = s_curr;
            for coeffs in &TRUE_PEAK_PHASES {
                // coeffs[k] multiplies x[i - k]
                let mut acc = 0.0;
                for (k, &c) in coeffs.iter().enumerate() {
                    acc += c * history[(head + taps - k) % taps];
                }
                let interp = acc.abs();
                if interp > max_true_abs {
                    max_true_abs = interp;
                    peak_frame = i.saturating_sub(TRUE_PEAK_FILTER_DELAY).min(n.saturating_sub(1));
                }
            }
            head = (head + 1) % taps;
        }
    }

    let sample_peak_dbfs = 20.0 * max_sample_abs.log10();
    let true_peak_dbtp = 20.0 * max_true_abs.log10();
    let timestamp = peak_frame as f64 / f64::from(buffer.sample_rate.max(1));
    (sample_peak_dbfs, true_peak_dbtp, timestamp)
}

/// Performs a full ITU-R BS.1770-4 loudness, phase, electrical, and AI-artifact audit.
pub fn audit_audio(buffer: &AudioBuffer) -> AuditReport {
    let sr = f64::from(buffer.sample_rate.max(1));
    let frames = buffer.num_frames();
    let ch_count = buffer.num_channels();

    // 1. K-weight all channels
    let mut k_channels = Vec::with_capacity(ch_count);
    for ch in &buffer.channels {
        let mut shelf = Biquad::k_weight_pre_shelf(sr);
        let mut rlb = Biquad::k_weight_rlb_highpass(sr);
        let mut k_ch = Vec::with_capacity(frames);
        for &x in ch {
            k_ch.push(rlb.process_sample(shelf.process_sample(x)));
        }
        k_channels.push(k_ch);
    }

    // 2. 100ms sub-blocks for BS.1770-4 gating (400ms Momentary = 4 sub-blocks, 3s Short-Term = 30 sub-blocks)
    let sub_len = (sr * 0.1).round().max(1.0) as usize;
    let num_subs = frames / sub_len;
    let mut sub_energies = Vec::with_capacity(num_subs);

    for s in 0..num_subs {
        let start = s * sub_len;
        let end = start + sub_len;
        let mut sum_sq = 0.0;
        for k_ch in &k_channels {
            let mut ch_sq = 0.0;
            for &v in &k_ch[start..end] {
                ch_sq += v * v;
            }
            sum_sq += ch_sq / (sub_len as f64);
        }
        sub_energies.push(sum_sq);
    }

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
    let mut short_term_max_lufs = momentary_max_lufs;
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
        let gated_rel: Vec<f64> = gated_abs
            .into_iter()
            .filter(|&e| e > rel_gate_e)
            .collect();
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

        let mut pos = 0;
        while pos + fft_size <= mono.len() {
            for i in 0..fft_size {
                buf[i] = Complex::new((mono[pos + i] - dc_offset) * window[i], 0.0);
            }
            fft.process(&mut buf);
            let mags: Vec<f64> = buf[..=fft_size / 2]
                .iter()
                .map(|c| (c.re * c.re + c.im * c.im) / ((fft_size * fft_size) as f64))
                .collect();

            for (dst, &m) in avg_mag_sq.iter_mut().zip(mags.iter()) {
                *dst += m;
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

    let hum_50_60hz_db = band_db(45.0, 65.0) - rms_db;
    let box_mud_200_500hz_db = band_db(200.0, 500.0) - rms_db;
    let harshness_2_6khz_db = band_db(2000.0, 6000.0) - rms_db;
    let sibilance_5_14khz_db = band_db(5000.0, 14000.0) - rms_db;

    // Composite AI Fizz Index (0..100): measures 2-6 kHz spectral whistle spike + upper-mid energy excess + sub phase smear
    let whistle_db = 20.0 * max_peak_over_median_2_6k.log10();
    let whistle_score = ((whistle_db - 6.0) * 4.5).clamp(0.0, 60.0);
    let harsh_score = ((harshness_2_6khz_db + 8.0) * 3.0).clamp(0.0, 25.0);
    let sub_smear_score = ((0.85 - sub_160hz_correlation) * 30.0).clamp(0.0, 15.0);
    let ai_fizz_index = (whistle_score + harsh_score + sub_smear_score).clamp(0.0, 100.0);

    let distrokid_ready =
        (-16.5..=-7.5).contains(&integrated_lufs) && true_peak_dbtp <= -0.95 && clipped_samples == 0;

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

/// Masters an `AudioBuffer` to a target Integrated LUFS and strict True-Peak ceiling (`-1.0 dBTP`).
pub fn normalize_and_limit(
    buffer: &AudioBuffer,
    target_lufs: f64,
    ceiling_dbtp: f64,
) -> Result<AudioBuffer> {
    if !target_lufs.is_finite() || !(-40.0..=-4.0).contains(&target_lufs) {
        return Err(ForgeError::InvalidParameter(format!(
            "target_lufs ({target_lufs}) must be between -40.0 and -4.0 LUFS"
        )));
    }
    let clean_ceiling_dbtp = sanitize_param_f64(ceiling_dbtp, -1.0, -24.0, 0.0);
    let initial = audit_audio(buffer);
    let makeup_db = if initial.integrated_lufs <= -65.0 {
        0.0
    } else {
        (target_lufs - initial.integrated_lufs).clamp(-24.0, 24.0)
    };
    let makeup_gain = 10.0_f64.powf(makeup_db / 20.0);
    let ceiling_lin = 10.0_f64.powf((clean_ceiling_dbtp - 0.04) / 20.0);

    let mut out = buffer.clone();
    let frames = out.num_frames();
    let ch_count = out.num_channels();
    if frames == 0 {
        return Ok(out);
    }

    // Apply initial makeup gain (sanitizing any non-finite samples)
    for ch in &mut out.channels {
        for s in ch.iter_mut() {
            *s = sanitize_sample_f64(*s) * makeup_gain;
        }
    }

    // Compute per-frame peak envelope including 4x inter-sample peaks
    let mut inst_peak = vec![0.0_f64; frames];
    for ch in &out.channels {
        for i in 0..frames {
            let mut pk = ch[i].abs();
            if i >= 1 && i + 2 < frames {
                let interp = (-0.0625 * ch[i - 1]
                    + 0.5625 * ch[i]
                    + 0.5625 * ch[i + 1]
                    - 0.0625 * ch[i + 2])
                    .abs();
                if interp > pk {
                    pk = interp;
                }
            }
            if pk > inst_peak[i] {
                inst_peak[i] = pk;
            }
        }
    }

    // 3ms lookahead window + smooth 50ms release envelope
    let sr = f64::from(out.sample_rate.max(1));
    let lookahead = ((sr * 0.003).round() as usize).max(4);
    let release_coeff = (-1.0 / (sr * 0.050)).exp();

    let mut gain_env = vec![1.0_f64; frames];
    let mut current_gain = 1.0_f64;

    for i in 0..frames {
        let win_end = (i + lookahead).min(frames);
        let mut max_future = inst_peak[i];
        for &p in &inst_peak[i..win_end] {
            if p > max_future {
                max_future = p;
            }
        }
        let target_g = if max_future > ceiling_lin {
            ceiling_lin / max_future
        } else {
            1.0
        };
        if target_g < current_gain {
            // Fast lookahead attack
            current_gain = target_g;
        } else {
            current_gain = release_coeff * current_gain + (1.0 - release_coeff) * target_g;
        }
        gain_env[i] = current_gain;
    }

    for ch in 0..ch_count {
        for (s, &g) in out.channels[ch].iter_mut().zip(gain_env.iter()) {
            *s *= g;
        }
    }

    // Final safety trim if 4x true-peak slightly exceeds clean_ceiling_dbtp
    let (_, post_tp_db, _) = compute_true_peak(&out);
    if post_tp_db > clean_ceiling_dbtp {
        let trim = 10.0_f64.powf((clean_ceiling_dbtp - post_tp_db - 0.02) / 20.0);
        for ch in &mut out.channels {
            for s in ch.iter_mut() {
                *s *= trim;
            }
        }
    }

    Ok(out)
}
