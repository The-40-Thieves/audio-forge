//! Zero-allocation real-time low-latency vocal pitch correction engine (**Forge Vocal Tune**).
//!
//! Combines a streaming YIN F0 detector (`75..=1050 Hz`, 128-sample analysis hop) with
//! scale-locked pitch quantization, retune speed / humanize smoothing, and a stereo-locked
//! dual-grain Hann-windowed PSOLA ring-buffer pitch shifter.

use std::f64::consts::PI;
use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

/// Musical scale presets for [`RealtimeVocalTune`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VocalTuneScaleMode {
    #[default]
    Chromatic,
    Major,
    Minor,
    PentatonicMinor,
    PentatonicMajor,
}

impl VocalTuneScaleMode {
    /// Returns the 12-semitone interval mask relative to the root pitch class.
    #[inline]
    pub const fn interval_mask(self) -> [bool; 12] {
        match self {
            Self::Chromatic => [true; 12],
            Self::Major => [
                true, false, true, false, true, true, false, true, false, true, false, true,
            ],
            Self::Minor => [
                true, false, true, true, false, true, false, true, true, false, true, false,
            ],
            Self::PentatonicMinor => [
                true, false, false, true, false, true, false, true, false, false, true, false,
            ],
            Self::PentatonicMajor => [
                true, false, true, false, true, false, false, true, false, true, false, false,
            ],
        }
    }
}

/// Real-time zero-allocation vocal pitch corrector (Auto-Tune Pro / MetaTune style).
///
/// Latency is `grain_len / 2` (~`10.7 ms` at 48 kHz). All ring buffers and YIN scratch arrays
/// are allocated once in [`Self::new`] so `process_stereo_sample` and `reset` never allocate.
pub struct RealtimeVocalTune {
    sample_rate: f64,
    ring_l: Vec<f64>,
    ring_r: Vec<f64>,
    ring_mask: usize,
    write_pos: usize,
    grain_len: usize,
    grain_phase: f64,
    yin_win: usize,
    yin_max_tau: usize,
    yin_min_tau: usize,
    yin_hop: usize,
    hop_counter: usize,
    yin_buf: Vec<f64>,
    yin_prefix_sq: Vec<f64>,
    yin_diff: Vec<f64>,
    fft_fwd: Arc<dyn Fft<f64>>,
    fft_inv: Arc<dyn Fft<f64>>,
    fft_a: Vec<Complex<f64>>,
    fft_b: Vec<Complex<f64>>,
    fft_scratch: Vec<Complex<f64>>,
    key_root: u8,
    scale_mode: VocalTuneScaleMode,
    retune_speed_ms: f64,
    retune_amount: f64,
    humanize: f64,
    formant_shift_st: f64,
    formant_eq_l: forge_dsp::Biquad,
    formant_eq_r: forge_dsp::Biquad,
    applied_formant_st: f64,
    wet_mix: f64,
    current_ratio: f64,
    target_ratio: f64,
    detected_hz: f64,
    target_hz: f64,
    correction_cents: f64,
}

impl RealtimeVocalTune {
    /// Computes the power-of-two YIN window size scaled to `sr` so `yin_win / 2` always covers
    /// vocal fundamentals down to `75 Hz` even at `96 kHz` / `192 kHz` / `384 kHz`.
    #[inline]
    fn yin_win_for_sr(sr: f64) -> usize {
        let min_half = (sr / 75.0).ceil() as usize;
        (min_half.saturating_mul(2)).max(2048).next_power_of_two()
    }

    #[inline]
    fn yin_hop_for_sr(sr: f64) -> usize {
        ((sr * (128.0 / 48_000.0)).round() as usize).clamp(64, 1024)
    }

    #[inline]
    fn grain_len_for_sr(sr: f64) -> usize {
        ((sr * 0.0213).round() as usize).clamp(256, 4096)
    }

    pub fn new(sample_rate: f64) -> Self {
        let sr = if sample_rate.is_finite() {
            sample_rate.clamp(8_000.0, 384_000.0)
        } else {
            48_000.0
        };
        let yin_win = Self::yin_win_for_sr(sr);
        let grain_len = Self::grain_len_for_sr(sr);
        let ring_len = (yin_win.max(grain_len).saturating_mul(2))
            .max(8192)
            .next_power_of_two();
        let yin_min_tau = ((sr / 1050.0).floor() as usize).clamp(8, yin_win / 2 - 2);
        let yin_max_tau = ((sr / 75.0).ceil() as usize).clamp(yin_min_tau + 1, yin_win / 2);
        let yin_hop = Self::yin_hop_for_sr(sr);

        let mut planner = FftPlanner::<f64>::new();
        let fft_fwd = planner.plan_fft_forward(yin_win);
        let fft_inv = planner.plan_fft_inverse(yin_win);
        let scratch_len = fft_fwd
            .get_inplace_scratch_len()
            .max(fft_inv.get_inplace_scratch_len());

        Self {
            sample_rate: sr,
            ring_l: vec![0.0; ring_len],
            ring_r: vec![0.0; ring_len],
            ring_mask: ring_len - 1,
            write_pos: 0,
            grain_len,
            grain_phase: 0.0,
            yin_win,
            yin_max_tau,
            yin_min_tau,
            yin_hop,
            hop_counter: 0,
            yin_buf: vec![0.0; yin_win],
            yin_prefix_sq: vec![0.0; yin_win + 1],
            yin_diff: vec![0.0; yin_win / 2 + 1],
            fft_fwd,
            fft_inv,
            fft_a: vec![Complex::new(0.0, 0.0); yin_win],
            fft_b: vec![Complex::new(0.0, 0.0); yin_win],
            fft_scratch: vec![Complex::new(0.0, 0.0); scratch_len],
            key_root: 0,
            scale_mode: VocalTuneScaleMode::Chromatic,
            retune_speed_ms: 25.0,
            retune_amount: 0.85,
            humanize: 0.15,
            formant_shift_st: 0.0,
            formant_eq_l: forge_dsp::Biquad::peaking_eq(sr, 1200.0, 0.8, 0.0),
            formant_eq_r: forge_dsp::Biquad::peaking_eq(sr, 1200.0, 0.8, 0.0),
            applied_formant_st: 0.0,
            wet_mix: 1.0,
            current_ratio: 1.0,
            target_ratio: 1.0,
            detected_hz: 0.0,
            target_hz: 0.0,
            correction_cents: 0.0,
        }
    }

    /// Resets all ring buffers and pitch tracking state in place.
    ///
    /// Performs **zero heap allocations** whenever `sample_rate` does not require a larger FFT
    /// window than already allocated (including all DAW stop/seek resets at the active rate).
    pub fn reset(&mut self, sample_rate: f64) {
        let sr = if sample_rate.is_finite() {
            sample_rate.clamp(8_000.0, 384_000.0)
        } else {
            48_000.0
        };
        let needed_win = Self::yin_win_for_sr(sr);
        let grain_len = Self::grain_len_for_sr(sr);
        if needed_win > self.yin_win || grain_len.saturating_mul(2) > self.ring_l.len() {
            let mut rebuilt = Self::new(sr);
            rebuilt.key_root = self.key_root;
            rebuilt.scale_mode = self.scale_mode;
            rebuilt.retune_speed_ms = self.retune_speed_ms;
            rebuilt.retune_amount = self.retune_amount;
            rebuilt.humanize = self.humanize;
            rebuilt.wet_mix = self.wet_mix;
            rebuilt.set_formant_shift_st(self.formant_shift_st);
            *self = rebuilt;
            return;
        }
        self.sample_rate = sr;
        self.yin_min_tau = ((sr / 1050.0).floor() as usize).clamp(8, self.yin_win / 2 - 2);
        self.yin_max_tau =
            ((sr / 75.0).ceil() as usize).clamp(self.yin_min_tau + 1, self.yin_win / 2);
        self.yin_hop = Self::yin_hop_for_sr(sr);
        self.grain_len = grain_len;
        self.ring_l.fill(0.0);
        self.ring_r.fill(0.0);
        self.write_pos = 0;
        self.grain_phase = 0.0;
        self.hop_counter = 0;
        self.yin_buf.fill(0.0);
        self.yin_prefix_sq.fill(0.0);
        self.yin_diff.fill(0.0);
        self.current_ratio = 1.0;
        self.target_ratio = 1.0;
        self.detected_hz = 0.0;
        self.target_hz = 0.0;
        self.correction_cents = 0.0;
        self.formant_eq_l = forge_dsp::Biquad::peaking_eq(sr, 1200.0, 0.8, 0.0);
        self.formant_eq_r = forge_dsp::Biquad::peaking_eq(sr, 1200.0, 0.8, 0.0);
        self.applied_formant_st = 0.0;
    }

    /// Algorithmic latency in samples (`grain_len / 2 + 16`, matching the latency-aligned dry tap).
    #[inline]
    pub fn latency_samples(&self) -> u32 {
        (self.grain_len / 2 + 16) as u32
    }

    #[inline]
    pub fn set_key_and_scale(&mut self, key_root: u8, mode: VocalTuneScaleMode) {
        self.key_root = key_root % 12;
        self.scale_mode = mode;
    }

    #[inline]
    pub fn set_retune_speed_ms(&mut self, ms: f64) {
        self.retune_speed_ms = if ms.is_finite() {
            ms.clamp(0.0, 100.0)
        } else {
            25.0
        };
    }

    #[inline]
    pub fn set_retune_amount(&mut self, amount: f64) {
        self.retune_amount = if amount.is_finite() {
            amount.clamp(0.0, 1.0)
        } else {
            0.85
        };
    }

    #[inline]
    pub fn set_humanize(&mut self, humanize: f64) {
        self.humanize = if humanize.is_finite() {
            humanize.clamp(0.0, 1.0)
        } else {
            0.15
        };
    }

    #[inline]
    pub fn set_formant_shift_st(&mut self, semitones: f64) {
        self.formant_shift_st = if semitones.is_finite() {
            semitones.clamp(-12.0, 12.0)
        } else {
            0.0
        };
        if (self.formant_shift_st - self.applied_formant_st).abs() > 0.05 {
            self.applied_formant_st = self.formant_shift_st;
            let fc = (1200.0 * 2.0_f64.powf(self.formant_shift_st / 18.0)).clamp(350.0, 4200.0);
            let gain_db = (self.formant_shift_st * 0.45).clamp(-5.5, 5.5);
            let design = forge_dsp::Biquad::peaking_eq(self.sample_rate, fc, 0.85, gain_db);
            self.formant_eq_l.set_coeffs(&design);
            self.formant_eq_r.set_coeffs(&design);
        }
    }

    #[inline]
    pub fn set_wet_mix(&mut self, mix: f64) {
        self.wet_mix = if mix.is_finite() {
            mix.clamp(0.0, 1.0)
        } else {
            1.0
        };
    }

    #[inline]
    pub fn detected_hz(&self) -> f64 {
        self.detected_hz
    }

    #[inline]
    pub fn target_hz(&self) -> f64 {
        self.target_hz
    }

    #[inline]
    pub fn correction_cents(&self) -> f64 {
        self.correction_cents
    }

    /// Snaps continuous `midi` pitch to the nearest allowed pitch in `(key_root, scale_mode)`.
    #[inline]
    fn snap_midi_to_scale(midi: f64, key_root: u8, mode: VocalTuneScaleMode) -> f64 {
        let mask = mode.interval_mask();
        let center = midi.round() as i32;
        let mut best_midi = center as f64;
        let mut best_dist = f64::INFINITY;
        for cand in (center - 6)..=(center + 6) {
            let rel = (cand - i32::from(key_root)).rem_euclid(12) as usize;
            if mask[rel] {
                let d = (f64::from(cand) - midi).abs();
                if d < best_dist {
                    best_dist = d;
                    best_midi = f64::from(cand);
                }
            }
        }
        best_midi
    }

    fn run_yin_hop(&mut self) {
        let n = self.yin_win;
        let start = self.write_pos.wrapping_sub(n);
        let mut rms_acc = 0.0_f64;
        for (i, dst) in self.yin_buf.iter_mut().enumerate().take(n) {
            let idx = start.wrapping_add(i) & self.ring_mask;
            let mono = 0.5 * (self.ring_l[idx] + self.ring_r[idx]);
            *dst = mono;
            rms_acc += mono * mono;
        }
        let rms = (rms_acc / (n as f64)).sqrt();
        if rms < 0.005 {
            self.detected_hz = 0.0;
            self.target_hz = 0.0;
            self.correction_cents = 0.0;
            self.target_ratio = 1.0;
            return;
        }

        let half = n / 2;
        // Build cumulative squared energy prefix sums: O(N).
        self.yin_prefix_sq[0] = 0.0;
        for i in 0..n {
            let x = self.yin_buf[i];
            self.yin_prefix_sq[i + 1] = self.yin_prefix_sq[i] + x * x;
        }
        let e0 = self.yin_prefix_sq[half];

        // Compute exact linear cross-correlation r(tau) = sum_{j=0..half} x[j]*x[j+tau]
        // via O(N log N) FFT with pre-allocated scratch buffers (zero heap allocations).
        for j in 0..half {
            self.fft_a[j] = Complex::new(self.yin_buf[j], 0.0);
        }
        for c in &mut self.fft_a[half..n] {
            *c = Complex::new(0.0, 0.0);
        }
        for (dst, &src) in self.fft_b.iter_mut().zip(self.yin_buf.iter()).take(n) {
            *dst = Complex::new(src, 0.0);
        }
        self.fft_fwd
            .process_with_scratch(&mut self.fft_a[..n], &mut self.fft_scratch);
        self.fft_fwd
            .process_with_scratch(&mut self.fft_b[..n], &mut self.fft_scratch);
        for k in 0..n {
            self.fft_a[k] = self.fft_b[k] * self.fft_a[k].conj();
        }
        self.fft_inv
            .process_with_scratch(&mut self.fft_a[..n], &mut self.fft_scratch);
        let inv_n = 1.0 / (n as f64);

        self.yin_diff[0] = 1.0;
        let mut running_sum = 0.0_f64;
        let mut best_tau = 0_usize;
        let mut best_cmnd = 1.0_f64;
        let mut found_thresh = false;
        const YIN_THRESH: f64 = 0.18;

        for tau in 1..=self.yin_max_tau {
            let e_tau = self.yin_prefix_sq[tau + half] - self.yin_prefix_sq[tau];
            let r_tau = self.fft_a[tau].re * inv_n;
            let sum = (e0 + e_tau - 2.0 * r_tau).max(0.0);
            running_sum += sum;
            let cmnd = if running_sum > 1e-12 {
                sum * (tau as f64) / running_sum
            } else {
                1.0
            };
            self.yin_diff[tau] = cmnd;

            if tau >= self.yin_min_tau {
                if cmnd < YIN_THRESH {
                    if !found_thresh || cmnd < best_cmnd {
                        best_cmnd = cmnd;
                        best_tau = tau;
                        found_thresh = true;
                    }
                } else if found_thresh {
                    break;
                } else if cmnd < best_cmnd {
                    best_cmnd = cmnd;
                    best_tau = tau;
                }
            }
        }

        if best_tau < self.yin_min_tau || best_cmnd > 0.35 {
            self.detected_hz = 0.0;
            self.target_hz = 0.0;
            self.correction_cents = 0.0;
            self.target_ratio = 1.0;
            return;
        }

        // Parabolic refinement around best_tau
        let refined_tau = if best_tau > self.yin_min_tau && best_tau < self.yin_max_tau {
            let s0 = self.yin_diff[best_tau - 1];
            let s1 = self.yin_diff[best_tau];
            let s2 = self.yin_diff[best_tau + 1];
            let denom = 2.0 * (2.0 * s1 - s2 - s0);
            if denom.abs() > 1e-9 {
                (best_tau as f64) + ((s2 - s0) / denom).clamp(-0.5, 0.5)
            } else {
                best_tau as f64
            }
        } else {
            best_tau as f64
        };

        let f0 = self.sample_rate / refined_tau.max(1.0);
        let midi = 69.0 + 12.0 * (f0 / 440.0).log2();
        let snapped = Self::snap_midi_to_scale(midi, self.key_root, self.scale_mode);
        let mut delta_st = (snapped - midi) * self.retune_amount;

        // Humanize deadzone: preserve subtle natural vibrato within +/- (humanize * 0.25) semitones
        let deadzone = self.humanize * 0.25;
        if delta_st.abs() < deadzone {
            delta_st *= 0.2;
        }

        let clamped_st = delta_st.clamp(-6.0, 6.0);
        self.detected_hz = f0;
        self.target_hz = 440.0 * 2.0_f64.powf((midi + clamped_st - 69.0) / 12.0);
        self.correction_cents = clamped_st * 100.0;
        self.target_ratio = 2.0_f64.powf(clamped_st / 12.0).clamp(0.5, 2.0);
    }

    #[inline]
    fn read_linear(ring: &[f64], mask: usize, write_pos: usize, delay: f64) -> f64 {
        let d0 = delay.floor() as usize;
        let frac = delay - (d0 as f64);
        let i0 = write_pos.wrapping_sub(d0) & mask;
        let i1 = write_pos.wrapping_sub(d0 + 1) & mask;
        ring[i0] * (1.0 - frac) + ring[i1] * frac
    }

    /// Processes one stereo audio frame with zero heap allocations.
    #[inline]
    pub fn process_stereo_sample(&mut self, in_l: f32, in_r: f32) -> (f32, f32) {
        let xl = if in_l.is_finite() {
            f64::from(in_l).clamp(-16.0, 16.0)
        } else {
            0.0
        };
        let xr = if in_r.is_finite() {
            f64::from(in_r).clamp(-16.0, 16.0)
        } else {
            0.0
        };

        let wp = self.write_pos & self.ring_mask;
        self.ring_l[wp] = xl;
        self.ring_r[wp] = xr;

        self.hop_counter += 1;
        if self.hop_counter >= self.yin_hop {
            self.hop_counter = 0;
            self.run_yin_hop();
        }

        // Smooth pitch ratio toward target according to retune_speed_ms
        if self.retune_speed_ms <= 0.1 {
            self.current_ratio = self.target_ratio;
        } else {
            let alpha = (-1.0 / (self.retune_speed_ms * 0.001 * self.sample_rate)).exp();
            self.current_ratio = alpha * self.current_ratio + (1.0 - alpha) * self.target_ratio;
        }

        // Latency-aligned dry tap (grain_len / 2 + 16)
        let dry_delay = (self.grain_len / 2 + 16) as f64;
        let dry_l = Self::read_linear(&self.ring_l, self.ring_mask, self.write_pos, dry_delay);
        let dry_r = Self::read_linear(&self.ring_r, self.ring_mask, self.write_pos, dry_delay);

        // When correction is effectively off (ratio == 1.0 and amount == 0), bypass grain shift
        // transparently so disabling correction never colors vocal tone.
        let (mut wet_l, mut wet_r) = if self.retune_amount <= 1e-4
            && (self.current_ratio - 1.0).abs() < 1e-4
        {
            self.target_ratio = 1.0;
            (dry_l, dry_r)
        } else {
            let g = self.grain_len as f64;
            let rate = 1.0 - self.current_ratio;
            self.grain_phase = (self.grain_phase + rate).rem_euclid(g);
            let phase_a = self.grain_phase;
            let phase_b = (phase_a + g * 0.5).rem_euclid(g);

            let w_a = 0.5 * (1.0 - (2.0 * PI * phase_a / g).cos());
            let w_b = 1.0 - w_a;

            let delay_a = phase_a + 16.0;
            let delay_b = phase_b + 16.0;

            (
                w_a * Self::read_linear(&self.ring_l, self.ring_mask, self.write_pos, delay_a)
                    + w_b
                        * Self::read_linear(&self.ring_l, self.ring_mask, self.write_pos, delay_b),
                w_a * Self::read_linear(&self.ring_r, self.ring_mask, self.write_pos, delay_a)
                    + w_b
                        * Self::read_linear(&self.ring_r, self.ring_mask, self.write_pos, delay_b),
            )
        };

        if self.applied_formant_st.abs() > 0.05 {
            wet_l = self.formant_eq_l.process_sample(wet_l);
            wet_r = self.formant_eq_r.process_sample(wet_r);
        }

        self.write_pos = self.write_pos.wrapping_add(1);

        let mix = self.wet_mix;
        let out_l = dry_l * (1.0 - mix) + wet_l * mix;
        let out_r = dry_r * (1.0 - mix) + wet_r * mix;

        let clean_l = if out_l.abs() < 1e-25 { 0.0 } else { out_l };
        let clean_r = if out_r.abs() < 1e-25 { 0.0 } else { out_r };
        (clean_l as f32, clean_r as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed_sine_and_check_pitch(
        sr: f64,
        scale: VocalTuneScaleMode,
        freq_hz: f64,
        amp: f64,
        frames: usize,
        tol_hz: f64,
    ) -> RealtimeVocalTune {
        let mut tuner = RealtimeVocalTune::new(sr);
        tuner.set_key_and_scale(0, scale);
        tuner.set_retune_speed_ms(5.0);
        tuner.set_retune_amount(1.0);
        tuner.set_humanize(0.0);
        tuner.set_wet_mix(1.0);
        for i in 0..frames {
            let s = (amp * (2.0 * PI * freq_hz * (i as f64) / sr).sin()) as f32;
            let (l, r) = tuner.process_stereo_sample(s, s);
            assert!(l.is_finite() && r.is_finite());
        }
        assert!(
            (tuner.detected_hz() - freq_hz).abs() < tol_hz,
            "sr={sr}: expected ~{freq_hz} Hz, got {}",
            tuner.detected_hz()
        );
        tuner
    }

    #[test]
    fn realtime_vocal_tune_snaps_sharp_pitch_to_scale_and_self_heals_nan() {
        // Feed a 452 Hz sine (about +46.6 cents sharp of A4 = 440 Hz)
        let mut tuner =
            feed_sine_and_check_pitch(48_000.0, VocalTuneScaleMode::Major, 452.0, 0.4, 8192, 4.0);
        assert!(
            (tuner.target_hz() - 440.0).abs() < 2.0,
            "target_hz = {}",
            tuner.target_hz()
        );
        assert!(
            tuner.correction_cents() < -30.0,
            "expected downward correction toward A4, got {} cents",
            tuner.correction_cents()
        );

        // NaN/Inf self-healing
        let (nl, nr) = tuner.process_stereo_sample(f32::NAN, f32::INFINITY);
        assert!(nl.is_finite() && nr.is_finite());
    }

    #[test]
    fn realtime_vocal_tune_detects_low_baritone_at_96k_and_192k() {
        for &sr in &[96_000.0_f64, 192_000.0_f64] {
            // 85 Hz (baritone/bass vocal well below the old 93.75 / 187.5 Hz blind spot)
            let frames = (sr * 0.12) as usize;
            feed_sine_and_check_pitch(sr, VocalTuneScaleMode::Chromatic, 85.0, 0.45, frames, 2.5);
        }
    }
}
