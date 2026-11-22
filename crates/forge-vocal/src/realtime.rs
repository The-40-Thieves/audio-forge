//! Zero-allocation real-time low-latency vocal pitch correction engine (**Forge Vocal Tune**).
//!
//! Combines a streaming YIN F0 detector (`75..=1050 Hz`, 128-sample analysis hop) with
//! scale-locked pitch quantization, retune speed / humanize smoothing, and a stereo-locked
//! dual-grain Hann-windowed PSOLA ring-buffer pitch shifter.

use std::f64::consts::PI;

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
    yin_diff: Vec<f64>,
    key_root: u8,
    scale_mode: VocalTuneScaleMode,
    retune_speed_ms: f64,
    retune_amount: f64,
    humanize: f64,
    wet_mix: f64,
    current_ratio: f64,
    target_ratio: f64,
    detected_hz: f64,
    target_hz: f64,
    correction_cents: f64,
}

impl RealtimeVocalTune {
    const RING_LEN: usize = 8192;
    const YIN_WIN: usize = 1024;
    const YIN_HOP: usize = 128;

    pub fn new(sample_rate: f64) -> Self {
        let sr = if sample_rate.is_finite() {
            sample_rate.clamp(8_000.0, 384_000.0)
        } else {
            48_000.0
        };
        let yin_min_tau = ((sr / 1050.0).floor() as usize).clamp(8, Self::YIN_WIN / 2 - 2);
        let yin_max_tau = ((sr / 75.0).ceil() as usize).clamp(yin_min_tau + 1, Self::YIN_WIN / 2);
        let grain_len = ((sr * 0.0213).round() as usize).clamp(256, 2048);

        Self {
            sample_rate: sr,
            ring_l: vec![0.0; Self::RING_LEN],
            ring_r: vec![0.0; Self::RING_LEN],
            ring_mask: Self::RING_LEN - 1,
            write_pos: 0,
            grain_len,
            grain_phase: 0.0,
            yin_win: Self::YIN_WIN,
            yin_max_tau,
            yin_min_tau,
            yin_hop: Self::YIN_HOP,
            hop_counter: 0,
            yin_buf: vec![0.0; Self::YIN_WIN],
            yin_diff: vec![0.0; Self::YIN_WIN / 2 + 1],
            key_root: 0,
            scale_mode: VocalTuneScaleMode::Chromatic,
            retune_speed_ms: 25.0,
            retune_amount: 0.85,
            humanize: 0.15,
            wet_mix: 1.0,
            current_ratio: 1.0,
            target_ratio: 1.0,
            detected_hz: 0.0,
            target_hz: 0.0,
            correction_cents: 0.0,
        }
    }

    /// Resets all ring buffers and pitch tracking state in place with zero heap allocations.
    pub fn reset(&mut self, sample_rate: f64) {
        let sr = if sample_rate.is_finite() {
            sample_rate.clamp(8_000.0, 384_000.0)
        } else {
            48_000.0
        };
        self.sample_rate = sr;
        self.yin_min_tau = ((sr / 1050.0).floor() as usize).clamp(8, self.yin_win / 2 - 2);
        self.yin_max_tau =
            ((sr / 75.0).ceil() as usize).clamp(self.yin_min_tau + 1, self.yin_win / 2);
        self.grain_len = ((sr * 0.0213).round() as usize).clamp(256, 2048);
        self.ring_l.fill(0.0);
        self.ring_r.fill(0.0);
        self.write_pos = 0;
        self.grain_phase = 0.0;
        self.hop_counter = 0;
        self.yin_buf.fill(0.0);
        self.yin_diff.fill(0.0);
        self.current_ratio = 1.0;
        self.target_ratio = 1.0;
        self.detected_hz = 0.0;
        self.target_hz = 0.0;
        self.correction_cents = 0.0;
    }

    /// Algorithmic latency in samples (`grain_len / 2`, ~`10.7 ms` at 48 kHz).
    #[inline]
    pub fn latency_samples(&self) -> u32 {
        (self.grain_len / 2) as u32
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
        self.yin_diff[0] = 1.0;
        let mut running_sum = 0.0_f64;
        let mut best_tau = 0_usize;
        let mut best_cmnd = 1.0_f64;
        let mut found_thresh = false;
        const YIN_THRESH: f64 = 0.18;

        for tau in 1..=self.yin_max_tau {
            let mut sum = 0.0_f64;
            for j in 0..half {
                let d = self.yin_buf[j] - self.yin_buf[j + tau];
                sum += d * d;
            }
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

        // Dual-grain Hann-windowed PSOLA ring-buffer pitch shifter (stereo-locked)
        let g = self.grain_len as f64;
        let rate = 1.0 - self.current_ratio;
        self.grain_phase = (self.grain_phase + rate).rem_euclid(g);
        let phase_a = self.grain_phase;
        let phase_b = (phase_a + g * 0.5).rem_euclid(g);

        let w_a = 0.5 * (1.0 - (2.0 * PI * phase_a / g).cos());
        let w_b = 1.0 - w_a;

        let delay_a = phase_a + 16.0;
        let delay_b = phase_b + 16.0;

        let wet_l = w_a * Self::read_linear(&self.ring_l, self.ring_mask, self.write_pos, delay_a)
            + w_b * Self::read_linear(&self.ring_l, self.ring_mask, self.write_pos, delay_b);
        let wet_r = w_a * Self::read_linear(&self.ring_r, self.ring_mask, self.write_pos, delay_a)
            + w_b * Self::read_linear(&self.ring_r, self.ring_mask, self.write_pos, delay_b);

        // Latency-aligned dry tap (grain_len / 2 + 16)
        let dry_delay = (self.grain_len / 2 + 16) as f64;
        let dry_l = Self::read_linear(&self.ring_l, self.ring_mask, self.write_pos, dry_delay);
        let dry_r = Self::read_linear(&self.ring_r, self.ring_mask, self.write_pos, dry_delay);

        self.write_pos = self.write_pos.wrapping_add(1);

        let mix = self.wet_mix;
        let out_l = dry_l * (1.0 - mix) + wet_l * mix;
        let out_r = dry_r * (1.0 - mix) + wet_r * mix;

        (out_l as f32, out_r as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn realtime_vocal_tune_snaps_sharp_pitch_to_scale_and_self_heals_nan() {
        let sr = 48_000.0;
        let mut tuner = RealtimeVocalTune::new(sr);
        tuner.set_key_and_scale(0, VocalTuneScaleMode::Major); // C Major
        tuner.set_retune_speed_ms(5.0);
        tuner.set_retune_amount(1.0);
        tuner.set_humanize(0.0);
        tuner.set_wet_mix(1.0);

        // Feed a 452 Hz sine (about +46.6 cents sharp of A4 = 440 Hz)
        for i in 0..8192 {
            let s = (0.4 * (2.0 * PI * 452.0 * (i as f64) / sr).sin()) as f32;
            let (l, r) = tuner.process_stereo_sample(s, s);
            assert!(l.is_finite() && r.is_finite());
        }
        assert!(
            (tuner.detected_hz() - 452.0).abs() < 4.0,
            "detected_hz = {}",
            tuner.detected_hz()
        );
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
}
