//! Windowed-sinc (Kaiser) offline and zero-allocation real-time polyphase resampling.

use std::f64::consts::PI;

use forge_core::{AudioBuffer, ForgeError, MAX_AUDIO_SAMPLES, Result};

use crate::loudness_limiter::{flush_denormal_f64, sanitize_sample_f32, sanitize_sample_f64};

const KAISER_BETA: f64 = 6.5;
const BASE_HALF_TAPS: usize = 24;
const MAX_HALF_TAPS: usize = 96;

/// Zeroth-order modified Bessel function of the first kind `I0(x)` for Kaiser windows.
fn bessel_i0(x: f64) -> f64 {
    let half_x = 0.5 * x;
    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;
    for k in 1..28 {
        let ratio = half_x / f64::from(k);
        term *= ratio * ratio;
        sum += term;
        if term < 1e-16 * sum {
            break;
        }
    }
    sum
}

#[inline]
fn normalized_sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        let px = PI * x;
        px.sin() / px
    }
}

/// Evaluates a normalized Kaiser-windowed sinc weight centered around fractional offset `d`.
#[inline]
fn kaiser_sinc_weight(d: f64, cutoff: f64, half_width: f64, inv_i0_beta: f64) -> f64 {
    let u = d / half_width;
    if u.abs() >= 1.0 {
        return 0.0;
    }
    let win = bessel_i0(KAISER_BETA * (1.0 - u * u).max(0.0).sqrt()) * inv_i0_beta;
    cutoff * normalized_sinc(cutoff * d) * win
}

/// Resamples a multi-channel planar [`AudioBuffer`] to `target_sr` using a DC-normalized
/// Kaiser-windowed sinc kernel with anti-aliasing low-pass scaling when downsampling.
pub fn resample_buffer_sinc(buf: &AudioBuffer, target_sr: u32) -> Result<AudioBuffer> {
    if target_sr == 0 {
        return Err(ForgeError::InvalidParameter(
            "Target sample rate must be greater than 0 Hz".to_string(),
        ));
    }
    if buf.sample_rate == target_sr {
        let mut out = buf.clone();
        for ch in &mut out.channels {
            for s in ch {
                *s = sanitize_sample_f64(*s);
            }
        }
        return Ok(out);
    }
    let in_frames = buf.num_frames();
    if in_frames == 0 {
        return AudioBuffer::new(buf.num_channels(), 0, target_sr);
    }

    let in_sr = f64::from(buf.sample_rate.max(1));
    let out_sr = f64::from(target_sr);
    let ratio = out_sr / in_sr;
    let out_frames_f = ((in_frames as f64) * ratio).round().max(1.0);
    if out_frames_f > (MAX_AUDIO_SAMPLES as f64) {
        return Err(ForgeError::InvalidParameter(
            "Resampled frame count exceeds safe buffer limit".to_string(),
        ));
    }
    let out_frames = out_frames_f as usize;
    if out_frames.saturating_mul(buf.num_channels()) > MAX_AUDIO_SAMPLES {
        return Err(ForgeError::InvalidParameter(format!(
            "Resampled buffer exceeds MAX_AUDIO_SAMPLES ({MAX_AUDIO_SAMPLES})"
        )));
    }

    let inv_ratio = in_sr / out_sr;
    let cutoff = ratio.clamp(0.02, 1.0);
    let half_taps = ((BASE_HALF_TAPS as f64) / cutoff)
        .ceil()
        .clamp(BASE_HALF_TAPS as f64, MAX_HALF_TAPS as f64) as usize;
    let half_width = half_taps as f64;
    let inv_i0_beta = 1.0 / bessel_i0(KAISER_BETA);
    let max_idx = (in_frames - 1) as isize;
    let k_start = -(half_taps as isize - 1);
    let k_end = half_taps as isize;

    let mut out_channels = Vec::with_capacity(buf.num_channels());
    for src in &buf.channels {
        let mut dst = Vec::with_capacity(out_frames);
        for i in 0..out_frames {
            let pos = (i as f64) * inv_ratio;
            let center = pos.floor() as isize;
            let frac = pos - (center as f64);
            let mut acc = 0.0_f64;
            let mut w_sum = 0.0_f64;
            for k in k_start..=k_end {
                let d = (k as f64) - frac;
                let w = kaiser_sinc_weight(d, cutoff, half_width, inv_i0_beta);
                if w == 0.0 {
                    continue;
                }
                let idx = (center + k).clamp(0, max_idx) as usize;
                acc += w * sanitize_sample_f64(src[idx]);
                w_sum += w;
            }
            let sample = if w_sum.abs() > 1e-12 {
                acc / w_sum
            } else {
                0.0
            };
            dst.push(sanitize_sample_f64(sample));
        }
        out_channels.push(dst);
    }

    AudioBuffer::from_channels(out_channels, target_sr)
}

const RT_TAPS: usize = 32;
const RT_HALF_TAPS: usize = RT_TAPS / 2;
const RT_PHASES: usize = 512;
const RT_HIST_LEN: usize = 64;

/// Pre-allocated, zero-allocation polyphase windowed-sinc stereo resampler for real-time audio.
pub struct RealtimeSincResampler {
    in_sr: f64,
    out_sr: f64,
    step: f64,
    phase_pos: f64,
    hist_l: [f64; RT_HIST_LEN],
    hist_r: [f64; RT_HIST_LEN],
    write_pos: usize,
    /// Flattened polyphase kernel table (`RT_PHASES * RT_TAPS`), pre-allocated in `new`.
    table: Vec<f64>,
}

impl RealtimeSincResampler {
    pub fn new(in_sr: f64, out_sr: f64) -> Self {
        let mut resampler = Self {
            in_sr: 48_000.0,
            out_sr: 48_000.0,
            step: 1.0,
            phase_pos: 0.0,
            hist_l: [0.0; RT_HIST_LEN],
            hist_r: [0.0; RT_HIST_LEN],
            write_pos: 0,
            table: vec![0.0; RT_PHASES * RT_TAPS],
        };
        resampler.reset(in_sr, out_sr);
        resampler
    }

    /// Resets circular buffers and rebuilds the polyphase filter table in place with zero allocations.
    pub fn reset(&mut self, in_sr: f64, out_sr: f64) {
        self.in_sr = if in_sr.is_finite() {
            in_sr.clamp(8_000.0, 768_000.0)
        } else {
            48_000.0
        };
        self.out_sr = if out_sr.is_finite() {
            out_sr.clamp(8_000.0, 768_000.0)
        } else {
            48_000.0
        };
        self.step = self.in_sr / self.out_sr;
        self.phase_pos = 0.0;
        self.hist_l.fill(0.0);
        self.hist_r.fill(0.0);
        self.write_pos = 0;

        let cutoff = (self.out_sr / self.in_sr).clamp(0.05, 1.0);
        let half_width = RT_HALF_TAPS as f64;
        let inv_i0_beta = 1.0 / bessel_i0(KAISER_BETA);
        let k_start = -(RT_HALF_TAPS as isize - 1);

        for p in 0..RT_PHASES {
            let frac = (p as f64) / (RT_PHASES as f64);
            let row = &mut self.table[p * RT_TAPS..(p + 1) * RT_TAPS];
            let mut w_sum = 0.0_f64;
            for (t_idx, w_slot) in row.iter_mut().enumerate() {
                let k = k_start + (t_idx as isize);
                let d = (k as f64) - frac;
                let w = kaiser_sinc_weight(d, cutoff, half_width, inv_i0_beta);
                *w_slot = w;
                w_sum += w;
            }
            if w_sum.abs() > 1e-12 {
                let inv = 1.0 / w_sum;
                for w_slot in row {
                    *w_slot *= inv;
                }
            }
        }
    }

    /// Feeds one stereo input sample and emits 0 or more resampled output frames into `out_l`/`out_r`.
    #[inline]
    pub fn process_stereo_sample(
        &mut self,
        in_l: f32,
        in_r: f32,
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> usize {
        let mask = RT_HIST_LEN - 1;
        self.hist_l[self.write_pos & mask] = sanitize_sample_f32(in_l);
        self.hist_r[self.write_pos & mask] = sanitize_sample_f32(in_r);
        self.write_pos = self.write_pos.wrapping_add(1);

        if (self.in_sr - self.out_sr).abs() < 1e-6 {
            if !out_l.is_empty() && !out_r.is_empty() {
                let delay_idx = self.write_pos.wrapping_sub(RT_HALF_TAPS) & mask;
                out_l[0] = self.hist_l[delay_idx] as f32;
                out_r[0] = self.hist_r[delay_idx] as f32;
                return 1;
            }
            return 0;
        }

        let max_out = out_l.len().min(out_r.len());
        let mut emitted = 0_usize;
        while self.phase_pos < 1.0 && emitted < max_out {
            let frac = self.phase_pos.clamp(0.0, 0.999_999);
            let phase_idx = ((frac * (RT_PHASES as f64)) as usize).min(RT_PHASES - 1);
            let row = &self.table[phase_idx * RT_TAPS..(phase_idx + 1) * RT_TAPS];
            let base = self.write_pos.wrapping_sub(RT_TAPS + 1);
            let mut acc_l = 0.0_f64;
            let mut acc_r = 0.0_f64;
            for (t_idx, &w) in row.iter().enumerate() {
                let idx = base.wrapping_add(t_idx + 1) & mask;
                acc_l += w * self.hist_l[idx];
                acc_r += w * self.hist_r[idx];
            }
            out_l[emitted] = flush_denormal_f64(acc_l) as f32;
            out_r[emitted] = flush_denormal_f64(acc_r) as f32;
            emitted += 1;
            self.phase_pos += self.step;
        }
        self.phase_pos -= 1.0;
        emitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn sinc_resampler_preserves_dc_unity_across_rates() {
        for (in_sr, out_sr) in [(44_100, 48_000), (48_000, 44_100), (96_000, 48_000)] {
            let dc = AudioBuffer::from_channels(vec![vec![0.42; 4_000], vec![-0.31; 4_000]], in_sr)
                .expect("dc buf");
            let resampled = resample_buffer_sinc(&dc, out_sr).expect("resample");
            assert_eq!(resampled.sample_rate, out_sr);
            for (ch, expected) in resampled.channels.iter().zip([0.42_f64, -0.31]) {
                let max_err = ch.iter().map(|s| (s - expected).abs()).fold(0.0, f64::max);
                assert!(
                    max_err < 1e-6,
                    "{in_sr}->{out_sr} DC drift {max_err} > 1e-6"
                );
            }
        }
    }

    #[test]
    fn sinc_resampler_preserves_1khz_sine_and_attenuates_aliasing() {
        let in_sr = 96_000_u32;
        let out_sr = 48_000_u32;
        let n = 48_000_usize;

        let tone: Vec<f64> = (0..n)
            .map(|i| 0.5 * (2.0 * PI * 1_000.0 * (i as f64) / f64::from(in_sr)).sin())
            .collect();
        let buf = AudioBuffer::from_channels(vec![tone], in_sr).expect("tone");
        let down = resample_buffer_sinc(&buf, out_sr).expect("downsample");
        // Check interior RMSE vs analytical 1 kHz sine
        let margin = 128;
        let sq_err: f64 = down.channels[0][margin..down.num_frames() - margin]
            .iter()
            .enumerate()
            .map(|(idx, &y)| {
                let i = idx + margin;
                let expected = 0.5 * (2.0 * PI * 1_000.0 * (i as f64) / f64::from(out_sr)).sin();
                (y - expected).powi(2)
            })
            .sum();
        let rmse = (sq_err / ((down.num_frames() - 2 * margin) as f64)).sqrt();
        assert!(rmse < 5e-4, "1 kHz sine RMSE {rmse} too high");

        // 36 kHz supersonic tone above 48 kHz Nyquist (24 kHz) must be strongly attenuated
        let high_tone: Vec<f64> = (0..n)
            .map(|i| 0.5 * (2.0 * PI * 36_000.0 * (i as f64) / f64::from(in_sr)).sin())
            .collect();
        let high_buf = AudioBuffer::from_channels(vec![high_tone], in_sr).expect("high");
        let high_down = resample_buffer_sinc(&high_buf, out_sr).expect("downsample high");
        let out_rms = (high_down.channels[0][margin..high_down.num_frames() - margin]
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            / ((high_down.num_frames() - 2 * margin) as f64))
            .sqrt();
        assert!(
            out_rms < 1e-3,
            "36 kHz above-Nyquist alias energy not attenuated: rms={out_rms}"
        );
    }

    #[test]
    fn realtime_sinc_resampler_sanitizes_nan_and_matches_output_length() {
        let mut rt = RealtimeSincResampler::new(44_100.0, 48_000.0);
        let mut out_l = [0.0_f32; 8];
        let mut out_r = [0.0_f32; 8];
        let mut total = 0_usize;
        for i in 0..4_410 {
            let (l, r) = if i == 100 {
                (f32::NAN, f32::INFINITY)
            } else {
                let t = (i as f32) / 44_100.0;
                (0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin(), 0.25)
            };
            let n = rt.process_stereo_sample(l, r, &mut out_l, &mut out_r);
            for k in 0..n {
                assert!(out_l[k].is_finite() && out_r[k].is_finite());
            }
            total += n;
        }
        assert!(
            (4780..=4820).contains(&total),
            "expected ~4800 output samples from 0.1s 44.1->48k, got {total}"
        );
    }
}
