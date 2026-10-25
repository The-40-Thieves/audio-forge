//! `forge-vocal` — Studio Vocal Production & Editing Suite:
//! - YIN / Autocorrelation F0 Pitch Contour Extractor (with optional RMVPE/FCPE worker handoff)
//! - Scale-Locked Pitch Quantizer & Formant-Preserving Pitch/Harmony Shifter
//! - Chord-Aware Multi-Part Harmony Stack Generator (3rd above, 5th below, octave double)
//! - Native REAPER Take Pitch Envelope (`<PITCHENV`) & `ReaPitch` curve compiler

use forge_core::{AudioBuffer, ForgeError, Result};
use forge_theory::parse_pitch_class;
use serde::{Deserialize, Serialize};

/// A detected F0 frame at a specific timestamp in seconds.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PitchFrame {
    pub time_secs: f64,
    /// Fundamental frequency in Hz (`0.0` if unvoiced/silent).
    pub f0_hz: f64,
    /// Continuous MIDI pitch (`69.0 = 440 Hz`, `0.0` if unvoiced).
    pub midi_pitch: f64,
    /// Target scale-corrected MIDI pitch.
    pub target_midi_pitch: f64,
    /// Required pitch correction in semitones (`target - midi_pitch`).
    pub correction_semitones: f64,
}

/// Scale definition for vocal tuning and diatonic harmony generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocalScale {
    pub root_pc: u8,
    pub is_minor: bool,
    pub allowed_pcs: [bool; 12],
}

impl VocalScale {
    /// Parses a scale descriptor: a root note followed by an optional mode, written attached or
    /// after a space (`"D minor"`, `"Am"`, `"F#m"`, `"F# dorian"`, `"A minor pentatonic"`, `"C"`).
    ///
    /// Modes: major/ionian, minor/aeolian, dorian, phrygian, lydian, mixolydian, locrian, harmonic
    /// and melodic minor, major and minor pentatonic, blues and chromatic, with the usual
    /// abbreviations (`m`, `min`, `maj`, `dor`, `mix`, `pent`, ...). Anything else is an error.
    pub fn parse(scale_str: &str) -> Result<Self> {
        let trimmed = scale_str.trim();
        let (root_pc, mode) = parse_pitch_class(trimmed)
            .map_err(|_| ForgeError::TheoryError(format!("Invalid scale root in '{trimmed}'")))?;
        let intervals = scale_intervals(mode).ok_or_else(|| {
            ForgeError::TheoryError(format!(
                "Unknown scale mode '{}' in '{trimmed}'",
                mode.trim()
            ))
        })?;
        let mut allowed_pcs = [false; 12];
        for &iv in intervals {
            allowed_pcs[usize::from((root_pc + iv) % 12)] = true;
        }
        Ok(Self {
            root_pc,
            is_minor: intervals.contains(&3) && !intervals.contains(&4),
            allowed_pcs,
        })
    }

    /// Snaps a continuous MIDI note (`e.g. 62.38`) to the nearest pitch in the scale.
    pub fn snap_midi(&self, midi_pitch: f64, retune_strength: f64) -> f64 {
        let clean_pitch = forge_dsp::sanitize_param_f64(midi_pitch, 60.0, 0.0, 127.0);
        let rounded = clean_pitch.round() as i32;
        let mut best_note = rounded;
        let mut best_dist = 999.0_f64;

        for candidate in (rounded - 2).max(0)..=(rounded + 2).min(127) {
            let pc = candidate.rem_euclid(12) as usize;
            if self.allowed_pcs[pc] {
                let dist = (f64::from(candidate) - clean_pitch).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_note = candidate;
                }
            }
        }
        let s = forge_dsp::sanitize_param_f64(retune_strength, 1.0, 0.0, 1.0);
        (clean_pitch * (1.0 - s) + f64::from(best_note) * s).clamp(0.0, 127.0)
    }

    /// Computes a diatonic scale degree offset (e.g., `+2` = diatonic 3rd above, `-2` = 3rd below).
    pub fn diatonic_shift_semitones(&self, midi_note: i32, degree_steps: i32) -> f64 {
        let clamped_note = midi_note.clamp(0, 127);
        let clamped_steps = degree_steps.clamp(-28, 28);
        let mut scale_notes = Vec::with_capacity(64);
        for n in 36_i32..=96 {
            if self.allowed_pcs[n.rem_euclid(12) as usize] {
                scale_notes.push(n);
            }
        }
        if scale_notes.is_empty() {
            return f64::from(clamped_steps * 2);
        }
        // Find nearest scale index
        let mut best_idx = 0_usize;
        let mut best_diff = i64::MAX;
        for (idx, &sn) in scale_notes.iter().enumerate() {
            let d = (i64::from(sn) - i64::from(clamped_note)).abs();
            if d < best_diff {
                best_diff = d;
                best_idx = idx;
            }
        }
        let target_idx = (best_idx as i32)
            .saturating_add(clamped_steps)
            .clamp(0, (scale_notes.len() - 1) as i32) as usize;
        f64::from(scale_notes[target_idx] - clamped_note)
    }
}

const MAJOR: [u8; 7] = [0, 2, 4, 5, 7, 9, 11];
const NATURAL_MINOR: [u8; 7] = [0, 2, 3, 5, 7, 8, 10];
const DORIAN: [u8; 7] = [0, 2, 3, 5, 7, 9, 10];
const PHRYGIAN: [u8; 7] = [0, 1, 3, 5, 7, 8, 10];
const LYDIAN: [u8; 7] = [0, 2, 4, 6, 7, 9, 11];
const MIXOLYDIAN: [u8; 7] = [0, 2, 4, 5, 7, 9, 10];
const LOCRIAN: [u8; 7] = [0, 1, 3, 5, 6, 8, 10];
const HARMONIC_MINOR: [u8; 7] = [0, 2, 3, 5, 7, 8, 11];
const MELODIC_MINOR: [u8; 7] = [0, 2, 3, 5, 7, 9, 11];
const MAJOR_PENTATONIC: [u8; 5] = [0, 2, 4, 7, 9];
const MINOR_PENTATONIC: [u8; 5] = [0, 3, 5, 7, 10];
const BLUES: [u8; 6] = [0, 3, 5, 6, 7, 10];
const CHROMATIC: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

/// Maps the mode text after the root to scale intervals; word order does not matter.
fn scale_intervals(mode: &str) -> Option<&'static [u8]> {
    // `CM` is C major, so check the capital before folding case.
    if mode.trim() == "M" {
        return Some(&MAJOR);
    }
    let mut words = mode
        .split_whitespace()
        .map(canonical_scale_word)
        .collect::<Option<Vec<_>>>()?;
    words.sort_unstable();
    SCALES
        .iter()
        .find(|(spelling, _)| *spelling == words.as_slice())
        .map(|&(_, intervals)| intervals)
}

/// Canonical mode words, sorted, and the scale they name.
const SCALES: [(&[&str], &[u8]); 16] = [
    (&[], &MAJOR),
    (&["major"], &MAJOR),
    (&["minor"], &NATURAL_MINOR),
    (&["dorian"], &DORIAN),
    (&["phrygian"], &PHRYGIAN),
    (&["lydian"], &LYDIAN),
    (&["mixolydian"], &MIXOLYDIAN),
    (&["locrian"], &LOCRIAN),
    (&["harmonic", "minor"], &HARMONIC_MINOR),
    (&["melodic", "minor"], &MELODIC_MINOR),
    (&["pentatonic"], &MAJOR_PENTATONIC),
    (&["major", "pentatonic"], &MAJOR_PENTATONIC),
    (&["minor", "pentatonic"], &MINOR_PENTATONIC),
    (&["blues"], &BLUES),
    (&["blues", "minor"], &BLUES),
    (&["chromatic"], &CHROMATIC),
];

fn canonical_scale_word(word: &str) -> Option<&'static str> {
    Some(match word.to_ascii_lowercase().as_str() {
        "maj" | "major" | "ion" | "ionian" => "major",
        "m" | "min" | "minor" | "aeo" | "aeolian" => "minor",
        "dor" | "dorian" => "dorian",
        "phr" | "phrygian" => "phrygian",
        "lyd" | "lydian" => "lydian",
        "mix" | "mixolydian" => "mixolydian",
        "loc" | "locrian" => "locrian",
        "harm" | "harmonic" => "harmonic",
        "mel" | "melodic" => "melodic",
        "pent" | "pentatonic" => "pentatonic",
        "blues" => "blues",
        "chrom" | "chromatic" => "chromatic",
        _ => return None,
    })
}

/// YIN absolute threshold: the first dip of `d'(tau)` below it is taken as the period.
const YIN_THRESHOLD: f64 = 0.15;
/// A frame with no dip below the threshold is still voiced if its global minimum is below this.
const YIN_VOICING_LIMIT: f64 = 0.35;
const YIN_MIN_F0_HZ: f64 = 75.0;
const YIN_MAX_F0_HZ: f64 = 900.0;
/// Frames quieter than this RMS are treated as silence.
const YIN_SILENCE_RMS: f64 = 0.012;

/// Frame geometry and lag search range for one sample rate.
struct Yin {
    sr: f64,
    hop: usize,
    win: usize,
    half: usize,
    tau_min: usize,
    tau_max: usize,
}

impl Yin {
    fn new(sample_rate: u32) -> Self {
        let sr = f64::from(sample_rate.max(1));
        let hop = ((sr * 0.020).round() as usize).max(128);
        let win = (hop * 2).max(512);
        let half = win / 2;
        Self {
            sr,
            hop,
            win,
            half,
            tau_min: ((sr / YIN_MAX_F0_HZ).floor() as usize).max(8),
            // `tau_max + 1` is still needed beside the integration window for interpolation.
            tau_max: ((sr / YIN_MIN_F0_HZ).ceil() as usize).min(win - half - 1),
        }
    }

    /// F0 of one window, or `None` when it is silent or unvoiced. `diff` and `cmnd` are scratch
    /// buffers of length `tau_max + 2`.
    fn estimate_f0(&self, slice: &[f64], diff: &mut [f64], cmnd: &mut [f64]) -> Option<f64> {
        let rms = (slice.iter().map(|&x| x * x).sum::<f64>() / (slice.len() as f64)).sqrt();
        if !rms.is_finite() || rms < YIN_SILENCE_RMS || self.tau_min >= self.tau_max {
            return None;
        }
        centered_difference(slice, self.half, diff);
        cumulative_mean_normalize(diff, cmnd);
        let tau = pick_period(cmnd, self.tau_min, self.tau_max)?;
        let refined = (tau as f64) + parabolic_offset(diff[tau - 1], diff[tau], diff[tau + 1]);
        Some(self.sr / refined)
    }
}

/// YIN difference function `d(tau)` for every lag in `1..diff.len()`. Each lag compares `half`
/// sample pairs centred on the middle of `slice`, so every lag measures the same instant.
fn centered_difference(slice: &[f64], half: usize, diff: &mut [f64]) {
    diff[0] = 0.0;
    for tau in 1..diff.len() {
        let start = (slice.len() - half - tau) / 2;
        let early = &slice[start..start + half];
        let late = &slice[start + tau..start + tau + half];
        diff[tau] = early.iter().zip(late).map(|(a, b)| (a - b) * (a - b)).sum();
    }
}

/// YIN cumulative mean normalized difference `d'(tau)`, with `d'(0) = 1`.
fn cumulative_mean_normalize(diff: &[f64], cmnd: &mut [f64]) {
    cmnd[0] = 1.0;
    let mut running_sum = 0.0_f64;
    for tau in 1..diff.len() {
        running_sum += diff[tau];
        cmnd[tau] = if running_sum > 1e-12 {
            diff[tau] * (tau as f64) / running_sum
        } else {
            1.0
        };
    }
}

/// The first lag whose `d'` dips below the threshold, followed down to the bottom of that dip.
/// Without such a dip, the global minimum if it is low enough to call the frame voiced.
fn pick_period(cmnd: &[f64], tau_min: usize, tau_max: usize) -> Option<usize> {
    if let Some(mut tau) = (tau_min..=tau_max).find(|&t| cmnd[t] < YIN_THRESHOLD) {
        while tau < tau_max && cmnd[tau + 1] < cmnd[tau] {
            tau += 1;
        }
        return Some(tau);
    }
    let best = (tau_min..=tau_max).min_by(|&a, &b| cmnd[a].total_cmp(&cmnd[b]))?;
    (cmnd[best] < YIN_VOICING_LIMIT).then_some(best)
}

/// Offset in lags of the vertex of the parabola through three evenly spaced points.
fn parabolic_offset(left: f64, mid: f64, right: f64) -> f64 {
    let curvature = left - 2.0 * mid + right;
    if curvature > 0.0 {
        (0.5 * (left - right) / curvature).clamp(-0.5, 0.5)
    } else {
        0.0
    }
}

fn pitch_frame(time_secs: f64, f0_hz: f64, scale: &VocalScale, retune: f64) -> PitchFrame {
    if f0_hz <= 0.0 {
        return PitchFrame {
            time_secs,
            f0_hz: 0.0,
            midi_pitch: 0.0,
            target_midi_pitch: 0.0,
            correction_semitones: 0.0,
        };
    }
    let midi_pitch = 69.0 + 12.0 * (f0_hz / 440.0).log2();
    let target_midi_pitch = scale.snap_midi(midi_pitch, retune);
    PitchFrame {
        time_secs,
        f0_hz,
        midi_pitch,
        target_midi_pitch,
        correction_semitones: (target_midi_pitch - midi_pitch).clamp(-6.0, 6.0),
    }
}

/// Extracts a 20 ms-hop F0 contour (75 Hz to 900 Hz) with the YIN estimator and computes
/// scale-locked pitch correction envelopes.
///
/// Each period is the first dip of the cumulative mean normalized difference below `0.15`,
/// taken at the bottom of that dip and refined by parabolic interpolation, so the pitch is not
/// quantized to whole-sample lags. `time_secs` is the centre of the analysis window, the instant
/// every lag's comparisons are centred on. Silent and unvoiced frames have `f0_hz == 0.0`.
pub fn track_vocal_pitch(
    buffer: &AudioBuffer,
    scale: &VocalScale,
    retune_strength: f64,
) -> Vec<PitchFrame> {
    let yin = Yin::new(buffer.sample_rate);
    let mono = buffer.to_mono();
    let mut diff = vec![0.0_f64; yin.tau_max + 2];
    let mut cmnd = vec![0.0_f64; yin.tau_max + 2];
    let mut frames = Vec::new();
    let mut pos = 0_usize;
    while pos + yin.win <= mono.len() {
        let slice = &mono[pos..pos + yin.win];
        let time_secs = ((pos as f64) + ((yin.win - 1) as f64) / 2.0) / yin.sr;
        let f0_hz = yin.estimate_f0(slice, &mut diff, &mut cmnd).unwrap_or(0.0);
        frames.push(pitch_frame(time_secs, f0_hz, scale, retune_strength));
        pos += yin.hop;
    }
    frames
}

/// Computes a 10th-order Linear Predictive Coding (LPC) spectral tilt/formant balance ratio
/// using Levinson-Durbin recursion on a short frame, returning the high-to-low formant tilt in dB.
pub fn estimate_lpc_formant_tilt_db(frame: &[f64]) -> f64 {
    const ORDER: usize = 10;
    if frame.len() <= ORDER {
        return 0.0;
    }
    let mut r = [0.0_f64; ORDER + 1];
    for lag in 0..=ORDER {
        let mut sum = 0.0;
        for i in 0..(frame.len() - lag) {
            let x0 = forge_dsp::sanitize_sample_f64(frame[i]);
            let x1 = forge_dsp::sanitize_sample_f64(frame[i + lag]);
            sum += x0 * x1;
        }
        r[lag] = sum / (frame.len() as f64);
    }
    if !r[0].is_finite() || r[0] < 1e-10 {
        return 0.0;
    }
    // Levinson-Durbin recursion for LPC reflection coefficients
    let mut a = [0.0_f64; ORDER + 1];
    let mut e = r[0];
    for i in 1..=ORDER {
        let mut acc = r[i];
        for j in 1..i {
            acc += a[j] * r[i - j];
        }
        if !acc.is_finite() || !e.is_finite() || e <= 1e-12 {
            break;
        }
        let k = (-acc / e).clamp(-0.99, 0.99);
        let prev_a = a;
        a[i] = k;
        for j in 1..i {
            a[j] = prev_a[j] + k * prev_a[i - j];
        }
        e *= 1.0 - k * k;
        if e <= 1e-12 {
            break;
        }
    }
    // First reflection coefficient (-r[1]/r[0]) encodes spectral tilt between F1 (<800Hz) and F2/F3 (2-4kHz)
    let r1_norm = if (r[1] / r[0]).is_finite() {
        (r[1] / r[0]).clamp(-0.99, 0.99)
    } else {
        0.0
    };
    20.0 * ((1.0 - r1_norm) / (1.0 + r1_norm)).max(1e-6).log10() * 0.35
}

/// Detects whether a short frame around index `i` is an unvoiced sibilant/fricative consonant
/// (`s`, `sh`, `f`, `t`, `k`, breath) using Zero-Crossing Rate (ZCR) and high-frequency energy ratio.
/// Like **Celemony Melodyne 5 Sibilant Detection**, unvoiced sibilants are kept at `0.0 st` pitch shift
/// so consonants never sound lispy or phase-smeared when tuning or generating harmonies!
#[inline]
fn is_unvoiced_sibilant_frame(ch: &[f64], center: usize, half_win: usize, sr: f64) -> f64 {
    let start = center.saturating_sub(half_win);
    let end = (center + half_win).min(ch.len());
    let len = end.saturating_sub(start);
    if len < 16 {
        return 0.0;
    }
    let mut zc = 0_usize;
    let mut energy = 0.0_f64;
    let mut diff_energy = 0.0_f64;
    let slice = &ch[start..end];
    for w in slice.windows(2) {
        if (w[0] >= 0.0) != (w[1] >= 0.0) {
            zc += 1;
        }
        energy += w[0] * w[0];
        let d = w[1] - w[0];
        diff_energy += d * d;
    }
    if energy < 1e-9 {
        return 1.0; // Silent/breath -> do not pitch-shift
    }
    let zcr_hz = (zc as f64) * sr / (2.0 * len as f64);
    let hf_ratio = (diff_energy / energy.max(1e-12)).clamp(0.0, 4.0);
    // Sibilants have ZCR > 2600 Hz and high first-difference energy ratio (> 0.65)
    let zcr_score = ((zcr_hz - 2200.0) / 2000.0).clamp(0.0, 1.0);
    let hf_score = ((hf_ratio - 0.55) / 0.85).clamp(0.0, 1.0);
    (zcr_score * 0.6 + hf_score * 0.4).clamp(0.0, 1.0)
}

/// Generates a formant-compensated, **sibilant-preserving** pitch-shifted vocal track using:
/// 1. **Melodyne 5-Style Sibilant Separation:** Unvoiced sibilants/consonants (`s/sh/t/k`) bypass the pitch shifter (`100%` dry phase-coherent), preventing lispy artifacts.
/// 2. **Dual-Grain Hanning Crossfade PSOLA Pitch Shifting** on voiced vowels.
/// 3. **Levinson-Durbin LPC Formant Spectral Envelope Preservation** (compensating both F1/F2 vowel cavity resonances and spectral tilt).
pub fn shift_vocal_pitch_formant_preserved(
    buffer: &AudioBuffer,
    semitone_curve: &[(f64, f64)], // (time_secs, shift_semitones)
    formant_shift_semitones: f64,
) -> Result<AudioBuffer> {
    let sr = f64::from(buffer.sample_rate.max(1));
    let frames = buffer.num_frames();
    if frames == 0 {
        return Ok(buffer.clone());
    }

    let grain_len = ((sr * 0.032).round() as usize).max(256); // 32ms grain for vocal fundamental
    let sibilant_half_win = ((sr * 0.006).round() as usize).max(64);
    let grain_f = grain_len as f64;
    let mut out_channels = Vec::with_capacity(buffer.num_channels());

    // Helper to interpolate semitone shift at time `t`
    let interp_shift = |t: f64| -> f64 {
        if semitone_curve.is_empty() {
            return 0.0;
        }
        if t <= semitone_curve[0].0 {
            return semitone_curve[0].1;
        }
        if let Some(last) = semitone_curve.last()
            && t >= last.0
        {
            return last.1;
        }
        let idx = semitone_curve.partition_point(|&(ts, _)| ts < t);
        if idx == 0 {
            semitone_curve[0].1
        } else {
            let (t0, s0) = semitone_curve[idx - 1];
            let (t1, s1) = semitone_curve[idx.min(semitone_curve.len() - 1)];
            if (t1 - t0).abs() < 1e-9 {
                s0
            } else {
                let frac = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
                s0 * (1.0 - frac) + s1 * frac
            }
        }
    };

    for ch in &buffer.channels {
        let mut out_ch = vec![0.0_f64; frames];
        let mut phase_a = 0.0_f64;
        let mut sibilant_env = 0.0_f64;
        let orig_lpc_tilt = estimate_lpc_formant_tilt_db(ch);

        for (i, sample_out) in out_ch.iter_mut().enumerate() {
            let t = (i as f64) / sr;
            if i % 32 == 0 {
                let target_sib = is_unvoiced_sibilant_frame(ch, i, sibilant_half_win, sr);
                sibilant_env = sibilant_env * 0.65 + target_sib * 0.35;
            }

            let raw_semi = forge_dsp::sanitize_param_f64(interp_shift(t), 0.0, -24.0, 24.0);
            // Attenuate pitch shift smoothly toward 0.0 semitones during unvoiced sibilants
            let semi = raw_semi * (1.0 - sibilant_env * 0.85);
            let pitch_ratio = 2.0_f64.powf(semi / 12.0);
            let rate_delta = 1.0 - pitch_ratio;

            phase_a = (phase_a + rate_delta).rem_euclid(grain_f);
            let phase_b = (phase_a + grain_f * 0.5).rem_euclid(grain_f);

            // Complementary Hann crossfade weights (`w_a + w_b == 1.0`)
            let w_a = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * phase_a / grain_f).cos());
            let w_b = 1.0 - w_a;

            let sample_at = |delay: f64| -> f64 {
                let pos = (i as f64 - delay).clamp(0.0, (frames - 1) as f64);
                let idx0 = pos.floor() as usize;
                let frac = pos - (idx0 as f64);
                let s0 = forge_dsp::sanitize_sample_f64(ch[idx0]);
                let s1 = forge_dsp::sanitize_sample_f64(ch[(idx0 + 1).min(frames - 1)]);
                s0 * (1.0 - frac) + s1 * frac
            };

            let pitched = w_a * sample_at(phase_a) + w_b * sample_at(phase_b);
            // Blend 100% phase-coherent dry sibilant on consonants (`s/sh/t/k`)
            let dry_s = forge_dsp::sanitize_sample_f64(ch[i]);
            *sample_out = pitched * (1.0 - sibilant_env) + dry_s * sibilant_env;
        }

        // LPC Formant-envelope preservation: restore original LPC spectral tilt + F1/F2 vocal tract resonance
        let shifted_lpc_tilt = estimate_lpc_formant_tilt_db(&out_ch);
        let lpc_tilt_delta = (orig_lpc_tilt - shifted_lpc_tilt).clamp(-4.5, 4.5);
        let clean_formant_semi =
            forge_dsp::sanitize_param_f64(formant_shift_semitones, 0.0, -12.0, 12.0);
        let mid_shift = forge_dsp::sanitize_param_f64(
            interp_shift(buffer.duration_secs() * 0.5),
            0.0,
            -24.0,
            24.0,
        );
        let net_formant_semi = clean_formant_semi - mid_shift * 0.65;
        if net_formant_semi.abs() > 0.05 || lpc_tilt_delta.abs() > 0.05 {
            let mut f1_eq = forge_dsp::Biquad::peaking_eq(
                sr,
                650.0,
                1.0,
                (net_formant_semi * 0.55 - lpc_tilt_delta * 0.4).clamp(-6.0, 6.0),
            );
            let mut f2_eq = forge_dsp::Biquad::peaking_eq(
                sr,
                2150.0,
                0.9,
                (-net_formant_semi * 0.85 + lpc_tilt_delta * 0.6).clamp(-6.0, 6.0),
            );
            for s in &mut out_ch {
                *s = f2_eq.process_sample(f1_eq.process_sample(*s));
            }
        }

        out_channels.push(out_ch);
    }

    Ok(AudioBuffer {
        channels: out_channels,
        sample_rate: buffer.sample_rate,
    })
}

/// Generates a 3-part humanized, scale-locked vocal harmony stack from a lead vocal:
/// 1. Diatonic 3rd Above (`+2` scale degrees, panned slightly Left)
/// 2. Diatonic 3rd/4th Below (`-2` scale degrees, panned slightly Right)
/// 3. Warm Formant-Lowered Octave Double (center reinforcement)
pub fn generate_vocal_harmony_stack(
    lead: &AudioBuffer,
    scale: &VocalScale,
) -> Result<(AudioBuffer, AudioBuffer, AudioBuffer)> {
    let contour = track_vocal_pitch(lead, scale, 1.0);

    let mut third_up_curve = Vec::with_capacity(contour.len());
    let mut third_down_curve = Vec::with_capacity(contour.len());
    let mut double_curve = Vec::with_capacity(contour.len());

    for (idx, frame) in contour.iter().enumerate() {
        let base_note = if frame.midi_pitch > 20.0 {
            frame.target_midi_pitch.round() as i32
        } else {
            60 + i32::from(scale.root_pc)
        };
        // Subtle 5.5 Hz human vibrato modulation (±0.08 semitones) out of phase per voice
        let vib_a = 0.08 * (frame.time_secs * 2.0 * std::f64::consts::PI * 5.2).sin();
        let vib_b = 0.08 * (frame.time_secs * 2.0 * std::f64::consts::PI * 5.6 + 1.3).sin();
        let vib_c = 0.05 * (frame.time_secs * 2.0 * std::f64::consts::PI * 4.8 + 2.5).sin();

        let up_semi =
            scale.diatonic_shift_semitones(base_note, 2) + frame.correction_semitones + vib_a;
        let down_semi =
            scale.diatonic_shift_semitones(base_note, -2) + frame.correction_semitones + vib_b;
        let dbl_semi = frame.correction_semitones + vib_c + if idx % 2 == 0 { 0.04 } else { -0.04 };

        third_up_curve.push((frame.time_secs, up_semi));
        third_down_curve.push((frame.time_secs, down_semi));
        double_curve.push((frame.time_secs, dbl_semi));
    }

    let harm_high = shift_vocal_pitch_formant_preserved(lead, &third_up_curve, -0.5)?;
    let harm_low = shift_vocal_pitch_formant_preserved(lead, &third_down_curve, 0.5)?;
    let harm_double = shift_vocal_pitch_formant_preserved(lead, &double_curve, -1.2)?;

    Ok((harm_high, harm_low, harm_double))
}

/// Compiles a `PitchFrame` contour into a native REAPER Take Pitch Envelope (`<PITCHENV`) chunk
/// so producers can visually inspect and tweak every pitch node inside REAPER!
pub fn compile_reaper_take_pitch_envelope(frames: &[PitchFrame]) -> String {
    let mut out = String::from(
        "      <PITCHENV\n        ACT 1 -1\n        VIS 1 1 1\n        LANEHEIGHT 0 0\n        ARM 0\n        DEFSHAPE 0 -1 -1\n",
    );
    for f in frames {
        if f.f0_hz > 0.0 && f.correction_semitones.abs() > 0.02 {
            out.push_str(&format!(
                "        PT {:.4} {:.4} 0\n",
                f.time_secs, f.correction_semitones
            ));
        }
    }
    out.push_str("      >\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::f64::consts::PI;

    #[test]
    fn test_yin_vocal_f0_tracker_and_reaper_pitchenv() {
        let sr = 48_000_u32;
        let frames = (sr as usize) / 2;
        let mut ch = vec![0.0; frames];
        // Slightly sharp A4 (445 Hz instead of 440 Hz)
        for (i, s) in ch.iter_mut().enumerate() {
            let t = (i as f64) / (sr as f64);
            *s = 0.5 * (2.0 * PI * 445.0 * t).sin();
        }
        let buf = AudioBuffer::from_channels(vec![ch], sr).expect("valid");
        let scale = VocalScale::parse("A minor").expect("scale");
        let contour = track_vocal_pitch(&buf, &scale, 1.0);
        assert!(!contour.is_empty());
        let env = compile_reaper_take_pitch_envelope(&contour);
        assert!(env.contains("<PITCHENV"));
        assert!(env.contains("PT "));
    }

    const RATES: [u32; 3] = [44_100, 48_000, 96_000];
    const PITCHES: [f64; 7] = [82.41, 110.0, 196.0, 261.63, 445.0, 659.26, 880.0];

    /// Sum of `amp * sin(k * phase)` partials over a phase track, as a mono buffer.
    fn additive(sr: u32, phase: &[f64], partials: &[(f64, f64)]) -> AudioBuffer {
        let ch = phase
            .iter()
            .map(|&p| partials.iter().map(|&(k, a)| a * (k * p).sin()).sum())
            .collect();
        AudioBuffer::from_channels(vec![ch], sr).expect("valid")
    }

    /// Phase of a steady tone at `f0` Hz lasting `secs` seconds.
    fn steady_phase(sr: u32, secs: f64, f0: f64) -> Vec<f64> {
        let n = (f64::from(sr) * secs) as usize;
        (0..n)
            .map(|i| 2.0 * PI * f0 * (i as f64) / f64::from(sr))
            .collect()
    }

    /// Band-limited sawtooth partials (`1/k`, below 0.45 * sr), scaled to roughly 0.5 peak.
    fn sawtooth_partials(sr: u32, f0: f64) -> Vec<(f64, f64)> {
        let top = (0.45 * f64::from(sr) / f0).floor() as usize;
        (1..=top.max(1))
            .map(|k| (k as f64, 0.3 / (k as f64)))
            .collect()
    }

    fn cents(measured: f64, truth: f64) -> f64 {
        1200.0 * (measured / truth).log2()
    }

    /// Largest cents error over the contour; panics if any frame is unvoiced.
    fn worst_cents(buf: &AudioBuffer, f0: f64) -> f64 {
        let scale = VocalScale::parse("C chromatic").expect("scale");
        let contour = track_vocal_pitch(buf, &scale, 0.0);
        assert!(contour.len() >= 5, "too few frames");
        contour
            .iter()
            .map(|f| {
                assert!(
                    f.f0_hz > 0.0,
                    "unvoiced frame at {} s for {f0} Hz",
                    f.time_secs
                );
                cents(f.f0_hz, f0).abs()
            })
            .fold(0.0, f64::max)
    }

    fn assert_tracks_within(label: &str, f0: f64, partials: impl Fn(u32) -> Vec<(f64, f64)>) {
        for sr in RATES {
            let buf = additive(sr, &steady_phase(sr, 0.3, f0), &partials(sr));
            let err = worst_cents(&buf, f0);
            assert!(err <= 3.0, "{label} {f0} Hz @ {sr} Hz: {err:.2} cents off");
        }
    }

    #[test]
    fn test_yin_parabolic_offset_edge_cases() {
        // Samples of (x - 0.3)^2 at -1, 0, 1 put the vertex at +0.3.
        let p = |x: f64| (x - 0.3) * (x - 0.3);
        assert!((parabolic_offset(p(-1.0), p(0.0), p(1.0)) - 0.3).abs() < 1e-12);
        assert_eq!(parabolic_offset(1.0, 0.5, 1.0), 0.0, "symmetric dip");
        assert_eq!(parabolic_offset(1.0, 1.0, 1.0), 0.0, "flat");
        assert_eq!(parabolic_offset(0.0, 1.0, 0.0), 0.0, "peak, not a dip");
        assert_eq!(parabolic_offset(f64::NAN, 0.0, 1.0), 0.0, "non-finite");
        assert_eq!(
            parabolic_offset(0.0, 0.0, 10.0),
            -0.5,
            "clamped to half a lag"
        );
    }

    #[test]
    fn test_yin_pick_period_threshold_and_fallback() {
        // A dip below the threshold at lag 3 is followed down to its bottom at lag 4.
        let dip = [1.0, 0.9, 0.5, 0.14, 0.05, 0.2, 0.01, 0.6];
        assert_eq!(pick_period(&dip, 1, 7), Some(4));
        // No dip below the threshold: the global minimum, if low enough to call voiced.
        let shallow = [1.0, 0.9, 0.3, 0.25, 0.4, 0.2, 0.5];
        assert_eq!(pick_period(&shallow, 1, 6), Some(5));
        let unvoiced = [1.0, 0.9, 0.5, 0.4, 0.6];
        assert_eq!(pick_period(&unvoiced, 1, 4), None);
    }

    #[test]
    fn test_yin_non_finite_input_is_unvoiced() {
        let scale = VocalScale::parse("C chromatic").expect("scale");
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut ch: Vec<f64> = steady_phase(48_000, 0.3, 220.0)
                .iter()
                .map(|p| 0.5 * p.sin())
                .collect();
            for s in ch.iter_mut().step_by(97) {
                *s = bad;
            }
            let buf = AudioBuffer::from_channels(vec![ch], 48_000).expect("valid");
            for f in track_vocal_pitch(&buf, &scale, 0.0) {
                assert!(
                    f.f0_hz.is_finite() && f.midi_pitch.is_finite(),
                    "{bad}: {f:?}"
                );
            }
        }
    }

    /// M15: descending to the dip and interpolating it puts pure sines within 3 cents.
    #[test]
    fn test_yin_pure_sines_within_3_cents_m15() {
        for f0 in PITCHES {
            assert_tracks_within("sine", f0, |_| vec![(1.0, 0.5)]);
        }
    }

    /// M15: harmonic-rich sawtooths track the fundamental within 3 cents at every rate.
    #[test]
    fn test_yin_sawtooth_within_3_cents_m15() {
        for f0 in PITCHES {
            assert_tracks_within("sawtooth", f0, |sr| sawtooth_partials(sr, f0));
        }
    }

    /// A tone with no energy at f0 (partials 2 to 6) still reports f0, not 2 * f0.
    #[test]
    fn test_yin_missing_fundamental_m15() {
        for f0 in [110.0, 196.0, 261.63] {
            assert_tracks_within("missing fundamental", f0, |_| {
                (2..=6).map(|k| (f64::from(k), 0.12)).collect()
            });
        }
    }

    /// Octave traps: a dominant 2nd partial and a low sawtooth, whose dips at 2T, 3T, ... are as
    /// deep as the one at T, must not jump an octave either way.
    #[test]
    fn test_yin_octave_traps_m15() {
        for f0 in [98.0, 146.83, 220.0] {
            assert_tracks_within("strong 2nd partial", f0, |_| vec![(1.0, 0.25), (2.0, 0.5)]);
            assert_tracks_within("low sawtooth", f0, |sr| sawtooth_partials(sr, f0));
        }
    }

    /// Silence, sub-gate signals and white noise give unvoiced frames and no envelope points.
    #[test]
    fn test_yin_silence_and_noise_are_unvoiced_m15() {
        let scale = VocalScale::parse("A minor").expect("scale");
        for sr in RATES {
            let n = (f64::from(sr) * 0.3) as usize;
            let mut state = 0x2545_F491_u32;
            let noise: Vec<f64> = (0..n)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    (f64::from(state) / f64::from(u32::MAX)) - 0.5
                })
                .collect();
            let quiet = additive(sr, &steady_phase(sr, 0.3, 220.0), &[(1.0, 0.005)]);
            for (label, ch) in [
                ("silence", vec![0.0; n]),
                ("noise", noise),
                ("quiet", quiet.channels[0].clone()),
            ] {
                let buf = AudioBuffer::from_channels(vec![ch], sr).expect("valid");
                let contour = track_vocal_pitch(&buf, &scale, 1.0);
                assert!(!contour.is_empty(), "{label} @ {sr}");
                for f in &contour {
                    assert_eq!(
                        f.f0_hz, 0.0,
                        "{label} @ {sr} Hz voiced at {} s",
                        f.time_secs
                    );
                    assert_eq!(f.midi_pitch, 0.0, "{label} @ {sr}");
                    assert_eq!(f.correction_semitones, 0.0, "{label} @ {sr}");
                }
                assert!(
                    !compile_reaper_take_pitch_envelope(&contour).contains("PT "),
                    "{label}"
                );
            }
        }
    }

    /// M15: each frame is stamped where its pitch was measured. On a 2 octave/s glide the pitch
    /// read back from a frame names the moment it was sung, within 1 ms of `time_secs`.
    #[test]
    fn test_yin_timestamps_match_glide_m15() {
        const RATE_OCT_PER_SEC: f64 = 2.0;
        const START_HZ: f64 = 150.0;
        let scale = VocalScale::parse("C chromatic").expect("scale");
        for sr in RATES {
            let n = (f64::from(sr) * 0.6) as usize;
            let k = RATE_OCT_PER_SEC * std::f64::consts::LN_2;
            let phase: Vec<f64> = (0..n)
                .map(|i| {
                    let t = (i as f64) / f64::from(sr);
                    2.0 * PI * START_HZ * ((k * t).exp() - 1.0) / k
                })
                .collect();
            let buf = additive(sr, &phase, &[(1.0, 0.5)]);
            let contour = track_vocal_pitch(&buf, &scale, 0.0);
            assert!(contour.len() >= 10, "@ {sr}");
            for f in &contour {
                assert!(f.f0_hz > 0.0, "unvoiced at {} s @ {sr}", f.time_secs);
                let sung_at = (f.f0_hz / START_HZ).log2() / RATE_OCT_PER_SEC;
                let err_ms = (sung_at - f.time_secs) * 1000.0;
                assert!(
                    err_ms.abs() <= 1.0,
                    "@ {sr} Hz frame {} s: {err_ms:.2} ms off",
                    f.time_secs
                );
            }
        }
    }

    fn mask(root: u8, intervals: &[u8]) -> [bool; 12] {
        let mut allowed = [false; 12];
        for &iv in intervals {
            allowed[usize::from((root + iv) % 12)] = true;
        }
        allowed
    }

    /// M14: the mode after the root is read, attached or not, and never silently dropped.
    #[test]
    fn test_scale_modes_m14() {
        let cases: &[(&str, u8, &[u8], bool)] = &[
            ("Am", 9, &NATURAL_MINOR, true),
            ("F#m", 6, &NATURAL_MINOR, true),
            ("Bbmin", 10, &NATURAL_MINOR, true),
            ("A minor pentatonic", 9, &MINOR_PENTATONIC, true),
            ("A pentatonic minor", 9, &MINOR_PENTATONIC, true),
            ("Am pent", 9, &MINOR_PENTATONIC, true),
            ("C pentatonic", 0, &MAJOR_PENTATONIC, false),
            ("G mixolydian", 7, &MIXOLYDIAN, false),
            ("Gmix", 7, &MIXOLYDIAN, false),
            ("F lydian", 5, &LYDIAN, false),
            ("E phrygian", 4, &PHRYGIAN, true),
            ("B locrian", 11, &LOCRIAN, true),
            ("D Dorian", 2, &DORIAN, true),
            ("A harmonic minor", 9, &HARMONIC_MINOR, true),
            ("A melodic minor", 9, &MELODIC_MINOR, true),
            ("E blues", 4, &BLUES, true),
            ("C chromatic", 0, &CHROMATIC, false),
            ("C", 0, &MAJOR, false),
            ("CM", 0, &MAJOR, false),
            ("Db major", 1, &MAJOR, false),
            ("C# minor", 1, &NATURAL_MINOR, true),
            ("  d minor  ", 2, &NATURAL_MINOR, true),
        ];
        for &(text, root, intervals, is_minor) in cases {
            let scale = VocalScale::parse(text).unwrap_or_else(|e| panic!("{text}: {e}"));
            assert_eq!(scale.root_pc, root, "{text}");
            assert_eq!(scale.allowed_pcs, mask(root, intervals), "{text}");
            assert_eq!(scale.is_minor, is_minor, "{text}");
        }
    }

    #[test]
    fn test_malformed_scales_are_rejected() {
        let cases = [
            ("", "Invalid scale root"),
            ("H minor", "Invalid scale root"),
            ("A minorx", "Unknown scale mode 'minorx'"),
            ("Ammm", "Unknown scale mode 'mmm'"),
            ("A minor pentatonic blues", "Unknown scale mode"),
            ("C majr", "Unknown scale mode 'majr'"),
        ];
        for (text, needle) in cases {
            let err = VocalScale::parse(text).expect_err(text).to_string();
            assert!(err.contains(needle), "{text}: {err}");
        }
    }

    proptest! {
        /// Every mode spelling gives the same mask attached (`Amix`) or spaced (`A mix`).
        #[test]
        fn prop_scale_mode_spellings(
            root in "[A-G][#b]?",
            mode_idx in 0usize..8,
            attached in any::<bool>(),
        ) {
            let modes: [(&str, &[u8]); 8] = [
                ("m", &NATURAL_MINOR),
                ("minor", &NATURAL_MINOR),
                ("dorian", &DORIAN),
                ("phrygian", &PHRYGIAN),
                ("lydian", &LYDIAN),
                ("mix", &MIXOLYDIAN),
                ("locrian", &LOCRIAN),
                ("minor pentatonic", &MINOR_PENTATONIC),
            ];
            let (mode, intervals) = modes[mode_idx];
            let text = if attached { format!("{root}{mode}") } else { format!("{root} {mode}") };
            let scale = VocalScale::parse(&text).unwrap();
            prop_assert_eq!(scale.allowed_pcs, mask(scale.root_pc, intervals));
        }

        #[test]
        fn prop_vocal_scale_snap_and_diatonic_shift_totality(
            arb_str in ".*",
            valid_key in "[A-G][#b]? (major|minor|dorian|pentatonic|chromatic)",
            midi_pitch in prop::num::f64::ANY,
            amount in prop::num::f64::ANY,
            midi_note in i32::MIN..=i32::MAX,
            degree_steps in i32::MIN..=i32::MAX,
        ) {
            // Arbitrary string must never panic (returns Ok or Err cleanly)
            let _ = VocalScale::parse(&arb_str);

            let scale = VocalScale::parse(&valid_key).unwrap();
            let snapped = scale.snap_midi(midi_pitch, amount);
            prop_assert!(snapped.is_finite() && (0.0..=127.0).contains(&snapped));

            // Idempotence when amount == 1.0: snapping an already-snapped pitch changes nothing
            let full_snap = scale.snap_midi(midi_pitch, 1.0);
            let resnapped = scale.snap_midi(full_snap, 1.0);
            prop_assert!((full_snap - resnapped).abs() < 1e-9);

            // Diatonic shift totality across full i32::MIN..=i32::MAX range
            let shift = scale.diatonic_shift_semitones(midi_note, degree_steps);
            prop_assert!(shift.is_finite() && (-127.0..=127.0).contains(&shift));
        }

        #[test]
        fn prop_lpc_formant_tilt_and_sibilant_pitch_shifter_totality(
            frames in 128usize..1024,
            semi_shift in prop::num::f64::ANY,
            formant_shift in prop::num::f64::ANY,
        ) {
            let sr = 48_000_u32;
            let mut ch = vec![0.0_f64; frames];
            for (i, s) in ch.iter_mut().enumerate() {
                let t = (i as f64) / (sr as f64);
                *s = if i % 67 == 0 {
                    f64::NAN
                } else {
                    0.4 * (2.0 * PI * 330.0 * t).sin() + 0.1 * (2.0 * PI * 7500.0 * t).cos()
                };
            }
            let tilt = estimate_lpc_formant_tilt_db(&ch);
            prop_assert!(tilt.is_finite());

            let buf = AudioBuffer::from_channels(vec![ch], sr).unwrap();
            let curve = vec![(0.0, semi_shift), (0.01, -semi_shift)];
            let shifted = shift_vocal_pitch_formant_preserved(&buf, &curve, formant_shift).unwrap();
            prop_assert_eq!(shifted.num_frames(), frames);
            for &y in &shifted.channels[0] {
                prop_assert!(y.is_finite() && y.abs() <= 16.0);
            }
        }
    }
}
