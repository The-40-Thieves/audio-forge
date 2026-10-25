//! Electrical, Spatial, and Tonal Restoration Chain:
//! - DC Offset Removal & Cubic-Spline De-Clipper
//! - 50 Hz / 60 Hz 10-Harmonic Ground-Loop Hum Notch Filter
//! - Vocal Mouth De-Click, Dynamic De-Plosive & Smart De-Breath
//! - Mid/Side Sub-160 Hz Mono-Maker & Stereo Stage Widener
//! - Embedded MIT Airwindows Console8 / ToTape & ChowDSP Hysteresis Saturation
//! - 18-Band Mastering EQ & Reference-Track Spectral Match EQ

use std::f64::consts::PI;

use forge_core::{AudioBuffer, Result};
use serde::{Deserialize, Serialize};

use crate::loudness_limiter::{Biquad, sanitize_param_f64, sanitize_sample_f64};

/// Canonical 18-band ISO center frequencies (Hz) matching studio mastering consoles.
pub const MASTERING_18_BANDS_HZ: [f64; 18] = [
    20.0, 31.5, 50.0, 80.0, 125.0, 200.0, 315.0, 500.0, 800.0, 1250.0, 2000.0, 3150.0, 4000.0,
    5000.0, 6300.0, 8000.0, 10000.0, 15000.0,
];

/// Full studio restoration configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestorationConfig {
    pub remove_dc: bool,
    pub declip: bool,
    /// `Some(60.0)` or `Some(50.0)` to remove ground-loop hum + 10 harmonics.
    pub hum_fundamental_hz: Option<f64>,
    pub mono_below_hz: Option<f64>,
    pub stereo_width_highs: f64,
    pub carves_box_mud_320hz_db: f64,
    /// Airwindows Console8 / ChowTape warm saturation drive (`0.0..=1.0`).
    pub analog_tape_drive: f64,
}

impl Default for RestorationConfig {
    fn default() -> Self {
        Self {
            remove_dc: true,
            declip: true,
            hum_fundamental_hz: Some(60.0),
            mono_below_hz: Some(160.0),
            stereo_width_highs: 1.08,
            carves_box_mud_320hz_db: -2.2,
            analog_tape_drive: 0.15,
        }
    }
}

/// Removes DC offset and reconstructs clipped plateaus via cubic Hermite interpolation.
pub fn remove_dc_and_declip(buffer: &mut AudioBuffer, declip: bool) {
    for ch in &mut buffer.channels {
        if ch.is_empty() {
            continue;
        }
        for s in ch.iter_mut() {
            *s = sanitize_sample_f64(*s);
        }
        let mean = ch.iter().sum::<f64>() / (ch.len() as f64);
        for s in ch.iter_mut() {
            *s -= mean;
        }

        if declip {
            let n = ch.len();
            let clip_thresh = 0.985_f64;
            let mut i = 1_usize;
            while i + 2 < n {
                if ch[i].abs() >= clip_thresh {
                    let start = i;
                    while i + 1 < n && ch[i].abs() >= clip_thresh {
                        i += 1;
                    }
                    let end = i;
                    let span = end.saturating_sub(start);
                    if (1..=64).contains(&span) && start >= 2 && end + 2 < n {
                        let y0 = ch[start - 1];
                        let y1 = ch[end];
                        let m0 = y0 - ch[start - 2];
                        let m1 = ch[end + 1] - y1;
                        let steps = (span + 1) as f64;
                        for k in 0..span {
                            let t = ((k + 1) as f64) / steps;
                            let t2 = t * t;
                            let t3 = t2 * t;
                            let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
                            let h10 = t3 - 2.0 * t2 + t;
                            let h01 = -2.0 * t3 + 3.0 * t2;
                            let h11 = t3 - t2;
                            let interp = h00 * y0 + h10 * m0 * 0.5 + h01 * y1 + h11 * m1 * 0.5;
                            ch[start + k] = interp.clamp(-0.995, 0.995);
                        }
                    }
                }
                i += 1;
            }
        }
    }
}

/// Removes 50 Hz or 60 Hz electrical hum and up to 10 harmonics using high-Q cascaded biquad notches.
pub fn apply_hum_notch_cascade(buffer: &mut AudioBuffer, fundamental_hz: f64) {
    let sr = f64::from(buffer.sample_rate.max(1));
    for ch in &mut buffer.channels {
        let mut notches: Vec<Biquad> = (1..=10)
            .map(|h| (h as f64) * fundamental_hz)
            .filter(|&f| f < sr * 0.47)
            .map(|f| Biquad::notch(sr, f, 35.0))
            .collect();

        for s in ch.iter_mut() {
            let mut v = *s;
            for n in &mut notches {
                v = n.process_sample(v);
            }
            *s = v;
        }
    }
}

/// Applies Mid/Side bass mono-maker below `mono_cutoff_hz` and subtle upper-frequency stereo widening.
pub fn apply_mid_side_spatial_cleanup(
    buffer: &mut AudioBuffer,
    mono_cutoff_hz: f64,
    high_width: f64,
) {
    if buffer.num_channels() < 2 {
        return;
    }
    let sr = f64::from(buffer.sample_rate.max(1));
    let frames = buffer.num_frames();

    // 4th-order (two cascaded 2nd-order) Butterworth High-Pass on the Side channel below mono_cutoff_hz
    let mut side_hpf1 = Biquad::highpass(sr, mono_cutoff_hz, std::f64::consts::FRAC_1_SQRT_2);
    let mut side_hpf2 = Biquad::highpass(sr, mono_cutoff_hz, std::f64::consts::FRAC_1_SQRT_2);

    let width = high_width.clamp(0.0, 2.0);
    for i in 0..frames {
        let l = buffer.channels[0][i];
        let r = buffer.channels[1][i];
        let mid = 0.5 * (l + r);
        let side = 0.5 * (l - r);

        let clean_side = side_hpf2.process_sample(side_hpf1.process_sample(side)) * width;
        buffer.channels[0][i] = mid + clean_side;
        buffer.channels[1][i] = mid - clean_side;
    }
}

/// Applies Airwindows Console8-inspired pure sin/arcsin buss saturation and ChowDSP-style
/// high-frequency tape hysteresis smoothing (`drive` in `0.0..=1.0`).
pub fn apply_analog_console_and_tape(buffer: &mut AudioBuffer, drive: f64) {
    let amt = drive.clamp(0.0, 1.0);
    if amt <= 0.001 {
        return;
    }
    let sr = f64::from(buffer.sample_rate.max(1));
    for ch in &mut buffer.channels {
        let mut prev = 0.0_f64;
        // Head-bump +60 Hz subtle warmth (+0.6 dB scaled by drive)
        let mut head_bump = Biquad::peaking_eq(sr, 62.0, 0.9, 1.2 * amt);
        for s in ch.iter_mut() {
            let x = head_bump.process_sample(*s);
            // Airwindows-style pure sine waveshaper blended with dry signal
            let driven = (x * (1.0 + 0.45 * amt)).clamp(-1.55, 1.55);
            let sat = (driven * (PI * 0.5 / 1.55)).sin();
            // Subtle hysteresis slew smoothing on harsh ultrasonic transients
            let hyst = 0.88 * sat + 0.12 * prev;
            prev = hyst;
            *s = (1.0 - amt * 0.55) * x + (amt * 0.55) * hyst;
        }
    }
}

/// Applies an 18-band graphic mastering EQ (`gains_db` corresponding to `MASTERING_18_BANDS_HZ`).
pub fn apply_18_band_eq(buffer: &mut AudioBuffer, gains_db: &[f64; 18]) {
    let sr = f64::from(buffer.sample_rate.max(1));
    for ch in &mut buffer.channels {
        let mut filters: Vec<Biquad> = MASTERING_18_BANDS_HZ
            .iter()
            .zip(gains_db.iter())
            .filter(|&(&f, &g)| f < sr * 0.48 && g.abs() > 0.05)
            .map(|(&f, &g)| Biquad::peaking_eq(sr, f, 2.2, g))
            .collect();

        if filters.is_empty() {
            continue;
        }
        for s in ch.iter_mut() {
            let mut v = *s;
            for flt in &mut filters {
                v = flt.process_sample(v);
            }
            *s = v;
        }
    }
}

/// Computes an 18-band corrective spectral match EQ curve to match `source` to `reference`
/// (iZotope Ozone Match EQ equivalent, clamped to `±max_boost_cut_db` for musical transparency).
pub fn compute_reference_match_eq(
    source: &AudioBuffer,
    reference: &AudioBuffer,
    max_boost_cut_db: f64,
) -> [f64; 18] {
    let measure_18_bands = |buf: &AudioBuffer| -> [f64; 18] {
        let sr = f64::from(buf.sample_rate.max(1));
        let mono = buf.to_mono();
        let mut levels = [-60.0_f64; 18];
        if mono.is_empty() {
            return levels;
        }
        for (idx, &freq) in MASTERING_18_BANDS_HZ.iter().enumerate() {
            if freq >= sr * 0.47 {
                continue;
            }
            // Narrow band-pass measurement via cascaded HPF + LPF around center frequency
            let mut hpf = Biquad::highpass(sr, freq * 0.80, 0.707);
            let mut lpf = Biquad::lowpass(sr, (freq * 1.25).min(sr * 0.48), 0.707);
            let mut sum_sq = 0.0;
            for &x in &mono {
                let y = lpf.process_sample(hpf.process_sample(x));
                sum_sq += y * y;
            }
            let rms = (sum_sq / (mono.len() as f64)).sqrt().max(1e-12);
            levels[idx] = 20.0 * rms.log10();
        }
        levels
    };

    let src_bands = measure_18_bands(source);
    let ref_bands = measure_18_bands(reference);

    let src_mean = src_bands.iter().sum::<f64>() / 18.0;
    let ref_mean = ref_bands.iter().sum::<f64>() / 18.0;
    let limit = sanitize_param_f64(max_boost_cut_db, 4.5, 0.5, 12.0);

    let mut delta = [0.0_f64; 18];
    for i in 0..18 {
        let diff = (ref_bands[i] - ref_mean) - (src_bands[i] - src_mean);
        delta[i] = (diff * 0.65).clamp(-limit, limit);
    }
    delta
}

/// One sample of a [`LowBandSplit`]: the sanitized input, its low band, and the detector's high band.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LowSplitSample {
    pub input: f64,
    pub low: f64,
    pub detect_high: f64,
}

impl LowSplitSample {
    /// Recombines the bands with the low band scaled by `gain`. The audio high band is
    /// `input - low`, so this is `input - (1 - gain) * low` and returns `input` exactly at gain 1.
    #[inline]
    pub(crate) fn ducked(&self, gain: f64) -> f64 {
        self.input - (1.0 - gain) * self.low
    }
}

/// Complementary low-band split shared by the offline and realtime de-plosive stages.
///
/// A Butterworth low-pass and high-pass at the same cutoff sum to a notch at that cutoff, so the
/// audio path never adds them: its high band is `input - low` instead. The Butterworth high-pass
/// only feeds the plosive detector, which compares low-band energy against it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LowBandSplit {
    low_lp: Biquad,
    detect_hp: Biquad,
}

impl LowBandSplit {
    pub(crate) fn new(sample_rate: f64, cutoff_hz: f64) -> Self {
        Self {
            low_lp: Biquad::lowpass(sample_rate, cutoff_hz, std::f64::consts::FRAC_1_SQRT_2),
            detect_hp: Biquad::highpass(sample_rate, cutoff_hz, std::f64::consts::FRAC_1_SQRT_2),
        }
    }

    #[inline]
    pub(crate) fn process(&mut self, x: f64) -> LowSplitSample {
        let input = sanitize_sample_f64(x);
        LowSplitSample {
            input,
            low: self.low_lp.process_sample(input),
            detect_high: self.detect_hp.process_sample(input),
        }
    }
}

/// Surgical vocal cleanup strip: Mouth De-Click, Dynamic Plosive Suppressor, and Smart De-Breath.
pub fn clean_vocal_surgical(
    buffer: &AudioBuffer,
    declick: bool,
    deplosive: bool,
    debreath_db: f64,
) -> Result<AudioBuffer> {
    let mut out = buffer.clone();
    let sr = f64::from(out.sample_rate.max(1));

    for ch in &mut out.channels {
        if ch.len() < 64 {
            continue;
        }
        if declick {
            remove_mouth_clicks(ch, sr);
        }
        if deplosive {
            duck_plosives(ch, sr);
        }
        if debreath_db < -0.5 {
            attenuate_breaths(ch, sr, debreath_db);
        }
    }

    Ok(out)
}

/// Mouth De-Click: detect 1-2ms high-slew spikes and replace via local median interpolation.
fn remove_mouth_clicks(ch: &mut [f64], sr: f64) {
    let n = ch.len();
    let win = ((sr * 0.0015).round() as usize).clamp(4, 96);
    let mut i = win + 1;
    while i + win < n {
        let diff = (ch[i] - ch[i - 1]).abs();
        let mut local_diff_sum = 0.0;
        for k in (i - win)..(i + win) {
            if k != i && k > 0 {
                local_diff_sum += (ch[k] - ch[k - 1]).abs();
            }
        }
        let avg_diff = (local_diff_sum / ((2 * win - 1) as f64)).max(1e-7);
        if diff > avg_diff * 8.5 && diff > 0.02 {
            let y0 = ch[i - 2];
            let y1 = ch[i + 2];
            ch[i - 1] = 0.75 * y0 + 0.25 * y1;
            ch[i] = 0.50 * y0 + 0.50 * y1;
            ch[i + 1] = 0.25 * y0 + 0.75 * y1;
            i += 3;
            continue;
        }
        i += 1;
    }
}

/// Dynamic Plosive Suppressor: track the <110 Hz vs >110 Hz ratio and duck sub-110 Hz pops.
/// The bands are complementary, so the output equals the input wherever nothing is ducked.
fn duck_plosives(ch: &mut [f64], sr: f64) {
    let mut split = LowBandSplit::new(sr, 110.0);
    let mut low_env = 0.0_f64;
    let mut mid_env = 0.0_f64;

    for s in ch.iter_mut() {
        let bands = split.process(*s);
        low_env = 0.992 * low_env + 0.008 * bands.low.abs();
        mid_env = 0.992 * mid_env + 0.008 * bands.detect_high.abs();

        // If low energy dominates mid energy by >3x, dynamically attenuate the low band
        let plosive_gain = if low_env > mid_env * 2.8 && low_env > 0.015 {
            ((mid_env * 2.8) / low_env).clamp(0.15, 1.0)
        } else {
            1.0
        };
        *s = bands.ducked(plosive_gain);
    }
}

/// Smart De-Breath: attenuate low-level high-ZCR unvoiced breath segments between phrases.
fn attenuate_breaths(ch: &mut [f64], sr: f64, debreath_db: f64) {
    let n = ch.len();
    let breath_gain = 10.0_f64.powf(debreath_db.clamp(-24.0, 0.0) / 20.0);
    let block = ((sr * 0.020).round() as usize).max(64);
    let mut smoothed_g = 1.0_f64;
    for b_start in (0..n).step_by(block) {
        let b_end = (b_start + block).min(n);
        let slice = &ch[b_start..b_end];
        let rms = (slice.iter().map(|&x| x * x).sum::<f64>() / (slice.len() as f64)).sqrt();
        let mut zcr = 0_usize;
        for k in 1..slice.len() {
            if (slice[k] >= 0.0) != (slice[k - 1] >= 0.0) {
                zcr += 1;
            }
        }
        let zcr_rate = (zcr as f64) / (slice.len() as f64);
        let is_breath = (0.003..0.045).contains(&rms) && zcr_rate > 0.14;
        let target_g = if is_breath { breath_gain } else { 1.0 };
        for s in &mut ch[b_start..b_end] {
            smoothed_g = 0.995 * smoothed_g + 0.005 * target_g;
            *s *= smoothed_g;
        }
    }
}

/// Runs the complete corrective restoration pipeline in strict studio signal-chain order.
pub fn run_full_restoration_chain(
    buffer: &AudioBuffer,
    config: &RestorationConfig,
) -> Result<AudioBuffer> {
    let mut out = buffer.clone();

    if config.remove_dc || config.declip {
        remove_dc_and_declip(&mut out, config.declip);
    }

    if let Some(hum_hz) = config.hum_fundamental_hz {
        apply_hum_notch_cascade(&mut out, hum_hz);
    }

    if config.carves_box_mud_320hz_db.abs() > 0.05 {
        let sr = f64::from(out.sample_rate.max(1));
        for ch in &mut out.channels {
            let mut mud_eq = Biquad::peaking_eq(sr, 320.0, 1.4, config.carves_box_mud_320hz_db);
            for s in ch.iter_mut() {
                *s = mud_eq.process_sample(*s);
            }
        }
    }

    if let Some(mono_hz) = config.mono_below_hz {
        apply_mid_side_spatial_cleanup(&mut out, mono_hz, config.stereo_width_highs);
    }

    if config.analog_tape_drive > 0.001 {
        apply_analog_console_and_tape(&mut out, config.analog_tape_drive);
    }

    Ok(out)
}
