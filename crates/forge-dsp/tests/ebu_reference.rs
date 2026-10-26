//! EBU Tech 3341 (loudness metering) and EBU Tech 3342 (loudness range) reference-signal
//! compliance tests for `audit_audio` / `compute_true_peak`.
//!
//! The reference signals are synthesized in-process instead of loading the EBU WAV test set:
//! every case is a stereo 1 kHz sine (or the specified true-peak sine) at 48 kHz, with identical
//! content in both channels, as described in the EBU documents.
//!
//! Tolerances follow the EBU "EBU Mode" requirements:
//! - Integrated / Momentary / Short-Term loudness: +/-0.1 LU
//! - Loudness Range: +/-1 LU
//! - True Peak: +0.2 / -0.4 dB

use std::f64::consts::PI;

use forge_core::AudioBuffer;
use forge_dsp::{audit_audio, compute_true_peak};

const SR: u32 = 48_000;
const LOUDNESS_TOL_LU: f64 = 0.1;
const LRA_TOL_LU: f64 = 1.0;

/// Builds a stereo buffer from `(level_dbfs, duration_secs)` segments of a continuous-phase
/// 1 kHz sine. Level is the sine's peak amplitude in dBFS, matching the EBU test-set definition.
fn stereo_sine_segments(segments: &[(f64, f64)]) -> AudioBuffer {
    let mut ch = Vec::new();
    let mut n: usize = 0;
    for &(level_dbfs, secs) in segments {
        let amp = 10.0_f64.powf(level_dbfs / 20.0);
        let frames = (secs * f64::from(SR)).round() as usize;
        for _ in 0..frames {
            ch.push(amp * (2.0 * PI * 1000.0 * n as f64 / f64::from(SR)).sin());
            n += 1;
        }
    }
    AudioBuffer::from_channels(vec![ch.clone(), ch], SR).expect("valid stereo buffer")
}

/// Stereo sine at `freq_hz` with initial phase `phase_deg` and peak amplitude `amp`.
///
/// A 10 ms raised-cosine fade is applied at both ends: an abrupt onset is a step, and any
/// band-limited reconstruction (including the BS.1770-4 interpolator) rings on it and overreads.
/// The EBU cases measure the steady-state inter-sample peak, which the fades leave untouched.
fn stereo_sine_phase(freq_hz: f64, phase_deg: f64, amp: f64, secs: f64) -> AudioBuffer {
    let frames = (secs * f64::from(SR)).round() as usize;
    let fade = (0.01 * f64::from(SR)) as usize;
    let phase = phase_deg.to_radians();
    let ch: Vec<f64> = (0..frames)
        .map(|i| {
            let edge = i.min(frames - 1 - i);
            let gain = if edge < fade {
                0.5 - 0.5 * (PI * edge as f64 / fade as f64).cos()
            } else {
                1.0
            };
            gain * amp * (2.0 * PI * freq_hz * i as f64 / f64::from(SR) + phase).sin()
        })
        .collect();
    AudioBuffer::from_channels(vec![ch.clone(), ch], SR).expect("valid stereo buffer")
}

fn assert_within(name: &str, measured: f64, expected: f64, tol: f64) {
    assert!(
        (measured - expected).abs() <= tol,
        "{name}: measured {measured:.3}, expected {expected:.1} +/- {tol}"
    );
}

// ---------------------------------------------------------------------------------------------
// EBU Tech 3341: Integrated loudness (cases 1-5)
// ---------------------------------------------------------------------------------------------

#[test]
fn tech3341_case1_integrated_minus23() {
    let r = audit_audio(&stereo_sine_segments(&[(-23.0, 20.0)]));
    assert_within(
        "3341 #1 integrated",
        r.integrated_lufs,
        -23.0,
        LOUDNESS_TOL_LU,
    );
}

#[test]
fn tech3341_case2_integrated_minus33() {
    let r = audit_audio(&stereo_sine_segments(&[(-33.0, 20.0)]));
    assert_within(
        "3341 #2 integrated",
        r.integrated_lufs,
        -33.0,
        LOUDNESS_TOL_LU,
    );
}

#[test]
fn tech3341_case3_relative_gate() {
    let r = audit_audio(&stereo_sine_segments(&[
        (-36.0, 10.0),
        (-23.0, 60.0),
        (-36.0, 10.0),
    ]));
    assert_within(
        "3341 #3 integrated",
        r.integrated_lufs,
        -23.0,
        LOUDNESS_TOL_LU,
    );
}

#[test]
fn tech3341_case4_absolute_and_relative_gate() {
    let r = audit_audio(&stereo_sine_segments(&[
        (-72.0, 10.0),
        (-36.0, 10.0),
        (-23.0, 60.0),
        (-36.0, 10.0),
        (-72.0, 10.0),
    ]));
    assert_within(
        "3341 #4 integrated",
        r.integrated_lufs,
        -23.0,
        LOUDNESS_TOL_LU,
    );
}

#[test]
fn tech3341_case5_level_steps() {
    let r = audit_audio(&stereo_sine_segments(&[
        (-26.0, 20.0),
        (-20.0, 20.1),
        (-26.0, 20.0),
    ]));
    assert_within(
        "3341 #5 integrated",
        r.integrated_lufs,
        -23.0,
        LOUDNESS_TOL_LU,
    );
}

// ---------------------------------------------------------------------------------------------
// EBU Tech 3341: Momentary and Short-Term loudness on a steady signal
// ---------------------------------------------------------------------------------------------

#[test]
fn tech3341_steady_momentary_and_short_term() {
    let r = audit_audio(&stereo_sine_segments(&[(-23.0, 20.0)]));
    assert_within(
        "3341 momentary max",
        r.momentary_max_lufs,
        -23.0,
        LOUDNESS_TOL_LU,
    );
    assert_within(
        "3341 short-term max",
        r.short_term_max_lufs,
        -23.0,
        LOUDNESS_TOL_LU,
    );
}

// ---------------------------------------------------------------------------------------------
// EBU Tech 3341: True Peak (cases 15-19), 48 kHz
// ---------------------------------------------------------------------------------------------

fn assert_true_peak(name: &str, buffer: &AudioBuffer, expected_dbtp: f64) {
    let (_, tp, _) = compute_true_peak(buffer);
    assert!(
        tp <= expected_dbtp + 0.2 && tp >= expected_dbtp - 0.4,
        "{name}: measured {tp:.3} dBTP, expected {expected_dbtp:.1} (+0.2 / -0.4)"
    );
}

#[test]
fn tech3341_case15_true_peak_fs4_0deg() {
    let amp = 10.0_f64.powf(-6.0 / 20.0);
    assert_true_peak(
        "3341 #15",
        &stereo_sine_phase(12_000.0, 0.0, amp, 1.0),
        -6.0,
    );
}

#[test]
fn tech3341_case16_true_peak_fs4_45deg() {
    let amp = 10.0_f64.powf(-6.0 / 20.0);
    assert_true_peak(
        "3341 #16",
        &stereo_sine_phase(12_000.0, 45.0, amp, 1.0),
        -6.0,
    );
}

#[test]
fn tech3341_case17_true_peak_fs6_60deg() {
    let amp = 10.0_f64.powf(-6.0 / 20.0);
    assert_true_peak(
        "3341 #17",
        &stereo_sine_phase(8_000.0, 60.0, amp, 1.0),
        -6.0,
    );
}

#[test]
fn tech3341_case18_true_peak_fs8_67_5deg() {
    let amp = 10.0_f64.powf(-6.0 / 20.0);
    assert_true_peak(
        "3341 #18",
        &stereo_sine_phase(6_000.0, 67.5, amp, 1.0),
        -6.0,
    );
}

#[test]
fn tech3341_case19_true_peak_plus3() {
    // Fs/4 at 45 degrees with sample peaks at 0 dBFS: inter-sample peak is +3.01 dBTP.
    let amp = 2.0_f64.sqrt();
    assert_true_peak(
        "3341 #19",
        &stereo_sine_phase(12_000.0, 45.0, amp, 1.0),
        3.0,
    );
}

// ---------------------------------------------------------------------------------------------
// EBU Tech 3342: Loudness Range (cases 1-4)
// ---------------------------------------------------------------------------------------------

#[test]
fn tech3342_case1_lra_10() {
    let r = audit_audio(&stereo_sine_segments(&[(-20.0, 20.0), (-30.0, 20.0)]));
    assert_within("3342 #1 LRA", r.loudness_range_lu, 10.0, LRA_TOL_LU);
}

#[test]
fn tech3342_case2_lra_5() {
    let r = audit_audio(&stereo_sine_segments(&[(-20.0, 20.0), (-15.0, 20.0)]));
    assert_within("3342 #2 LRA", r.loudness_range_lu, 5.0, LRA_TOL_LU);
}

#[test]
fn tech3342_case3_lra_20() {
    let r = audit_audio(&stereo_sine_segments(&[(-40.0, 20.0), (-20.0, 20.0)]));
    assert_within("3342 #3 LRA", r.loudness_range_lu, 20.0, LRA_TOL_LU);
}

#[test]
fn tech3342_case4_lra_15() {
    let r = audit_audio(&stereo_sine_segments(&[
        (-50.0, 20.0),
        (-35.0, 20.0),
        (-20.0, 20.0),
        (-35.0, 20.0),
        (-50.0, 20.0),
    ]));
    assert_within("3342 #4 LRA", r.loudness_range_lu, 15.0, LRA_TOL_LU);
}

// ---------------------------------------------------------------------------------------------
// True-peak timestamp and filter-tail behaviour
// ---------------------------------------------------------------------------------------------

/// Mono half-band sinc pulse (band-limited to Fs/4) peaking at 1.0 at fractional frame `t0`.
fn half_band_pulse(frames: usize, t0: f64) -> AudioBuffer {
    let ch: Vec<f64> = (0..frames)
        .map(|i| {
            let x = (i as f64 - t0) * 0.5;
            if x.abs() < 1e-12 {
                1.0
            } else {
                (PI * x).sin() / (PI * x)
            }
        })
        .collect();
    AudioBuffer::from_channels(vec![ch], SR).expect("valid mono buffer")
}

#[test]
fn true_peak_timestamp_resolves_inter_sample_position() {
    // The Annex 2 prototype is symmetric about upsampled tap 23.5, so its 4x output grid sits at
    // input positions k + 0.125 + m/4. Place the pulse on that grid to check the phase offset.
    let t0 = 1000.375;
    let (_, tp, ts) = compute_true_peak(&half_band_pulse(2048, t0));
    let frames = ts * f64::from(SR);
    assert!(
        tp.abs() < 0.2,
        "inter-sample pulse peak read {tp:.3} dBTP, expected ~0"
    );
    assert!(
        (frames - t0).abs() <= 0.0625,
        "peak frame {frames:.3}, expected {t0} +/- 0.0625"
    );
}

#[test]
fn true_peak_catches_inter_sample_peak_at_buffer_end() {
    // Peak sits between the 4th- and 3rd-to-last samples: the interpolator only reaches it
    // through the zero-padded filter tail.
    let frames = 2048;
    let t0 = frames as f64 - 3.5;
    let (sp, tp, ts) = compute_true_peak(&half_band_pulse(frames, t0));
    assert!(
        tp > sp + 0.3,
        "tail peak {tp:.3} dBTP not above sample peak {sp:.3} dBFS"
    );
    assert!(
        (ts * f64::from(SR) - t0).abs() <= 0.25,
        "tail peak frame {:.3}, expected {t0}",
        ts * f64::from(SR)
    );
}

// ---------------------------------------------------------------------------------------------
// Short-term max and silence (issues #19 and #16)
// ---------------------------------------------------------------------------------------------

#[test]
fn short_term_max_is_not_set_by_a_single_momentary_burst() {
    // A 400 ms burst 20 dB over a quiet bed: it sets the momentary max, but every 3 s window
    // averages it with 2.6 s of the bed.
    let r = audit_audio(&stereo_sine_segments(&[
        (-40.0, 5.0),
        (-20.0, 0.4),
        (-40.0, 5.0),
    ]));
    assert!(
        r.short_term_max_lufs < r.momentary_max_lufs - 5.0,
        "short-term max {:.2} LUFS follows momentary max {:.2} LUFS",
        r.short_term_max_lufs,
        r.momentary_max_lufs
    );
}

#[test]
fn digital_silence_has_no_fizz_or_band_readings() {
    let silence = AudioBuffer::from_channels(vec![vec![0.0; 10 * SR as usize]; 2], SR)
        .expect("valid stereo buffer");
    let r = audit_audio(&silence);
    assert_eq!(r.ai_fizz_index, 0.0);
    for (name, db) in [
        ("hum", r.hum_50_60hz_db),
        ("mud", r.box_mud_200_500hz_db),
        ("harshness", r.harshness_2_6khz_db),
        ("sibilance", r.sibilance_5_14khz_db),
    ] {
        assert_eq!(db, forge_dsp::SILENT_BAND_DB, "{name}");
    }
    assert!(r.recommendations.is_empty(), "{:?}", r.recommendations);
}

#[test]
fn a_loud_clip_shorter_than_one_gating_block_is_not_treated_as_silence() {
    // 50 ms of a 4.2 kHz whistle: no 100 ms block, so integrated loudness reads -70 LUFS, but
    // the clip is loud and long enough for the 2048-point spectral analysis.
    let frames = (0.05 * f64::from(SR)) as usize;
    let ch: Vec<f64> = (0..frames)
        .map(|n| 0.5 * (2.0 * PI * 4200.0 * n as f64 / f64::from(SR)).sin())
        .collect();
    let r = audit_audio(&AudioBuffer::from_channels(vec![ch.clone(), ch], SR).expect("buffer"));
    assert!(
        r.harshness_2_6khz_db > forge_dsp::SILENT_BAND_DB + 50.0,
        "{}",
        r.harshness_2_6khz_db
    );
    assert!(r.ai_fizz_index > 0.0, "fizz index {}", r.ai_fizz_index);
}
