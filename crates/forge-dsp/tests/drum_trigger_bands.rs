//! The drum trigger's snare band must not fire on kick-only or hat-only audio.
//!
//! If broadband energy leaks into the snare band, every kick or hat transient also passes the
//! snare onset test and fires a spurious D1. The positive control checks the band still fires
//! on an actual snare-like hit.

use std::f64::consts::PI;

use forge_dsp::{Biquad, RealtimeDrumTrigger};

const SR: f64 = 48_000.0;
/// Spacing between hits, long enough for every detector to settle and leave its holdoff.
const HIT_SPACING: usize = 24_000;
const HITS: usize = 8;
/// Kick, snare and hat slots of `RealtimeDrumTrigger::process_stereo_sample`.
const KICK: usize = 0;
const SNARE: usize = 1;
const HAT: usize = 2;

/// Deterministic white noise in `[-1, 1]` from a 64-bit LCG.
fn noise(len: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            2.0 * ((state >> 11) as f64 / (1_u64 << 53) as f64) - 1.0
        })
        .collect()
}

/// A 55 Hz sine kick with a 150 ms decay, peaking at `amp`.
fn kick_hit(amp: f64) -> Vec<f64> {
    (0..HIT_SPACING)
        .map(|i| {
            let t = i as f64 / SR;
            amp * (-t / 0.15).exp() * (2.0 * PI * 55.0 * t).sin()
        })
        .collect()
}

/// White noise high-passed twice at 8 kHz with a 40 ms decay, scaled by `amp`.
fn hat_hit(amp: f64) -> Vec<f64> {
    let mut hp1 = Biquad::highpass(SR, 8_000.0, 0.707);
    let mut hp2 = Biquad::highpass(SR, 8_000.0, 0.707);
    noise(HIT_SPACING, 0x9e37_79b9_7f4a_7c15)
        .into_iter()
        .enumerate()
        .map(|(i, n)| {
            let t = i as f64 / SR;
            amp * (-t / 0.04).exp() * hp2.process_sample(hp1.process_sample(n))
        })
        .collect()
}

/// A 200 Hz body plus 1-2 kHz noise with a 90 ms decay, scaled by `amp`.
fn snare_hit(amp: f64) -> Vec<f64> {
    let mut bp = Biquad::bandpass(SR, 1_400.0, 1.0);
    noise(HIT_SPACING, 0x2545_f491_4f6c_dd1d)
        .into_iter()
        .enumerate()
        .map(|(i, n)| {
            let t = i as f64 / SR;
            let body = 0.5 * (2.0 * PI * 200.0 * t).sin();
            amp * (-t / 0.09).exp() * (body + 1.5 * bp.process_sample(n))
        })
        .collect()
}

/// Feeds `HITS` repeats of `hit` and counts triggers per slot.
fn count_triggers(hit: &[f64], threshold_db: f64) -> [usize; 3] {
    let mut trig = RealtimeDrumTrigger::new(SR);
    let mut counts = [0; 3];
    for _ in 0..HITS {
        for &s in hit {
            let out = trig.process_stereo_sample(s as f32, s as f32, threshold_db);
            for (slot, fired) in out.iter().enumerate() {
                counts[slot] += usize::from(fired.is_some());
            }
        }
    }
    counts
}

#[test]
fn kick_only_audio_never_fires_the_snare() {
    // At the plugin's default -24 dB threshold.
    for amp in [0.25, 0.5, 0.9] {
        let counts = count_triggers(&kick_hit(amp), -24.0);
        assert_eq!(counts[KICK], HITS, "kick {amp}: {counts:?}");
        assert_eq!(counts[SNARE], 0, "kick {amp} fired the snare: {counts:?}");
    }
}

#[test]
fn hat_only_audio_never_fires_the_snare() {
    for amp in [0.25, 0.5, 0.9] {
        for threshold_db in [-24.0, -36.0, -48.0] {
            let counts = count_triggers(&hat_hit(amp), threshold_db);
            assert_eq!(
                counts[HAT], HITS,
                "hat {amp} at {threshold_db} dB: {counts:?}"
            );
            assert_eq!(
                counts[SNARE], 0,
                "hat {amp} at {threshold_db} dB fired the snare: {counts:?}"
            );
        }
    }
}

#[test]
fn snare_hits_still_fire_the_snare() {
    for amp in [0.25, 0.5, 0.9] {
        let counts = count_triggers(&snare_hit(amp), -24.0);
        assert!(counts[SNARE] >= HITS, "snare {amp}: {counts:?}");
    }
}
