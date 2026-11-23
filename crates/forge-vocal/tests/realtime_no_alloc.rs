//! Verifies that `RealtimeVocalTune::process_stereo_sample()` and same-rate `reset()` perform
//! strictly **zero heap allocations** across 48 kHz, 96 kHz, and 192 kHz.

use std::f64::consts::PI;

use assert_no_alloc::{AllocDisabler, assert_no_alloc, violation_count};
use forge_vocal::{RealtimeVocalTune, VocalTuneScaleMode};

#[global_allocator]
static ALLOC: AllocDisabler = AllocDisabler;

#[test]
fn realtime_vocal_tune_process_and_reset_are_zero_alloc() {
    for &sr in &[48_000.0_f64, 96_000.0_f64, 192_000.0_f64] {
        let mut tuner = RealtimeVocalTune::new(sr);
        tuner.set_key_and_scale(0, VocalTuneScaleMode::Major);
        tuner.set_retune_speed_ms(10.0);
        tuner.set_retune_amount(1.0);
        tuner.set_formant_shift_st(2.0);

        let frames = (sr * 0.05) as usize;
        let before = violation_count();
        assert_no_alloc(|| {
            for i in 0..frames {
                let s = (0.4 * (2.0 * PI * 220.0 * (i as f64) / sr).sin()) as f32;
                let (l, r) = tuner.process_stereo_sample(s, s);
                assert!(l.is_finite() && r.is_finite());
            }
            tuner.reset(sr);
            let (l, r) = tuner.process_stereo_sample(0.1, -0.1);
            assert!(l.is_finite() && r.is_finite());
        });
        let after = violation_count();
        assert_eq!(
            after - before,
            0,
            "RealtimeVocalTune allocated on the audio thread at sr={sr}"
        );
    }
}
