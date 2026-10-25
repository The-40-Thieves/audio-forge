//! Offline DSP robustness (THE-1598).
//!
//! 1. `apply_spectral_unmask` stays finite when its config or its input holds `NaN` / `±Inf`.
//! 2. `apply_spectral_unmask` rejects a ragged target instead of panicking.
//! 3. `run_full_restoration_chain` honours `remove_dc` independently of `declip`.

use forge_core::{AudioBuffer, ForgeError};
use forge_dsp::{
    RestorationConfig, UnmaskConfig, apply_spectral_unmask, run_full_restoration_chain,
};
use proptest::prelude::*;

const SR: u32 = 48_000;

/// Any `f64`, with the non-finite values drawn often enough to be hit in every run.
fn wild_f64() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(f64::NAN),
        Just(f64::INFINITY),
        Just(f64::NEG_INFINITY),
        -1.0e6_f64..1.0e6_f64,
        prop::num::f64::ANY,
    ]
}

/// Stereo test tone: target and sidechain share the band so the unmasker actually ducks.
fn stereo_tone(frames: usize, amp: f64) -> Vec<Vec<f64>> {
    let ch: Vec<f64> = (0..frames)
        .map(|i| amp * (2.0 * std::f64::consts::PI * 880.0 * (i as f64) / f64::from(SR)).sin())
        .collect();
    vec![ch.clone(), ch.iter().map(|x| -0.8 * x).collect()]
}

fn assert_all_finite(buf: &AudioBuffer) -> Result<(), TestCaseError> {
    for (c, ch) in buf.channels.iter().enumerate() {
        for (i, &s) in ch.iter().enumerate() {
            prop_assert!(s.is_finite(), "channel {} sample {} is {}", c, i, s);
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn prop_unmask_output_is_finite_for_non_finite_config(
        frames in 256_usize..1536,
        low_cut_hz in wild_f64(),
        high_cut_hz in wild_f64(),
        max_reduction_db in wild_f64(),
        sensitivity in wild_f64(),
    ) {
        let buf = AudioBuffer::from_channels(stereo_tone(frames, 0.5), SR).unwrap();
        let cfg = UnmaskConfig { low_cut_hz, high_cut_hz, max_reduction_db, sensitivity };
        let out = apply_spectral_unmask(&buf, &buf, &cfg).expect("unmask succeeded");
        prop_assert_eq!(out.num_frames(), frames);
        assert_all_finite(&out)?;
    }

    #[test]
    fn prop_unmask_output_is_finite_for_non_finite_samples(
        frames in 256_usize..1536,
        poison in prop::collection::vec((0_usize..1536, 0_usize..2, wild_f64()), 1..6),
    ) {
        let mut target = stereo_tone(frames, 0.5);
        let mut sidechain = stereo_tone(frames, 0.7);
        for &(pos, ch, val) in &poison {
            target[ch][pos % frames] = val;
            sidechain[1 - ch][(pos * 7) % frames] = val;
        }
        let target = AudioBuffer::from_channels(target, SR).unwrap();
        let sidechain = AudioBuffer::from_channels(sidechain, SR).unwrap();
        let cfg = UnmaskConfig::vocal_over_instruments();
        let out = apply_spectral_unmask(&target, &sidechain, &cfg).expect("unmask succeeded");
        prop_assert_eq!(out.num_frames(), frames);
        assert_all_finite(&out)?;
    }
}

#[test]
fn unmask_rejects_a_ragged_target_instead_of_panicking() {
    // Public fields let a caller bypass `AudioBuffer::from_channels` validation.
    let target = AudioBuffer {
        channels: vec![vec![0.1; 3000], vec![0.1; 2000]],
        sample_rate: SR,
    };
    let sidechain = AudioBuffer::from_channels(vec![vec![0.1; 3000]], SR).unwrap();
    let result = apply_spectral_unmask(&target, &sidechain, &UnmaskConfig::kick_over_bass());
    match result {
        Err(ForgeError::InvalidParameter(msg)) => assert!(msg.contains("Ragged"), "{msg}"),
        other => panic!("expected a ragged-buffer error, got {other:?}"),
    }
}

/// Restoration config with only the DC and de-clip stages enabled.
fn dc_declip_only(remove_dc: bool, declip: bool) -> RestorationConfig {
    RestorationConfig {
        remove_dc,
        declip,
        hum_fundamental_hz: None,
        mono_below_hz: None,
        stereo_width_highs: 1.0,
        carves_box_mud_320hz_db: 0.0,
        analog_tape_drive: 0.0,
    }
}

fn channel_mean(ch: &[f64]) -> f64 {
    ch.iter().sum::<f64>() / (ch.len() as f64)
}

#[test]
fn remove_dc_false_with_declip_true_keeps_the_dc_offset() {
    let dc = 0.25;
    let ch: Vec<f64> = (0..4800)
        .map(|i| dc + 0.2 * (2.0 * std::f64::consts::PI * 440.0 * (i as f64) / f64::from(SR)).sin())
        .collect();
    let buf = AudioBuffer::from_channels(vec![ch.clone(), ch], SR).unwrap();

    let kept = run_full_restoration_chain(&buf, &dc_declip_only(false, true)).unwrap();
    for ch in &kept.channels {
        let mean = channel_mean(ch);
        assert!(
            (mean - dc).abs() < 1e-3,
            "remove_dc=false must keep DC {dc}, got mean {mean}"
        );
    }

    let removed = run_full_restoration_chain(&buf, &dc_declip_only(true, true)).unwrap();
    for ch in &removed.channels {
        let mean = channel_mean(ch);
        assert!(
            mean.abs() < 1e-9,
            "remove_dc=true must remove DC, got mean {mean}"
        );
    }
}
