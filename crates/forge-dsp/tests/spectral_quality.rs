//! Quality and perceptual invariants for Piece #1 Spectral DSP
//! (`spectral_defizz` and `spectral_unmask`).

use std::f64::consts::PI;

use forge_core::AudioBuffer;
use forge_dsp::{DeFizzConfig, UnmaskConfig, apply_spectral_defizz, apply_spectral_unmask};

fn goertzel_rms(samples: &[f64], freq_hz: f64, sr: u32) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let omega = 2.0 * PI * freq_hz / f64::from(sr);
    let mut re = 0.0_f64;
    let mut im = 0.0_f64;
    for (i, &x) in samples.iter().enumerate() {
        let phase = omega * (i as f64);
        re += x * phase.cos();
        im -= x * phase.sin();
    }
    let n = samples.len() as f64;
    (2.0 * (re * re + im * im).sqrt()) / (n * 2.0_f64.sqrt())
}

fn pseudo_white(len: usize, seed: u64, amp: f64) -> Vec<f64> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            // xorshift64*
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let val = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
            let norm = ((val >> 11) as f64) / ((1_u64 << 53) as f64) * 2.0 - 1.0;
            norm * amp
        })
        .collect()
}

#[test]
fn defizz_preserves_stereo_image_on_asymmetric_resonance() {
    let sr = 48_000_u32;
    let frames = 48_000_usize;
    let noise_l = pseudo_white(frames, 0x1234_5678, 0.04);
    let noise_r = pseudo_white(frames, 0x9ABC_DEF0, 0.04);

    // Panned harsh 3.5 kHz resonance: 3:1 amplitude ratio between L and R
    let whistle_hz = 3500.0;
    let mut left = Vec::with_capacity(frames);
    let mut right = Vec::with_capacity(frames);
    for i in 0..frames {
        let t = (i as f64) / f64::from(sr);
        let s = (2.0 * PI * whistle_hz * t).sin();
        left.push(noise_l[i] + 0.45 * s);
        right.push(noise_r[i] + 0.15 * s);
    }

    let buf = AudioBuffer::from_channels(vec![left, right], sr).unwrap();
    let out = apply_spectral_defizz(&buf, &DeFizzConfig::default()).unwrap();

    // Measure steady-state middle half
    let slice = frames / 4..3 * frames / 4;
    let in_l = goertzel_rms(&buf.channels[0][slice.clone()], whistle_hz, sr);
    let in_r = goertzel_rms(&buf.channels[1][slice.clone()], whistle_hz, sr);
    let out_l = goertzel_rms(&out.channels[0][slice.clone()], whistle_hz, sr);
    let out_r = goertzel_rms(&out.channels[1][slice], whistle_hz, sr);

    // Must actually attenuate the harsh whistle
    assert!(
        out_l < in_l * 0.7,
        "expected whistle suppression: in_l={in_l:.4}, out_l={out_l:.4}"
    );

    // Stereo L/R ratio at the resonance frequency must stay preserved (linked gain)
    let in_ratio = in_l / in_r;
    let out_ratio = out_l / out_r;
    assert!(
        (out_ratio - in_ratio).abs() / in_ratio < 0.05,
        "stereo image shifted: in_ratio={in_ratio:.3}, out_ratio={out_ratio:.3}"
    );
}

#[test]
fn defizz_sharpness_below_8_changes_neighborhood_width() {
    let sr = 48_000_u32;
    let frames = 24_000_usize;
    let base_noise = pseudo_white(frames, 0xCAFE_BABE, 0.02);

    // A cluster of tones around 3.5 kHz separated by ~90 Hz: narrow sharpness sees the
    // neighbor tones as local floor; wider sharpness averages more of the quiet noise bed.
    let mut ch = Vec::with_capacity(frames);
    for (i, &n) in base_noise.iter().enumerate() {
        let t = (i as f64) / f64::from(sr);
        let cluster = 0.25 * (2.0 * PI * 3500.0 * t).sin()
            + 0.20 * (2.0 * PI * 3410.0 * t).sin()
            + 0.20 * (2.0 * PI * 3590.0 * t).sin();
        ch.push(n + cluster);
    }

    let buf = AudioBuffer::from_channels(vec![ch], sr).unwrap();
    let cfg_narrow = DeFizzConfig {
        sharpness_bins: 2,
        ..DeFizzConfig::default()
    };
    let cfg_default = DeFizzConfig {
        sharpness_bins: 6,
        ..DeFizzConfig::default()
    };

    let out_narrow = apply_spectral_defizz(&buf, &cfg_narrow).unwrap();
    let out_default = apply_spectral_defizz(&buf, &cfg_default).unwrap();

    let slice = frames / 4..3 * frames / 4;
    let rms_narrow = goertzel_rms(&out_narrow.channels[0][slice.clone()], 3500.0, sr);
    let rms_default = goertzel_rms(&out_default.channels[0][slice], 3500.0, sr);

    assert!(
        (rms_narrow - rms_default).abs() > 1e-3,
        "sharpness_bins=2 and sharpness_bins=6 should differ (narrow={rms_narrow:.5}, default={rms_default:.5})"
    );
}

#[test]
fn unmask_has_smooth_skirt_at_band_edge() {
    let sr = 48_000_u32;
    let frames = 24_000_usize;
    let cfg = UnmaskConfig {
        low_cut_hz: 1000.0,
        high_cut_hz: 4000.0,
        max_reduction_db: 12.0,
        sensitivity: 1.5,
        ..UnmaskConfig::default()
    };

    let measure_gain = |freq_hz: f64| -> f64 {
        let ch: Vec<f64> = (0..frames)
            .map(|i| 0.5 * (2.0 * PI * freq_hz * (i as f64) / f64::from(sr)).sin())
            .collect();
        let buf = AudioBuffer::from_channels(vec![ch], sr).unwrap();
        let out = apply_spectral_unmask(&buf, &buf, &cfg).unwrap();
        let slice = frames / 4..3 * frames / 4;
        let in_rms = goertzel_rms(&buf.channels[0][slice.clone()], freq_hz, sr);
        let out_rms = goertzel_rms(&out.channels[0][slice], freq_hz, sr);
        out_rms / in_rms
    };

    let g_in_band = measure_gain(2000.0);
    // ~1/6 octave below 1000 Hz: right in the middle of the 1/3-octave raised-cosine skirt
    let g_skirt = measure_gain(890.0);
    let g_out_of_band = measure_gain(400.0);

    assert!(
        g_in_band < g_skirt && g_skirt < g_out_of_band,
        "expected smooth monotonic skirt: in_band={g_in_band:.4}, skirt={g_skirt:.4}, out_of_band={g_out_of_band:.4}"
    );
    assert!(
        (g_out_of_band - 1.0).abs() < 0.02,
        "out-of-band tone should be untouched: {g_out_of_band:.4}"
    );
}

#[test]
fn unmask_envelope_timing_is_consistent_across_sample_rates() {
    let cfg = UnmaskConfig {
        low_cut_hz: 200.0,
        high_cut_hz: 6000.0,
        max_reduction_db: 12.0,
        sensitivity: 1.5,
        ..UnmaskConfig::default()
    };

    let measure_burst_gain = |sr: u32| -> f64 {
        let frames = (sr as usize) / 2; // 0.5 s
        let onset = (sr as usize) / 10; // 0.1 s
        let target_ch: Vec<f64> = (0..frames)
            .map(|i| 0.4 * (2.0 * PI * 1000.0 * (i as f64) / f64::from(sr)).sin())
            .collect();
        let sc_ch: Vec<f64> = (0..frames)
            .map(|i| {
                if i >= onset {
                    0.6 * (2.0 * PI * 1000.0 * (i as f64) / f64::from(sr)).sin()
                } else {
                    0.0
                }
            })
            .collect();
        let target = AudioBuffer::from_channels(vec![target_ch], sr).unwrap();
        let sc = AudioBuffer::from_channels(vec![sc_ch], sr).unwrap();
        let out = apply_spectral_unmask(&target, &sc, &cfg).unwrap();

        // Measure gain in physical window 25ms..55ms after onset
        let w_start = onset + (sr as usize) * 25 / 1000;
        let w_end = onset + (sr as usize) * 55 / 1000;
        let in_rms = goertzel_rms(&target.channels[0][w_start..w_end], 1000.0, sr);
        let out_rms = goertzel_rms(&out.channels[0][w_start..w_end], 1000.0, sr);
        out_rms / in_rms
    };

    let g_48k = measure_burst_gain(48_000);
    let g_96k = measure_burst_gain(96_000);

    assert!(
        (g_48k - g_96k).abs() < 0.08,
        "attack envelope diverged across sample rates: 48k={g_48k:.4}, 96k={g_96k:.4}"
    );
}

#[test]
fn defizz_handles_non_finite_band_weights() {
    let sr = 48_000_u32;
    let frames = 4096_usize;
    let ch = pseudo_white(frames, 0xDEAD_BEEF, 0.2);
    let buf = AudioBuffer::from_channels(vec![ch], sr).unwrap();

    for &(fw, sw) in &[
        (f64::NAN, 1.0),
        (1.0, f64::NAN),
        (f64::INFINITY, f64::NEG_INFINITY),
    ] {
        let cfg = DeFizzConfig {
            fizz_band_weight: fw,
            sibilance_band_weight: sw,
            ..DeFizzConfig::default()
        };
        let out = apply_spectral_defizz(&buf, &cfg).unwrap();
        assert!(
            out.channels[0].iter().all(|s| s.is_finite()),
            "non-finite output for weights ({fw}, {sw})"
        );
    }
}

#[test]
fn defizz_tilt_shifts_suppression_toward_highs() {
    let sr = 48_000_u32;
    let frames = 24_000_usize;
    let mut ch = pseudo_white(frames, 0xCAFE_1234, 0.08);
    for (i, s) in ch.iter_mut().enumerate() {
        let t = (i as f64) / f64::from(sr);
        *s += 0.35 * (2.0 * PI * 8500.0 * t).sin();
    }
    let buf = AudioBuffer::from_channels(vec![ch], sr).unwrap();

    let dark = apply_spectral_defizz(
        &buf,
        &DeFizzConfig {
            tilt_db_per_oct: -6.0,
            ..DeFizzConfig::default()
        },
    )
    .unwrap();
    let bright = apply_spectral_defizz(
        &buf,
        &DeFizzConfig {
            tilt_db_per_oct: 6.0,
            ..DeFizzConfig::default()
        },
    )
    .unwrap();

    let slice = frames / 4..3 * frames / 4;
    let rms_dark = goertzel_rms(&dark.channels[0][slice.clone()], 8500.0, sr);
    let rms_bright = goertzel_rms(&bright.channels[0][slice], 8500.0, sr);
    assert!(
        rms_bright < rms_dark * 0.90,
        "positive tilt (+6 dB/oct) should suppress 8.5 kHz more strongly than negative tilt (-6 dB/oct): bright={rms_bright:.5}, dark={rms_dark:.5}"
    );
}

#[test]
fn defizz_mid_side_balance_isolates_mid_and_preserves_side() {
    let sr = 48_000_u32;
    let frames = 24_000_usize;
    let noise_l = pseudo_white(frames, 0x1111_2222, 0.05);
    let noise_r = pseudo_white(frames, 0x3333_4444, 0.05);
    let mut left = vec![0.0; frames];
    let mut right = vec![0.0; frames];
    for i in 0..frames {
        let t = (i as f64) / f64::from(sr);
        // 3.5 kHz resonance purely in Mid (L == R), 4.5 kHz resonance purely in Side (L == -R)
        let mid_tone = 0.30 * (2.0 * PI * 3500.0 * t).sin();
        let side_tone = 0.30 * (2.0 * PI * 4500.0 * t).sin();
        left[i] = noise_l[i] + mid_tone + side_tone;
        right[i] = noise_r[i] + mid_tone - side_tone;
    }
    let buf = AudioBuffer::from_channels(vec![left, right], sr).unwrap();

    let mid_only = apply_spectral_defizz(
        &buf,
        &DeFizzConfig {
            mid_side_balance: -1.0,
            max_depth_db: 12.0,
            ..DeFizzConfig::default()
        },
    )
    .unwrap();

    let slice = frames / 4..3 * frames / 4;
    let mid_in: Vec<f64> = slice
        .clone()
        .map(|i| 0.5 * (buf.channels[0][i] + buf.channels[1][i]))
        .collect();
    let mid_out: Vec<f64> = slice
        .clone()
        .map(|i| 0.5 * (mid_only.channels[0][i] + mid_only.channels[1][i]))
        .collect();
    let side_in: Vec<f64> = slice
        .clone()
        .map(|i| 0.5 * (buf.channels[0][i] - buf.channels[1][i]))
        .collect();
    let side_out: Vec<f64> = slice
        .map(|i| 0.5 * (mid_only.channels[0][i] - mid_only.channels[1][i]))
        .collect();

    let mid_ratio = goertzel_rms(&mid_out, 3500.0, sr) / goertzel_rms(&mid_in, 3500.0, sr);
    let side_ratio = goertzel_rms(&side_out, 4500.0, sr) / goertzel_rms(&side_in, 4500.0, sr);

    assert!(
        mid_ratio < 0.70,
        "Mid-only mode should strongly attenuate Mid 3.5 kHz resonance: ratio={mid_ratio:.4}"
    );
    assert!(
        (side_ratio - 1.0).abs() < 0.05,
        "Mid-only mode should preserve Side 4.5 kHz resonance: ratio={side_ratio:.4}"
    );
}

#[test]
fn defizz_unlinked_stereo_preserves_clean_channel() {
    let sr = 48_000_u32;
    let frames = 24_000_usize;
    let mut left = pseudo_white(frames, 0xAAAA_BBBB, 0.05);
    let right = pseudo_white(frames, 0xCCCC_DDDD, 0.05);
    for (i, s) in left.iter_mut().enumerate() {
        let t = (i as f64) / f64::from(sr);
        *s += 0.35 * (2.0 * PI * 3500.0 * t).sin();
    }
    let buf = AudioBuffer::from_channels(vec![left, right], sr).unwrap();

    let unlinked = apply_spectral_defizz(
        &buf,
        &DeFizzConfig {
            stereo_link: 0.0,
            max_depth_db: 12.0,
            ..DeFizzConfig::default()
        },
    )
    .unwrap();
    let linked = apply_spectral_defizz(
        &buf,
        &DeFizzConfig {
            stereo_link: 1.0,
            max_depth_db: 12.0,
            ..DeFizzConfig::default()
        },
    )
    .unwrap();

    let slice = frames / 4..3 * frames / 4;
    let r_in = goertzel_rms(&buf.channels[1][slice.clone()], 3500.0, sr);
    let r_unlinked = goertzel_rms(&unlinked.channels[1][slice.clone()], 3500.0, sr) / r_in;
    let r_linked = goertzel_rms(&linked.channels[1][slice], 3500.0, sr) / r_in;

    assert!(
        r_unlinked > r_linked + 0.10,
        "unlinked stereo should avoid ducking the clean right channel compared to linked: unlinked={r_unlinked:.4}, linked={r_linked:.4}"
    );
}

#[test]
fn unmask_delta_listen_isolates_carved_pocket_and_ms_preserves_sides() {
    let sr = 48_000_u32;
    let frames = 24_000_usize;
    let mut left = vec![0.0; frames];
    let mut right = vec![0.0; frames];
    let mut sc = vec![0.0; frames];
    for i in 0..frames {
        let t = (i as f64) / f64::from(sr);
        let mid_1k = 0.4 * (2.0 * PI * 1000.0 * t).sin();
        let side_1k = 0.4 * (2.0 * PI * 1000.0 * t).sin();
        left[i] = mid_1k + side_1k;
        right[i] = mid_1k - side_1k;
        sc[i] = 0.7 * (2.0 * PI * 1000.0 * t).sin();
    }
    let target = AudioBuffer::from_channels(vec![left, right], sr).unwrap();
    let sidechain = AudioBuffer::from_channels(vec![sc], sr).unwrap();

    let cfg_wet = UnmaskConfig {
        low_cut_hz: 200.0,
        high_cut_hz: 4000.0,
        max_reduction_db: 12.0,
        sensitivity: 1.5,
        delta_listen: false,
        mid_side_balance: -1.0, // Carve Mid only, preserve stereo Sides!
    };
    let cfg_delta = UnmaskConfig {
        delta_listen: true,
        ..cfg_wet
    };

    let wet = apply_spectral_unmask(&target, &sidechain, &cfg_wet).unwrap();
    let delta = apply_spectral_unmask(&target, &sidechain, &cfg_delta).unwrap();

    let slice = frames / 4..3 * frames / 4;
    let side_in: Vec<f64> = slice
        .clone()
        .map(|i| 0.5 * (target.channels[0][i] - target.channels[1][i]))
        .collect();
    let side_wet: Vec<f64> = slice
        .clone()
        .map(|i| 0.5 * (wet.channels[0][i] - wet.channels[1][i]))
        .collect();
    let mid_in: Vec<f64> = slice
        .clone()
        .map(|i| 0.5 * (target.channels[0][i] + target.channels[1][i]))
        .collect();
    let mid_wet: Vec<f64> = slice
        .clone()
        .map(|i| 0.5 * (wet.channels[0][i] + wet.channels[1][i]))
        .collect();
    let mid_delta: Vec<f64> = slice
        .map(|i| 0.5 * (delta.channels[0][i] + delta.channels[1][i]))
        .collect();

    let side_ratio = goertzel_rms(&side_wet, 1000.0, sr) / goertzel_rms(&side_in, 1000.0, sr);
    let mid_wet_ratio = goertzel_rms(&mid_wet, 1000.0, sr) / goertzel_rms(&mid_in, 1000.0, sr);
    let mid_delta_ratio = goertzel_rms(&mid_delta, 1000.0, sr) / goertzel_rms(&mid_in, 1000.0, sr);

    assert!(
        (side_ratio - 1.0).abs() < 0.03,
        "Mid-only unmask must preserve stereo Side width: side_ratio={side_ratio:.4}"
    );
    assert!(
        mid_wet_ratio < 0.55,
        "Mid-only unmask must carve Mid 1 kHz pocket: mid_wet_ratio={mid_wet_ratio:.4}"
    );
    assert!(
        (mid_wet_ratio + mid_delta_ratio - 1.0).abs() < 0.05,
        "Delta + Wet must reconstruct dry magnitude: wet={mid_wet_ratio:.4}, delta={mid_delta_ratio:.4}"
    );
}
