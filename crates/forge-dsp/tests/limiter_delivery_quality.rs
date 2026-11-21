//! Quality and mastering invariants for Piece #2 (Loudness, Limiter & Delivery).

use std::f64::consts::PI;

use forge_core::AudioBuffer;
use forge_dsp::{DeliveryOptions, DeliveryTarget, audit_audio, deliver, normalize_and_limit};

const SR: u32 = 48_000;

/// 6-second punchy mix with high crest factor: sustained musical bed plus strong periodic
/// transient hits every 250 ms so the limiter must actively work when pushing to loud targets.
fn punchy_high_crest_mix(sr: u32) -> AudioBuffer {
    let frames = (sr as usize) * 6;
    let beat_samples = (sr as usize) / 4; // 4 hits/sec
    let hit_len = (sr as usize) * 18 / 1000; // 18 ms transient burst
    let mut ch = Vec::with_capacity(frames);
    for i in 0..frames {
        let t = (i as f64) / f64::from(sr);
        let bed = 0.08 * (2.0 * PI * 110.0 * t).sin() + 0.05 * (2.0 * PI * 440.0 * t).sin();
        let pos_in_beat = i % beat_samples;
        let hit = if pos_in_beat < hit_len {
            let env = (-(pos_in_beat as f64) / (f64::from(sr) * 0.006)).exp();
            0.90 * env * (2.0 * PI * 180.0 * t).sin()
        } else {
            0.0
        };
        ch.push(bed + hit);
    }
    AudioBuffer::from_channels(vec![ch.clone(), ch], sr).unwrap()
}

#[test]
fn normalize_and_limit_converges_on_punchy_high_crest_mix() {
    let mix = punchy_high_crest_mix(SR);
    let target_lufs = -11.0;
    let ceiling_dbtp = -1.0;

    let mastered = normalize_and_limit(&mix, target_lufs, ceiling_dbtp).unwrap();
    let report = audit_audio(&mastered);

    assert!(
        (report.integrated_lufs - target_lufs).abs() <= 0.35,
        "expected closed-loop convergence near {target_lufs} LUFS, got {:.2} LUFS",
        report.integrated_lufs
    );
    assert!(
        report.true_peak_dbtp <= ceiling_dbtp,
        "true peak {:.2} dBTP exceeded ceiling {ceiling_dbtp} dBTP",
        report.true_peak_dbtp
    );
    assert_eq!(report.clipped_samples, 0);
}

#[test]
fn dual_stage_release_recovers_faster_on_transient_than_sustained_overload() {
    let frames = (SR as usize) * 2;
    let bed_amp = 0.04;
    let over_amp = 0.70; // ~+9 dB over -12.0 dBTP ceiling (0.251 lin)
    let ceiling_dbtp = -12.0;
    let end_sample = (SR as usize) / 2; // overload ends at t = 0.5 s

    let make_signal = |overload_ms: usize| -> AudioBuffer {
        let overload_samples = (SR as usize) * overload_ms / 1000;
        let start_sample = end_sample - overload_samples;
        let ch: Vec<f64> = (0..frames)
            .map(|i| {
                let t = (i as f64) / f64::from(SR);
                let tone = (2.0 * PI * 1000.0 * t).sin();
                if i >= start_sample && i < end_sample {
                    over_amp * tone
                } else {
                    bed_amp * tone
                }
            })
            .collect();
        AudioBuffer::from_channels(vec![ch.clone(), ch], SR).unwrap()
    };

    let transient_buf = make_signal(3);
    let sustained_buf = make_signal(250);

    let lufs_t = audit_audio(&transient_buf)
        .integrated_lufs
        .clamp(-36.0, -6.0);
    let lufs_s = audit_audio(&sustained_buf)
        .integrated_lufs
        .clamp(-36.0, -6.0);

    let out_t = normalize_and_limit(&transient_buf, lufs_t, ceiling_dbtp).unwrap();
    let out_s = normalize_and_limit(&sustained_buf, lufs_s, ceiling_dbtp).unwrap();

    let rms = |s: &[f64]| -> f64 { (s.iter().map(|x| x * x).sum::<f64>() / s.len() as f64).sqrt() };

    // Recovery window: 70ms..110ms after overload ends; far bed: 1.5s..1.8s (fully recovered).
    let rec_start = end_sample + (SR as usize) * 70 / 1000;
    let rec_end = end_sample + (SR as usize) * 110 / 1000;
    let far_start = (SR as usize) * 15 / 10;
    let far_end = (SR as usize) * 18 / 10;

    let recovery_ratio_transient =
        rms(&out_t.channels[0][rec_start..rec_end]) / rms(&out_t.channels[0][far_start..far_end]);
    let recovery_ratio_sustained =
        rms(&out_s.channels[0][rec_start..rec_end]) / rms(&out_s.channels[0][far_start..far_end]);

    assert!(
        recovery_ratio_transient > recovery_ratio_sustained + 0.05,
        "expected isolated transient to recover faster than sustained overload: \
         transient={recovery_ratio_transient:.3}, sustained={recovery_ratio_sustained:.3}"
    );
}

#[test]
fn audit_reports_dc_offset_phase_inversion_and_low_crest_factor() {
    let frames = SR as usize;

    // 1. DC offset
    let dc_ch: Vec<f64> = (0..frames)
        .map(|i| 0.15 + 0.1 * (2.0 * PI * 440.0 * (i as f64) / f64::from(SR)).sin())
        .collect();
    let dc_report = audit_audio(&AudioBuffer::from_channels(vec![dc_ch], SR).unwrap());
    assert!(
        dc_report
            .recommendations
            .iter()
            .any(|r| r.contains("DC offset")),
        "expected DC offset recommendation, got {:?}",
        dc_report.recommendations
    );

    // 2. Out-of-phase stereo
    let l: Vec<f64> = (0..frames)
        .map(|i| 0.25 * (2.0 * PI * 440.0 * (i as f64) / f64::from(SR)).sin())
        .collect();
    let r: Vec<f64> = l.iter().map(|&x| -x).collect();
    let phase_report = audit_audio(&AudioBuffer::from_channels(vec![l, r], SR).unwrap());
    assert!(
        phase_report
            .recommendations
            .iter()
            .any(|r| r.contains("Negative broadband stereo correlation")),
        "expected negative correlation recommendation, got {:?}",
        phase_report.recommendations
    );

    // 3. Heavily squashed square-like wave (crest factor ~ 0 dB)
    let sq: Vec<f64> = (0..frames)
        .map(|i| {
            if (2.0 * PI * 220.0 * (i as f64) / f64::from(SR)).sin() >= 0.0 {
                0.5
            } else {
                -0.5
            }
        })
        .collect();
    let squash_report = audit_audio(&AudioBuffer::from_channels(vec![sq], SR).unwrap());
    assert!(
        squash_report
            .recommendations
            .iter()
            .any(|r| r.contains("Low crest factor")),
        "expected low crest factor recommendation, got {:?}",
        squash_report.recommendations
    );
}

#[test]
fn deliver_hits_spotify_target_on_punchy_mix() {
    let mix = punchy_high_crest_mix(SR);
    let (_, report) = deliver(&mix, &DeliveryOptions::for_target(DeliveryTarget::Spotify)).unwrap();
    assert!(
        report.ready,
        "expected Spotify delivery ready: {:#?}",
        report.checks
    );
    assert!(
        (report.after.integrated_lufs - (-14.0)).abs() <= 0.4,
        "expected tight loudness hit near -14.0 LUFS, got {:.2}",
        report.after.integrated_lufs
    );
}

#[test]
fn limiter_styles_all_obey_ceiling_and_differ_in_recovery() {
    use forge_dsp::{LimiterStyle, MasteringLimiterConfig, normalize_and_limit_with_config};

    let mix = punchy_high_crest_mix(SR);
    let mut rms_by_style = Vec::new();
    for style in [
        LimiterStyle::Transparent,
        LimiterStyle::Punchy,
        LimiterStyle::Dynamic,
        LimiterStyle::Aggressive,
    ] {
        let cfg = MasteringLimiterConfig {
            target_lufs: -10.0,
            ceiling_dbtp: -1.0,
            style,
            ..MasteringLimiterConfig::default()
        };
        let out = normalize_and_limit_with_config(&mix, &cfg).unwrap();
        let report = audit_audio(&out);
        assert!(
            report.true_peak_dbtp <= -1.0 + 1e-6,
            "style {style:?} exceeded -1.0 dBTP ceiling: {:.4}",
            report.true_peak_dbtp
        );
        let rms =
            (out.channels[0].iter().map(|s| s * s).sum::<f64>() / out.num_frames() as f64).sqrt();
        rms_by_style.push((style, rms));
    }
    let dynamic_rms = rms_by_style[2].1;
    let aggressive_rms = rms_by_style[3].1;
    assert!(
        (aggressive_rms - dynamic_rms).abs() > 1e-4,
        "Aggressive and Dynamic limiter styles should produce distinct envelope dynamics"
    );

    // Dithered 16-bit mastering must still strictly obey the requested true-peak ceiling.
    let dithered = normalize_and_limit_with_config(
        &mix,
        &MasteringLimiterConfig {
            target_lufs: -10.0,
            ceiling_dbtp: -1.0,
            dither: forge_dsp::DitherMode::NoiseShapedHighPass,
            dither_bits: 16,
            ..MasteringLimiterConfig::default()
        },
    )
    .unwrap();
    let dithered_report = audit_audio(&dithered);
    assert!(
        dithered_report.true_peak_dbtp <= -1.0 + 1e-6,
        "16-bit noise-shaped dithered master exceeded -1.0 dBTP ceiling: {:.4}",
        dithered_report.true_peak_dbtp
    );
}

#[test]
fn unlinked_channel_limiter_preserves_quiet_channel_during_asymmetric_transient() {
    use forge_dsp::{MasteringLimiterConfig, normalize_and_limit_with_config};

    let frames = (SR as usize) * 2;
    let mut left = vec![0.0; frames];
    let mut right = vec![0.0; frames];
    for i in 0..frames {
        let t = (i as f64) / f64::from(SR);
        let base = 0.25 * (2.0 * PI * 440.0 * t).sin();
        left[i] = base;
        right[i] = base;
        // Inject a strong transient spike on Left ONLY at 0.5 s
        if (24_000..24_120).contains(&i) {
            left[i] += 1.8 * (2.0 * PI * 1000.0 * t).sin();
        }
    }
    let buf = AudioBuffer::from_channels(vec![left, right], SR).unwrap();

    let linked = normalize_and_limit_with_config(
        &buf,
        &MasteringLimiterConfig {
            target_lufs: -14.0,
            ceiling_dbtp: -1.0,
            channel_link: 1.0,
            ..MasteringLimiterConfig::default()
        },
    )
    .unwrap();
    let unlinked = normalize_and_limit_with_config(
        &buf,
        &MasteringLimiterConfig {
            target_lufs: -14.0,
            ceiling_dbtp: -1.0,
            channel_link: 0.0,
            ..MasteringLimiterConfig::default()
        },
    )
    .unwrap();

    let slice = 24_000..24_600;
    let r_linked_rms = (linked.channels[1][slice.clone()]
        .iter()
        .map(|s| s * s)
        .sum::<f64>()
        / slice.len() as f64)
        .sqrt();
    let r_unlinked_rms = (unlinked.channels[1][slice.clone()]
        .iter()
        .map(|s| s * s)
        .sum::<f64>()
        / slice.len() as f64)
        .sqrt();

    assert!(
        r_unlinked_rms > r_linked_rms * 1.15,
        "unlinked limiter should avoid ducking the unaffected right channel: unlinked={r_unlinked_rms:.5}, linked={r_linked_rms:.5}"
    );
}

#[test]
fn noise_shaped_dither_lowers_midband_quantization_noise_floor_at_16bit() {
    use forge_dsp::{DitherMode, apply_dither};
    use rustfft::{FftPlanner, num_complex::Complex};

    let n = 16_384_usize;
    // Ultra-quiet -84 dBFS 1 kHz sine (around 2 LSB at 16-bit)
    let amp = 10.0_f64.powf(-84.0 / 20.0);
    let ch: Vec<f64> = (0..n)
        .map(|i| amp * (2.0 * PI * 1000.0 * (i as f64) / f64::from(SR)).sin())
        .collect();
    let buf = AudioBuffer::from_channels(vec![ch.clone()], SR).unwrap();

    let tpdf = apply_dither(&buf, 16, DitherMode::Tpdf).unwrap();
    let shaped = apply_dither(&buf, 16, DitherMode::NoiseShapedHighPass).unwrap();

    let midband_noise_power = |processed: &[f64]| -> f64 {
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(n);
        let mut spec: Vec<Complex<f64>> = processed
            .iter()
            .zip(&ch)
            .map(|(&p, &orig)| Complex::new(p - orig, 0.0))
            .collect();
        fft.process(&mut spec);
        let bin_hz = f64::from(SR) / (n as f64);
        let b_lo = (500.0 / bin_hz).round() as usize;
        let b_hi = (5000.0 / bin_hz).round() as usize;
        spec[b_lo..=b_hi].iter().map(|c| c.norm_sqr()).sum::<f64>()
    };

    let p_tpdf = midband_noise_power(&tpdf.channels[0]);
    let p_shaped = midband_noise_power(&shaped.channels[0]);
    let improvement_db = 10.0 * (p_tpdf / p_shaped.max(1e-30)).log10();

    assert!(
        improvement_db > 6.0,
        "2nd-order psychoacoustic noise shaping should reduce 0.5–5 kHz noise floor by > 6 dB over flat TPDF, got {improvement_db:.2} dB"
    );
    assert!(shaped.channels[0].iter().all(|s| s.is_finite()));
}
