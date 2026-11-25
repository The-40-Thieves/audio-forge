use std::f64::consts::PI;

use forge_core::AudioBuffer;
use forge_dsp::{
    DeFizzConfig, DeliveryOptions, DeliveryTarget, RestorationConfig,
    align_vocal_double_dtw_elastic, analyze_beat_grid, apply_spectral_defizz, audit_audio, deliver,
    quantize_to_constant_bpm, remove_dc_and_declip, sum_audio_buffers, warp_buffer_wsola,
};

fn make_sine_stereo(freq_hz: f64, amp: f64, sr: u32, secs: f64) -> AudioBuffer {
    let frames = ((f64::from(sr) * secs).round() as usize).max(1);
    let ch: Vec<f64> = (0..frames)
        .map(|i| amp * (2.0 * PI * freq_hz * (i as f64) / f64::from(sr)).sin())
        .collect();
    AudioBuffer::from_channels(vec![ch.clone(), ch], sr).expect("valid sine buffer")
}

#[test]
fn test_audit_measures_side_shimmer_bandwidth_and_transient_sharpness() {
    let sr = 48_000_u32;
    let steady = make_sine_stereo(440.0, 0.25, sr, 1.5);
    let steady_audit = audit_audio(&steady);
    assert!(steady_audit.high_mid_stereo_correlation > 0.5);
    assert!(steady_audit.effective_bandwidth_hz <= 6000.0);

    // Uncorrelated stereo high-frequency shimmer + sharp Kick impulse bursts
    let frames = (sr as usize) * 2;
    let mut left = vec![0.0_f64; frames];
    let mut right = vec![0.0_f64; frames];
    for i in 0..frames {
        let t = (i as f64) / f64::from(sr);
        let base = 0.15 * (2.0 * PI * 180.0 * t).sin();
        // Opposite-phase harsh 4.8kHz/9.5kHz swirl in Left vs Right
        let fizz_l = 0.18 * (2.0 * PI * 4850.0 * t).sin() + 0.10 * (2.0 * PI * 9600.0 * t).sin();
        let fizz_r = -0.18 * (2.0 * PI * 4850.0 * t).cos() + 0.10 * (2.0 * PI * 10200.0 * t).sin();
        // Periodic 5 ms sharp kick burst every 0.5s
        let phase_in_beat = i % (sr as usize / 2);
        let transient = if phase_in_beat < 240 {
            0.65 * (-((phase_in_beat as f64) / 60.0)).exp()
        } else {
            0.0
        };
        left[i] = base + fizz_l + transient;
        right[i] = base + fizz_r + transient;
    }
    let harsh = AudioBuffer::from_channels(vec![left, right], sr).expect("valid harsh buffer");
    let harsh_audit = audit_audio(&harsh);
    assert!(
        harsh_audit.effective_bandwidth_hz > steady_audit.effective_bandwidth_hz,
        "expected wider bandwidth on 10kHz track: harsh={} steady={}",
        harsh_audit.effective_bandwidth_hz,
        steady_audit.effective_bandwidth_hz
    );
    assert!(
        harsh_audit.high_mid_stereo_correlation < 0.4,
        "expected low 2-12kHz correlation on phasey stereo shimmer: {}",
        harsh_audit.high_mid_stereo_correlation
    );
    assert!(
        harsh_audit.transient_sharpness_db > steady_audit.transient_sharpness_db,
        "expected higher transient sharpness with impulses: harsh={} steady={}",
        harsh_audit.transient_sharpness_db,
        steady_audit.transient_sharpness_db
    );
}

#[test]
fn test_adaptive_plateau_declipper_repairs_normalized_flat_tops() {
    let sr = 48_000_u32;
    let frames = 4000; // Exact 60 cycles of 720 Hz -> zero DC asymmetry after mean removal
    let ceiling = 0.891_25; // -1 dBFS normalized plateau below legacy 0.985 threshold
    let mut ch = vec![0.0_f64; frames];
    for (i, sample) in ch.iter_mut().enumerate() {
        // 720 Hz clipped arch spans ~18 samples (inside 3..=64 Hermite declip window)
        let raw = 1.25 * (2.0 * PI * 720.0 * (i as f64) / f64::from(sr)).sin();
        *sample = raw.clamp(-ceiling, ceiling);
    }
    let mut buf = AudioBuffer::from_channels(vec![ch], sr).expect("valid");
    remove_dc_and_declip(&mut buf, true);

    let peak_after = buf.channels[0]
        .iter()
        .copied()
        .fold(0.0_f64, |a, b| a.max(b.abs()));
    assert!(
        peak_after > ceiling + 0.01,
        "expected adaptive Hermite interpolation above {ceiling}, got {peak_after}"
    );
    // Ensure flat runs >= 4 samples at the ceiling were eliminated
    let flat_runs = buf.channels[0]
        .windows(4)
        .filter(|w| {
            (w[0] - w[1]).abs() < 1e-6
                && (w[1] - w[2]).abs() < 1e-6
                && (w[2] - w[3]).abs() < 1e-6
                && w[0].abs() > 0.85
        })
        .count();
    assert_eq!(flat_runs, 0, "plateau runs should be reconstructed");
}

#[test]
fn test_restoration_config_from_audit_and_sum_audio_buffers() {
    let sr = 48_000_u32;
    let s1 = make_sine_stereo(220.0, 0.25, sr, 0.5);
    let s2 = make_sine_stereo(660.0, 0.15, sr, 0.25);

    let audit = audit_audio(&s1);
    let rest_ai = RestorationConfig::from_audit(&audit, true);
    assert!(
        (rest_ai.stereo_width_highs - 1.0).abs() < 1e-9,
        "AI track mode must disable artificial high-band widening"
    );

    let summed = sum_audio_buffers(&[&s1, &s2], s1.num_frames(), sr).expect("sum succeeds");
    assert_eq!(summed.num_channels(), 2);
    assert_eq!(summed.num_frames(), s1.num_frames());
    // In first quarter-second, s1 + s2 active; after 0.25s, only s1
    assert!((summed.channels[0][100] - (s1.channels[0][100] + s2.channels[0][100])).abs() < 1e-9);
    assert!((summed.channels[0][sr as usize / 3] - s1.channels[0][sr as usize / 3]).abs() < 1e-9);

    assert!(sum_audio_buffers(&[], 100, sr).is_err());
}

#[test]
fn test_defizz_ai_presets_hpss_hard_mode_and_ultrasonic_lpf() {
    let sr = 48_000_u32;
    let frames = sr as usize;
    let mut voice = Vec::with_capacity(frames);
    let mut ultra_buzz = Vec::with_capacity(frames);
    for i in 0..frames {
        let t = (i as f64) / f64::from(sr);
        let warm = 0.30 * (2.0 * PI * 160.0 * t).sin();
        let air_buzz = 0.25 * (2.0 * PI * 18_500.0 * t).sin();
        voice.push(warm + air_buzz);
        ultra_buzz.push(air_buzz);
    }
    let buf = AudioBuffer::from_channels(vec![voice.clone(), voice], sr).expect("valid");
    let cleaned = apply_spectral_defizz(&buf, &DeFizzConfig::ai_vocal()).expect("defizz vocal");

    // 16 kHz ultrasonic LPF inside ai_vocal() should attenuate the 18.5 kHz codec buzz strongly
    let out_rms = (cleaned.channels[0].iter().map(|x| x * x).sum::<f64>() / (frames as f64)).sqrt();
    let warm_only = make_sine_stereo(160.0, 0.30, sr, 1.0);
    let warm_rms =
        (warm_only.channels[0].iter().map(|x| x * x).sum::<f64>() / (frames as f64)).sqrt();
    let raw_rms = (buf.channels[0].iter().map(|x| x * x).sum::<f64>() / (frames as f64)).sqrt();
    assert!(
        out_rms < raw_rms && (out_rms - warm_rms).abs() < 0.04,
        "expected 18.5kHz buzz filtered cleanly: raw_rms={raw_rms:.4} out_rms={out_rms:.4} warm_rms={warm_rms:.4}"
    );

    let mut report = audit_audio(&buf);
    report.ai_fizz_index = 72.0;
    report.mid_fizz_p95_db = 4.0;
    report.side_fizz_p95_db = 9.5;
    let ai_cfg = DeFizzConfig::ai_mix_from_audit(&report);
    assert!(ai_cfg.hard_mode);
    assert!(ai_cfg.mid_side_balance > 0.15);
    assert!(ai_cfg.hpss_shimmer_suppress >= 0.5);
    let _ = ultra_buzz;
}

#[test]
fn test_deliver_ai_track_mode_passes_checks_and_reduces_fizz() {
    let sr = 48_000_u32;
    let frames = sr as usize;
    let mut ch = vec![0.0_f64; frames];
    for (i, s) in ch.iter_mut().enumerate() {
        let t = (i as f64) / f64::from(sr);
        *s = 0.28 * (2.0 * PI * 220.0 * t).sin() + 0.35 * (2.0 * PI * 4250.0 * t).sin();
    }
    let buf = AudioBuffer::from_channels(vec![ch.clone(), ch], sr).expect("valid");
    let mut opts = DeliveryOptions::for_target(DeliveryTarget::Spotify);
    opts.ai_track_mode = true;
    let (_master, report) = deliver(&buf, &opts).expect("deliver ai_track_mode");
    assert!(report.ready, "checks failed: {:?}", report.checks);
    assert!(report.after.ai_fizz_index < report.before.ai_fizz_index);
}

#[test]
fn test_beat_grid_tracks_drifting_tempo_and_wsola_quantizes() {
    let sr = 48_000_u32;
    let secs = 10.0;
    let frames = (sr as usize) * 10;
    let mut ch = vec![0.0_f64; frames];
    // Place sharp 100 Hz drum transients at drifting intervals (105 BPM first half -> 130 BPM second half)
    let mut t = 0.25_f64;
    let mut pulse_times = Vec::new();
    while t < secs - 0.25 {
        pulse_times.push(t);
        let bpm = if t < 5.0 { 105.0 } else { 130.0 };
        t += 60.0 / bpm;
    }
    for &pt in &pulse_times {
        let start = (pt * f64::from(sr)).round() as usize;
        for k in 0..800 {
            if start + k < frames {
                let env = (-(k as f64) / 140.0).exp();
                ch[start + k] += 0.85 * env * (2.0 * PI * 110.0 * (k as f64) / f64::from(sr)).sin();
            }
        }
    }
    let buf = AudioBuffer::from_channels(vec![ch.clone(), ch], sr).expect("valid click track");
    let grid = analyze_beat_grid(&buf, Some(120.0));
    assert!(
        grid.beats_secs.len() >= 8,
        "expected >= 8 beats, got {}",
        grid.beats_secs.len()
    );
    assert!(
        (95.0..=145.0).contains(&grid.median_bpm),
        "unexpected median_bpm: {}",
        grid.median_bpm
    );

    // Explicit quantization onto constant 120 BPM
    let (quantized, markers) =
        quantize_to_constant_bpm(&buf, &pulse_times, 120.0).expect("quantize grid");
    assert_eq!(markers.len(), pulse_times.len());
    assert_eq!(quantized.num_channels(), 2);
    assert!(quantized.channels.iter().flatten().all(|x| x.is_finite()));

    let warped = warp_buffer_wsola(&buf, |idx| (idx as f64) * 0.9, buf.num_frames());
    assert_eq!(warped.num_frames(), buf.num_frames());
    assert!(warped.channels.iter().flatten().all(|x| x.is_finite()));
}

#[test]
fn test_realtime_opensoothe_ulpf_and_ms_telemetry() {
    use forge_dsp::RealtimeOpenSoothe;

    let sr = 48_000.0_f64;
    let mut soothe = RealtimeOpenSoothe::new(sr);
    soothe.set_ultrasonic_lpf_hz(14_000.0);
    assert!((soothe.ultrasonic_lpf_hz() - 14_000.0).abs() < 1e-6);

    // Feed out-of-phase 6 kHz fizz on Side + 20 kHz ultrasonic energy for several hops
    let mut out_rms_sq = 0.0_f64;
    let total_samples = 4096_usize;
    for i in 0..total_samples {
        let t = (i as f64) / sr;
        let ultra = 0.4 * (2.0 * PI * 20_000.0 * t).sin() as f32;
        let side_fizz = 0.25 * (2.0 * PI * 6_000.0 * t).sin() as f32;
        let (yl, yr) =
            soothe.process_stereo_sample(ultra + side_fizz, ultra - side_fizz, 3.5, 9.0, 8, false);
        assert!(yl.is_finite() && yr.is_finite());
        if i >= 3072 {
            out_rms_sq += 0.5 * (f64::from(yl) * f64::from(yl) + f64::from(yr) * f64::from(yr));
        }
    }
    let corr = soothe.high_mid_correlation();
    assert!((-1.0..=1.0).contains(&corr) && corr.is_finite());
    assert!(
        corr < 0.0,
        "expected negative high-mid correlation for out-of-phase 6 kHz fizz, got {corr}"
    );
    assert!(soothe.side_fizz_db() > soothe.mid_fizz_db());
    let out_rms = (out_rms_sq / 1024.0).sqrt();
    assert!(
        out_rms < 0.35,
        "expected attenuation of 20k+6k energy, rms={out_rms}"
    );
}

#[test]
fn test_elastic_dtw_aligns_vocal_double() {
    let sr = 48_000_u32;
    let frames = (sr as usize) * 2;
    let mut lead = vec![0.0_f64; frames];
    let mut dub = vec![0.0_f64; frames];
    for i in 12_000..42_000 {
        let t = (i as f64) / f64::from(sr);
        let note = 0.4 * (2.0 * PI * 330.0 * t).sin();
        lead[i] = note;
        dub[i + 2400] = note; // 50 ms late double
    }
    let lead_buf = AudioBuffer::from_channels(vec![lead.clone(), lead], sr).expect("lead");
    let dub_buf = AudioBuffer::from_channels(vec![dub.clone(), dub], sr).expect("dub");
    let (aligned, markers) =
        align_vocal_double_dtw_elastic(&lead_buf, &dub_buf, 120.0).expect("elastic dtw");
    assert_eq!(aligned.num_frames(), lead_buf.num_frames());
    assert!(!markers.is_empty());
    assert!(aligned.channels.iter().flatten().all(|x| x.is_finite()));
}

#[test]
fn test_defizz_ultrasonic_lpf_low_sample_rates_never_panic() {
    for (sr, lpf) in [(8_000_u32, 3_000.0), (1_500_u32, 500.0)] {
        let len = sr as usize;
        let ch: Vec<f64> = (0..len)
            .map(|i| 0.25 * (2.0 * PI * 300.0 * (i as f64) / f64::from(sr)).sin())
            .collect();
        let buf = AudioBuffer::from_channels(vec![ch.clone(), ch], sr).expect("low sr buf");
        let cfg = DeFizzConfig {
            ultrasonic_lpf_hz: Some(lpf),
            ..Default::default()
        };
        let out = apply_spectral_defizz(&buf, &cfg).expect("defizz low sr");
        assert_eq!(out.num_frames(), len);
        assert!(out.channels.iter().flatten().all(|s| s.is_finite()));
    }
}

#[test]
fn test_wsola_identity_map_preserves_non_silent_tail_without_replay() {
    let frames = 3072_usize;
    let sr = 48_000_u32;
    let mut ch = vec![0.0_f64; frames];
    for (i, s) in ch.iter_mut().enumerate() {
        *s = if i < 1536 {
            0.35 * (2.0 * PI * 220.0 * (i as f64) / f64::from(sr)).sin()
        } else {
            // Distinct unique non-silent tail ramp + high tone so any earlier replay is caught
            0.55 * (2.0 * PI * 880.0 * (i as f64) / f64::from(sr)).sin()
                * ((i - 1536) as f64 / 1536.0)
        };
    }
    let buf = AudioBuffer::from_channels(vec![ch.clone(), ch], sr).expect("tail buf");
    let warped = warp_buffer_wsola(&buf, |f| f as f64, frames);
    assert_eq!(warped.num_frames(), frames);
    let max_tail_err = warped.channels[0][1536..frames]
        .iter()
        .zip(&buf.channels[0][1536..frames])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_tail_err < 1e-5,
        "WSOLA identity mapping corrupted audio tail: max_tail_err={max_tail_err}"
    );
}
