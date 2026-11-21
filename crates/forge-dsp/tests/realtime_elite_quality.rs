//! Quality, continuity, and sample-rate invariants for Piece #4
//! (Real-Time DSP & Elite Processors: `elite_dsp.rs` & `realtime.rs`).

use std::f64::consts::PI;

use forge_dsp::{RealtimeDensityAndClipper, RealtimeFdnReverb, RealtimeTransientTonalSplit};

#[test]
fn spaceverb_modulated_tail_is_smooth_without_zipper_clicks() {
    let sr = 48_000.0;
    let mut verb = RealtimeFdnReverb::new(sr);

    // Excite with a smooth 440 Hz sine burst (150 ms)
    let excite = (sr * 0.15) as usize;
    for i in 0..excite {
        let env = (PI * (i as f64) / (excite as f64)).sin();
        let s = (0.3 * env * (2.0 * PI * 440.0 * (i as f64) / sr).sin()) as f32;
        let _ = verb.process_stereo_sample(s, s, 3.0, 0.4, 0.0, 0.0, 1.0);
    }

    // Collect 0.5 s of modulated tail (LFO sweeps through multiple full-sample crossings)
    let tail_len = (sr * 0.5) as usize;
    let mut tail = Vec::with_capacity(tail_len);
    for _ in 0..tail_len {
        let (l, _) = verb.process_stereo_sample(0.0, 0.0, 3.0, 0.4, 0.0, 0.0, 1.0);
        tail.push(f64::from(l));
    }

    let rms = (tail.iter().map(|x| x * x).sum::<f64>() / (tail.len() as f64)).sqrt();
    assert!(
        rms > 1e-4,
        "expected audible reverb tail, got rms={rms:.3e}"
    );

    // For a 440 Hz damped sine tail at 48 kHz, max natural sample-to-sample step is
    // ~2*pi*440/48000 ≈ 0.058 * amplitude. Integer-jump zipper glitches produce steps >> 0.5 * rms.
    let max_step = tail
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_step / rms < 0.45,
        "detected zipper discontinuity in modulated FDN tail: max_step/rms = {:.3}",
        max_step / rms
    );
}

#[test]
fn spaceverb_reset_clears_state_in_place() {
    let sr = 48_000.0;
    let mut verb = RealtimeFdnReverb::new(sr);
    for i in 0..4096 {
        let s = (0.5 * (2.0 * PI * 330.0 * (i as f64) / sr).sin()) as f32;
        let _ = verb.process_stereo_sample(s, s, 4.0, 0.3, 0.4, 0.3, 0.8);
    }

    verb.reset(sr);
    for _ in 0..512 {
        let (l, r) = verb.process_stereo_sample(0.0, 0.0, 4.0, 0.3, 0.4, 0.3, 0.8);
        assert_eq!(l, 0.0, "expected silent output after in-place reset");
        assert_eq!(r, 0.0, "expected silent output after in-place reset");
    }
}

#[test]
fn density_clipper_has_smooth_gain_across_minus_48db_floor() {
    let sr = 48_000.0;
    let measure_gain_db = |input_db: f64| -> f64 {
        let mut proc = RealtimeDensityAndClipper::new(sr);
        let amp = 10.0_f64.powf(input_db / 20.0) * std::f64::consts::SQRT_2;
        let n = 4800_usize;
        let mut out_sq = 0.0_f64;
        let mut in_sq = 0.0_f64;
        for i in 0..n {
            let s = amp * (2.0 * PI * 1000.0 * (i as f64) / sr).sin();
            let (ol, _) = proc.process_stereo_sample(s as f32, s as f32, 1.0, 0.0, 0.0);
            if i >= n / 2 {
                in_sq += s * s;
                out_sq += f64::from(ol) * f64::from(ol);
            }
        }
        10.0 * (out_sq / in_sq.max(1e-18)).log10()
    };

    // Step across -54 dB -> -50 dB -> -48 dB -> -46 dB in 2 dB steps:
    // Previously there was a ~9 dB cliff right across -48 dB; now every 2 dB step is smooth.
    let levels = [-56.0, -54.0, -52.0, -50.0, -48.0, -46.0, -44.0];
    let gains: Vec<f64> = levels.iter().map(|&db| measure_gain_db(db)).collect();
    for (pair_lvls, pair_gains) in levels.windows(2).zip(gains.windows(2)) {
        let jump = (pair_gains[1] - pair_gains[0]).abs();
        assert!(
            jump < 3.0,
            "hard gain cliff between {} dB and {} dB: {:.2} dB -> {:.2} dB (jump {jump:.2} dB)",
            pair_lvls[0],
            pair_lvls[1],
            pair_gains[0],
            pair_gains[1]
        );
    }
}

#[test]
fn spliteq_transient_detection_is_consistent_across_48k_and_96k() {
    let measure_transient_peak = |sr: f64| -> f64 {
        let mut split = RealtimeTransientTonalSplit::new(sr);
        let frames = (sr * 0.25) as usize;
        let onset = (sr * 0.10) as usize;
        let burst = (sr * 0.004) as usize;
        let mut peak = 0.0_f64;

        for i in 0..frames {
            let t = (i as f64) / sr;
            let amp = if i >= onset && i < onset + burst {
                0.70
            } else {
                0.08
            };
            let s = (amp * (2.0 * PI * 1000.0 * t).sin()) as f32;
            // audition_mode = 1 -> Solo Transients Only
            let (ol, _) = split.process_stereo_sample(s, s, [0.0; 3], [0.0; 3], 1.5, 1);
            if i >= onset && i < onset + burst {
                peak = peak.max(f64::from(ol).abs());
            }
        }
        peak
    };

    let p48 = measure_transient_peak(48_000.0);
    let p96 = measure_transient_peak(96_000.0);

    assert!(
        p48 > 0.2 && p96 > 0.2,
        "expected strong transient extraction: 48k={p48:.3}, 96k={p96:.3}"
    );
    assert!(
        (p48 - p96).abs() / p48 < 0.15,
        "SplitEQ transient response diverged across sample rates: 48k={p48:.3}, 96k={p96:.3}"
    );
}
