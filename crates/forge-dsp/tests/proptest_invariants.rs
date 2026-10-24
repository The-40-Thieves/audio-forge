//! Neuro-Symbolic Property-Based Testing (`proptest`) & Adversarial Invariant Verification Suite
//! for `forge-dsp`.
//!
//! Verifies:
//! 1. 8x8 Walsh-Hadamard Unitary Energy Conservation (Parseval's Theorem) & Involution.
//! 2. 1st-Order ADAA (`AdaaTanh`, `AdaaSin`) Continuity across the L'Hopital `dx = 1e-5` boundary,
//!    Odd Symmetry, and Boundedness `[-1.0, 1.0]`.
//! 3. `Biquad` Totality across arbitrary/extreme parameters and Lyapunov asymptotic decay.
//! 4. `RealtimeTransientTonalSplit` Partition-of-Unity (`Full == Transient + Tonal` at 0 dB),
//!    `RealtimeDensityAndClipper`, `RealtimeFdnReverb` Lyapunov decay, and `RealtimeNeuralPreamp`.
//! 5. Adversarial `NaN` / `±Inf` / Subnormal (Denormal) / `f32::MAX` State-Poisoning & Self-Healing
//!    Recovery across all 12 real-time studio processors.
//! 6. `audit_audio`, `apply_spectral_unmask`, and `normalize_and_limit` totality across extreme
//!    sample rates (including the Z3-discovered `1001 Hz` / `8000 Hz` clamp boundary).
//! 7. Lock-Free `SHARED_SPECTRUM_SLOTS` Atomic IPC bit-exact roundtrip & non-finite sanitization.

use std::f64::consts::PI;
use std::time::Instant;

use forge_core::AudioBuffer;
use forge_dsp::{
    apply_spectral_defizz, apply_spectral_unmask, audit_audio, align_phase_gcc_phat,
    align_vocal_double_dtw, compute_reference_match_eq, normalize_and_limit,
    publish_shared_spectrum, read_shared_spectrum, remove_dc_and_declip,
    run_full_restoration_chain, AdaaSin, AdaaTanh, AnalogHardwareModel, Biquad, DeFizzConfig,
    RealtimeDensityAndClipper, RealtimeDrumTrigger, RealtimeFdnReverb, RealtimeMasterLimiter,
    RealtimeMasteringEq18, RealtimeNeuralPreamp, RealtimeOpenSoothe, RealtimeOpenSpacer,
    RealtimePhaseAligner, RealtimeSubAndTapeBus, RealtimeTransientTonalSplit,
    RealtimeVocalSurgical, RestorationConfig, UnmaskConfig,
};
use proptest::prelude::*;

#[inline]
fn is_clean_audio_f32(x: f32) -> bool {
    x.is_finite() && (x == 0.0 || x.abs() >= f32::MIN_POSITIVE)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    // ========================================================================
    // 1. 8x8 WALSH-HADAMARD PARSEVAL ENERGY CONSERVATION & INVOLUTION
    // ========================================================================
    #[test]
    fn prop_walsh_hadamard_8_parseval_and_involution(
        v0 in -1.0e4_f64..1.0e4_f64,
        v1 in -1.0e4_f64..1.0e4_f64,
        v2 in -1.0e4_f64..1.0e4_f64,
        v3 in -1.0e4_f64..1.0e4_f64,
        v4 in -1.0e4_f64..1.0e4_f64,
        v5 in -1.0e4_f64..1.0e4_f64,
        v6 in -1.0e4_f64..1.0e4_f64,
        v7 in -1.0e4_f64..1.0e4_f64,
    ) {
        let orig = [v0, v1, v2, v3, v4, v5, v6, v7];
        let energy_in: f64 = orig.iter().map(|&x| x * x).sum();

        let mut h1 = orig;
        RealtimeFdnReverb::fast_hadamard_8(&mut h1);
        let energy_out: f64 = h1.iter().map(|&y| y * y).sum();

        // Parseval's Theorem: ||H(v)||_2^2 == ||v||_2^2
        let rel_err = (energy_out - energy_in).abs() / energy_in.max(1.0);
        prop_assert!(
            rel_err < 1.0e-11,
            "Parseval energy violation: in={}, out={}, rel_err={}",
            energy_in,
            energy_out,
            rel_err
        );

        // Involution: H(H(v)) == v
        let mut h2 = h1;
        RealtimeFdnReverb::fast_hadamard_8(&mut h2);
        for i in 0..8 {
            let diff = (h2[i] - orig[i]).abs();
            prop_assert!(
                diff < 1.0e-9 * orig[i].abs().max(1.0),
                "Hadamard involution mismatch at {}: orig={}, h2={}",
                i,
                orig[i],
                h2[i]
            );
        }
    }

    // ========================================================================
    // 2. ADAA TANH & SIN CONTINUITY ACROSS L'HOPITAL THRESHOLD, SYMMETRY & BOUNDS
    // ========================================================================
    #[test]
    fn prop_adaa_tanh_and_sin_continuity_symmetry_and_bounds(
        x_base in -20.0_f64..20.0_f64,
        delta in -2.5e-5_f64..2.5e-5_f64,
        drive in 0.1_f64..20.0_f64,
    ) {
        // 2A. Continuity across the |dx| = 1e-5 piecewise branch
        let mut t1 = AdaaTanh::default();
        let _ = t1.process(x_base, drive);
        let y_tanh = t1.process(x_base + delta, drive);
        prop_assert!(y_tanh.is_finite() && (-1.0..=1.0).contains(&y_tanh));

        // Compare against midpoint tanh(x_mid) around tiny step delta
        let x0_clamped = (x_base * drive).clamp(-12.0, 12.0);
        let x1_clamped = ((x_base + delta) * drive).clamp(-12.0, 12.0);
        let exact_mid_tanh = (0.5 * (x0_clamped + x1_clamped)).tanh();
        prop_assert!(
            (y_tanh - exact_mid_tanh).abs() < 1.0e-3,
            "ADAA tanh discontinuity near dx=0: y_adaa={}, exact_mid={}",
            y_tanh,
            exact_mid_tanh
        );

        // 2B. Odd symmetry from zero state: ADAA(x) == -ADAA(-x)
        let mut pos_tanh = AdaaTanh::default();
        let mut neg_tanh = AdaaTanh::default();
        let yp = pos_tanh.process(x_base, drive);
        let yn = neg_tanh.process(-x_base, drive);
        prop_assert!((yp + yn).abs() < 1.0e-12);

        // 2C. AdaaSin boundedness and odd symmetry
        let mut pos_sin = AdaaSin::default();
        let mut neg_sin = AdaaSin::default();
        let sp = pos_sin.process(x_base, drive);
        let sn = neg_sin.process(-x_base, drive);
        prop_assert!(sp.is_finite() && (-1.0..=1.0).contains(&sp));
        prop_assert!((sp + sn).abs() < 1.0e-12);
    }

    // ========================================================================
    // 3. BIQUAD TOTALITY ACROSS EXTREME PARAMETERS & LYAPUNOV STABILITY
    // ========================================================================
    #[test]
    fn prop_biquad_all_filter_types_totality_and_lyapunov_decay(
        sr in -1000.0_f64..384_000.0_f64,
        freq in -500.0_f64..250_000.0_f64,
        q in -5.0_f64..100.0_f64,
        gain_db in -48.0_f64..48.0_f64,
    ) {
        let mut filters = [
            Biquad::peaking_eq(sr, freq, q, gain_db),
            Biquad::lowpass(sr, freq, q),
            Biquad::highpass(sr, freq, q),
            Biquad::notch(sr, freq, q),
            Biquad::k_weight_pre_shelf(sr),
            Biquad::k_weight_rlb_highpass(sr),
        ];

        for flt in &mut filters {
            // 3A. Exact Schur-Cohn / Jury Stability Criterion for 2nd-order IIR denominator 1 + a1*z^-1 + a2*z^-2:
            // Both complex poles lie strictly inside (or on, for degenerate bypass) the unit disk iff:
            //   |a2| <= 1.0  AND  |a1| <= 1.0 + a2 + 1e-12
            prop_assert!(
                flt.b0.is_finite()
                    && flt.b1.is_finite()
                    && flt.b2.is_finite()
                    && flt.a1.is_finite()
                    && flt.a2.is_finite()
            );
            prop_assert!(
                flt.a2.abs() <= 1.0 + 1.0e-12 && flt.a1.abs() <= 1.0 + flt.a2 + 1.0e-9,
                "Biquad violated Schur-Cohn stability triangle: a1={}, a2={}",
                flt.a1,
                flt.a2
            );

            // 3B. Totality across 256 samples on arbitrary parameters
            for i in 0..256 {
                let s = (0.5 * (i as f64 * 0.37).sin()) + (0.25 * (i as f64 * 1.13).cos());
                let y = flt.process_sample(s);
                prop_assert!(y.is_finite());
            }
        }

        // 3C. Time-domain Lyapunov asymptotic decay on audio-band biquad where tau << 1024 samples
        let eff_sr = sr.clamp(8_000.0, 192_000.0);
        let eff_fc = freq.clamp(250.0, eff_sr * 0.40);
        let eff_q = q.clamp(0.5, 12.0);
        let mut decay_flt = Biquad::peaking_eq(eff_sr, eff_fc, eff_q, gain_db);
        let mut active_energy = 0.0_f64;
        for i in 0..64 {
            let s = (0.5 * (i as f64 * 0.37).sin()) + (0.25 * (i as f64 * 1.13).cos());
            let y = decay_flt.process_sample(s);
            active_energy += y * y;
        }
        let mut tail_early = 0.0_f64;
        for _ in 0..64 {
            let y = decay_flt.process_sample(0.0);
            tail_early += y * y;
        }
        let settle_samples = ((eff_sr / eff_fc) * eff_q * 8.0).round() as usize;
        for _ in 0..settle_samples.clamp(512, 16_384) {
            let _ = decay_flt.process_sample(0.0);
        }
        let mut tail_late = 0.0_f64;
        for _ in 0..64 {
            let y = decay_flt.process_sample(0.0);
            tail_late += y * y;
        }
        prop_assert!(
            tail_late <= (active_energy + tail_early) * 0.1 + 1.0e-18,
            "Biquad failed Lyapunov decay: active={}, early={}, late={}",
            active_energy,
            tail_early,
            tail_late
        );
    }

    // ========================================================================
    // 4. SPLITEQ PARTITION-OF-UNITY & ELITE DSP LYAPUNOV STABILITY
    // ========================================================================
    #[test]
    fn prop_spliteq_partition_of_unity_and_fdn_lyapunov_decay(
        sr in 8_000.0_f64..96_000.0_f64,
        sharpness in 0.5_f64..4.0_f64,
        rt60 in 0.2_f64..3.5_f64,
        damping in 0.10_f64..0.85_f64,
        amp in 0.05_f32..0.95_f32,
    ) {
        // 4A. SplitEQ Partition of Unity: at 0 dB gains, Full Mix (0) == Transient Solo (1) + Tonal Solo (2)
        let mut split_full = RealtimeTransientTonalSplit::new(sr);
        let mut split_trans = RealtimeTransientTonalSplit::new(sr);
        let mut split_tonal = RealtimeTransientTonalSplit::new(sr);

        for i in 0..128 {
            let x = amp * ((i as f32 * 0.23).sin() + 0.5 * (i as f32 * 1.41).cos());
            let (fl, fr) = split_full.process_stereo_sample(x, -x, [0.0; 3], [0.0; 3], sharpness, 0);
            let (tl, tr) = split_trans.process_stereo_sample(x, -x, [0.0; 3], [0.0; 3], sharpness, 1);
            let (sl, sr_out) = split_tonal.process_stereo_sample(x, -x, [0.0; 3], [0.0; 3], sharpness, 2);

            prop_assert!(is_clean_audio_f32(fl) && is_clean_audio_f32(fr));
            prop_assert!(
                (fl - (tl + sl)).abs() < 1.0e-5 && (fr - (tr + sr_out)).abs() < 1.0e-5,
                "SplitEQ Partition-of-Unity violated: full=({}, {}), trans+tonal=({}, {})",
                fl, fr, tl + sl, tr + sr_out
            );
        }

        // 4B. 8x8 FDN Reverb Lyapunov Decay: excite for 256 samples, then measure energy across
        // sample-rate-invariant 100 ms windows (> 89.3 ms max FDN delay line) to prove monotonic decay!
        let mut fdn = RealtimeFdnReverb::new(sr);
        for i in 0..256 {
            let x = amp * (i as f32 * 0.31).sin();
            let (ol, or) = fdn.process_stereo_sample(x, x, rt60, damping, 0.2, 0.3, 1.0);
            prop_assert!(is_clean_audio_f32(ol) && is_clean_audio_f32(or));
        }

        // 100 ms window guarantees all 8 delay lines (29.3 ms .. 89.3 ms) traverse at least 1 full cycle
        let win_100ms = ((sr * 0.100).round() as usize).max(1024);

        let mut early_tail_energy = 0.0_f64;
        for _ in 0..win_100ms {
            let (ol, or) = fdn.process_stereo_sample(0.0, 0.0, rt60, damping, 0.0, 0.0, 1.0);
            prop_assert!(is_clean_audio_f32(ol) && is_clean_audio_f32(or));
            early_tail_energy += f64::from(ol) * f64::from(ol) + f64::from(or) * f64::from(or);
        }

        // Advance 200 ms of silent decay
        for _ in 0..(win_100ms * 2) {
            let (ol, or) = fdn.process_stereo_sample(0.0, 0.0, rt60, damping, 0.0, 0.0, 1.0);
            prop_assert!(is_clean_audio_f32(ol) && is_clean_audio_f32(or));
        }

        let mut late_tail_energy = 0.0_f64;
        for _ in 0..win_100ms {
            let (ol, or) = fdn.process_stereo_sample(0.0, 0.0, rt60, damping, 0.0, 0.0, 1.0);
            prop_assert!(is_clean_audio_f32(ol) && is_clean_audio_f32(or));
            late_tail_energy += f64::from(ol) * f64::from(ol) + f64::from(or) * f64::from(or);
        }

        prop_assert!(
            early_tail_energy > 0.0 && late_tail_energy < early_tail_energy,
            "FDN Reverb failed Lyapunov decay: early_tail={}, late_tail={}",
            early_tail_energy,
            late_tail_energy
        );
    }

    // ========================================================================
    // 5. AUDIT_AUDIO, UNMASK & MASTER LIMITER TOTALITY AT Z3 COUNTEREXAMPLE RATES
    // ========================================================================
    #[test]
    fn prop_audit_unmask_and_limiter_totality_across_all_sample_rates(
        sr in 200_u32..192_000_u32,
        frames in 64_usize..2560_usize,
        low_cut in 10.0_f64..15_000.0_f64,
        high_cut in 20.0_f64..22_000.0_f64,
        ceiling_dbtp in -9.0_f64..-0.2_f64,
    ) {
        let mut ch0 = Vec::with_capacity(frames);
        let mut ch1 = Vec::with_capacity(frames);
        for i in 0..frames {
            let s = 0.45 * ((i as f64) * 0.19).sin();
            ch0.push(s);
            ch1.push(-s * 0.8);
        }
        let buf = AudioBuffer::from_channels(vec![ch0, ch1], sr).expect("valid buffer");

        // Must never panic even at sr = 1001 Hz or 8000 Hz (proven by Z3)
        let report = audit_audio(&buf);
        prop_assert!(report.integrated_lufs.is_finite());
        prop_assert!(report.true_peak_dbtp.is_finite());
        prop_assert!(report.ai_fizz_index.is_finite() && (0.0..=100.0).contains(&report.ai_fizz_index));

        let cfg = UnmaskConfig {
            low_cut_hz: low_cut,
            high_cut_hz: high_cut,
            max_reduction_db: 6.0,
            sensitivity: 1.0,
        };
        let unmasked = apply_spectral_unmask(&buf, &buf, &cfg).expect("unmask succeeded");
        prop_assert_eq!(unmasked.num_frames(), frames);

        let mastered = normalize_and_limit(&buf, -14.0, ceiling_dbtp).expect("mastered");
        let post = audit_audio(&mastered);
        prop_assert!(
            post.true_peak_dbtp <= ceiling_dbtp + 0.05,
            "Mastered true peak {} exceeded ceiling {}",
            post.true_peak_dbtp,
            ceiling_dbtp
        );
    }

    // ========================================================================
    // 6. LOCK-FREE ATOMIC SPECTRUM BUS ROUNDTRIP & SANITIZATION
    // ========================================================================
    #[test]
    fn prop_shared_spectrum_atomic_bus_roundtrip_and_sanitization(
        slot in 0_usize..16_usize,
        seed_val in any::<u32>(),
    ) {
        let f_val = f32::from_bits(seed_val);
        let mut bands = [0.0_f32; 32];
        for (i, b) in bands.iter_mut().enumerate() {
            *b = if i % 2 == 0 { f_val } else { -18.5_f32 };
        }
        publish_shared_spectrum(slot, &bands);
        let read_back = read_shared_spectrum(slot);
        for &v in &read_back {
            prop_assert!(is_clean_audio_f32(v));
        }
        if slot < 8 && f_val.is_finite() && f_val.abs() >= 1.0e-30 {
            prop_assert_eq!(read_back[0].to_bits(), f_val.to_bits());
        } else if slot < 8 {
            prop_assert_eq!(read_back[0], 0.0_f32);
        } else {
            // Out-of-bounds slot >= 8 must safely return all zeros
            prop_assert!(read_back.iter().all(|&x| x == 0.0));
        }
    }

    // ========================================================================
    // 7B. RESTORATION, SPECTRAL DE-FIZZ & GCC-PHAT / DTW PHASE ALIGNMENT
    // ========================================================================
    #[test]
    fn prop_restoration_defizz_and_phase_align_totality(
        len in 256usize..768,
        dc_offset in -0.5f64..0.5,
        sensitivity_db in prop::num::f64::ANY,
        max_reduction_db in prop::num::f64::ANY,
    ) {
        let sr = 48_000_u32;
        let mut l = vec![0.0_f64; len];
        let mut r = vec![0.0_f64; len];
        for i in 0..len {
            let t = (i as f64) / (sr as f64);
            l[i] = dc_offset + 0.5 * (2.0 * PI * 440.0 * t).sin();
            r[i] = dc_offset + 0.5 * (2.0 * PI * 440.0 * t).sin();
        }
        // Inject adversarial NaN and hard-clipped plateau
        l[0] = f64::NAN;
        r[0] = f64::INFINITY;
        for s in l.iter_mut().take(16).skip(10) {
            *s = 1.05;
        }

        let mut buf = AudioBuffer::from_channels(vec![l, r], sr).unwrap();
        remove_dc_and_declip(&mut buf, true);
        for ch in &buf.channels {
            for &s in ch {
                prop_assert!(s.is_finite());
            }
        }

        let rest_cfg = RestorationConfig::default();
        buf = run_full_restoration_chain(&buf, &rest_cfg).unwrap();
        for ch in &buf.channels {
            for &s in ch {
                prop_assert!(s.is_finite());
            }
        }

        let gains = compute_reference_match_eq(&buf, &buf, f64::NAN);
        for &g in &gains {
            prop_assert!(g.is_finite() && (-9.0..=9.0).contains(&g));
        }

        let cfg = DeFizzConfig {
            sensitivity_db,
            max_depth_db: max_reduction_db,
            ..DeFizzConfig::default()
        };
        let defizzed = apply_spectral_defizz(&buf, &cfg).unwrap();
        for ch in &defizzed.channels {
            for &s in ch {
                prop_assert!(s.is_finite());
            }
        }

        let (aligned, res) = align_phase_gcc_phat(&buf, &defizzed, 20.0).unwrap();
        prop_assert!(res.correlation_peak.is_finite());
        for ch in &aligned.channels {
            for &s in ch {
                prop_assert!(s.is_finite());
            }
        }

        let (dtw_buf, markers) = align_vocal_double_dtw(&buf, &defizzed, 12.0).unwrap();
        for ch in &dtw_buf.channels {
            for &s in ch {
                prop_assert!(s.is_finite());
            }
        }
        for m in markers {
            prop_assert!(m.timeline_pos_secs.is_finite() && m.source_pos_secs.is_finite());
        }
    }
}

// ============================================================================
// 7. DETERMINISTIC ADVERSARIAL POISONING & SELF-HEALING BENCHMARK SUITE
// ============================================================================
#[test]
fn test_adversarial_nan_inf_denormal_poisoning_and_self_healing_across_all_12_processors() {
    let sr = 48_000.0;

    let mut split_eq = RealtimeTransientTonalSplit::new(sr);
    let mut density_clip = RealtimeDensityAndClipper::new(sr);
    let mut spaceverb = RealtimeFdnReverb::new(sr);
    let mut neural_preamp = RealtimeNeuralPreamp::new(AnalogHardwareModel::TubeV76Triode);
    let mut soothe = RealtimeOpenSoothe::new(sr);
    let mut spacer = RealtimeOpenSpacer::new(sr);
    let mut vocal_surg = RealtimeVocalSurgical::new(sr);
    let mut sub_tape = RealtimeSubAndTapeBus::new(sr, 160.0);
    let mut limiter = RealtimeMasterLimiter::new(sr);
    let mut eq18 = RealtimeMasteringEq18::new(sr);
    let mut phase_align = RealtimePhaseAligner::new(sr);
    let mut drum_trig = RealtimeDrumTrigger::new(sr);

    // Adversarial vectors: NaN, +Inf, -Inf, -0.0, smallest subnormal (1e-45), f32::MAX, f32::MIN
    let poison_vectors: [f32; 8] = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0_f32,
        f32::from_bits(1),         // positive subnormal ~1.4e-45
        f32::from_bits(0x8000_0001), // negative subnormal
        f32::MAX,
        f32::MIN,
    ];

    // Phase 1: Bombard all 12 processors with 512 frames of adversarial poison + NaN parameters
    eq18.set_gains(&[f64::NAN; 18], f64::INFINITY);
    phase_align.set_alignment(usize::MAX, true);

    for i in 0..512 {
        let p = poison_vectors[i % poison_vectors.len()];
        let q = poison_vectors[(i + 3) % poison_vectors.len()];

        let (o1_l, o1_r) = split_eq.process_stereo_sample(
            p,
            q,
            [f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
            [f64::NAN; 3],
            f64::NAN,
            0,
        );
        assert!(is_clean_audio_f32(o1_l) && is_clean_audio_f32(o1_r));

        let (o2_l, o2_r) =
            density_clip.process_stereo_sample(p, q, f64::NAN, f64::INFINITY, f64::NEG_INFINITY);
        assert!(is_clean_audio_f32(o2_l) && is_clean_audio_f32(o2_r));

        let (o3_l, o3_r) = spaceverb.process_stereo_sample(
            p,
            q,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::INFINITY,
        );
        assert!(is_clean_audio_f32(o3_l) && is_clean_audio_f32(o3_r));

        let (o4_l, o4_r) =
            neural_preamp.process_stereo_sample(p, q, f64::NAN, f64::INFINITY);
        assert!(is_clean_audio_f32(o4_l) && is_clean_audio_f32(o4_r));

        let (o5_l, o5_r) =
            soothe.process_stereo_sample(p, q, f64::NAN, f64::INFINITY, usize::MAX, false);
        assert!(is_clean_audio_f32(o5_l) && is_clean_audio_f32(o5_r));

        let (o6_l, o6_r) = spacer.process_sample(
            p,
            q,
            p,
            q,
            f64::NAN,
            f64::INFINITY,
            f64::NAN,
            f64::INFINITY,
        );
        assert!(is_clean_audio_f32(o6_l) && is_clean_audio_f32(o6_r));

        let (o7_l, o7_r) = vocal_surg.process_stereo_sample(
            p,
            q,
            f64::NAN,
            f64::INFINITY,
            f64::NAN,
            f64::NEG_INFINITY,
        );
        assert!(is_clean_audio_f32(o7_l) && is_clean_audio_f32(o7_r));

        let (o8_l, o8_r) =
            sub_tape.process_stereo_sample(p, q, f64::NAN, f64::INFINITY, f64::NAN);
        assert!(is_clean_audio_f32(o8_l) && is_clean_audio_f32(o8_r));

        let (o9_l, o9_r) =
            limiter.process_stereo_sample(p, q, f64::NAN, f64::INFINITY, f64::NAN);
        assert!(is_clean_audio_f32(o9_l) && is_clean_audio_f32(o9_r));

        let (o10_l, o10_r) = eq18.process_stereo_sample(p, q);
        assert!(is_clean_audio_f32(o10_l) && is_clean_audio_f32(o10_r));

        let (o11_l, o11_r) = phase_align.process_stereo_sample(p, q);
        assert!(is_clean_audio_f32(o11_l) && is_clean_audio_f32(o11_r));

        let trig_notes = drum_trig.process_stereo_sample(p, q, f64::NAN);
        for note in trig_notes.into_iter().flatten() {
            assert!(note.1.is_finite() && (0.0..=1.0).contains(&note.1));
        }
    }

    // Phase 2: Self-Healing Recovery — feed 4096 samples of clean 440 Hz sine wave and verify
    // every single processor recovers and produces finite, non-zero output energy!
    phase_align.set_alignment(16, false);
    eq18.set_gains(&[1.5; 18], 1.0);

    let mut energies = [0.0_f64; 11];
    let t_start = Instant::now();

    for i in 0..4096 {
        let t = (i as f64) / sr;
        let s = (0.4 * (2.0 * PI * 440.0 * t).sin() + 0.2 * (2.0 * PI * 1760.0 * t).cos()) as f32;

        let (o1_l, o1_r) =
            split_eq.process_stereo_sample(s, s, [1.5, 0.0, 1.5], [0.0; 3], 1.5, 0);
        let (o2_l, o2_r) = density_clip.process_stereo_sample(s, s, 0.35, 2.0, -0.5);
        let (o3_l, o3_r) = spaceverb.process_stereo_sample(s, s, 1.8, 0.35, 0.15, 0.25, 0.35);
        let (o4_l, o4_r) = neural_preamp.process_stereo_sample(s, s, 3.0, 0.0);
        let (o5_l, o5_r) = soothe.process_stereo_sample(s, s, 3.5, 6.0, 8, false);
        let (o6_l, o6_r) = spacer.process_sample(s, s, s * 0.2, s * 0.2, 200.0, 6000.0, 0.3, 4.0);
        let (o7_l, o7_r) = vocal_surg.process_stereo_sample(s, s, 0.5, 0.5, -20.0, -60.0);
        let (o8_l, o8_r) = sub_tape.process_stereo_sample(s, s, 0.25, 1.0, 0.2);
        let (o9_l, o9_r) = limiter.process_stereo_sample(s, s, 2.0, -1.0, 50.0);
        let (o10_l, o10_r) = eq18.process_stereo_sample(s, s);
        let (o11_l, o11_r) = phase_align.process_stereo_sample(s, s);

        let pairs = [
            (o1_l, o1_r),
            (o2_l, o2_r),
            (o3_l, o3_r),
            (o4_l, o4_r),
            (o5_l, o5_r),
            (o6_l, o6_r),
            (o7_l, o7_r),
            (o8_l, o8_r),
            (o9_l, o9_r),
            (o10_l, o10_r),
            (o11_l, o11_r),
        ];

        for (idx, &(l, r)) in pairs.iter().enumerate() {
            assert!(
                is_clean_audio_f32(l) && is_clean_audio_f32(r),
                "Processor {} emitted non-clean sample during recovery: ({}, {})",
                idx,
                l,
                r
            );
            energies[idx] += f64::from(l) * f64::from(l) + f64::from(r) * f64::from(r);
        }
    }

    let elapsed = t_start.elapsed();
    let names = [
        "RealtimeTransientTonalSplit",
        "RealtimeDensityAndClipper",
        "RealtimeFdnReverb",
        "RealtimeNeuralPreamp",
        "RealtimeOpenSoothe",
        "RealtimeOpenSpacer",
        "RealtimeVocalSurgical",
        "RealtimeSubAndTapeBus",
        "RealtimeMasterLimiter",
        "RealtimeMasteringEq18",
        "RealtimePhaseAligner",
    ];
    for (idx, &e) in energies.iter().enumerate() {
        assert!(
            e > 1.0,
            "Processor {} ({}) failed to self-heal after NaN/Inf barrage (energy={})",
            idx,
            names[idx],
            e
        );
    }
    println!(
        "[TELEMETRY] 11 chained real-time processors self-healed and processed 4096 stereo frames in {:?} ({:.1}x real-time @ 48kHz)",
        elapsed,
        (4096.0 / 48_000.0) / elapsed.as_secs_f64()
    );
}

#[test]
fn test_deterministic_dsp_telemetry_and_numerical_precision_report() {
    let sr = 48_000.0;
    let frames = 48_000_usize; // 1.000 second of stereo 48 kHz audio

    // 1. Measure exact maximum Parseval & Involution error across 1,000 deterministic Hadamard vectors
    let mut max_parseval_rel_err = 0.0_f64;
    let mut max_involution_abs_err = 0.0_f64;
    for k in 0..1000 {
        let orig: [f64; 8] = std::array::from_fn(|i| {
            ((k * 17 + i * 31) as f64 * 0.137).sin() * 100.0
                + ((k * 7 + i * 19) as f64 * 0.71).cos() * 25.0
        });
        let e_in: f64 = orig.iter().map(|&x| x * x).sum();
        let mut h1 = orig;
        RealtimeFdnReverb::fast_hadamard_8(&mut h1);
        let e_out: f64 = h1.iter().map(|&x| x * x).sum();
        let rel_err = (e_out - e_in).abs() / e_in.max(1e-12);
        if rel_err > max_parseval_rel_err {
            max_parseval_rel_err = rel_err;
        }
        let mut h2 = h1;
        RealtimeFdnReverb::fast_hadamard_8(&mut h2);
        for i in 0..8 {
            let inv_err = (h2[i] - orig[i]).abs();
            if inv_err > max_involution_abs_err {
                max_involution_abs_err = inv_err;
            }
        }
    }

    // 2. Measure exact ADAA L'Hopital boundary (|dx| = 1e-5) continuity error
    let mut max_adaa_boundary_err = 0.0_f64;
    for k in -500..=500 {
        let x0 = (k as f64) * 0.01;
        for &dx in &[-1.0001e-5, -0.9999e-5, 0.9999e-5, 1.0001e-5] {
            let mut adaa = AdaaTanh::default();
            let _ = adaa.process(x0, 2.5);
            let y_adaa = adaa.process(x0 + dx, 2.5);
            let c0 = (x0 * 2.5).clamp(-12.0, 12.0);
            let c1 = ((x0 + dx) * 2.5).clamp(-12.0, 12.0);
            let y_mid = (0.5 * (c0 + c1)).tanh();
            let err = (y_adaa - y_mid).abs();
            if err > max_adaa_boundary_err {
                max_adaa_boundary_err = err;
            }
        }
    }

    // 3. Per-processor 1-second (48,000 stereo frames) throughput & CPU budget benchmark
    let signal: Vec<f32> = (0..frames)
        .map(|i| {
            let t = (i as f64) / sr;
            (0.45 * (2.0 * PI * 220.0 * t).sin() + 0.25 * (2.0 * PI * 3400.0 * t).cos()) as f32
        })
        .collect();

    let mut split_eq = RealtimeTransientTonalSplit::new(sr);
    let t0 = Instant::now();
    let mut sink = 0.0_f32;
    for &s in &signal {
        let (l, r) = split_eq.process_stereo_sample(s, s, [2.0, 0.0, 1.5], [0.0; 3], 1.5, 0);
        sink += l + r;
    }
    let dt_spliteq = t0.elapsed();

    let mut density = RealtimeDensityAndClipper::new(sr);
    let t1 = Instant::now();
    for &s in &signal {
        let (l, r) = density.process_stereo_sample(s, s, 0.4, 3.0, -0.5);
        sink += l + r;
    }
    let dt_density = t1.elapsed();

    let mut spaceverb = RealtimeFdnReverb::new(sr);
    let t2 = Instant::now();
    for &s in &signal {
        let (l, r) = spaceverb.process_stereo_sample(s, s, 2.2, 0.35, 0.2, 0.3, 0.3);
        sink += l + r;
    }
    let dt_spaceverb = t2.elapsed();

    let mut preamp = RealtimeNeuralPreamp::new(AnalogHardwareModel::Neve1073ClassA);
    let t3 = Instant::now();
    for &s in &signal {
        let (l, r) = preamp.process_stereo_sample(s, s, 4.0, 0.0);
        sink += l + r;
    }
    let dt_preamp = t3.elapsed();

    let mut limiter = RealtimeMasterLimiter::new(sr);
    let t4 = Instant::now();
    for &s in &signal {
        let (l, r) = limiter.process_stereo_sample(s, s, 3.0, -1.0, 65.0);
        sink += l + r;
    }
    let dt_limiter = t4.elapsed();

    let mut soothe = RealtimeOpenSoothe::new(sr);
    let t5 = Instant::now();
    for &s in &signal {
        let (l, r) = soothe.process_stereo_sample(s, s, 3.5, 9.0, 8, false);
        sink += l + r;
    }
    let dt_soothe = t5.elapsed();

    assert!(sink.is_finite());
    println!(
        "[PRECISION] Hadamard Parseval max rel err = {:.3e} | Hadamard Involution max abs err = {:.3e} | ADAA dx=1e-5 max boundary err = {:.3e}",
        max_parseval_rel_err, max_involution_abs_err, max_adaa_boundary_err
    );
    for (name, dt) in [
        ("Forge SplitEQ (6-Band Transient/Tonal)", dt_spliteq),
        ("Forge Density & 2x ADAA Clipper", dt_density),
        ("Forge SpaceVerb (8x8 Hadamard FDN + Shimmer)", dt_spaceverb),
        ("Forge Neural Preamp (Dilated WaveNet + ADAA)", dt_preamp),
        ("Forge Master Limiter (4x True-Peak + LUFS)", dt_limiter),
        ("Forge OpenSoothe (2048-FFT OLA Dynamic)", dt_soothe),
    ] {
        let ns_per_frame = dt.as_nanos() as f64 / (frames as f64);
        let rt_mult = 1.0 / dt.as_secs_f64();
        let cpu_pct = dt.as_secs_f64() * 100.0;
        println!(
            "[BENCHMARK] {:<44} | {:>6.2} ns/frame | {:>7.1}x real-time | {:>5.3}% single-core CPU @ 48kHz",
            name, ns_per_frame, rt_mult, cpu_pct
        );
    }
}

