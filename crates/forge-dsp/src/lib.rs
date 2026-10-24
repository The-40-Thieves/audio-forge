//! `forge-dsp` — Studio DSP, Restoration, Spectral Suppression, Unmasking, Alignment & Mastering.

pub mod elite_dsp;
pub mod loudness_limiter;
pub mod phase_align;
pub mod realtime;
pub mod restoration;
pub mod spectral_defizz;
pub mod spectral_unmask;

pub use elite_dsp::{
    compute_genre_tonal_balance_eq, publish_shared_spectrum, read_shared_spectrum, AdaaSin,
    AdaaTanh, AllpassDisperser, AnalogHardwareModel, AutoGainCompensator, GenreTonalTarget,
    JilesAthertonHysteresis, NamWaveNetProfile, Oversampler2x, RealtimeDensityAndClipper,
    RealtimeFdnReverb, RealtimeNeuralPreamp, RealtimeSpectrumAnalyzer32,
    RealtimeTransientTonalSplit, SHARED_SPECTRUM_SLOTS,
};
pub use loudness_limiter::{
    audit_audio, compute_true_peak, flush_denormal_f64, normalize_and_limit, sanitize_param_f64,
    sanitize_sample_f32, sanitize_sample_f64, Biquad, CytomicSvf,
};
pub use phase_align::{
    align_phase_gcc_phat, align_vocal_double_dtw, PhaseAlignmentResult, StretchMarker,
};
pub use realtime::{
    ParamSmoother, RealtimeDrumTrigger, RealtimeMasterLimiter, RealtimeMasteringEq18,
    RealtimeOpenSoothe, RealtimeOpenSpacer, RealtimePhaseAligner, RealtimeSubAndTapeBus,
    RealtimeVocalSurgical,
};
pub use restoration::{
    apply_18_band_eq, apply_analog_console_and_tape, apply_hum_notch_cascade,
    apply_mid_side_spatial_cleanup, clean_vocal_surgical, compute_reference_match_eq,
    remove_dc_and_declip, run_full_restoration_chain, RestorationConfig, MASTERING_18_BANDS_HZ,
};
pub use spectral_defizz::{apply_spectral_defizz, DeFizzConfig};
pub use spectral_unmask::{apply_spectral_unmask, UnmaskConfig};

#[cfg(test)]
mod tests {
    use super::*;
    use forge_core::AudioBuffer;
    use std::f64::consts::PI;

    #[test]
    fn test_bs1770_loudness_and_brickwall_true_peak_limiter() {
        let sr = 48_000_u32;
        let frames = (sr as usize) * 2; // 2 seconds
        let mut ch0 = vec![0.0; frames];
        let mut ch1 = vec![0.0; frames];
        for i in 0..frames {
            let t = (i as f64) / (sr as f64);
            let s = 0.25 * (2.0 * PI * 997.0 * t).sin();
            ch0[i] = s;
            ch1[i] = s;
        }
        let buf = AudioBuffer::from_channels(vec![ch0, ch1], sr).expect("valid buffer");
        let report = audit_audio(&buf);
        assert!(report.integrated_lufs < -8.0 && report.integrated_lufs > -16.0);

        let mastered = normalize_and_limit(&buf, -14.0, -1.0).expect("mastered");
        let post = audit_audio(&mastered);
        assert!(
            post.true_peak_dbtp <= -0.95,
            "True peak {} exceeded ceiling",
            post.true_peak_dbtp
        );
    }

    #[test]
    fn test_opensoothe_spectral_defizz_suppresses_harsh_whistle() {
        let sr = 48_000_u32;
        let frames = sr as usize; // 1 second
        let mut ch = vec![0.0; frames];
        for (i, sample) in ch.iter_mut().enumerate() {
            let t = (i as f64) / (sr as f64);
            // Warm 220 Hz fundamental + harsh 4200 Hz metallic AI whistle
            let warm = 0.3 * (2.0 * PI * 220.0 * t).sin();
            let harsh = 0.45 * (2.0 * PI * 4200.0 * t).sin();
            *sample = warm + harsh;
        }
        let buf = AudioBuffer::from_channels(vec![ch.clone(), ch], sr).expect("valid");
        let pre = audit_audio(&buf);
        let cleaned = apply_spectral_defizz(&buf, &DeFizzConfig::default()).expect("defizzed");
        let post = audit_audio(&cleaned);
        assert!(
            post.ai_fizz_index < pre.ai_fizz_index,
            "Expected fizz index to drop: before={}, after={}",
            pre.ai_fizz_index,
            post.ai_fizz_index
        );
    }

    #[test]
    fn test_gcc_phat_recovers_exact_delay_and_inverted_polarity() {
        let sr = 48_000_u32;
        let frames = 16_384;
        let delay = 23_isize;
        let mut ref_ch = vec![0.0; frames];
        for (i, s) in ref_ch.iter_mut().enumerate() {
            let t = (i as f64) / (sr as f64);
            *s = 0.5 * (2.0 * PI * 310.0 * t).sin()
                + 0.3 * (2.0 * PI * 1150.0 * t * (1.0 + 0.2 * t)).cos();
        }
        let mut tgt_ch = vec![0.0; frames];
        for i in (delay as usize)..frames {
            // Delayed by +23 samples and polarity-inverted
            tgt_ch[i] = -ref_ch[i - (delay as usize)];
        }
        let ref_buf = AudioBuffer::from_channels(vec![ref_ch], sr).expect("ref");
        let tgt_buf = AudioBuffer::from_channels(vec![tgt_ch], sr).expect("tgt");
        let (_aligned, res) = align_phase_gcc_phat(&ref_buf, &tgt_buf, 20.0).expect("aligned");
        assert_eq!(res.delay_samples, delay);
        assert!(res.invert_polarity);
    }

    #[test]
    fn test_realtime_zero_alloc_streaming_processors() {
        let sr = 48_000.0;
        let mut soothe = RealtimeOpenSoothe::new(sr);
        assert_eq!(soothe.latency_samples(), 1536);

        let mut limiter = RealtimeMasterLimiter::new(sr);
        let mut vocal_strip = RealtimeVocalSurgical::new(sr);
        let mut sub_tape = RealtimeSubAndTapeBus::new(sr, 160.0);
        let mut drum_trig = RealtimeDrumTrigger::new(sr);

        let mut triggered_kick = false;
        for i in 0..4096 {
            let t = (i as f64) / sr;
            // Kick burst at sample 500 + harsh 4.2kHz sine
            let kick = if (500..750).contains(&i) {
                0.85 * (2.0 * PI * 68.0 * t).sin()
            } else {
                0.0
            };
            let s = (kick + 0.3 * (2.0 * PI * 4200.0 * t).sin()) as f32;
            let (v_l, v_r) = vocal_strip.process_stereo_sample(s, s, 0.7, 0.7, -20.0, -50.0);
            let (d_l, d_r) = soothe.process_stereo_sample(v_l, v_r, 3.5, 9.0, 8, false);
            let (b_l, b_r) = sub_tape.process_stereo_sample(d_l, d_r, 0.3, 1.05, 0.25);
            let (out_l, out_r) = limiter.process_stereo_sample(b_l, b_r, 3.0, -1.0, 85.0);
            assert!(out_l.abs() <= 0.90 && out_r.abs() <= 0.90);

            let notes = drum_trig.process_stereo_sample(s, s, -24.0);
            if notes[0].is_some() {
                triggered_kick = true;
            }
        }
        assert!(triggered_kick, "Expected real-time kick trigger on C1");
    }

    #[test]
    fn test_elite_dsp_spliteq_density_adaa_spaceverb_neural_and_shared_bus() {
        let sr = 48_000.0;
        let mut split_eq = RealtimeTransientTonalSplit::new(sr);
        let mut density_clip = RealtimeDensityAndClipper::new(sr);
        let mut spaceverb = RealtimeFdnReverb::new(sr);
        let mut preamp = RealtimeNeuralPreamp::new(AnalogHardwareModel::Neve1073ClassA);

        let mut energy_out = 0.0_f32;
        for i in 0..2048 {
            let t = (i as f64) / sr;
            let s = (0.4 * (2.0 * PI * 220.0 * t).sin() + 0.2 * (2.0 * PI * 3200.0 * t).cos()) as f32;
            let (sp_l, sp_r) = split_eq.process_stereo_sample(
                s,
                s,
                [2.0, 0.0, -2.0],
                [0.5, 0.0, 1.5],
                1.5,
                0,
            );
            let (dn_l, dn_r) = density_clip.process_stereo_sample(sp_l, sp_r, 0.4, 3.0, -0.8);
            let (pr_l, pr_r) = preamp.process_stereo_sample(dn_l, dn_r, 4.0, 0.0);
            let (rv_l, rv_r) = spaceverb.process_stereo_sample(pr_l, pr_r, 0.65, 0.4, 0.15, 0.5, 0.25);
            assert!(rv_l.is_finite() && rv_r.is_finite());
            energy_out += rv_l * rv_l + rv_r * rv_r;
        }
        assert!(energy_out > 1.0);

        // Verify Cross-Instance Shared Spectrum Bus
        let mut test_bands = [0.0_f32; 32];
        test_bands[5] = -12.5;
        publish_shared_spectrum(2, &test_bands);
        let read_back = read_shared_spectrum(2);
        assert!((read_back[5] - (-12.5)).abs() < 1e-4);
    }
}

