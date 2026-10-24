//! `forge-dsp` — Studio DSP, Restoration, Spectral Suppression, Unmasking, Alignment & Mastering.

pub mod loudness_limiter;
pub mod phase_align;
pub mod restoration;
pub mod spectral_defizz;
pub mod spectral_unmask;

pub use loudness_limiter::{audit_audio, compute_true_peak, normalize_and_limit, Biquad};
pub use phase_align::{
    align_phase_gcc_phat, align_vocal_double_dtw, PhaseAlignmentResult, StretchMarker,
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
}

