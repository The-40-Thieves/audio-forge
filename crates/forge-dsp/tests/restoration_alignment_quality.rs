//! Quality and perceptual invariants for Piece #3 (Restoration & Alignment):
//! `restoration.rs`, `phase_align.rs`, and `dtw_align.rs`.

use std::f64::consts::PI;

use forge_core::AudioBuffer;
use forge_dsp::{align_phase_gcc_phat, clean_vocal_surgical, remove_dc_and_declip};

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    let sum: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    (sum / (a.len().min(b.len()).max(1) as f64)).sqrt()
}

#[test]
fn declip_reconstructs_sine_peak_above_clip_threshold() {
    let sr = 48_000_u32;
    let frames = 4800_usize;
    let true_sig: Vec<f64> = (0..frames)
        .map(|i| 1.25 * (2.0 * PI * 440.0 * (i as f64) / f64::from(sr)).sin())
        .collect();
    let clipped: Vec<f64> = true_sig.iter().map(|&x| x.clamp(-0.99, 0.99)).collect();

    let clipped_err = rmse(&true_sig[200..frames - 200], &clipped[200..frames - 200]);

    let mut buf = AudioBuffer::from_channels(vec![clipped], sr).unwrap();
    remove_dc_and_declip(&mut buf, true);

    let peak = buf.channels[0][200..frames - 200]
        .iter()
        .fold(0.0_f64, |m, &x| m.max(x.abs()));
    let repaired_err = rmse(
        &true_sig[200..frames - 200],
        &buf.channels[0][200..frames - 200],
    );

    assert!(
        peak > 1.05,
        "expected reconstructed arch peak > 1.05 (not re-clipped at 0.995), got {peak:.4}"
    );
    assert!(
        repaired_err < clipped_err * 0.65,
        "expected de-clipper to substantially reduce RMSE: clipped={clipped_err:.5}, repaired={repaired_err:.5}"
    );
}

#[test]
fn debreath_attenuates_unvoiced_breath_consistently_at_48k_and_96k() {
    for sr in [48_000_u32, 96_000_u32] {
        let frames = (sr as usize) / 2; // 0.5 s
        // Low-level ~5.5 kHz breath-like unvoiced signal (RMS ~ 0.018)
        let ch: Vec<f64> = (0..frames)
            .map(|i| 0.025 * (2.0 * PI * 5500.0 * (i as f64) / f64::from(sr)).sin())
            .collect();
        let buf = AudioBuffer::from_channels(vec![ch], sr).unwrap();
        let out = clean_vocal_surgical(&buf, false, false, -12.0).unwrap();

        let rms =
            |s: &[f64]| -> f64 { (s.iter().map(|x| x * x).sum::<f64>() / (s.len() as f64)).sqrt() };
        let slice = frames / 4..3 * frames / 4;
        let in_rms = rms(&buf.channels[0][slice.clone()]);
        let out_rms = rms(&out.channels[0][slice]);
        let atten_db = 20.0 * (out_rms / in_rms).log10();

        assert!(
            atten_db < -6.0,
            "expected breath attenuation < -6 dB at {sr} Hz, got {atten_db:.2} dB"
        );
    }
}

#[test]
fn declick_removes_isolated_mouth_click_cleanly() {
    let sr = 48_000_u32;
    let frames = 4800_usize;
    let clean: Vec<f64> = (0..frames)
        .map(|i| 0.25 * (2.0 * PI * 300.0 * (i as f64) / f64::from(sr)).sin())
        .collect();
    let mut clicked = clean.clone();
    clicked[2000] += 0.55; // sharp mouth click

    let buf = AudioBuffer::from_channels(vec![clicked], sr).unwrap();
    let out = clean_vocal_surgical(&buf, true, false, 0.0).unwrap();

    let residual_at_click = (out.channels[0][2000] - clean[2000]).abs();
    assert!(
        residual_at_click < 0.05,
        "expected click to be suppressed below 0.05 residual, got {residual_at_click:.4}"
    );
    // Unaffected region stays exact
    let far_err = rmse(&clean[3000..4000], &out.channels[0][3000..4000]);
    assert!(
        far_err < 1e-12,
        "clean region should be untouched, got RMSE {far_err:.3e}"
    );
}

#[test]
fn gcc_phat_recovers_fractional_half_sample_delay() {
    let sr = 48_000_u32;
    let frames = 48_000_usize;
    let true_delay = 12.5_f64;

    // Band-limited multi-sine signal evaluated analytically at `t` and `t - true_delay/sr`
    let eval = |sample_pos: f64| -> f64 {
        let t = sample_pos / f64::from(sr);
        let mut s = 0.0;
        for h in 1..=25 {
            let f = 180.0 * (h as f64);
            let phase = (h as f64) * 0.73;
            s += (0.8 / (h as f64).sqrt()) * (2.0 * PI * f * t + phase).sin();
        }
        s * 0.15
    };

    let ref_ch: Vec<f64> = (0..frames).map(|i| eval(i as f64)).collect();
    let tgt_ch: Vec<f64> = (0..frames).map(|i| eval((i as f64) - true_delay)).collect();

    let ref_buf = AudioBuffer::from_channels(vec![ref_ch.clone()], sr).unwrap();
    let tgt_buf = AudioBuffer::from_channels(vec![tgt_ch], sr).unwrap();

    let (aligned, res) = align_phase_gcc_phat(&ref_buf, &tgt_buf, 20.0).unwrap();
    let expected_ms = true_delay * 1000.0 / f64::from(sr);

    assert!(
        (res.delay_ms - expected_ms).abs() < 0.005,
        "expected sub-sample delay near {expected_ms:.4} ms, got {:.4} ms ({res:?})",
        res.delay_ms
    );
    assert!(!res.invert_polarity);

    // Aligned output should closely match reference in the interior (much better than integer lag)
    let interior = 1000..frames - 1000;
    let aligned_err = rmse(&ref_ch[interior.clone()], &aligned.channels[0][interior]);
    assert!(
        aligned_err < 0.015,
        "expected sub-sample aligned RMSE < 0.015, got {aligned_err:.5}"
    );
}
