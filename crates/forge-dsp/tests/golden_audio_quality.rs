//! Golden audio quality regression suite for DSP fidelity, resampling, limiting, and spatial QC.

use std::f64::consts::PI;
use std::io::Cursor;
use std::path::Path;

use forge_core::AudioBuffer;
use forge_dsp::{
    ChannelLayout, RealtimeDensityAndClipper, RealtimeMasterLimiter, audit_audio,
    audit_spatial_adm_bwf_reader, bs1770_5_channel_weights, compute_bs1770_5_loudness,
    resample_buffer_sinc,
};
use rustfft::{FftPlanner, num_complex::Complex};

const FFT_LEN: usize = 4096;

fn sine_buffer(
    num_channels: usize,
    sample_rate: u32,
    secs: f64,
    freq_hz: f64,
    amp: f64,
) -> AudioBuffer {
    let frames = ((sample_rate as f64) * secs).round() as usize;
    let mut buf = AudioBuffer::new(num_channels, frames, sample_rate).unwrap();
    for ch in &mut buf.channels {
        for (i, s) in ch.iter_mut().enumerate() {
            *s = amp * (2.0 * PI * freq_hz * (i as f64) / f64::from(sample_rate)).sin();
        }
    }
    buf
}

fn hann_power_spectrum(samples: &[f64]) -> Vec<f64> {
    assert_eq!(samples.len(), FFT_LEN);
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(FFT_LEN);
    let mut buf: Vec<Complex<f64>> = samples
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let w = 0.5 * (1.0 - (2.0 * PI * (i as f64) / (FFT_LEN as f64)).cos());
            Complex::new(x * w, 0.0)
        })
        .collect();
    fft.process(&mut buf);
    buf.iter()
        .take(FFT_LEN / 2 + 1)
        .map(|c| c.norm_sqr())
        .collect()
}

fn measure_hann_sinad_db(samples: &[f64], sample_rate: f64, target_hz: f64) -> f64 {
    let power = hann_power_spectrum(samples);
    let k0 = (target_hz * (FFT_LEN as f64) / sample_rate).round() as usize;
    let lo = k0.saturating_sub(2).max(3);
    let hi = (k0 + 2).min(FFT_LEN / 2 - 1);
    let mut sig_p = 0.0_f64;
    let mut nd_p = 0.0_f64;
    for (k, &p) in power.iter().enumerate().take(FFT_LEN / 2).skip(3) {
        if k >= lo && k <= hi {
            sig_p += p;
        } else {
            nd_p += p;
        }
    }
    10.0 * (sig_p / nd_p.max(1e-24)).log10()
}

fn measure_alias_suppression_db(samples: &[f64], sample_rate: f64, f0_hz: f64) -> f64 {
    let power = hann_power_spectrum(samples);
    let k1 = (f0_hz * (FFT_LEN as f64) / sample_rate).round() as usize;
    let k2 = 2 * k1;
    let k3 = 3 * k1;
    let fund_p: f64 = power[k1.saturating_sub(2)..=(k1 + 2).min(FFT_LEN / 2 - 1)]
        .iter()
        .sum();
    let mut max_alias_p = 1e-24_f64;
    for (k, &p) in power.iter().enumerate().take(FFT_LEN / 2).skip(3) {
        let near_harm = [k1, k2, k3]
            .iter()
            .any(|&h| h < FFT_LEN / 2 && k.abs_diff(h) <= 3);
        if !near_harm {
            max_alias_p = max_alias_p.max(p);
        }
    }
    10.0 * (fund_p / max_alias_p).log10()
}

fn push_wav_chunk(out: &mut Vec<u8>, id: &[u8; 4], payload: &[u8]) {
    out.extend_from_slice(id);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    if payload.len() % 2 == 1 {
        out.push(0);
    }
}

fn build_adm_wav_bytes(sample_rate: u32, amp: f64, include_adm: bool) -> Vec<u8> {
    let channels: u16 = 6;
    let bits: u16 = 24;
    let block_align = channels * (bits / 8);
    let frames = sample_rate as usize;
    let mut pcm = Vec::with_capacity(frames * usize::from(block_align));
    for i in 0..frames {
        let s = amp * (2.0 * PI * 1000.0 * (i as f64) / f64::from(sample_rate)).sin();
        let q = (s * 8_388_607.0).round().clamp(-8_388_608.0, 8_388_607.0) as i32;
        let bytes = q.to_le_bytes();
        for ch in 0..usize::from(channels) {
            let sample_bytes = if ch < 3 { &bytes[0..3] } else { &[0, 0, 0] };
            pcm.extend_from_slice(sample_bytes);
        }
    }

    let mut chunks = Vec::new();
    let mut fmt = Vec::with_capacity(16);
    fmt.extend_from_slice(&1_u16.to_le_bytes());
    fmt.extend_from_slice(&channels.to_le_bytes());
    fmt.extend_from_slice(&sample_rate.to_le_bytes());
    fmt.extend_from_slice(&(sample_rate * u32::from(block_align)).to_le_bytes());
    fmt.extend_from_slice(&block_align.to_le_bytes());
    fmt.extend_from_slice(&bits.to_le_bytes());
    push_wav_chunk(&mut chunks, b"fmt ", &fmt);

    if include_adm {
        push_wav_chunk(&mut chunks, b"bext", &[0_u8; 602]);
        let mut chna = vec![0_u8; 40];
        chna[0..2].copy_from_slice(&1_u16.to_le_bytes());
        chna[2..4].copy_from_slice(&1_u16.to_le_bytes());
        push_wav_chunk(&mut chunks, b"chna", &chna);
        let axml = br#"<?xml version="1.0" encoding="UTF-8"?><ebuCoreMain><coreMetadata><format><audioFormatExtended><audioProgramme audioProgrammeID="APR_1001" audioProgrammeName="AtmosMix"/><audioObject audioObjectID="AO_1001" binauralRenderMode="near"/></audioFormatExtended></format></coreMetadata></ebuCoreMain>"#;
        push_wav_chunk(&mut chunks, b"axml", axml);
    }
    push_wav_chunk(&mut chunks, b"data", &pcm);

    let mut wav = Vec::with_capacity(12 + chunks.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&((4 + chunks.len()) as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(&chunks);
    wav
}

#[test]
fn density_clipper_bypass_sinad_and_4x_adaa_alias_rejection() {
    let sr = 48_000.0;
    let f_clean = 64.0 * sr / (FFT_LEN as f64); // 750 Hz coherent
    let mut clean_proc = RealtimeDensityAndClipper::new(sr);
    let mut clean_out = Vec::with_capacity(FFT_LEN);
    for i in 0..(FFT_LEN * 2) {
        let x = (0.25 * (2.0 * PI * f_clean * (i as f64) / sr).sin()) as f32;
        let (yl, _) = clean_proc.process_stereo_sample(x, x, 0.0, 0.0, 0.0);
        if i >= FFT_LEN {
            clean_out.push(f64::from(yl));
        }
    }
    let clean_sinad = measure_hann_sinad_db(&clean_out, sr, f_clean);
    assert!(
        clean_sinad > 80.0,
        "expected bypass SINAD > 80 dB, got {clean_sinad:.2}"
    );

    let f_hf = 640.0 * sr / (FFT_LEN as f64); // 7500 Hz coherent (5th/7th harmonics > Nyquist)
    let ceil_lin = 10.0_f64.powf(-1.0 / 20.0);
    let drive_lin = 10.0_f64.powf(12.0 / 20.0);
    let mut drv_proc = RealtimeDensityAndClipper::new(sr);
    let mut adaa_out = Vec::with_capacity(FFT_LEN);
    let mut naive_out = Vec::with_capacity(FFT_LEN);
    for i in 0..(FFT_LEN * 2) {
        let x = 0.8 * (2.0 * PI * f_hf * (i as f64) / sr).sin();
        let (yl, _) = drv_proc.process_stereo_sample(x as f32, x as f32, 0.0, 12.0, -1.0);
        assert!(yl.is_finite() && f64::from(yl).abs() <= ceil_lin + 1e-4);
        if i >= FFT_LEN {
            adaa_out.push(f64::from(yl));
            naive_out.push((x * drive_lin).tanh());
        }
    }
    let adaa_suppression = measure_alias_suppression_db(&adaa_out, sr, f_hf);
    let naive_suppression = measure_alias_suppression_db(&naive_out, sr, f_hf);
    assert!(
        adaa_suppression > 40.0 && adaa_suppression > naive_suppression + 15.0,
        "expected 4x ADAA alias suppression > 40 dB and +15 dB over naive (adaa={adaa_suppression:.2}, naive={naive_suppression:.2})"
    );
}

#[test]
fn sinc_resampler_passband_flatness_and_sinad_across_44k_48k_96k() {
    let freq_hz = 750.0; // exact coherent bin at both 48 kHz (k=64) and 96 kHz (k=32) for N=4096
    for (src_sr, dst_sr) in [(44_100_u32, 48_000_u32), (48_000_u32, 96_000_u32)] {
        let inp = sine_buffer(1, src_sr, 0.5, freq_hz, 0.5);
        let out = resample_buffer_sinc(&inp, dst_sr).unwrap();
        assert_eq!(out.sample_rate, dst_sr);
        let skip = (f64::from(dst_sr) * 0.05) as usize;
        let steady = &out.channels[0][skip..skip + FFT_LEN];
        let rms_in = (0.5_f64 * 0.5_f64 / 2.0).sqrt();
        let rms_out = (steady.iter().map(|s| s * s).sum::<f64>() / (steady.len() as f64)).sqrt();
        let gain_db = 20.0 * (rms_out / rms_in).log10();
        assert!(
            gain_db.abs() < 0.05,
            "resampler {src_sr}->{dst_sr} passband gain drift {gain_db:.4} dB"
        );
        let sinad = measure_hann_sinad_db(steady, f64::from(dst_sr), freq_hz);
        assert!(
            sinad > 75.0,
            "resampler {src_sr}->{dst_sr} SINAD expected > 75 dB, got {sinad:.2}"
        );
    }
}

#[test]
fn master_limiter_preserves_clean_passband_and_enforces_true_peak_ceiling() {
    let sr = 48_000.0;
    let f_clean = 85.0 * sr / (FFT_LEN as f64); // ~996 Hz coherent
    let mut clean_lim = RealtimeMasterLimiter::new(sr);
    let mut clean_out = Vec::with_capacity(FFT_LEN);
    for i in 0..(FFT_LEN * 2) {
        let x = (0.2 * (2.0 * PI * f_clean * (i as f64) / sr).sin()) as f32;
        let (yl, _) = clean_lim.process_stereo_sample(x, x, 0.0, -1.0, 85.0);
        if i >= FFT_LEN {
            clean_out.push(f64::from(yl));
        }
    }
    let clean_sinad = measure_hann_sinad_db(&clean_out, sr, f_clean);
    assert!(
        clean_sinad > 85.0,
        "expected limiter passband SINAD > 85 dB, got {clean_sinad:.2}"
    );

    let frames = 24_000_usize;
    let mut driven_buf = AudioBuffer::new(2, frames, 48_000).unwrap();
    let mut drv_lim = RealtimeMasterLimiter::new(sr);
    for i in 0..frames {
        let t = (i as f64) / sr;
        let x =
            (0.55 * (2.0 * PI * 1000.0 * t).sin() + 0.30 * (2.0 * PI * 9970.0 * t).sin()) as f32;
        let (yl, yr) = drv_lim.process_stereo_sample(x, x, 8.0, -1.0, 85.0);
        driven_buf.channels[0][i] = f64::from(yl);
        driven_buf.channels[1][i] = f64::from(yr);
    }
    let tp = audit_audio(&driven_buf).true_peak_dbtp;
    assert!(
        tp <= -0.95,
        "expected limiter true peak <= -0.95 dBTP, got {tp:.3}"
    );
}

#[test]
fn bs1770_5_multichannel_excludes_lfe_and_weights_surrounds_plus_1_5_db() {
    assert_eq!(
        bs1770_5_channel_weights(6),
        vec![1.0, 1.0, 1.0, 0.0, 1.41, 1.41]
    );
    let mut lfe_only = AudioBuffer::new(6, 48_000, 48_000).unwrap();
    let amp_lfe = 10.0_f64.powf(-12.0 / 20.0);
    for i in 0..48_000 {
        lfe_only.channels[3][i] = amp_lfe * (2.0 * PI * 100.0 * (i as f64) / 48_000.0).sin();
    }
    let (lfe_lkfs, _, _, _) =
        compute_bs1770_5_loudness(&lfe_only, Some(ChannelLayout::Surround5_1));
    assert!(
        lfe_lkfs <= -70.0,
        "expected LFE-only LKFS <= -70, got {lfe_lkfs:.2}"
    );

    let amp_20 = 10.0_f64.powf(-20.0 / 20.0);
    let tone = sine_buffer(1, 48_000, 1.0, 1000.0, amp_20);
    let mut fl_buf = AudioBuffer::new(6, 48_000, 48_000).unwrap();
    let mut ls_buf = AudioBuffer::new(6, 48_000, 48_000).unwrap();
    fl_buf.channels[0] = tone.channels[0].clone();
    ls_buf.channels[4] = tone.channels[0].clone();

    let (fl_lkfs, _, _, _) = compute_bs1770_5_loudness(&fl_buf, Some(ChannelLayout::Surround5_1));
    let (ls_lkfs, _, _, _) = compute_bs1770_5_loudness(&ls_buf, Some(ChannelLayout::Surround5_1));
    let diff = ls_lkfs - fl_lkfs;
    assert!(
        (diff - 1.492).abs() <= 0.05,
        "expected +1.492 LU surround boost, got {diff:.4}"
    );
}

#[test]
fn spatial_adm_bwf_qc_validates_atmos_pass_and_rejects_loud_or_missing_adm() {
    let pass_wav = build_adm_wav_bytes(48_000, 0.04, true);
    let pass_rep =
        audit_spatial_adm_bwf_reader(&mut Cursor::new(pass_wav), Path::new("pass.wav")).unwrap();
    assert!(
        pass_rep.atmos_ready,
        "expected compliant Atmos mix to pass: {:?}",
        pass_rep.checks
    );
    assert!(
        pass_rep
            .checks
            .iter()
            .filter(|c| c.mandatory)
            .all(|c| c.passed)
    );

    let loud_wav = build_adm_wav_bytes(48_000, 0.5, true);
    let loud_rep =
        audit_spatial_adm_bwf_reader(&mut Cursor::new(loud_wav), Path::new("loud.wav")).unwrap();
    assert!(!loud_rep.atmos_ready);
    assert!(
        !loud_rep
            .checks
            .iter()
            .find(|c| c.name == "integrated_loudness_max_neg18_lkfs")
            .unwrap()
            .passed
    );

    let plain_wav = build_adm_wav_bytes(48_000, 0.04, false);
    let plain_rep =
        audit_spatial_adm_bwf_reader(&mut Cursor::new(plain_wav), Path::new("plain.wav")).unwrap();
    assert!(!plain_rep.atmos_ready);
    assert!(
        !plain_rep
            .checks
            .iter()
            .find(|c| c.name == "adm_chunks_present")
            .unwrap()
            .passed
    );
}
