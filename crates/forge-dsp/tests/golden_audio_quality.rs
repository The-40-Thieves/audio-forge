//! Golden audio quality regression suite for DSP fidelity, resampling, limiting, and spatial QC.

use std::f64::consts::PI;
use std::io::Cursor;
use std::path::Path;

use forge_core::AdmProgrammeKind;
use forge_core::AudioBuffer;
use forge_dsp::{
    ChannelLayout, LoudnessVerificationStatus, RealtimeDensityAndClipper, RealtimeMasterLimiter,
    ReferenceLoudspeakerConfig, SpatialQcOptions, SpatialRenderMode, audit_audio,
    audit_spatial_adm_bwf_reader, audit_spatial_adm_bwf_reader_with_options,
    bs1770_5_channel_weights, bs1770_5_position_weight, compute_bs1770_5_loudness,
    compute_vbap_gains, resample_buffer_sinc,
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

fn push_chna_rec(buf: &mut Vec<u8>, track: u16, uid: &str, track_ref: &str, pack_ref: &str) {
    buf.extend_from_slice(&track.to_le_bytes());
    let mut pad = [0_u8; 38];
    let ub = uid.as_bytes();
    let tb = track_ref.as_bytes();
    let pb = pack_ref.as_bytes();
    pad[..ub.len().min(12)].copy_from_slice(&ub[..ub.len().min(12)]);
    pad[12..12 + tb.len().min(14)].copy_from_slice(&tb[..tb.len().min(14)]);
    pad[26..26 + pb.len().min(11)].copy_from_slice(&pb[..pb.len().min(11)]);
    buf.extend_from_slice(&pad);
}

fn encode_24bit_pcm(channels: &[Vec<f64>]) -> Vec<u8> {
    let n_ch = channels.len();
    let n_frames = channels.first().map_or(0, Vec::len);
    let mut pcm = Vec::with_capacity(n_frames * n_ch * 3);
    for i in 0..n_frames {
        for ch in channels {
            let q = (ch[i] * 8_388_607.0)
                .round()
                .clamp(-8_388_608.0, 8_388_607.0) as i32;
            pcm.extend_from_slice(&q.to_le_bytes()[0..3]);
        }
    }
    pcm
}

fn build_wav_with_adm(
    sample_rate: u32,
    channels: &[Vec<f64>],
    adm_chunks: Option<(&[u8], &[u8])>,
) -> Vec<u8> {
    let n_ch = channels.len() as u16;
    let block_align = n_ch * 3;
    let mut chunks = Vec::new();
    let mut fmt = Vec::with_capacity(16);
    fmt.extend_from_slice(&1_u16.to_le_bytes());
    fmt.extend_from_slice(&n_ch.to_le_bytes());
    fmt.extend_from_slice(&sample_rate.to_le_bytes());
    fmt.extend_from_slice(&(sample_rate * u32::from(block_align)).to_le_bytes());
    fmt.extend_from_slice(&block_align.to_le_bytes());
    fmt.extend_from_slice(&24_u16.to_le_bytes());
    push_wav_chunk(&mut chunks, b"fmt ", &fmt);
    if let Some((chna, axml)) = adm_chunks {
        push_wav_chunk(&mut chunks, b"bext", &[0_u8; 602]);
        push_wav_chunk(&mut chunks, b"chna", chna);
        push_wav_chunk(&mut chunks, b"axml", axml);
    }
    push_wav_chunk(&mut chunks, b"data", &encode_24bit_pcm(channels));
    let mut wav = Vec::with_capacity(12 + chunks.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&((4 + chunks.len()) as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(&chunks);
    wav
}

fn build_permuted_5_1_bed_wav(amp_main: f64, amp_lfe: f64) -> Vec<u8> {
    let t = sine_buffer(1, 48_000, 1.0, 1000.0, amp_main)
        .channels
        .pop()
        .unwrap();
    let lfe = sine_buffer(1, 48_000, 1.0, 80.0, amp_lfe)
        .channels
        .pop()
        .unwrap();
    // Permuted order: [C, LFE, Ls, Rs, L, R]
    let channels = [t.clone(), lfe, t.clone(), t.clone(), t.clone(), t];
    let mut chna = Vec::new();
    chna.extend_from_slice(&6_u16.to_le_bytes());
    chna.extend_from_slice(&6_u16.to_le_bytes());
    let defs = [
        "AT_00010003_01",
        "AT_00010004_01",
        "AT_00010005_01",
        "AT_00010006_01",
        "AT_00010001_01",
        "AT_00010002_01",
    ];
    for (idx, tr) in defs.iter().enumerate() {
        push_chna_rec(
            &mut chna,
            (idx + 1) as u16,
            &format!("ATU_0000000{}", idx + 1),
            tr,
            "AP_00010002",
        );
    }
    let axml = br#"<ebuCoreMain><audioFormatExtended>
        <audioProgramme audioProgrammeID="APR_1"><audioContentIDRef>ACO_1</audioContentIDRef></audioProgramme>
        <audioContent audioContentID="ACO_1"><audioObjectIDRef>AO_BED</audioObjectIDRef></audioContent>
        <audioObject audioObjectID="AO_BED" binauralRenderMode="near">
            <audioTrackUIDRef>ATU_00000001</audioTrackUIDRef><audioTrackUIDRef>ATU_00000002</audioTrackUIDRef>
            <audioTrackUIDRef>ATU_00000003</audioTrackUIDRef><audioTrackUIDRef>ATU_00000004</audioTrackUIDRef>
            <audioTrackUIDRef>ATU_00000005</audioTrackUIDRef><audioTrackUIDRef>ATU_00000006</audioTrackUIDRef>
        </audioObject>
        <audioTrackUID UID="ATU_00000001"><audioTrackFormatIDRef>AT_00010003_01</audioTrackFormatIDRef></audioTrackUID>
        <audioTrackUID UID="ATU_00000002"><audioTrackFormatIDRef>AT_00010004_01</audioTrackFormatIDRef></audioTrackUID>
        <audioTrackUID UID="ATU_00000003"><audioTrackFormatIDRef>AT_00010005_01</audioTrackFormatIDRef></audioTrackUID>
        <audioTrackUID UID="ATU_00000004"><audioTrackFormatIDRef>AT_00010006_01</audioTrackFormatIDRef></audioTrackUID>
        <audioTrackUID UID="ATU_00000005"><audioTrackFormatIDRef>AT_00010001_01</audioTrackFormatIDRef></audioTrackUID>
        <audioTrackUID UID="ATU_00000006"><audioTrackFormatIDRef>AT_00010002_01</audioTrackFormatIDRef></audioTrackUID>
    </audioFormatExtended></ebuCoreMain>"#;
    build_wav_with_adm(48_000, &channels, Some((&chna, axml)))
}

fn build_object_adm_wav(raw_amp: f64, gain_db_second_half: f64) -> Vec<u8> {
    let mono = sine_buffer(1, 48_000, 2.0, 1000.0, raw_amp).channels;
    let mut chna = Vec::new();
    chna.extend_from_slice(&1_u16.to_le_bytes());
    chna.extend_from_slice(&1_u16.to_le_bytes());
    push_chna_rec(
        &mut chna,
        1,
        "ATU_00000010",
        "AT_00031001_01",
        "AP_00031001",
    );
    let axml = format!(
        r#"<ebuCoreMain><audioFormatExtended>
        <audioProgramme audioProgrammeID="APR_1"><audioContentIDRef>ACO_1</audioContentIDRef></audioProgramme>
        <audioContent audioContentID="ACO_1"><audioObjectIDRef>AO_1</audioObjectIDRef></audioContent>
        <audioObject audioObjectID="AO_1" binauralRenderMode="mid"><audioPackFormatIDRef>AP_00031001</audioPackFormatIDRef><audioTrackUIDRef>ATU_00000010</audioTrackUIDRef></audioObject>
        <audioPackFormat audioPackFormatID="AP_00031001" typeDefinition="Objects"><audioChannelFormatIDRef>AC_00031001</audioChannelFormatIDRef></audioPackFormat>
        <audioChannelFormat audioChannelFormatID="AC_00031001" typeDefinition="Objects">
            <audioBlockFormat rtime="00:00:00.00000" duration="00:00:01.00000"><position coordinate="azimuth">0.0</position><position coordinate="elevation">0.0</position><gain gainUnit="linear">1.0</gain></audioBlockFormat>
            <audioBlockFormat rtime="00:00:01.00000" duration="00:00:01.00000"><position coordinate="azimuth">90.0</position><position coordinate="elevation">0.0</position><gain gainUnit="dB">{gain_db_second_half:.2}</gain></audioBlockFormat>
        </audioChannelFormat>
        <audioTrackFormat audioTrackFormatID="AT_00031001_01"><audioStreamFormatIDRef>AS_00031001</audioStreamFormatIDRef></audioTrackFormat>
        <audioStreamFormat audioStreamFormatID="AS_00031001"><audioChannelFormatIDRef>AC_00031001</audioChannelFormatIDRef><audioPackFormatIDRef>AP_00031001</audioPackFormatIDRef></audioStreamFormat>
        <audioTrackUID UID="ATU_00000010"><audioTrackFormatIDRef>AT_00031001_01</audioTrackFormatIDRef><audioPackFormatIDRef>AP_00031001</audioPackFormatIDRef></audioTrackUID>
    </audioFormatExtended></ebuCoreMain>"#
    );
    build_wav_with_adm(48_000, &mono, Some((&chna, axml.as_bytes())))
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
    assert_eq!(
        bs1770_5_channel_weights(8),
        vec![1.0, 1.0, 1.0, 0.0, 1.41, 1.41, 1.0, 1.0]
    );
    assert_eq!(
        bs1770_5_channel_weights(12),
        vec![1.0, 1.0, 1.0, 0.0, 1.41, 1.41, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    );
    assert_eq!(bs1770_5_position_weight(135.0, 0.0, false), 1.0);
    assert_eq!(bs1770_5_position_weight(90.0, 30.0, false), 1.0);
    assert_eq!(bs1770_5_position_weight(90.0, 0.0, false), 1.41);
    assert_eq!(bs1770_5_position_weight(0.0, 0.0, true), 0.0);

    let mut lfe_only = AudioBuffer::new(6, 48_000, 48_000).unwrap();
    let amp_lfe = 10.0_f64.powf(-12.0 / 20.0);
    for (i, sample) in lfe_only.channels[3].iter_mut().enumerate() {
        *sample = amp_lfe * (2.0 * PI * 100.0 * (i as f64) / 48_000.0).sin();
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
fn vbap_grid_conserves_constant_power_energy() {
    for az in [-180.0, -135.0, -90.0, -45.0, 0.0, 30.0, 90.0, 110.0, 150.0] {
        for el in [0.0, 15.0, 30.0, 60.0, 90.0] {
            let g = compute_vbap_gains(ReferenceLoudspeakerConfig::Layout7_1_4, az, el);
            assert_eq!(g[3], 0.0, "LFE must never receive VBAP energy");
            let e: f64 = g.iter().map(|v| v * v).sum();
            assert!(
                (e - 1.0).abs() < 1e-6,
                "VBAP power drift at ({az}, {el}): {e}"
            );
        }
    }
}

#[test]
fn spatial_adm_bwf_qc_audits_permuted_beds_and_rendered_objects() {
    let bed_wav = build_permuted_5_1_bed_wav(0.03, 0.5);
    let bed_rep =
        audit_spatial_adm_bwf_reader(&mut Cursor::new(bed_wav), Path::new("bed.wav")).unwrap();
    assert_eq!(bed_rep.programme_kind, AdmProgrammeKind::ChannelBedOnly);
    assert_eq!(
        bed_rep.loudness_status,
        LoudnessVerificationStatus::VerifiedBedDirectSpeakers
    );
    assert_eq!(
        bed_rep.channel_weights,
        vec![1.0, 0.0, 1.41, 1.41, 1.0, 1.0]
    );
    assert!(
        bed_rep.preliminary_qc_passed,
        "permuted bed checks: {:?}",
        bed_rep.checks
    );
    assert!(bed_rep.certification_disclaimer.contains("NOT Dolby Atmos"));

    // Object with quiet raw track (-24 dBFS) boosted +12 dB in block 2 exceeds -18 LKFS only after rendering
    let obj_loud = build_object_adm_wav(0.06, 12.0);
    let obj_rep =
        audit_spatial_adm_bwf_reader(&mut Cursor::new(obj_loud), Path::new("obj_loud.wav"))
            .unwrap();
    assert_eq!(obj_rep.programme_kind, AdmProgrammeKind::ObjectOnly);
    assert_eq!(
        obj_rep.loudness_status,
        LoudnessVerificationStatus::VerifiedRenderedObjects
    );
    assert!(!obj_rep.preliminary_qc_passed);
    assert!(obj_rep.integrated_lkfs.unwrap() > -18.0);
}

#[test]
fn spatial_adm_bwf_qc_marks_unrendered_missing_and_invalid_adm_unverified() {
    let obj_loud = build_object_adm_wav(0.06, 12.0);
    let no_render_opts = SpatialQcOptions {
        render_mode: SpatialRenderMode::NoRenderer,
    };
    let no_rep = audit_spatial_adm_bwf_reader_with_options(
        &mut Cursor::new(obj_loud),
        Path::new("obj.wav"),
        no_render_opts,
    )
    .unwrap();
    assert_eq!(
        no_rep.loudness_status,
        LoudnessVerificationStatus::UnverifiedNoRenderer
    );
    assert!(no_rep.integrated_lkfs.is_none());
    assert!(!no_rep.preliminary_qc_passed);

    let plain_chs = vec![vec![0.02; 48_000]; 6];
    let plain = build_wav_with_adm(48_000, &plain_chs, None);
    let plain_rep =
        audit_spatial_adm_bwf_reader(&mut Cursor::new(plain), Path::new("plain.wav")).unwrap();
    assert_eq!(
        plain_rep.loudness_status,
        LoudnessVerificationStatus::UnverifiedMissingAdm
    );
    assert!(plain_rep.integrated_lkfs.is_none());
    assert!(!plain_rep.preliminary_qc_passed);

    let bad = build_wav_with_adm(
        48_000,
        &plain_chs[..1],
        Some((&[1, 0, 1, 0], b"<ebuCoreMain><broken>")),
    );
    let bad_rep =
        audit_spatial_adm_bwf_reader(&mut Cursor::new(bad), Path::new("bad.wav")).unwrap();
    assert_eq!(
        bad_rep.programme_kind,
        AdmProgrammeKind::UnsupportedOrMalformed
    );
    assert!(!bad_rep.preliminary_qc_passed);
}
