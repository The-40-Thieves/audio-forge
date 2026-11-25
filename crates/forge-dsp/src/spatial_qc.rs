//! Dolby Atmos / Spatial ADM BWF (ITU-R BS.2076 + ITU-R BS.1770-5) delivery quality control.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use forge_core::{AdmBwfInfo, AudioBuffer, Result, parse_adm_bwf_reader};
use serde::{Deserialize, Serialize};

use crate::{ChannelLayout, audit_audio, bs1770_5_channel_weights, compute_bs1770_5_loudness};

/// One spatial Atmos / ADM BWF delivery verification check.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpatialQcCheck {
    pub name: String,
    pub passed: bool,
    pub mandatory: bool,
    pub detail: String,
}

/// Complete ITU-R BS.1770-5 + Dolby Atmos ADM BWF delivery QC report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialQcReport {
    pub layout: ChannelLayout,
    pub channel_weights: Vec<f64>,
    pub sample_rate: u32,
    pub bits_per_sample: u16,
    pub integrated_lkfs: f64,
    pub short_term_max_lkfs: f64,
    pub momentary_max_lkfs: f64,
    pub loudness_range_lu: f64,
    pub true_peak_dbtp: f64,
    pub adm: AdmBwfInfo,
    pub atmos_ready: bool,
    pub checks: Vec<SpatialQcCheck>,
}

fn push_check(
    checks: &mut Vec<SpatialQcCheck>,
    name: &str,
    passed: bool,
    mandatory: bool,
    detail: String,
) {
    checks.push(SpatialQcCheck {
        name: name.to_string(),
        passed,
        mandatory,
        detail,
    });
}

fn evaluate_format_and_loudness_checks(
    checks: &mut Vec<SpatialQcCheck>,
    adm: &AdmBwfInfo,
    integrated_lkfs: f64,
    true_peak_dbtp: f64,
) {
    let sr = adm.sample_rate;
    let bits = adm.bits_per_sample;
    push_check(
        checks,
        "sample_rate_48khz",
        sr == 48_000,
        true,
        format!("Sample rate is {sr} Hz (Dolby Atmos requires 48000 Hz)"),
    );
    push_check(
        checks,
        "bit_depth_24bit",
        bits == 24,
        true,
        format!("Bit depth is {bits}-bit (Dolby Atmos requires 24-bit PCM)"),
    );
    push_check(
        checks,
        "integrated_loudness_max_neg18_lkfs",
        integrated_lkfs <= -18.0 && integrated_lkfs > -70.0,
        true,
        format!(
            "Integrated loudness is {integrated_lkfs:.2} LKFS (must be <= -18.0 LKFS and above silence gate)"
        ),
    );
    push_check(
        checks,
        "true_peak_max_neg1_dbtp",
        true_peak_dbtp <= -1.0,
        true,
        format!("True peak is {true_peak_dbtp:.2} dBTP (ceiling -1.0 dBTP)"),
    );
}

fn evaluate_adm_metadata_checks(checks: &mut Vec<SpatialQcCheck>, adm: &AdmBwfInfo) {
    push_check(
        checks,
        "adm_chunks_present",
        adm.has_axml && adm.has_chna,
        true,
        format!("axml={}, chna={}", adm.has_axml, adm.has_chna),
    );
    push_check(
        checks,
        "adm_structure_valid",
        adm.adm_programmes >= 1 && adm.adm_objects >= 1,
        true,
        format!(
            "programmes={}, objects={}",
            adm.adm_programmes, adm.adm_objects
        ),
    );
    let mode_detail = if adm.binaural_render_modes.is_empty() {
        "No binauralRenderMode tags found (advisory)".to_string()
    } else {
        format!("Modes: {}", adm.binaural_render_modes.join(", "))
    };
    push_check(
        checks,
        "binaural_render_mode_tagged",
        !adm.binaural_render_modes.is_empty(),
        false,
        mode_detail,
    );
}

fn evaluate_atmos_checks(
    adm: &AdmBwfInfo,
    integrated_lkfs: f64,
    true_peak_dbtp: f64,
) -> Vec<SpatialQcCheck> {
    let mut checks = Vec::with_capacity(7);
    evaluate_format_and_loudness_checks(&mut checks, adm, integrated_lkfs, true_peak_dbtp);
    evaluate_adm_metadata_checks(&mut checks, adm);
    checks
}

/// Audits an open WAV/RF64 ADM BWF stream against ITU-R BS.1770-5 and Dolby Atmos delivery criteria.
pub fn audit_spatial_adm_bwf_reader<R: Read + Seek>(
    reader: &mut R,
    name: &Path,
) -> Result<SpatialQcReport> {
    let origin = name.display().to_string();
    reader.seek(SeekFrom::Start(0))?;
    let adm = parse_adm_bwf_reader(reader, &origin)?;
    reader.seek(SeekFrom::Start(0))?;
    let buf = AudioBuffer::load_from_reader(reader, name)?;

    let layout = ChannelLayout::infer(buf.num_channels());
    let channel_weights = bs1770_5_channel_weights(buf.num_channels());
    let (integrated_lkfs, short_term_max_lkfs, momentary_max_lkfs, loudness_range_lu) =
        compute_bs1770_5_loudness(&buf, Some(layout));
    let base_audit = audit_audio(&buf);
    let checks = evaluate_atmos_checks(&adm, integrated_lkfs, base_audit.true_peak_dbtp);
    let atmos_ready = checks.iter().filter(|c| c.mandatory).all(|c| c.passed);

    Ok(SpatialQcReport {
        layout,
        channel_weights,
        sample_rate: adm.sample_rate,
        bits_per_sample: adm.bits_per_sample,
        integrated_lkfs,
        short_term_max_lkfs,
        momentary_max_lkfs,
        loudness_range_lu,
        true_peak_dbtp: base_audit.true_peak_dbtp,
        adm,
        atmos_ready,
        checks,
    })
}

/// Audits an ADM BWF `.wav` file on disk for BS.1770-5 loudness and Dolby Atmos compliance.
pub fn audit_spatial_adm_bwf<P: AsRef<Path>>(path: P) -> Result<SpatialQcReport> {
    let p = path.as_ref();
    let mut file = std::fs::File::open(p).map_err(|e| forge_core::ForgeError::AudioIo {
        path: p.display().to_string(),
        message: e.to_string(),
    })?;
    audit_spatial_adm_bwf_reader(&mut file, p)
}
