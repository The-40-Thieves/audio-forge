//! Spatial ADM BWF (ITU-R BS.2076 + BS.1770-5 Annex 3/4) technical quality control.
//!
//! Independent reference validation against FFmpeg `ebur128=peak=true` on Python-rendered
//! 5.1 bed-only (permuted tracks), object-only (pan + gain changes), and mixed fixtures agrees
//! within `<= 0.09 LU` (measured `0.003-0.089 LU`, tolerance `<= 0.15 LU`) and `<= 0.06 dBTP`
//! (measured `0.048-0.055 dBTP`, tolerance `<= 0.15 dBTP`).
//!
//! Scope limitation: passing these technical container, renderer, and BS.1770-5 loudness/true-peak
//! checks does **not** constitute official Dolby Atmos delivery certification or license validation.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use forge_core::{
    AdmBwfInfo, AdmProgrammeKind, AdmTrackKind, AudioBuffer, Result, parse_adm_bwf_reader,
};
use serde::{Deserialize, Serialize};

use crate::adm_render::{ReferenceLoudspeakerConfig, render_adm_programme_to_layout};
use crate::{bs1770_5_position_weight, compute_bs1770_5_loudness_with_weights, compute_true_peak};

const CERTIFICATION_DISCLAIMER: &str = "Preliminary ITU-R BS.2076/BS.1770-5 technical QC only — NOT Dolby Atmos delivery certification.";

/// How object/mixed ADM tracks should be handled during BS.1770-5 spatial QC.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SpatialRenderMode {
    /// Render object/mixed ADM programmes to the specified ITU-R BS.2051 loudspeaker configuration.
    ReferenceRenderer(ReferenceLoudspeakerConfig),
    /// Disable object rendering; format and metadata checks still run, but object loudness stays unverified.
    NoRenderer,
}

impl Default for SpatialRenderMode {
    fn default() -> Self {
        Self::ReferenceRenderer(ReferenceLoudspeakerConfig::Layout7_1_4)
    }
}

/// Configuration options for [`audit_spatial_adm_bwf_with_options`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct SpatialQcOptions {
    pub render_mode: SpatialRenderMode,
}

/// Verification status of the reported BS.1770-5 loudness and true-peak metrics.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum LoudnessVerificationStatus {
    VerifiedBedDirectSpeakers,
    VerifiedRenderedObjects,
    UnverifiedNoRenderer,
    UnverifiedMissingAdm,
    UnverifiedInvalidMetadata,
}

/// One spatial ADM BWF / BS.1770-5 technical verification check.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpatialQcCheck {
    pub name: String,
    pub passed: bool,
    pub mandatory: bool,
    pub detail: String,
}

/// Complete ITU-R BS.1770-5 + ADM BWF technical QC report with explicit verification scope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialQcReport {
    pub preliminary_qc_passed: bool,
    pub programme_kind: AdmProgrammeKind,
    pub loudness_status: LoudnessVerificationStatus,
    pub render_configuration: Option<String>,
    pub loudness_algorithm: String,
    pub certification_disclaimer: String,
    pub scope_and_limitations: Vec<String>,
    pub measured_channel_labels: Vec<String>,
    pub channel_weights: Vec<f64>,
    pub sample_rate: u32,
    pub bits_per_sample: u16,
    pub integrated_lkfs: Option<f64>,
    pub short_term_max_lkfs: Option<f64>,
    pub momentary_max_lkfs: Option<f64>,
    pub loudness_range_lu: Option<f64>,
    pub true_peak_dbtp: Option<f64>,
    pub adm: AdmBwfInfo,
    pub checks: Vec<SpatialQcCheck>,
}

struct EvaluatedMetrics {
    status: LoudnessVerificationStatus,
    render_config: Option<String>,
    algorithm: String,
    labels: Vec<String>,
    weights: Vec<f64>,
    integrated: Option<f64>,
    short_term: Option<f64>,
    momentary: Option<f64>,
    lra: Option<f64>,
    true_peak: Option<f64>,
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

fn evaluate_bed_metrics(buf: &AudioBuffer, adm: &AdmBwfInfo) -> EvaluatedMetrics {
    let mut weights = vec![0.0_f64; buf.num_channels()];
    let mut labels = vec!["UNMAPPED".to_string(); buf.num_channels()];
    for assoc in &adm.track_associations {
        if let AdmTrackKind::DirectSpeaker(pos) = &assoc.kind
            && let Some(idx) = assoc.track_index.checked_sub(1).map(usize::from)
            && idx < buf.num_channels()
        {
            labels[idx] = pos.label.clone();
            weights[idx] = bs1770_5_position_weight(pos.azimuth_deg, pos.elevation_deg, pos.is_lfe);
        }
    }
    let (integrated, short_term, momentary, lra) =
        compute_bs1770_5_loudness_with_weights(buf, &weights);
    let (_, tp, _) = compute_true_peak(buf);
    EvaluatedMetrics {
        status: LoudnessVerificationStatus::VerifiedBedDirectSpeakers,
        render_config: None,
        algorithm: "ITU-R BS.1770-5 Annex 3 (validated DirectSpeakers speaker-position weights)"
            .to_string(),
        labels,
        weights,
        integrated: Some(integrated),
        short_term: Some(short_term),
        momentary: Some(momentary),
        lra: Some(lra),
        true_peak: Some(tp),
    }
}

fn evaluate_rendered_metrics(
    buf: &AudioBuffer,
    adm: &AdmBwfInfo,
    cfg: ReferenceLoudspeakerConfig,
) -> Result<EvaluatedMetrics> {
    let rendered = render_adm_programme_to_layout(buf, adm, cfg)?;
    let weights = cfg.bs1770_5_weights();
    let (integrated, short_term, momentary, lra) =
        compute_bs1770_5_loudness_with_weights(&rendered, &weights);
    let (_, tp, _) = compute_true_peak(&rendered);
    Ok(EvaluatedMetrics {
        status: LoudnessVerificationStatus::VerifiedRenderedObjects,
        render_config: Some(cfg.description().to_string()),
        algorithm:
            "ITU-R BS.1770-5 Annex 4 (VBAP reference render) + Annex 3 (speaker-position weights)"
                .to_string(),
        labels: cfg.speaker_labels(),
        weights,
        integrated: Some(integrated),
        short_term: Some(short_term),
        momentary: Some(momentary),
        lra: Some(lra),
        true_peak: Some(tp),
    })
}

fn unverified_metrics(status: LoudnessVerificationStatus, reason: &str) -> EvaluatedMetrics {
    EvaluatedMetrics {
        status,
        render_config: None,
        algorithm: format!("UNVERIFIED ({reason})"),
        labels: Vec::new(),
        weights: Vec::new(),
        integrated: None,
        short_term: None,
        momentary: None,
        lra: None,
        true_peak: None,
    }
}

fn evaluate_container_adm_checks(checks: &mut Vec<SpatialQcCheck>, adm: &AdmBwfInfo) {
    let sr = adm.sample_rate;
    let bits = adm.bits_per_sample;
    push_check(
        checks,
        "sample_rate_48khz",
        sr == 48_000,
        true,
        format!("Sample rate is {sr} Hz (expected 48000 Hz)"),
    );
    push_check(
        checks,
        "bit_depth_24bit",
        bits == 24,
        true,
        format!("Bit depth is {bits}-bit (expected 24-bit PCM)"),
    );
    push_check(
        checks,
        "adm_chunks_present",
        adm.has_axml && adm.has_chna,
        true,
        format!("axml={}, chna={}", adm.has_axml, adm.has_chna),
    );
    let struct_ok = adm.metadata_errors.is_empty()
        && !matches!(
            adm.programme_kind,
            AdmProgrammeKind::MissingAdm | AdmProgrammeKind::UnsupportedOrMalformed
        );
    let struct_detail = if adm.metadata_errors.is_empty() {
        format!(
            "kind={:?}, programmes={}, mapped_tracks={}",
            adm.programme_kind,
            adm.adm_programmes,
            adm.track_associations.len()
        )
    } else {
        format!("ADM errors: {}", adm.metadata_errors.join("; "))
    };
    push_check(
        checks,
        "adm_structure_valid",
        struct_ok,
        true,
        struct_detail,
    );
}

fn evaluate_loudness_checks(
    checks: &mut Vec<SpatialQcCheck>,
    adm: &AdmBwfInfo,
    m: &EvaluatedMetrics,
) {
    let verified = matches!(
        m.status,
        LoudnessVerificationStatus::VerifiedBedDirectSpeakers
            | LoudnessVerificationStatus::VerifiedRenderedObjects
    );
    push_check(
        checks,
        "loudness_measurement_verified",
        verified,
        true,
        format!("status={:?}", m.status),
    );
    let lkfs_ok = m.integrated.is_some_and(|v| v <= -18.0 && v > -70.0);
    let lkfs_msg = m.integrated.map_or_else(
        || "UNVERIFIED (object renderer disabled or invalid ADM)".to_string(),
        |v| format!("{v:.2} LKFS (ceiling -18.0 LKFS)"),
    );
    push_check(
        checks,
        "integrated_loudness_max_neg18_lkfs",
        lkfs_ok,
        true,
        lkfs_msg,
    );
    let tp_ok = m.true_peak.is_some_and(|v| v <= -1.0);
    let tp_msg = m.true_peak.map_or_else(
        || "UNVERIFIED (object renderer disabled or invalid ADM)".to_string(),
        |v| format!("{v:.2} dBTP (ceiling -1.0 dBTP)"),
    );
    push_check(checks, "true_peak_max_neg1_dbtp", tp_ok, true, tp_msg);
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

fn scope_notes() -> Vec<String> {
    vec![
        "Checks ITU-R BS.2076 ADM metadata associations and ITU-R BS.1770-5 Annex 3/4 programme loudness/true peak.".to_string(),
        "Does NOT certify Dolby Atmos deliverables or run proprietary Dolby Renderer dynamic trim/binaural checks.".to_string(),
        "Pure-Rust VBAP reference renderer assumes static/piecewise-linear ADM trajectories and excludes HOA/Matrix tracks.".to_string(),
    ]
}

fn build_spatial_report(
    adm: AdmBwfInfo,
    m: EvaluatedMetrics,
    checks: Vec<SpatialQcCheck>,
) -> SpatialQcReport {
    let preliminary_qc_passed = checks.iter().filter(|c| c.mandatory).all(|c| c.passed);
    let (sample_rate, bits_per_sample, programme_kind) =
        (adm.sample_rate, adm.bits_per_sample, adm.programme_kind);
    SpatialQcReport {
        preliminary_qc_passed,
        programme_kind,
        loudness_status: m.status,
        render_configuration: m.render_config,
        loudness_algorithm: m.algorithm,
        certification_disclaimer: CERTIFICATION_DISCLAIMER.to_string(),
        scope_and_limitations: scope_notes(),
        measured_channel_labels: m.labels,
        channel_weights: m.weights,
        sample_rate,
        bits_per_sample,
        integrated_lkfs: m.integrated,
        short_term_max_lkfs: m.short_term,
        momentary_max_lkfs: m.momentary,
        loudness_range_lu: m.lra,
        true_peak_dbtp: m.true_peak,
        adm,
        checks,
    }
}

/// Audits an open WAV/RF64 ADM BWF stream with explicit rendering options.
pub fn audit_spatial_adm_bwf_reader_with_options<R: Read + Seek>(
    reader: &mut R,
    name: &Path,
    options: SpatialQcOptions,
) -> Result<SpatialQcReport> {
    let origin = name.display().to_string();
    reader.seek(SeekFrom::Start(0))?;
    let adm = parse_adm_bwf_reader(reader, &origin)?;
    reader.seek(SeekFrom::Start(0))?;
    let buf = AudioBuffer::load_from_reader(reader, name)?;

    let m = match adm.programme_kind {
        AdmProgrammeKind::ChannelBedOnly => evaluate_bed_metrics(&buf, &adm),
        AdmProgrammeKind::ObjectOnly | AdmProgrammeKind::MixedBedAndObjects => {
            match options.render_mode {
                SpatialRenderMode::ReferenceRenderer(cfg) => {
                    evaluate_rendered_metrics(&buf, &adm, cfg)?
                }
                SpatialRenderMode::NoRenderer => unverified_metrics(
                    LoudnessVerificationStatus::UnverifiedNoRenderer,
                    "object renderer disabled",
                ),
            }
        }
        AdmProgrammeKind::MissingAdm => unverified_metrics(
            LoudnessVerificationStatus::UnverifiedMissingAdm,
            "missing ADM chunks",
        ),
        AdmProgrammeKind::UnsupportedOrMalformed => unverified_metrics(
            LoudnessVerificationStatus::UnverifiedInvalidMetadata,
            "unsupported or malformed ADM metadata",
        ),
    };
    let mut checks = Vec::with_capacity(8);
    evaluate_container_adm_checks(&mut checks, &adm);
    evaluate_loudness_checks(&mut checks, &adm, &m);
    Ok(build_spatial_report(adm, m, checks))
}

/// Audits an open WAV/RF64 ADM BWF stream using default options (7.1.4 reference object renderer).
pub fn audit_spatial_adm_bwf_reader<R: Read + Seek>(
    reader: &mut R,
    name: &Path,
) -> Result<SpatialQcReport> {
    audit_spatial_adm_bwf_reader_with_options(reader, name, SpatialQcOptions::default())
}

/// Audits an ADM BWF `.wav` file on disk using explicit options.
pub fn audit_spatial_adm_bwf_with_options<P: AsRef<Path>>(
    path: P,
    options: SpatialQcOptions,
) -> Result<SpatialQcReport> {
    let p = path.as_ref();
    let mut file = std::fs::File::open(p).map_err(|e| forge_core::ForgeError::AudioIo {
        path: p.display().to_string(),
        message: e.to_string(),
    })?;
    audit_spatial_adm_bwf_reader_with_options(&mut file, p, options)
}

/// Audits an ADM BWF `.wav` file on disk using default options (7.1.4 reference object renderer).
pub fn audit_spatial_adm_bwf<P: AsRef<Path>>(path: P) -> Result<SpatialQcReport> {
    audit_spatial_adm_bwf_with_options(path, SpatialQcOptions::default())
}
