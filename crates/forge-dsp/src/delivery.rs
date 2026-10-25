//! Delivery pipeline: repair, de-fizz, master and verify a mix against a distribution target.
//!
//! This is the shared engine behind `forge deliver` and the `forge_deliver` MCP tool. Every run
//! audits the input and the result with the BS.1770-4 / EBU R128 meter and returns explicit
//! pass/fail checks, so an agent (or a person) never has to guess whether a file is ready.
//!
//! Platform loudness values are the playback-normalization reference levels the services
//! publish (integrated LUFS and a recommended true-peak ceiling). They change over time, so
//! check the platform's own documentation before relying on them for a release.

use forge_core::{AudioBuffer, AuditReport, ForgeError, Result};
use serde::{Deserialize, Serialize};

use crate::elite_dsp::{GenreTonalTarget, compute_genre_tonal_balance_eq};
use crate::loudness_limiter::{audit_audio, normalize_and_limit};
use crate::restoration::{
    RestorationConfig, apply_18_band_eq, clean_vocal_surgical, compute_reference_match_eq,
    run_full_restoration_chain,
};
use crate::spectral_defizz::{DeFizzConfig, apply_spectral_defizz};

/// Integrated loudness may land within this many LU of the target and still pass. The limiter
/// can only reduce level, so dense material often lands slightly under the target.
pub const LOUDNESS_TOLERANCE_LU: f64 = 1.0;

/// Lowest sample rate accepted as a delivery master.
pub const MIN_DELIVERY_SAMPLE_RATE: u32 = 44_100;

/// A distribution target: where the master is going and the loudness it should meet.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryTarget {
    Spotify,
    AppleMusic,
    YouTube,
    AmazonMusic,
    Tidal,
    Deezer,
    /// Podcasts (Apple Podcasts / Spotify spoken-word guidance).
    Podcast,
    /// EBU R128 broadcast (Europe).
    EbuR128,
    /// ATSC A/85 broadcast (US).
    AtscA85,
    /// Any explicit loudness and true-peak ceiling.
    Custom {
        lufs: f64,
        ceiling_dbtp: f64,
    },
}

impl DeliveryTarget {
    /// Every named target, for listing in CLIs and tool schemas.
    pub const NAMED: [DeliveryTarget; 9] = [
        DeliveryTarget::Spotify,
        DeliveryTarget::AppleMusic,
        DeliveryTarget::YouTube,
        DeliveryTarget::AmazonMusic,
        DeliveryTarget::Tidal,
        DeliveryTarget::Deezer,
        DeliveryTarget::Podcast,
        DeliveryTarget::EbuR128,
        DeliveryTarget::AtscA85,
    ];

    /// Integrated loudness target in LUFS.
    pub fn loudness_lufs(self) -> f64 {
        match self {
            Self::Spotify | Self::YouTube | Self::AmazonMusic | Self::Tidal => -14.0,
            Self::Deezer => -15.0,
            Self::AppleMusic | Self::Podcast => -16.0,
            Self::EbuR128 => -23.0,
            Self::AtscA85 => -24.0,
            Self::Custom { lufs, .. } => lufs,
        }
    }

    /// Maximum true peak in dBTP.
    pub fn ceiling_dbtp(self) -> f64 {
        match self {
            Self::AtscA85 => -2.0,
            Self::Custom { ceiling_dbtp, .. } => ceiling_dbtp,
            _ => -1.0,
        }
    }

    /// Stable machine name (`spotify`, `apple_music`, ..., `custom`).
    pub fn name(self) -> &'static str {
        match self {
            Self::Spotify => "spotify",
            Self::AppleMusic => "apple_music",
            Self::YouTube => "youtube",
            Self::AmazonMusic => "amazon_music",
            Self::Tidal => "tidal",
            Self::Deezer => "deezer",
            Self::Podcast => "podcast",
            Self::EbuR128 => "ebu_r128",
            Self::AtscA85 => "atsc_a85",
            Self::Custom { .. } => "custom",
        }
    }

    /// Parses a named target. Case, spaces, `-` and `_` are ignored, so `"Apple Music"`,
    /// `"apple-music"` and `"apple_music"` are all accepted.
    pub fn parse(raw: &str) -> Result<Self> {
        let key: String = raw
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .map(|c| c.to_ascii_lowercase())
            .collect();
        Self::NAMED
            .into_iter()
            .find(|t| t.name().replace('_', "") == key)
            .ok_or_else(|| {
                let names: Vec<&str> = Self::NAMED.iter().map(|t| t.name()).collect();
                ForgeError::InvalidParameter(format!(
                    "Unknown delivery target '{raw}'. Expected one of: {}",
                    names.join(", ")
                ))
            })
    }

    /// Builds a custom target, validating the ranges the mastering stage accepts.
    pub fn custom(lufs: f64, ceiling_dbtp: f64) -> Result<Self> {
        if !lufs.is_finite() || !(-40.0..=-4.0).contains(&lufs) {
            return Err(ForgeError::InvalidParameter(format!(
                "Custom target loudness ({lufs}) must be between -40.0 and -4.0 LUFS"
            )));
        }
        if !ceiling_dbtp.is_finite() || !(-24.0..=0.0).contains(&ceiling_dbtp) {
            return Err(ForgeError::InvalidParameter(format!(
                "Custom true-peak ceiling ({ceiling_dbtp}) must be between -24.0 and 0.0 dBTP"
            )));
        }
        Ok(Self::Custom { lufs, ceiling_dbtp })
    }
}

/// Optional tonal correction applied before mastering.
#[derive(Debug, Clone)]
pub enum ToneMatch {
    /// Match the 18-band spectrum of a reference track (max +/-4.5 dB per band).
    Reference(AudioBuffer),
    /// Steer towards a genre tonal-balance curve (max +/-3.5 dB per band).
    Genre(GenreTonalTarget),
}

/// What the pipeline should do on the way to the target.
#[derive(Debug, Clone)]
pub struct DeliveryOptions {
    pub target: DeliveryTarget,
    /// DC removal, de-clip and hum notch (`run_full_restoration_chain`).
    pub repair: bool,
    /// OpenSoothe-style dynamic spectral de-fizz for AI-generated or harsh material.
    pub defizz: bool,
    /// Extra vocal clean-up (de-click, de-plosive, breath gate). Use on vocal stems only.
    pub vocal: bool,
    pub tone_match: Option<ToneMatch>,
}

impl DeliveryOptions {
    /// Repair and de-fizz on, vocal clean-up and tone matching off.
    pub fn for_target(target: DeliveryTarget) -> Self {
        Self {
            target,
            repair: true,
            defizz: true,
            vocal: false,
            tone_match: None,
        }
    }
}

/// One objective delivery requirement and whether the result meets it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryCheck {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

/// Everything an agent needs to decide whether the master can ship.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryReport {
    pub target: String,
    pub target_lufs: f64,
    pub ceiling_dbtp: f64,
    /// Processing stages actually applied, in order.
    pub stages: Vec<String>,
    pub before: AuditReport,
    pub after: AuditReport,
    pub checks: Vec<DeliveryCheck>,
    /// `true` when every check passed.
    pub ready: bool,
}

/// Runs the delivery pipeline and returns the master plus a verified report.
pub fn deliver(
    input: &AudioBuffer,
    options: &DeliveryOptions,
) -> Result<(AudioBuffer, DeliveryReport)> {
    let target = options.target;
    let before = audit_audio(input);
    let mut stages = Vec::new();
    let mut work = input.clone();

    if options.repair {
        work = run_full_restoration_chain(&work, &RestorationConfig::default())?;
        stages.push("repair: DC removal, de-clip, hum notch".to_string());
    }
    if options.vocal {
        work = clean_vocal_surgical(&work, true, true, -12.0)?;
        stages.push("vocal: de-click, de-plosive, breath gate".to_string());
    }
    if options.defizz {
        work = apply_spectral_defizz(&work, &DeFizzConfig::default())?;
        stages.push("defizz: dynamic spectral resonance suppression".to_string());
    }
    match &options.tone_match {
        Some(ToneMatch::Reference(reference)) => {
            let curve = compute_reference_match_eq(&work, reference, 4.5);
            apply_18_band_eq(&mut work, &curve);
            stages.push("tone: 18-band reference match EQ".to_string());
        }
        Some(ToneMatch::Genre(genre)) => {
            let curve = compute_genre_tonal_balance_eq(&work, *genre, 3.5);
            apply_18_band_eq(&mut work, &curve);
            stages.push("tone: 18-band genre tonal balance EQ".to_string());
        }
        None => {}
    }

    let mastered = normalize_and_limit(&work, target.loudness_lufs(), target.ceiling_dbtp())?;
    stages.push(format!(
        "master: {:.1} LUFS, {:.1} dBTP true-peak limit",
        target.loudness_lufs(),
        target.ceiling_dbtp()
    ));

    let after = audit_audio(&mastered);
    let checks = delivery_checks(&after, target);
    let ready = checks.iter().all(|c| c.passed);
    let report = DeliveryReport {
        target: target.name().to_string(),
        target_lufs: target.loudness_lufs(),
        ceiling_dbtp: target.ceiling_dbtp(),
        stages,
        before,
        after,
        checks,
        ready,
    };
    Ok((mastered, report))
}

/// Checks an audit against a target's objective delivery requirements.
pub fn delivery_checks(audit: &AuditReport, target: DeliveryTarget) -> Vec<DeliveryCheck> {
    let lufs_delta = audit.integrated_lufs - target.loudness_lufs();
    vec![
        DeliveryCheck {
            name: "integrated_loudness".to_string(),
            passed: lufs_delta.abs() <= LOUDNESS_TOLERANCE_LU,
            detail: format!(
                "{:.2} LUFS (target {:.1} +/- {LOUDNESS_TOLERANCE_LU:.1} LU)",
                audit.integrated_lufs,
                target.loudness_lufs()
            ),
        },
        DeliveryCheck {
            name: "true_peak".to_string(),
            passed: audit.true_peak_dbtp <= target.ceiling_dbtp(),
            detail: format!(
                "{:.2} dBTP (ceiling {:.1} dBTP)",
                audit.true_peak_dbtp,
                target.ceiling_dbtp()
            ),
        },
        DeliveryCheck {
            name: "no_clipping".to_string(),
            passed: audit.clipped_samples == 0,
            detail: format!("{} clipped samples", audit.clipped_samples),
        },
        DeliveryCheck {
            name: "sample_rate".to_string(),
            passed: audit.sample_rate >= MIN_DELIVERY_SAMPLE_RATE,
            detail: format!(
                "{} Hz (minimum {MIN_DELIVERY_SAMPLE_RATE} Hz)",
                audit.sample_rate
            ),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// 10 s stereo mix-like signal: a 110 Hz tone, a 3.2 kHz tone and broadband hash.
    fn test_mix(level: f64, sample_rate: u32) -> AudioBuffer {
        let n = sample_rate as usize * 10;
        let mut seed = 0x2545_f491_u32;
        let ch: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / f64::from(sample_rate);
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                let noise = (f64::from(seed) / f64::from(u32::MAX) - 0.5) * 0.2;
                level
                    * (0.6 * (2.0 * PI * 110.0 * t).sin()
                        + 0.3 * (2.0 * PI * 3200.0 * t).sin()
                        + noise)
            })
            .collect();
        AudioBuffer::from_channels(vec![ch.clone(), ch], sample_rate).expect("valid buffer")
    }

    #[test]
    fn parses_named_targets_loosely() {
        assert_eq!(
            DeliveryTarget::parse("Apple Music").unwrap(),
            DeliveryTarget::AppleMusic
        );
        assert_eq!(
            DeliveryTarget::parse("ebu-r128").unwrap(),
            DeliveryTarget::EbuR128
        );
        assert_eq!(
            DeliveryTarget::parse("SPOTIFY").unwrap(),
            DeliveryTarget::Spotify
        );
        assert!(DeliveryTarget::parse("myspace").is_err());
        for t in DeliveryTarget::NAMED {
            assert_eq!(DeliveryTarget::parse(t.name()).unwrap(), t);
        }
    }

    #[test]
    fn custom_target_validates_ranges() {
        assert!(DeliveryTarget::custom(-14.0, -1.0).is_ok());
        assert!(DeliveryTarget::custom(-2.0, -1.0).is_err());
        assert!(DeliveryTarget::custom(-14.0, 1.0).is_err());
        assert!(DeliveryTarget::custom(f64::NAN, -1.0).is_err());
    }

    #[test]
    fn quiet_mix_is_delivered_ready_for_each_streaming_target() {
        let quiet = test_mix(0.05, 48_000);
        for target in [
            DeliveryTarget::Spotify,
            DeliveryTarget::AppleMusic,
            DeliveryTarget::Deezer,
            DeliveryTarget::EbuR128,
            DeliveryTarget::AtscA85,
        ] {
            let (_, report) = deliver(&quiet, &DeliveryOptions::for_target(target)).unwrap();
            assert!(
                report.ready,
                "{}: not ready: {:#?}",
                target.name(),
                report.checks
            );
            assert!(report.after.true_peak_dbtp <= target.ceiling_dbtp());
            assert!(report.before.integrated_lufs < report.after.integrated_lufs);
        }
    }

    #[test]
    fn hot_mix_is_brought_under_the_ceiling() {
        let hot = test_mix(0.95, 48_000);
        let (_, report) =
            deliver(&hot, &DeliveryOptions::for_target(DeliveryTarget::Spotify)).unwrap();
        assert!(report.before.true_peak_dbtp > -1.0);
        assert!(report.after.true_peak_dbtp <= -1.0, "{:#?}", report.checks);
        assert!(report.ready, "{:#?}", report.checks);
    }

    #[test]
    fn low_sample_rate_is_reported_not_ready() {
        let low = test_mix(0.05, 22_050);
        let (_, report) =
            deliver(&low, &DeliveryOptions::for_target(DeliveryTarget::Spotify)).unwrap();
        let sr = report
            .checks
            .iter()
            .find(|c| c.name == "sample_rate")
            .unwrap();
        assert!(!sr.passed);
        assert!(!report.ready);
    }

    #[test]
    fn stages_reflect_options() {
        let mix = test_mix(0.1, 48_000);
        let mut opts = DeliveryOptions::for_target(DeliveryTarget::Tidal);
        opts.repair = false;
        opts.defizz = false;
        let (_, report) = deliver(&mix, &opts).unwrap();
        assert_eq!(report.stages.len(), 1);
        assert!(report.stages[0].starts_with("master:"));
    }
}
