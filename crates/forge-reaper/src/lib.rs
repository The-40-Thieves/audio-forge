//! `forge-reaper` — Native REAPER 7.82 `.RPP` Multitrack Project Builder, JSFX Installer,
//! Live UDP OSC (`127.0.0.1:9000`) Bridge, and ISO 3901 ISRC / ID3v2.3 Metadata Engine.

use std::net::UdpSocket;
use std::path::{Path, PathBuf};
use std::process::Command;

use forge_core::{ForgeError, Result};
use forge_dsp::StretchMarker;
use forge_vocal::{compile_reaper_take_pitch_envelope, PitchFrame};
use rosc::{encoder, OscMessage, OscPacket, OscType};
use serde::{Deserialize, Serialize};

/// Embedded JSFX plugin source files for zero-dependency 1-command installation into REAPER.
pub const JSFX_OPENSOOTHE: &str = include_str!("../../../reaper-jsfx/Forge_OpenSoothe_DeFizz.jsfx");
pub const JSFX_OPENSPACER: &str = include_str!("../../../reaper-jsfx/Forge_OpenSpacer_Unmask.jsfx");
pub const JSFX_SUBBASS: &str = include_str!("../../../reaper-jsfx/Forge_SubBass_Reinforce.jsfx");
pub const JSFX_VOCAL_SURGICAL: &str =
    include_str!("../../../reaper-jsfx/Forge_Vocal_Surgical.jsfx");

/// Installs all 4 ForgeAudio JSFX plugins into REAPER's `Effects/ForgeAudio` directory.
pub fn install_jsfx_plugins(custom_reaper_dir: Option<&Path>) -> Result<Vec<PathBuf>> {
    let base_dir = if let Some(d) = custom_reaper_dir {
        d.to_path_buf()
    } else if let Ok(appdata) = std::env::var("APPDATA") {
        PathBuf::from(appdata)
            .join("REAPER")
            .join("Effects")
            .join("ForgeAudio")
    } else {
        PathBuf::from("reaper-effects-installed")
    };

    std::fs::create_dir_all(&base_dir)?;

    let plugins = [
        ("Forge_OpenSoothe_DeFizz.jsfx", JSFX_OPENSOOTHE),
        ("Forge_OpenSpacer_Unmask.jsfx", JSFX_OPENSPACER),
        ("Forge_SubBass_Reinforce.jsfx", JSFX_SUBBASS),
        ("Forge_Vocal_Surgical.jsfx", JSFX_VOCAL_SURGICAL),
    ];

    let mut installed = Vec::with_capacity(plugins.len());
    for (filename, content) in plugins {
        let p = base_dir.join(filename);
        std::fs::write(&p, content)?;
        installed.push(p);
    }
    Ok(installed)
}

/// Validates a 12-character ISO 3901 International Standard Recording Code (`CC-XXX-YY-NNNNN` or `CCXXXYYNNNNN`).
pub fn validate_isrc(raw_isrc: &str) -> Result<String> {
    let cleaned: String = raw_isrc
        .trim()
        .chars()
        .filter(|&c| c != '-')
        .map(|c| c.to_ascii_uppercase())
        .collect();

    if cleaned.len() != 12 {
        return Err(ForgeError::IsrcError(format!(
            "ISRC '{raw_isrc}' must contain exactly 12 alphanumeric characters (got {})",
            cleaned.len()
        )));
    }
    let bytes = cleaned.as_bytes();
    if !bytes[0].is_ascii_alphabetic() || !bytes[1].is_ascii_alphabetic() {
        return Err(ForgeError::IsrcError(
            "First 2 characters of ISRC must be an ISO 3166-1 alpha-2 country code (e.g. US, GB, QM)"
                .to_string(),
        ));
    }
    if !bytes[2..5].iter().all(u8::is_ascii_alphanumeric) {
        return Err(ForgeError::IsrcError(
            "Characters 3-5 of ISRC must be a 3-character alphanumeric registrant code".to_string(),
        ));
    }
    if !bytes[5..12].iter().all(u8::is_ascii_digit) {
        return Err(ForgeError::IsrcError(
            "Last 7 characters of ISRC (YYNNNNN) must be numeric digits".to_string(),
        ));
    }
    Ok(cleaned)
}

/// Release metadata contract for DistroKid / Spotify / Apple Music delivery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseMetadata {
    pub title: String,
    pub artist: String,
    pub album: String,
    pub isrc: String,
    pub bpm: Option<u32>,
    pub key: Option<String>,
    pub year: Option<u32>,
}

/// Writes validated ISO 3901 ISRC (`TSRC`) and ID3v2.3 / RIFF BWF metadata to an audio file via FFmpeg.
pub fn write_distrokid_metadata<P: AsRef<Path>>(
    input_path: P,
    output_path: P,
    meta: &ReleaseMetadata,
) -> Result<String> {
    let canonical_isrc = validate_isrc(&meta.isrc)?;
    let in_ref = input_path.as_ref();
    let out_ref = output_path.as_ref();

    if let Some(parent) = out_ref.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let mut args = vec![
        "-hide_banner".to_string(),
        "-loglevel".to_string(),
        "error".to_string(),
        "-y".to_string(),
        "-i".to_string(),
        in_ref.to_string_lossy().to_string(),
        "-c".to_string(),
        "copy".to_string(),
        "-write_id3v2".to_string(),
        "1".to_string(),
        "-id3v2_version".to_string(),
        "3".to_string(),
        "-metadata".to_string(),
        format!("title={}", meta.title),
        "-metadata".to_string(),
        format!("artist={}", meta.artist),
        "-metadata".to_string(),
        format!("album={}", meta.album),
        "-metadata".to_string(),
        format!("ISRC={canonical_isrc}"),
        "-metadata".to_string(),
        format!("TSRC={canonical_isrc}"),
    ];

    if let Some(bpm) = meta.bpm {
        args.push("-metadata".to_string());
        args.push(format!("TBPM={bpm}"));
    }
    if let Some(ref key) = meta.key {
        args.push("-metadata".to_string());
        args.push(format!("TKEY={key}"));
    }
    if let Some(year) = meta.year {
        args.push("-metadata".to_string());
        args.push(format!("date={year}"));
    }
    args.push(out_ref.to_string_lossy().to_string());

    let output = Command::new("ffmpeg")
        .args(&args)
        .output()
        .map_err(|e| ForgeError::ReaperError(format!("FFmpeg metadata tagger failed: {e}")))?;

    if !output.status.success() {
        return Err(ForgeError::ReaperError(format!(
            "FFmpeg metadata write failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(canonical_isrc)
}

/// Track definition for programmatic REAPER 7.82 `.RPP` project generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReaperTrackSpec {
    pub name: String,
    pub file_path: Option<PathBuf>,
    pub is_midi: bool,
    pub duration_secs: f64,
    /// Linear volume fader (`1.0` = 0.0 dB, `0.707` = -3.0 dB).
    pub volume_lin: f64,
    /// Pan (`-1.0` = hard Left, `0.0` = Center, `+1.0` = hard Right).
    pub pan: f64,
    /// RGB peak color (`0x01BBGGRR`).
    pub color: u32,
    /// Optional JSFX plugin names to insert on this track.
    pub jsfx_inserts: Vec<String>,
    /// Optional sidechain receive from another 0-indexed track (`source_track_idx` → Ch 3/4).
    pub sidechain_from_track: Option<usize>,
    /// Optional stretch markers (from `FastDTW` vocal alignment or tempo-grid warping).
    pub stretch_markers: Vec<StretchMarker>,
    /// Optional vocal pitch correction envelope frames.
    pub pitch_envelope: Vec<PitchFrame>,
}

/// Section marker on the REAPER master timeline (e.g. `[Intro]`, `[Verse 1]`, `[Chorus]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReaperMarker {
    pub id: u32,
    pub time_secs: f64,
    pub name: String,
    pub color: u32,
}

/// Complete REAPER 7.82 `.RPP` multitrack session builder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReaperProjectBuilder {
    pub title: String,
    pub bpm: f64,
    pub sample_rate: u32,
    pub isrc: Option<String>,
    pub markers: Vec<ReaperMarker>,
    pub tracks: Vec<ReaperTrackSpec>,
}

#[inline]
fn sanitize_rpp_string(s: &str) -> String {
    s.chars()
        .filter(|&c| c != '"' && c != '\r' && c != '\n' && c != '\0')
        .collect()
}

impl ReaperProjectBuilder {
    pub fn new(title: &str, bpm: f64, sample_rate: u32) -> Self {
        let clean_bpm = forge_dsp::sanitize_param_f64(bpm, 120.0, 40.0, 260.0);
        Self {
            title: sanitize_rpp_string(title),
            bpm: clean_bpm,
            sample_rate: sample_rate.clamp(8_000, 768_000),
            isrc: None,
            markers: Vec::new(),
            tracks: Vec::new(),
        }
    }

    pub fn add_default_song_structure_markers(&mut self) {
        let clean_bpm = forge_dsp::sanitize_param_f64(self.bpm, 120.0, 40.0, 260.0);
        let bar_secs = (60.0 / clean_bpm) * 4.0;
        let sections = [
            (1, 0.0, "[Intro]", 0x01FF_AA44),
            (2, bar_secs * 4.0, "[Verse 1]", 0x0144_CC88),
            (3, bar_secs * 12.0, "[Pre-Chorus]", 0x0144_AACC),
            (4, bar_secs * 16.0, "[Chorus]", 0x0144_66FF),
            (5, bar_secs * 24.0, "[Verse 2]", 0x0144_CC88),
            (6, bar_secs * 32.0, "[Chorus 2]", 0x0144_66FF),
            (7, bar_secs * 40.0, "[Bridge / Solo]", 0x01CC_66FF),
            (8, bar_secs * 48.0, "[Final Chorus]", 0x0144_66FF),
            (9, bar_secs * 56.0, "[Outro]", 0x01FF_AA44),
        ];
        for (id, t, name, color) in sections {
            self.markers.push(ReaperMarker {
                id,
                time_secs: t,
                name: name.to_string(),
                color,
            });
        }
    }

    /// Compiles the complete REAPER 7.82 `.RPP` project text.
    pub fn build_rpp_string(&self) -> String {
        let clean_bpm = forge_dsp::sanitize_param_f64(self.bpm, 120.0, 40.0, 260.0);
        let clean_sr = self.sample_rate.clamp(8_000, 768_000);
        let clean_title = sanitize_rpp_string(&self.title);
        let mut rpp = String::with_capacity(16384);
        rpp.push_str(&format!(
            "<REAPER_PROJECT 0.1 \"7.82/win64\" 1761280000\n  RIPPLE 0\n  GROUPOVERRIDE 0 0 0\n  AUTOXFADE 129\n  ENVATTACH 3\n  SAMPLERATE {} 0 0\n  TEMPO {:.4} 4 4\n",
            clean_sr, clean_bpm
        ));

        if let Some(ref isrc) = self.isrc {
            let safe_isrc = sanitize_rpp_string(isrc);
            rpp.push_str("  <RENDER_METADATA\n");
            rpp.push_str(&format!("    TAG \"ISRC\" \"{safe_isrc}\"\n"));
            rpp.push_str(&format!("    TAG \"ID3:TSRC\" \"{safe_isrc}\"\n"));
            rpp.push_str(&format!("    TAG \"ID3:TIT2\" \"{clean_title}\"\n"));
            rpp.push_str("  >\n");
        }

        for m in &self.markers {
            let safe_t = forge_dsp::sanitize_param_f64(m.time_secs, 0.0, 0.0, 86_400.0);
            let safe_name = sanitize_rpp_string(&m.name);
            rpp.push_str(&format!(
                "  MARKER {} {:.4} \"{}\" 0 {} 1 R\n",
                m.id, safe_t, safe_name, m.color
            ));
        }

        // Master Bus FX Chain (OpenSoothe DeFizz + SubBass Mono-Maker)
        rpp.push_str("  <MASTERFXLIST\n    <JS ForgeAudio/Forge_OpenSoothe_DeFizz.jsfx \"\"\n      3.5 9.0 1.25 1.0 160 0\n    >\n  >\n");

        for (idx, tr) in self.tracks.iter().enumerate() {
            let nchan = if tr.sidechain_from_track.is_some() {
                4
            } else {
                2
            };
            let safe_tr_name = sanitize_rpp_string(&tr.name);
            let safe_vol = forge_dsp::sanitize_param_f64(tr.volume_lin, 1.0, 0.0, 4.0);
            let safe_pan = forge_dsp::sanitize_param_f64(tr.pan, 0.0, -1.0, 1.0);
            rpp.push_str(&format!(
                "  <TRACK {{FORGE-TRACK-{:04}}}\n    NAME \"{}\"\n    PEAKCOL {}\n    VOLPAN {:.4} {:.4} -1 -1 1\n    MUTESOLO 0 0 0\n    IPHASE 0\n    NCHAN {}\n",
                idx + 1,
                safe_tr_name,
                tr.color,
                safe_vol,
                safe_pan,
                nchan
            ));

            // Optional sidechain receive onto channels 3/4 (1024 = destination ch 3/4)
            if let Some(src_idx) = tr.sidechain_from_track {
                rpp.push_str(&format!(
                    "    AUXRECV {src_idx} 0 1.0 0.0 0 0 0 0 1024 -1:U 0 -1 ''\n"
                ));
            }

            if !tr.jsfx_inserts.is_empty() {
                rpp.push_str("    <FXCHAIN\n      SHOW 0\n      LASTSEL 0\n      DOCKED 0\n");
                for jsfx in &tr.jsfx_inserts {
                    let safe_jsfx = sanitize_rpp_string(jsfx);
                    rpp.push_str(&format!("      <JS ForgeAudio/{safe_jsfx} \"\"\n      >\n"));
                }
                rpp.push_str("    >\n");
            }

            if let Some(ref path) = tr.file_path {
                let norm_path = sanitize_rpp_string(&path.to_string_lossy().replace('\\', "/"));
                let src_type = if tr.is_midi { "MIDI" } else { "WAVE" };
                let safe_dur = forge_dsp::sanitize_param_f64(tr.duration_secs, 1.0, 1.0, 86_400.0);
                // PLAYRATE 1 1 0 229376 0 -> 229376 selects elastique 3.3.3 Pro pitch-preserving engine
                rpp.push_str(&format!(
                    "    <ITEM\n      POSITION 0.0000\n      LENGTH {:.4}\n      LOOP 0\n      ALLTAKES 0\n      FADEIN 1 0.005 0 1 0 0 0\n      FADEOUT 1 0.010 0 1 0 0 0\n      NAME \"{}\"\n      VOLPAN 1.0 0.0 1.0 -1.0\n      PLAYRATE 1.0 1 0.0 229376 0 0.0025\n",
                    safe_dur,
                    safe_tr_name
                ));

                if !tr.stretch_markers.is_empty() {
                    rpp.push_str("      SM ");
                    for sm in &tr.stretch_markers {
                        rpp.push_str(&format!(
                            "{:.5} {:.5} 0.0 + ",
                            sm.timeline_pos_secs, sm.source_pos_secs
                        ));
                    }
                    rpp.push('\n');
                }

                if !tr.pitch_envelope.is_empty() {
                    rpp.push_str(&compile_reaper_take_pitch_envelope(&tr.pitch_envelope));
                }

                rpp.push_str(&format!(
                    "      <SOURCE {src_type}\n        FILE \"{norm_path}\"\n      >\n    >\n"
                ));
            }

            rpp.push_str("  >\n");
        }

        rpp.push_str(">\n");
        rpp
    }

    /// Writes the `.RPP` project to disk.
    pub fn save_rpp<P: AsRef<Path>>(&self, path: P) -> Result<PathBuf> {
        let p = path.as_ref();
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(p, self.build_rpp_string())?;
        Ok(p.to_path_buf())
    }
}

/// Sends a live UDP Open Sound Control (OSC) packet to REAPER (`127.0.0.1:9000` by default).
pub fn send_reaper_osc(host_port: &str, address: &str, value: Option<f32>) -> Result<()> {
    let socket = UdpSocket::bind("127.0.0.1:0")
        .map_err(|e| ForgeError::ReaperError(format!("Failed to bind UDP OSC socket: {e}")))?;
    let args = match value {
        Some(v) => vec![OscType::Float(v)],
        None => Vec::new(),
    };
    let packet = OscPacket::Message(OscMessage {
        addr: address.to_string(),
        args,
    });
    let bytes = encoder::encode(&packet)
        .map_err(|e| ForgeError::ReaperError(format!("OSC encoding error: {e}")))?;
    socket
        .send_to(&bytes, host_port)
        .map_err(|e| ForgeError::ReaperError(format!("OSC send failed to {host_port}: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_iso3901_isrc_validation() {
        let valid = validate_isrc("US-S1Z-26-00001").expect("valid ISRC");
        assert_eq!(valid, "USS1Z2600001");
        assert!(validate_isrc("INVALID123").is_err());
    }

    #[test]
    fn test_reaper_rpp_builder_emits_sidechain_and_jsfx() {
        let mut rpp = ReaperProjectBuilder::new("TestSession", 124.0, 48_000);
        rpp.isrc = Some("USS1Z2600001".to_string());
        rpp.add_default_song_structure_markers();
        rpp.tracks.push(ReaperTrackSpec {
            name: "Lead Vox".to_string(),
            file_path: None,
            is_midi: false,
            duration_secs: 10.0,
            volume_lin: 1.0,
            pan: 0.0,
            color: 0x0155_88FF,
            jsfx_inserts: vec!["Forge_OpenSoothe_DeFizz.jsfx".to_string()],
            sidechain_from_track: Some(0),
            stretch_markers: Vec::new(),
            pitch_envelope: Vec::new(),
        });
        let text = rpp.build_rpp_string();
        assert!(text.contains("<REAPER_PROJECT 0.1 \"7.82/win64\""));
        assert!(text.contains("TAG \"ISRC\" \"USS1Z2600001\""));
        assert!(text.contains("AUXRECV 0 0 1.0 0.0 0 0 0 0 1024"));
        assert!(text.contains("Forge_OpenSoothe_DeFizz.jsfx"));
    }

    proptest! {
        #[test]
        fn prop_iso3901_isrc_validator_bijection_and_rejection(
            cc in "[A-Za-z]{2}",
            reg in "[A-Za-z0-9]{3}",
            yr in "[0-9]{2}",
            desig in "[0-9]{5}",
            arb in ".*",
        ) {
            let formatted = format!("{cc}-{reg}-{yr}-{desig}");
            let clean = validate_isrc(&formatted).unwrap();
            prop_assert_eq!(clean.len(), 12);
            prop_assert!(clean.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit()));

            // Arbitrary string: if validate_isrc succeeds, it must satisfy ISO 3901 invariants
            if let Ok(ok_code) = validate_isrc(&arb) {
                prop_assert_eq!(ok_code.len(), 12);
                prop_assert!(ok_code[0..2].chars().all(|c| c.is_ascii_uppercase()));
                prop_assert!(ok_code[2..5].chars().all(|c| c.is_ascii_alphanumeric()));
                prop_assert!(ok_code[5..12].chars().all(|c| c.is_ascii_digit()));
            }
        }

        #[test]
        fn prop_reaper_rpp_builder_injection_immunity_and_structural_validity(
            title in ".*",
            track_name in ".*",
            bpm in prop::num::f64::ANY,
            sr in 0u32..1_000_000,
            vol in prop::num::f64::ANY,
            pan in prop::num::f64::ANY,
            dur in prop::num::f64::ANY,
        ) {
            let mut rpp = ReaperProjectBuilder::new(&title, bpm, sr);
            rpp.isrc = Some(title.clone());
            rpp.add_default_song_structure_markers();
            rpp.tracks.push(ReaperTrackSpec {
                name: track_name,
                file_path: Some(PathBuf::from("C:\\temp\\weird\"file\nname.wav")),
                is_midi: false,
                duration_secs: dur,
                volume_lin: vol,
                pan,
                color: 0x0155_88FF,
                jsfx_inserts: vec!["Forge\"Inject\n.jsfx".to_string()],
                sidechain_from_track: Some(0),
                stretch_markers: vec![StretchMarker {
                    timeline_pos_secs: 1.0,
                    source_pos_secs: 1.0,
                }],
                pitch_envelope: Vec::new(),
            });
            let out = rpp.build_rpp_string();

            // Must never emit NaN or Inf into the .RPP file
            prop_assert!(!out.contains("NaN"));
            prop_assert!(!out.contains("inf"));
            prop_assert!(!out.contains("Inf"));

            // Every `<` chunk header line must have a matching `>` closing line
            let open_chunks = out.lines().filter(|l| l.trim_start().starts_with('<')).count();
            let close_chunks = out.lines().filter(|l| l.trim() == ">").count();
            prop_assert_eq!(open_chunks, close_chunks);
        }
    }
}
