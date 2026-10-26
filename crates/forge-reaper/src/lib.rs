//! `forge-reaper` — Native REAPER 7.82 `.RPP` Multitrack Project Builder, JSFX Installer,
//! Live UDP OSC (`127.0.0.1:9000`) Bridge, and ISO 3901 ISRC / ID3v2.3 Metadata Engine.

use std::ffi::OsString;
use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr, ToSocketAddrs, UdpSocket};
use std::path::{Path, PathBuf};
use std::process::Command;

use forge_core::{ForgeError, Result};
use forge_dsp::StretchMarker;
use forge_vocal::{PitchFrame, compile_reaper_take_pitch_envelope};
use rosc::{OscMessage, OscPacket, OscType, encoder};
use serde::{Deserialize, Serialize};

/// Embedded JSFX plugin source files for zero-dependency 1-command installation into REAPER.
pub const JSFX_OPENSOOTHE: &str = include_str!("../../../reaper-jsfx/Forge_OpenSoothe_DeFizz.jsfx");
pub const JSFX_OPENSPACER: &str = include_str!("../../../reaper-jsfx/Forge_OpenSpacer_Unmask.jsfx");
pub const JSFX_SUBBASS: &str = include_str!("../../../reaper-jsfx/Forge_SubBass_Reinforce.jsfx");
pub const JSFX_VOCAL_SURGICAL: &str =
    include_str!("../../../reaper-jsfx/Forge_Vocal_Surgical.jsfx");

/// Folder under REAPER's `Effects` directory that the generated `.rpp` FX references
/// (`<JS "ForgeAudio/...">`) point into.
pub const JSFX_SUBDIR: &str = "ForgeAudio";

/// The JSFX folder inside a REAPER resource folder: `<resource>/Effects/ForgeAudio`.
pub fn jsfx_install_dir(resource_dir: &Path) -> PathBuf {
    resource_dir.join("Effects").join(JSFX_SUBDIR)
}

/// The environment values that locate REAPER's resource folder.
#[derive(Debug, Default)]
struct ResourceEnv {
    appdata: Option<OsString>,
    home: Option<OsString>,
    xdg_config_home: Option<OsString>,
}

impl ResourceEnv {
    fn from_process() -> Self {
        Self {
            appdata: std::env::var_os("APPDATA"),
            home: std::env::var_os("HOME"),
            xdg_config_home: std::env::var_os("XDG_CONFIG_HOME"),
        }
    }
}

/// REAPER's default resource folder for `os` (a `std::env::consts::OS` value), or `None`
/// when the variable it is derived from is unset or empty.
fn default_resource_dir(os: &str, env: &ResourceEnv) -> Option<PathBuf> {
    let set = |v: &Option<OsString>| v.clone().filter(|s| !s.is_empty()).map(PathBuf::from);
    match os {
        "windows" => set(&env.appdata).map(|d| d.join("REAPER")),
        "macos" => set(&env.home).map(|h| h.join("Library/Application Support/REAPER")),
        _ => set(&env.xdg_config_home)
            .or_else(|| set(&env.home).map(|h| h.join(".config")))
            .map(|c| c.join("REAPER")),
    }
}

fn find_resource_dir(os: &str, env: &ResourceEnv) -> Result<PathBuf> {
    match default_resource_dir(os, env) {
        Some(dir) if dir.is_dir() => Ok(dir),
        Some(dir) => Err(ForgeError::ReaperError(format!(
            "REAPER resource folder {} does not exist; install REAPER or pass an explicit JSFX directory",
            dir.display()
        ))),
        None => Err(ForgeError::ReaperError(format!(
            "cannot locate REAPER's resource folder on {os}: the environment variable it is derived from is not set; pass an explicit JSFX directory"
        ))),
    }
}

/// Finds this platform's REAPER resource folder (`%APPDATA%\REAPER` on Windows,
/// `~/Library/Application Support/REAPER` on macOS, `$XDG_CONFIG_HOME/REAPER` or
/// `~/.config/REAPER` elsewhere). Fails when it cannot be derived or does not exist.
pub fn reaper_resource_dir() -> Result<PathBuf> {
    find_resource_dir(std::env::consts::OS, &ResourceEnv::from_process())
}

/// Installs all 4 ForgeAudio JSFX plugins into `custom_dir`, or by default into
/// `Effects/ForgeAudio` under [`reaper_resource_dir`], which is where the generated
/// sessions' FX references resolve.
pub fn install_jsfx_plugins(custom_dir: Option<&Path>) -> Result<Vec<PathBuf>> {
    let base_dir = match custom_dir {
        Some(d) => d.to_path_buf(),
        None => jsfx_install_dir(&reaper_resource_dir()?),
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

/// Deletes an intermediate file when dropped, so it is removed on every exit path,
/// including a failed FFmpeg tag.
#[derive(Debug)]
pub struct TempFileGuard {
    path: PathBuf,
}

impl TempFileGuard {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
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

    if let Some(parent) = out_ref.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
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

/// Quotes `s` as one `.rpp` token. The format has no escape sequences: REAPER delimits a
/// string with `"`, `'` or a backtick, whichever the string does not contain, and when it
/// contains all three it turns backticks into `'`. Line breaks and NULs would end the
/// line, so they are dropped.
fn quote_rpp(s: &str) -> String {
    let mut text: String = s
        .chars()
        .filter(|c| !matches!(c, '\r' | '\n' | '\0'))
        .collect();
    let quote = match ['"', '\'', '`'].into_iter().find(|q| !text.contains(*q)) {
        Some(q) => q,
        None => {
            text = text.replace('`', "'");
            '`'
        }
    };
    format!("{quote}{text}{quote}")
}

/// A JSFX reference as REAPER resolves it: relative to its `Effects` folder.
fn jsfx_ref(name: &str) -> String {
    quote_rpp(&format!("{JSFX_SUBDIR}/{name}"))
}

/// The quoted `FILE` path. REAPER resolves a relative path against the `.rpp`'s folder,
/// not the working directory it was given in, so relative paths are made absolute.
fn rpp_media_path(path: &Path) -> String {
    quote_rpp(&absolute_media_path(path).to_string_lossy())
}

/// `path` as it is written to the `.rpp`: unchanged when absolute, otherwise joined onto the
/// working directory.
fn absolute_media_path(path: &Path) -> PathBuf {
    match std::env::current_dir() {
        Ok(cwd) => media_path_under(&cwd, path),
        Err(_) => path.to_path_buf(),
    }
}

/// `path` resolved against `cwd`, so the exact absolute path can be validated and written.
fn media_path_under(cwd: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    }
}

/// Rejects a media path that [`rpp_media_path`] could not write back unchanged.
fn check_media_path(path: &Path) -> Result<()> {
    let unrepresentable = |why: &str| {
        Err(ForgeError::ReaperError(format!(
            "media path {} {why}, so it cannot be referenced from an .rpp",
            path.display()
        )))
    };
    let Some(text) = path.to_str() else {
        return unrepresentable("is not valid UTF-8");
    };
    if text.contains(['\r', '\n', '\0']) {
        return unrepresentable("contains a line break or NUL");
    }
    if ['"', '\'', '`'].iter().all(|q| text.contains(*q)) {
        return unrepresentable("contains all three quote characters");
    }
    Ok(())
}

fn push_fx_chain(rpp: &mut String, jsfx_inserts: &[String]) {
    if jsfx_inserts.is_empty() {
        return;
    }
    rpp.push_str("    <FXCHAIN\n      SHOW 0\n      LASTSEL 0\n      DOCKED 0\n");
    for jsfx in jsfx_inserts {
        rpp.push_str(&format!("      <JS {} \"\"\n      >\n", jsfx_ref(jsfx)));
    }
    rpp.push_str("    >\n");
}

fn push_item(rpp: &mut String, tr: &ReaperTrackSpec, path: &Path, quoted_name: &str) {
    let src_type = if tr.is_midi { "MIDI" } else { "WAVE" };
    let safe_dur = forge_dsp::sanitize_param_f64(tr.duration_secs, 1.0, 1.0, 86_400.0);
    // PLAYRATE 1 1 0 229376 0 -> 229376 selects elastique 3.3.3 Pro pitch-preserving engine
    rpp.push_str(&format!(
        "    <ITEM\n      POSITION 0.0000\n      LENGTH {safe_dur:.4}\n      LOOP 0\n      ALLTAKES 0\n      FADEIN 1 0.005 0 1 0 0 0\n      FADEOUT 1 0.010 0 1 0 0 0\n      NAME {quoted_name}\n      VOLPAN 1.0 0.0 1.0 -1.0\n      PLAYRATE 1.0 1 0.0 229376 0 0.0025\n"
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
        "      <SOURCE {src_type}\n        FILE {}\n      >\n    >\n",
        rpp_media_path(path)
    ));
}

fn push_track(rpp: &mut String, idx: usize, tr: &ReaperTrackSpec) {
    let nchan = if tr.sidechain_from_track.is_some() {
        4
    } else {
        2
    };
    let name = quote_rpp(&tr.name);
    let safe_vol = forge_dsp::sanitize_param_f64(tr.volume_lin, 1.0, 0.0, 4.0);
    let safe_pan = forge_dsp::sanitize_param_f64(tr.pan, 0.0, -1.0, 1.0);
    rpp.push_str(&format!(
        "  <TRACK {{FORGE-TRACK-{:04}}}\n    NAME {}\n    PEAKCOL {}\n    VOLPAN {:.4} {:.4} -1 -1 1\n    MUTESOLO 0 0 0\n    IPHASE 0\n    NCHAN {}\n",
        idx + 1,
        name,
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

    push_fx_chain(rpp, &tr.jsfx_inserts);
    if let Some(ref path) = tr.file_path {
        push_item(rpp, tr, path, &name);
    }
    rpp.push_str("  >\n");
}

impl ReaperProjectBuilder {
    pub fn new(title: &str, bpm: f64, sample_rate: u32) -> Self {
        let clean_bpm = forge_dsp::sanitize_param_f64(bpm, 120.0, 40.0, 260.0);
        Self {
            title: title.to_string(),
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

    fn push_header(&self, rpp: &mut String) {
        let clean_bpm = forge_dsp::sanitize_param_f64(self.bpm, 120.0, 40.0, 260.0);
        let clean_sr = self.sample_rate.clamp(8_000, 768_000);
        rpp.push_str(&format!(
            "<REAPER_PROJECT 0.1 \"7.82/win64\" 1761280000\n  RIPPLE 0\n  GROUPOVERRIDE 0 0 0\n  AUTOXFADE 129\n  ENVATTACH 3\n  SAMPLERATE {} 0 0\n  TEMPO {:.4} 4 4\n",
            clean_sr, clean_bpm
        ));

        if let Some(ref isrc) = self.isrc {
            let safe_isrc = quote_rpp(isrc);
            rpp.push_str("  <RENDER_METADATA\n");
            rpp.push_str(&format!("    TAG \"ISRC\" {safe_isrc}\n"));
            rpp.push_str(&format!("    TAG \"ID3:TSRC\" {safe_isrc}\n"));
            rpp.push_str(&format!(
                "    TAG \"ID3:TIT2\" {}\n",
                quote_rpp(&self.title)
            ));
            rpp.push_str("  >\n");
        }

        for m in &self.markers {
            let safe_t = forge_dsp::sanitize_param_f64(m.time_secs, 0.0, 0.0, 86_400.0);
            rpp.push_str(&format!(
                "  MARKER {} {:.4} {} 0 {} 1 R\n",
                m.id,
                safe_t,
                quote_rpp(&m.name),
                m.color
            ));
        }
    }

    /// Compiles the complete REAPER 7.82 `.RPP` project text.
    pub fn build_rpp_string(&self) -> String {
        let mut rpp = String::with_capacity(16384);
        self.push_header(&mut rpp);

        // Master Bus FX Chain (OpenSoothe DeFizz + SubBass Mono-Maker)
        rpp.push_str(&format!(
            "  <MASTERFXLIST\n    <JS {} \"\"\n      3.5 9.0 1.25 1.0 160 0\n    >\n  >\n",
            jsfx_ref("Forge_OpenSoothe_DeFizz.jsfx")
        ));

        for (idx, tr) in self.tracks.iter().enumerate() {
            push_track(&mut rpp, idx, tr);
        }

        rpp.push_str(">\n");
        rpp
    }

    /// Checks every track's media path as it will be written: resolved against `cwd`, so a
    /// working directory that cannot be represented is caught too. [`Self::save_rpp`] runs
    /// this itself; callers that write through [`Self::write_rpp`] run it first.
    pub fn check_media_paths(&self, cwd: &Path) -> Result<()> {
        for media in self.tracks.iter().filter_map(|tr| tr.file_path.as_deref()) {
            check_media_path(&media_path_under(cwd, media))?;
        }
        Ok(())
    }

    /// Writes the `.RPP` project to disk. Fails, writing nothing, when a track's media path
    /// cannot be represented in the file.
    pub fn save_rpp<P: AsRef<Path>>(&self, path: P) -> Result<PathBuf> {
        self.check_media_paths(&std::env::current_dir()?)?;
        let p = path.as_ref();
        if let Some(parent) = p.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        self.write_rpp(std::fs::File::create(p)?)?;
        Ok(p.to_path_buf())
    }

    /// Writes the `.RPP` project text to `writer` and flushes it. [`Self::save_rpp`] creates
    /// the file and delegates here.
    pub fn write_rpp<W: std::io::Write>(&self, mut writer: W) -> Result<()> {
        writer.write_all(self.build_rpp_string().as_bytes())?;
        writer.flush()?;
        Ok(())
    }
}

/// Rejects an address that is not an OSC address pattern: it must start with `/` and hold
/// only printable ASCII, with no spaces, `#` or `,`.
fn validate_osc_address(address: &str) -> Result<()> {
    let valid = address.len() > 1
        && address.starts_with('/')
        && address
            .bytes()
            .all(|b| b.is_ascii_graphic() && b != b'#' && b != b',');
    if valid {
        Ok(())
    } else {
        Err(ForgeError::ReaperError(format!(
            "OSC address {address:?} must start with '/' and contain only printable ASCII without spaces, '#' or ','"
        )))
    }
}

fn resolve_osc_host(host_port: &str) -> Result<SocketAddr> {
    let invalid = |why: String| {
        ForgeError::ReaperError(format!(
            "OSC host {host_port:?} is not a valid host:port: {why}"
        ))
    };
    host_port
        .to_socket_addrs()
        .map_err(|e| invalid(e.to_string()))?
        .next()
        .ok_or_else(|| invalid("it resolved to no address".to_string()))
}

/// The local address to send from: loopback for a loopback destination, otherwise the
/// unspecified address of the destination's family, so the OS can route the packet.
fn osc_bind_addr(dest: &SocketAddr) -> SocketAddr {
    let ip = match (dest.is_ipv4(), dest.ip().is_loopback()) {
        (true, true) => Ipv4Addr::LOCALHOST.into(),
        (true, false) => Ipv4Addr::UNSPECIFIED.into(),
        (false, true) => Ipv6Addr::LOCALHOST.into(),
        (false, false) => Ipv6Addr::UNSPECIFIED.into(),
    };
    SocketAddr::new(ip, 0)
}

/// Sends a live UDP Open Sound Control (OSC) packet to REAPER (`127.0.0.1:9000` by default).
/// UDP has no acknowledgement, so `Ok` means the packet was sent, not that REAPER received it.
pub fn send_reaper_osc(host_port: &str, address: &str, value: Option<f32>) -> Result<()> {
    validate_osc_address(address)?;
    let dest = resolve_osc_host(host_port)?;
    let socket = UdpSocket::bind(osc_bind_addr(&dest))
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
        .send_to(&bytes, dest)
        .map_err(|e| ForgeError::ReaperError(format!("OSC send failed to {host_port}: {e}")))?;
    Ok(())
}

/// Every bundled JSFX plugin as `(file name, source)`, the set [`install_jsfx_plugins`]
/// writes (a test keeps the two in sync). Callers that must create the files themselves (through a directory handle, for
/// example) can write these entries directly.
pub const JSFX_PLUGINS: [(&str, &str); 4] = [
    ("Forge_OpenSoothe_DeFizz.jsfx", JSFX_OPENSOOTHE),
    ("Forge_OpenSpacer_Unmask.jsfx", JSFX_OPENSPACER),
    ("Forge_SubBass_Reinforce.jsfx", JSFX_SUBBASS),
    ("Forge_Vocal_Surgical.jsfx", JSFX_VOCAL_SURGICAL),
];

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::time::Duration;

    #[test]
    fn test_iso3901_isrc_validation() {
        let valid = validate_isrc("US-S1Z-26-00001").expect("valid ISRC");
        assert_eq!(valid, "USS1Z2600001");
        assert!(validate_isrc("INVALID123").is_err());
    }

    fn track(
        name: &str,
        file_path: Option<PathBuf>,
        is_midi: bool,
        jsfx: &[&str],
    ) -> ReaperTrackSpec {
        ReaperTrackSpec {
            name: name.to_string(),
            file_path,
            is_midi,
            duration_secs: 10.0,
            volume_lin: 1.0,
            pan: 0.0,
            color: 0x0155_88FF,
            jsfx_inserts: jsfx.iter().map(ToString::to_string).collect(),
            sidechain_from_track: None,
            stretch_markers: Vec::new(),
            pitch_envelope: Vec::new(),
        }
    }

    /// The token after `prefix` on an `.rpp` line, unquoted the way REAPER reads it.
    fn rpp_token(line: &str, prefix: &str) -> Option<String> {
        let rest = line.trim_start().strip_prefix(prefix)?.trim_start();
        let quote = rest.chars().next()?;
        if !matches!(quote, '"' | '\'' | '`') {
            return rest.split_whitespace().next().map(str::to_string);
        }
        let body = &rest[1..];
        body.find(quote).map(|end| body[..end].to_string())
    }

    fn tokens(rpp: &str, prefix: &str) -> Vec<String> {
        rpp.lines().filter_map(|l| rpp_token(l, prefix)).collect()
    }

    #[test]
    fn test_reaper_rpp_builder_emits_sidechain_and_jsfx() {
        let mut rpp = ReaperProjectBuilder::new("TestSession", 124.0, 48_000);
        rpp.isrc = Some("USS1Z2600001".to_string());
        rpp.add_default_song_structure_markers();
        let mut vox = track("Lead Vox", None, false, &["Forge_OpenSoothe_DeFizz.jsfx"]);
        vox.sidechain_from_track = Some(0);
        rpp.tracks.push(vox);
        let text = rpp.build_rpp_string();
        assert!(text.contains("<REAPER_PROJECT 0.1 \"7.82/win64\""));
        assert!(text.contains("TAG \"ISRC\" \"USS1Z2600001\""));
        assert!(text.contains("AUXRECV 0 0 1.0 0.0 0 0 0 0 1024"));
        assert!(text.contains("Forge_OpenSoothe_DeFizz.jsfx"));
    }

    /// The ticket's round trip: a session written under a relative `output_dir` must
    /// reference media and FX that exist where REAPER looks for them. REAPER resolves a
    /// relative `FILE` against the `.rpp`'s folder and a `<JS>` path against its `Effects`.
    #[test]
    fn session_with_relative_output_dir_resolves_every_file_and_fx() {
        let cwd = std::env::current_dir().expect("cwd");
        let tmp = tempfile::tempdir_in(&cwd).expect("tempdir");
        let rel = tmp.path().strip_prefix(&cwd).expect("under cwd");
        let out = rel.join("session out");
        assert!(out.is_relative());

        let vocal = out.join("stems").join("Lead Vox O'Brien caf\u{e9}.wav");
        let midi = out.join("midi").join("02_Voiced_Chords.mid");
        for f in [&vocal, &midi] {
            std::fs::create_dir_all(f.parent().expect("parent")).expect("mkdir");
            std::fs::write(f, b"stub").expect("write stub");
        }
        let resource = tmp.path().join("REAPER");
        install_jsfx_plugins(Some(&jsfx_install_dir(&resource))).expect("install");

        let mut rpp = ReaperProjectBuilder::new("Session", 120.0, 48_000);
        let fx = ["Forge_Vocal_Surgical.jsfx", "Forge_OpenSoothe_DeFizz.jsfx"];
        rpp.tracks
            .push(track("01 Lead Vocal", Some(vocal), false, &fx));
        rpp.tracks.push(track("02 MIDI", Some(midi), true, &[]));
        let rpp_path = rpp
            .save_rpp(out.join("ForgeAudio_Session.rpp"))
            .expect("save");
        let text = std::fs::read_to_string(&rpp_path).expect("read rpp");

        let project_dir = rpp_path.parent().expect("rpp folder");
        let files = tokens(&text, "FILE ");
        assert_eq!(files.len(), 2, "{text}");
        for f in files {
            let resolved = project_dir.join(&f);
            assert!(
                resolved.is_file(),
                "FILE {f} resolves to missing {}",
                resolved.display()
            );
        }
        let effects = resource.join("Effects");
        let js = tokens(&text, "<JS ");
        assert_eq!(js.len(), 3, "{text}");
        for f in js {
            assert!(
                effects.join(&f).is_file(),
                "JS {f} is not under {}",
                effects.display()
            );
        }
    }

    #[test]
    fn rpp_strings_keep_quotes_by_switching_delimiter() {
        assert_eq!(quote_rpp(r#"say "hi".wav"#), r#"'say "hi".wav'"#);
        assert_eq!(quote_rpp(r#"it's "x""#), r#"`it's "x"`"#);
        assert_eq!(quote_rpp("a\"b'c`d"), "`a\"b'c'd`");
        assert_eq!(quote_rpp("line\r\nbreak\0"), "\"linebreak\"");

        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path().join("Jane \"JD\" Doe").join("M\u{fa}sica");
        let media = dir.join("take 'two'.wav");
        let mut rpp = ReaperProjectBuilder::new("My \"Big\" Song", 120.0, 48_000);
        rpp.isrc = Some("USS1Z2600001".to_string());
        rpp.tracks
            .push(track("Lead \"Vox\"", Some(media.clone()), false, &[]));
        let text = rpp.build_rpp_string();
        assert_eq!(tokens(&text, "FILE "), [media.to_str().expect("utf-8")]);
        assert_eq!(tokens(&text, "NAME ")[0], "Lead \"Vox\"");
        assert!(
            text.contains("TAG \"ID3:TIT2\" 'My \"Big\" Song'"),
            "{text}"
        );
    }

    #[test]
    fn save_rpp_rejects_media_paths_it_cannot_represent() {
        let tmp = tempfile::tempdir().expect("tempdir");
        for bad in ["stems/bad\nname.wav", "stems/cr\rname.wav", "a\"b'c`d.wav"] {
            let mut rpp = ReaperProjectBuilder::new("S", 120.0, 48_000);
            rpp.tracks
                .push(track("T", Some(PathBuf::from(bad)), false, &[]));
            let target = tmp.path().join("s.rpp");
            let err = rpp.save_rpp(&target).expect_err(bad);
            assert!(err.to_string().contains("cannot be referenced"), "{err}");
            assert!(!target.exists(), "{bad:?} still wrote the session");
        }
    }

    #[test]
    fn relative_media_paths_are_validated_with_their_working_directory() {
        // A relative path that is fine on its own must still be rejected when the directory
        // it resolves under cannot be written to an .rpp unchanged.
        let cwd = Path::new("/sessions/session\nfolder");
        let mut rpp = ReaperProjectBuilder::new("S", 120.0, 48_000);
        rpp.tracks
            .push(track("T", Some(PathBuf::from("stem.wav")), false, &[]));
        let err = rpp
            .check_media_paths(cwd)
            .expect_err("newline in the working directory");
        assert!(err.to_string().contains("line break"), "{err}");
        assert!(rpp.check_media_paths(Path::new("/sessions/ok")).is_ok());
        let abs = std::env::current_dir().expect("cwd").join("stem.wav");
        assert_eq!(media_path_under(cwd, &abs), abs);
    }

    #[test]
    fn resource_dir_lookup_fails_instead_of_falling_back() {
        for os in ["windows", "macos", "linux"] {
            let err = find_resource_dir(os, &ResourceEnv::default()).expect_err(os);
            assert!(err.to_string().contains("cannot locate"), "{os}: {err}");
        }
        let tmp = tempfile::tempdir().expect("tempdir");
        let env = ResourceEnv {
            appdata: Some(tmp.path().join("no-such-dir").into()),
            ..ResourceEnv::default()
        };
        let err = find_resource_dir("windows", &env).expect_err("missing folder");
        assert!(err.to_string().contains("does not exist"), "{err}");
    }

    #[test]
    fn resource_dir_follows_each_platform_convention() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        let expected = [
            root.join("AppData").join("REAPER"),
            root.join("home/Library/Application Support/REAPER"),
            root.join("home/.config/REAPER"),
            root.join("xdg/REAPER"),
        ];
        for dir in &expected {
            std::fs::create_dir_all(dir).expect("mkdir");
        }
        let env = |appdata: &str, xdg: &str| ResourceEnv {
            appdata: Some(root.join(appdata).into()),
            home: Some(root.join("home").into()),
            xdg_config_home: Some(if xdg.is_empty() {
                OsString::new()
            } else {
                root.join(xdg).into_os_string()
            }),
        };
        let found = |os: &str, e: &ResourceEnv| find_resource_dir(os, e).expect(os);
        assert_eq!(found("windows", &env("AppData", "")), expected[0]);
        assert_eq!(found("macos", &env("AppData", "")), expected[1]);
        assert_eq!(found("linux", &env("AppData", "")), expected[2]);
        assert_eq!(found("linux", &env("AppData", "xdg")), expected[3]);
        assert_eq!(
            jsfx_install_dir(&expected[0]),
            expected[0].join("Effects").join("ForgeAudio")
        );
    }

    #[test]
    fn temp_file_guard_removes_the_file_on_an_error_return() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("master.pre_tag.wav");
        let failing_tag = |p: &Path| -> Result<()> {
            let guard = TempFileGuard::new(p);
            std::fs::write(guard.path(), b"render")?;
            Err(ForgeError::ReaperError("ffmpeg failed".into()))
        };
        assert!(failing_tag(&path).is_err());
        assert!(!path.exists(), "the intermediate render was left behind");
    }

    #[test]
    fn osc_rejects_addresses_that_are_not_osc_patterns() {
        let rx = UdpSocket::bind("127.0.0.1:0").expect("bind receiver");
        rx.set_read_timeout(Some(Duration::from_secs(5)))
            .expect("timeout");
        let host = rx.local_addr().expect("addr").to_string();
        for bad in [
            "action/40044",
            "",
            "/",
            "/has space",
            "/a,b",
            "/a#b",
            "/caf\u{e9}",
        ] {
            assert!(
                send_reaper_osc(&host, bad, None).is_err(),
                "{bad:?} was sent"
            );
        }
        send_reaper_osc(&host, "/action/40044", Some(1.0)).expect("valid send");
        let mut buf = [0u8; 256];
        let (n, _) = rx.recv_from(&mut buf).expect("packet");
        match rosc::decoder::decode_udp(&buf[..n]).expect("decode").1 {
            OscPacket::Message(m) => assert_eq!(m.addr, "/action/40044"),
            other => panic!("unexpected packet {other:?}"),
        }
    }

    #[test]
    fn osc_sends_from_an_address_that_can_reach_the_destination() {
        let bind = |dest: &str| osc_bind_addr(&dest.parse().expect("socket addr")).to_string();
        assert_eq!(bind("127.0.0.1:9000"), "127.0.0.1:0");
        assert_eq!(bind("192.168.1.20:9000"), "0.0.0.0:0");
        assert_eq!(bind("[::1]:9000"), "[::1]:0");
        assert_eq!(bind("[2001:db8::1]:9000"), "[::]:0");
        assert!(resolve_osc_host("127.0.0.1").is_err());
        assert!(resolve_osc_host("not a host:9000").is_err());
        assert!(resolve_osc_host("127.0.0.1:9000").is_ok());
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

    #[test]
    fn rpp_writer_matches_save_rpp_and_jsfx_list_matches_install() {
        let rpp = ReaperProjectBuilder::new("Writer Session", 96.0, 48_000);
        let mut bytes = Vec::new();
        rpp.write_rpp(&mut bytes).expect("write");
        assert_eq!(bytes, rpp.build_rpp_string().into_bytes());

        let dir = tempfile::tempdir().expect("tempdir");
        let saved = rpp
            .save_rpp(dir.path().join("nested").join("s.rpp"))
            .expect("save");
        assert_eq!(std::fs::read(&saved).expect("read"), bytes);

        let installed = install_jsfx_plugins(Some(&dir.path().join("fx"))).expect("install");
        assert_eq!(installed.len(), JSFX_PLUGINS.len());
        for (path, (name, source)) in installed.iter().zip(JSFX_PLUGINS) {
            assert!(path.ends_with(name));
            assert_eq!(std::fs::read_to_string(path).expect("read"), source);
        }
    }
}
