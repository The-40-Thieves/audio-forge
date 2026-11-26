//! Bounded pure-Rust ITU-R BS.2076 / BS.2094 ADM `axml` XML parser and track resolver.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

const MAX_ADM_ELEMENTS: usize = 131_072;
const MAX_ADM_BLOCKS_PER_CHANNEL: usize = 65_536;
const MAX_XML_DEPTH: usize = 64;

/// ITU-R BS.2076 ADM `typeDefinition` classification.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AdmTypeDefinition {
    DirectSpeakers,
    Objects,
    Matrix,
    Hoa,
    Binaural,
    Unknown(String),
}

/// Speaker label and spherical coordinates (degrees) for a `DirectSpeakers` channel.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdmSpeakerPosition {
    pub label: String,
    pub azimuth_deg: f64,
    pub elevation_deg: f64,
    pub is_lfe: bool,
}

/// One time-varying spatial metadata block inside an `Objects` `audioChannelFormat`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdmObjectBlock {
    pub start_secs: f64,
    pub duration_secs: Option<f64>,
    pub azimuth_deg: f64,
    pub elevation_deg: f64,
    pub distance: f64,
    pub gain: f64,
    pub jump_position: bool,
    pub interpolation_secs: Option<f64>,
}

/// Resolved audio semantics for one WAV/RF64 track.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AdmTrackKind {
    DirectSpeaker(AdmSpeakerPosition),
    Object {
        object_gain: f64,
        blocks: Vec<AdmObjectBlock>,
    },
    Unsupported(AdmTypeDefinition),
}

/// Validated `chna` + `axml` association binding one WAV track (`1`-based) to ADM metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdmTrackAssociation {
    pub track_index: u16,
    pub uid: String,
    pub object_id: Option<String>,
    pub binaural_render_mode: Option<String>,
    pub kind: AdmTrackKind,
}

/// High-level programme taxonomy derived strictly from ADM `axml` + `chna` metadata.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AdmProgrammeKind {
    ChannelBedOnly,
    ObjectOnly,
    MixedBedAndObjects,
    MissingAdm,
    UnsupportedOrMalformed,
}

/// Raw 40-byte EBU Tech 3364 / ITU-R BS.2076 `chna` audioID entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChnaAudioId {
    pub track_index: u16,
    pub uid: String,
    pub track_ref: String,
    pub pack_ref: String,
}

#[derive(Debug, Default)]
pub(crate) struct RawBlock {
    pub start_secs: f64,
    pub duration_secs: Option<f64>,
    pub speaker_label: Option<String>,
    pub cartesian: bool,
    pub az_or_x: f64,
    pub el_or_y: f64,
    pub dist_or_z: Option<f64>,
    pub gain: f64,
    pub jump_position: bool,
    pub interpolation_secs: Option<f64>,
}

#[derive(Debug, Default)]
pub(crate) struct XmlAudioObject {
    pub gain: f64,
    pub binaural_mode: Option<String>,
    pub pack_refs: Vec<String>,
    pub uid_refs: Vec<String>,
    pub object_refs: Vec<String>,
}

#[derive(Debug, Default)]
pub(crate) struct XmlPackFormat {
    pub type_def: Option<AdmTypeDefinition>,
    pub channel_refs: Vec<String>,
    pub pack_refs: Vec<String>,
}

#[derive(Debug, Default)]
pub(crate) struct XmlChannelFormat {
    pub type_def: Option<AdmTypeDefinition>,
    pub blocks: Vec<RawBlock>,
}

#[derive(Debug, Default)]
pub(crate) struct XmlTrackUid {
    pub track_ref: Option<String>,
    pub channel_ref: Option<String>,
    pub pack_ref: Option<String>,
}

#[derive(Debug, Default)]
pub(crate) struct ParsedAxml {
    pub programmes: Vec<(String, Vec<String>)>,
    pub contents: HashMap<String, Vec<String>>,
    pub objects: HashMap<String, XmlAudioObject>,
    pub pack_formats: HashMap<String, XmlPackFormat>,
    pub channel_formats: HashMap<String, XmlChannelFormat>,
    pub track_formats: HashMap<String, String>,
    pub stream_formats: HashMap<String, (Option<String>, Option<String>)>,
    pub track_uids: HashMap<String, XmlTrackUid>,
    pub binaural_modes: Vec<String>,
}

#[derive(Debug)]
struct ActiveBlock {
    start_secs: f64,
    duration_secs: Option<f64>,
    speaker_label: Option<String>,
    cartesian: bool,
    az: f64,
    el: f64,
    dist: f64,
    x: f64,
    y: f64,
    z: f64,
    gain: f64,
    jump_position: bool,
    interpolation_secs: Option<f64>,
}

impl ActiveBlock {
    fn into_raw(self) -> RawBlock {
        if self.cartesian {
            RawBlock {
                start_secs: self.start_secs,
                duration_secs: self.duration_secs,
                speaker_label: self.speaker_label,
                cartesian: true,
                az_or_x: self.x,
                el_or_y: self.y,
                dist_or_z: Some(self.z),
                gain: self.gain,
                jump_position: self.jump_position,
                interpolation_secs: self.interpolation_secs,
            }
        } else {
            RawBlock {
                start_secs: self.start_secs,
                duration_secs: self.duration_secs,
                speaker_label: self.speaker_label,
                cartesian: false,
                az_or_x: self.az,
                el_or_y: self.el,
                dist_or_z: Some(self.dist),
                gain: self.gain,
                jump_position: self.jump_position,
                interpolation_secs: self.interpolation_secs,
            }
        }
    }
}

#[derive(Debug, Default)]
struct ParserState {
    parsed: ParsedAxml,
    current_prog: Option<(String, Vec<String>)>,
    current_content: Option<(String, Vec<String>)>,
    current_object: Option<(String, XmlAudioObject)>,
    current_pack: Option<(String, XmlPackFormat)>,
    current_channel: Option<(String, XmlChannelFormat)>,
    current_block: Option<ActiveBlock>,
    current_track_fmt: Option<(String, Option<String>)>,
    current_stream_fmt: Option<(String, Option<String>, Option<String>)>,
    current_uid: Option<(String, XmlTrackUid)>,
    last_position_coord: Option<String>,
    last_gain_db: bool,
}

fn unescape_xml(text: &str) -> String {
    text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

fn local_name(name: &str) -> &str {
    name.rsplit(':').next().unwrap_or(name)
}

fn push_unique(vec: &mut Vec<String>, val: &str) {
    let trimmed = val.trim();
    if !trimmed.is_empty() && !vec.iter().any(|x| x == trimmed) {
        vec.push(trimmed.to_string());
    }
}

fn attr_get<'a>(attrs: &'a [(String, String)], key: &str) -> Option<&'a str> {
    attrs
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(key))
        .map(|(_, v)| v.as_str())
}

fn parse_gain_value(raw: &str, is_db: bool) -> Result<f64, String> {
    let v: f64 = raw
        .trim()
        .parse()
        .map_err(|_| format!("Invalid ADM <gain> value '{raw}'"))?;
    if !v.is_finite() {
        return Err("Non-finite ADM <gain> value".to_string());
    }
    Ok(if is_db { 10.0_f64.powf(v / 20.0) } else { v })
}

/// Parses ITU-R BS.2076 ADM timestamps (`HH:MM:SS.fffff` or `HH:MM:SS.fffffSxxxxx`).
pub(crate) fn parse_adm_timecode(raw: &str) -> Result<f64, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(0.0);
    }
    let main_part = trimmed.split('S').next().unwrap_or(trimmed);
    let mut parts = main_part.split(':');
    let hh: f64 = parts
        .next()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("Invalid ADM timecode '{raw}'"))?;
    let mm: f64 = parts
        .next()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("Invalid ADM timecode '{raw}'"))?;
    let ss: f64 = parts
        .next()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("Invalid ADM timecode '{raw}'"))?;
    if parts.next().is_some() || !hh.is_finite() || !mm.is_finite() || !ss.is_finite() {
        return Err(format!("Invalid ADM timecode '{raw}'"));
    }
    let base = hh * 3600.0 + mm * 60.0 + ss;
    if let Some(sub) = trimmed.split('S').nth(1) {
        let mut sub_parts = sub.split('/');
        let num: f64 = sub_parts
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| format!("Invalid ADM rational timecode '{raw}'"))?;
        let den: f64 = sub_parts.next().and_then(|s| s.parse().ok()).unwrap_or(1.0);
        if den <= 0.0 || !num.is_finite() || !den.is_finite() {
            return Err(format!("Invalid ADM rational timecode '{raw}'"));
        }
        return Ok((base + num / den).max(0.0));
    }
    Ok(base.max(0.0))
}

/// Converts ADM Cartesian `(X, Y, Z)` (`+Y` front, `-X` left, `+Z` up) to spherical `(az_deg, el_deg, dist)`.
pub(crate) fn cartesian_to_spherical_deg(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let dist = (x * x + y * y + z * z).sqrt();
    if dist < 1e-12 {
        return (0.0, 0.0, 0.0);
    }
    let az = (-x).atan2(y).to_degrees();
    let el = (z / dist).clamp(-1.0, 1.0).asin().to_degrees();
    (az, el, dist)
}

fn parse_type_def(
    type_def: Option<&str>,
    type_label: Option<&str>,
    id: &str,
) -> Option<AdmTypeDefinition> {
    if let Some(td) = type_def {
        return Some(match td.trim().to_ascii_lowercase().as_str() {
            "directspeakers" => AdmTypeDefinition::DirectSpeakers,
            "objects" => AdmTypeDefinition::Objects,
            "matrix" => AdmTypeDefinition::Matrix,
            "hoa" | "highorderambisonics" | "higherorderambisonics" => AdmTypeDefinition::Hoa,
            "binaural" => AdmTypeDefinition::Binaural,
            other => AdmTypeDefinition::Unknown(other.to_string()),
        });
    }
    match type_label.map(str::trim) {
        Some("0001") => Some(AdmTypeDefinition::DirectSpeakers),
        Some("0002") => Some(AdmTypeDefinition::Matrix),
        Some("0003") => Some(AdmTypeDefinition::Objects),
        Some("0004") => Some(AdmTypeDefinition::Hoa),
        Some("0005") => Some(AdmTypeDefinition::Binaural),
        _ => infer_type_from_common_id(id),
    }
}

pub(crate) fn infer_type_from_common_id(id: &str) -> Option<AdmTypeDefinition> {
    let code = id.split('_').nth(1)?;
    if code.starts_with("0001") {
        Some(AdmTypeDefinition::DirectSpeakers)
    } else if code.starts_with("0002") {
        Some(AdmTypeDefinition::Matrix)
    } else if code.starts_with("0003") {
        Some(AdmTypeDefinition::Objects)
    } else if code.starts_with("0004") {
        Some(AdmTypeDefinition::Hoa)
    } else if code.starts_with("0005") {
        Some(AdmTypeDefinition::Binaural)
    } else {
        None
    }
}

fn spk(label: &str, azimuth_deg: f64, elevation_deg: f64, is_lfe: bool) -> AdmSpeakerPosition {
    AdmSpeakerPosition {
        label: label.to_string(),
        azimuth_deg,
        elevation_deg,
        is_lfe,
    }
}

const SPEAKER_ALIASES: &[(&[&str], &str, f64, f64, bool)] = &[
    (
        &["L", "FL", "M+030", "LEFT", "FRONTLEFT"],
        "M+030",
        30.0,
        0.0,
        false,
    ),
    (
        &["R", "FR", "M-030", "RIGHT", "FRONTRIGHT"],
        "M-030",
        -30.0,
        0.0,
        false,
    ),
    (
        &["C", "FC", "M+000", "M-000", "CENTRE", "CENTER"],
        "M+000",
        0.0,
        0.0,
        false,
    ),
    (&["LFE", "LFE1", "LFE2", "SUB"], "LFE", 0.0, 0.0, true),
    (
        &["LS", "SL", "M+110", "LEFT_SURROUND", "SURROUNDLEFT"],
        "M+110",
        110.0,
        0.0,
        false,
    ),
    (
        &["RS", "SR", "M-110", "RIGHT_SURROUND", "SURROUNDRIGHT"],
        "M-110",
        -110.0,
        0.0,
        false,
    ),
    (&["LSS", "M+090", "SIDELEFT"], "M+090", 90.0, 0.0, false),
    (&["RSS", "M-090", "SIDERIGHT"], "M-090", -90.0, 0.0, false),
    (
        &["LRS", "BL", "LB", "M+135", "BACKLEFT"],
        "M+135",
        135.0,
        0.0,
        false,
    ),
    (
        &["RRS", "BR", "RB", "M-135", "BACKRIGHT"],
        "M-135",
        -135.0,
        0.0,
        false,
    ),
    (
        &["BC", "M+180", "M-180", "BACKCENTRE"],
        "M+180",
        180.0,
        0.0,
        false,
    ),
    (
        &["LTF", "VHL", "U+030", "TOPFRONTLEFT"],
        "U+030",
        30.0,
        30.0,
        false,
    ),
    (
        &["RTF", "VHR", "U-030", "TOPFRONTRIGHT"],
        "U-030",
        -30.0,
        30.0,
        false,
    ),
    (&["U+045"], "U+045", 45.0, 30.0, false),
    (&["U-045"], "U-045", -45.0, 30.0, false),
    (&["U+090", "LTM", "TOPSIDELEFT"], "U+090", 90.0, 30.0, false),
    (
        &["U-090", "RTM", "TOPSIDERIGHT"],
        "U-090",
        -90.0,
        30.0,
        false,
    ),
    (
        &["LTR", "U+110", "TOPSURROUNDLEFT"],
        "U+110",
        110.0,
        30.0,
        false,
    ),
    (
        &["RTR", "U-110", "TOPSURROUNDRIGHT"],
        "U-110",
        -110.0,
        30.0,
        false,
    ),
    (&["U+135", "TOPBACKLEFT"], "U+135", 135.0, 30.0, false),
    (&["U-135", "TOPBACKRIGHT"], "U-135", -135.0, 30.0, false),
    (&["U+000", "TOPFRONTCENTRE"], "U+000", 0.0, 30.0, false),
    (&["U+180", "TOPBACKCENTRE"], "U+180", 180.0, 30.0, false),
    (&["T+000", "TOPCENTRE"], "T+000", 0.0, 90.0, false),
];

const COMMON_DEF_CHANNELS: &[(&str, &str, f64, f64, bool)] = &[
    ("00010001", "M+030", 30.0, 0.0, false),
    ("00010002", "M-030", -30.0, 0.0, false),
    ("00010003", "M+000", 0.0, 0.0, false),
    ("00010004", "LFE", 0.0, 0.0, true),
    ("00010017", "LFE", 0.0, 0.0, true),
    ("00010018", "LFE", 0.0, 0.0, true),
    ("00010005", "M+110", 110.0, 0.0, false),
    ("00010006", "M-110", -110.0, 0.0, false),
    ("00010007", "M+022", 22.5, 0.0, false),
    ("00010008", "M-022", -22.5, 0.0, false),
    ("00010009", "M+180", 180.0, 0.0, false),
    ("0001000a", "M+090", 90.0, 0.0, false),
    ("0001000b", "M-090", -90.0, 0.0, false),
    ("0001000c", "T+000", 0.0, 90.0, false),
    ("0001000d", "U+030", 30.0, 30.0, false),
    ("0001000e", "U+000", 0.0, 30.0, false),
    ("0001000f", "U-030", -30.0, 30.0, false),
    ("00010010", "U+110", 110.0, 30.0, false),
    ("00010011", "U+180", 180.0, 30.0, false),
    ("00010012", "U-110", -110.0, 30.0, false),
    ("00010013", "U+090", 90.0, 30.0, false),
    ("00010014", "U-090", -90.0, 30.0, false),
    ("00010015", "U+135", 135.0, 30.0, false),
    ("00010016", "U-135", -135.0, 30.0, false),
];

/// Resolves standard ITU-R BS.2051 / ADM speaker labels (`M+030`, `L`, `LFE`, `U+030`, etc.).
pub(crate) fn lookup_speaker_label(raw: &str) -> Option<AdmSpeakerPosition> {
    let upper = raw.trim().to_ascii_uppercase();
    SPEAKER_ALIASES
        .iter()
        .find(|(keys, ..)| keys.iter().any(|&k| k == upper))
        .map(|(_, lbl, az, el, lfe)| {
            let out_lbl = if *lfe && upper.starts_with("LFE") {
                &upper
            } else {
                *lbl
            };
            spk(out_lbl, *az, *el, *lfe)
        })
        .or_else(|| parse_bs2051_label(&upper))
}

fn parse_bs2051_label(upper: &str) -> Option<AdmSpeakerPosition> {
    let (el, rest) = if let Some(r) = upper.strip_prefix('M') {
        (0.0, r)
    } else if let Some(r) = upper.strip_prefix('U') {
        (30.0, r)
    } else if let Some(r) = upper.strip_prefix('B') {
        (-15.0, r)
    } else {
        (90.0, upper.strip_prefix('T')?)
    };
    let az: f64 = rest.parse().ok()?;
    if !az.is_finite() || az.abs() > 180.0 {
        return None;
    }
    Some(spk(upper, az, el, false))
}

/// Resolves ITU-R BS.2094 common definition `AC_0001xxxx` (or derived `AT_0001xxxx_01`).
pub(crate) fn lookup_common_channel_id(id: &str) -> Option<AdmSpeakerPosition> {
    let lower = id.split('_').nth(1)?.to_ascii_lowercase();
    COMMON_DEF_CHANNELS
        .iter()
        .find(|(code, ..)| *code == lower)
        .map(|(_, lbl, az, el, lfe)| spk(lbl, *az, *el, *lfe))
}

fn skip_ws(bytes: &[u8], mut i: usize) -> usize {
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    i
}

fn read_single_attr(
    rest: &str,
    bytes: &[u8],
    mut i: usize,
) -> Result<(String, String, usize), String> {
    let name_start = i;
    while i < bytes.len() && bytes[i] != b'=' && !bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    let key = rest[name_start..i].trim().to_string();
    i = skip_ws(bytes, i);
    if i >= bytes.len() || bytes[i] != b'=' {
        return Err(format!("Malformed XML attribute near '{key}'"));
    }
    i = skip_ws(bytes, i + 1);
    let quote = *bytes
        .get(i)
        .filter(|&&b| b == b'"' || b == b'\'')
        .ok_or_else(|| format!("Missing quote for XML attribute '{key}'"))?;
    i += 1;
    let val_start = i;
    while i < bytes.len() && bytes[i] != quote {
        i += 1;
    }
    if i >= bytes.len() {
        return Err(format!("Unclosed attribute value for '{key}'"));
    }
    Ok((key, unescape_xml(&rest[val_start..i]), i + 1))
}

fn parse_attributes(rest: &str) -> Result<(Vec<(String, String)>, bool), String> {
    let mut attrs = Vec::new();
    let bytes = rest.as_bytes();
    let mut i = skip_ws(bytes, 0);
    while i < bytes.len() {
        if bytes[i] == b'/' && i + 1 == bytes.len() {
            return Ok((attrs, true));
        }
        let (k, v, next) = read_single_attr(rest, bytes, i)?;
        attrs.push((k, v));
        i = skip_ws(bytes, next);
    }
    Ok((attrs, false))
}

fn attr_str(attrs: &[(String, String)], key: &str) -> String {
    attr_get(attrs, key).unwrap_or("").to_string()
}

fn open_object_node(state: &mut ParserState, attrs: &[(String, String)]) {
    let mut obj = XmlAudioObject {
        gain: 1.0,
        ..Default::default()
    };
    if let Some(m) = attr_get(attrs, "binauralRenderMode") {
        push_unique(&mut state.parsed.binaural_modes, m);
        obj.binaural_mode = Some(m.trim().to_string());
    }
    state.current_object = Some((attr_str(attrs, "audioObjectID"), obj));
}

fn open_format_node(state: &mut ParserState, tag: &str, attrs: &[(String, String)]) -> bool {
    match tag {
        "audioPackFormat" => {
            let id = attr_str(attrs, "audioPackFormatID");
            let td = parse_type_def(
                attr_get(attrs, "typeDefinition"),
                attr_get(attrs, "typeLabel"),
                &id,
            );
            state.current_pack = Some((
                id,
                XmlPackFormat {
                    type_def: td,
                    ..Default::default()
                },
            ));
        }
        "audioChannelFormat" => {
            let id = attr_str(attrs, "audioChannelFormatID");
            let td = parse_type_def(
                attr_get(attrs, "typeDefinition"),
                attr_get(attrs, "typeLabel"),
                &id,
            );
            state.current_channel = Some((
                id,
                XmlChannelFormat {
                    type_def: td,
                    blocks: Vec::new(),
                },
            ));
        }
        _ => return false,
    }
    true
}

fn start_hierarchy_element(state: &mut ParserState, tag: &str, attrs: &[(String, String)]) -> bool {
    match tag {
        "audioProgramme" => {
            state.current_prog = Some((attr_str(attrs, "audioProgrammeID"), Vec::new()));
        }
        "audioContent" => {
            state.current_content = Some((attr_str(attrs, "audioContentID"), Vec::new()));
        }
        "audioObject" => open_object_node(state, attrs),
        _ => return open_format_node(state, tag, attrs),
    }
    true
}

fn start_element(
    state: &mut ParserState,
    tag: &str,
    attrs: &[(String, String)],
) -> Result<(), String> {
    if start_hierarchy_element(state, tag, attrs) {
        return Ok(());
    }
    match tag {
        "audioBlockFormat" => start_block_element(state, attrs)?,
        "audioTrackFormat" => {
            state.current_track_fmt = Some((attr_str(attrs, "audioTrackFormatID"), None));
        }
        "audioStreamFormat" => {
            state.current_stream_fmt = Some((attr_str(attrs, "audioStreamFormatID"), None, None));
        }
        "audioTrackUID" => {
            state.current_uid = Some((attr_str(attrs, "UID"), XmlTrackUid::default()));
        }
        "position" => {
            state.last_position_coord = attr_get(attrs, "coordinate").map(str::to_ascii_lowercase);
        }
        "gain" => {
            state.last_gain_db =
                attr_get(attrs, "gainUnit").is_some_and(|u| u.eq_ignore_ascii_case("db"));
        }
        "jumpPosition" => {
            if let Some(blk) = &mut state.current_block {
                blk.interpolation_secs = attr_get(attrs, "interpolationLength")
                    .map(parse_adm_timecode)
                    .transpose()?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn start_block_element(state: &mut ParserState, attrs: &[(String, String)]) -> Result<(), String> {
    let start_secs = attr_get(attrs, "rtime")
        .map(parse_adm_timecode)
        .transpose()?
        .unwrap_or(0.0);
    let duration_secs = attr_get(attrs, "duration")
        .map(parse_adm_timecode)
        .transpose()?;
    state.current_block = Some(ActiveBlock {
        start_secs,
        duration_secs,
        speaker_label: None,
        cartesian: false,
        az: 0.0,
        el: 0.0,
        dist: 1.0,
        x: 0.0,
        y: 1.0,
        z: 0.0,
        gain: 1.0,
        jump_position: false,
        interpolation_secs: None,
    });
    Ok(())
}

fn apply_ref_element_text(state: &mut ParserState, tag: &str, val: &str) -> bool {
    match tag {
        "audioContentIDRef" => {
            if let Some((_, refs)) = &mut state.current_prog {
                push_unique(refs, val);
            }
        }
        "audioObjectIDRef" => apply_object_ref(state, val),
        "audioPackFormatIDRef" => apply_pack_ref(state, val),
        "audioChannelFormatIDRef" => apply_channel_ref(state, val),
        "audioTrackUIDRef" => {
            if let Some((_, obj)) = &mut state.current_object {
                push_unique(&mut obj.uid_refs, val);
            }
        }
        "audioTrackFormatIDRef" => {
            if let Some((_, uid)) = &mut state.current_uid {
                uid.track_ref = Some(val.to_string());
            }
        }
        "audioStreamFormatIDRef" => {
            if let Some((_, sref)) = &mut state.current_track_fmt {
                *sref = Some(val.to_string());
            }
        }
        "binauralRenderMode" => apply_binaural_text(state, val),
        _ => return false,
    }
    true
}

fn apply_element_text(state: &mut ParserState, tag: &str, text: &str) -> Result<(), String> {
    let val = text.trim();
    if !val.is_empty()
        && !apply_ref_element_text(state, tag, val)
        && matches!(
            tag,
            "speakerLabel" | "cartesian" | "position" | "gain" | "jumpPosition"
        )
    {
        apply_block_or_object_field(state, tag, val)?;
    }
    Ok(())
}

fn apply_object_ref(state: &mut ParserState, val: &str) {
    if let Some((_, refs)) = &mut state.current_content {
        push_unique(refs, val);
    } else if let Some((_, obj)) = &mut state.current_object {
        push_unique(&mut obj.object_refs, val);
    }
}

fn apply_binaural_text(state: &mut ParserState, val: &str) {
    push_unique(&mut state.parsed.binaural_modes, val);
    if let Some((_, obj)) = &mut state.current_object {
        obj.binaural_mode = Some(val.to_string());
    }
}

fn apply_pack_ref(state: &mut ParserState, val: &str) {
    if let Some((_, obj)) = &mut state.current_object {
        push_unique(&mut obj.pack_refs, val);
    } else if let Some((_, pack)) = &mut state.current_pack {
        push_unique(&mut pack.pack_refs, val);
    } else if let Some((_, _, pack_ref)) = &mut state.current_stream_fmt {
        *pack_ref = Some(val.to_string());
    } else if let Some((_, uid)) = &mut state.current_uid {
        uid.pack_ref = Some(val.to_string());
    }
}

fn apply_channel_ref(state: &mut ParserState, val: &str) {
    if let Some((_, pack)) = &mut state.current_pack {
        push_unique(&mut pack.channel_refs, val);
    } else if let Some((_, ch_ref, _)) = &mut state.current_stream_fmt {
        *ch_ref = Some(val.to_string());
    } else if let Some((_, uid)) = &mut state.current_uid {
        uid.channel_ref = Some(val.to_string());
    }
}

fn apply_position_coord(
    blk: &mut ActiveBlock,
    coord: Option<&str>,
    val: &str,
) -> Result<(), String> {
    let num: f64 = val
        .parse()
        .map_err(|_| format!("Invalid ADM <position> coordinate '{val}'"))?;
    if !num.is_finite() {
        return Err("Non-finite ADM <position> coordinate".to_string());
    }
    match coord {
        Some("azimuth") => blk.az = num,
        Some("elevation") => blk.el = num,
        Some("distance") => blk.dist = num,
        Some("x") => blk.x = num,
        Some("y") => blk.y = num,
        Some("z") => blk.z = num,
        _ => {}
    }
    Ok(())
}

fn apply_block_or_object_field(
    state: &mut ParserState,
    tag: &str,
    val: &str,
) -> Result<(), String> {
    if tag == "gain" {
        let g = parse_gain_value(val, state.last_gain_db)?;
        if let Some(blk) = &mut state.current_block {
            blk.gain = g;
        } else if let Some((_, obj)) = &mut state.current_object {
            obj.gain = g;
        }
        return Ok(());
    }
    let Some(blk) = &mut state.current_block else {
        return Ok(());
    };
    let flag = val == "1" || val.eq_ignore_ascii_case("true");
    match tag {
        "speakerLabel" => blk.speaker_label = Some(val.to_string()),
        "cartesian" => blk.cartesian = flag,
        "jumpPosition" => blk.jump_position = flag,
        "position" => apply_position_coord(blk, state.last_position_coord.as_deref(), val)?,
        _ => {}
    }
    Ok(())
}

fn end_hierarchy_element(state: &mut ParserState, tag: &str) -> bool {
    match tag {
        "audioProgramme" => {
            if let Some(p) = state.current_prog.take() {
                state.parsed.programmes.push(p);
            }
        }
        "audioContent" => {
            if let Some((id, refs)) = state.current_content.take() {
                state.parsed.contents.insert(id, refs);
            }
        }
        "audioObject" => {
            if let Some((id, obj)) = state.current_object.take() {
                state.parsed.objects.insert(id, obj);
            }
        }
        "audioPackFormat" => {
            if let Some((id, pack)) = state.current_pack.take() {
                state.parsed.pack_formats.insert(id, pack);
            }
        }
        "audioChannelFormat" => {
            if let Some((id, ch)) = state.current_channel.take() {
                state.parsed.channel_formats.insert(id, ch);
            }
        }
        _ => return false,
    }
    true
}

fn end_element(state: &mut ParserState, tag: &str) -> Result<(), String> {
    if end_hierarchy_element(state, tag) {
        return Ok(());
    }
    match tag {
        "audioBlockFormat" => {
            if let Some(blk) = state.current_block.take()
                && let Some((_, ch)) = &mut state.current_channel
            {
                if ch.blocks.len() >= MAX_ADM_BLOCKS_PER_CHANNEL {
                    return Err(
                        "ADM audioChannelFormat exceeds MAX_ADM_BLOCKS_PER_CHANNEL".to_string()
                    );
                }
                ch.blocks.push(blk.into_raw());
            }
        }
        "audioTrackFormat" => {
            if let Some((id, Some(sref))) = state.current_track_fmt.take() {
                state.parsed.track_formats.insert(id, sref);
            }
        }
        "audioStreamFormat" => {
            if let Some((id, ch, pack)) = state.current_stream_fmt.take() {
                state.parsed.stream_formats.insert(id, (ch, pack));
            }
        }
        "audioTrackUID" => {
            if let Some((id, uid)) = state.current_uid.take() {
                state.parsed.track_uids.insert(id, uid);
            }
        }
        "position" => state.last_position_coord = None,
        "gain" => state.last_gain_db = false,
        _ => {}
    }
    Ok(())
}

fn find_unquoted_gt(s: &str, start: usize) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut quote: Option<u8> = None;
    let mut i = start;
    while i < bytes.len() {
        let b = bytes[i];
        match quote {
            Some(q) if b == q => quote = None,
            None if b == b'"' || b == b'\'' => quote = Some(b),
            None if b == b'>' => return Some(i),
            _ => {}
        }
        i += 1;
    }
    None
}

fn skip_special_markup(xml: &str, start: usize) -> Result<Option<usize>, String> {
    let rest = &xml[start..];
    for (prefix, end_tok, err) in [
        ("<!--", "-->", "Unclosed XML comment in ADM axml"),
        ("<?", "?>", "Unclosed XML declaration in ADM axml"),
        ("<![CDATA[", "]]>", "Unclosed CDATA in ADM axml"),
        ("<!", ">", "Unclosed DOCTYPE/markup in ADM axml"),
    ] {
        if rest.starts_with(prefix) {
            let end = rest.find(end_tok).ok_or(err)?;
            return Ok(Some(start + end + end_tok.len()));
        }
    }
    Ok(None)
}

fn process_closing_tag(
    inside: &str,
    stack: &mut Vec<String>,
    state: &mut ParserState,
) -> Result<bool, String> {
    let Some(close_raw) = inside.strip_prefix('/') else {
        return Ok(false);
    };
    let close_tag = local_name(close_raw.trim());
    let expected = stack
        .pop()
        .ok_or_else(|| format!("Unexpected closing </{close_tag}>"))?;
    if expected != close_tag {
        return Err(format!(
            "Mismatched closing </{close_tag}> (expected </{expected}>)"
        ));
    }
    end_element(state, close_tag)?;
    Ok(true)
}

fn step_xml_tag(
    xml: &str,
    lt: usize,
    stack: &mut Vec<String>,
    state: &mut ParserState,
) -> Result<usize, String> {
    if let Some(next_pos) = skip_special_markup(xml, lt)? {
        return Ok(next_pos);
    }
    let gt = find_unquoted_gt(xml, lt + 1).ok_or("Unclosed XML tag in ADM axml")?;
    let inside = xml[lt + 1..gt].trim();
    if process_closing_tag(inside, stack, state)? {
        return Ok(gt + 1);
    }
    let mut split = inside.splitn(2, char::is_whitespace);
    let raw_name = split.next().unwrap_or("");
    let tag_name = local_name(raw_name.trim_end_matches('/'));
    if tag_name.is_empty() {
        return Err("Empty XML tag name in ADM axml".to_string());
    }
    let (attrs, attrs_self_close) = split
        .next()
        .map(|s| parse_attributes(s.trim()))
        .transpose()?
        .unwrap_or((Vec::new(), false));
    start_element(state, tag_name, &attrs)?;
    if raw_name.ends_with('/') || attrs_self_close {
        end_element(state, tag_name)?;
    } else if stack.len() >= MAX_XML_DEPTH {
        return Err("ADM axml nesting depth exceeds limit".to_string());
    } else {
        stack.push(tag_name.to_string());
    }
    Ok(gt + 1)
}

pub(crate) fn parse_adm_axml_document(xml: &str) -> Result<ParsedAxml, String> {
    let mut state = ParserState::default();
    let mut stack: Vec<String> = Vec::new();
    let mut pos = 0;
    let mut elements = 0usize;
    while pos < xml.len() {
        let Some(rel_lt) = xml[pos..].find('<') else {
            break;
        };
        let lt = pos + rel_lt;
        if lt > pos
            && let Some(top) = stack.last()
        {
            apply_element_text(&mut state, top, &unescape_xml(&xml[pos..lt]))?;
        }
        elements += 1;
        if elements > MAX_ADM_ELEMENTS {
            return Err("ADM axml exceeds MAX_ADM_ELEMENTS".to_string());
        }
        pos = step_xml_tag(xml, lt, &mut stack, &mut state)?;
    }
    if !stack.is_empty() {
        return Err(format!(
            "Unclosed XML element <{}> in ADM axml",
            stack.join(">/<")
        ));
    }
    if elements == 0 {
        return Err("ADM axml contains no XML elements".to_string());
    }
    Ok(state.parsed)
}

fn collect_object_map(parsed: &ParsedAxml) -> HashMap<String, (String, Option<String>, f64)> {
    let mut uid_meta: HashMap<String, (String, Option<String>, f64)> = HashMap::new();
    let mut active_objs = HashSet::new();
    for (_, content_refs) in &parsed.programmes {
        for cref in content_refs {
            if let Some(obj_refs) = parsed.contents.get(cref) {
                for oref in obj_refs {
                    active_objs.insert(oref.clone());
                }
            }
        }
    }
    let obj_iter: Vec<&String> = if active_objs.is_empty() {
        parsed.objects.keys().collect()
    } else {
        active_objs
            .iter()
            .filter(|id| parsed.objects.contains_key(*id))
            .collect()
    };
    for oid in obj_iter {
        if let Some(obj) = parsed.objects.get(oid) {
            for uid in &obj.uid_refs {
                uid_meta.insert(
                    uid.clone(),
                    (oid.clone(), obj.binaural_mode.clone(), obj.gain),
                );
            }
        }
    }
    uid_meta
}

fn resolve_channel_and_pack_type(
    parsed: &ParsedAxml,
    entry: &ChnaAudioId,
) -> (Option<String>, Option<String>, Option<AdmTypeDefinition>) {
    let uid_node = parsed.track_uids.get(&entry.uid);
    let track_ref = if entry.track_ref.is_empty() {
        uid_node.and_then(|u| u.track_ref.clone())
    } else {
        Some(entry.track_ref.clone())
    };
    let stream_ids = track_ref
        .as_deref()
        .and_then(|tr| parsed.track_formats.get(tr))
        .and_then(|sf| parsed.stream_formats.get(sf));

    let ch_id = uid_node
        .and_then(|u| u.channel_ref.clone())
        .or_else(|| stream_ids.and_then(|(c, _)| c.clone()))
        .or_else(|| {
            (!entry.track_ref.is_empty()).then(|| derive_channel_id_from_track(&entry.track_ref))
        });
    let pack_id = (!entry.pack_ref.is_empty())
        .then(|| entry.pack_ref.clone())
        .or_else(|| uid_node.and_then(|u| u.pack_ref.clone()))
        .or_else(|| stream_ids.and_then(|(_, p)| p.clone()));

    let td = ch_id
        .as_deref()
        .and_then(|c| {
            parsed
                .channel_formats
                .get(c)
                .and_then(|cf| cf.type_def.clone())
        })
        .or_else(|| {
            pack_id.as_deref().and_then(|p| {
                parsed
                    .pack_formats
                    .get(p)
                    .and_then(|pf| pf.type_def.clone())
            })
        })
        .or_else(|| ch_id.as_deref().and_then(infer_type_from_common_id))
        .or_else(|| pack_id.as_deref().and_then(infer_type_from_common_id))
        .or_else(|| infer_type_from_common_id(&entry.track_ref));

    (ch_id, pack_id, td)
}

fn derive_channel_id_from_track(track_ref: &str) -> String {
    if let Some(code) = track_ref.strip_prefix("AT_") {
        let base = code.split('_').next().unwrap_or(code);
        return format!("AC_{base}");
    }
    track_ref.to_string()
}

fn block_to_speaker_pos(ch_id: &str, blocks: &[RawBlock]) -> Option<AdmSpeakerPosition> {
    if let Some(first) = blocks.first() {
        if let Some(lbl) = &first.speaker_label
            && let Some(pos) = lookup_speaker_label(lbl)
        {
            return Some(pos);
        }
        if first.cartesian {
            let (az, el, _) = cartesian_to_spherical_deg(
                first.az_or_x,
                first.el_or_y,
                first.dist_or_z.unwrap_or(0.0),
            );
            return Some(spk(&format!("XY({az:.0},{el:.0})"), az, el, false));
        }
        return Some(spk(
            &format!("AZ({:.0},{:.0})", first.az_or_x, first.el_or_y),
            first.az_or_x,
            first.el_or_y,
            false,
        ));
    }
    lookup_common_channel_id(ch_id)
}

fn block_to_object_block(raw: &RawBlock) -> AdmObjectBlock {
    let (azimuth_deg, elevation_deg, distance) = if raw.cartesian {
        cartesian_to_spherical_deg(raw.az_or_x, raw.el_or_y, raw.dist_or_z.unwrap_or(0.0))
    } else {
        (raw.az_or_x, raw.el_or_y, raw.dist_or_z.unwrap_or(1.0))
    };
    AdmObjectBlock {
        start_secs: raw.start_secs,
        duration_secs: raw.duration_secs,
        azimuth_deg,
        elevation_deg,
        distance,
        gain: raw.gain,
        jump_position: raw.jump_position,
        interpolation_secs: raw.interpolation_secs,
    }
}

fn build_track_kind(
    parsed: &ParsedAxml,
    entry: &ChnaAudioId,
    obj_gain: f64,
) -> Result<AdmTrackKind, String> {
    let (ch_id, _pack_id, td_opt) = resolve_channel_and_pack_type(parsed, entry);
    let td = td_opt.ok_or_else(|| {
        format!(
            "Track {} (UID '{}') lacks ADM typeDefinition",
            entry.track_index, entry.uid
        )
    })?;
    let blocks = ch_id
        .as_deref()
        .and_then(|c| parsed.channel_formats.get(c))
        .map(|cf| cf.blocks.as_slice())
        .unwrap_or(&[]);
    match td {
        AdmTypeDefinition::DirectSpeakers => {
            let cid = ch_id.as_deref().ok_or_else(|| {
                format!(
                    "Track {} DirectSpeakers lacks audioChannelFormat",
                    entry.track_index
                )
            })?;
            let pos = block_to_speaker_pos(cid, blocks).ok_or_else(|| {
                format!(
                    "Track {} DirectSpeakers '{cid}' unknown speaker",
                    entry.track_index
                )
            })?;
            Ok(AdmTrackKind::DirectSpeaker(pos))
        }
        AdmTypeDefinition::Objects => Ok(AdmTrackKind::Object {
            object_gain: obj_gain,
            blocks: blocks.iter().map(block_to_object_block).collect(),
        }),
        other => Ok(AdmTrackKind::Unsupported(other)),
    }
}

fn resolve_single_entry(
    parsed: &ParsedAxml,
    uid_meta: &HashMap<String, (String, Option<String>, f64)>,
    entry: &ChnaAudioId,
    total_channels: u16,
    seen_tracks: &mut HashSet<u16>,
) -> Result<Option<AdmTrackAssociation>, String> {
    if entry.track_index == 0 {
        return Ok(None);
    }
    if entry.track_index > total_channels {
        return Err(format!(
            "chna track_index {} exceeds fmt channels ({total_channels})",
            entry.track_index
        ));
    }
    if !seen_tracks.insert(entry.track_index) {
        return Err(format!(
            "Duplicate active chna mapping for track_index {}",
            entry.track_index
        ));
    }
    if !parsed.track_uids.contains_key(&entry.uid) {
        return Err(format!(
            "chna UID '{}' on track {} missing in axml",
            entry.uid, entry.track_index
        ));
    }
    let (obj_id, binaural_mode, obj_gain) = uid_meta.get(&entry.uid).cloned().ok_or_else(|| {
        format!(
            "audioTrackUID '{}' on track {} not in active audioObject",
            entry.uid, entry.track_index
        )
    })?;
    let kind = build_track_kind(parsed, entry, obj_gain)?;
    Ok(Some(AdmTrackAssociation {
        track_index: entry.track_index,
        uid: entry.uid.clone(),
        object_id: Some(obj_id),
        binaural_render_mode: binaural_mode,
        kind,
    }))
}

pub(crate) fn resolve_adm_tracks(
    parsed: &ParsedAxml,
    chna_entries: &[ChnaAudioId],
    total_channels: u16,
    errors: &mut Vec<String>,
) -> Vec<AdmTrackAssociation> {
    let uid_meta = collect_object_map(parsed);
    let mut seen_tracks: HashSet<u16> = HashSet::new();
    let mut associations = Vec::new();
    for entry in chna_entries {
        match resolve_single_entry(parsed, &uid_meta, entry, total_channels, &mut seen_tracks) {
            Ok(Some(assoc)) => associations.push(assoc),
            Ok(None) => {}
            Err(msg) => errors.push(msg),
        }
    }
    associations.sort_by_key(|a| a.track_index);
    associations
}
