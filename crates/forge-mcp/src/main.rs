//! # `forge-mcp` — Model Context Protocol (MCP) JSON-RPC 2.0 Server
//!
//! Zero-overhead native Rust stdio MCP server exposing all ForgeAudio DSP,
//! music theory, vocal surgery, GPU worker, and REAPER 7.82 session tools to
//! AI agents (Antigravity, Claude Code, Cursor, REAPER AI scripts).

use forge_core::{AudioBuffer, ForgeError};
use forge_dsp::{
    align_phase_gcc_phat, align_vocal_double_dtw, apply_18_band_eq, apply_spectral_defizz,
    apply_spectral_unmask, audit_audio, clean_vocal_surgical, compute_reference_match_eq,
    normalize_and_limit, run_full_restoration_chain, DeFizzConfig, RestorationConfig, UnmaskConfig,
};
use forge_reaper::{
    install_jsfx_plugins, validate_isrc, write_distrokid_metadata, ReaperProjectBuilder,
    ReaperTrackSpec, ReleaseMetadata,
};
use forge_theory::{
    export_abc_score, extract_drum_midi_triggers, reharmonize_progression, solve_voice_leading,
    write_midi_file, GrooveStyle, VoicingStyle,
};
use forge_vocal::{
    generate_vocal_harmony_stack, shift_vocal_pitch_formant_preserved, track_vocal_pitch,
    VocalScale,
};
use forge_worker::{separate_stems, WorkerBackend};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let Ok(raw) = line else { break };
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }

        let req: JsonRpcRequest = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(e) => {
                let resp = JsonRpcResponse {
                    jsonrpc: "2.0",
                    id: Value::Null,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32700,
                        message: format!("Parse error: {e}"),
                    }),
                };
                let _ = writeln!(
                    stdout,
                    "{}",
                    serde_json::to_string(&resp).unwrap_or_default()
                );
                let _ = stdout.flush();
                continue;
            }
        };

        let Some(req_id) = req.id.clone() else {
            continue;
        };

        let response = handle_request(req_id, &req.method, &req.params);
        if let Ok(serialized) = serde_json::to_string(&response) {
            let _ = writeln!(stdout, "{serialized}");
            let _ = stdout.flush();
        }
    }
}

fn handle_request(id: Value, method: &str, params: &Value) -> JsonRpcResponse {
    match method {
        "initialize" => JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "forge-mcp",
                    "version": env!("CARGO_PKG_VERSION")
                }
            })),
            error: None,
        },
        "tools/list" => JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({
                "tools": tool_definitions()
            })),
            error: None,
        },
        "tools/call" => {
            let tool_name = params
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let args = params
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));

            match dispatch_tool(tool_name, &args) {
                Ok(val) => JsonRpcResponse {
                    jsonrpc: "2.0",
                    id,
                    result: Some(json!({
                        "content": [{
                            "type": "text",
                            "text": serde_json::to_string_pretty(&val).unwrap_or_default()
                        }],
                        "isError": false
                    })),
                    error: None,
                },
                Err(err) => JsonRpcResponse {
                    jsonrpc: "2.0",
                    id,
                    result: Some(json!({
                        "content": [{
                            "type": "text",
                            "text": format!("ForgeAudio Error: {err}")
                        }],
                        "isError": true
                    })),
                    error: None,
                },
            }
        }
        _ => JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(JsonRpcError {
                code: -32601,
                message: format!("Method not found: {method}"),
            }),
        },
    }
}

fn tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "name": "forge_audit_audio",
            "description": "Run an ITU-R BS.1770-4 / EBU R128 loudness, 4x polyphase True Peak (dBTP), LRA, and AI Spectral Fizz Index (2.5-6.5 kHz) audit on an audio file.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to audio file" }
                },
                "required": ["path"]
            }
        }),
        json!({
            "name": "forge_restore_and_defizz",
            "description": "Clean AI-generated or live audio: DC offset, cubic Hermite de-clip, 50/60Hz hum removal, vocal de-click/de-plosive/de-breath, and OpenSoothe dynamic spectral de-fizz.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input": { "type": "string" },
                    "output": { "type": "string" },
                    "vocal": { "type": "boolean", "default": false },
                    "defizz_depth_db": { "type": "number", "default": 9.0 },
                    "sensitivity_db": { "type": "number", "default": 3.5 }
                },
                "required": ["input", "output"]
            }
        }),
        json!({
            "name": "forge_spectral_unmask",
            "description": "Run OpenSpacer (Trackspacer equivalent) sidechain STFT dynamic frequency unmasking between a sidechain trigger (vocal or kick) and target bed (music or bass).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "target": { "type": "string" },
                    "sidechain": { "type": "string" },
                    "output": { "type": "string" },
                    "kick_bass_mode": { "type": "boolean", "default": false }
                },
                "required": ["target", "sidechain", "output"]
            }
        }),
        json!({
            "name": "forge_phase_align",
            "description": "Align multi-mic phase via GCC-PHAT (Auto-Align equivalent) or vocal doubles via FastDTW stretch markers (VocAlign equivalent).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "reference": { "type": "string" },
                    "target": { "type": "string" },
                    "output": { "type": "string" },
                    "vocal_dtw": { "type": "boolean", "default": false }
                },
                "required": ["reference", "target", "output"]
            }
        }),
        json!({
            "name": "forge_vocal_suite",
            "description": "Run YIN pitch tracking, formant-preserving pitch correction, 3-part chord-aware vocal harmony generation, and REAPER <PITCHENV compilation.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input": { "type": "string" },
                    "output_dir": { "type": "string" },
                    "scale": { "type": "string", "default": "D minor" },
                    "retune_strength": { "type": "number", "default": 0.85 },
                    "generate_harmonies": { "type": "boolean", "default": true }
                },
                "required": ["input", "output_dir"]
            }
        }),
        json!({
            "name": "forge_music_theory_midi",
            "description": "Generate voice-led MIDI chords, Neo-Soul/Jazz reharmonizations, Dilla/MPC humanized grooves, and ABC notation.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "chords": { "type": "string" },
                    "output_midi": { "type": "string" },
                    "bpm": { "type": "number", "default": 92.0 },
                    "style": { "type": "string", "default": "neo-soul" }
                },
                "required": ["chords", "output_midi"]
            }
        }),
        json!({
            "name": "forge_master_distrokid",
            "description": "Master stereo audio to -14.0 LUFS / -1.0 dBTP with Reference Match EQ, 18-band Mastering EQ, and optional ISO 3901 ISRC tagging.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input": { "type": "string" },
                    "output": { "type": "string" },
                    "reference": { "type": "string" },
                    "target_lufs": { "type": "number", "default": -14.0 },
                    "ceiling_dbtp": { "type": "number", "default": -1.0 },
                    "isrc": { "type": "string" },
                    "title": { "type": "string" },
                    "artist": { "type": "string" }
                },
                "required": ["input", "output"]
            }
        }),
        json!({
            "name": "forge_ingest_to_reaper",
            "description": "Full autonomous pipeline: separate stems, clean & de-fizz audio, extract drum MIDI, build voice-led chords, master to -14 LUFS, install JSFX plugins, and generate a REAPER 7.82 .RPP project.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input": { "type": "string" },
                    "output_dir": { "type": "string" },
                    "bpm": { "type": "number", "default": 120.0 },
                    "scale": { "type": "string", "default": "D minor" },
                    "chords": { "type": "string", "default": "Dm9 Bbmaj7 Gm9 A7b13" }
                },
                "required": ["input", "output_dir"]
            }
        }),
    ]
}

fn dispatch_tool(name: &str, args: &Value) -> Result<Value, ForgeError> {
    match name {
        "forge_audit_audio" => {
            let path = args
                .get("path")
                .and_then(Value::as_str)
                .ok_or_else(|| ForgeError::InvalidParameter("Missing 'path'".into()))?;
            let buf = AudioBuffer::load(Path::new(path))?;
            let report = audit_audio(&buf);
            Ok(serde_json::to_value(report)?)
        }
        "forge_restore_and_defizz" => {
            let input = args
                .get("input")
                .and_then(Value::as_str)
                .ok_or_else(|| ForgeError::InvalidParameter("Missing 'input'".into()))?;
            let output = args
                .get("output")
                .and_then(Value::as_str)
                .ok_or_else(|| ForgeError::InvalidParameter("Missing 'output'".into()))?;
            let vocal = args.get("vocal").and_then(Value::as_bool).unwrap_or(false);
            let depth = args
                .get("defizz_depth_db")
                .and_then(Value::as_f64)
                .unwrap_or(9.0);
            let sensitivity = args
                .get("sensitivity_db")
                .and_then(Value::as_f64)
                .unwrap_or(3.5);

            let raw = AudioBuffer::load(Path::new(input))?;
            let before = audit_audio(&raw);
            let mut stage1 = run_full_restoration_chain(&raw, &RestorationConfig::default())?;
            if vocal {
                stage1 = clean_vocal_surgical(&stage1, true, true, -12.0)?;
            }
            let cleaned = apply_spectral_defizz(
                &stage1,
                &DeFizzConfig {
                    sensitivity_db: sensitivity,
                    max_depth_db: depth,
                    ..DeFizzConfig::default()
                },
            )?;
            let after = audit_audio(&cleaned);
            cleaned.save_wav(Path::new(output), 24)?;
            Ok(json!({
                "output": output,
                "before_fizz_index": before.ai_fizz_index,
                "after_fizz_index": after.ai_fizz_index,
                "after_lufs": after.integrated_lufs
            }))
        }
        "forge_spectral_unmask" => {
            let target = args
                .get("target")
                .and_then(Value::as_str)
                .ok_or_else(|| ForgeError::InvalidParameter("Missing 'target'".into()))?;
            let sidechain = args
                .get("sidechain")
                .and_then(Value::as_str)
                .ok_or_else(|| ForgeError::InvalidParameter("Missing 'sidechain'".into()))?;
            let output = args
                .get("output")
                .and_then(Value::as_str)
                .ok_or_else(|| ForgeError::InvalidParameter("Missing 'output'".into()))?;
            let kick_bass = args
                .get("kick_bass_mode")
                .and_then(Value::as_bool)
                .unwrap_or(false);

            let target_buf = AudioBuffer::load(Path::new(target))?;
            let sc_buf = AudioBuffer::load(Path::new(sidechain))?;
            let cfg = if kick_bass {
                UnmaskConfig::kick_over_bass()
            } else {
                UnmaskConfig::vocal_over_instruments()
            };
            let unmasked = apply_spectral_unmask(&target_buf, &sc_buf, &cfg)?;
            unmasked.save_wav(Path::new(output), 24)?;
            Ok(json!({ "output": output, "kick_bass_mode": kick_bass }))
        }
        "forge_phase_align" => {
            let reference = args
                .get("reference")
                .and_then(Value::as_str)
                .ok_or_else(|| ForgeError::InvalidParameter("Missing 'reference'".into()))?;
            let target = args
                .get("target")
                .and_then(Value::as_str)
                .ok_or_else(|| ForgeError::InvalidParameter("Missing 'target'".into()))?;
            let output = args
                .get("output")
                .and_then(Value::as_str)
                .ok_or_else(|| ForgeError::InvalidParameter("Missing 'output'".into()))?;
            let vocal_dtw = args
                .get("vocal_dtw")
                .and_then(Value::as_bool)
                .unwrap_or(false);

            let ref_buf = AudioBuffer::load(Path::new(reference))?;
            let tgt_buf = AudioBuffer::load(Path::new(target))?;
            if vocal_dtw {
                let (warped, markers) = align_vocal_double_dtw(&ref_buf, &tgt_buf, 120.0)?;
                warped.save_wav(Path::new(output), 24)?;
                Ok(json!({
                    "mode": "vocal_dtw",
                    "output": output,
                    "stretch_markers_count": markers.len()
                }))
            } else {
                let (aligned, info) = align_phase_gcc_phat(&ref_buf, &tgt_buf, 35.0)?;
                aligned.save_wav(Path::new(output), 24)?;
                Ok(serde_json::to_value(info)?)
            }
        }
        "forge_vocal_suite" => {
            let input = args
                .get("input")
                .and_then(Value::as_str)
                .ok_or_else(|| ForgeError::InvalidParameter("Missing 'input'".into()))?;
            let output_dir = PathBuf::from(
                args.get("output_dir")
                    .and_then(Value::as_str)
                    .ok_or_else(|| ForgeError::InvalidParameter("Missing 'output_dir'".into()))?,
            );
            std::fs::create_dir_all(&output_dir)?;
            let scale_str = args
                .get("scale")
                .and_then(Value::as_str)
                .unwrap_or("D minor");
            let retune = args
                .get("retune_strength")
                .and_then(Value::as_f64)
                .unwrap_or(0.85);
            let gen_harmonies = args
                .get("generate_harmonies")
                .and_then(Value::as_bool)
                .unwrap_or(true);

            let raw = AudioBuffer::load(Path::new(input))?;
            let clean = clean_vocal_surgical(&raw, true, true, -12.0)?;
            let scale = VocalScale::parse(scale_str)?;
            let contour = track_vocal_pitch(&clean, &scale, retune);
            let corr_curve: Vec<(f64, f64)> = contour
                .iter()
                .map(|f| (f.time_secs, f.correction_semitones))
                .collect();
            let tuned = shift_vocal_pitch_formant_preserved(&clean, &corr_curve, 0.0)?;
            let tuned_path = output_dir.join("Vocal_Lead_Tuned_Clean.wav");
            tuned.save_wav(&tuned_path, 24)?;

            let mut harmonies = Vec::new();
            if gen_harmonies {
                let (h_up, h_down, h_dbl) = generate_vocal_harmony_stack(&tuned, &scale)?;
                let p_up = output_dir.join("Vocal_Harmony_3rd_Above.wav");
                let p_down = output_dir.join("Vocal_Harmony_3rd_Below.wav");
                let p_dbl = output_dir.join("Vocal_Formant_Double.wav");
                h_up.save_wav(&p_up, 24)?;
                h_down.save_wav(&p_down, 24)?;
                h_dbl.save_wav(&p_dbl, 24)?;
                harmonies.extend([
                    p_up.to_string_lossy().into_owned(),
                    p_down.to_string_lossy().into_owned(),
                    p_dbl.to_string_lossy().into_owned(),
                ]);
            }
            Ok(json!({
                "tuned_vocal": tuned_path,
                "harmonies": harmonies,
                "frames_tracked": contour.len()
            }))
        }
        "forge_music_theory_midi" => {
            let chords_str = args
                .get("chords")
                .and_then(Value::as_str)
                .ok_or_else(|| ForgeError::InvalidParameter("Missing 'chords'".into()))?;
            let output_midi = args
                .get("output_midi")
                .and_then(Value::as_str)
                .ok_or_else(|| ForgeError::InvalidParameter("Missing 'output_midi'".into()))?;
            let bpm = args.get("bpm").and_then(Value::as_f64).unwrap_or(92.0);
            let raw_chords: Vec<String> = chords_str
                .split_whitespace()
                .map(ToString::to_string)
                .collect();
            let reharm = reharmonize_progression(&raw_chords, VoicingStyle::NeoSoulRootless);
            let (chord_notes, bass_notes) = solve_voice_leading(
                &reharm,
                VoicingStyle::NeoSoulRootless,
                GrooveStyle::NeoSoulDilla,
                480,
                4,
            )?;
            write_midi_file(
                Path::new(output_midi),
                bpm,
                480,
                &[("Voiced_Chords", &chord_notes), ("Sub_Bass", &bass_notes)],
            )?;
            let abc = export_abc_score("ForgeAudio Progression", "C", bpm.round() as u32, &reharm);
            Ok(json!({
                "output_midi": output_midi,
                "abc_notation": abc,
                "reharmonized_chords": reharm
            }))
        }
        "forge_master_distrokid" => {
            let input = args
                .get("input")
                .and_then(Value::as_str)
                .ok_or_else(|| ForgeError::InvalidParameter("Missing 'input'".into()))?;
            let output = args
                .get("output")
                .and_then(Value::as_str)
                .ok_or_else(|| ForgeError::InvalidParameter("Missing 'output'".into()))?;
            let target_lufs = args
                .get("target_lufs")
                .and_then(Value::as_f64)
                .unwrap_or(-14.0);
            let ceiling_dbtp = args
                .get("ceiling_dbtp")
                .and_then(Value::as_f64)
                .unwrap_or(-1.0);

            let mut buf = AudioBuffer::load(Path::new(input))?;
            if let Some(ref_str) = args.get("reference").and_then(Value::as_str) {
                let ref_buf = AudioBuffer::load(Path::new(ref_str))?;
                let curve = compute_reference_match_eq(&buf, &ref_buf, 4.5);
                apply_18_band_eq(&mut buf, &curve);
            }
            let mastered = normalize_and_limit(&buf, target_lufs, ceiling_dbtp)?;
            let out_path = PathBuf::from(output);
            if let Some(raw_isrc) = args.get("isrc").and_then(Value::as_str) {
                let canonical = validate_isrc(raw_isrc)?;
                let tmp = out_path.with_extension("pre_tag.wav");
                mastered.save_wav(&tmp, 24)?;
                let meta = ReleaseMetadata {
                    title: args
                        .get("title")
                        .and_then(Value::as_str)
                        .unwrap_or("Untitled")
                        .into(),
                    artist: args
                        .get("artist")
                        .and_then(Value::as_str)
                        .unwrap_or("ForgeAudio")
                        .into(),
                    album: "Single".into(),
                    isrc: canonical,
                    bpm: None,
                    key: None,
                    year: Some(2026),
                };
                write_distrokid_metadata(&tmp, &out_path, &meta)?;
                let _ = std::fs::remove_file(&tmp);
            } else {
                mastered.save_wav(&out_path, 24)?;
            }
            let report = audit_audio(&mastered);
            Ok(serde_json::to_value(report)?)
        }
        "forge_ingest_to_reaper" => {
            let input = PathBuf::from(
                args.get("input")
                    .and_then(Value::as_str)
                    .ok_or_else(|| ForgeError::InvalidParameter("Missing 'input'".into()))?,
            );
            let output_dir = PathBuf::from(
                args.get("output_dir")
                    .and_then(Value::as_str)
                    .ok_or_else(|| ForgeError::InvalidParameter("Missing 'output_dir'".into()))?,
            );
            let bpm = args.get("bpm").and_then(Value::as_f64).unwrap_or(120.0);
            let scale_str = args
                .get("scale")
                .and_then(Value::as_str)
                .unwrap_or("D minor");
            let chords = args
                .get("chords")
                .and_then(Value::as_str)
                .unwrap_or("Dm9 Bbmaj7 Gm9 A7b13");

            std::fs::create_dir_all(&output_dir)?;
            let stems_dir = output_dir.join("stems");
            let midi_dir = output_dir.join("midi");
            std::fs::create_dir_all(&stems_dir)?;
            std::fs::create_dir_all(&midi_dir)?;

            let bundle = separate_stems(&input, &stems_dir, &WorkerBackend::Auto)?;
            let parsed_scale = VocalScale::parse(scale_str)?;
            let mut vocal_pitch_frames = Vec::new();
            if let Some(ref voc_path) = bundle.vocals {
                let voc_buf = AudioBuffer::load(voc_path)?;
                let voc_clean = clean_vocal_surgical(&voc_buf, true, true, -10.0)?;
                let voc_defizz = apply_spectral_defizz(&voc_clean, &DeFizzConfig::default())?;
                vocal_pitch_frames = track_vocal_pitch(&voc_defizz, &parsed_scale, 0.85);
                voc_defizz.save_wav(voc_path, 24)?;
            }

            let drum_src = bundle.drums.as_ref().unwrap_or(&input);
            let drum_buf = AudioBuffer::load(drum_src)?;
            let duration_secs = drum_buf.duration_secs();
            let (kick_notes, snare_notes, hat_notes) =
                extract_drum_midi_triggers(&drum_buf, bpm, 480);
            let drum_midi_path = midi_dir.join("01_Drum_Triggers.mid");
            write_midi_file(
                &drum_midi_path,
                bpm,
                480,
                &[
                    ("Kick_Trigger_C1", &kick_notes),
                    ("Snare_Trigger_D1", &snare_notes),
                    ("HiHat_Trigger_F#1", &hat_notes),
                ],
            )?;

            let raw_chords: Vec<String> =
                chords.split_whitespace().map(ToString::to_string).collect();
            let reharm = reharmonize_progression(&raw_chords, VoicingStyle::NeoSoulRootless);
            let (chord_notes, bass_notes) = solve_voice_leading(
                &reharm,
                VoicingStyle::NeoSoulRootless,
                GrooveStyle::NeoSoulDilla,
                480,
                4,
            )?;
            let chords_midi_path = midi_dir.join("02_Voiced_Chords.mid");
            write_midi_file(
                &chords_midi_path,
                bpm,
                480,
                &[("Voiced_Chords", &chord_notes), ("Sub_Bass", &bass_notes)],
            )?;

            let full_mix = AudioBuffer::load(&input)?;
            let restored = run_full_restoration_chain(&full_mix, &RestorationConfig::default())?;
            let defizzed = apply_spectral_defizz(&restored, &DeFizzConfig::default())?;
            let mastered = normalize_and_limit(&defizzed, -14.0, -1.0)?;
            let master_path = output_dir.join("Master_DistroKid_24bit_48kHz.wav");
            mastered.save_wav(&master_path, 24)?;
            let master_report = audit_audio(&mastered);

            let jsfx_dir = output_dir.join("Effects");
            let _ = install_jsfx_plugins(Some(&jsfx_dir))?;

            let mut rpp =
                ReaperProjectBuilder::new("ForgeAudio Studio Session", bpm, full_mix.sample_rate);
            rpp.add_default_song_structure_markers();
            rpp.tracks.push(ReaperTrackSpec {
                name: "01 Lead Vocal".to_string(),
                file_path: bundle.vocals,
                is_midi: false,
                duration_secs,
                volume_lin: 0.85,
                pan: 0.0,
                color: 0x0155_88FF,
                jsfx_inserts: vec![
                    "Forge_Vocal_Surgical.jsfx".to_string(),
                    "Forge_OpenSoothe_DeFizz.jsfx".to_string(),
                ],
                sidechain_from_track: None,
                stretch_markers: Vec::new(),
                pitch_envelope: vocal_pitch_frames,
            });
            rpp.tracks.push(ReaperTrackSpec {
                name: "02 MIDI Chords & Triggers".to_string(),
                file_path: Some(chords_midi_path.clone()),
                is_midi: true,
                duration_secs,
                volume_lin: 0.85,
                pan: 0.0,
                color: 0x01FF_CC44,
                jsfx_inserts: Vec::new(),
                sidechain_from_track: None,
                stretch_markers: Vec::new(),
                pitch_envelope: Vec::new(),
            });
            let rpp_path = output_dir.join("ForgeAudio_Session.rpp");
            rpp.save_rpp(&rpp_path)?;

            Ok(json!({
                "rpp_project": rpp_path,
                "master_wav": master_path,
                "midi_file": chords_midi_path,
                "master_lufs": master_report.integrated_lufs,
                "master_true_peak_dbtp": master_report.true_peak_dbtp,
                "ai_fizz_index": master_report.ai_fizz_index
            }))
        }
        _ => Err(ForgeError::InvalidParameter(format!(
            "Unknown tool: {name}"
        ))),
    }
}
