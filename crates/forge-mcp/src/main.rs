//! # `forge-mcp` — Model Context Protocol (MCP) JSON-RPC 2.0 Server
//!
//! Zero-overhead native Rust stdio MCP server exposing all ForgeAudio DSP,
//! music theory, vocal surgery, GPU worker, and REAPER 7.82 session tools to
//! AI agents (Antigravity, Claude Code, Cursor, REAPER AI scripts).
//!
//! File access is confined: every path argument must resolve inside the roots listed in
//! `FORGE_MCP_ROOT` (default: the directory the server was launched from). See [`sandbox`].

mod sandbox;

use forge_core::{AudioBuffer, ForgeError};
use forge_dsp::{
    DeFizzConfig, DeliveryOptions, DeliveryTarget, GenreTonalTarget, RestorationConfig, ToneMatch,
    UnmaskConfig, align_phase_gcc_phat, align_vocal_double_dtw, apply_18_band_eq,
    apply_spectral_defizz, apply_spectral_unmask, audit_audio, clean_vocal_surgical,
    compute_reference_match_eq, deliver, normalize_and_limit, run_full_restoration_chain,
};
use forge_reaper::{
    ReaperProjectBuilder, ReaperTrackSpec, ReleaseMetadata, install_jsfx_plugins, validate_isrc,
    write_distrokid_metadata,
};
use forge_theory::{
    GrooveStyle, VoicingStyle, export_abc_score, extract_drum_midi_triggers,
    reharmonize_progression, solve_voice_leading, write_midi_file,
};
use forge_vocal::{
    VocalScale, generate_vocal_harmony_stack, shift_vocal_pitch_formant_preserved,
    track_vocal_pitch,
};
use forge_worker::{WorkerBackend, separate_stems};
use sandbox::PathSandbox;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::any::Any;
use std::io::{self, BufRead, Write};
use std::panic::{self, AssertUnwindSafe};
use std::path::PathBuf;

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

/// Result of a successful tool call. `success` is false when the tool ran but its outcome
/// is a failure the caller must act on (e.g. a delivery that is not `ready`).
struct ToolOutput {
    value: Value,
    success: bool,
}

impl From<Value> for ToolOutput {
    fn from(value: Value) -> Self {
        Self {
            value,
            success: true,
        }
    }
}

fn main() {
    let sandbox = match PathSandbox::from_env() {
        Ok(sb) => sb,
        Err(e) => {
            eprintln!("forge-mcp: cannot set up the file sandbox: {e}");
            std::process::exit(1);
        }
    };
    let roots: Vec<String> = sandbox.roots().map(|r| r.display().to_string()).collect();
    eprintln!("forge-mcp: file access confined to {}", roots.join(", "));

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    if let Err(e) = serve(stdin.lock(), &mut stdout, &sandbox) {
        eprintln!("forge-mcp: stdio failure: {e}");
        std::process::exit(1);
    }
}

/// Reads newline-delimited JSON-RPC requests until EOF, answering each on `output`.
/// A malformed line (bad JSON or invalid UTF-8) gets a -32700 reply; the loop keeps going.
fn serve<R: BufRead, W: Write>(
    mut input: R,
    output: &mut W,
    sandbox: &PathSandbox,
) -> io::Result<()> {
    let mut line = Vec::new();
    loop {
        line.clear();
        if input.read_until(b'\n', &mut line)? == 0 {
            return Ok(());
        }
        let response = match std::str::from_utf8(&line) {
            Ok(text) => process_line(text, sandbox),
            Err(e) => Some(parse_error(format!("Parse error: invalid UTF-8: {e}"))),
        };
        if let Some(resp) = response {
            let serialized = serde_json::to_string(&resp).map_err(io::Error::other)?;
            writeln!(output, "{serialized}")?;
            output.flush()?;
        }
    }
}

/// Handles one request line. Returns `None` for blank lines and notifications (no `id`).
fn process_line(text: &str, sandbox: &PathSandbox) -> Option<JsonRpcResponse> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    let req: JsonRpcRequest = match serde_json::from_str(trimmed) {
        Ok(r) => r,
        Err(e) => return Some(parse_error(format!("Parse error: {e}"))),
    };
    let id = req.id?;
    Some(handle_request(id, &req.method, &req.params, sandbox))
}

fn parse_error(message: String) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0",
        id: Value::Null,
        result: None,
        error: Some(JsonRpcError {
            code: -32700,
            message,
        }),
    }
}

fn panic_message(payload: &(dyn Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("unknown panic")
}

fn handle_request(
    id: Value,
    method: &str,
    params: &Value,
    sandbox: &PathSandbox,
) -> JsonRpcResponse {
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

            // A panic in DSP code must fail this call, not take down the server.
            let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
                dispatch_tool(tool_name, &args, sandbox)
            }));
            let (text, is_error) = match outcome {
                Ok(Ok(out)) => (
                    serde_json::to_string_pretty(&out.value).unwrap_or_default(),
                    !out.success,
                ),
                Ok(Err(err)) => (format!("ForgeAudio Error: {err}"), true),
                Err(payload) => (
                    format!(
                        "ForgeAudio internal error: tool '{tool_name}' panicked: {}",
                        panic_message(payload.as_ref())
                    ),
                    true,
                ),
            };
            JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: Some(json!({
                    "content": [{ "type": "text", "text": text }],
                    "isError": is_error
                })),
                error: None,
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
            "name": "forge_deliver",
            "description": "Deliver a mix to a distribution target in one call: audit, repair (DC, clipping, hum), de-fizz, optional reference/genre tone match, loudness-normalize and true-peak limit, then re-audit. Returns before/after BS.1770-4 audits and explicit pass/fail checks (loudness within 1 LU, true peak under ceiling, no clipping, sample rate >= 44.1 kHz). Writes a 24-bit WAV and a .delivery.json report next to it.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input": { "type": "string", "description": "Input mix or stem (WAV)" },
                    "output": { "type": "string", "description": "Output master WAV path" },
                    "target": { "type": "string", "enum": delivery_target_names(), "default": "spotify" },
                    "target_lufs": { "type": "number", "description": "Required when target is 'custom'" },
                    "ceiling_dbtp": { "type": "number", "description": "Required when target is 'custom'" },
                    "repair": { "type": "boolean", "default": true },
                    "defizz": { "type": "boolean", "default": true },
                    "vocal": { "type": "boolean", "default": false, "description": "Vocal clean-up; use on vocal stems only" },
                    "reference": { "type": "string", "description": "Reference track WAV for 18-band tone match" },
                    "genre": { "type": "string", "description": "Genre tonal-balance target (pop, hiphop, neosoul, edm, rock, jazz); ignored when reference is set" }
                },
                "required": ["input", "output"]
            }
        }),
        json!({
            "name": "forge_ingest_to_reaper",
            "description": "Full autonomous pipeline: separate stems, clean & de-fizz audio, extract drum MIDI, build voice-led chords, deliver a verified master for the chosen target, install JSFX plugins, and generate a REAPER 7.82 .RPP project.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input": { "type": "string" },
                    "output_dir": { "type": "string" },
                    "target": { "type": "string", "enum": delivery_target_names(), "default": "spotify" },
                    "target_lufs": { "type": "number", "description": "Required when target is 'custom'" },
                    "ceiling_dbtp": { "type": "number", "description": "Required when target is 'custom'" },
                    "bpm": { "type": "number", "default": 120.0 },
                    "scale": { "type": "string", "default": "D minor" },
                    "chords": { "type": "string", "default": "Dm9 Bbmaj7 Gm9 A7b13" }
                },
                "required": ["input", "output_dir"]
            }
        }),
    ]
}

fn delivery_target_names() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = DeliveryTarget::NAMED.iter().map(|t| t.name()).collect();
    names.push("custom");
    names
}

/// Reads `target` (default `spotify`), or `target_lufs` + `ceiling_dbtp` for `custom`.
fn delivery_target_from_args(args: &Value) -> Result<DeliveryTarget, ForgeError> {
    let raw = args
        .get("target")
        .and_then(Value::as_str)
        .unwrap_or("spotify");
    if !raw.eq_ignore_ascii_case("custom") {
        return DeliveryTarget::parse(raw);
    }
    let lufs = args.get("target_lufs").and_then(Value::as_f64);
    let ceiling = args.get("ceiling_dbtp").and_then(Value::as_f64);
    match (lufs, ceiling) {
        (Some(lufs), Some(ceiling)) => DeliveryTarget::custom(lufs, ceiling),
        _ => Err(ForgeError::InvalidParameter(
            "target 'custom' requires 'target_lufs' and 'ceiling_dbtp'".into(),
        )),
    }
}

/// Reads a required path argument and resolves it inside the sandbox.
fn path_arg(args: &Value, key: &str, sandbox: &PathSandbox) -> Result<PathBuf, ForgeError> {
    let raw = args
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| ForgeError::InvalidParameter(format!("Missing '{key}'")))?;
    sandbox.resolve(raw)
}

/// Reads an optional path argument and resolves it inside the sandbox.
fn optional_path_arg(
    args: &Value,
    key: &str,
    sandbox: &PathSandbox,
) -> Result<Option<PathBuf>, ForgeError> {
    args.get(key)
        .and_then(Value::as_str)
        .map(|raw| sandbox.resolve(raw))
        .transpose()
}

/// Runs one tool. All path arguments are resolved through `sandbox` before any file I/O.
fn dispatch_tool(
    name: &str,
    args: &Value,
    sandbox: &PathSandbox,
) -> Result<ToolOutput, ForgeError> {
    match name {
        "forge_deliver" => {
            let input = path_arg(args, "input", sandbox)?;
            let output = path_arg(args, "output", sandbox)?;
            let reference = optional_path_arg(args, "reference", sandbox)?;
            let flag = |key: &str, default: bool| {
                args.get(key).and_then(Value::as_bool).unwrap_or(default)
            };
            let mut options = DeliveryOptions::for_target(delivery_target_from_args(args)?);
            options.repair = flag("repair", true);
            options.defizz = flag("defizz", true);
            options.vocal = flag("vocal", false);
            if let Some(reference) = reference {
                options.tone_match = Some(ToneMatch::Reference(AudioBuffer::load(&reference)?));
            } else if let Some(genre) = args.get("genre").and_then(Value::as_str) {
                options.tone_match = Some(ToneMatch::Genre(GenreTonalTarget::parse(genre)));
            }

            let buf = AudioBuffer::load(&input)?;
            let (master, report) = deliver(&buf, &options)?;
            if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent)?;
            }
            master.save_wav(&output, 24)?;
            let report_path = output.with_extension("delivery.json");
            std::fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;
            Ok(ToolOutput {
                success: report.ready,
                value: json!({
                    "output": output,
                    "report_path": report_path,
                    "report": report
                }),
            })
        }
        "forge_audit_audio" => {
            let path = path_arg(args, "path", sandbox)?;
            let buf = AudioBuffer::load(&path)?;
            let report = audit_audio(&buf);
            Ok(serde_json::to_value(report)?.into())
        }
        "forge_restore_and_defizz" => {
            let input = path_arg(args, "input", sandbox)?;
            let output = path_arg(args, "output", sandbox)?;
            let vocal = args.get("vocal").and_then(Value::as_bool).unwrap_or(false);
            let depth = args
                .get("defizz_depth_db")
                .and_then(Value::as_f64)
                .unwrap_or(9.0);
            let sensitivity = args
                .get("sensitivity_db")
                .and_then(Value::as_f64)
                .unwrap_or(3.5);

            let raw = AudioBuffer::load(&input)?;
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
            cleaned.save_wav(&output, 24)?;
            Ok(json!({
                "output": output,
                "before_fizz_index": before.ai_fizz_index,
                "after_fizz_index": after.ai_fizz_index,
                "after_lufs": after.integrated_lufs
            })
            .into())
        }
        "forge_spectral_unmask" => {
            let target = path_arg(args, "target", sandbox)?;
            let sidechain = path_arg(args, "sidechain", sandbox)?;
            let output = path_arg(args, "output", sandbox)?;
            let kick_bass = args
                .get("kick_bass_mode")
                .and_then(Value::as_bool)
                .unwrap_or(false);

            let target_buf = AudioBuffer::load(&target)?;
            let sc_buf = AudioBuffer::load(&sidechain)?;
            let cfg = if kick_bass {
                UnmaskConfig::kick_over_bass()
            } else {
                UnmaskConfig::vocal_over_instruments()
            };
            let unmasked = apply_spectral_unmask(&target_buf, &sc_buf, &cfg)?;
            unmasked.save_wav(&output, 24)?;
            Ok(json!({ "output": output, "kick_bass_mode": kick_bass }).into())
        }
        "forge_phase_align" => {
            let reference = path_arg(args, "reference", sandbox)?;
            let target = path_arg(args, "target", sandbox)?;
            let output = path_arg(args, "output", sandbox)?;
            let vocal_dtw = args
                .get("vocal_dtw")
                .and_then(Value::as_bool)
                .unwrap_or(false);

            let ref_buf = AudioBuffer::load(&reference)?;
            let tgt_buf = AudioBuffer::load(&target)?;
            if vocal_dtw {
                let (warped, markers) = align_vocal_double_dtw(&ref_buf, &tgt_buf, 120.0)?;
                warped.save_wav(&output, 24)?;
                Ok(json!({
                    "mode": "vocal_dtw",
                    "output": output,
                    "stretch_markers_count": markers.len()
                })
                .into())
            } else {
                let (aligned, info) = align_phase_gcc_phat(&ref_buf, &tgt_buf, 35.0)?;
                aligned.save_wav(&output, 24)?;
                Ok(serde_json::to_value(info)?.into())
            }
        }
        "forge_vocal_suite" => {
            let input = path_arg(args, "input", sandbox)?;
            let output_dir = path_arg(args, "output_dir", sandbox)?;
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

            let raw = AudioBuffer::load(&input)?;
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
            })
            .into())
        }
        "forge_music_theory_midi" => {
            let chords_str = args
                .get("chords")
                .and_then(Value::as_str)
                .ok_or_else(|| ForgeError::InvalidParameter("Missing 'chords'".into()))?;
            let output_midi = path_arg(args, "output_midi", sandbox)?;
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
                &output_midi,
                bpm,
                480,
                &[("Voiced_Chords", &chord_notes), ("Sub_Bass", &bass_notes)],
            )?;
            let abc = export_abc_score("ForgeAudio Progression", "C", bpm.round() as u32, &reharm);
            Ok(json!({
                "output_midi": output_midi,
                "abc_notation": abc,
                "reharmonized_chords": reharm
            })
            .into())
        }
        "forge_master_distrokid" => {
            let input = path_arg(args, "input", sandbox)?;
            // Absolute after resolution, so it can never be read by ffmpeg as an option.
            let out_path = path_arg(args, "output", sandbox)?;
            let reference = optional_path_arg(args, "reference", sandbox)?;
            let target_lufs = args
                .get("target_lufs")
                .and_then(Value::as_f64)
                .unwrap_or(-14.0);
            let ceiling_dbtp = args
                .get("ceiling_dbtp")
                .and_then(Value::as_f64)
                .unwrap_or(-1.0);

            let mut buf = AudioBuffer::load(&input)?;
            if let Some(reference) = reference {
                let ref_buf = AudioBuffer::load(&reference)?;
                let curve = compute_reference_match_eq(&buf, &ref_buf, 4.5);
                apply_18_band_eq(&mut buf, &curve);
            }
            let mastered = normalize_and_limit(&buf, target_lufs, ceiling_dbtp)?;
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
            Ok(serde_json::to_value(report)?.into())
        }
        "forge_ingest_to_reaper" => {
            let input = path_arg(args, "input", sandbox)?;
            let output_dir = path_arg(args, "output_dir", sandbox)?;
            let target = delivery_target_from_args(args)?;
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
            let (mastered, delivery) = deliver(&full_mix, &DeliveryOptions::for_target(target))?;
            let master_path = output_dir.join(format!("Master_{}_24bit.wav", target.name()));
            mastered.save_wav(&master_path, 24)?;
            let delivery_path = master_path.with_extension("delivery.json");
            std::fs::write(&delivery_path, serde_json::to_string_pretty(&delivery)?)?;

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

            Ok(ToolOutput {
                success: delivery.ready,
                value: json!({
                    "rpp_project": rpp_path,
                    "master_wav": master_path,
                    "midi_file": chords_midi_path,
                    "delivery_report": delivery_path,
                    "target": delivery.target,
                    "ready": delivery.ready,
                    "checks": delivery.checks,
                    "master_lufs": delivery.after.integrated_lufs,
                    "master_true_peak_dbtp": delivery.after.true_peak_dbtp,
                    "ai_fizz_index": delivery.after.ai_fizz_index
                }),
            })
        }
        _ => Err(ForgeError::InvalidParameter(format!(
            "Unknown tool: {name}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    use std::path::Path;

    fn sandbox_in(dir: &Path) -> PathSandbox {
        PathSandbox::new([dir.to_path_buf()]).expect("sandbox")
    }

    /// 10 s stereo mix-like signal (two tones plus broadband hash), as in the delivery tests.
    fn write_test_mix(path: &Path, level: f64, sample_rate: u32) {
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
        AudioBuffer::from_channels(vec![ch.clone(), ch], sample_rate)
            .expect("valid buffer")
            .save_wav(path, 24)
            .expect("write fixture");
    }

    /// Runs `tools/call` and returns the MCP result object.
    fn call_tool(sandbox: &PathSandbox, name: &str, arguments: Value) -> Value {
        let resp = handle_request(
            json!(1),
            "tools/call",
            &json!({ "name": name, "arguments": arguments }),
            sandbox,
        );
        assert!(resp.error.is_none(), "unexpected JSON-RPC error");
        resp.result.expect("tool result")
    }

    fn is_error(result: &Value) -> bool {
        result["isError"].as_bool().expect("isError present")
    }

    fn text(result: &Value) -> &str {
        result["content"][0]["text"].as_str().expect("text content")
    }

    #[test]
    fn output_outside_root_is_rejected_before_any_write() {
        let root = tempfile::tempdir().expect("tempdir");
        let elsewhere = tempfile::tempdir().expect("tempdir");
        let sb = sandbox_in(root.path());
        write_test_mix(&root.path().join("mix.wav"), 0.05, 48_000);

        let victim = elsewhere.path().join("victim.wav");
        let result = call_tool(
            &sb,
            "forge_deliver",
            json!({ "input": "mix.wav", "output": victim.to_string_lossy() }),
        );
        assert!(is_error(&result));
        assert!(
            text(&result).contains("outside the allowed roots"),
            "{}",
            text(&result)
        );
        assert!(!victim.exists());
        assert!(!victim.with_extension("delivery.json").exists());
    }

    #[test]
    fn parent_dir_traversal_is_rejected_for_every_path_argument() {
        let root = tempfile::tempdir().expect("tempdir");
        let sb = sandbox_in(root.path());
        let cases = [
            ("forge_audit_audio", json!({ "path": "../secret.wav" })),
            (
                "forge_restore_and_defizz",
                json!({ "input": "in.wav", "output": "../out.wav" }),
            ),
            (
                "forge_spectral_unmask",
                json!({ "target": "t.wav", "sidechain": "../s.wav", "output": "o.wav" }),
            ),
            (
                "forge_phase_align",
                json!({ "reference": "../r.wav", "target": "t.wav", "output": "o.wav" }),
            ),
            (
                "forge_vocal_suite",
                json!({ "input": "v.wav", "output_dir": "stems/../../escape" }),
            ),
            (
                "forge_music_theory_midi",
                json!({ "chords": "Cmaj7", "output_midi": "../x.mid" }),
            ),
            (
                "forge_master_distrokid",
                json!({ "input": "in.wav", "output": "o.wav", "reference": "../r.wav" }),
            ),
            (
                "forge_ingest_to_reaper",
                json!({ "input": "in.wav", "output_dir": "../session" }),
            ),
        ];
        for (tool, args) in cases {
            let result = call_tool(&sb, tool, args);
            assert!(is_error(&result), "{tool} accepted a '..' path");
            assert!(text(&result).contains("'..'"), "{tool}: {}", text(&result));
        }
        assert!(!root.path().join("..").join("session").exists());
    }

    #[test]
    fn not_ready_delivery_reports_is_error() {
        let root = tempfile::tempdir().expect("tempdir");
        let sb = sandbox_in(root.path());
        // 22.05 kHz fails the sample-rate check, so the delivery cannot be `ready`.
        write_test_mix(&root.path().join("lofi.wav"), 0.05, 22_050);

        let result = call_tool(
            &sb,
            "forge_deliver",
            json!({ "input": "lofi.wav", "output": "out/master.wav" }),
        );
        assert!(is_error(&result), "not-ready delivery reported success");
        let body: Value = serde_json::from_str(text(&result)).expect("JSON body");
        assert_eq!(body["report"]["ready"], json!(false));
        assert!(root.path().join("out/master.wav").exists());
    }

    #[test]
    fn ready_delivery_reports_success() {
        let root = tempfile::tempdir().expect("tempdir");
        let sb = sandbox_in(root.path());
        write_test_mix(&root.path().join("mix.wav"), 0.05, 48_000);

        let result = call_tool(
            &sb,
            "forge_deliver",
            json!({ "input": "mix.wav", "output": "master.wav" }),
        );
        assert!(!is_error(&result), "{}", text(&result));
        let body: Value = serde_json::from_str(text(&result)).expect("JSON body");
        assert_eq!(body["report"]["ready"], json!(true));
    }

    #[test]
    fn invalid_utf8_line_gets_parse_error_and_server_keeps_running() {
        let root = tempfile::tempdir().expect("tempdir");
        let sb = sandbox_in(root.path());
        let mut input = b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"\xff\xfe\"}\n".to_vec();
        input.extend_from_slice(b"{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"initialize\"}\n");

        let mut output = Vec::new();
        serve(input.as_slice(), &mut output, &sb).expect("serve");
        let lines: Vec<Value> = String::from_utf8(output)
            .expect("utf-8 output")
            .lines()
            .map(|l| serde_json::from_str(l).expect("JSON line"))
            .collect();

        assert_eq!(lines.len(), 2, "{lines:?}");
        assert_eq!(lines[0]["error"]["code"], json!(-32700));
        assert!(
            lines[0]["error"]["message"]
                .as_str()
                .unwrap()
                .contains("invalid UTF-8")
        );
        assert_eq!(lines[1]["id"], json!(2));
        assert_eq!(lines[1]["result"]["serverInfo"]["name"], json!("forge-mcp"));
    }

    #[test]
    fn malformed_json_and_notifications_do_not_stop_the_loop() {
        let root = tempfile::tempdir().expect("tempdir");
        let sb = sandbox_in(root.path());
        let input = b"not json\n\n{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"tools/list\"}";

        let mut output = Vec::new();
        serve(input.as_slice(), &mut output, &sb).expect("serve");
        let lines: Vec<Value> = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();

        assert_eq!(lines.len(), 2, "{lines:?}");
        assert_eq!(lines[0]["error"]["code"], json!(-32700));
        assert_eq!(lines[1]["id"], json!(7));
        assert!(lines[1]["result"]["tools"].as_array().unwrap().len() >= 9);
    }

    #[test]
    fn panic_payloads_are_reported() {
        let caught = panic::catch_unwind(|| panic!("boom {}", 42)).unwrap_err();
        assert_eq!(panic_message(caught.as_ref()), "boom 42");
        let caught = panic::catch_unwind(|| panic!("static")).unwrap_err();
        assert_eq!(panic_message(caught.as_ref()), "static");
    }
}
