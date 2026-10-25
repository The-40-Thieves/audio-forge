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
    GrooveStyle, NoteEvent, VoicingStyle, export_abc_score, extract_drum_midi_triggers,
    reharmonize_progression, solve_voice_leading, write_midi_file,
};
use forge_vocal::{
    PitchFrame, VocalScale, generate_vocal_harmony_stack, shift_vocal_pitch_formant_preserved,
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
        "tools/call" => JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(call_tool_result(params, sandbox)),
            error: None,
        },
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

/// Builds the MCP `tools/call` result: `{ content, isError }`.
fn call_tool_result(params: &Value, sandbox: &PathSandbox) -> Value {
    let tool_name = params
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let (text, is_error) = run_tool(tool_name, &args, sandbox);
    json!({
        "content": [{ "type": "text", "text": text }],
        "isError": is_error
    })
}

/// Runs a tool and renders its outcome as `(text, is_error)`. A panic in DSP code fails
/// this call instead of taking down the server.
fn run_tool(tool_name: &str, args: &Value, sandbox: &PathSandbox) -> (String, bool) {
    let outcome = panic::catch_unwind(AssertUnwindSafe(|| dispatch_tool(tool_name, args, sandbox)));
    match outcome {
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
    match args.get(key) {
        None | Some(Value::Null) => Err(ForgeError::InvalidParameter(format!("Missing '{key}'"))),
        Some(value) => resolve_path_value(key, value, sandbox),
    }
}

/// Reads an optional path argument and resolves it inside the sandbox. Only an absent or
/// `null` field means "not given"; any other non-string value is rejected.
fn optional_path_arg(
    args: &Value,
    key: &str,
    sandbox: &PathSandbox,
) -> Result<Option<PathBuf>, ForgeError> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => resolve_path_value(key, value, sandbox).map(Some),
    }
}

fn resolve_path_value(
    key: &str,
    value: &Value,
    sandbox: &PathSandbox,
) -> Result<PathBuf, ForgeError> {
    let raw = value
        .as_str()
        .ok_or_else(|| ForgeError::InvalidParameter(format!("'{key}' must be a string path")))?;
    sandbox.resolve(raw)
}

fn str_arg<'a>(args: &'a Value, key: &str) -> Result<&'a str, ForgeError> {
    args.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| ForgeError::InvalidParameter(format!("Missing '{key}'")))
}

fn bool_arg(args: &Value, key: &str, default: bool) -> bool {
    args.get(key).and_then(Value::as_bool).unwrap_or(default)
}

fn f64_arg(args: &Value, key: &str, default: f64) -> f64 {
    args.get(key).and_then(Value::as_f64).unwrap_or(default)
}

fn str_arg_or<'a>(args: &'a Value, key: &str, default: &'a str) -> &'a str {
    args.get(key).and_then(Value::as_str).unwrap_or(default)
}

type ToolHandler = fn(&Value, &PathSandbox) -> Result<ToolOutput, ForgeError>;

/// Runs one tool. Every path argument, and every path derived from one, is resolved
/// through `sandbox` before any file I/O.
fn dispatch_tool(
    name: &str,
    args: &Value,
    sandbox: &PathSandbox,
) -> Result<ToolOutput, ForgeError> {
    let handler: ToolHandler = match name {
        "forge_deliver" => tool_deliver,
        "forge_audit_audio" => tool_audit_audio,
        "forge_restore_and_defizz" => tool_restore_and_defizz,
        "forge_spectral_unmask" => tool_spectral_unmask,
        "forge_phase_align" => tool_phase_align,
        "forge_vocal_suite" => tool_vocal_suite,
        "forge_music_theory_midi" => tool_music_theory_midi,
        "forge_master_distrokid" => tool_master_distrokid,
        "forge_ingest_to_reaper" => tool_ingest_to_reaper,
        _ => {
            return Err(ForgeError::InvalidParameter(format!(
                "Unknown tool: {name}"
            )));
        }
    };
    handler(args, sandbox)
}

fn tool_deliver(args: &Value, sandbox: &PathSandbox) -> Result<ToolOutput, ForgeError> {
    let input = path_arg(args, "input", sandbox)?;
    let output = path_arg(args, "output", sandbox)?;
    let report_path = sandbox.check(&output.with_extension("delivery.json"))?;
    let reference = optional_path_arg(args, "reference", sandbox)?;
    let mut options = DeliveryOptions::for_target(delivery_target_from_args(args)?);
    options.repair = bool_arg(args, "repair", true);
    options.defizz = bool_arg(args, "defizz", true);
    options.vocal = bool_arg(args, "vocal", false);
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

fn tool_audit_audio(args: &Value, sandbox: &PathSandbox) -> Result<ToolOutput, ForgeError> {
    let path = path_arg(args, "path", sandbox)?;
    let buf = AudioBuffer::load(&path)?;
    Ok(serde_json::to_value(audit_audio(&buf))?.into())
}

fn tool_restore_and_defizz(args: &Value, sandbox: &PathSandbox) -> Result<ToolOutput, ForgeError> {
    let input = path_arg(args, "input", sandbox)?;
    let output = path_arg(args, "output", sandbox)?;
    let config = DeFizzConfig {
        sensitivity_db: f64_arg(args, "sensitivity_db", 3.5),
        max_depth_db: f64_arg(args, "defizz_depth_db", 9.0),
        ..DeFizzConfig::default()
    };

    let raw = AudioBuffer::load(&input)?;
    let before = audit_audio(&raw);
    let mut stage1 = run_full_restoration_chain(&raw, &RestorationConfig::default())?;
    if bool_arg(args, "vocal", false) {
        stage1 = clean_vocal_surgical(&stage1, true, true, -12.0)?;
    }
    let cleaned = apply_spectral_defizz(&stage1, &config)?;
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

fn tool_spectral_unmask(args: &Value, sandbox: &PathSandbox) -> Result<ToolOutput, ForgeError> {
    let target = path_arg(args, "target", sandbox)?;
    let sidechain = path_arg(args, "sidechain", sandbox)?;
    let output = path_arg(args, "output", sandbox)?;
    let kick_bass = bool_arg(args, "kick_bass_mode", false);

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

fn tool_phase_align(args: &Value, sandbox: &PathSandbox) -> Result<ToolOutput, ForgeError> {
    let reference = path_arg(args, "reference", sandbox)?;
    let target = path_arg(args, "target", sandbox)?;
    let output = path_arg(args, "output", sandbox)?;

    let ref_buf = AudioBuffer::load(&reference)?;
    let tgt_buf = AudioBuffer::load(&target)?;
    if bool_arg(args, "vocal_dtw", false) {
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

fn tool_vocal_suite(args: &Value, sandbox: &PathSandbox) -> Result<ToolOutput, ForgeError> {
    let input = path_arg(args, "input", sandbox)?;
    let output_dir = path_arg(args, "output_dir", sandbox)?;
    let tuned_path = sandbox.check(&output_dir.join("Vocal_Lead_Tuned_Clean.wav"))?;
    let harmony_paths = [
        sandbox.check(&output_dir.join("Vocal_Harmony_3rd_Above.wav"))?,
        sandbox.check(&output_dir.join("Vocal_Harmony_3rd_Below.wav"))?,
        sandbox.check(&output_dir.join("Vocal_Formant_Double.wav"))?,
    ];
    let scale = VocalScale::parse(str_arg_or(args, "scale", "D minor"))?;
    let retune = f64_arg(args, "retune_strength", 0.85);
    std::fs::create_dir_all(&output_dir)?;

    let raw = AudioBuffer::load(&input)?;
    let clean = clean_vocal_surgical(&raw, true, true, -12.0)?;
    let contour = track_vocal_pitch(&clean, &scale, retune);
    let corr_curve: Vec<(f64, f64)> = contour
        .iter()
        .map(|f| (f.time_secs, f.correction_semitones))
        .collect();
    let tuned = shift_vocal_pitch_formant_preserved(&clean, &corr_curve, 0.0)?;
    tuned.save_wav(&tuned_path, 24)?;

    let mut harmonies = Vec::new();
    if bool_arg(args, "generate_harmonies", true) {
        let (h_up, h_down, h_dbl) = generate_vocal_harmony_stack(&tuned, &scale)?;
        for (buf, path) in [h_up, h_down, h_dbl].iter().zip(&harmony_paths) {
            buf.save_wav(path, 24)?;
            harmonies.push(path.to_string_lossy().into_owned());
        }
    }
    Ok(json!({
        "tuned_vocal": tuned_path,
        "harmonies": harmonies,
        "frames_tracked": contour.len()
    })
    .into())
}

/// Reharmonized chord symbols, then the voiced chord notes and the bass line.
type VoicedChords = (Vec<String>, Vec<NoteEvent>, Vec<NoteEvent>);

/// Reharmonizes a whitespace-separated chord list and voices it with the neo-soul groove.
fn voice_chords(chords: &str) -> Result<VoicedChords, ForgeError> {
    let raw_chords: Vec<String> = chords.split_whitespace().map(ToString::to_string).collect();
    let reharm = reharmonize_progression(&raw_chords, VoicingStyle::NeoSoulRootless);
    let (chord_notes, bass_notes) = solve_voice_leading(
        &reharm,
        VoicingStyle::NeoSoulRootless,
        GrooveStyle::NeoSoulDilla,
        480,
        4,
    )?;
    Ok((reharm, chord_notes, bass_notes))
}

fn tool_music_theory_midi(args: &Value, sandbox: &PathSandbox) -> Result<ToolOutput, ForgeError> {
    let chords = str_arg(args, "chords")?;
    let output_midi = path_arg(args, "output_midi", sandbox)?;
    let bpm = f64_arg(args, "bpm", 92.0);
    let (reharm, chord_notes, bass_notes) = voice_chords(chords)?;
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

fn tool_master_distrokid(args: &Value, sandbox: &PathSandbox) -> Result<ToolOutput, ForgeError> {
    let input = path_arg(args, "input", sandbox)?;
    // Absolute after resolution, so it can never be read by ffmpeg as an option.
    let out_path = path_arg(args, "output", sandbox)?;
    let tmp = sandbox.check(&out_path.with_extension("pre_tag.wav"))?;
    let reference = optional_path_arg(args, "reference", sandbox)?;

    let mut buf = AudioBuffer::load(&input)?;
    if let Some(reference) = reference {
        let ref_buf = AudioBuffer::load(&reference)?;
        let curve = compute_reference_match_eq(&buf, &ref_buf, 4.5);
        apply_18_band_eq(&mut buf, &curve);
    }
    let mastered = normalize_and_limit(
        &buf,
        f64_arg(args, "target_lufs", -14.0),
        f64_arg(args, "ceiling_dbtp", -1.0),
    )?;
    if let Some(raw_isrc) = args.get("isrc").and_then(Value::as_str) {
        let meta = ReleaseMetadata {
            title: str_arg_or(args, "title", "Untitled").into(),
            artist: str_arg_or(args, "artist", "ForgeAudio").into(),
            album: "Single".into(),
            isrc: validate_isrc(raw_isrc)?,
            bpm: None,
            key: None,
            year: Some(2026),
        };
        mastered.save_wav(&tmp, 24)?;
        write_distrokid_metadata(&tmp, &out_path, &meta)?;
        let _ = std::fs::remove_file(&tmp);
    } else {
        mastered.save_wav(&out_path, 24)?;
    }
    Ok(serde_json::to_value(audit_audio(&mastered))?.into())
}

/// Every destination `forge_ingest_to_reaper` writes, each checked against the sandbox
/// before any work starts.
struct IngestPaths {
    output_dir: PathBuf,
    stems_dir: PathBuf,
    midi_dir: PathBuf,
    drum_midi: PathBuf,
    chords_midi: PathBuf,
    master: PathBuf,
    delivery: PathBuf,
    jsfx_dir: PathBuf,
    rpp: PathBuf,
}

impl IngestPaths {
    fn new(
        output_dir: PathBuf,
        target: DeliveryTarget,
        sandbox: &PathSandbox,
    ) -> Result<Self, ForgeError> {
        let check = |rel: &str| sandbox.check(&output_dir.join(rel));
        let master = check(&format!("Master_{}_24bit.wav", target.name()))?;
        Ok(Self {
            stems_dir: check("stems")?,
            midi_dir: check("midi")?,
            drum_midi: check("midi/01_Drum_Triggers.mid")?,
            chords_midi: check("midi/02_Voiced_Chords.mid")?,
            delivery: sandbox.check(&master.with_extension("delivery.json"))?,
            master,
            jsfx_dir: check("Effects")?,
            rpp: check("ForgeAudio_Session.rpp")?,
            output_dir,
        })
    }
}

fn tool_ingest_to_reaper(args: &Value, sandbox: &PathSandbox) -> Result<ToolOutput, ForgeError> {
    let input = path_arg(args, "input", sandbox)?;
    let target = delivery_target_from_args(args)?;
    let paths = IngestPaths::new(path_arg(args, "output_dir", sandbox)?, target, sandbox)?;
    let bpm = f64_arg(args, "bpm", 120.0);
    let scale = VocalScale::parse(str_arg_or(args, "scale", "D minor"))?;
    let chords = str_arg_or(args, "chords", "Dm9 Bbmaj7 Gm9 A7b13");

    std::fs::create_dir_all(&paths.output_dir)?;
    std::fs::create_dir_all(&paths.stems_dir)?;
    std::fs::create_dir_all(&paths.midi_dir)?;

    let bundle = separate_stems(&input, &paths.stems_dir, &WorkerBackend::Auto)?;
    let vocal_pitch_frames = match bundle.vocals {
        Some(ref voc_path) => clean_ingested_vocal(&sandbox.check(voc_path)?, &scale)?,
        None => Vec::new(),
    };

    let drum_buf = AudioBuffer::load(bundle.drums.as_ref().unwrap_or(&input))?;
    let duration_secs = drum_buf.duration_secs();
    let (kick_notes, snare_notes, hat_notes) = extract_drum_midi_triggers(&drum_buf, bpm, 480);
    write_midi_file(
        &paths.drum_midi,
        bpm,
        480,
        &[
            ("Kick_Trigger_C1", &kick_notes),
            ("Snare_Trigger_D1", &snare_notes),
            ("HiHat_Trigger_F#1", &hat_notes),
        ],
    )?;

    let (_, chord_notes, bass_notes) = voice_chords(chords)?;
    write_midi_file(
        &paths.chords_midi,
        bpm,
        480,
        &[("Voiced_Chords", &chord_notes), ("Sub_Bass", &bass_notes)],
    )?;

    let full_mix = AudioBuffer::load(&input)?;
    let (mastered, delivery) = deliver(&full_mix, &DeliveryOptions::for_target(target))?;
    mastered.save_wav(&paths.master, 24)?;
    std::fs::write(&paths.delivery, serde_json::to_string_pretty(&delivery)?)?;

    let _ = install_jsfx_plugins(Some(&paths.jsfx_dir))?;
    let session = SessionSpec {
        bpm,
        sample_rate: full_mix.sample_rate,
        duration_secs,
        vocal: bundle.vocals,
        vocal_pitch_frames,
        chords_midi: paths.chords_midi.clone(),
    };
    build_session(session).save_rpp(&paths.rpp)?;

    Ok(ToolOutput {
        success: delivery.ready,
        value: json!({
            "rpp_project": paths.rpp,
            "master_wav": paths.master,
            "midi_file": paths.chords_midi,
            "delivery_report": paths.delivery,
            "target": delivery.target,
            "ready": delivery.ready,
            "checks": delivery.checks,
            "master_lufs": delivery.after.integrated_lufs,
            "master_true_peak_dbtp": delivery.after.true_peak_dbtp,
            "ai_fizz_index": delivery.after.ai_fizz_index
        }),
    })
}

/// Cleans and de-fizzes the separated vocal in place; returns its pitch contour.
fn clean_ingested_vocal(
    voc_path: &std::path::Path,
    scale: &VocalScale,
) -> Result<Vec<PitchFrame>, ForgeError> {
    let voc_buf = AudioBuffer::load(voc_path)?;
    let voc_clean = clean_vocal_surgical(&voc_buf, true, true, -10.0)?;
    let voc_defizz = apply_spectral_defizz(&voc_clean, &DeFizzConfig::default())?;
    let frames = track_vocal_pitch(&voc_defizz, scale, 0.85);
    voc_defizz.save_wav(voc_path, 24)?;
    Ok(frames)
}

/// Inputs for the generated REAPER session.
struct SessionSpec {
    bpm: f64,
    sample_rate: u32,
    duration_secs: f64,
    vocal: Option<PathBuf>,
    vocal_pitch_frames: Vec<PitchFrame>,
    chords_midi: PathBuf,
}

fn build_session(spec: SessionSpec) -> ReaperProjectBuilder {
    let mut rpp =
        ReaperProjectBuilder::new("ForgeAudio Studio Session", spec.bpm, spec.sample_rate);
    rpp.add_default_song_structure_markers();
    rpp.tracks.push(ReaperTrackSpec {
        name: "01 Lead Vocal".to_string(),
        file_path: spec.vocal,
        is_midi: false,
        duration_secs: spec.duration_secs,
        volume_lin: 0.85,
        pan: 0.0,
        color: 0x0155_88FF,
        jsfx_inserts: vec![
            "Forge_Vocal_Surgical.jsfx".to_string(),
            "Forge_OpenSoothe_DeFizz.jsfx".to_string(),
        ],
        sidechain_from_track: None,
        stretch_markers: Vec::new(),
        pitch_envelope: spec.vocal_pitch_frames,
    });
    rpp.tracks.push(ReaperTrackSpec {
        name: "02 MIDI Chords & Triggers".to_string(),
        file_path: Some(spec.chords_midi),
        is_midi: true,
        duration_secs: spec.duration_secs,
        volume_lin: 0.85,
        pan: 0.0,
        color: 0x01FF_CC44,
        jsfx_inserts: Vec::new(),
        sidechain_from_track: None,
        stretch_markers: Vec::new(),
        pitch_envelope: Vec::new(),
    });
    rpp
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
    fn non_string_path_arguments_are_rejected() {
        let root = tempfile::tempdir().expect("tempdir");
        let sb = sandbox_in(root.path());
        write_test_mix(&root.path().join("mix.wav"), 0.05, 48_000);

        let result = call_tool(
            &sb,
            "forge_deliver",
            json!({ "input": "mix.wav", "output": "master.wav", "reference": 5 }),
        );
        assert!(is_error(&result));
        assert!(
            text(&result).contains("'reference' must be a string path"),
            "{}",
            text(&result)
        );
        assert!(!root.path().join("master.wav").exists());

        let result = call_tool(&sb, "forge_audit_audio", json!({ "path": ["mix.wav"] }));
        assert!(text(&result).contains("'path' must be a string path"));

        // An explicit null optional is treated as absent.
        let result = call_tool(
            &sb,
            "forge_deliver",
            json!({ "input": "mix.wav", "output": "master.wav", "reference": null }),
        );
        assert!(!is_error(&result), "{}", text(&result));
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_report_path_cannot_redirect_the_sidecar_write() {
        let root = tempfile::tempdir().expect("tempdir");
        let elsewhere = tempfile::tempdir().expect("tempdir");
        let sb = sandbox_in(root.path());
        write_test_mix(&root.path().join("mix.wav"), 0.05, 48_000);
        let victim = elsewhere.path().join("victim.json");
        std::fs::write(&victim, "keep").expect("write victim");
        std::os::unix::fs::symlink(&victim, root.path().join("master.delivery.json"))
            .expect("symlink");

        let result = call_tool(
            &sb,
            "forge_deliver",
            json!({ "input": "mix.wav", "output": "master.wav" }),
        );
        assert!(is_error(&result));
        assert_eq!(std::fs::read_to_string(&victim).unwrap(), "keep");
        assert!(!root.path().join("master.wav").exists());
    }

    #[test]
    fn panic_payloads_are_reported() {
        let caught = panic::catch_unwind(|| panic!("boom {}", 42)).unwrap_err();
        assert_eq!(panic_message(caught.as_ref()), "boom 42");
        let caught = panic::catch_unwind(|| panic!("static")).unwrap_err();
        assert_eq!(panic_message(caught.as_ref()), "static");
    }
}
