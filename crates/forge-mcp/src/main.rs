//! # `forge-mcp` — Model Context Protocol (MCP) JSON-RPC 2.0 Server
//!
//! Zero-overhead native Rust stdio MCP server exposing all ForgeAudio DSP,
//! music theory, vocal surgery, GPU worker, and REAPER 7.82 session tools to
//! AI agents (Antigravity, Claude Code, Cursor, REAPER AI scripts).
//!
//! File access is confined: every path argument must resolve inside the roots listed in
//! `FORGE_MCP_ROOT` (default: the directory the server was launched from). See [`sandbox`].
//! Files are then opened through handles anchored at those roots; see [`anchored`] and,
//! for the FFmpeg and worker steps that still take path names, [`external`].

mod anchored;
mod external;
mod sandbox;
#[cfg(all(test, unix))]
mod toctou_tests;

use forge_core::{AudioBuffer, ForgeError, StemBundle};
use forge_dsp::{
    DeFizzConfig, DeliveryOptions, DeliveryTarget, DitherMode, GenreTonalTarget, LimiterStyle,
    RestorationConfig, ToneMatch, UnmaskConfig, align_phase_gcc_phat, align_vocal_double_dtw,
    apply_18_band_eq, apply_spectral_defizz, apply_spectral_unmask, audit_audio,
    clean_vocal_surgical, compute_reference_match_eq, deliver, normalize_and_limit,
    run_full_restoration_chain,
};
use forge_reaper::{
    ReaperProjectBuilder, ReaperTrackSpec, ReleaseMetadata, install_jsfx_plugins, jsfx_install_dir,
    reaper_resource_dir, validate_isrc,
};
use forge_theory::{
    GrooveStyle, NoteEvent, VoicingStyle, export_abc_score, extract_drum_midi_triggers,
    reharmonize_progression, solve_voice_leading, validate_bpm,
};
use forge_vocal::{
    PitchFrame, VocalScale, generate_vocal_harmony_stack, shift_vocal_pitch_formant_preserved,
    track_vocal_pitch,
};
use forge_worker::SeparationConfig;
use sandbox::PathSandbox;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::any::Any;
use std::io::{self, BufRead, Write};
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::time::Duration;

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

const MAX_JSONRPC_LINE_BYTES: u64 = 4 * 1024 * 1024;

/// Reads newline-delimited JSON-RPC requests until EOF, answering each on `output`.
/// A malformed line (bad JSON, invalid UTF-8, or >4 MiB) gets a -32700 reply; the loop keeps going.
fn serve<R: BufRead, W: Write>(
    mut input: R,
    output: &mut W,
    sandbox: &PathSandbox,
) -> io::Result<()> {
    let mut line = Vec::new();
    loop {
        line.clear();
        let n =
            io::Read::take(&mut input, MAX_JSONRPC_LINE_BYTES + 1).read_until(b'\n', &mut line)?;
        if n == 0 {
            return Ok(());
        }
        let response = if (n as u64) > MAX_JSONRPC_LINE_BYTES && !line.ends_with(b"\n") {
            // Drain the remainder of the oversized line safely without buffering it in RAM
            let mut discard = [0_u8; 4096];
            loop {
                let buf = input.fill_buf()?;
                if buf.is_empty() {
                    break;
                }
                if let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                    input.consume(pos + 1);
                    break;
                }
                let len = buf.len().min(discard.len());
                discard[..len].copy_from_slice(&buf[..len]);
                input.consume(len);
            }
            Some(parse_error(format!(
                "Parse error: request line exceeds {MAX_JSONRPC_LINE_BYTES} byte limit"
            )))
        } else {
            match std::str::from_utf8(&line) {
                Ok(text) => process_line(text, sandbox),
                Err(e) => Some(parse_error(format!("Parse error: invalid UTF-8: {e}"))),
            }
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
                    "sensitivity_db": { "type": "number", "default": 3.5 },
                    "tilt": { "type": "number", "default": 0.0, "description": "Spectral tilt in dB/oct (-6.0..=6.0)" },
                    "stereo_link": { "type": "number", "default": 1.0, "description": "Stereo link ratio (0.0..=1.0)" },
                    "mid_side": { "type": "number", "default": 0.0, "description": "Mid/Side balance (-1.0=Mid only, 0.0=linked, +1.0=Side only)" },
                    "delta": { "type": "boolean", "default": false, "description": "Audition only the removed spectral resonance delta" }
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
                    "kick_bass_mode": { "type": "boolean", "default": false },
                    "mid_side": { "type": "number", "default": 0.0, "description": "Mid/Side balance (-1.0=Mid only, 0.0=linked, +1.0=Side only)" },
                    "delta": { "type": "boolean", "default": false, "description": "Audition only the carved sidechain delta" }
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
                    "genre": { "type": "string", "description": "Genre tonal-balance target (pop, hiphop, neosoul, edm, rock, jazz); ignored when reference is set" },
                    "limiter_style": { "type": "string", "enum": ["transparent", "punchy", "dynamic", "aggressive"], "default": "transparent" },
                    "channel_link": { "type": "number", "default": 1.0, "description": "Stereo limiter channel link ratio (0.0..=1.0)" },
                    "dither": { "type": "string", "enum": ["none", "tpdf", "noise-shaped"], "default": "none" },
                    "dither_bits": { "type": "integer", "default": 24, "description": "Target bit depth for dither (8..=24)" }
                },
                "required": ["input", "output"]
            }
        }),
        json!({
            "name": "forge_neural_denoise",
            "description": "Run neural audio denoising via DeepFilterNet3 or Mel-Band-RoFormer on the local Python GPU worker, confined to the MCP path sandbox.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input": { "type": "string", "description": "Input audio file" },
                    "output_dir": { "type": "string", "description": "Output directory for denoised audio" },
                    "backend": { "type": "string", "enum": ["auto", "deepfilternet", "deepfilternet3", "mel-band-roformer", "roformer"], "default": "auto" },
                    "attenuation_limit_db": { "type": "number", "description": "Optional attenuation limit in dB (0..=100, DeepFilterNet3)" },
                    "timeout_secs": { "type": "integer", "default": 300, "description": "Worker timeout in seconds (1..=3600)" }
                },
                "required": ["input", "output_dir"]
            }
        }),
        json!({
            "name": "forge_transcribe_midi",
            "description": "Run polyphonic audio-to-MIDI transcription via basic-pitch on the local Python GPU worker, confined to the MCP path sandbox.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input": { "type": "string", "description": "Input audio file" },
                    "output_dir": { "type": "string", "description": "Output directory for transcribed MIDI" },
                    "timeout_secs": { "type": "integer", "default": 300, "description": "Worker timeout in seconds (1..=3600)" }
                },
                "required": ["input", "output_dir"]
            }
        }),
        json!({
            "name": "forge_separate_stems",
            "description": "Studio-grade neural stem separation (BS-RoFormer 1297 / Mel-Band-RoFormer cascade, 2/4/6 stems + optional BS-RoFormer de-reverb & 5-piece MDX23C DrumSep) with pure-Rust DSP fallback, confined to the MCP path sandbox.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input": { "type": "string", "description": "Input audio file" },
                    "output_dir": { "type": "string", "description": "Output directory for separated WAV stems" },
                    "stems": { "type": "integer", "enum": [2, 4, 6], "default": 6, "description": "Primary stem count (2, 4, or 6)" },
                    "quality": { "type": "string", "enum": ["fast", "studio", "ultra"], "default": "studio", "description": "Separation quality tier" },
                    "dereverb": { "type": "boolean", "default": true, "description": "Run BS-RoFormer de-reverb on isolated vocals" },
                    "drum_sep": { "type": "boolean", "default": false, "description": "Also split isolated drums into Kick, Snare, Toms, Hi-Hat, and Cymbals" }
                },
                "required": ["input", "output_dir"]
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
                    "chords": { "type": "string", "default": "Dm9 Bbmaj7 Gm9 A7b13" },
                    "require_neural": { "type": "boolean", "default": false, "description": "Fail instead of falling back to the DSP stem split when no GPU or Colab worker produces stems" }
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

fn bounded_f64_arg(
    args: &Value,
    key: &str,
    default: f64,
    min: f64,
    max: f64,
) -> Result<f64, ForgeError> {
    let v = match args.get(key) {
        None | Some(Value::Null) => default,
        Some(raw) => raw
            .as_f64()
            .ok_or_else(|| ForgeError::InvalidParameter(format!("'{key}' must be a number")))?,
    };
    if !v.is_finite() || !(min..=max).contains(&v) {
        return Err(ForgeError::InvalidParameter(format!(
            "'{key}' ({v}) must be finite and in {min}..={max}"
        )));
    }
    Ok(v)
}

fn timeout_secs_arg(args: &Value) -> Result<Duration, ForgeError> {
    let secs = match args.get("timeout_secs") {
        None | Some(Value::Null) => 300_u64,
        Some(raw) => raw.as_u64().ok_or_else(|| {
            ForgeError::InvalidParameter("'timeout_secs' must be a positive integer".into())
        })?,
    };
    if !(1..=3600).contains(&secs) {
        return Err(ForgeError::InvalidParameter(format!(
            "'timeout_secs' ({secs}) must be in 1..=3600"
        )));
    }
    Ok(Duration::from_secs(secs))
}

fn parse_limiter_style(raw: &str) -> Result<LimiterStyle, ForgeError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "transparent" => Ok(LimiterStyle::Transparent),
        "punchy" => Ok(LimiterStyle::Punchy),
        "dynamic" => Ok(LimiterStyle::Dynamic),
        "aggressive" => Ok(LimiterStyle::Aggressive),
        other => Err(ForgeError::InvalidParameter(format!(
            "unknown 'limiter_style' '{other}' (expected transparent, punchy, dynamic, or aggressive)"
        ))),
    }
}

fn parse_dither_mode(raw: &str) -> Result<DitherMode, ForgeError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "none" => Ok(DitherMode::None),
        "tpdf" => Ok(DitherMode::Tpdf),
        "noise-shaped" | "noise_shaped" | "shaped" => Ok(DitherMode::NoiseShapedHighPass),
        other => Err(ForgeError::InvalidParameter(format!(
            "unknown 'dither' '{other}' (expected none, tpdf, or noise-shaped)"
        ))),
    }
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
        "forge_neural_denoise" => tool_neural_denoise,
        "forge_transcribe_midi" => tool_transcribe_midi,
        "forge_separate_stems" => tool_separate_stems,
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
    match args.get("limiter_style") {
        None | Some(Value::Null) => {}
        Some(raw) => {
            let style = raw.as_str().ok_or_else(|| {
                ForgeError::InvalidParameter("'limiter_style' must be a string".into())
            })?;
            options.limiter_style = parse_limiter_style(style)?;
        }
    }
    options.channel_link = bounded_f64_arg(args, "channel_link", 1.0, 0.0, 1.0)?;
    let explicit_dither = match args.get("dither") {
        None | Some(Value::Null) => false,
        Some(raw) => {
            let dither = raw
                .as_str()
                .ok_or_else(|| ForgeError::InvalidParameter("'dither' must be a string".into()))?;
            options.dither = parse_dither_mode(dither)?;
            true
        }
    };
    let dither_bits = match args.get("dither_bits") {
        None | Some(Value::Null) => 24_u16,
        Some(raw) => {
            let v = raw.as_u64().ok_or_else(|| {
                ForgeError::InvalidParameter("'dither_bits' must be an integer in 8..=24".into())
            })?;
            if !(8..=24).contains(&v) {
                return Err(ForgeError::InvalidParameter(format!(
                    "'dither_bits' ({v}) must be in 8..=24"
                )));
            }
            v as u16
        }
    };
    options.dither_bits = dither_bits;
    if let Some(reference) = reference {
        options.tone_match = Some(ToneMatch::Reference(sandbox.load_audio(&reference)?));
    } else if let Some(genre) = args.get("genre").and_then(Value::as_str) {
        options.tone_match = Some(ToneMatch::Genre(GenreTonalTarget::parse(genre)));
    }

    let buf = sandbox.load_audio(&input)?;
    let (master, report) = deliver(&buf, &options)?;
    if options.dither != DitherMode::None {
        let pcm_bits = if options.dither_bits <= 16 { 16 } else { 24 };
        sandbox.save_wav_no_dither(&master, &output, pcm_bits)?;
    } else if explicit_dither {
        sandbox.save_wav_no_dither(&master, &output, 24)?;
    } else {
        sandbox.save_wav(&master, &output, 24)?;
    }
    sandbox.write_json(&report_path, &report)?;
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
    let buf = sandbox.load_audio(&path)?;
    Ok(serde_json::to_value(audit_audio(&buf))?.into())
}

fn tool_restore_and_defizz(args: &Value, sandbox: &PathSandbox) -> Result<ToolOutput, ForgeError> {
    let input = path_arg(args, "input", sandbox)?;
    let output = path_arg(args, "output", sandbox)?;
    let config = DeFizzConfig {
        sensitivity_db: f64_arg(args, "sensitivity_db", 3.5),
        max_depth_db: f64_arg(args, "defizz_depth_db", 9.0),
        tilt_db_per_oct: bounded_f64_arg(args, "tilt", 0.0, -6.0, 6.0)?,
        stereo_link: bounded_f64_arg(args, "stereo_link", 1.0, 0.0, 1.0)?,
        mid_side_balance: bounded_f64_arg(args, "mid_side", 0.0, -1.0, 1.0)?,
        delta_listen: bool_arg(args, "delta", false),
        ..DeFizzConfig::default()
    };

    let raw = sandbox.load_audio(&input)?;
    let before = audit_audio(&raw);
    let mut stage1 = run_full_restoration_chain(&raw, &RestorationConfig::default())?;
    if bool_arg(args, "vocal", false) {
        stage1 = clean_vocal_surgical(&stage1, true, true, -12.0)?;
    }
    let cleaned = apply_spectral_defizz(&stage1, &config)?;
    let after = audit_audio(&cleaned);
    sandbox.save_wav(&cleaned, &output, 24)?;
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
    let mid_side = bounded_f64_arg(args, "mid_side", 0.0, -1.0, 1.0)?;
    let delta = bool_arg(args, "delta", false);

    let target_buf = sandbox.load_audio(&target)?;
    let sc_buf = sandbox.load_audio(&sidechain)?;
    let mut cfg = if kick_bass {
        UnmaskConfig::kick_over_bass()
    } else {
        UnmaskConfig::vocal_over_instruments()
    };
    cfg.mid_side_balance = mid_side;
    cfg.delta_listen = delta;
    let unmasked = apply_spectral_unmask(&target_buf, &sc_buf, &cfg)?;
    sandbox.save_wav(&unmasked, &output, 24)?;
    Ok(json!({
        "output": output,
        "kick_bass_mode": kick_bass,
        "mid_side": mid_side,
        "delta": delta
    })
    .into())
}

fn tool_neural_denoise(args: &Value, sandbox: &PathSandbox) -> Result<ToolOutput, ForgeError> {
    let input = path_arg(args, "input", sandbox)?;
    let output_dir = path_arg(args, "output_dir", sandbox)?;
    let raw_backend = match args.get("backend") {
        None | Some(Value::Null) => "auto",
        Some(v) => v
            .as_str()
            .ok_or_else(|| ForgeError::InvalidParameter("'backend' must be a string".into()))?,
    };
    let backend = match raw_backend.trim().to_ascii_lowercase().as_str() {
        "auto" => "auto",
        "deepfilternet" | "deepfilternet3" => "deepfilternet",
        "mel-band-roformer" | "roformer" => "mel-band-roformer",
        other => {
            return Err(ForgeError::InvalidParameter(format!(
                "unknown 'backend' '{other}' (expected auto, deepfilternet, deepfilternet3, mel-band-roformer, or roformer)"
            )));
        }
    };
    let attenuation_limit_db = match args.get("attenuation_limit_db") {
        None | Some(Value::Null) => None,
        Some(_) => Some(bounded_f64_arg(
            args,
            "attenuation_limit_db",
            0.0,
            0.0,
            100.0,
        )?),
    };
    let timeout = timeout_secs_arg(args)?;
    let res = external::neural_denoise_checked(
        sandbox,
        &input,
        &output_dir,
        backend,
        attenuation_limit_db,
        timeout,
    )?;
    Ok(serde_json::to_value(res)?.into())
}

fn tool_transcribe_midi(args: &Value, sandbox: &PathSandbox) -> Result<ToolOutput, ForgeError> {
    let input = path_arg(args, "input", sandbox)?;
    let output_dir = path_arg(args, "output_dir", sandbox)?;
    let timeout = timeout_secs_arg(args)?;
    let res = external::transcribe_midi_checked(sandbox, &input, &output_dir, timeout)?;
    Ok(serde_json::to_value(res)?.into())
}

fn tool_phase_align(args: &Value, sandbox: &PathSandbox) -> Result<ToolOutput, ForgeError> {
    let reference = path_arg(args, "reference", sandbox)?;
    let target = path_arg(args, "target", sandbox)?;
    let output = path_arg(args, "output", sandbox)?;

    let ref_buf = sandbox.load_audio(&reference)?;
    let tgt_buf = sandbox.load_audio(&target)?;
    if bool_arg(args, "vocal_dtw", false) {
        let (warped, markers) = align_vocal_double_dtw(&ref_buf, &tgt_buf, 120.0)?;
        sandbox.save_wav(&warped, &output, 24)?;
        Ok(json!({
            "mode": "vocal_dtw",
            "output": output,
            "stretch_markers_count": markers.len()
        })
        .into())
    } else {
        let (aligned, info) = align_phase_gcc_phat(&ref_buf, &tgt_buf, 35.0)?;
        sandbox.save_wav(&aligned, &output, 24)?;
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
    let retune = f64_arg(args, "retune_strength", 0.85).clamp(0.0, 1.0);
    sandbox.create_dir_all(&output_dir)?;

    let raw = sandbox.load_audio(&input)?;
    let clean = clean_vocal_surgical(&raw, true, true, -12.0)?;
    let contour = track_vocal_pitch(&clean, &scale, retune);
    let pitch_env = forge_vocal::compile_reaper_take_pitch_envelope(&contour);
    let corr_curve: Vec<(f64, f64)> = contour
        .iter()
        .map(|f| (f.time_secs, f.correction_semitones))
        .collect();
    let tuned = shift_vocal_pitch_formant_preserved(&clean, &corr_curve, 0.0)?;
    sandbox.save_wav(&tuned, &tuned_path, 24)?;

    let mut harmonies = Vec::new();
    if bool_arg(args, "generate_harmonies", true) {
        let (h_up, h_down, h_dbl) = generate_vocal_harmony_stack(&tuned, &scale)?;
        for (buf, path) in [h_up, h_down, h_dbl].iter().zip(&harmony_paths) {
            sandbox.save_wav(buf, path, 24)?;
            harmonies.push(path.to_string_lossy().into_owned());
        }
    }
    Ok(json!({
        "tuned_vocal": tuned_path,
        "harmonies": harmonies,
        "frames_tracked": contour.len(),
        "reaper_pitch_envelope": pitch_env
    })
    .into())
}

/// Reharmonized chord symbols, then the voiced chord notes and the bass line.
type VoicedChords = (Vec<String>, Vec<NoteEvent>, Vec<NoteEvent>);

fn parse_voicing_style(style: &str) -> VoicingStyle {
    match style.trim().to_ascii_lowercase().as_str() {
        "jazz" | "jazz-drop2" | "drop2" => VoicingStyle::Drop2,
        "pop" | "triad" | "close" => VoicingStyle::Close,
        "cinematic" => VoicingStyle::CinematicSpread,
        _ => VoicingStyle::NeoSoulRootless,
    }
}

/// Reharmonizes a whitespace-separated chord list and voices it with the selected style.
fn voice_chords(chords: &str, style: VoicingStyle) -> Result<VoicedChords, ForgeError> {
    let raw_chords: Vec<String> = chords.split_whitespace().map(ToString::to_string).collect();
    let reharm = reharmonize_progression(&raw_chords, style);
    let (chord_notes, bass_notes) =
        solve_voice_leading(&reharm, style, GrooveStyle::NeoSoulDilla, 480, 4)?;
    Ok((reharm, chord_notes, bass_notes))
}

fn tool_music_theory_midi(args: &Value, sandbox: &PathSandbox) -> Result<ToolOutput, ForgeError> {
    let chords = str_arg(args, "chords")?;
    let output_midi = path_arg(args, "output_midi", sandbox)?;
    let bpm = validate_bpm(f64_arg(args, "bpm", 92.0))?;
    let style = parse_voicing_style(str_arg_or(args, "style", "neo-soul"));
    let (reharm, chord_notes, bass_notes) = voice_chords(chords, style)?;
    sandbox.write_midi(
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

    let mut buf = sandbox.load_audio(&input)?;
    if let Some(reference) = reference {
        let ref_buf = sandbox.load_audio(&reference)?;
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
        sandbox.save_wav(&mastered, &tmp, 24)?;
        external::tag_master(sandbox, &tmp, &out_path, &meta)?;
    } else {
        sandbox.save_wav(&mastered, &out_path, 24)?;
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
            rpp: check("ForgeAudio_Session.rpp")?,
            output_dir,
        })
    }
}

/// Installs the JSFX into `Effects/ForgeAudio` under REAPER's resource folder, the only
/// place the session's `<JS "ForgeAudio/...">` references resolve. These are fixed files at
/// a fixed location, not a client path, so they are not confined to the sandbox. A missing
/// REAPER install is reported rather than failing the rest of the ingest.
fn install_session_jsfx(resource_dir: Result<PathBuf, ForgeError>) -> Value {
    let installed =
        resource_dir.and_then(|dir| install_jsfx_plugins(Some(&jsfx_install_dir(&dir))));
    match installed {
        Ok(files) => json!({ "installed": true, "files": files }),
        Err(e) => json!({ "installed": false, "error": e.to_string() }),
    }
}

fn tool_ingest_to_reaper(args: &Value, sandbox: &PathSandbox) -> Result<ToolOutput, ForgeError> {
    let input = path_arg(args, "input", sandbox)?;
    let target = delivery_target_from_args(args)?;
    let paths = IngestPaths::new(path_arg(args, "output_dir", sandbox)?, target, sandbox)?;
    let bpm = validate_bpm(f64_arg(args, "bpm", 120.0))?;
    let scale = VocalScale::parse(str_arg_or(args, "scale", "D minor"))?;
    let chords = str_arg_or(args, "chords", "Dm9 Bbmaj7 Gm9 A7b13");

    sandbox.create_dir_all(&paths.output_dir)?;
    sandbox.create_dir_all(&paths.midi_dir)?;

    let require_neural = bool_arg(args, "require_neural", false);
    let (bundle, stem_separation) =
        separate_ingest_stems(sandbox, &input, &paths.stems_dir, require_neural)?;
    let vocal_pitch_frames = ingest_vocal_frames(sandbox, bundle.vocals.as_deref(), &scale)?;

    let drum_buf = sandbox.load_audio(&sandbox.check(bundle.drums.as_ref().unwrap_or(&input))?)?;
    let duration_secs = drum_buf.duration_secs();
    write_ingest_midi(sandbox, &paths, &drum_buf, bpm, chords)?;

    let full_mix = sandbox.load_audio(&input)?;
    let (mastered, delivery) = deliver(&full_mix, &DeliveryOptions::for_target(target))?;
    sandbox.save_wav(&mastered, &paths.master, 24)?;
    sandbox.write_json(&paths.delivery, &delivery)?;

    let session = SessionSpec {
        bpm,
        sample_rate: full_mix.sample_rate,
        duration_secs,
        vocal: bundle.vocals,
        drums: bundle.drums,
        bass: bundle.bass,
        other: bundle.other,
        vocal_pitch_frames,
        chords_midi: paths.chords_midi.clone(),
        drum_midi: paths.drum_midi.clone(),
    };
    sandbox.save_rpp(&build_session(session), &paths.rpp)?;

    Ok(ToolOutput {
        success: delivery.ready,
        value: json!({
            "rpp_project": paths.rpp,
            "master_wav": paths.master,
            "midi_file": paths.chords_midi,
            "delivery_report": paths.delivery,
            "jsfx": install_session_jsfx(reaper_resource_dir()),
            "target": delivery.target,
            "ready": delivery.ready,
            "checks": delivery.checks,
            "master_lufs": delivery.after.integrated_lufs,
            "master_true_peak_dbtp": delivery.after.true_peak_dbtp,
            "ai_fizz_index": delivery.after.ai_fizz_index,
            "stem_separation": stem_separation
        }),
    })
}

/// Separates stems on the backend the environment selects (`FORGE_WORKER_BACKEND`) and
/// returns them with `{"backend", "fallbacks"}` so callers can see which backend ran. The
/// worker is an external step, so [`external::separate_stems_checked`] re-checks its paths
/// and verifies every stem through the root handle.
fn separate_ingest_stems(
    sandbox: &PathSandbox,
    input: &Path,
    stems_dir: &Path,
    require_neural: bool,
) -> Result<(StemBundle, Value), ForgeError> {
    let mut config = SeparationConfig::from_env();
    config.require_neural |= require_neural;
    let report = external::separate_stems_checked(sandbox, input, stems_dir, &config)?;
    let info = json!({ "backend": report.backend, "fallbacks": report.fallbacks });
    Ok((report.bundle, info))
}

/// Writes the drum-trigger MIDI (from `drum_buf`) and the voice-led chord MIDI.
fn write_ingest_midi(
    sandbox: &PathSandbox,
    paths: &IngestPaths,
    drum_buf: &AudioBuffer,
    bpm: f64,
    chords: &str,
) -> Result<(), ForgeError> {
    let (kick_notes, snare_notes, hat_notes) = extract_drum_midi_triggers(drum_buf, bpm, 480);
    sandbox.write_midi(
        &paths.drum_midi,
        bpm,
        480,
        &[
            ("Kick_Trigger_C1", &kick_notes),
            ("Snare_Trigger_D1", &snare_notes),
            ("HiHat_Trigger_F#1", &hat_notes),
        ],
    )?;
    let (_, chord_notes, bass_notes) = voice_chords(chords, VoicingStyle::NeoSoulRootless)?;
    sandbox.write_midi(
        &paths.chords_midi,
        bpm,
        480,
        &[("Voiced_Chords", &chord_notes), ("Sub_Bass", &bass_notes)],
    )
}

/// The pitch contour of the separated vocal (cleaned in place), or none without a vocal.
fn ingest_vocal_frames(
    sandbox: &PathSandbox,
    vocal: Option<&std::path::Path>,
    scale: &VocalScale,
) -> Result<Vec<PitchFrame>, ForgeError> {
    match vocal {
        Some(voc_path) => clean_ingested_vocal(sandbox, &sandbox.check(voc_path)?, scale),
        None => Ok(Vec::new()),
    }
}

/// Cleans and de-fizzes the separated vocal in place; returns its pitch contour.
fn clean_ingested_vocal(
    sandbox: &PathSandbox,
    voc_path: &std::path::Path,
    scale: &VocalScale,
) -> Result<Vec<PitchFrame>, ForgeError> {
    let voc_buf = sandbox.load_audio(voc_path)?;
    let voc_clean = clean_vocal_surgical(&voc_buf, true, true, -10.0)?;
    let voc_defizz = apply_spectral_defizz(&voc_clean, &DeFizzConfig::default())?;
    let frames = track_vocal_pitch(&voc_defizz, scale, 0.85);
    sandbox.save_wav(&voc_defizz, voc_path, 24)?;
    Ok(frames)
}

/// Inputs for the generated REAPER session.
struct SessionSpec {
    bpm: f64,
    sample_rate: u32,
    duration_secs: f64,
    vocal: Option<PathBuf>,
    drums: Option<PathBuf>,
    bass: Option<PathBuf>,
    other: Option<PathBuf>,
    vocal_pitch_frames: Vec<PitchFrame>,
    chords_midi: PathBuf,
    drum_midi: PathBuf,
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
    let has_drums = spec.drums.is_some();
    if let Some(drums) = spec.drums {
        rpp.tracks.push(ReaperTrackSpec {
            name: "02 Drums".to_string(),
            file_path: Some(drums),
            is_midi: false,
            duration_secs: spec.duration_secs,
            volume_lin: 0.90,
            pan: 0.0,
            color: 0x01FF_5555,
            jsfx_inserts: Vec::new(),
            sidechain_from_track: None,
            stretch_markers: Vec::new(),
            pitch_envelope: Vec::new(),
        });
    }
    if let Some(bass) = spec.bass {
        rpp.tracks.push(ReaperTrackSpec {
            name: "03 Bass".to_string(),
            file_path: Some(bass),
            is_midi: false,
            duration_secs: spec.duration_secs,
            volume_lin: 0.88,
            pan: 0.0,
            color: 0x01AA_66FF,
            jsfx_inserts: vec!["Forge_OpenSpacer_Unmask.jsfx".to_string()],
            sidechain_from_track: if has_drums { Some(1) } else { None },
            stretch_markers: Vec::new(),
            pitch_envelope: Vec::new(),
        });
    }
    if let Some(other) = spec.other {
        rpp.tracks.push(ReaperTrackSpec {
            name: "04 Instruments / Other".to_string(),
            file_path: Some(other),
            is_midi: false,
            duration_secs: spec.duration_secs,
            volume_lin: 0.82,
            pan: 0.0,
            color: 0x0144_DD88,
            jsfx_inserts: vec!["Forge_OpenSpacer_Unmask.jsfx".to_string()],
            sidechain_from_track: Some(0),
            stretch_markers: Vec::new(),
            pitch_envelope: Vec::new(),
        });
    }
    rpp.tracks.push(ReaperTrackSpec {
        name: "05 MIDI Chords & Sub-Bass".to_string(),
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
    rpp.tracks.push(ReaperTrackSpec {
        name: "06 Drum MIDI Triggers".to_string(),
        file_path: Some(spec.drum_midi),
        is_midi: true,
        duration_secs: spec.duration_secs,
        volume_lin: 0.85,
        pan: 0.0,
        color: 0x01FF_8844,
        jsfx_inserts: Vec::new(),
        sidechain_from_track: None,
        stretch_markers: Vec::new(),
        pitch_envelope: Vec::new(),
    });
    rpp
}

fn tool_separate_stems(args: &Value, sandbox: &PathSandbox) -> Result<ToolOutput, ForgeError> {
    let input = path_arg(args, "input", sandbox)?;
    let output_dir = path_arg(args, "output_dir", sandbox)?;
    let stems_raw = args.get("stems").and_then(Value::as_u64).unwrap_or(6);
    let stems = match stems_raw {
        2 | 4 | 6 => stems_raw as u8,
        other => {
            return Err(ForgeError::InvalidParameter(format!(
                "stems must be 2, 4, or 6, got {other}"
            )));
        }
    };
    let quality_str = args
        .get("quality")
        .and_then(Value::as_str)
        .unwrap_or("studio");
    let quality = forge_worker::parse_quality(quality_str)?;
    let dereverb = args
        .get("dereverb")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let drum_sep = args
        .get("drum_sep")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let mut cfg = SeparationConfig::from_env();
    cfg.stems = stems;
    cfg.quality = quality;
    cfg.dereverb = dereverb;
    cfg.drum_sep = drum_sep;

    let report = external::separate_stems_checked(sandbox, &input, &output_dir, &cfg)?;
    let payload = serde_json::to_value(&report)
        .map_err(|e| ForgeError::InvalidParameter(format!("Failed to serialize report: {e}")))?;
    Ok(payload.into())
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
            (
                "forge_neural_denoise",
                json!({ "input": "../in.wav", "output_dir": "out" }),
            ),
            (
                "forge_transcribe_midi",
                json!({ "input": "in.wav", "output_dir": "../midi_out" }),
            ),
            (
                "forge_separate_stems",
                json!({ "input": "in.wav", "output_dir": "../stems_out" }),
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
    fn out_of_range_bpm_and_ceiling_are_tool_errors() {
        let root = tempfile::tempdir().expect("tempdir");
        let sb = sandbox_in(root.path());
        write_test_mix(&root.path().join("mix.wav"), 0.05, 48_000);

        let theory = call_tool(
            &sb,
            "forge_music_theory_midi",
            json!({ "chords": "Dm9 G13", "output_midi": "x.mid", "bpm": 0 }),
        );
        assert!(is_error(&theory), "{}", text(&theory));
        assert!(text(&theory).contains("bpm"), "{}", text(&theory));
        assert!(!root.path().join("x.mid").exists());

        let master = call_tool(
            &sb,
            "forge_master_distrokid",
            json!({ "input": "mix.wav", "output": "master.wav", "ceiling_dbtp": 6.0 }),
        );
        assert!(is_error(&master), "{}", text(&master));
        assert!(text(&master).contains("ceiling_dbtp"), "{}", text(&master));
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

    fn repo_doc(relative: &str) -> String {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(relative);
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
    }

    fn advertised_tool_names() -> Vec<String> {
        tool_definitions()
            .iter()
            .map(|t| t["name"].as_str().expect("tool name").to_string())
            .collect()
    }

    #[test]
    fn readme_tool_table_matches_the_server() {
        let readme = repo_doc("README.md");
        let section = readme
            .split("### MCP tools")
            .nth(1)
            .and_then(|rest| rest.split("\n### ").next())
            .expect("README.md has a '### MCP tools' section");
        let mut documented: Vec<String> = section
            .lines()
            .filter_map(|line| line.strip_prefix("| `"))
            .filter_map(|rest| rest.split('`').next())
            .map(str::to_string)
            .collect();
        let mut served = advertised_tool_names();
        documented.sort();
        served.sort();
        assert_eq!(
            documented, served,
            "README.md '### MCP tools' table is out of sync with tool_definitions()"
        );
    }

    #[test]
    fn architecture_guide_lists_every_tool() {
        let guide = repo_doc("docs/ARCHITECTURE_AND_ENGINEERING_GUIDE.md");
        let served = advertised_tool_names();
        let count = format!("exposing {} structured tools", served.len());
        assert!(
            guide.contains(&count),
            "architecture guide should say \"{count}\" to match tool_definitions()"
        );
        for name in &served {
            assert!(
                guide.contains(&format!("`{name}`")),
                "architecture guide does not mention MCP tool `{name}`"
            );
        }
    }

    #[test]
    fn pre_tag_render_is_removed_when_tagging_fails() {
        let root = tempfile::tempdir().expect("tempdir");
        let sb = sandbox_in(root.path());
        write_test_mix(&root.path().join("mix.wav"), 0.05, 48_000);
        // A directory at the output path makes ffmpeg fail (as does ffmpeg being absent).
        std::fs::create_dir(root.path().join("master.wav")).expect("mkdir");

        let result = call_tool(
            &sb,
            "forge_master_distrokid",
            json!({ "input": "mix.wav", "output": "master.wav", "isrc": "US-S1Z-26-00001" }),
        );
        assert!(is_error(&result), "{}", text(&result));
        assert!(!root.path().join("master.pre_tag.wav").exists());
    }

    #[test]
    fn session_jsfx_is_installed_where_the_rpp_references_it() {
        let resource = tempfile::tempdir().expect("tempdir");
        let report = install_session_jsfx(Ok(resource.path().to_path_buf()));
        assert_eq!(report["installed"], true, "{report}");

        let session = build_session(SessionSpec {
            bpm: 120.0,
            sample_rate: 48_000,
            duration_secs: 10.0,
            vocal: Some(resource.path().join("vocals.wav")),
            drums: None,
            bass: None,
            other: None,
            vocal_pitch_frames: Vec::new(),
            chords_midi: resource.path().join("chords.mid"),
            drum_midi: resource.path().join("drums.mid"),
        });
        let effects = resource.path().join("Effects");
        let mut refs = 0;
        for line in session.build_rpp_string().lines() {
            let Some(rest) = line.trim_start().strip_prefix("<JS \"") else {
                continue;
            };
            let fx = &rest[..rest.find('"').expect("closing quote")];
            assert!(
                effects.join(fx).is_file(),
                "{fx} is not under {}",
                effects.display()
            );
            refs += 1;
        }
        assert_eq!(refs, 3);

        let missing = install_session_jsfx(Err(ForgeError::ReaperError("no REAPER".into())));
        assert_eq!(missing["installed"], false);
        assert!(
            missing["error"]
                .as_str()
                .expect("error")
                .contains("no REAPER")
        );
    }

    #[test]
    fn wave1_dsp_params_accepted_and_validated() {
        let root = tempfile::tempdir().expect("tempdir");
        let sb = sandbox_in(root.path());
        write_test_mix(&root.path().join("mix.wav"), 0.08, 48_000);

        // Valid Wave 1 params on forge_deliver
        let res = call_tool(
            &sb,
            "forge_deliver",
            json!({
                "input": "mix.wav",
                "output": "out_punchy_16.wav",
                "target": "spotify",
                "limiter_style": "punchy",
                "channel_link": 0.5,
                "dither": "noise-shaped",
                "dither_bits": 16
            }),
        );
        assert!(!is_error(&res), "{}", text(&res));
        assert!(root.path().join("out_punchy_16.wav").is_file());

        // Invalid limiter_style rejected
        let bad_style = call_tool(
            &sb,
            "forge_deliver",
            json!({
                "input": "mix.wav",
                "output": "bad.wav",
                "limiter_style": "hyper_extreme"
            }),
        );
        assert!(is_error(&bad_style));
        assert!(text(&bad_style).contains("limiter_style"));

        // Invalid dither_bits rejected
        let bad_bits = call_tool(
            &sb,
            "forge_deliver",
            json!({
                "input": "mix.wav",
                "output": "bad.wav",
                "dither_bits": 4
            }),
        );
        assert!(is_error(&bad_bits));

        // Valid Wave 1 params on forge_restore_and_defizz
        let restored = call_tool(
            &sb,
            "forge_restore_and_defizz",
            json!({
                "input": "mix.wav",
                "output": "restored.wav",
                "tilt": -2.0,
                "stereo_link": 0.75,
                "mid_side": 0.5,
                "delta": true
            }),
        );
        assert!(!is_error(&restored), "{}", text(&restored));

        // Out-of-range tilt rejected
        let bad_tilt = call_tool(
            &sb,
            "forge_restore_and_defizz",
            json!({
                "input": "mix.wav",
                "output": "restored_bad.wav",
                "tilt": 12.0
            }),
        );
        assert!(is_error(&bad_tilt));

        // Valid Wave 1 params on forge_spectral_unmask
        let unmasked = call_tool(
            &sb,
            "forge_spectral_unmask",
            json!({
                "target": "mix.wav",
                "sidechain": "mix.wav",
                "output": "unmasked.wav",
                "mid_side": -0.5,
                "delta": true
            }),
        );
        assert!(!is_error(&unmasked), "{}", text(&unmasked));

        // Invalid neural denoise backend rejected
        let bad_backend = call_tool(
            &sb,
            "forge_neural_denoise",
            json!({
                "input": "mix.wav",
                "output_dir": "denoised",
                "backend": "nonexistent"
            }),
        );
        assert!(is_error(&bad_backend));
        assert!(text(&bad_backend).contains("backend"));
    }
}
