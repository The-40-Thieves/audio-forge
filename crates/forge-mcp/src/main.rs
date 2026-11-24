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
    align_vocal_double_dtw_elastic, analyze_beat_grid, apply_18_band_eq, apply_spectral_defizz,
    apply_spectral_unmask, audit_audio, clean_vocal_surgical, compute_reference_match_eq, deliver,
    normalize_and_limit, quantize_to_constant_bpm, run_full_restoration_chain,
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
use forge_worker::{DrumHit, SeparationConfig};
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

fn schema_prop(ty: &str, default: Option<Value>, enum_vals: Option<Value>, desc: &str) -> Value {
    let mut map = serde_json::Map::new();
    map.insert("type".to_string(), json!(ty));
    if let Some(vals) = enum_vals {
        map.insert("enum".to_string(), vals);
    }
    if let Some(def) = default {
        map.insert("default".to_string(), def);
    }
    if !desc.is_empty() {
        map.insert("description".to_string(), json!(desc));
    }
    Value::Object(map)
}

fn prop_str(default: Option<&str>, desc: &str) -> Value {
    schema_prop("string", default.map(|d| json!(d)), None, desc)
}

fn prop_bool(default: bool, desc: &str) -> Value {
    schema_prop("boolean", Some(json!(default)), None, desc)
}

fn prop_num(default: Option<f64>, desc: &str) -> Value {
    schema_prop("number", default.map(|d| json!(d)), None, desc)
}

fn prop_int(default: Option<i64>, desc: &str) -> Value {
    schema_prop("integer", default.map(|d| json!(d)), None, desc)
}

fn prop_enum<T: Serialize>(ty: &str, values: &[T], default: Value, desc: &str) -> Value {
    schema_prop(ty, Some(default), Some(json!(values)), desc)
}

fn mcp_tool(name: &str, description: &str, props: &[(&str, Value)], required: &[&str]) -> Value {
    let mut properties = serde_json::Map::new();
    for (k, v) in props {
        properties.insert((*k).to_string(), v.clone());
    }
    json!({
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": Value::Object(properties),
            "required": required
        }
    })
}

fn worker_tool(
    name: &str,
    desc: &str,
    input_desc: &str,
    output_dir_desc: Option<&str>,
    backends: &[&str],
    default_timeout_secs: i64,
    extra: &[(&str, Value)],
) -> Value {
    let mut props = vec![("input", prop_str(None, input_desc))];
    let mut required = vec!["input"];
    if let Some(dir_desc) = output_dir_desc {
        props.push(("output_dir", prop_str(None, dir_desc)));
        required.push("output_dir");
    }
    props.push(("backend", prop_enum("string", backends, json!("auto"), "")));
    props.extend_from_slice(extra);
    props.push((
        "timeout_secs",
        prop_int(
            Some(default_timeout_secs),
            "Worker timeout in seconds (1..=3600)",
        ),
    ));
    mcp_tool(name, desc, &props, &required)
}

fn tool_definitions() -> Vec<Value> {
    let targets = delivery_target_names();
    let mid_side_prop = || {
        prop_num(
            Some(0.0),
            "Mid/Side balance (-1.0=Mid only, 0.0=linked, +1.0=Side only)",
        )
    };
    let mut separate_props = vec![
        ("input", prop_str(None, "Input audio file")),
        (
            "output_dir",
            prop_str(None, "Output directory for separated WAV stems"),
        ),
        (
            "stems",
            prop_enum(
                "integer",
                &[2, 4, 6],
                json!(6),
                "Primary stem count (2, 4, or 6)",
            ),
        ),
        (
            "quality",
            prop_enum(
                "string",
                &["fast", "studio", "ultra"],
                json!("studio"),
                "Separation quality tier",
            ),
        ),
    ];
    for &(flag, default, desc) in &[
        (
            "dereverb",
            true,
            "Run BS-RoFormer de-reverb on isolated vocals",
        ),
        (
            "drum_sep",
            false,
            "Also split isolated drums into Kick, Snare, Toms, Hi-Hat, and Cymbals",
        ),
        (
            "split_lead_backing",
            false,
            "Also split isolated vocals into Lead Vocals and Backing Vocals / Harmonies (RoFormer Karaoke)",
        ),
        (
            "split_male_female",
            false,
            "Also split isolated vocals into Male and Female vocal stems",
        ),
        (
            "split_guitar",
            false,
            "Also split isolated guitar into Electric Guitar and Acoustic Guitar",
        ),
        (
            "split_orchestral",
            false,
            "Also extract Strings, Brass, Woodwinds, and Synth from residual Other stem",
        ),
        (
            "split_crowd",
            false,
            "Also extract live Crowd / Audience / Room Noise from residual Other stem",
        ),
        (
            "all_stems",
            false,
            "Enable full 23-slot hierarchical pro studio cascade (all sub-splitters)",
        ),
        (
            "restore",
            false,
            "Run optional Apollo / spectral bandwidth restoration post-pass on separated stems",
        ),
        (
            "debleed",
            false,
            "Run Mel-Band-RoFormer stem bleed suppression pass on separated stems",
        ),
        (
            "de_aspiration",
            false,
            "Run Aspiration Mel-Band-RoFormer breath/aspiration cleanup on isolated vocals",
        ),
        (
            "dereverb_echo",
            false,
            "Use Mel-Band-RoFormer de-reverb/de-echo pass on isolated vocals",
        ),
    ] {
        separate_props.push((flag, prop_bool(default, desc)));
    }

    vec![
        mcp_tool(
            "forge_audit_audio",
            "Run an ITU-R BS.1770-4 / EBU R128 loudness, 4x polyphase True Peak (dBTP), LRA, Mid/Side fizz, bandwidth cutoff, transient sharpness, and AI Spectral Fizz Index audit on an audio file.",
            &[("path", prop_str(None, "Path to audio file"))],
            &["path"],
        ),
        mcp_tool(
            "forge_restore_and_defizz",
            "Clean AI-generated or live audio: DC offset, cubic Hermite de-clip, 50/60Hz hum removal, vocal de-click/de-plosive/de-breath, HPSS shimmer suppression, and OpenSoothe dynamic spectral de-fizz.",
            &[
                ("input", prop_str(None, "")),
                ("output", prop_str(None, "")),
                ("vocal", prop_bool(false, "")),
                ("defizz_depth_db", prop_num(Some(9.0), "")),
                ("sensitivity_db", prop_num(Some(3.5), "")),
                (
                    "tilt",
                    prop_num(Some(0.0), "Spectral tilt in dB/oct (-6.0..=6.0)"),
                ),
                (
                    "stereo_link",
                    prop_num(Some(1.0), "Stereo link ratio (0.0..=1.0)"),
                ),
                ("mid_side", mid_side_prop()),
                (
                    "delta",
                    prop_bool(false, "Audition only the removed spectral resonance delta"),
                ),
                (
                    "hard_mode",
                    prop_bool(
                        false,
                        "Enable steeper suppression slope for stubborn AI codec buzz",
                    ),
                ),
                (
                    "hpss_shimmer",
                    prop_num(
                        Some(0.0),
                        "HPSS erratic frame shimmer suppression boost (0.0..=1.0)",
                    ),
                ),
                (
                    "ultrasonic_lpf_hz",
                    prop_num(
                        None,
                        "Optional gentle ultrasonic LPF cutoff Hz (1000..=22000)",
                    ),
                ),
                (
                    "ai_mode",
                    prop_bool(
                        false,
                        "Derive telemetry-driven restoration and de-fizz settings automatically via audit_audio",
                    ),
                ),
            ],
            &["input", "output"],
        ),
        mcp_tool(
            "forge_spectral_unmask",
            "Run OpenSpacer (Trackspacer equivalent) sidechain STFT dynamic frequency unmasking between a sidechain trigger (vocal or kick) and target bed (music or bass).",
            &[
                ("target", prop_str(None, "")),
                ("sidechain", prop_str(None, "")),
                ("output", prop_str(None, "")),
                ("kick_bass_mode", prop_bool(false, "")),
                ("mid_side", mid_side_prop()),
                (
                    "delta",
                    prop_bool(false, "Audition only the carved sidechain delta"),
                ),
            ],
            &["target", "sidechain", "output"],
        ),
        mcp_tool(
            "forge_phase_align",
            "Align multi-mic phase via GCC-PHAT (Auto-Align equivalent) or vocal doubles via FastDTW stretch markers (VocAlign equivalent).",
            &[
                ("reference", prop_str(None, "")),
                ("target", prop_str(None, "")),
                ("output", prop_str(None, "")),
                ("vocal_dtw", prop_bool(false, "")),
                (
                    "elastic",
                    prop_bool(
                        false,
                        "Use WSOLA elastic stretch warping when vocal_dtw is true",
                    ),
                ),
            ],
            &["reference", "target", "output"],
        ),
        mcp_tool(
            "forge_vocal_suite",
            "Run YIN pitch tracking, formant-preserving pitch correction, 3-part chord-aware vocal harmony generation, and REAPER <PITCHENV compilation.",
            &[
                ("input", prop_str(None, "")),
                ("output_dir", prop_str(None, "")),
                ("scale", prop_str(Some("D minor"), "")),
                ("retune_strength", prop_num(Some(0.85), "")),
                ("generate_harmonies", prop_bool(true, "")),
            ],
            &["input", "output_dir"],
        ),
        mcp_tool(
            "forge_music_theory_midi",
            "Generate voice-led MIDI chords, Neo-Soul/Jazz reharmonizations, Dilla/MPC humanized grooves, and ABC notation.",
            &[
                ("chords", prop_str(None, "")),
                ("output_midi", prop_str(None, "")),
                ("bpm", prop_num(Some(92.0), "")),
                ("style", prop_str(Some("neo-soul"), "")),
            ],
            &["chords", "output_midi"],
        ),
        mcp_tool(
            "forge_master_distrokid",
            "Master stereo audio to -14.0 LUFS / -1.0 dBTP with Reference Match EQ, 18-band Mastering EQ, and optional ISO 3901 ISRC tagging.",
            &[
                ("input", prop_str(None, "")),
                ("output", prop_str(None, "")),
                ("reference", prop_str(None, "")),
                ("target_lufs", prop_num(Some(-14.0), "")),
                ("ceiling_dbtp", prop_num(Some(-1.0), "")),
                ("isrc", prop_str(None, "")),
                ("title", prop_str(None, "")),
                ("artist", prop_str(None, "")),
            ],
            &["input", "output"],
        ),
        mcp_tool(
            "forge_deliver",
            "Deliver a mix to a distribution target in one call: audit, repair (DC, clipping, hum), de-fizz, optional reference/genre tone match, loudness-normalize and true-peak limit, then re-audit. Returns before/after BS.1770-4 audits and explicit pass/fail checks (loudness within 1 LU, true peak under ceiling, no clipping, sample rate >= 44.1 kHz). Writes a 24-bit WAV and a .delivery.json report next to it.",
            &[
                ("input", prop_str(None, "Input mix or stem (WAV)")),
                ("output", prop_str(None, "Output master WAV path")),
                (
                    "target",
                    prop_enum("string", &targets, json!("spotify"), ""),
                ),
                (
                    "target_lufs",
                    prop_num(None, "Required when target is 'custom'"),
                ),
                (
                    "ceiling_dbtp",
                    prop_num(None, "Required when target is 'custom'"),
                ),
                ("repair", prop_bool(true, "")),
                ("defizz", prop_bool(true, "")),
                (
                    "vocal",
                    prop_bool(false, "Vocal clean-up; use on vocal stems only"),
                ),
                (
                    "reference",
                    prop_str(None, "Reference track WAV for 18-band tone match"),
                ),
                (
                    "genre",
                    prop_str(
                        None,
                        "Genre tonal-balance target (pop, hiphop, neosoul, edm, rock, jazz); ignored when reference is set",
                    ),
                ),
                (
                    "limiter_style",
                    prop_enum(
                        "string",
                        &["transparent", "punchy", "dynamic", "aggressive"],
                        json!("transparent"),
                        "",
                    ),
                ),
                (
                    "channel_link",
                    prop_num(Some(1.0), "Stereo limiter channel link ratio (0.0..=1.0)"),
                ),
                (
                    "dither",
                    prop_enum(
                        "string",
                        &["none", "tpdf", "noise-shaped"],
                        json!("none"),
                        "",
                    ),
                ),
                (
                    "dither_bits",
                    prop_int(Some(24), "Target bit depth for dither (8..=24)"),
                ),
                (
                    "ai_track_mode",
                    prop_bool(
                        false,
                        "Treat input as AI-generated audio (telemetry-driven de-fizz, avoids artificial high widening)",
                    ),
                ),
                (
                    "hpss_shimmer",
                    prop_num(
                        None,
                        "Override HPSS erratic shimmer suppression ratio (0.0..=1.0)",
                    ),
                ),
            ],
            &["input", "output"],
        ),
        worker_tool(
            "forge_neural_denoise",
            "Run neural audio denoising via DeepFilterNet3 or Mel-Band-RoFormer on the local Python GPU worker, confined to the MCP path sandbox.",
            "Input audio file",
            Some("Output directory for denoised audio"),
            &[
                "auto",
                "deepfilternet",
                "deepfilternet3",
                "mel-band-roformer",
                "roformer",
            ],
            300,
            &[(
                "attenuation_limit_db",
                prop_num(
                    None,
                    "Optional attenuation limit in dB (0..=100, DeepFilterNet3)",
                ),
            )],
        ),
        mcp_tool(
            "forge_transcribe_midi",
            "Run polyphonic audio-to-MIDI transcription via basic-pitch on the local Python GPU worker, confined to the MCP path sandbox.",
            &[
                ("input", prop_str(None, "Input audio file")),
                (
                    "output_dir",
                    prop_str(None, "Output directory for transcribed MIDI"),
                ),
                (
                    "timeout_secs",
                    prop_int(Some(300), "Worker timeout in seconds (1..=3600)"),
                ),
            ],
            &["input", "output_dir"],
        ),
        mcp_tool(
            "forge_separate_stems",
            "Studio-grade neural stem separation (BS-RoFormer 1297 / Mel-Band-RoFormer / BS-RoFormer SW cascade, 2/4/6 stems + de-reverb/de-echo/debleed/de-aspiration, Karaoke Lead/Backing split, 5-piece MDX23C DrumSep & bandwidth restoration) with pure-Rust DSP fallback, confined to the MCP path sandbox.",
            &separate_props,
            &["input", "output_dir"],
        ),
        mcp_tool(
            "forge_daw_inspect",
            "Validate and inspect a ForgeStudio DAW `.forgeproj` session (BPM, sample rate, tracks, audio objects, MIDI clips, aux buses, total PDC) within the MCP path sandbox.",
            &[(
                "project",
                prop_str(None, "Path to `.forgeproj` session file"),
            )],
            &["project"],
        ),
        mcp_tool(
            "forge_daw_bounce",
            "Headless 24-bit WAV mastering bounce (with ITU-R BS.1770-4 audit) and optional batch stem export from a ForgeStudio DAW `.forgeproj` session within the MCP path sandbox.",
            &[
                (
                    "project",
                    prop_str(None, "Path to `.forgeproj` session file"),
                ),
                (
                    "output",
                    prop_str(None, "Output 24-bit WAV master bounce path"),
                ),
                (
                    "stems_dir",
                    prop_str(
                        None,
                        "Optional directory for batch per-track stem WAV export",
                    ),
                ),
            ],
            &["project", "output"],
        ),
        mcp_tool(
            "forge_ingest_to_reaper",
            "Full autonomous AI-track pipeline: separate stems, clean & de-fizz vocal/other stems, unmask instruments, optionally layer drum one-shots, track beat grid / quantize stems or emit <TEMPOENVEX>, re-sum clean stems and deliver a verified master, install JSFX, and generate a REAPER 7.82 .RPP project.",
            &[
                ("input", prop_str(None, "")),
                ("output_dir", prop_str(None, "")),
                (
                    "target",
                    prop_enum("string", &targets, json!("spotify"), ""),
                ),
                (
                    "target_lufs",
                    prop_num(None, "Required when target is 'custom'"),
                ),
                (
                    "ceiling_dbtp",
                    prop_num(None, "Required when target is 'custom'"),
                ),
                ("bpm", prop_num(Some(120.0), "")),
                ("scale", prop_str(Some("D minor"), "")),
                ("chords", prop_str(Some("Dm9 Bbmaj7 Gm9 A7b13"), "")),
                (
                    "require_neural",
                    prop_bool(
                        false,
                        "Fail instead of falling back to the DSP stem split when no GPU or Colab worker produces stems",
                    ),
                ),
                (
                    "drum_samples_dir",
                    prop_str(
                        None,
                        "Optional sandboxed directory of .wav drum one-shots for phase-locked Kick/Snare/HiHat reinforcement",
                    ),
                ),
                (
                    "drum_replace_mix",
                    prop_num(
                        Some(0.5),
                        "Wet mix ratio for sample-layered drums (0.0..=1.0)",
                    ),
                ),
                (
                    "quantize_grid",
                    prop_bool(
                        false,
                        "WSOLA time-warp separated stems onto a constant bpm grid instead of emitting variable <TEMPOENVEX>",
                    ),
                ),
            ],
            &["input", "output_dir"],
        ),
        worker_tool(
            "forge_neural_restore",
            "Run neural bandwidth restoration and AI artifact cleanup (Apollo, FlowHigh, AudioSR, SonicMaster, or spectral exciter fallback) within the MCP path sandbox.",
            "Input audio file",
            Some("Output directory for restored audio"),
            &[
                "auto",
                "apollo",
                "flowhigh",
                "audiosr",
                "sonicmaster",
                "exciter",
            ],
            600,
            &[
                (
                    "prompt",
                    prop_str(None, "Optional instruction prompt (used by SonicMaster)"),
                ),
                (
                    "chunk_secs",
                    prop_num(None, "Processing chunk length in seconds (0.5..=120.0)"),
                ),
            ],
        ),
        worker_tool(
            "forge_vocal_rescue",
            "Rescue robotic or phasey AI vocals (Resemble Enhance, Seed-VC zero-shot timbre, or RoFormer chain) within the MCP path sandbox.",
            "Input vocal stem",
            Some("Output directory for rescued vocal"),
            &["auto", "resemble-enhance", "seed-vc", "roformer-clean"],
            600,
            &[
                (
                    "reference_voice",
                    prop_str(
                        None,
                        "Optional clean human reference vocal WAV inside sandbox",
                    ),
                ),
                (
                    "dereverb",
                    prop_bool(true, "Run de-reverb/de-echo cleaning"),
                ),
                (
                    "de_aspiration",
                    prop_bool(true, "Run vocal aspiration suppression"),
                ),
            ],
        ),
        worker_tool(
            "forge_beat_track",
            "Extract beats, downbeats, median BPM, and tempo-drift map (beat-this, librosa, energy, or pure-Rust onset DP) within the MCP path sandbox.",
            "Input audio file",
            None,
            &[
                "auto",
                "beat-this",
                "librosa",
                "energy",
                "rust",
                "pure_rust_onset",
            ],
            300,
            &[(
                "hint_bpm",
                prop_num(None, "Optional BPM hint (30.0..=300.0) for Rust fallback"),
            )],
        ),
        worker_tool(
            "forge_evaluate_quality",
            "Evaluate perceptual audio quality (Meta Audiobox Aesthetics PQ/PC/CE/CU, SingMOS, or acoustic proxy) within the MCP path sandbox.",
            "Input audio file",
            None,
            &["auto", "audiobox", "singmos", "acoustic-proxy", "rust"],
            300,
            &[
                (
                    "reference",
                    prop_str(
                        None,
                        "Optional reference WAV inside sandbox for paired comparison",
                    ),
                ),
                (
                    "vocal",
                    prop_bool(false, "Prioritize vocal quality (SingMOS)"),
                ),
            ],
        ),
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

fn optional_bounded_f64_arg(
    args: &Value,
    key: &str,
    min: f64,
    max: f64,
) -> Result<Option<f64>, ForgeError> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(_) => Ok(Some(bounded_f64_arg(args, key, min, min, max)?)),
    }
}

fn validate_backend_arg<'a>(args: &Value, allowed: &[&'a str]) -> Result<&'a str, ForgeError> {
    let raw = match args.get("backend") {
        None | Some(Value::Null) => "auto",
        Some(v) => v
            .as_str()
            .ok_or_else(|| ForgeError::InvalidParameter("'backend' must be a string".into()))?,
    }
    .trim();
    allowed
        .iter()
        .copied()
        .find(|c| c.eq_ignore_ascii_case(raw))
        .ok_or_else(|| {
            ForgeError::InvalidParameter(format!(
                "unknown 'backend' '{raw}' (expected one of {})",
                allowed.join(", ")
            ))
        })
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
        "forge_daw_inspect" => tool_daw_inspect,
        "forge_daw_bounce" => tool_daw_bounce,
        "forge_neural_restore" => tool_neural_restore,
        "forge_vocal_rescue" => tool_vocal_rescue,
        "forge_beat_track" => tool_beat_track,
        "forge_evaluate_quality" => tool_evaluate_quality,
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
    options.ai_track_mode = bool_arg(args, "ai_track_mode", false);
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
    let hpss_shimmer = optional_bounded_f64_arg(args, "hpss_shimmer", 0.0, 1.0)?;

    let buf = sandbox.load_audio(&input)?;
    if let Some(shimmer) = hpss_shimmer {
        let mut cfg = DeFizzConfig::ai_mix_from_audit(&audit_audio(&buf));
        cfg.hpss_shimmer_suppress = shimmer;
        options.defizz_config = Some(cfg);
    }
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
    let ai_mode = bool_arg(args, "ai_mode", false);
    let vocal = bool_arg(args, "vocal", false);
    let sens_opt = optional_bounded_f64_arg(args, "sensitivity_db", 0.1, 24.0)?;
    let depth_opt = optional_bounded_f64_arg(args, "defizz_depth_db", 0.0, 24.0)?;
    let tilt = bounded_f64_arg(args, "tilt", 0.0, -6.0, 6.0)?;
    let stereo_link = bounded_f64_arg(args, "stereo_link", 1.0, 0.0, 1.0)?;
    let mid_side = bounded_f64_arg(args, "mid_side", 0.0, -1.0, 1.0)?;
    let delta = bool_arg(args, "delta", false);
    let hard_mode = bool_arg(args, "hard_mode", false);
    let hpss_shimmer = bounded_f64_arg(args, "hpss_shimmer", 0.0, 0.0, 1.0)?;
    let ultrasonic_lpf_hz = optional_bounded_f64_arg(args, "ultrasonic_lpf_hz", 1000.0, 22_000.0)?;

    let raw = sandbox.load_audio(&input)?;
    let before = audit_audio(&raw);
    let rest_cfg = if ai_mode {
        RestorationConfig::from_audit(&before, true)
    } else {
        RestorationConfig::default()
    };
    let mut stage1 = run_full_restoration_chain(&raw, &rest_cfg)?;
    if vocal {
        stage1 = clean_vocal_surgical(&stage1, true, true, -12.0)?;
    }
    let mut config = if ai_mode {
        DeFizzConfig::ai_mix_from_audit(&audit_audio(&stage1))
    } else {
        DeFizzConfig::default()
    };
    if let Some(s) = sens_opt {
        config.sensitivity_db = s;
    }
    if let Some(d) = depth_opt {
        config.max_depth_db = d;
    }
    config.tilt_db_per_oct = tilt;
    config.stereo_link = stereo_link;
    if args.get("mid_side").and_then(Value::as_f64).is_some() || !ai_mode {
        config.mid_side_balance = mid_side;
    }
    config.delta_listen = delta;
    config.hard_mode |= hard_mode;
    if hpss_shimmer > 0.0 {
        config.hpss_shimmer_suppress = hpss_shimmer;
    }
    if let Some(lpf) = ultrasonic_lpf_hz {
        config.ultrasonic_lpf_hz = Some(lpf);
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
    let backend = match validate_backend_arg(
        args,
        &[
            "auto",
            "deepfilternet",
            "deepfilternet3",
            "mel-band-roformer",
            "roformer",
        ],
    )? {
        "deepfilternet3" => "deepfilternet",
        "roformer" => "mel-band-roformer",
        other => other,
    };
    let attenuation_limit_db = optional_bounded_f64_arg(args, "attenuation_limit_db", 0.0, 100.0)?;
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

fn tool_neural_restore(args: &Value, sandbox: &PathSandbox) -> Result<ToolOutput, ForgeError> {
    let input = path_arg(args, "input", sandbox)?;
    let output_dir = path_arg(args, "output_dir", sandbox)?;
    let backend = validate_backend_arg(
        args,
        &[
            "auto",
            "apollo",
            "flowhigh",
            "audiosr",
            "sonicmaster",
            "exciter",
        ],
    )?;
    let prompt = args.get("prompt").and_then(Value::as_str);
    let chunk_secs = optional_bounded_f64_arg(args, "chunk_secs", 0.5, 120.0)?;
    let timeout = timeout_secs_arg(args)?;
    let res = external::neural_restore_checked(
        sandbox,
        &input,
        &output_dir,
        backend,
        prompt,
        chunk_secs,
        timeout,
    )?;
    Ok(serde_json::to_value(res)?.into())
}

fn tool_vocal_rescue(args: &Value, sandbox: &PathSandbox) -> Result<ToolOutput, ForgeError> {
    let input = path_arg(args, "input", sandbox)?;
    let output_dir = path_arg(args, "output_dir", sandbox)?;
    let backend = validate_backend_arg(
        args,
        &["auto", "resemble-enhance", "seed-vc", "roformer-clean"],
    )?;
    let reference_voice = optional_path_arg(args, "reference_voice", sandbox)?;
    let dereverb = bool_arg(args, "dereverb", true);
    let de_aspiration = bool_arg(args, "de_aspiration", true);
    let timeout = timeout_secs_arg(args)?;
    let res = external::vocal_rescue_checked(
        sandbox,
        &input,
        &output_dir,
        backend,
        reference_voice.as_deref(),
        dereverb,
        de_aspiration,
        timeout,
    )?;
    Ok(serde_json::to_value(res)?.into())
}

fn tool_beat_track(args: &Value, sandbox: &PathSandbox) -> Result<ToolOutput, ForgeError> {
    let input = path_arg(args, "input", sandbox)?;
    let hint_bpm = optional_bounded_f64_arg(args, "hint_bpm", 30.0, 300.0)?;
    let backend = validate_backend_arg(
        args,
        &[
            "auto",
            "beat-this",
            "librosa",
            "energy",
            "rust",
            "pure_rust_onset",
        ],
    )?;
    let timeout = timeout_secs_arg(args)?;
    let res = external::beat_track_checked(sandbox, &input, hint_bpm, backend, timeout)?;
    Ok(serde_json::to_value(res)?.into())
}

fn tool_evaluate_quality(args: &Value, sandbox: &PathSandbox) -> Result<ToolOutput, ForgeError> {
    let input = path_arg(args, "input", sandbox)?;
    let reference = optional_path_arg(args, "reference", sandbox)?;
    let vocal = bool_arg(args, "vocal", false);
    let backend = validate_backend_arg(
        args,
        &["auto", "audiobox", "singmos", "acoustic-proxy", "rust"],
    )?;
    let timeout = timeout_secs_arg(args)?;
    let res = external::evaluate_quality_checked(
        sandbox,
        &input,
        reference.as_deref(),
        vocal,
        backend,
        timeout,
    )?;
    Ok(serde_json::to_value(res)?.into())
}

fn tool_phase_align(args: &Value, sandbox: &PathSandbox) -> Result<ToolOutput, ForgeError> {
    let reference = path_arg(args, "reference", sandbox)?;
    let target = path_arg(args, "target", sandbox)?;
    let output = path_arg(args, "output", sandbox)?;
    let elastic = bool_arg(args, "elastic", false);

    let ref_buf = sandbox.load_audio(&reference)?;
    let tgt_buf = sandbox.load_audio(&target)?;
    if bool_arg(args, "vocal_dtw", false) {
        let (warped, markers) = if elastic {
            align_vocal_double_dtw_elastic(&ref_buf, &tgt_buf, 120.0)?
        } else {
            align_vocal_double_dtw(&ref_buf, &tgt_buf, 120.0)?
        };
        sandbox.save_wav(&warped, &output, 24)?;
        Ok(json!({
            "mode": if elastic { "vocal_dtw_elastic" } else { "vocal_dtw" },
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
    let drum_samples_dir = optional_path_arg(args, "drum_samples_dir", sandbox)?;
    let drum_replace_mix = bounded_f64_arg(args, "drum_replace_mix", 0.5, 0.0, 1.0)?;
    let quantize_grid = bool_arg(args, "quantize_grid", false);

    sandbox.create_dir_all(&paths.output_dir)?;
    sandbox.create_dir_all(&paths.midi_dir)?;

    let full_mix = sandbox.load_audio(&input)?;
    let beat_grid = analyze_beat_grid(&full_mix, Some(bpm));

    let require_neural = bool_arg(args, "require_neural", false);
    let (mut bundle, stem_separation) =
        separate_ingest_stems(sandbox, &input, &paths.stems_dir, require_neural)?;

    let mut primary_stems: Vec<(PathBuf, AudioBuffer)> = Vec::new();
    let (vocal_pitch_frames, cleaned_vocal) =
        ingest_vocal_frames(sandbox, bundle.vocals.as_deref(), &scale)?;
    if let (Some(voc_path), Some(voc_buf)) = (&bundle.vocals, &cleaned_vocal) {
        primary_stems.push((sandbox.check(voc_path)?, voc_buf.clone()));
    }

    if let Some(ref other_path) = bundle.other {
        let checked = sandbox.check(other_path)?;
        let other_clean = clean_ingested_other(sandbox, &checked, cleaned_vocal.as_ref())?;
        primary_stems.push((checked, other_clean));
    }

    let mut drum_buf =
        sandbox.load_audio(&sandbox.check(bundle.drums.as_ref().unwrap_or(&input))?)?;
    let (kick_notes, snare_notes, hat_notes) =
        write_ingest_midi(sandbox, &paths, &drum_buf, bpm, chords)?;

    let mut drum_hits_layered = 0_usize;
    if let Some(ref samples_dir) = drum_samples_dir {
        let tv = |ns: &[NoteEvent]| -> Vec<(u32, u8)> {
            ns.iter().map(|n| (n.start_tick, n.velocity)).collect()
        };
        let hits = DrumHit::from_note_groups(
            [
                ("Kick", tv(&kick_notes)),
                ("Snare", tv(&snare_notes)),
                ("HiHat", tv(&hat_notes)),
            ],
            bpm,
            480,
            drum_buf.sample_rate,
        );
        let (layered, count) = external::render_drum_layer_checked(
            sandbox,
            samples_dir,
            drum_buf.num_frames(),
            drum_buf.sample_rate,
            &hits,
            drum_replace_mix,
            &drum_buf,
        )?;
        drum_hits_layered = count;
        let out_drum = match bundle.drums {
            Some(ref p) => sandbox.check(p)?,
            None => sandbox.check(&paths.stems_dir.join("Drums_Layered.wav"))?,
        };
        sandbox.save_wav(&layered, &out_drum, 24)?;
        bundle.drums = Some(out_drum);
        drum_buf = layered;
    }
    if let Some(ref drums_path) = bundle.drums {
        primary_stems.push((sandbox.check(drums_path)?, drum_buf));
    }

    for opt_path in [&bundle.bass, &bundle.guitar, &bundle.keys]
        .into_iter()
        .flatten()
    {
        let checked = sandbox.check(opt_path)?;
        let buf = sandbox.load_audio(&checked)?;
        primary_stems.push((checked, buf));
    }

    let mut quantized_applied = false;
    if quantize_grid && beat_grid.beats_secs.len() >= 2 {
        for (path, buf) in &mut primary_stems {
            let (warped, _) = quantize_to_constant_bpm(buf, &beat_grid.beats_secs, bpm)?;
            sandbox.save_wav(&warped, path, 24)?;
            *buf = warped;
        }
        quantized_applied = true;
    }

    let target_frames = primary_stems
        .iter()
        .map(|(_, b)| b.num_frames())
        .max()
        .unwrap_or_else(|| full_mix.num_frames());
    let duration_secs = (target_frames as f64) / f64::from(full_mix.sample_rate.max(1));
    let refs: Vec<&AudioBuffer> = primary_stems.iter().map(|(_, b)| b).collect();
    let (mastered, delivery) = forge_dsp::resum_and_deliver_ai_stems(&refs, &full_mix, target)?;
    sandbox.save_wav(&mastered, &paths.master, 24)?;
    sandbox.write_json(&paths.delivery, &delivery)?;

    let tempo_map = if !quantized_applied && beat_grid.max_bpm_drift >= 1.5 {
        beat_grid.tempo_map.clone()
    } else {
        Vec::new()
    };
    let session = SessionSpec {
        bpm,
        sample_rate: full_mix.sample_rate,
        duration_secs,
        vocal: bundle.vocals,
        drums: bundle.drums,
        bass: bundle.bass,
        other: bundle.other,
        guitar: bundle.guitar,
        keys: bundle.keys,
        vocal_pitch_frames,
        tempo_map,
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
            "stem_separation": stem_separation,
            "beat_grid": {
                "median_bpm": beat_grid.median_bpm,
                "max_bpm_drift": beat_grid.max_bpm_drift,
                "beats_detected": beat_grid.beats_secs.len(),
                "quantized": quantized_applied,
            },
            "drum_hits_layered": drum_hits_layered
        }),
    })
}

/// Separates stems on the backend the environment selects (`FORGE_WORKER_BACKEND`) and
/// returns them with `{"backend", "fallbacks", "warnings"}` so callers can see which backend ran. The
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
    let info = json!({
        "backend": report.backend,
        "fallbacks": report.fallbacks,
        "warnings": report.warnings
    });
    Ok((report.bundle, info))
}

type DrumTriggerMidi = (Vec<NoteEvent>, Vec<NoteEvent>, Vec<NoteEvent>);

/// Writes the drum-trigger MIDI (from `drum_buf`) and the voice-led chord MIDI, returning triggers.
fn write_ingest_midi(
    sandbox: &PathSandbox,
    paths: &IngestPaths,
    drum_buf: &AudioBuffer,
    bpm: f64,
    chords: &str,
) -> Result<DrumTriggerMidi, ForgeError> {
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
    )?;
    Ok((kick_notes, snare_notes, hat_notes))
}

/// The pitch contour and cleaned buffer of the separated vocal, or none without a vocal.
fn ingest_vocal_frames(
    sandbox: &PathSandbox,
    vocal: Option<&std::path::Path>,
    scale: &VocalScale,
) -> Result<(Vec<PitchFrame>, Option<AudioBuffer>), ForgeError> {
    match vocal {
        Some(voc_path) => clean_ingested_vocal(sandbox, &sandbox.check(voc_path)?, scale),
        None => Ok((Vec::new(), None)),
    }
}

/// Cleans and de-fizzes the separated vocal in place using `ai_vocal`; returns pitch frames & buffer.
fn clean_ingested_vocal(
    sandbox: &PathSandbox,
    voc_path: &std::path::Path,
    scale: &VocalScale,
) -> Result<(Vec<PitchFrame>, Option<AudioBuffer>), ForgeError> {
    let voc_buf = sandbox.load_audio(voc_path)?;
    let voc_clean = clean_vocal_surgical(&voc_buf, true, true, -10.0)?;
    let voc_defizz = apply_spectral_defizz(&voc_clean, &DeFizzConfig::ai_vocal())?;
    let frames = track_vocal_pitch(&voc_defizz, scale, 0.85);
    sandbox.save_wav(&voc_defizz, voc_path, 24)?;
    Ok((frames, Some(voc_defizz)))
}

/// De-fizzes and optionally vocal-unmasks the `other` instrumental stem in place.
fn clean_ingested_other(
    sandbox: &PathSandbox,
    other_path: &std::path::Path,
    cleaned_vocal: Option<&AudioBuffer>,
) -> Result<AudioBuffer, ForgeError> {
    let other_buf = sandbox.load_audio(other_path)?;
    let defizz_cfg = DeFizzConfig::ai_mix_from_audit(&audit_audio(&other_buf));
    let mut other_clean = apply_spectral_defizz(&other_buf, &defizz_cfg)?;
    if let Some(voc) = cleaned_vocal {
        other_clean =
            apply_spectral_unmask(&other_clean, voc, &UnmaskConfig::vocal_over_instruments())?;
    }
    sandbox.save_wav(&other_clean, other_path, 24)?;
    Ok(other_clean)
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
    guitar: Option<PathBuf>,
    keys: Option<PathBuf>,
    vocal_pitch_frames: Vec<PitchFrame>,
    tempo_map: Vec<forge_dsp::TempoPoint>,
    chords_midi: PathBuf,
    drum_midi: PathBuf,
}

fn build_session(spec: SessionSpec) -> ReaperProjectBuilder {
    let mut rpp =
        ReaperProjectBuilder::new("ForgeAudio Studio Session", spec.bpm, spec.sample_rate);
    rpp.tempo_map = spec.tempo_map;
    rpp.add_default_song_structure_markers();
    rpp.tracks.push(
        ReaperTrackSpec::audio(
            "01 Lead Vocal",
            spec.vocal,
            spec.duration_secs,
            0.85,
            0x0155_88FF,
        )
        .with_jsfx(["Forge_Vocal_Surgical.jsfx", "Forge_OpenSoothe_DeFizz.jsfx"])
        .with_pitch_envelope(spec.vocal_pitch_frames),
    );
    let has_drums = spec.drums.is_some();
    if let Some(drums) = spec.drums {
        rpp.tracks.push(ReaperTrackSpec::audio(
            "02 Drums",
            Some(drums),
            spec.duration_secs,
            0.90,
            0x01FF_5555,
        ));
    }
    if let Some(bass) = spec.bass {
        rpp.tracks.push(
            ReaperTrackSpec::audio("03 Bass", Some(bass), spec.duration_secs, 0.88, 0x01AA_66FF)
                .with_jsfx(["Forge_OpenSpacer_Unmask.jsfx"])
                .with_sidechain(if has_drums { Some(1) } else { None }),
        );
    }
    if let Some(other) = spec.other {
        rpp.tracks.push(
            ReaperTrackSpec::audio(
                "04 Instruments / Other",
                Some(other),
                spec.duration_secs,
                0.82,
                0x0144_DD88,
            )
            .with_jsfx(["Forge_OpenSpacer_Unmask.jsfx"])
            .with_sidechain(Some(0)),
        );
    }
    rpp.tracks.push(ReaperTrackSpec::midi(
        "05 MIDI Chords & Sub-Bass",
        spec.chords_midi,
        spec.duration_secs,
        0.85,
        0x01FF_CC44,
    ));
    rpp.tracks.push(ReaperTrackSpec::midi(
        "06 Drum MIDI Triggers",
        spec.drum_midi,
        spec.duration_secs,
        0.85,
        0x01FF_8844,
    ));
    if let Some(guitar) = spec.guitar {
        rpp.tracks.push(ReaperTrackSpec::audio(
            "07 Guitar",
            Some(guitar),
            spec.duration_secs,
            0.80,
            0x0144_AAFF,
        ));
    }
    if let Some(keys) = spec.keys {
        rpp.tracks.push(ReaperTrackSpec::audio(
            "08 Keys / Piano",
            Some(keys),
            spec.duration_secs,
            0.80,
            0x01AA_FF44,
        ));
    }
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
    let split_lead_backing = bool_arg(args, "split_lead_backing", false);
    let split_male_female = bool_arg(args, "split_male_female", false);
    let split_guitar = bool_arg(args, "split_guitar", false);
    let split_orchestral = bool_arg(args, "split_orchestral", false);
    let split_crowd = bool_arg(args, "split_crowd", false);
    let all_stems = bool_arg(args, "all_stems", false);
    let restore = bool_arg(args, "restore", false);
    let debleed = bool_arg(args, "debleed", false);
    let de_aspiration = bool_arg(args, "de_aspiration", false);
    let dereverb_echo = bool_arg(args, "dereverb_echo", false);

    let mut cfg = SeparationConfig::from_env();
    cfg.stems = stems;
    cfg.quality = quality;
    cfg.dereverb = dereverb;
    cfg.drum_sep = drum_sep;
    cfg.split_lead_backing = split_lead_backing;
    cfg.split_male_female = split_male_female;
    cfg.split_guitar = split_guitar;
    cfg.split_orchestral = split_orchestral;
    cfg.split_crowd = split_crowd;
    if all_stems {
        cfg.enable_all_stems();
        cfg.dereverb = dereverb;
    }
    cfg.restore = restore;
    cfg.debleed = debleed;
    cfg.de_aspiration = de_aspiration;
    cfg.dereverb_echo = dereverb_echo;

    let report = external::separate_stems_checked(sandbox, &input, &output_dir, &cfg)?;
    let payload = serde_json::to_value(&report)
        .map_err(|e| ForgeError::InvalidParameter(format!("Failed to serialize report: {e}")))?;
    Ok(payload.into())
}

fn load_sandboxed_daw_project(
    project_path: &std::path::Path,
    sandbox: &PathSandbox,
    require_all_clips: bool,
) -> Result<
    (
        forge_engine::DawProject,
        forge_engine::ClipCache,
        Vec<String>,
    ),
    ForgeError,
> {
    const MAX_PROJ_BYTES: u64 = 16 * 1024 * 1024;
    let file = sandbox.open_read(project_path)?;
    let mut raw = String::new();
    std::io::Read::read_to_string(&mut std::io::Read::take(file, MAX_PROJ_BYTES + 1), &mut raw)
        .map_err(ForgeError::Io)?;
    if (raw.len() as u64) > MAX_PROJ_BYTES {
        return Err(ForgeError::InvalidParameter(
            "Project file exceeds 16 MiB maximum size".to_string(),
        ));
    }
    let proj = forge_engine::DawProject::from_json(&raw)?;
    let mut cache = forge_engine::ClipCache::default();
    let mut missing_clips = Vec::new();
    for tr in &proj.tracks {
        for obj in &tr.objects {
            let safe_clip = sandbox.check(&obj.source_path)?;
            match sandbox.load_audio(&safe_clip) {
                Ok(buf) => {
                    cache.insert_memory_clip(obj.source_path.clone(), buf);
                }
                Err(e) if !require_all_clips => {
                    missing_clips.push(format!("{}: {e}", safe_clip.display()));
                }
                Err(e) => return Err(e),
            }
        }
    }
    Ok((proj, cache, missing_clips))
}

fn tool_daw_inspect(args: &Value, sandbox: &PathSandbox) -> Result<ToolOutput, ForgeError> {
    let project_path = path_arg(args, "project", sandbox)?;
    let (proj, cache, missing_clips) = load_sandboxed_daw_project(&project_path, sandbox, false)?;
    let engine = forge_engine::ObjectOrientedEngine::compile(&proj, &cache);
    let summary =
        serde_json::to_value(proj.inspect_summary(engine.total_pdc_samples(), missing_clips))?;
    Ok(summary.into())
}

fn tool_daw_bounce(args: &Value, sandbox: &PathSandbox) -> Result<ToolOutput, ForgeError> {
    let project_path = path_arg(args, "project", sandbox)?;
    let output_path = path_arg(args, "output", sandbox)?;
    let stems_dir = optional_path_arg(args, "stems_dir", sandbox)?;

    let (proj, cache, _) = load_sandboxed_daw_project(&project_path, sandbox, true)?;
    let base_frames = usize::try_from(proj.total_length_samples()).map_err(|_| {
        ForgeError::InvalidParameter("Project duration overflows platform pointer width".into())
    })?;
    let tail_frames = forge_engine::ObjectOrientedEngine::compute_reverb_tail_frames(&proj);
    let total = base_frames.saturating_add(tail_frames);
    let master_buf = forge_engine::ObjectOrientedEngine::bounce_offline(&proj, &cache, total)?;
    sandbox.save_wav(&master_buf, &output_path, 24)?;
    let report = audit_audio(&master_buf);

    let stems_exported = if let Some(dir) = stems_dir {
        sandbox.create_dir_all(&dir)?;
        forge_engine::ObjectOrientedEngine::bounce_stems_with(
            &proj,
            &cache,
            |ti, safe_name, stem_buf| {
                let stem_path =
                    sandbox.check(&dir.join(format!("{:02}_{safe_name}.wav", ti + 1)))?;
                sandbox.save_wav(stem_buf, &stem_path, 24)?;
                Ok(())
            },
        )?
    } else {
        0
    };
    let payload = json!({
        "output": output_path.display().to_string(),
        "integrated_lufs": report.integrated_lufs,
        "true_peak_dbtp": report.true_peak_dbtp,
        "stems_exported": stems_exported,
    });
    Ok(payload.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn sandbox_in(dir: &Path) -> PathSandbox {
        PathSandbox::new([dir.to_path_buf()]).expect("sandbox")
    }

    /// 10 s stereo mix-like signal (two tones plus broadband hash), as in the delivery tests.
    fn write_test_mix(path: &Path, level: f64, sample_rate: u32) {
        forge_dsp::delivery::synthetic_delivery_fixture(level, sample_rate)
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
            (
                "forge_daw_inspect",
                json!({ "project": "../session.forgeproj" }),
            ),
            (
                "forge_daw_bounce",
                json!({ "project": "session.forgeproj", "output": "../bounce.wav" }),
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

    fn assert_delivery_ready(sample_rate: u32, output_rel: &str, expect_ready: bool) {
        let root = tempfile::tempdir().expect("tempdir");
        let sb = sandbox_in(root.path());
        write_test_mix(&root.path().join("input.wav"), 0.05, sample_rate);

        let result = call_tool(
            &sb,
            "forge_deliver",
            json!({ "input": "input.wav", "output": output_rel }),
        );
        assert_eq!(
            is_error(&result),
            !expect_ready,
            "ready={expect_ready}: {}",
            text(&result)
        );
        let body: Value = serde_json::from_str(text(&result)).expect("JSON body");
        assert_eq!(body["report"]["ready"], json!(expect_ready));
        assert!(root.path().join(output_rel).exists());
    }

    #[test]
    fn not_ready_delivery_reports_is_error() {
        // 22.05 kHz fails the sample-rate check, so the delivery cannot be `ready`.
        assert_delivery_ready(22_050, "out/master.wav", false);
    }

    #[test]
    fn ready_delivery_reports_success() {
        assert_delivery_ready(48_000, "master.wav", true);
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
            guitar: None,
            keys: None,
            vocal_pitch_frames: Vec::new(),
            tempo_map: Vec::new(),
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

    #[test]
    fn ai_suite_tools_sandbox_enforcement_and_ingest_features() {
        let root = tempfile::tempdir().expect("tempdir");
        let sb = sandbox_in(root.path());

        // Write drifting-tempo click+harmonic WAV (100 BPM -> 132 BPM) so beat grid + <TEMPOENVEX>
        // trigger while maintaining sustained bed RMS so delivery hits -14 LUFS under -1 dBTP.
        let sr = 48_000_u32;
        let frames = (sr as usize) * 8;
        let mut ch: Vec<f64> = (0..frames)
            .map(|i| {
                let ts = (i as f64) / f64::from(sr);
                0.22 * (2.0 * std::f64::consts::PI * 220.0 * ts).sin()
            })
            .collect();
        let mut t = 0.2_f64;
        while t < 7.7 {
            let start = (t * f64::from(sr)).round() as usize;
            for k in 0..600 {
                if start + k < frames {
                    let env = (-(k as f64) / 100.0).exp();
                    ch[start + k] += 0.35
                        * env
                        * (2.0 * std::f64::consts::PI * 105.0 * (k as f64) / f64::from(sr)).sin();
                }
            }
            let bpm = if t < 4.0 { 100.0 } else { 132.0 };
            t += 60.0 / bpm;
        }
        AudioBuffer::from_channels(vec![ch.clone(), ch], sr)
            .unwrap()
            .save_wav(root.path().join("ai_song.wav"), 24)
            .unwrap();

        // Write one kick drum one-shot into a sandboxed folder
        let kit_dir = root.path().join("kit");
        std::fs::create_dir_all(&kit_dir).unwrap();
        let kick: Vec<f64> = (0..1800)
            .map(|i| {
                0.8 * (-(i as f64) / 250.0).exp()
                    * (2.0 * std::f64::consts::PI * 65.0 * (i as f64) / f64::from(sr)).sin()
            })
            .collect();
        AudioBuffer::from_channels(vec![kick.clone(), kick], sr)
            .unwrap()
            .save_wav(kit_dir.join("kick.wav"), 24)
            .unwrap();

        // 1. Beat track & evaluate quality via pure-Rust fallbacks inside sandbox
        let beats = call_tool(
            &sb,
            "forge_beat_track",
            json!({ "input": "ai_song.wav", "hint_bpm": 115.0, "backend": "rust" }),
        );
        assert!(!is_error(&beats), "{}", text(&beats));
        let beats_json: Value = serde_json::from_str(text(&beats)).unwrap();
        assert_eq!(beats_json["backend"], "pure_rust_onset");
        assert!(!beats_json["beats_secs"].as_array().unwrap().is_empty());

        let eval = call_tool(
            &sb,
            "forge_evaluate_quality",
            json!({ "input": "ai_song.wav", "reference": "ai_song.wav", "vocal": true, "backend": "rust" }),
        );
        assert!(!is_error(&eval), "{}", text(&eval));
        let eval_json: Value = serde_json::from_str(text(&eval)).unwrap();
        assert_eq!(eval_json["backend"], "rust-acoustic-proxy");

        // 2. Sandbox traversal rejection across new path arguments
        for (tool, args) in [
            (
                "forge_neural_restore",
                json!({ "input": "../escape.wav", "output_dir": "out" }),
            ),
            (
                "forge_vocal_rescue",
                json!({ "input": "ai_song.wav", "output_dir": "out", "reference_voice": "../escape.wav" }),
            ),
            (
                "forge_evaluate_quality",
                json!({ "input": "ai_song.wav", "reference": "../escape.wav" }),
            ),
            (
                "forge_ingest_to_reaper",
                json!({ "input": "ai_song.wav", "output_dir": "sess", "drum_samples_dir": "../escape_kit" }),
            ),
        ] {
            let err = call_tool(&sb, tool, args);
            assert!(is_error(&err), "{tool} must reject parent escape");
        }

        // 3. forge_deliver with ai_track_mode + hpss_shimmer
        let delivered = call_tool(
            &sb,
            "forge_deliver",
            json!({
                "input": "ai_song.wav",
                "output": "ai_master.wav",
                "ai_track_mode": true,
                "hpss_shimmer": 0.4
            }),
        );
        assert!(!is_error(&delivered), "{}", text(&delivered));

        // 4. forge_ingest_to_reaper (DSP fallback -> stem re-sum + drum layer + <TEMPOENVEX>)
        let ingested = call_tool(
            &sb,
            "forge_ingest_to_reaper",
            json!({
                "input": "ai_song.wav",
                "output_dir": "reaper_sess",
                "bpm": 115.0,
                "drum_samples_dir": "kit",
                "drum_replace_mix": 0.5
            }),
        );
        assert!(!is_error(&ingested), "{}", text(&ingested));
        let ingest_json: Value = serde_json::from_str(text(&ingested)).unwrap();
        assert!(ingest_json["beat_grid"]["beats_detected"].as_u64().unwrap() > 0);
        assert!(
            root.path()
                .join("reaper_sess/ForgeAudio_Session.rpp")
                .is_file()
        );
        let rpp_txt =
            std::fs::read_to_string(root.path().join("reaper_sess/ForgeAudio_Session.rpp"))
                .unwrap();
        if ingest_json["beat_grid"]["max_bpm_drift"]
            .as_f64()
            .unwrap_or(0.0)
            >= 1.5
        {
            assert!(
                rpp_txt.contains("<TEMPOENVEX"),
                "expected <TEMPOENVEX> in .rpp"
            );
        }
    }
}
