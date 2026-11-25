//! Synchronous tool handlers executed inside the sandboxed MCP worker context.

use crate::external;
use crate::sandbox::PathSandbox;
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
use rmcp::model::{CallToolResult, ContentBlock};
use serde_json::{Value, json};
use std::any::Any;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Result of a tool invocation. `success` is false when the tool ran but its outcome
/// is a failure the caller must act on (e.g. a delivery that is not `ready`).
pub(crate) struct ToolOutput {
    pub(crate) value: Value,
    pub(crate) success: bool,
}

impl From<Value> for ToolOutput {
    fn from(value: Value) -> Self {
        Self {
            value,
            success: true,
        }
    }
}

pub(crate) fn panic_message(payload: &(dyn Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("unknown panic")
}

/// Runs one tool under panic containment and maps its result to an MCP `CallToolResult`
/// containing both pretty-printed JSON text and structured `structuredContent`.
pub(crate) fn call_tool_result(
    tool_name: &str,
    args: &Value,
    sandbox: &PathSandbox,
) -> CallToolResult {
    let outcome = panic::catch_unwind(AssertUnwindSafe(|| dispatch_tool(tool_name, args, sandbox)));
    match outcome {
        Ok(Ok(out)) => {
            let pretty = serde_json::to_string_pretty(&out.value).unwrap_or_default();
            let mut result = if out.success {
                CallToolResult::structured(out.value)
            } else {
                CallToolResult::structured_error(out.value)
            };
            result.content = vec![ContentBlock::text(pretty)];
            result
        }
        Ok(Err(err)) => {
            CallToolResult::error(vec![ContentBlock::text(format!("ForgeAudio Error: {err}"))])
        }
        Err(payload) => CallToolResult::error(vec![ContentBlock::text(format!(
            "ForgeAudio internal error: tool '{tool_name}' panicked: {}",
            panic_message(payload.as_ref())
        ))]),
    }
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

fn path_arg(args: &Value, key: &str, sandbox: &PathSandbox) -> Result<PathBuf, ForgeError> {
    match args.get(key) {
        None | Some(Value::Null) => Err(ForgeError::InvalidParameter(format!("Missing '{key}'"))),
        Some(value) => resolve_path_value(key, value, sandbox),
    }
}

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

type VoicedChords = (Vec<String>, Vec<NoteEvent>, Vec<NoteEvent>);

fn parse_voicing_style(style: &str) -> VoicingStyle {
    match style.trim().to_ascii_lowercase().as_str() {
        "jazz" | "jazz-drop2" | "drop2" => VoicingStyle::Drop2,
        "pop" | "triad" | "close" => VoicingStyle::Close,
        "cinematic" => VoicingStyle::CinematicSpread,
        _ => VoicingStyle::NeoSoulRootless,
    }
}

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

pub(crate) fn install_session_jsfx(resource_dir: Result<PathBuf, ForgeError>) -> Value {
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
    let backend_override = args.get("backend").and_then(Value::as_str);
    let (mut bundle, stem_separation) = separate_ingest_stems(
        sandbox,
        &input,
        &paths.stems_dir,
        require_neural,
        backend_override,
    )?;

    let mut primary_stems: Vec<(PathBuf, AudioBuffer)> = Vec::new();
    let (mut vocal_pitch_frames, cleaned_vocal) =
        ingest_vocal_frames(sandbox, bundle.vocals.as_deref(), &scale)?;
    let mut vocal_stem_idx: Option<usize> = None;
    if let (Some(voc_path), Some(voc_buf)) = (&bundle.vocals, &cleaned_vocal) {
        vocal_stem_idx = Some(primary_stems.len());
        primary_stems.push((sandbox.check(voc_path)?, voc_buf.clone()));
    }

    if let Some(ref other_path) = bundle.other {
        let checked = sandbox.check(other_path)?;
        let other_clean = clean_ingested_other(sandbox, &checked, cleaned_vocal.as_ref())?;
        primary_stems.push((checked, other_clean));
    }

    let mut drum_buf =
        sandbox.load_audio(&sandbox.check(bundle.drums.as_ref().unwrap_or(&input))?)?;
    let (kick_notes, snare_notes, hat_notes) = extract_drum_midi_triggers(&drum_buf, bpm, 480);

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
    let mut drum_stem_idx: Option<usize> = None;
    if let Some(ref drums_path) = bundle.drums {
        drum_stem_idx = Some(primary_stems.len());
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
    let mut drum_for_midi = if let Some(d_idx) = drum_stem_idx {
        primary_stems[d_idx].1.clone()
    } else {
        full_mix.clone()
    };
    if quantize_grid && beat_grid.beats_secs.len() >= 2 {
        let (synced_pitch, synced_drums) = quantize_stems_and_sync_assets(
            &mut primary_stems,
            vocal_stem_idx,
            drum_stem_idx,
            &beat_grid.beats_secs,
            bpm,
            &scale,
        )?;
        for (path, buf) in &primary_stems {
            sandbox.save_wav(buf, path, 24)?;
        }
        if vocal_stem_idx.is_some() {
            vocal_pitch_frames = synced_pitch;
        }
        drum_for_midi = match synced_drums {
            Some(_) => primary_stems[drum_stem_idx.expect("idx checked")].1.clone(),
            None => quantize_to_constant_bpm(&full_mix, &beat_grid.beats_secs, bpm)?.0,
        };
        quantized_applied = true;
    }
    let _ = write_ingest_midi(sandbox, &paths, &drum_for_midi, bpm, chords)?;

    let refs: Vec<&AudioBuffer> = primary_stems.iter().map(|(_, b)| b).collect();
    let (mastered, delivery) = forge_dsp::resum_and_deliver_ai_stems(&refs, &full_mix, target)?;
    sandbox.save_wav(&mastered, &paths.master, 24)?;
    sandbox.write_json(&paths.delivery, &delivery)?;

    let duration_secs = primary_stems
        .iter()
        .map(|(_, b)| buffer_duration_secs(b))
        .chain(std::iter::once(buffer_duration_secs(&mastered)))
        .fold(0.0_f64, f64::max)
        .max(buffer_duration_secs(&full_mix));
    let stem_duration = |path: Option<&PathBuf>| -> f64 {
        path.and_then(|p| {
            primary_stems
                .iter()
                .find(|(sp, _)| sp == p)
                .map(|(_, b)| buffer_duration_secs(b))
        })
        .unwrap_or(duration_secs)
    };

    let tempo_map = if !quantized_applied && beat_grid.max_bpm_drift >= 1.5 {
        beat_grid.tempo_map.clone()
    } else {
        Vec::new()
    };
    let session = SessionSpec {
        bpm,
        sample_rate: full_mix.sample_rate,
        duration_secs,
        vocal_duration_secs: stem_duration(bundle.vocals.as_ref()),
        drums_duration_secs: stem_duration(bundle.drums.as_ref()),
        bass_duration_secs: stem_duration(bundle.bass.as_ref()),
        other_duration_secs: stem_duration(bundle.other.as_ref()),
        guitar_duration_secs: stem_duration(bundle.guitar.as_ref()),
        keys_duration_secs: stem_duration(bundle.keys.as_ref()),
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

fn separate_ingest_stems(
    sandbox: &PathSandbox,
    input: &Path,
    stems_dir: &Path,
    require_neural: bool,
    backend_override: Option<&str>,
) -> Result<(StemBundle, Value), ForgeError> {
    let mut config = SeparationConfig::from_env();
    config.require_neural |= require_neural;
    if let Some(raw) = backend_override {
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" | "" => {}
            "local" | "local_gpu" => {
                config.backend = forge_worker::WorkerBackend::LocalGpu;
            }
            "rust" | "dsp" | "pure_rust_dsp" => {
                config.backend = forge_worker::WorkerBackend::PureRustDsp;
            }
            "colab" | "remote" | "remote_colab" => {
                let url = std::env::var("FORGE_COLAB_WORKER_URL")
                    .ok()
                    .map(|u| u.trim().to_string())
                    .filter(|u| !u.is_empty())
                    .ok_or_else(|| {
                        ForgeError::WorkerError(
                            "backend='colab' requires FORGE_COLAB_WORKER_URL to be set".to_string(),
                        )
                    })?;
                config.backend = forge_worker::WorkerBackend::RemoteColab { base_url: url };
            }
            other => {
                return Err(ForgeError::InvalidParameter(format!(
                    "Invalid ingest separation backend '{other}' (expected auto, local, colab, or rust)"
                )));
            }
        }
    }
    let report = external::separate_stems_checked(sandbox, input, stems_dir, &config)?;
    let info = json!({
        "backend": report.backend,
        "fallbacks": report.fallbacks,
        "warnings": report.warnings
    });
    Ok((report.bundle, info))
}

type DrumTriggerMidi = (Vec<NoteEvent>, Vec<NoteEvent>, Vec<NoteEvent>);

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

fn ingest_vocal_frames(
    sandbox: &PathSandbox,
    vocal: Option<&Path>,
    scale: &VocalScale,
) -> Result<(Vec<PitchFrame>, Option<AudioBuffer>), ForgeError> {
    match vocal {
        Some(voc_path) => clean_ingested_vocal(sandbox, &sandbox.check(voc_path)?, scale),
        None => Ok((Vec::new(), None)),
    }
}

fn clean_ingested_vocal(
    sandbox: &PathSandbox,
    voc_path: &Path,
    scale: &VocalScale,
) -> Result<(Vec<PitchFrame>, Option<AudioBuffer>), ForgeError> {
    let voc_buf = sandbox.load_audio(voc_path)?;
    let voc_clean = clean_vocal_surgical(&voc_buf, true, true, -10.0)?;
    let voc_defizz = apply_spectral_defizz(&voc_clean, &DeFizzConfig::ai_vocal())?;
    let frames = track_vocal_pitch(&voc_defizz, scale, 0.85);
    sandbox.save_wav(&voc_defizz, voc_path, 24)?;
    Ok((frames, Some(voc_defizz)))
}

fn clean_ingested_other(
    sandbox: &PathSandbox,
    other_path: &Path,
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

pub(crate) fn buffer_duration_secs(buf: &AudioBuffer) -> f64 {
    (buf.num_frames() as f64) / f64::from(buf.sample_rate.max(1))
}

pub(crate) fn quantize_stems_and_sync_assets(
    primary_stems: &mut [(PathBuf, AudioBuffer)],
    vocal_idx: Option<usize>,
    drum_idx: Option<usize>,
    beats_secs: &[f64],
    bpm: f64,
    scale: &VocalScale,
) -> Result<(Vec<PitchFrame>, Option<DrumTriggerMidi>), ForgeError> {
    for (_, buf) in primary_stems.iter_mut() {
        let (warped, _) = quantize_to_constant_bpm(buf, beats_secs, bpm)?;
        *buf = warped;
    }
    let pitch_frames = vocal_idx
        .map(|idx| track_vocal_pitch(&primary_stems[idx].1, scale, 0.85))
        .unwrap_or_default();
    let drum_midi = drum_idx.map(|idx| extract_drum_midi_triggers(&primary_stems[idx].1, bpm, 480));
    Ok((pitch_frames, drum_midi))
}

pub(crate) struct SessionSpec {
    pub(crate) bpm: f64,
    pub(crate) sample_rate: u32,
    pub(crate) duration_secs: f64,
    pub(crate) vocal_duration_secs: f64,
    pub(crate) drums_duration_secs: f64,
    pub(crate) bass_duration_secs: f64,
    pub(crate) other_duration_secs: f64,
    pub(crate) guitar_duration_secs: f64,
    pub(crate) keys_duration_secs: f64,
    pub(crate) vocal: Option<PathBuf>,
    pub(crate) drums: Option<PathBuf>,
    pub(crate) bass: Option<PathBuf>,
    pub(crate) other: Option<PathBuf>,
    pub(crate) guitar: Option<PathBuf>,
    pub(crate) keys: Option<PathBuf>,
    pub(crate) vocal_pitch_frames: Vec<PitchFrame>,
    pub(crate) tempo_map: Vec<forge_dsp::TempoPoint>,
    pub(crate) chords_midi: PathBuf,
    pub(crate) drum_midi: PathBuf,
}

pub(crate) fn build_session(spec: SessionSpec) -> ReaperProjectBuilder {
    let mut rpp =
        ReaperProjectBuilder::new("ForgeAudio Studio Session", spec.bpm, spec.sample_rate);
    rpp.tempo_map = spec.tempo_map;
    rpp.add_default_song_structure_markers();
    rpp.tracks.push(
        ReaperTrackSpec::audio(
            "01 Lead Vocal",
            spec.vocal,
            spec.vocal_duration_secs,
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
            spec.drums_duration_secs,
            0.90,
            0x01FF_5555,
        ));
    }
    if let Some(bass) = spec.bass {
        rpp.tracks.push(
            ReaperTrackSpec::audio(
                "03 Bass",
                Some(bass),
                spec.bass_duration_secs,
                0.88,
                0x01AA_66FF,
            )
            .with_jsfx(["Forge_OpenSpacer_Unmask.jsfx"])
            .with_sidechain(if has_drums { Some(1) } else { None }),
        );
    }
    if let Some(other) = spec.other {
        rpp.tracks.push(
            ReaperTrackSpec::audio(
                "04 Instruments / Other",
                Some(other),
                spec.other_duration_secs,
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
            spec.guitar_duration_secs,
            0.80,
            0x0144_AAFF,
        ));
    }
    if let Some(keys) = spec.keys {
        rpp.tracks.push(ReaperTrackSpec::audio(
            "08 Keys / Piano",
            Some(keys),
            spec.keys_duration_secs,
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
    let quality = forge_worker::parse_quality(str_arg_or(args, "quality", "studio"))?;
    let dereverb = bool_arg(args, "dereverb", true);

    let mut cfg = SeparationConfig::from_env();
    cfg.stems = stems;
    cfg.quality = quality;
    cfg.dereverb = dereverb;
    cfg.drum_sep = bool_arg(args, "drum_sep", false);
    cfg.split_lead_backing = bool_arg(args, "split_lead_backing", false);
    cfg.split_male_female = bool_arg(args, "split_male_female", false);
    cfg.split_guitar = bool_arg(args, "split_guitar", false);
    cfg.split_orchestral = bool_arg(args, "split_orchestral", false);
    cfg.split_crowd = bool_arg(args, "split_crowd", false);
    if bool_arg(args, "all_stems", false) {
        cfg.enable_all_stems();
        cfg.dereverb = dereverb;
    }
    cfg.restore = bool_arg(args, "restore", false);
    cfg.debleed = bool_arg(args, "debleed", false);
    cfg.de_aspiration = bool_arg(args, "de_aspiration", false);
    cfg.dereverb_echo = bool_arg(args, "dereverb_echo", false);

    let report = external::separate_stems_checked(sandbox, &input, &output_dir, &cfg)?;
    let payload = serde_json::to_value(&report)
        .map_err(|e| ForgeError::InvalidParameter(format!("Failed to serialize report: {e}")))?;
    Ok(payload.into())
}

fn load_sandboxed_daw_project(
    project_path: &Path,
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
