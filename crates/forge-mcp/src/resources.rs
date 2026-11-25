//! MCP resources (`forge://targets`, `forge://models`, `forge://roots`) and workflow prompts.

use crate::sandbox::PathSandbox;
use forge_dsp::DeliveryTarget;
use forge_dsp::delivery::MIN_DELIVERY_SAMPLE_RATE;
use rmcp::model::{
    ErrorData, GetPromptResult, JsonObject, Prompt, PromptArgument, PromptMessage,
    ReadResourceResult, Resource, ResourceContents, Role,
};
use serde_json::{Value, json};

pub(crate) fn list_resources() -> Vec<Resource> {
    vec![
        Resource::new("forge://targets", "Delivery Targets")
            .with_title("Mastering & Distribution Delivery Targets")
            .with_description(
                "LUFS, true-peak ceiling, tolerance, and sample-rate policy for all supported distribution targets.",
            )
            .with_mime_type("application/json"),
        Resource::new("forge://models", "AI Audio Models")
            .with_title("ForgeAudio Neural Restoration & Separation Models")
            .with_description(
                "Verified checkpoints and backend capabilities for stem separation, denoising, neural bandwidth restoration, vocal rescue, beat tracking, and perceptual quality evaluation.",
            )
            .with_mime_type("application/json"),
        Resource::new("forge://roots", "Sandbox Roots")
            .with_title("Authorized PathSandbox Roots")
            .with_description(
                "Canonical filesystem root directories permitted for anchored MCP file I/O.",
            )
            .with_mime_type("application/json"),
        Resource::new("forge://apps/audit-dashboard", "Audio Audit Dashboard App")
            .with_title("ForgeAudio Telemetry & Loudness Dashboard")
            .with_description(
                "Interactive MCP Apps (ext-apps) HTML dashboard visualizing BS.1770 LUFS, dBTP, LRA, M/S fizz, and spectral bandwidth.",
            )
            .with_mime_type("text/html;profile=mcp-app"),
    ]
}

fn targets_payload() -> Value {
    let presets: Vec<Value> = DeliveryTarget::NAMED
        .iter()
        .map(|target| {
            json!({
                "name": target.name(),
                "loudness_lufs": target.loudness_lufs(),
                "tolerance_lu": target.loudness_tolerance_lu(),
                "ceiling_dbtp": target.ceiling_dbtp(),
                "min_sample_rate_hz": MIN_DELIVERY_SAMPLE_RATE,
            })
        })
        .collect();
    json!({
        "targets": presets,
        "custom_supported": true,
        "custom_requires": ["target_lufs", "ceiling_dbtp"],
    })
}

fn models_payload() -> Value {
    json!({
        "stem_separation": {
            "fast": "vocals_mel_band_roformer.ckpt",
            "studio": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            "six_stem": "BS-Roformer-SW.ckpt",
            "dereverb": "deverb_bs_roformer_8_384dim_10depth.ckpt",
            "dereverb_echo": "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt",
            "denoise": "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
            "debleed": "mel_band_roformer_bleed_suppressor_v1.ckpt",
            "de_aspiration": "aspiration_mel_band_roformer_sdr_18.9845.ckpt",
            "vocal_revive": "bs_roformer_vocals_revive_v3e_unwa.ckpt",
            "drumsep": "MDX23C-DrumSep-aufr33-jarredou.ckpt",
            "karaoke": "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        },
        "backends": {
            "neural_denoise": ["auto", "deepfilternet3", "mel-band-roformer"],
            "neural_restore": ["auto", "apollo", "flowhigh", "audiosr", "sonicmaster", "exciter"],
            "vocal_rescue": ["auto", "resemble-enhance", "seed-vc", "roformer-clean"],
            "beat_track": ["auto", "beat-this", "librosa", "energy", "rust"],
            "evaluate_quality": ["auto", "audiobox", "singmos", "acoustic-proxy", "rust"],
            "f0_track": ["auto", "rmvpe", "fcpe", "autocorr"],
        }
    })
}

fn roots_payload(sandbox: &PathSandbox) -> Value {
    let roots: Vec<String> = sandbox.roots().map(|p| p.display().to_string()).collect();
    json!({ "roots": roots })
}

fn audit_dashboard_html() -> &'static str {
    r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>ForgeAudio Audit Dashboard</title>
<style>
  :root { color-scheme: dark; --bg:#0b1120; --card:#131c31; --fg:#e2e8f0; --muted:#94a3b8; --accent:#38bdf8; --warn:#fbbf24; }
  body { margin:0; padding:16px; font-family:ui-sans-serif,system-ui,sans-serif; background:var(--bg); color:var(--fg); }
  h1 { margin:0 0 8px; font-size:16px; color:var(--accent); letter-spacing:.04em; text-transform:uppercase; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:10px; margin-top:12px; }
  .card { background:var(--card); border:1px solid #24324d; border-radius:8px; padding:10px 12px; }
  .lbl { font-size:11px; color:var(--muted); text-transform:uppercase; }
  .val { font-size:20px; font-weight:700; margin-top:4px; font-variant-numeric:tabular-nums; }
  #status { font-size:12px; color:var(--muted); margin-top:10px; }
</style>
</head>
<body>
  <h1>ForgeAudio Studio Telemetry</h1>
  <div class="grid">
    <div class="card"><div class="lbl">Integrated LUFS</div><div class="val" id="lufs">--</div></div>
    <div class="card"><div class="lbl">True Peak (dBTP)</div><div class="val" id="tp">--</div></div>
    <div class="card"><div class="lbl">LRA (LU)</div><div class="val" id="lra">--</div></div>
    <div class="card"><div class="lbl">AI Fizz Index</div><div class="val" id="fizz">--</div></div>
    <div class="card"><div class="lbl">M/S Corr (2-12k)</div><div class="val" id="corr">--</div></div>
    <div class="card"><div class="lbl">Bandwidth (Hz)</div><div class="val" id="bw">--</div></div>
  </div>
  <div id="status">Waiting for MCP tool telemetry payload via postMessage...</div>
<script>
window.addEventListener("message", (ev) => {
  const d = (ev.data && (ev.data.structuredContent || ev.data.params || ev.data)) || {};
  const fmt = (v, n=2) => (typeof v === "number" && isFinite(v)) ? v.toFixed(n) : "--";
  document.getElementById("lufs").textContent = fmt(d.integrated_lufs, 2);
  document.getElementById("tp").textContent = fmt(d.true_peak_dbtp, 2);
  document.getElementById("lra").textContent = fmt(d.lra_lu ?? d.loudness_range_lu, 2);
  document.getElementById("fizz").textContent = fmt(d.ai_fizz_index, 1);
  document.getElementById("corr").textContent = fmt(d.stereo_correlation ?? d.high_mid_stereo_correlation, 2);
  document.getElementById("bw").textContent = fmt(d.bandwidth_cutoff_hz ?? d.effective_bandwidth_hz, 0);
  document.getElementById("status").textContent = "Updated live from ForgeAudio MCP telemetry.";
});
</script>
</body>
</html>"#
}

pub(crate) fn read_resource(
    uri: &str,
    sandbox: &PathSandbox,
) -> Result<ReadResourceResult, ErrorData> {
    if uri == "forge://apps/audit-dashboard" {
        return Ok(ReadResourceResult::new(vec![
            ResourceContents::text(audit_dashboard_html(), uri)
                .with_mime_type("text/html;profile=mcp-app"),
        ]));
    }
    let payload = match uri {
        "forge://targets" => targets_payload(),
        "forge://models" => models_payload(),
        "forge://roots" => roots_payload(sandbox),
        _ => {
            return Err(ErrorData::resource_not_found(
                format!("Unknown resource URI: {uri}"),
                None,
            ));
        }
    };
    let text = serde_json::to_string_pretty(&payload)
        .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
    Ok(ReadResourceResult::new(vec![
        ResourceContents::text(text, uri).with_mime_type("application/json"),
    ]))
}

fn arg_def(name: &str, desc: &str, required: bool) -> PromptArgument {
    PromptArgument::new(name)
        .with_description(desc)
        .with_required(required)
}

pub(crate) fn list_prompts() -> Vec<Prompt> {
    vec![
        Prompt::new(
            "master_ai_track",
            Some("Audit, repair AI fizz/drift, and deliver a verified master WAV."),
            Some(vec![
                arg_def("input_wav", "Input AI-generated audio path inside sandbox", true),
                arg_def("target", "Distribution target (default: spotify)", false),
                arg_def("genre", "Optional genre tonal target (pop, hiphop, neosoul, edm, rock, jazz)", false),
            ]),
        )
        .with_title("Master AI Track Workflow"),
        Prompt::new(
            "rescue_ai_vocals",
            Some("Run surgical vocal cleanup, OpenSoothe de-fizz, OpenSpacer unmasking, and harmony tuning."),
            Some(vec![
                arg_def("vocal_wav", "Isolated AI vocal WAV inside sandbox", true),
                arg_def("backing_wav", "Optional backing bed WAV to unmask for the vocal", false),
                arg_def("key", "Musical key/scale for vocal tuning (default: D minor)", false),
            ]),
        )
        .with_title("Rescue AI Vocals Workflow"),
        Prompt::new(
            "prepare_reaper_session",
            Some("Separate stems, align beat grid / pitch envelopes, and assemble a REAPER 7.82 .RPP project."),
            Some(vec![
                arg_def("input_audio", "Input mix audio path inside sandbox", true),
                arg_def("output_dir", "Destination directory for stems, MIDI, master, and .RPP", true),
                arg_def("bpm", "Optional initial tempo BPM (default: 120)", false),
            ]),
        )
        .with_title("Prepare REAPER 7.82 Session"),
    ]
}

fn prompt_arg<'a>(args: Option<&'a JsonObject>, key: &str) -> Option<&'a str> {
    args.and_then(|map| map.get(key))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

fn require_prompt_arg<'a>(args: Option<&'a JsonObject>, key: &str) -> Result<&'a str, ErrorData> {
    prompt_arg(args, key).ok_or_else(|| {
        ErrorData::invalid_params(format!("Missing required prompt argument '{key}'"), None)
    })
}

pub(crate) fn get_prompt(
    name: &str,
    args: Option<&JsonObject>,
) -> Result<GetPromptResult, ErrorData> {
    let (description, body) = match name {
        "master_ai_track" => {
            let input_wav = require_prompt_arg(args, "input_wav")?;
            let target = prompt_arg(args, "target").unwrap_or("spotify");
            let genre = prompt_arg(args, "genre").unwrap_or("pop");
            (
                "End-to-end AI track inspection, de-fizz restoration, and verified mastering workflow",
                format!(
                    "Master `{input_wav}` for target `{target}` (genre `{genre}`):\n\
                     1. Call `forge_audit_audio` on `{input_wav}` to inspect integrated LUFS, True Peak, M/S fizz, and AI Fizz Index.\n\
                     2. Call `forge_beat_track` on `{input_wav}` to measure median BPM and tempo drift.\n\
                     3. Call `forge_restore_and_defizz` with `ai_mode: true` if fizz or clipping is elevated, or run `forge_deliver` directly with `ai_track_mode: true`, `target: \"{target}\"`, and `genre: \"{genre}\"`.\n\
                     4. Verify `ready: true` and all checks in the returned delivery report."
                ),
            )
        }
        "rescue_ai_vocals" => {
            let vocal_wav = require_prompt_arg(args, "vocal_wav")?;
            let key = prompt_arg(args, "key").unwrap_or("D minor");
            let backing_step = match prompt_arg(args, "backing_wav") {
                Some(backing) => format!(
                    "3. Call `forge_spectral_unmask` with `target: \"{backing}\"` and sidechain set to the cleaned vocal.\n"
                ),
                None => String::new(),
            };
            (
                "AI vocal artifact removal, spectral de-fizz, unmasking, and harmony tuning workflow",
                format!(
                    "Rescue `{vocal_wav}` in `{key}`:\n\
                     1. Run `forge_vocal_rescue` if neural dereverb/timbre repair is available, or `forge_restore_and_defizz` with `vocal: true` and `ai_mode: true`.\n\
                     2. Run `forge_vocal_suite` on the cleaned vocal with `scale: \"{key}\"` to generate tuned lead + harmonies and a REAPER `<PITCHENV>` envelope.\n\
                     {backing_step}4. Audit the final vocal with `forge_audit_audio`."
                ),
            )
        }
        "prepare_reaper_session" => {
            let input_audio = require_prompt_arg(args, "input_audio")?;
            let output_dir = require_prompt_arg(args, "output_dir")?;
            let bpm = prompt_arg(args, "bpm").unwrap_or("120");
            (
                "Full autonomous stem separation, MIDI trigger/chord extraction, and REAPER 7.82 .RPP build",
                format!(
                    "Prepare a REAPER 7.82 session from `{input_audio}` into `{output_dir}` at {bpm} BPM:\n\
                     1. Read `forge://models` and `forge://targets` if selecting custom model/target settings.\n\
                     2. Call `forge_ingest_to_reaper` with `input: \"{input_audio}\"`, `output_dir: \"{output_dir}\"`, `bpm: {bpm}`.\n\
                     3. Summarize the generated `.RPP`, delivered master LUFS/dBTP checks, beat-grid drift, and installed JSFX status."
                ),
            )
        }
        _ => {
            return Err(ErrorData::invalid_params(
                format!("Unknown prompt: {name}"),
                None,
            ));
        }
    };

    Ok(
        GetPromptResult::new(vec![PromptMessage::new_text(Role::User, body)])
            .with_description(description),
    )
}
