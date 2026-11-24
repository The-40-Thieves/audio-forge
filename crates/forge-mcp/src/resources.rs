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
            "fast": "mel_band_roformer_vocals_fv4_gabox.ckpt",
            "studio": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            "six_stem": "bs_roformer_sw_fixed.ckpt",
            "dereverb": "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
            "dereverb_echo": "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt",
            "debleed": "mel_band_roformer_bleed_suppressor_v1.ckpt",
            "de_aspiration": "aspiration_mel_band_roformer_less_aggr_sdr_18.1201.ckpt",
            "drumsep": "MDX23C-8KFFT-InstVoc_HQ_DrumSep_5stem.ckpt",
            "karaoke": "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        },
        "backends": {
            "neural_denoise": ["auto", "deepfilternet3", "mel-band-roformer"],
            "neural_restore": ["auto", "apollo", "flowhigh", "audiosr", "sonicmaster", "exciter"],
            "vocal_rescue": ["auto", "resemble-enhance", "seed-vc", "roformer-clean"],
            "beat_track": ["auto", "beat-this", "librosa", "energy", "rust"],
            "evaluate_quality": ["auto", "audiobox", "singmos", "acoustic-proxy", "rust"],
            "f0_track": ["auto", "rmvpe", "torchcrepe", "librosa-yin", "autocorr"],
        }
    })
}

fn roots_payload(sandbox: &PathSandbox) -> Value {
    let roots: Vec<String> = sandbox.roots().map(|p| p.display().to_string()).collect();
    json!({ "roots": roots })
}

pub(crate) fn read_resource(
    uri: &str,
    sandbox: &PathSandbox,
) -> Result<ReadResourceResult, ErrorData> {
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
