//! MCP tool schemas (`inputSchema`, `outputSchema`, `ToolAnnotations`) for all 18 ForgeAudio tools.

use forge_dsp::DeliveryTarget;
use rmcp::model::{JsonObject, Tool, ToolAnnotations};
use serde::Serialize;
use serde_json::{Map, Value, json};
use std::sync::Arc;

#[derive(Clone, Copy)]
enum ToolKind {
    ReadOnly,
    ReadOpenWorld,
    WriteLocal,
    WriteExternal,
}

fn annotations(kind: ToolKind, title: &str) -> ToolAnnotations {
    let (read_only, idempotent, open_world) = match kind {
        ToolKind::ReadOnly => (true, true, false),
        ToolKind::ReadOpenWorld => (true, true, true),
        ToolKind::WriteLocal => (false, true, false),
        ToolKind::WriteExternal => (false, false, true),
    };
    ToolAnnotations::with_title(title)
        .read_only(read_only)
        .destructive(false)
        .idempotent(idempotent)
        .open_world(open_world)
}

fn schema_prop(ty: &str, default: Option<Value>, enum_vals: Option<Value>, desc: &str) -> Value {
    let mut map = Map::new();
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

fn str_fields(names: &[&'static str]) -> Vec<(&'static str, Value)> {
    names.iter().map(|&k| (k, prop_str(None, ""))).collect()
}

fn with_props(
    mut base: Vec<(&'static str, Value)>,
    extra: Vec<(&'static str, Value)>,
) -> Vec<(&'static str, Value)> {
    base.extend(extra);
    base
}

fn mid_side_prop() -> Value {
    prop_num(
        Some(0.0),
        "Mid/Side balance (-1.0=Mid only, 0.0=linked, +1.0=Side only)",
    )
}

fn object_schema(
    props: &[(&str, Value)],
    required: Option<&[&str]>,
    additional_properties: Option<bool>,
) -> Arc<JsonObject> {
    let mut properties = Map::new();
    for (k, v) in props {
        properties.insert((*k).to_string(), v.clone());
    }
    let mut root = Map::new();
    root.insert("type".to_string(), json!("object"));
    root.insert("properties".to_string(), Value::Object(properties));
    if let Some(req) = required {
        root.insert("required".to_string(), json!(req));
    }
    if let Some(add) = additional_properties {
        root.insert("additionalProperties".to_string(), json!(add));
    }
    Arc::new(root)
}

fn out_fields(fields: &[(&str, &str)]) -> Arc<JsonObject> {
    let props: Vec<(&str, Value)> = fields
        .iter()
        .map(|&(name, ty)| (name, schema_prop(ty, None, None, "")))
        .collect();
    object_schema(&props, None, Some(true))
}

fn out_audit() -> Arc<JsonObject> {
    out_fields(&[
        ("integrated_lufs", "number"),
        ("true_peak_dbtp", "number"),
        ("lra_lu", "number"),
        ("ai_fizz_index", "number"),
        ("stereo_correlation", "number"),
        ("bandwidth_cutoff_hz", "number"),
        ("transient_sharpness", "number"),
    ])
}

fn out_backend_output() -> Arc<JsonObject> {
    out_fields(&[("backend", "string"), ("output", "string")])
}

fn delivery_target_props(targets: &[&str]) -> Vec<(&'static str, Value)> {
    vec![
        ("target", prop_enum("string", targets, json!("spotify"), "")),
        (
            "target_lufs",
            prop_num(None, "Required when target is 'custom'"),
        ),
        (
            "ceiling_dbtp",
            prop_num(None, "Required when target is 'custom'"),
        ),
    ]
}

fn mcp_tool(
    name: &'static str,
    title: &str,
    description: &'static str,
    kind: ToolKind,
    props: &[(&str, Value)],
    required: &[&str],
    output_schema: Arc<JsonObject>,
) -> Tool {
    Tool::new(
        name,
        description,
        object_schema(props, Some(required), None),
    )
    .with_title(title)
    .with_raw_output_schema(output_schema)
    .with_annotations(annotations(kind, title))
}

fn req_str_tool(
    name: &'static str,
    title: &str,
    description: &'static str,
    kind: ToolKind,
    required: &[&'static str],
    extra: Vec<(&'static str, Value)>,
    output_schema: Arc<JsonObject>,
) -> Tool {
    let props = with_props(str_fields(required), extra);
    mcp_tool(
        name,
        title,
        description,
        kind,
        &props,
        required,
        output_schema,
    )
}

const DENOISE_BACKENDS: &[&str] = &[
    "auto",
    "deepfilternet",
    "deepfilternet3",
    "mel-band-roformer",
    "roformer",
];
const RESTORE_BACKENDS: &[&str] = &[
    "auto",
    "apollo",
    "flowhigh",
    "audiosr",
    "sonicmaster",
    "exciter",
];
const RESCUE_BACKENDS: &[&str] = &["auto", "resemble-enhance", "seed-vc", "roformer-clean"];
const BEAT_BACKENDS: &[&str] = &[
    "auto",
    "beat-this",
    "librosa",
    "energy",
    "rust",
    "pure_rust_onset",
];
const EVAL_BACKENDS: &[&str] = &["auto", "audiobox", "singmos", "acoustic-proxy", "rust"];

fn worker_tool(
    meta: (&'static str, &str, &'static str, ToolKind),
    output_dir_desc: Option<&str>,
    backends: &[&str],
    default_timeout_secs: i64,
    extra: Vec<(&'static str, Value)>,
    output_schema: Arc<JsonObject>,
) -> Tool {
    let (name, title, desc, kind) = meta;
    let mut props = vec![("input", prop_str(None, "Input audio file"))];
    let mut required = vec!["input"];
    if let Some(dir_desc) = output_dir_desc {
        props.push(("output_dir", prop_str(None, dir_desc)));
        required.push("output_dir");
    }
    props.push(("backend", prop_enum("string", backends, json!("auto"), "")));
    props.extend(extra);
    props.push((
        "timeout_secs",
        prop_int(
            Some(default_timeout_secs),
            "Worker timeout in seconds (1..=3600)",
        ),
    ));
    mcp_tool(name, title, desc, kind, &props, &required, output_schema)
}

pub(crate) fn delivery_target_names() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = DeliveryTarget::NAMED.iter().map(|t| t.name()).collect();
    names.push("custom");
    names
}

fn build_separate_props() -> Vec<(&'static str, Value)> {
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
    separate_props
}

pub(crate) fn build_tools() -> Vec<Tool> {
    let targets = delivery_target_names();
    let separate_props = build_separate_props();

    let restore_props = with_props(
        str_fields(&["input", "output"]),
        vec![
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
    );

    let deliver_props = with_props(
        with_props(
            vec![
                ("input", prop_str(None, "Input mix or stem (WAV)")),
                ("output", prop_str(None, "Output master WAV path")),
            ],
            delivery_target_props(&targets),
        ),
        vec![
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
    );

    let ingest_props = with_props(
        with_props(
            str_fields(&["input", "output_dir"]),
            delivery_target_props(&targets),
        ),
        vec![
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
                "backend",
                prop_enum(
                    "string",
                    &["auto", "local", "colab", "rust"],
                    json!("auto"),
                    "Stem separation backend override (defaults to env FORGE_WORKER_BACKEND or auto)",
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
    );

    let defizz_out = out_fields(&[
        ("output", "string"),
        ("before_fizz_index", "number"),
        ("after_fizz_index", "number"),
        ("after_lufs", "number"),
    ]);
    let unmask_extra = vec![
        ("kick_bass_mode", prop_bool(false, "")),
        ("mid_side", mid_side_prop()),
        (
            "delta",
            prop_bool(false, "Audition only the carved sidechain delta"),
        ),
    ];
    let unmask_out = out_fields(&[
        ("output", "string"),
        ("kick_bass_mode", "boolean"),
        ("mid_side", "number"),
        ("delta", "boolean"),
    ]);

    vec![
        mcp_tool(
            "forge_audit_audio",
            "Audit Audio (BS.1770-4 & AI Fizz)",
            "Run an ITU-R BS.1770-4 / EBU R128 loudness, 4x polyphase True Peak (dBTP), LRA, Mid/Side fizz, bandwidth cutoff, transient sharpness, and AI Spectral Fizz Index audit on an audio file.",
            ToolKind::ReadOnly,
            &[("path", prop_str(None, "Path to audio file"))],
            &["path"],
            out_audit(),
        ),
        mcp_tool(
            "forge_restore_and_defizz",
            "Restore Audio & OpenSoothe De-Fizz",
            "Clean AI-generated or live audio: DC offset, cubic Hermite de-clip, 50/60Hz hum removal, vocal de-click/de-plosive/de-breath, HPSS shimmer suppression, and OpenSoothe dynamic spectral de-fizz.",
            ToolKind::WriteLocal,
            &restore_props,
            &["input", "output"],
            defizz_out,
        ),
        req_str_tool(
            "forge_spectral_unmask",
            "OpenSpacer Spectral Unmasking",
            "Run OpenSpacer (Trackspacer equivalent) sidechain STFT dynamic frequency unmasking between a sidechain trigger (vocal or kick) and target bed (music or bass).",
            ToolKind::WriteLocal,
            &["target", "sidechain", "output"],
            unmask_extra,
            unmask_out,
        ),
        req_str_tool(
            "forge_phase_align",
            "Phase & Vocal Double Alignment",
            "Align multi-mic phase via GCC-PHAT (Auto-Align equivalent) or vocal doubles via FastDTW stretch markers (VocAlign equivalent).",
            ToolKind::WriteLocal,
            &["reference", "target", "output"],
            vec![
                ("vocal_dtw", prop_bool(false, "")),
                (
                    "elastic",
                    prop_bool(
                        false,
                        "Use WSOLA elastic stretch warping when vocal_dtw is true",
                    ),
                ),
            ],
            out_fields(&[("mode", "string"), ("output", "string")]),
        ),
        req_str_tool(
            "forge_vocal_suite",
            "Vocal Pitch Surgery & Harmonies",
            "Run YIN pitch tracking, formant-preserving pitch correction, 3-part chord-aware vocal harmony generation, and REAPER <PITCHENV> compilation.",
            ToolKind::WriteLocal,
            &["input", "output_dir"],
            vec![
                ("scale", prop_str(Some("D minor"), "")),
                ("retune_strength", prop_num(Some(0.85), "")),
                ("generate_harmonies", prop_bool(true, "")),
            ],
            out_fields(&[
                ("tuned_vocal", "string"),
                ("harmonies", "array"),
                ("frames_tracked", "integer"),
                ("reaper_pitch_envelope", "string"),
            ]),
        ),
        req_str_tool(
            "forge_music_theory_midi",
            "Music Theory Chords & Groove MIDI",
            "Generate voice-led MIDI chords, Neo-Soul/Jazz reharmonizations, Dilla/MPC humanized grooves, and ABC notation.",
            ToolKind::WriteLocal,
            &["chords", "output_midi"],
            vec![
                ("bpm", prop_num(Some(92.0), "")),
                ("style", prop_str(Some("neo-soul"), "")),
            ],
            out_fields(&[
                ("output_midi", "string"),
                ("abc_notation", "string"),
                ("reharmonized_chords", "array"),
            ]),
        ),
        mcp_tool(
            "forge_master_distrokid",
            "Master & ISRC Tag Audio",
            "Master stereo audio to -14.0 LUFS / -1.0 dBTP with Reference Match EQ, 18-band Mastering EQ, and optional ISO 3901 ISRC tagging.",
            ToolKind::WriteLocal,
            &with_props(
                str_fields(&["input", "output", "reference", "isrc", "title", "artist"]),
                vec![
                    ("target_lufs", prop_num(Some(-14.0), "")),
                    ("ceiling_dbtp", prop_num(Some(-1.0), "")),
                ],
            ),
            &["input", "output"],
            out_audit(),
        ),
        mcp_tool(
            "forge_deliver",
            "Deliver Distribution Master",
            "Deliver a mix to a distribution target in one call: audit, repair (DC, clipping, hum), de-fizz, optional reference/genre tone match, loudness-normalize and true-peak limit, then re-audit. Returns before/after BS.1770-4 audits and explicit pass/fail checks (loudness within 1 LU, true peak under ceiling, no clipping, sample rate >= 44.1 kHz). Writes a 24-bit WAV and a .delivery.json report next to it.",
            ToolKind::WriteLocal,
            &deliver_props,
            &["input", "output"],
            out_fields(&[
                ("output", "string"),
                ("report_path", "string"),
                ("report", "object"),
            ]),
        ),
        worker_tool(
            (
                "forge_neural_denoise",
                "Neural Denoise (DeepFilterNet3 / RoFormer)",
                "Run neural audio denoising via DeepFilterNet3 or Mel-Band-RoFormer on the local Python GPU worker, confined to the MCP path sandbox.",
                ToolKind::WriteExternal,
            ),
            Some("Output directory for denoised audio"),
            DENOISE_BACKENDS,
            300,
            vec![(
                "attenuation_limit_db",
                prop_num(
                    None,
                    "Optional attenuation limit in dB (0..=100, DeepFilterNet3)",
                ),
            )],
            out_backend_output(),
        ),
        mcp_tool(
            "forge_transcribe_midi",
            "Polyphonic Audio-to-MIDI Transcription",
            "Run polyphonic audio-to-MIDI transcription via basic-pitch on the local Python GPU worker, confined to the MCP path sandbox.",
            ToolKind::WriteExternal,
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
            out_fields(&[("backend", "string"), ("midi", "string")]),
        ),
        mcp_tool(
            "forge_separate_stems",
            "Neural Stem Separation Cascade",
            "Studio-grade neural stem separation (BS-RoFormer 1297 / Mel-Band-RoFormer / BS-RoFormer SW cascade, 2/4/6 stems + de-reverb/de-echo/debleed/de-aspiration, Karaoke Lead/Backing split, 5-piece MDX23C DrumSep & bandwidth restoration) with pure-Rust DSP fallback, confined to the MCP path sandbox.",
            ToolKind::WriteExternal,
            &separate_props,
            &["input", "output_dir"],
            out_fields(&[
                ("backend", "string"),
                ("bundle", "object"),
                ("fallbacks", "array"),
                ("warnings", "array"),
            ]),
        ),
        mcp_tool(
            "forge_daw_inspect",
            "Inspect ForgeStudio Session (.forgeproj)",
            "Validate and inspect a ForgeStudio DAW `.forgeproj` session (BPM, sample rate, tracks, audio objects, MIDI clips, aux buses, total PDC) within the MCP path sandbox.",
            ToolKind::ReadOnly,
            &[(
                "project",
                prop_str(None, "Path to `.forgeproj` session file"),
            )],
            &["project"],
            out_fields(&[
                ("name", "string"),
                ("bpm", "number"),
                ("sample_rate", "integer"),
                ("tracks", "array"),
                ("total_pdc_samples", "integer"),
            ]),
        ),
        mcp_tool(
            "forge_daw_bounce",
            "Bounce ForgeStudio Session (.forgeproj)",
            "Headless 24-bit WAV mastering bounce (with ITU-R BS.1770-4 audit) and optional batch stem export from a ForgeStudio DAW `.forgeproj` session within the MCP path sandbox.",
            ToolKind::WriteLocal,
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
            out_fields(&[
                ("output", "string"),
                ("integrated_lufs", "number"),
                ("true_peak_dbtp", "number"),
                ("stems_exported", "integer"),
            ]),
        ),
        mcp_tool(
            "forge_ingest_to_reaper",
            "Autonomous AI-Track REAPER Ingest",
            "Full autonomous AI-track pipeline: separate stems, clean & de-fizz vocal/other stems, unmask instruments, optionally layer drum one-shots, track beat grid / quantize stems or emit <TEMPOENVEX>, re-sum clean stems and deliver a verified master, install JSFX, and generate a REAPER 7.82 .RPP project.",
            ToolKind::WriteExternal,
            &ingest_props,
            &["input", "output_dir"],
            out_fields(&[
                ("rpp_project", "string"),
                ("master_wav", "string"),
                ("midi_file", "string"),
                ("delivery_report", "string"),
                ("target", "string"),
                ("ready", "boolean"),
                ("master_lufs", "number"),
                ("master_true_peak_dbtp", "number"),
                ("ai_fizz_index", "number"),
                ("drum_hits_layered", "integer"),
            ]),
        ),
        worker_tool(
            (
                "forge_neural_restore",
                "Neural Bandwidth Restoration (Apollo / AudioSR)",
                "Run neural bandwidth restoration and AI artifact cleanup (Apollo, FlowHigh, AudioSR, SonicMaster, or spectral exciter fallback) within the MCP path sandbox.",
                ToolKind::WriteExternal,
            ),
            Some("Output directory for restored audio"),
            RESTORE_BACKENDS,
            600,
            vec![
                (
                    "prompt",
                    prop_str(None, "Optional instruction prompt (used by SonicMaster)"),
                ),
                (
                    "chunk_secs",
                    prop_num(None, "Processing chunk length in seconds (0.5..=120.0)"),
                ),
            ],
            out_backend_output(),
        ),
        worker_tool(
            (
                "forge_vocal_rescue",
                "Neural AI Vocal Rescue",
                "Rescue robotic or phasey AI vocals (Resemble Enhance, Seed-VC zero-shot timbre, or RoFormer chain) within the MCP path sandbox.",
                ToolKind::WriteExternal,
            ),
            Some("Output directory for rescued vocal"),
            RESCUE_BACKENDS,
            600,
            vec![
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
            out_backend_output(),
        ),
        worker_tool(
            (
                "forge_beat_track",
                "Beat & Tempo Drift Tracker",
                "Extract beats, downbeats, median BPM, and tempo-drift map (beat-this, librosa, energy, or pure-Rust onset DP) within the MCP path sandbox.",
                ToolKind::ReadOpenWorld,
            ),
            None,
            BEAT_BACKENDS,
            300,
            vec![(
                "hint_bpm",
                prop_num(None, "Optional BPM hint (30.0..=300.0) for Rust fallback"),
            )],
            out_fields(&[
                ("backend", "string"),
                ("median_bpm", "number"),
                ("max_bpm_drift", "number"),
                ("beats_secs", "array"),
            ]),
        ),
        worker_tool(
            (
                "forge_evaluate_quality",
                "Perceptual Quality Assessment",
                "Evaluate perceptual audio quality (Meta Audiobox Aesthetics PQ/PC/CE/CU, SingMOS, or acoustic proxy) within the MCP path sandbox.",
                ToolKind::ReadOpenWorld,
            ),
            None,
            EVAL_BACKENDS,
            300,
            vec![
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
            out_fields(&[("backend", "string")]),
        ),
    ]
}
