//! # `forge-plug` — Native Rust VST3 & CLAP Real-Time Studio Plugin Suite
#![allow(semicolon_in_expressions_from_non_local_macros)]
//!
//! Built on `nih-plug` and `forge-dsp::realtime` with **zero audio-thread heap allocations**
//! (`assert_process_allocs` enabled).
//!
//! Implements 10 VST3/CLAP studio processors (exported individually from `plugins/forge-*`,
//! or together as one bundle when `all-in-one-export` is enabled):
//! 1. **Forge OpenSoothe De-Fizz** (2048-pt OLA STFT Dynamic Spectral Resonance & AI Fizz Suppressor)
//! 2. **Forge OpenSpacer Unmask** (4-Channel Sidechain Dynamic Spectral Unmasker)
//! 3. **Forge Vocal Surgical Strip** (Zero-Latency Mouth De-Click, Plosive Suppressor, De-Esser & Breath Gate)
//! 4. **Forge SubMono & Console Tape** (Linkwitz-Riley `<160 Hz` Mono-Maker, Sub Reinforce & Console8/ChowTape Warmth)
//! 5. **Forge Master True-Peak Limiter** (4x Inter-Sample True-Peak `-1.0 dBTP` Lookahead Brickwall Limiter)
//! 6. **Forge Drum Trigger & Theory MIDI** (Real-Time Audio-to-MIDI Drum Replacer `C1/D1/F#1` + Neo-Soul Chord Voicer)
//! 7. **Forge Phase & Match EQ** (Phase/Polarity Aligner + 18-Band Mastering EQ & Genre Tonal Balance)
//! 8. **Forge SplitEQ** (3-Band Structural Transient/Tonal Equalizer & Shaper)
//! 9. **Forge Density & Clipper** (3-Band Upward Density Compressor & 2x ADAA Soft Clipper)
//! 10. **Forge SpaceVerb & Preamp** (8x8 Hadamard FDN Reverb with Ducking/Shimmer + Neural WaveNet Preamp)

mod ui;

use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

#[cfg(test)]
use forge_dsp::read_shared_spectrum;
use forge_dsp::{
    AnalogHardwareModel, AutoGainCompensator, DitherMode, DrumTriggerNotes, GenreTonalTarget,
    IspOversampling, LimiterStyle, NoteMsg, RealtimeDensityAndClipper, RealtimeDrumTrigger,
    RealtimeFdnReverb, RealtimeMasterLimiter, RealtimeMasteringEq18, RealtimeNeuralPreamp,
    RealtimeOpenSoothe, RealtimeOpenSpacer, RealtimePhaseAligner, RealtimeSpectrumAnalyzer32,
    RealtimeSubAndTapeBus, RealtimeTransientTonalSplit, RealtimeVocalSurgical, ReverbRoomGeometry,
    SpectrumBusOwner, clear_shared_spectrum, publish_shared_spectrum, shared_spectrum_slot_index,
};
use forge_vocal::{RealtimeVocalTune, VocalTuneScaleMode};
use nih_plug::prelude::*;
use nih_plug_egui::{EguiState, create_egui_editor, egui, widgets};

pub use ui::{
    AbBankState, FactoryPreset, GuiTelemetry, OutputLoudnessMeter, ParamRef, UserPreset,
    export_user_preset_json, import_user_preset_json,
};
use ui::{
    CanvasDragHandle, draw_ab_and_presets_bar, draw_ab_and_presets_bar_persisted,
    draw_forge_studio_header_and_spectrum, draw_forge_studio_header_and_spectrum_interactive,
};

const OPENSOOTHE_PRESETS: &[FactoryPreset] = &[
    FactoryPreset {
        name: "Vocal De-Harsh",
        values: &[
            ("sens", 0.25),
            ("depth", 0.35),
            ("sharp", 0.25),
            ("delta", 0.0),
            ("submono", 0.0),
            ("tape", 0.0),
            ("limit", 0.0),
            ("ceil", 0.85),
        ],
    },
    FactoryPreset {
        name: "AI Fizz Rescue",
        values: &[
            ("sens", 0.15),
            ("depth", 0.60),
            ("sharp", 0.35),
            ("delta", 0.0),
            ("submono", 1.0),
            ("tape", 0.15),
            ("limit", 1.0),
            ("ceil", 0.85),
        ],
    },
    FactoryPreset {
        name: "Gentle Master",
        values: &[
            ("sens", 0.40),
            ("depth", 0.20),
            ("sharp", 0.50),
            ("delta", 0.0),
            ("submono", 0.0),
            ("tape", 0.10),
            ("limit", 1.0),
            ("ceil", 0.85),
        ],
    },
];

const OPENSPACER_PRESETS: &[FactoryPreset] = &[
    FactoryPreset {
        name: "Vocal Pocket",
        values: &[
            ("amt", 0.45),
            ("maxcut", 0.35),
            ("lowhz", 0.18),
            ("highhz", 0.45),
            ("kickmode", 0.0),
        ],
    },
    FactoryPreset {
        name: "Kick -> Sub",
        values: &[
            ("amt", 0.65),
            ("maxcut", 0.50),
            ("lowhz", 0.02),
            ("highhz", 0.08),
            ("kickmode", 1.0),
        ],
    },
];

const VOCAL_SURGICAL_PRESETS: &[FactoryPreset] = &[
    FactoryPreset {
        name: "Clean Lead",
        values: &[
            ("declick", 0.40),
            ("deplosive", 0.45),
            ("deess", 0.55),
            ("gate", 0.35),
        ],
    },
    FactoryPreset {
        name: "Broadcast Dry",
        values: &[
            ("declick", 0.60),
            ("deplosive", 0.70),
            ("deess", 0.45),
            ("gate", 0.50),
        ],
    },
];

const SUB_TAPE_PRESETS: &[FactoryPreset] = &[
    FactoryPreset {
        name: "Tight Club Sub",
        values: &[
            ("cutoff", 0.38),
            ("sub", 0.35),
            ("width", 0.55),
            ("drive", 0.25),
        ],
    },
    FactoryPreset {
        name: "Warm 15 IPS",
        values: &[
            ("cutoff", 0.30),
            ("sub", 0.20),
            ("width", 0.50),
            ("drive", 0.55),
        ],
    },
];

const LIMITER_PRESETS: &[FactoryPreset] = &[
    FactoryPreset {
        name: "Spotify -1 dBTP",
        values: &[("drive", 0.45), ("ceil", 0.85), ("rel", 0.18)],
    },
    FactoryPreset {
        name: "Club Loud",
        values: &[("drive", 0.65), ("ceil", 0.90), ("rel", 0.08)],
    },
    FactoryPreset {
        name: "Transparent Master",
        values: &[("drive", 0.40), ("ceil", 0.85), ("rel", 0.35)],
    },
];

const DRUM_TRIGGER_PRESETS: &[FactoryPreset] = &[
    FactoryPreset {
        name: "Tight Kit",
        values: &[("thresh", 0.55), ("neosoul", 0.0)],
    },
    FactoryPreset {
        name: "Neo-Soul Chords",
        values: &[("thresh", 0.50), ("neosoul", 1.0)],
    },
];

const PHASE_EQ_PRESETS: &[FactoryPreset] = &[
    FactoryPreset {
        name: "AI De-Harsh Flat",
        values: &[
            ("inv", 0.0),
            ("eqamt", 0.50),
            ("genre", 0.0),
            ("autogain", 1.0),
        ],
    },
    FactoryPreset {
        name: "Modern Hip-Hop",
        values: &[
            ("inv", 0.0),
            ("eqamt", 0.70),
            ("genre", 0.25),
            ("autogain", 1.0),
        ],
    },
];

const SPLITEQ_PRESETS: &[FactoryPreset] = &[
    FactoryPreset {
        name: "Drum Punch",
        values: &[
            ("tr_low", 0.62),
            ("tr_mid", 0.55),
            ("tr_high", 0.60),
            ("tn_low", 0.48),
            ("tn_mid", 0.45),
            ("tn_high", 0.50),
        ],
    },
    FactoryPreset {
        name: "Vocal Air",
        values: &[
            ("tr_low", 0.50),
            ("tr_mid", 0.45),
            ("tr_high", 0.45),
            ("tn_low", 0.50),
            ("tn_mid", 0.54),
            ("tn_high", 0.62),
        ],
    },
];

const DENSITY_CLIPPER_PRESETS: &[FactoryPreset] = &[
    FactoryPreset {
        name: "Glue Bus",
        values: &[("density", 0.25), ("clip", 0.20), ("ceil", 0.85)],
    },
    FactoryPreset {
        name: "Loud Master Clip",
        values: &[("density", 0.40), ("clip", 0.45), ("ceil", 0.90)],
    },
];

const SPACEVERB_PRESETS: &[FactoryPreset] = &[
    FactoryPreset {
        name: "Vocal Hall",
        values: &[
            ("model", 0.0),
            ("pdrive", 0.25),
            ("decay", 0.60),
            ("damp", 0.45),
            ("shimmer", 0.12),
            ("duck", 0.60),
            ("mix", 0.22),
        ],
    },
    FactoryPreset {
        name: "Lush Shimmer",
        values: &[
            ("model", 0.67),
            ("pdrive", 0.35),
            ("decay", 0.82),
            ("damp", 0.30),
            ("shimmer", 0.45),
            ("duck", 0.40),
            ("mix", 0.38),
        ],
    },
];

const VOCAL_TUNE_PRESETS: &[FactoryPreset] = &[
    FactoryPreset {
        name: "Natural Pop",
        values: &[
            ("speed", 0.25),
            ("amt", 0.78),
            ("human", 0.30),
            ("mix", 1.0),
        ],
    },
    FactoryPreset {
        name: "Hard Trap Tune",
        values: &[("speed", 0.0), ("amt", 1.0), ("human", 0.0), ("mix", 1.0)],
    },
    FactoryPreset {
        name: "Subtle Polish",
        values: &[
            ("speed", 0.45),
            ("amt", 0.55),
            ("human", 0.45),
            ("mix", 0.85),
        ],
    },
];

// ============================================================================
// 1. FORGE OPENSOOTHE DE-FIZZ (CLAP + VST3 Flagship)
// ============================================================================

pub struct ForgeOpenSoothePlug {
    params: Arc<OpenSootheParams>,
    telemetry: Arc<GuiTelemetry>,
    dsp: RealtimeOpenSoothe,
    sub_tape_dsp: RealtimeSubAndTapeBus,
    limiter_dsp: RealtimeMasterLimiter,
    /// Host sample rate from the last `initialize`, reused by `reset` (which gets none).
    sample_rate: f64,
    /// Shared-bus slot this instance last published to, cleared when it changes or on drop.
    published_slot: Option<usize>,
    /// This instance's identity on the shared bus, so its clears never empty a slot another
    /// instance has published to since.
    bus_owner: SpectrumBusOwner,
}

#[derive(Params)]
pub struct OpenSootheParams {
    #[persist = "editor-state"]
    pub editor_state: Arc<EguiState>,
    #[persist = "ab-state"]
    pub ab_state: Arc<Mutex<String>>,
    #[id = "sens"]
    pub sensitivity_db: FloatParam,
    #[id = "depth"]
    pub max_depth_db: FloatParam,
    #[id = "sharp"]
    pub sharpness_bins: IntParam,
    #[id = "delta"]
    pub delta_listen: BoolParam,
    #[id = "submono"]
    pub enable_sub_tape: BoolParam,
    #[id = "tape"]
    pub tape_drive: FloatParam,
    #[id = "limit"]
    pub enable_limiter: BoolParam,
    #[id = "ceil"]
    pub ceiling_dbtp: FloatParam,
    #[id = "bus_slot"]
    pub shared_bus_slot: IntParam,
    #[id = "tilt"]
    pub tilt_db_per_oct: FloatParam,
    #[id = "slink"]
    pub stereo_link: FloatParam,
    #[id = "msbal"]
    pub mid_side_balance: FloatParam,
    #[id = "lowlat"]
    pub low_latency_mode: BoolParam,
    #[id = "atk"]
    pub attack_ms: FloatParam,
    #[id = "rel"]
    pub release_ms: FloatParam,
    #[id = "hard"]
    pub hard_mode: BoolParam,
    #[id = "foc_lo"]
    pub focus_low_db: FloatParam,
    #[id = "foc_mid"]
    pub focus_mid_db: FloatParam,
    #[id = "foc_hi"]
    pub focus_high_db: FloatParam,
}

impl Default for OpenSootheParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(720, 460),
            ab_state: Arc::new(Mutex::new(String::new())),
            sensitivity_db: FloatParam::new(
                "Sensitivity",
                3.5,
                FloatRange::Linear {
                    min: 0.5,
                    max: 12.0,
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(2))
            .with_smoother(SmoothingStyle::Linear(20.0))
            .with_unit(" dB"),
            max_depth_db: FloatParam::new(
                "Max Suppression",
                9.0,
                FloatRange::Linear {
                    min: 0.5,
                    max: 20.0,
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(2))
            .with_smoother(SmoothingStyle::Linear(20.0))
            .with_unit(" dB"),
            sharpness_bins: IntParam::new(
                "Sharpness Radius",
                8,
                IntRange::Linear { min: 4, max: 20 },
            )
            .with_unit(" bins"),
            delta_listen: BoolParam::new("Delta Listen (Removed Fizz)", false),
            enable_sub_tape: BoolParam::new("Enable SubMono + Tape Warmth", true),
            tape_drive: FloatParam::new(
                "Console8 / Tape Drive",
                0.20,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(3))
            .with_smoother(SmoothingStyle::Linear(20.0)),
            enable_limiter: BoolParam::new("Enable -1.0 dBTP Brickwall", false),
            ceiling_dbtp: FloatParam::new(
                "True-Peak Ceiling",
                -1.0,
                FloatRange::Linear {
                    min: -6.0,
                    max: -0.1,
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(2))
            .with_unit(" dBTP"),
            shared_bus_slot: IntParam::new(
                "Shared Spectrum Bus Slot",
                0,
                IntRange::Linear { min: 0, max: 8 },
            )
            .with_value_to_string(Arc::new(bus_slot_to_string))
            .with_string_to_value(Arc::new(bus_slot_from_string)),
            tilt_db_per_oct: FloatParam::new(
                "Spectral Tilt",
                0.0,
                FloatRange::Linear {
                    min: -6.0,
                    max: 6.0,
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(2))
            .with_unit(" dB/oct"),
            stereo_link: FloatParam::new(
                "Stereo Link",
                1.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),
            mid_side_balance: FloatParam::new(
                "Mid/Side Focus",
                0.0,
                FloatRange::Linear {
                    min: -1.0,
                    max: 1.0,
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(2)),
            low_latency_mode: BoolParam::new("Low-Latency Tracking", false),
            attack_ms: FloatParam::new(
                "Attack",
                10.0,
                FloatRange::Skewed {
                    min: 1.0,
                    max: 100.0,
                    factor: FloatRange::skew_factor(-1.5),
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(1))
            .with_unit(" ms"),
            release_ms: FloatParam::new(
                "Release",
                55.0,
                FloatRange::Skewed {
                    min: 10.0,
                    max: 500.0,
                    factor: FloatRange::skew_factor(-1.5),
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(1))
            .with_unit(" ms"),
            hard_mode: BoolParam::new("Hard Mode (Surgical)", false),
            focus_low_db: FloatParam::new(
                "Focus Low (<400 Hz)",
                0.0,
                FloatRange::Linear {
                    min: -12.0,
                    max: 12.0,
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(1))
            .with_unit(" dB"),
            focus_mid_db: FloatParam::new(
                "Focus Mid (0.4-4 kHz)",
                0.0,
                FloatRange::Linear {
                    min: -12.0,
                    max: 12.0,
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(1))
            .with_unit(" dB"),
            focus_high_db: FloatParam::new(
                "Focus High (>4 kHz)",
                0.0,
                FloatRange::Linear {
                    min: -12.0,
                    max: 12.0,
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(1))
            .with_unit(" dB"),
        }
    }
}

impl Default for ForgeOpenSoothePlug {
    fn default() -> Self {
        Self {
            params: Arc::new(OpenSootheParams::default()),
            telemetry: Arc::new(GuiTelemetry::default()),
            dsp: RealtimeOpenSoothe::new(48_000.0),
            sub_tape_dsp: RealtimeSubAndTapeBus::new(48_000.0, 160.0),
            limiter_dsp: RealtimeMasterLimiter::new(48_000.0),
            sample_rate: 48_000.0,
            published_slot: None,
            bus_owner: SpectrumBusOwner::unique(),
        }
    }
}

/// Slot letters for bus-slot parameter values `1..=8`.
const BUS_SLOT_LETTERS: &str = "ABCDEFGH";

/// Bus-slot parameter display: `0` is `Off`, `1..=8` are `Slot A..H`.
fn bus_slot_to_string(value: i32) -> String {
    usize::try_from(value)
        .ok()
        .and_then(|v| v.checked_sub(1))
        .and_then(|i| BUS_SLOT_LETTERS.get(i..=i))
        .map_or_else(|| String::from("Off"), |letter| format!("Slot {letter}"))
}

/// Parses bus-slot text back to its parameter value, ignoring case and surrounding spaces:
/// `Off`, `Slot A..H`, a bare letter `A..H`, or the plain value `0..8`.
fn bus_slot_from_string(text: &str) -> Option<i32> {
    let text = text.trim();
    if text.eq_ignore_ascii_case("off") {
        return Some(0);
    }
    if let Ok(value) = text.parse::<i32>() {
        return (0..=8).contains(&value).then_some(value);
    }
    let letter = match text.get(..4) {
        Some(prefix) if prefix.eq_ignore_ascii_case("slot") => text[4..].trim(),
        _ => text,
    };
    let mut chars = letter.chars();
    let (Some(c), None) = (chars.next(), chars.next()) else {
        return None;
    };
    let index = BUS_SLOT_LETTERS.find(c.to_ascii_uppercase())?;
    i32::try_from(index + 1).ok()
}

impl Drop for ForgeOpenSoothePlug {
    /// A removed instance must not leave its last frame on the bus. The clear only empties the
    /// slot if this instance published to it last, so it never silences another instance.
    fn drop(&mut self) {
        if let Some(slot) = self.published_slot.take() {
            clear_shared_spectrum(self.bus_owner, slot);
        }
    }
}

impl ForgeOpenSoothePlug {
    /// Publishes `bands` to the bus slot selected by `bus_param` (`0 = Off`). Moving to another
    /// slot or to `Off` clears the slot this instance published to before, unless another
    /// instance has published to it since (the last publisher owns a slot).
    fn publish_to_bus(&mut self, bus_param: usize, bands: &[f32; 32]) {
        let slot = shared_spectrum_slot_index(bus_param);
        if let Some(old) = self.published_slot.filter(|&old| Some(old) != slot) {
            clear_shared_spectrum(self.bus_owner, old);
        }
        self.published_slot = slot;
        if let Some(idx) = slot {
            publish_shared_spectrum(self.bus_owner, idx, bands);
        }
    }

    /// Total input-to-output delay. The limiter's lookahead delay stays in the signal path even
    /// when the limiter (or delta listen) is off, so this is constant and the host's delay
    /// compensation never goes stale when those switches change.
    fn latency_samples(&self) -> u32 {
        self.dsp.latency_samples() + self.limiter_dsp.latency_samples()
    }
}

impl Plugin for ForgeOpenSoothePlug {
    const NAME: &'static str = "Forge OpenSoothe De-Fizz";
    const VENDOR: &'static str = "The 40 Thieves (ForgeAudio)";
    const URL: &'static str = "https://github.com/The-40-Thieves/audio-forge";
    const EMAIL: &'static str = "audio@the40thieves.org";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),
        ..AudioIOLayout::const_default()
    }];

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;
    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        let params = self.params.clone();
        let telemetry = self.telemetry.clone();
        let initial_ab = AbBankState::load_from_persisted(&self.params.ab_state);
        create_egui_editor(
            self.params.editor_state.clone(),
            initial_ab,
            |_, _| {},
            move |egui_ctx, setter, ab_state| {
                egui::CentralPanel::default().show(egui_ctx, |ui| {
                    let slot = shared_spectrum_slot_index(params.shared_bus_slot.value() as usize);
                    draw_forge_studio_header_and_spectrum_interactive(
                        ui,
                        "FORGE OPENSOOTHE DE-FIZZ",
                        "2048-pt OLA STFT Dynamic Spectral Resonance & AI Metallic Whistle Suppressor",
                        &telemetry,
                        slot,
                        Some(setter),
                        Some(CanvasDragHandle {
                            label: "SENS",
                            param: &params.sensitivity_db,
                            color: egui::Color32::from_rgb(250, 204, 21),
                        }),
                    );
                    let param_refs = [
                        ("sens", ParamRef::Float(&params.sensitivity_db)),
                        ("depth", ParamRef::Float(&params.max_depth_db)),
                        ("sharp", ParamRef::Int(&params.sharpness_bins)),
                        ("delta", ParamRef::Bool(&params.delta_listen)),
                        ("submono", ParamRef::Bool(&params.enable_sub_tape)),
                        ("tape", ParamRef::Float(&params.tape_drive)),
                        ("limit", ParamRef::Bool(&params.enable_limiter)),
                        ("ceil", ParamRef::Float(&params.ceiling_dbtp)),
                        ("bus_slot", ParamRef::Int(&params.shared_bus_slot)),
                        ("tilt", ParamRef::Float(&params.tilt_db_per_oct)),
                        ("slink", ParamRef::Float(&params.stereo_link)),
                        ("ms", ParamRef::Float(&params.mid_side_balance)),
                        ("lowlat", ParamRef::Bool(&params.low_latency_mode)),
                        ("atk", ParamRef::Float(&params.attack_ms)),
                        ("rel", ParamRef::Float(&params.release_ms)),
                        ("hard", ParamRef::Bool(&params.hard_mode)),
                        ("foc_lo", ParamRef::Float(&params.focus_low_db)),
                        ("foc_mid", ParamRef::Float(&params.focus_mid_db)),
                        ("foc_hi", ParamRef::Float(&params.focus_high_db)),
                    ];
                    draw_ab_and_presets_bar_persisted(
                        ui,
                        setter,
                        &param_refs,
                        ab_state,
                        OPENSOOTHE_PRESETS,
                        Some(&params.ab_state),
                    );
                    egui::ScrollArea::vertical()
                        .auto_shrink([false; 2])
                        .show(ui, |ui| {
                            egui::Grid::new("opensoothe_grid")
                                .num_columns(2)
                                .spacing([16.0, 8.0])
                                .show(ui, |ui| {
                                    ui.label("Sensitivity (dB)");
                                    ui.add(widgets::ParamSlider::for_param(&params.sensitivity_db, setter));
                                    ui.end_row();

                                    ui.label("Max Suppression (dB)");
                                    ui.add(widgets::ParamSlider::for_param(&params.max_depth_db, setter));
                                    ui.end_row();

                                    ui.label("Sharpness Radius");
                                    ui.add(widgets::ParamSlider::for_param(&params.sharpness_bins, setter));
                                    ui.end_row();

                                    ui.label("Delta Listen (Fizz Only)");
                                    ui.add(widgets::ParamSlider::for_param(&params.delta_listen, setter));
                                    ui.end_row();

                                    ui.label("SubMono (<160 Hz) + Tape Bus");
                                    ui.add(widgets::ParamSlider::for_param(&params.enable_sub_tape, setter));
                                    ui.end_row();

                                    ui.label("Console8 / ChowTape Drive");
                                    ui.add(widgets::ParamSlider::for_param(&params.tape_drive, setter));
                                    ui.end_row();

                                    ui.label("True-Peak -1.0 dBTP Brickwall");
                                    ui.add(widgets::ParamSlider::for_param(&params.enable_limiter, setter));
                                    ui.end_row();

                                    ui.label("Limiter Ceiling (dBTP)");
                                    ui.add(widgets::ParamSlider::for_param(&params.ceiling_dbtp, setter));
                                    ui.end_row();

                                    ui.label("Cross-Instance Spectrum Slot");
                                    ui.add(widgets::ParamSlider::for_param(&params.shared_bus_slot, setter));
                                    ui.end_row();

                                    ui.label("Spectral Tilt (dB/oct)");
                                    ui.add(widgets::ParamSlider::for_param(&params.tilt_db_per_oct, setter));
                                    ui.end_row();

                                    ui.label("Stereo Link");
                                    ui.add(widgets::ParamSlider::for_param(&params.stereo_link, setter));
                                    ui.end_row();

                                    ui.label("Mid/Side Focus (-1 M .. +1 S)");
                                    ui.add(widgets::ParamSlider::for_param(&params.mid_side_balance, setter));
                                    ui.end_row();

                                    ui.label("Low-Latency Tracking (~5.3 ms)");
                                    ui.add(widgets::ParamSlider::for_param(&params.low_latency_mode, setter));
                                    ui.end_row();

                                    ui.label("Attack (ms)");
                                    ui.add(widgets::ParamSlider::for_param(&params.attack_ms, setter));
                                    ui.end_row();

                                    ui.label("Release (ms)");
                                    ui.add(widgets::ParamSlider::for_param(&params.release_ms, setter));
                                    ui.end_row();

                                    ui.label("Hard Mode (Surgical vs Soft)");
                                    ui.add(widgets::ParamSlider::for_param(&params.hard_mode, setter));
                                    ui.end_row();

                                    ui.label("Focus Low (<400 Hz)");
                                    ui.add(widgets::ParamSlider::for_param(&params.focus_low_db, setter));
                                    ui.end_row();

                                    ui.label("Focus Mid (0.4-4 kHz)");
                                    ui.add(widgets::ParamSlider::for_param(&params.focus_mid_db, setter));
                                    ui.end_row();

                                    ui.label("Focus High (>4 kHz)");
                                    ui.add(widgets::ParamSlider::for_param(&params.focus_high_db, setter));
                                    ui.end_row();
                                });
                        });
                });
            },
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        self.sample_rate = f64::from(buffer_config.sample_rate);
        self.dsp
            .set_low_latency(self.params.low_latency_mode.value());
        self.reset();
        context.set_latency_samples(self.latency_samples());
        true
    }

    /// Clears DSP state at the sample rate from `initialize`. May run on the audio thread, and
    /// none of these resets allocate.
    fn reset(&mut self) {
        self.dsp.reset(self.sample_rate);
        self.sub_tape_dsp.reset(self.sample_rate);
        self.limiter_dsp.reset(self.sample_rate);
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let low_lat = self.params.low_latency_mode.value();
        self.dsp.set_low_latency(low_lat);
        let sharp = self.params.sharpness_bins.value() as usize;
        let delta = self.params.delta_listen.value();
        let use_sub_tape = self.params.enable_sub_tape.value();
        let use_limiter = self.params.enable_limiter.value();
        let ceil = f64::from(self.params.ceiling_dbtp.value());
        let bus_slot = self.params.shared_bus_slot.value() as usize;
        self.dsp
            .set_tilt_db_per_oct(f64::from(self.params.tilt_db_per_oct.value()));
        self.dsp
            .set_stereo_link(f64::from(self.params.stereo_link.value()));
        self.dsp
            .set_mid_side_balance(f64::from(self.params.mid_side_balance.value()));
        self.dsp
            .set_attack_ms(f64::from(self.params.attack_ms.value()));
        self.dsp
            .set_release_ms(f64::from(self.params.release_ms.value()));
        self.dsp.set_hard_mode(self.params.hard_mode.value());
        self.dsp.set_focus_eq(
            f64::from(self.params.focus_low_db.value()),
            f64::from(self.params.focus_mid_db.value()),
            f64::from(self.params.focus_high_db.value()),
        );

        for mut channel_samples in buffer.iter_samples() {
            let sens = f64::from(self.params.sensitivity_db.smoothed.next());
            let depth = f64::from(self.params.max_depth_db.smoothed.next());
            let tape_drive = f64::from(self.params.tape_drive.smoothed.next());

            let in_l = *channel_samples.get_mut(0).unwrap_or(&mut 0.0);
            let in_r = *channel_samples.get_mut(1).unwrap_or(&mut 0.0);

            let (mut out_l, mut out_r) = self
                .dsp
                .process_stereo_sample(in_l, in_r, sens, depth, sharp, delta);

            if use_sub_tape && !delta {
                (out_l, out_r) = self
                    .sub_tape_dsp
                    .process_stereo_sample(out_l, out_r, 0.20, 1.06, tape_drive);
            }

            (out_l, out_r) = if use_limiter && !delta {
                self.limiter_dsp
                    .process_stereo_sample(out_l, out_r, 0.0, ceil, 85.0)
            } else {
                self.limiter_dsp.process_delay_only(out_l, out_r)
            };

            let gr_est = 20.0
                * (out_l.abs().max(out_r.abs()) / in_l.abs().max(in_r.abs()).max(1e-5))
                    .clamp(0.05, 1.0)
                    .log10();
            self.telemetry
                .update_peaks(in_l, in_r, out_l, out_r, gr_est);

            if let Some(s0) = channel_samples.get_mut(0) {
                *s0 = out_l;
            }
            if let Some(s1) = channel_samples.get_mut(1) {
                *s1 = out_r;
            }
        }

        // Publish true 2048-pt STFT 32-band spectral energy & dynamic GR curve to GUI & Cross-Instance Bus
        let bands = self.dsp.last_spectrum_32_db();
        let gr_curve = self.dsp.last_band_gr_32_db();
        self.telemetry.update_spectrum_and_gr(&bands, &gr_curve);
        self.publish_to_bus(bus_slot, &bands);

        ProcessStatus::Normal
    }
}

impl ClapPlugin for ForgeOpenSoothePlug {
    const CLAP_ID: &'static str = "org.the40thieves.forge-audio.opensoothe-defizz";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("Real-Time 2048-pt OLA STFT Dynamic Spectral Resonance & AI Metallic Fizz Suppressor");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Restoration,
        ClapFeature::Equalizer,
        ClapFeature::Mastering,
        ClapFeature::Stereo,
    ];
}

impl Vst3Plugin for ForgeOpenSoothePlug {
    const VST3_CLASS_ID: [u8; 16] = *b"ForgeOpenSoothe1";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Restoration,
        Vst3SubCategory::Mastering,
    ];
}

// ============================================================================
// 2. FORGE OPENSPACER SIDECHAIN UNMASKER (4-Channel Sidechain CLAP Plugin)
// ============================================================================

pub struct ForgeOpenSpacerPlug {
    params: Arc<OpenSpacerParams>,
    telemetry: Arc<GuiTelemetry>,
    analyzer: RealtimeSpectrumAnalyzer32,
    dsp: RealtimeOpenSpacer,
}

#[derive(Params)]
pub struct OpenSpacerParams {
    #[persist = "editor-state"]
    pub editor_state: Arc<EguiState>,
    #[id = "amt"]
    pub amount: FloatParam,
    #[id = "maxcut"]
    pub max_cut_db: FloatParam,
    #[id = "lowhz"]
    pub low_hz: FloatParam,
    #[id = "highhz"]
    pub high_hz: FloatParam,
    #[id = "kickmode"]
    pub kick_bass_mode: BoolParam,
    #[id = "delta"]
    pub delta_listen: BoolParam,
    #[id = "msbal"]
    pub mid_side_balance: FloatParam,
}

impl Default for OpenSpacerParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(680, 380),
            amount: FloatParam::new(
                "Unmask Amount",
                0.50,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(3))
            .with_smoother(SmoothingStyle::Linear(15.0)),
            max_cut_db: FloatParam::new(
                "Max Carve Depth",
                8.5,
                FloatRange::Linear {
                    min: 1.0,
                    max: 18.0,
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(2))
            .with_unit(" dB"),
            low_hz: FloatParam::new(
                "Low Crossover",
                200.0,
                FloatRange::Skewed {
                    min: 25.0,
                    max: 2000.0,
                    factor: FloatRange::skew_factor(-1.5),
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(2))
            .with_unit(" Hz"),
            high_hz: FloatParam::new(
                "High Crossover",
                7500.0,
                FloatRange::Skewed {
                    min: 120.0,
                    max: 16000.0,
                    factor: FloatRange::skew_factor(-1.5),
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(2))
            .with_unit(" Hz"),
            kick_bass_mode: BoolParam::new("Kick -> Sub-Bass Mode (30-130 Hz)", false),
            delta_listen: BoolParam::new("Delta Listen (Carved Pocket Only)", false),
            mid_side_balance: FloatParam::new(
                "Mid/Side Focus",
                0.0,
                FloatRange::Linear {
                    min: -1.0,
                    max: 1.0,
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(2)),
        }
    }
}

impl Default for ForgeOpenSpacerPlug {
    fn default() -> Self {
        Self {
            params: Arc::new(OpenSpacerParams::default()),
            telemetry: Arc::new(GuiTelemetry::default()),
            analyzer: RealtimeSpectrumAnalyzer32::new(48_000.0),
            dsp: RealtimeOpenSpacer::new(48_000.0),
        }
    }
}

impl Plugin for ForgeOpenSpacerPlug {
    const NAME: &'static str = "Forge OpenSpacer Unmask";
    const VENDOR: &'static str = "The 40 Thieves (ForgeAudio)";
    const URL: &'static str = "https://github.com/The-40-Thieves/audio-forge";
    const EMAIL: &'static str = "audio@the40thieves.org";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),
        aux_input_ports: &[new_nonzero_u32(2)],
        ..AudioIOLayout::const_default()
    }];

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;
    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        let params = self.params.clone();
        let telemetry = self.telemetry.clone();
        create_egui_editor(
            self.params.editor_state.clone(),
            AbBankState::default(),
            |_, _| {},
            move |egui_ctx, setter, ab_state| {
                egui::CentralPanel::default().show(egui_ctx, |ui| {
                    draw_forge_studio_header_and_spectrum(
                        ui,
                        "FORGE OPENSPACER UNMASK",
                        "4-Channel Sidechain Dynamic Spectral Unmasker (Vocal -> Music & Kick -> Sub-Bass)",
                        &telemetry,
                        None,
                    );
                    let param_refs = [
                        ("amt", ParamRef::Float(&params.amount)),
                        ("maxcut", ParamRef::Float(&params.max_cut_db)),
                        ("lowhz", ParamRef::Float(&params.low_hz)),
                        ("highhz", ParamRef::Float(&params.high_hz)),
                        ("kickmode", ParamRef::Bool(&params.kick_bass_mode)),
                    ];
                    draw_ab_and_presets_bar(ui, setter, &param_refs, ab_state, OPENSPACER_PRESETS);
                    egui::Grid::new("openspacer_grid")
                        .num_columns(2)
                        .spacing([16.0, 8.0])
                        .show(ui, |ui| {
                            ui.label("Unmask Amount");
                            ui.add(widgets::ParamSlider::for_param(&params.amount, setter));
                            ui.end_row();
                            ui.label("Max Carve Depth (dB)");
                            ui.add(widgets::ParamSlider::for_param(&params.max_cut_db, setter));
                            ui.end_row();
                            ui.label("Low Crossover (Hz)");
                            ui.add(widgets::ParamSlider::for_param(&params.low_hz, setter));
                            ui.end_row();
                            ui.label("High Crossover (Hz)");
                            ui.add(widgets::ParamSlider::for_param(&params.high_hz, setter));
                            ui.end_row();
                            ui.label("Kick -> Sub-Bass Mode");
                            ui.add(widgets::ParamSlider::for_param(&params.kick_bass_mode, setter));
                            ui.end_row();
                            ui.label("Delta Listen (Carved Only)");
                            ui.add(widgets::ParamSlider::for_param(&params.delta_listen, setter));
                            ui.end_row();
                            ui.label("Mid/Side Focus (-1 M .. +1 S)");
                            ui.add(widgets::ParamSlider::for_param(&params.mid_side_balance, setter));
                            ui.end_row();
                        });
                });
            },
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        let sr = f64::from(buffer_config.sample_rate);
        self.dsp.reset(sr);
        self.analyzer.reset(sr);
        context.set_latency_samples(self.dsp.latency_samples());
        true
    }

    fn reset(&mut self) {
        self.analyzer.clear_state();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let kick_mode = self.params.kick_bass_mode.value();
        let (low_hz, high_hz) = if kick_mode {
            (30.0, 135.0)
        } else {
            (
                f64::from(self.params.low_hz.value()),
                f64::from(self.params.high_hz.value()),
            )
        };
        let max_cut = f64::from(self.params.max_cut_db.value());
        self.dsp.set_delta_listen(self.params.delta_listen.value());
        self.dsp
            .set_mid_side_balance(f64::from(self.params.mid_side_balance.value()));
        let samples_len = buffer.samples();

        for i in 0..samples_len {
            let amt = f64::from(self.params.amount.smoothed.next());
            let main_slices = buffer.as_slice();
            let tgt_l = main_slices.first().map_or(0.0, |c| c[i]);
            let tgt_r = main_slices.get(1).map_or(0.0, |c| c[i]);

            let (sc_l, sc_r) = if let Some(sc_buf) = aux.inputs.first_mut() {
                let sc_slices = sc_buf.as_slice();
                (
                    sc_slices
                        .first()
                        .and_then(|c| c.get(i).copied())
                        .unwrap_or(0.0),
                    sc_slices
                        .get(1)
                        .and_then(|c| c.get(i).copied())
                        .unwrap_or(0.0),
                )
            } else {
                (0.0, 0.0)
            };

            let (out_l, out_r) = self
                .dsp
                .process_sample(tgt_l, tgt_r, sc_l, sc_r, low_hz, high_hz, amt, max_cut);

            self.analyzer.push_stereo_sample(out_l, out_r);
            let gr_est = 20.0
                * (out_l.abs().max(out_r.abs()) / tgt_l.abs().max(tgt_r.abs()).max(1e-5))
                    .clamp(0.05, 1.0)
                    .log10();
            self.telemetry
                .update_peaks(tgt_l, tgt_r, out_l, out_r, gr_est);

            let out_slices = buffer.as_slice();
            if let Some(c0) = out_slices.first_mut() {
                c0[i] = out_l;
            }
            if let Some(c1) = out_slices.get_mut(1) {
                c1[i] = out_r;
            }
        }
        let carve_db = self.dsp.last_carve_bands_db();
        self.telemetry
            .update_spectrum_and_gr(&self.analyzer.snapshot_db(), &carve_db);

        ProcessStatus::Normal
    }
}

impl ClapPlugin for ForgeOpenSpacerPlug {
    const CLAP_ID: &'static str = "org.the40thieves.forge-audio.openspacer-unmask";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("Sidechain Dynamic Spectral Unmasker (Vocal -> Music & Kick -> Bass)");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Equalizer,
        ClapFeature::Mixing,
        ClapFeature::Stereo,
    ];
}

impl Vst3Plugin for ForgeOpenSpacerPlug {
    const VST3_CLASS_ID: [u8; 16] = *b"ForgeOpenSpacer1";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Eq,
        Vst3SubCategory::Dynamics,
    ];
}

// ============================================================================
// 3. FORGE VOCAL SURGICAL STRIP (Zero-Latency Vocal Restoration)
// ============================================================================

pub struct ForgeVocalSurgicalPlug {
    params: Arc<VocalSurgicalParams>,
    telemetry: Arc<GuiTelemetry>,
    analyzer: RealtimeSpectrumAnalyzer32,
    dsp: RealtimeVocalSurgical,
}

#[derive(Params)]
pub struct VocalSurgicalParams {
    #[persist = "editor-state"]
    pub editor_state: Arc<EguiState>,
    #[id = "declick"]
    pub declick_sensitivity: FloatParam,
    #[id = "deplosive"]
    pub deplosive_strength: FloatParam,
    #[id = "deess"]
    pub deess_threshold_db: FloatParam,
    #[id = "gate"]
    pub breath_gate_db: FloatParam,
}

/// Ramp time for every Vocal Surgical control, in milliseconds. All four controls are linear
/// amounts or dB thresholds, so a linear ramp is the right curve for each of them.
const VOCAL_SURGICAL_SMOOTHING_MS: f32 = 20.0;

impl Default for VocalSurgicalParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(680, 380),
            declick_sensitivity: FloatParam::new(
                "Mouth De-Click",
                0.65,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(3))
            .with_smoother(SmoothingStyle::Linear(VOCAL_SURGICAL_SMOOTHING_MS)),
            deplosive_strength: FloatParam::new(
                "Plosive Duck (<115Hz)",
                0.75,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(3))
            .with_smoother(SmoothingStyle::Linear(VOCAL_SURGICAL_SMOOTHING_MS)),
            deess_threshold_db: FloatParam::new(
                "Split-Band De-Esser",
                -22.0,
                FloatRange::Linear {
                    min: -45.0,
                    max: 0.0,
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(2))
            .with_unit(" dB")
            .with_smoother(SmoothingStyle::Linear(VOCAL_SURGICAL_SMOOTHING_MS)),
            breath_gate_db: FloatParam::new(
                "Breath Gate Threshold",
                -48.0,
                FloatRange::Linear {
                    min: -70.0,
                    max: -20.0,
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(2))
            .with_unit(" dB")
            .with_smoother(SmoothingStyle::Linear(VOCAL_SURGICAL_SMOOTHING_MS)),
        }
    }
}

/// One sample's worth of Vocal Surgical control values, read from the parameter smoothers.
#[derive(Debug, Clone, Copy, PartialEq)]
struct VocalSurgicalControls {
    declick: f64,
    deplosive: f64,
    deess_db: f64,
    gate_db: f64,
}

/// Advances every Vocal Surgical smoother by one sample and returns the new values.
fn next_vocal_surgical_controls(params: &VocalSurgicalParams) -> VocalSurgicalControls {
    VocalSurgicalControls {
        declick: f64::from(params.declick_sensitivity.smoothed.next()),
        deplosive: f64::from(params.deplosive_strength.smoothed.next()),
        deess_db: f64::from(params.deess_threshold_db.smoothed.next()),
        gate_db: f64::from(params.breath_gate_db.smoothed.next()),
    }
}

impl Default for ForgeVocalSurgicalPlug {
    fn default() -> Self {
        Self {
            params: Arc::new(VocalSurgicalParams::default()),
            telemetry: Arc::new(GuiTelemetry::default()),
            analyzer: RealtimeSpectrumAnalyzer32::new(48_000.0),
            dsp: RealtimeVocalSurgical::new(48_000.0),
        }
    }
}

impl Plugin for ForgeVocalSurgicalPlug {
    const NAME: &'static str = "Forge Vocal Surgical Strip";
    const VENDOR: &'static str = "The 40 Thieves (ForgeAudio)";
    const URL: &'static str = "https://github.com/The-40-Thieves/audio-forge";
    const EMAIL: &'static str = "audio@the40thieves.org";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),
        ..AudioIOLayout::const_default()
    }];

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;
    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        let params = self.params.clone();
        let telemetry = self.telemetry.clone();
        create_egui_editor(
            self.params.editor_state.clone(),
            AbBankState::default(),
            |_, _| {},
            move |egui_ctx, setter, ab_state| {
                egui::CentralPanel::default().show(egui_ctx, |ui| {
                    draw_forge_studio_header_and_spectrum(
                        ui,
                        "FORGE VOCAL SURGICAL STRIP",
                        "Zero-Latency Mouth De-Click, Dynamic Plosive Suppressor, Split-Band De-Esser & Breath Gate",
                        &telemetry,
                        None,
                    );
                    let param_refs = [
                        ("declick", ParamRef::Float(&params.declick_sensitivity)),
                        ("deplosive", ParamRef::Float(&params.deplosive_strength)),
                        ("deess", ParamRef::Float(&params.deess_threshold_db)),
                        ("gate", ParamRef::Float(&params.breath_gate_db)),
                    ];
                    draw_ab_and_presets_bar(
                        ui,
                        setter,
                        &param_refs,
                        ab_state,
                        VOCAL_SURGICAL_PRESETS,
                    );
                    egui::Grid::new("vocal_surgical_grid")
                        .num_columns(2)
                        .spacing([16.0, 8.0])
                        .show(ui, |ui| {
                            ui.label("Mouth De-Click Sensitivity");
                            ui.add(widgets::ParamSlider::for_param(&params.declick_sensitivity, setter));
                            ui.end_row();
                            ui.label("Plosive Duck (<115 Hz)");
                            ui.add(widgets::ParamSlider::for_param(&params.deplosive_strength, setter));
                            ui.end_row();
                            ui.label("Split-Band De-Esser (dB)");
                            ui.add(widgets::ParamSlider::for_param(&params.deess_threshold_db, setter));
                            ui.end_row();
                            ui.label("Breath Gate Threshold (dB)");
                            ui.add(widgets::ParamSlider::for_param(&params.breath_gate_db, setter));
                            ui.end_row();
                        });
                });
            },
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        let sr = f64::from(buffer_config.sample_rate);
        self.dsp.reset(sr);
        self.analyzer.reset(sr);
        true
    }

    fn reset(&mut self) {
        self.analyzer.clear_state();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        for mut channel_samples in buffer.iter_samples() {
            let ctl = next_vocal_surgical_controls(&self.params);
            let in_l = *channel_samples.get_mut(0).unwrap_or(&mut 0.0);
            let in_r = *channel_samples.get_mut(1).unwrap_or(&mut 0.0);
            let (out_l, out_r) = self.dsp.process_stereo_sample(
                in_l,
                in_r,
                ctl.declick,
                ctl.deplosive,
                ctl.deess_db,
                ctl.gate_db,
            );
            self.analyzer.push_stereo_sample(out_l, out_r);
            let gr_est = 20.0
                * (out_l.abs().max(out_r.abs()) / in_l.abs().max(in_r.abs()).max(1e-5))
                    .clamp(0.05, 1.0)
                    .log10();
            self.telemetry
                .update_peaks(in_l, in_r, out_l, out_r, gr_est);
            if let Some(s0) = channel_samples.get_mut(0) {
                *s0 = out_l;
            }
            if let Some(s1) = channel_samples.get_mut(1) {
                *s1 = out_r;
            }
        }
        self.telemetry
            .update_spectrum_and_gr(&self.analyzer.snapshot_db(), &[0.0; 32]);
        ProcessStatus::Normal
    }
}

impl ClapPlugin for ForgeVocalSurgicalPlug {
    const CLAP_ID: &'static str = "org.the40thieves.forge-audio.vocal-surgical";
    const CLAP_DESCRIPTION: Option<&'static str> = Some(
        "Zero-Latency Vocal Mouth De-Click, Dynamic Plosive Suppressor, Split-Band De-Esser & Breath Gate",
    );
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Restoration,
        ClapFeature::Compressor,
        ClapFeature::Stereo,
    ];
}

impl Vst3Plugin for ForgeVocalSurgicalPlug {
    const VST3_CLASS_ID: [u8; 16] = *b"ForgeVocalSurg01";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Restoration,
        Vst3SubCategory::Dynamics,
    ];
}

// ============================================================================
// 4. FORGE SUBMONO & CONSOLE TAPE BUS
// ============================================================================

pub struct ForgeSubAndTapePlug {
    params: Arc<SubAndTapeParams>,
    telemetry: Arc<GuiTelemetry>,
    analyzer: RealtimeSpectrumAnalyzer32,
    dsp: RealtimeSubAndTapeBus,
}

#[derive(Params)]
pub struct SubAndTapeParams {
    #[persist = "editor-state"]
    pub editor_state: Arc<EguiState>,
    #[id = "cutoff"]
    pub mono_cutoff_hz: FloatParam,
    #[id = "sub"]
    pub sub_reinforce: FloatParam,
    #[id = "width"]
    pub stereo_width: FloatParam,
    #[id = "drive"]
    pub tape_drive: FloatParam,
}

impl Default for SubAndTapeParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(680, 380),
            mono_cutoff_hz: FloatParam::new(
                "Sub Mono Cutoff",
                160.0,
                FloatRange::Linear {
                    min: 50.0,
                    max: 300.0,
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(2))
            .with_unit(" Hz"),
            sub_reinforce: FloatParam::new(
                "Sub Harmonic Reinforce",
                0.25,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_smoother(SmoothingStyle::Linear(20.0))
            .with_value_to_string(formatters::v2s_f32_rounded(3)),
            stereo_width: FloatParam::new(
                "High Stereo Width",
                1.08,
                FloatRange::Linear { min: 0.0, max: 2.0 },
            )
            .with_smoother(SmoothingStyle::Linear(20.0))
            .with_value_to_string(formatters::v2s_f32_rounded(3)),
            tape_drive: FloatParam::new(
                "Console8 / ChowTape RK4 Drive",
                0.22,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_smoother(SmoothingStyle::Linear(20.0))
            .with_value_to_string(formatters::v2s_f32_rounded(3)),
        }
    }
}

impl Default for ForgeSubAndTapePlug {
    fn default() -> Self {
        Self {
            params: Arc::new(SubAndTapeParams::default()),
            telemetry: Arc::new(GuiTelemetry::default()),
            analyzer: RealtimeSpectrumAnalyzer32::new(48_000.0),
            dsp: RealtimeSubAndTapeBus::new(48_000.0, 160.0),
        }
    }
}

impl Plugin for ForgeSubAndTapePlug {
    const NAME: &'static str = "Forge SubMono & Console Tape";
    const VENDOR: &'static str = "The 40 Thieves (ForgeAudio)";
    const URL: &'static str = "https://github.com/The-40-Thieves/audio-forge";
    const EMAIL: &'static str = "audio@the40thieves.org";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),
        ..AudioIOLayout::const_default()
    }];

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;
    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        let params = self.params.clone();
        let telemetry = self.telemetry.clone();
        create_egui_editor(
            self.params.editor_state.clone(),
            AbBankState::default(),
            |_, _| {},
            move |egui_ctx, setter, ab_state| {
                egui::CentralPanel::default().show(egui_ctx, |ui| {
                    draw_forge_studio_header_and_spectrum(
                        ui,
                        "FORGE SUBMONO & CONSOLE TAPE (JILES-ATHERTON RK4)",
                        "Linkwitz-Riley <160Hz Mono-Maker, Allpass Disperser, Jiles-Atherton RK4 Hysteresis & Head-Bump",
                        &telemetry,
                        None,
                    );
                    let param_refs = [
                        ("cutoff", ParamRef::Float(&params.mono_cutoff_hz)),
                        ("sub", ParamRef::Float(&params.sub_reinforce)),
                        ("width", ParamRef::Float(&params.stereo_width)),
                        ("drive", ParamRef::Float(&params.tape_drive)),
                    ];
                    draw_ab_and_presets_bar(ui, setter, &param_refs, ab_state, SUB_TAPE_PRESETS);
                    egui::Grid::new("subtape_grid")
                        .num_columns(2)
                        .spacing([16.0, 8.0])
                        .show(ui, |ui| {
                            ui.label("Sub Mono Cutoff (Hz)");
                            ui.add(widgets::ParamSlider::for_param(&params.mono_cutoff_hz, setter));
                            ui.end_row();
                            ui.label("Sub Harmonic + Disperser Reinforce");
                            ui.add(widgets::ParamSlider::for_param(&params.sub_reinforce, setter));
                            ui.end_row();
                            ui.label("High Stereo Width");
                            ui.add(widgets::ParamSlider::for_param(&params.stereo_width, setter));
                            ui.end_row();
                            ui.label("Jiles-Atherton RK4 + Console8 Drive");
                            ui.add(widgets::ParamSlider::for_param(&params.tape_drive, setter));
                            ui.end_row();
                        });
                });
            },
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        let sr = f64::from(buffer_config.sample_rate);
        self.dsp = RealtimeSubAndTapeBus::new(sr, f64::from(self.params.mono_cutoff_hz.value()));
        self.analyzer.reset(sr);
        true
    }

    fn reset(&mut self) {
        self.analyzer.clear_state();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // Only sets the glide target; the DSP retunes its side high-passes in place.
        self.dsp
            .set_mono_cutoff(f64::from(self.params.mono_cutoff_hz.value()));

        for mut channel_samples in buffer.iter_samples() {
            let sub = f64::from(self.params.sub_reinforce.smoothed.next());
            let width = f64::from(self.params.stereo_width.smoothed.next());
            let drive = f64::from(self.params.tape_drive.smoothed.next());
            let in_l = *channel_samples.get_mut(0).unwrap_or(&mut 0.0);
            let in_r = *channel_samples.get_mut(1).unwrap_or(&mut 0.0);
            let (out_l, out_r) = self
                .dsp
                .process_stereo_sample(in_l, in_r, sub, width, drive);
            self.analyzer.push_stereo_sample(out_l, out_r);
            self.telemetry.update_peaks(in_l, in_r, out_l, out_r, 0.0);
            if let Some(s0) = channel_samples.get_mut(0) {
                *s0 = out_l;
            }
            if let Some(s1) = channel_samples.get_mut(1) {
                *s1 = out_r;
            }
        }
        self.telemetry
            .update_spectrum_and_gr(&self.analyzer.snapshot_db(), &[0.0; 32]);
        ProcessStatus::Normal
    }
}

impl ClapPlugin for ForgeSubAndTapePlug {
    const CLAP_ID: &'static str = "org.the40thieves.forge-audio.submono-tape";
    const CLAP_DESCRIPTION: Option<&'static str> = Some(
        "Linkwitz-Riley <160Hz Sub Mono-Maker, Harmonic Bass Reinforcer & Console8/ChowTape Saturation",
    );
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Mastering,
        ClapFeature::Distortion,
        ClapFeature::Stereo,
    ];
}

impl Vst3Plugin for ForgeSubAndTapePlug {
    const VST3_CLASS_ID: [u8; 16] = *b"ForgeSubAndTape1";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Mastering,
        Vst3SubCategory::Distortion,
    ];
}

// ============================================================================
// 5. FORGE MASTER TRUE-PEAK LIMITER
// ============================================================================

pub struct ForgeMasterLimiterPlug {
    params: Arc<MasterLimiterParams>,
    telemetry: Arc<GuiTelemetry>,
    analyzer: RealtimeSpectrumAnalyzer32,
    loudness_meter: OutputLoudnessMeter,
    dsp: RealtimeMasterLimiter,
    sample_rate: f64,
}

#[derive(Params)]
pub struct MasterLimiterParams {
    #[persist = "editor-state"]
    pub editor_state: Arc<EguiState>,
    #[persist = "ab-state"]
    pub ab_state: Arc<Mutex<String>>,
    #[id = "drive"]
    pub drive_db: FloatParam,
    #[id = "ceil"]
    pub ceiling_dbtp: FloatParam,
    #[id = "rel"]
    pub release_ms: FloatParam,
    #[id = "style"]
    pub limiter_style: IntParam,
    #[id = "clink"]
    pub channel_link: FloatParam,
    #[id = "dither"]
    pub dither_mode: IntParam,
    #[id = "isp8x"]
    pub isp_8x: BoolParam,
    #[id = "unity"]
    pub unity_audition: BoolParam,
}

impl Default for MasterLimiterParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(680, 420),
            ab_state: Arc::new(Mutex::new(String::new())),
            drive_db: FloatParam::new(
                "Input Gain / Drive",
                0.0,
                FloatRange::Linear {
                    min: -12.0,
                    max: 18.0,
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(2))
            .with_smoother(SmoothingStyle::Linear(15.0))
            .with_unit(" dB"),
            ceiling_dbtp: FloatParam::new(
                "True-Peak Ceiling",
                -1.0,
                FloatRange::Linear {
                    min: -6.0,
                    max: -0.1,
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(2))
            .with_unit(" dBTP"),
            release_ms: FloatParam::new(
                "Release",
                85.0,
                FloatRange::Linear {
                    min: 15.0,
                    max: 400.0,
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(2))
            .with_unit(" ms"),
            limiter_style: IntParam::new("Limiter Style", 0, IntRange::Linear { min: 0, max: 3 })
                .with_value_to_string(Arc::new(|v| match v {
                    0 => "Transparent".to_string(),
                    1 => "Punchy".to_string(),
                    2 => "Dynamic".to_string(),
                    _ => "Aggressive".to_string(),
                }))
                .with_string_to_value(Arc::new(|s| match s.trim().to_ascii_lowercase().as_str() {
                    "0" | "transparent" => Some(0),
                    "1" | "punchy" => Some(1),
                    "2" | "dynamic" => Some(2),
                    "3" | "aggressive" => Some(3),
                    _ => None,
                })),
            channel_link: FloatParam::new(
                "Channel Link",
                1.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),
            dither_mode: IntParam::new("Output Dither", 0, IntRange::Linear { min: 0, max: 2 })
                .with_value_to_string(Arc::new(|v| match v {
                    0 => "Off".to_string(),
                    1 => "TPDF 24-bit".to_string(),
                    _ => "Noise-Shaped 16-bit".to_string(),
                }))
                .with_string_to_value(Arc::new(|s| match s.trim().to_ascii_lowercase().as_str() {
                    "0" | "off" | "none" => Some(0),
                    "1" | "tpdf" | "tpdf 24-bit" => Some(1),
                    "2" | "noise-shaped" | "noise-shaped 16-bit" | "shaped" => Some(2),
                    _ => None,
                })),
            isp_8x: BoolParam::new("8x True-Peak Oversampling", false),
            unity_audition: BoolParam::new("1:1 Unity Gain Audition", false),
        }
    }
}

impl Default for ForgeMasterLimiterPlug {
    fn default() -> Self {
        Self {
            params: Arc::new(MasterLimiterParams::default()),
            telemetry: Arc::new(GuiTelemetry::default()),
            analyzer: RealtimeSpectrumAnalyzer32::new(48_000.0),
            loudness_meter: OutputLoudnessMeter::new(48_000.0),
            dsp: RealtimeMasterLimiter::new(48_000.0),
            sample_rate: 48_000.0,
        }
    }
}

impl Plugin for ForgeMasterLimiterPlug {
    const NAME: &'static str = "Forge Master True-Peak Limiter";
    const VENDOR: &'static str = "The 40 Thieves (ForgeAudio)";
    const URL: &'static str = "https://github.com/The-40-Thieves/audio-forge";
    const EMAIL: &'static str = "audio@the40thieves.org";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),
        ..AudioIOLayout::const_default()
    }];

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;
    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        let params = self.params.clone();
        let telemetry = self.telemetry.clone();
        let initial_ab = AbBankState::load_from_persisted(&self.params.ab_state);
        create_egui_editor(
            self.params.editor_state.clone(),
            initial_ab,
            |_, _| {},
            move |egui_ctx, setter, ab_state| {
                egui::CentralPanel::default().show(egui_ctx, |ui| {
                    draw_forge_studio_header_and_spectrum_interactive(
                        ui,
                        "FORGE MASTER TRUE-PEAK BRICKWALL LIMITER",
                        "Lookahead 4x/8x Inter-Sample True-Peak Brickwall Limiter (-1.0 dBTP) & ITU-R BS.1770-4 Meter",
                        &telemetry,
                        None,
                        Some(setter),
                        Some(CanvasDragHandle {
                            label: "CEIL",
                            param: &params.ceiling_dbtp,
                            color: egui::Color32::from_rgb(248, 113, 113),
                        }),
                    );
                    let param_refs = [
                        ("drive", ParamRef::Float(&params.drive_db)),
                        ("ceil", ParamRef::Float(&params.ceiling_dbtp)),
                        ("rel", ParamRef::Float(&params.release_ms)),
                        ("style", ParamRef::Int(&params.limiter_style)),
                        ("clink", ParamRef::Float(&params.channel_link)),
                        ("dither", ParamRef::Int(&params.dither_mode)),
                        ("isp8x", ParamRef::Bool(&params.isp_8x)),
                        ("unity", ParamRef::Bool(&params.unity_audition)),
                    ];
                    draw_ab_and_presets_bar_persisted(
                        ui,
                        setter,
                        &param_refs,
                        ab_state,
                        LIMITER_PRESETS,
                        Some(&params.ab_state),
                    );
                    egui::ScrollArea::vertical()
                        .auto_shrink([false; 2])
                        .show(ui, |ui| {
                            egui::Grid::new("limiter_grid")
                                .num_columns(2)
                                .spacing([16.0, 8.0])
                                .show(ui, |ui| {
                                    ui.label("Input Gain / Drive (dB)");
                                    ui.add(widgets::ParamSlider::for_param(&params.drive_db, setter));
                                    ui.end_row();
                                    ui.label("True-Peak Ceiling (dBTP)");
                                    ui.add(widgets::ParamSlider::for_param(&params.ceiling_dbtp, setter));
                                    ui.end_row();
                                    ui.label("Release Time (ms)");
                                    ui.add(widgets::ParamSlider::for_param(&params.release_ms, setter));
                                    ui.end_row();
                                    ui.label("Limiter Style");
                                    ui.add(widgets::ParamSlider::for_param(&params.limiter_style, setter));
                                    ui.end_row();
                                    ui.label("Channel Link");
                                    ui.add(widgets::ParamSlider::for_param(&params.channel_link, setter));
                                    ui.end_row();
                                    ui.label("Output Dither");
                                    ui.add(widgets::ParamSlider::for_param(&params.dither_mode, setter));
                                    ui.end_row();
                                    ui.label("8x ISP Oversampling");
                                    ui.add(widgets::ParamSlider::for_param(&params.isp_8x, setter));
                                    ui.end_row();
                                    ui.label("1:1 Unity Gain Audition");
                                    ui.add(widgets::ParamSlider::for_param(&params.unity_audition, setter));
                                    ui.end_row();
                                });
                        });
                });
            },
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        let sr = f64::from(buffer_config.sample_rate);
        self.sample_rate = sr;
        self.dsp.reset(sr);
        self.analyzer.reset(sr);
        self.loudness_meter.reset(sr);
        context.set_latency_samples(self.dsp.latency_samples());
        true
    }

    fn reset(&mut self) {
        self.dsp.reset(self.sample_rate);
        self.analyzer.reset(self.sample_rate);
        self.loudness_meter.clear();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let ceil = f64::from(self.params.ceiling_dbtp.value());
        let rel = f64::from(self.params.release_ms.value());
        let style = match self.params.limiter_style.value() {
            1 => LimiterStyle::Punchy,
            2 => LimiterStyle::Dynamic,
            3 => LimiterStyle::Aggressive,
            _ => LimiterStyle::Transparent,
        };
        let (dither_mode, dither_bits) = match self.params.dither_mode.value() {
            1 => (DitherMode::Tpdf, 24),
            2 => (DitherMode::NoiseShapedHighPass, 16),
            _ => (DitherMode::None, 24),
        };
        self.dsp.set_style(style);
        self.dsp
            .set_channel_link(f64::from(self.params.channel_link.value()));
        self.dsp.set_dither(dither_mode, dither_bits);
        self.dsp
            .set_isp_oversampling(if self.params.isp_8x.value() {
                IspOversampling::X8
            } else {
                IspOversampling::X4
            });
        let unity_audition = self.params.unity_audition.value();

        for mut channel_samples in buffer.iter_samples() {
            let drive = f64::from(self.params.drive_db.smoothed.next());
            let in_l = *channel_samples.get_mut(0).unwrap_or(&mut 0.0);
            let in_r = *channel_samples.get_mut(1).unwrap_or(&mut 0.0);
            let (mut out_l, mut out_r) =
                self.dsp.process_stereo_sample(in_l, in_r, drive, ceil, rel);
            if unity_audition && drive.abs() > 0.01 {
                let comp = 10.0_f32.powf((-drive as f32) / 20.0);
                out_l *= comp;
                out_r *= comp;
            }
            self.analyzer.push_stereo_sample(out_l, out_r);
            self.loudness_meter.push_output_sample(out_l, out_r);
            let (_lufs, _dbtp, gr) = self.dsp.telemetry();
            self.telemetry.update_peaks(in_l, in_r, out_l, out_r, -gr);
            if let Some(s0) = channel_samples.get_mut(0) {
                *s0 = out_l;
            }
            if let Some(s1) = channel_samples.get_mut(1) {
                *s1 = out_r;
            }
        }
        let (_lufs, _dbtp, gr) = self.dsp.telemetry();
        let (lufs_m, lufs_s, out_tp) = self.loudness_meter.snapshot();
        self.telemetry
            .update_loudness_and_tp(lufs_m, lufs_s, out_tp);
        self.telemetry
            .update_spectrum_and_gr(&self.analyzer.snapshot_db(), &[-gr; 32]);
        ProcessStatus::Normal
    }
}

impl ClapPlugin for ForgeMasterLimiterPlug {
    const CLAP_ID: &'static str = "org.the40thieves.forge-audio.master-limiter";
    const CLAP_DESCRIPTION: Option<&'static str> = Some(
        "Lookahead 4x Inter-Sample True-Peak Brickwall Limiter (-1.0 dBTP) & ITU-R BS.1770-4 Meter",
    );
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Limiter,
        ClapFeature::Mastering,
        ClapFeature::Stereo,
    ];
}

impl Vst3Plugin for ForgeMasterLimiterPlug {
    const VST3_CLASS_ID: [u8; 16] = *b"ForgeMasterLim01";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Mastering,
        Vst3SubCategory::Dynamics,
    ];
}

// ============================================================================
// 6. FORGE DRUM TRIGGER & NEO-SOUL CHORD MIDI PROCESSOR
// ============================================================================

pub struct ForgeDrumTriggerPlug {
    params: Arc<DrumTriggerParams>,
    telemetry: Arc<GuiTelemetry>,
    analyzer: RealtimeSpectrumAnalyzer32,
    dsp: RealtimeDrumTrigger,
    /// Generated notes still sounding, so each gets its NoteOff.
    notes: DrumTriggerNotes,
    /// Set by `reset`; the next block starts by releasing every generated note.
    release_pending: bool,
    was_playing: bool,
}

#[derive(Params)]
pub struct DrumTriggerParams {
    #[persist = "editor-state"]
    pub editor_state: Arc<EguiState>,
    #[id = "thresh"]
    pub threshold_db: FloatParam,
    #[id = "neosoul"]
    pub expand_midi_chords: BoolParam,
}

impl Default for DrumTriggerParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(640, 340),
            threshold_db: FloatParam::new(
                "Trigger Threshold",
                -24.0,
                FloatRange::Linear {
                    min: -48.0,
                    max: -6.0,
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(2))
            .with_unit(" dB"),
            expand_midi_chords: BoolParam::new("1-Finger MIDI -> Neo-Soul 9th Voicing", true),
        }
    }
}

impl Default for ForgeDrumTriggerPlug {
    fn default() -> Self {
        Self {
            params: Arc::new(DrumTriggerParams::default()),
            telemetry: Arc::new(GuiTelemetry::default()),
            analyzer: RealtimeSpectrumAnalyzer32::new(48_000.0),
            dsp: RealtimeDrumTrigger::new(48_000.0),
            notes: DrumTriggerNotes::new(48_000.0),
            release_pending: false,
            was_playing: false,
        }
    }
}

impl Plugin for ForgeDrumTriggerPlug {
    const NAME: &'static str = "Forge Drum Trigger & Theory MIDI";
    const VENDOR: &'static str = "The 40 Thieves (ForgeAudio)";
    const URL: &'static str = "https://github.com/The-40-Thieves/audio-forge";
    const EMAIL: &'static str = "audio@the40thieves.org";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),
        ..AudioIOLayout::const_default()
    }];

    const MIDI_INPUT: MidiConfig = MidiConfig::Basic;
    const MIDI_OUTPUT: MidiConfig = MidiConfig::Basic;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;
    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        let params = self.params.clone();
        let telemetry = self.telemetry.clone();
        create_egui_editor(
            self.params.editor_state.clone(),
            AbBankState::default(),
            |_, _| {},
            move |egui_ctx, setter, ab_state| {
                egui::CentralPanel::default().show(egui_ctx, |ui| {
                    draw_forge_studio_header_and_spectrum(
                        ui,
                        "FORGE DRUM TRIGGER & THEORY MIDI",
                        "Real-Time Audio-to-MIDI 3-Band Drum Transient Replacer (C1/D1/F#1) & Neo-Soul Chord Voicer",
                        &telemetry,
                        None,
                    );
                    let param_refs = [
                        ("thresh", ParamRef::Float(&params.threshold_db)),
                        ("neosoul", ParamRef::Bool(&params.expand_midi_chords)),
                    ];
                    draw_ab_and_presets_bar(
                        ui,
                        setter,
                        &param_refs,
                        ab_state,
                        DRUM_TRIGGER_PRESETS,
                    );
                    egui::Grid::new("drum_trigger_grid")
                        .num_columns(2)
                        .spacing([16.0, 8.0])
                        .show(ui, |ui| {
                            ui.label("Transient Trigger Threshold (dB)");
                            ui.add(widgets::ParamSlider::for_param(&params.threshold_db, setter));
                            ui.end_row();
                            ui.label("1-Finger MIDI -> Neo-Soul 9th Voicing");
                            ui.add(widgets::ParamSlider::for_param(&params.expand_midi_chords, setter));
                            ui.end_row();
                        });
                });
            },
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        let sr = f64::from(buffer_config.sample_rate);
        self.dsp.reset(sr);
        self.analyzer.reset(sr);
        self.notes.set_sample_rate(sr);
        true
    }

    /// Notes cannot be sent from here, so held notes are released at the start of the next block.
    fn reset(&mut self) {
        self.release_pending = true;
        self.analyzer.clear_state();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let thresh = f64::from(self.params.threshold_db.value());
        self.notes
            .set_chord_expansion(self.params.expand_midi_chords.value());
        self.release_on_reset_or_stop(context);

        // Output events must be in non-decreasing timing order, so incoming MIDI (1) is forwarded
        // in step with the audio-detected drum hits (2) rather than all up front.
        let mut pending = context.next_event();

        // 2. Detect sub-sample Kick (C1=36), Snare (D1=38), and HiHat (F#1=42) transients from audio
        for (sample_idx, mut channel_samples) in buffer.iter_samples().enumerate() {
            while let Some(event) = pending {
                if event.timing() as usize > sample_idx {
                    break;
                }
                self.forward_event(context, event);
                pending = context.next_event();
            }

            let in_l = *channel_samples.get_mut(0).unwrap_or(&mut 0.0);
            let in_r = *channel_samples.get_mut(1).unwrap_or(&mut 0.0);
            self.analyzer.push_stereo_sample(in_l, in_r);
            self.telemetry.update_peaks(in_l, in_r, in_l, in_r, 0.0);
            self.trigger_drums(context, sample_idx as u32, in_l, in_r, thresh);
        }
        // Events timed at or past the end of the block (should not happen, but never drop them).
        while let Some(event) = pending {
            self.forward_event(context, event);
            pending = context.next_event();
        }
        self.telemetry
            .update_spectrum_and_gr(&self.analyzer.snapshot_db(), &[0.0; 32]);

        ProcessStatus::Normal
    }
}

impl ForgeDrumTriggerPlug {
    /// Converts a tracked note to a host event. Forwarded roots keep their incoming voice ID and
    /// generated notes carry none (see [`NoteMsg`]), so a chord tone never shares the ID of the
    /// root that started it.
    fn note_event(msg: NoteMsg) -> PluginNoteEvent<Self> {
        match msg {
            NoteMsg::On {
                timing,
                channel,
                note,
                velocity,
                voice_id,
            } => NoteEvent::NoteOn {
                timing,
                voice_id,
                channel,
                note,
                velocity,
            },
            NoteMsg::Off {
                timing,
                channel,
                note,
                voice_id,
            } => NoteEvent::NoteOff {
                timing,
                voice_id,
                channel,
                note,
                velocity: 0.0,
            },
        }
    }

    /// Releases every tracked note, forwarded roots included, at the start of the block after a
    /// reset, or when the transport stops.
    fn release_on_reset_or_stop(&mut self, context: &mut impl ProcessContext<Self>) {
        let playing = context.transport().playing;
        let stopped = self.was_playing && !playing;
        self.was_playing = playing;
        if std::mem::take(&mut self.release_pending) || stopped {
            self.notes
                .release_all(0, &mut |msg| context.send_event(Self::note_event(msg)));
        }
    }

    /// Step 1: forwards an incoming event and, when enabled, expands 1-finger MIDI notes into
    /// rootless Neo-Soul 9th voicings (3, 7, 9, 13) at the same timing. NoteOns and NoteOffs go
    /// through the tracker, which shares each key's occupancy between forwarded roots and
    /// generated notes and releases only the chord of the voice a NoteOff names. Every other
    /// event is forwarded unchanged.
    fn forward_event(
        &mut self,
        context: &mut impl ProcessContext<Self>,
        event: PluginNoteEvent<Self>,
    ) {
        match event {
            NoteEvent::NoteOn {
                timing,
                voice_id,
                channel,
                note,
                velocity,
            } => self
                .notes
                .root_on(timing, channel, note, velocity, voice_id, &mut |msg| {
                    context.send_event(Self::note_event(msg));
                }),
            NoteEvent::NoteOff {
                timing,
                voice_id,
                channel,
                note,
                ..
            } => self
                .notes
                .root_off(timing, channel, note, voice_id, &mut |msg| {
                    context.send_event(Self::note_event(msg));
                }),
            other => context.send_event(other),
        }
    }

    /// Step 2: runs the transient detector for one frame and sends gated drum notes on GM
    /// channel 10, each released a fixed length later.
    fn trigger_drums(
        &mut self,
        context: &mut impl ProcessContext<Self>,
        timing: u32,
        in_l: f32,
        in_r: f32,
        thresh: f64,
    ) {
        let emit = &mut |msg| context.send_event(Self::note_event(msg));
        self.notes.tick(timing, emit);
        let triggers = self.dsp.process_stereo_sample(in_l, in_r, thresh);
        for (slot, trig) in triggers.into_iter().enumerate() {
            if let Some((midi_note, vel)) = trig {
                self.notes.drum_hit(timing, slot, midi_note, vel, emit);
            }
        }
    }
}

impl ClapPlugin for ForgeDrumTriggerPlug {
    const CLAP_ID: &'static str = "org.the40thieves.forge-audio.drum-trigger-theory";
    const CLAP_DESCRIPTION: Option<&'static str> = Some(
        "Real-Time Audio-to-MIDI 3-Band Drum Transient Replacer (C1/D1/F#1) & Neo-Soul Chord Voicer",
    );
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::NoteEffect,
        ClapFeature::Drum,
        ClapFeature::Utility,
        ClapFeature::Stereo,
    ];
}

impl Vst3Plugin for ForgeDrumTriggerPlug {
    const VST3_CLASS_ID: [u8; 16] = *b"ForgeDrumTrig001";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Drum,
        Vst3SubCategory::Tools,
    ];
}

// ============================================================================
// 7. FORGE PHASE-ALIGN & 18-BAND MASTERING / MATCH EQ
// ============================================================================

pub struct ForgePhaseAndMatchPlug {
    params: Arc<PhaseAndMatchParams>,
    telemetry: Arc<GuiTelemetry>,
    analyzer: RealtimeSpectrumAnalyzer32,
    auto_gain: AutoGainCompensator,
    phase_dsp: RealtimePhaseAligner,
    eq_dsp: RealtimeMasteringEq18,
}

#[derive(Params)]
pub struct PhaseAndMatchParams {
    #[persist = "editor-state"]
    pub editor_state: Arc<EguiState>,
    #[id = "delay"]
    pub delay_samples: IntParam,
    #[id = "inv"]
    pub invert_polarity: BoolParam,
    #[id = "eqamt"]
    pub ai_warmth_eq_amount: FloatParam,
    #[id = "genre"]
    pub genre_target_idx: IntParam,
    #[id = "autogain"]
    pub auto_gain_comp: BoolParam,
}

impl Default for PhaseAndMatchParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(680, 390),
            delay_samples: IntParam::new(
                "Phase Delay Offset",
                0,
                IntRange::Linear { min: 0, max: 2048 },
            )
            .with_unit(" spl"),
            invert_polarity: BoolParam::new("Invert Polarity (180 deg)", false),
            ai_warmth_eq_amount: FloatParam::new(
                "18-Band AI De-Harsh & Genre Target",
                0.50,
                FloatRange::Linear { min: 0.0, max: 1.5 },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(3)),
            genre_target_idx: IntParam::new(
                "Genre Target (0=Pop,1=Trap,2=RnB,3=EDM,4=Rock,5=Jazz)",
                0,
                IntRange::Linear { min: 0, max: 5 },
            ),
            auto_gain_comp: BoolParam::new("ZLEqualizer Auto-Gain Compensation", true),
        }
    }
}

impl Default for ForgePhaseAndMatchPlug {
    fn default() -> Self {
        Self {
            params: Arc::new(PhaseAndMatchParams::default()),
            telemetry: Arc::new(GuiTelemetry::default()),
            analyzer: RealtimeSpectrumAnalyzer32::new(48_000.0),
            auto_gain: AutoGainCompensator::default(),
            phase_dsp: RealtimePhaseAligner::new(48_000.0),
            eq_dsp: RealtimeMasteringEq18::new(48_000.0),
        }
    }
}

impl Plugin for ForgePhaseAndMatchPlug {
    const NAME: &'static str = "Forge PhaseAlign & 18-Band EQ";
    const VENDOR: &'static str = "The 40 Thieves (ForgeAudio)";
    const URL: &'static str = "https://github.com/The-40-Thieves/audio-forge";
    const EMAIL: &'static str = "audio@the40thieves.org";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),
        ..AudioIOLayout::const_default()
    }];

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;
    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        let params = self.params.clone();
        let telemetry = self.telemetry.clone();
        create_egui_editor(
            self.params.editor_state.clone(),
            AbBankState::default(),
            |_, _| {},
            move |egui_ctx, setter, ab_state| {
                egui::CentralPanel::default().show(egui_ctx, |ui| {
                    draw_forge_studio_header_and_spectrum(
                        ui,
                        "FORGE PHASE-ALIGN & 18-BAND MASTERING EQ",
                        "Sample-Accurate Phase Aligner, Nyquist De-Cramped 18-Band AI De-Harsh EQ, Bode Overlay & Auto-Gain",
                        &telemetry,
                        None,
                    );
                    let param_refs = [
                        ("delay", ParamRef::Int(&params.delay_samples)),
                        ("inv", ParamRef::Bool(&params.invert_polarity)),
                        ("eqamt", ParamRef::Float(&params.ai_warmth_eq_amount)),
                        ("genre", ParamRef::Int(&params.genre_target_idx)),
                        ("autogain", ParamRef::Bool(&params.auto_gain_comp)),
                    ];
                    draw_ab_and_presets_bar(ui, setter, &param_refs, ab_state, PHASE_EQ_PRESETS);
                    egui::Grid::new("phase_eq_grid")
                        .num_columns(2)
                        .spacing([16.0, 8.0])
                        .show(ui, |ui| {
                            ui.label("Phase Delay Offset (spl)");
                            ui.add(widgets::ParamSlider::for_param(&params.delay_samples, setter));
                            ui.end_row();
                            ui.label("Invert Polarity (180 deg)");
                            ui.add(widgets::ParamSlider::for_param(&params.invert_polarity, setter));
                            ui.end_row();
                            ui.label("18-Band AI De-Harsh & Genre Amount");
                            ui.add(widgets::ParamSlider::for_param(&params.ai_warmth_eq_amount, setter));
                            ui.end_row();
                            ui.label("Genre Target (0=Pop..5=Jazz)");
                            ui.add(widgets::ParamSlider::for_param(&params.genre_target_idx, setter));
                            ui.end_row();
                            ui.label("Auto-Gain Compensation (AGC)");
                            ui.add(widgets::ParamSlider::for_param(&params.auto_gain_comp, setter));
                            ui.end_row();
                        });
                });
            },
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        let sr = f64::from(buffer_config.sample_rate);
        self.phase_dsp = RealtimePhaseAligner::new(sr);
        self.eq_dsp = RealtimeMasteringEq18::new(sr);
        self.analyzer.reset(sr);
        self.auto_gain.set_sample_rate(sr);
        self.auto_gain.reset();
        true
    }

    fn reset(&mut self) {
        self.phase_dsp.reset();
        self.eq_dsp.reset();
        self.auto_gain.reset();
        self.analyzer.clear_state();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let delay = self.params.delay_samples.value().max(0) as usize;
        let inv = self.params.invert_polarity.value();
        let eq_amt = f64::from(self.params.ai_warmth_eq_amount.value());
        let use_agc = self.params.auto_gain_comp.value();
        let genre = match self.params.genre_target_idx.value() {
            1 => GenreTonalTarget::HipHopTrap,
            2 => GenreTonalTarget::NeoSoulRnB,
            3 => GenreTonalTarget::EdmClub,
            4 => GenreTonalTarget::RockMetal,
            5 => GenreTonalTarget::AcousticJazz,
            _ => GenreTonalTarget::ModernPop,
        };
        let target_curve = genre.target_curve_18_db();

        // Blend built-in Genre Tonal Balance Target with AI De-Harshing (3.5-5kHz fizz cut + 12.5-16kHz air)
        const AI_WARMTH_CURVE_18: [f64; 18] = [
            0.4, 0.8, 1.1, 0.6, 0.0, -0.5, -0.8, -0.4, 0.0, -0.3, -1.2, -2.2, -2.5, -1.8, -0.6,
            0.5, 1.1, 1.3,
        ];
        let mut combined = [0.0_f64; 18];
        for i in 0..18 {
            combined[i] = AI_WARMTH_CURVE_18[i] * 0.65 + target_curve[i] * 0.15;
        }
        self.phase_dsp.set_alignment(delay, inv);
        self.eq_dsp.set_gains(&combined, eq_amt);

        for mut channel_samples in buffer.iter_samples() {
            let in_l = *channel_samples.get_mut(0).unwrap_or(&mut 0.0);
            let in_r = *channel_samples.get_mut(1).unwrap_or(&mut 0.0);
            let (al_l, al_r) = self.phase_dsp.process_stereo_sample(in_l, in_r);
            let (eq_l, eq_r) = self.eq_dsp.process_stereo_sample(al_l, al_r);
            let (out_l, out_r) = self
                .auto_gain
                .process_stereo(in_l, in_r, eq_l, eq_r, use_agc);
            self.analyzer.push_stereo_sample(out_l, out_r);
            self.telemetry.update_peaks(in_l, in_r, out_l, out_r, 0.0);
            if let Some(s0) = channel_samples.get_mut(0) {
                *s0 = out_l;
            }
            if let Some(s1) = channel_samples.get_mut(1) {
                *s1 = out_r;
            }
        }

        // Interpolate 18-band EQ Bode response across 32 GUI curve points
        let mut bode_32 = [0.0_f32; 32];
        for (b, val) in bode_32.iter_mut().enumerate() {
            let idx = (b * 17) / 31;
            *val = (combined[idx] * eq_amt) as f32;
        }
        self.telemetry
            .update_spectrum_and_gr(&self.analyzer.snapshot_db(), &bode_32);

        ProcessStatus::Normal
    }
}

impl ClapPlugin for ForgePhaseAndMatchPlug {
    const CLAP_ID: &'static str = "org.the40thieves.forge-audio.phase-and-matcheq";
    const CLAP_DESCRIPTION: Option<&'static str> = Some(
        "Sample-Accurate Phase/Polarity Aligner, 18-Band AI De-Harsh EQ & Genre Tonal Balance Target",
    );
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Equalizer,
        ClapFeature::Mastering,
        ClapFeature::PhaseVocoder,
        ClapFeature::Stereo,
    ];
}

impl Vst3Plugin for ForgePhaseAndMatchPlug {
    const VST3_CLASS_ID: [u8; 16] = *b"ForgePhaseEq0001";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Eq,
        Vst3SubCategory::Mastering,
    ];
}

// ============================================================================
// 8. FORGE SPLITEQ TRANSIENT & TONAL SHAPER (Eventide SplitEQ / Spiff Rival)
// ============================================================================

pub struct ForgeSplitEqPlug {
    params: Arc<SplitEqParams>,
    telemetry: Arc<GuiTelemetry>,
    analyzer: RealtimeSpectrumAnalyzer32,
    dsp: RealtimeTransientTonalSplit,
}

#[derive(Params)]
pub struct SplitEqParams {
    #[persist = "editor-state"]
    pub editor_state: Arc<EguiState>,
    #[id = "tr_low"]
    pub trans_low_db: FloatParam,
    #[id = "tr_mid"]
    pub trans_mid_db: FloatParam,
    #[id = "tr_high"]
    pub trans_high_db: FloatParam,
    #[id = "tn_low"]
    pub tonal_low_db: FloatParam,
    #[id = "tn_mid"]
    pub tonal_mid_db: FloatParam,
    #[id = "tn_high"]
    pub tonal_high_db: FloatParam,
}

impl Default for SplitEqParams {
    fn default() -> Self {
        let mk_db = |name: &'static str, def: f32| {
            FloatParam::new(
                name,
                def,
                FloatRange::Linear {
                    min: -12.0,
                    max: 12.0,
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(2))
            .with_smoother(SmoothingStyle::Linear(15.0))
            .with_unit(" dB")
        };
        Self {
            editor_state: EguiState::from_size(680, 380),
            trans_low_db: mk_db("Low Transient (<180 Hz Punch)", 1.5),
            trans_mid_db: mk_db("Mid Transient (180-3.2kHz Attack)", 0.0),
            trans_high_db: mk_db("High Transient (>3.2kHz Click/Pick)", -2.0),
            tonal_low_db: mk_db("Low Tonal (<180 Hz Body)", 0.5),
            tonal_mid_db: mk_db("Mid Tonal (180-3.2kHz Sustain)", 0.0),
            tonal_high_db: mk_db("High Tonal (>3.2kHz Shimmer/Air)", 1.5),
        }
    }
}

impl Default for ForgeSplitEqPlug {
    fn default() -> Self {
        Self {
            params: Arc::new(SplitEqParams::default()),
            telemetry: Arc::new(GuiTelemetry::default()),
            analyzer: RealtimeSpectrumAnalyzer32::new(48_000.0),
            dsp: RealtimeTransientTonalSplit::new(48_000.0),
        }
    }
}

impl Plugin for ForgeSplitEqPlug {
    const NAME: &'static str = "Forge SplitEQ Transient & Tonal";
    const VENDOR: &'static str = "The 40 Thieves (ForgeAudio)";
    const URL: &'static str = "https://github.com/The-40-Thieves/audio-forge";
    const EMAIL: &'static str = "audio@the40thieves.org";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),
        ..AudioIOLayout::const_default()
    }];

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;
    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        let params = self.params.clone();
        let telemetry = self.telemetry.clone();
        create_egui_editor(
            self.params.editor_state.clone(),
            AbBankState::default(),
            |_, _| {},
            move |egui_ctx, setter, ab_state| {
                egui::CentralPanel::default().show(egui_ctx, |ui| {
                    draw_forge_studio_header_and_spectrum(
                        ui,
                        "FORGE SPLITEQ — TRANSIENT vs. TONAL",
                        "Structural 3-Band Transient Attack vs. Sustained Harmonic Body Shaper",
                        &telemetry,
                        None,
                    );
                    let param_refs = [
                        ("tr_low", ParamRef::Float(&params.trans_low_db)),
                        ("tr_mid", ParamRef::Float(&params.trans_mid_db)),
                        ("tr_high", ParamRef::Float(&params.trans_high_db)),
                        ("tn_low", ParamRef::Float(&params.tonal_low_db)),
                        ("tn_mid", ParamRef::Float(&params.tonal_mid_db)),
                        ("tn_high", ParamRef::Float(&params.tonal_high_db)),
                    ];
                    draw_ab_and_presets_bar(ui, setter, &param_refs, ab_state, SPLITEQ_PRESETS);
                    egui::Grid::new("spliteq_grid")
                        .num_columns(2)
                        .spacing([16.0, 8.0])
                        .show(ui, |ui| {
                            ui.label("Low Transient (<250 Hz)");
                            ui.add(widgets::ParamSlider::for_param(
                                &params.trans_low_db,
                                setter,
                            ));
                            ui.end_row();
                            ui.label("Mid Transient (250-3.5kHz)");
                            ui.add(widgets::ParamSlider::for_param(
                                &params.trans_mid_db,
                                setter,
                            ));
                            ui.end_row();
                            ui.label("High Transient (>3.5kHz)");
                            ui.add(widgets::ParamSlider::for_param(
                                &params.trans_high_db,
                                setter,
                            ));
                            ui.end_row();
                            ui.label("Low Tonal Body (<250 Hz)");
                            ui.add(widgets::ParamSlider::for_param(
                                &params.tonal_low_db,
                                setter,
                            ));
                            ui.end_row();
                            ui.label("Mid Tonal Sustain (250-3.5kHz)");
                            ui.add(widgets::ParamSlider::for_param(
                                &params.tonal_mid_db,
                                setter,
                            ));
                            ui.end_row();
                            ui.label("High Tonal Air (>3.5kHz)");
                            ui.add(widgets::ParamSlider::for_param(
                                &params.tonal_high_db,
                                setter,
                            ));
                            ui.end_row();
                        });
                });
            },
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        let sr = f64::from(buffer_config.sample_rate);
        self.dsp.reset(sr);
        self.analyzer.reset(sr);
        true
    }

    fn reset(&mut self) {
        self.analyzer.clear_state();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let mut last_tr = [0.0_f64; 3];
        let mut last_tn = [0.0_f64; 3];
        for mut channel_samples in buffer.iter_samples() {
            let tr = [
                f64::from(self.params.trans_low_db.smoothed.next()),
                f64::from(self.params.trans_mid_db.smoothed.next()),
                f64::from(self.params.trans_high_db.smoothed.next()),
            ];
            let tn = [
                f64::from(self.params.tonal_low_db.smoothed.next()),
                f64::from(self.params.tonal_mid_db.smoothed.next()),
                f64::from(self.params.tonal_high_db.smoothed.next()),
            ];
            last_tr = tr;
            last_tn = tn;
            let in_l = *channel_samples.get_mut(0).unwrap_or(&mut 0.0);
            let in_r = *channel_samples.get_mut(1).unwrap_or(&mut 0.0);
            let (out_l, out_r) = self.dsp.process_stereo_sample(in_l, in_r, tr, tn, 1.5, 0);
            self.analyzer.push_stereo_sample(out_l, out_r);
            self.telemetry.update_peaks(in_l, in_r, out_l, out_r, 0.0);
            if let Some(s0) = channel_samples.get_mut(0) {
                *s0 = out_l;
            }
            if let Some(s1) = channel_samples.get_mut(1) {
                *s1 = out_r;
            }
        }
        let mut split_curve = [0.0_f32; 32];
        for (b, val) in split_curve.iter_mut().enumerate() {
            let band_idx = if b < 10 {
                0
            } else if b < 22 {
                1
            } else {
                2
            };
            *val = (0.5 * (last_tr[band_idx] + last_tn[band_idx])) as f32;
        }
        self.telemetry
            .update_spectrum_and_gr(&self.analyzer.snapshot_db(), &split_curve);
        ProcessStatus::Normal
    }
}

impl ClapPlugin for ForgeSplitEqPlug {
    const CLAP_ID: &'static str = "org.the40thieves.forge-audio.spliteq-transient-tonal";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("Structural 3-Band Transient vs. Tonal Equalizer & Attack/Sustain Shaper");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Equalizer,
        ClapFeature::TransientShaper,
        ClapFeature::Stereo,
    ];
}

impl Vst3Plugin for ForgeSplitEqPlug {
    const VST3_CLASS_ID: [u8; 16] = *b"ForgeSplitEq0001";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Eq,
        Vst3SubCategory::Dynamics,
    ];
}

// ============================================================================
// 9. FORGE DENSITY & 3-BAND ADAA CLIPPER (oeksound Bloom / Neutron 5 Rival)
// ============================================================================

pub struct ForgeDensityClipperPlug {
    params: Arc<DensityClipperParams>,
    telemetry: Arc<GuiTelemetry>,
    analyzer: RealtimeSpectrumAnalyzer32,
    dsp: RealtimeDensityAndClipper,
}

#[derive(Params)]
pub struct DensityClipperParams {
    #[persist = "editor-state"]
    pub editor_state: Arc<EguiState>,
    #[id = "density"]
    pub upward_density: FloatParam,
    #[id = "clip"]
    pub clip_drive_db: FloatParam,
    #[id = "ceil"]
    pub ceiling_db: FloatParam,
}

impl Default for DensityClipperParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(640, 320),
            upward_density: FloatParam::new(
                "Upward Density (Whisper Lift)",
                0.35,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(3))
            .with_smoother(SmoothingStyle::Linear(15.0)),
            clip_drive_db: FloatParam::new(
                "3-Band 2x ADAA Clip Drive",
                2.5,
                FloatRange::Linear {
                    min: 0.0,
                    max: 12.0,
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(2))
            .with_smoother(SmoothingStyle::Linear(15.0))
            .with_unit(" dB"),
            ceiling_db: FloatParam::new(
                "Clipper Output Ceiling",
                -0.8,
                FloatRange::Linear {
                    min: -6.0,
                    max: -0.1,
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(2))
            .with_unit(" dB"),
        }
    }
}

impl Default for ForgeDensityClipperPlug {
    fn default() -> Self {
        Self {
            params: Arc::new(DensityClipperParams::default()),
            telemetry: Arc::new(GuiTelemetry::default()),
            analyzer: RealtimeSpectrumAnalyzer32::new(48_000.0),
            dsp: RealtimeDensityAndClipper::new(48_000.0),
        }
    }
}

impl Plugin for ForgeDensityClipperPlug {
    const NAME: &'static str = "Forge Density & 3-Band ADAA Clipper";
    const VENDOR: &'static str = "The 40 Thieves (ForgeAudio)";
    const URL: &'static str = "https://github.com/The-40-Thieves/audio-forge";
    const EMAIL: &'static str = "audio@the40thieves.org";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),
        ..AudioIOLayout::const_default()
    }];

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;
    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        let params = self.params.clone();
        let telemetry = self.telemetry.clone();
        create_egui_editor(
            self.params.editor_state.clone(),
            AbBankState::default(),
            |_, _| {},
            move |egui_ctx, setter, ab_state| {
                egui::CentralPanel::default().show(egui_ctx, |ui| {
                    draw_forge_studio_header_and_spectrum(
                        ui,
                        "FORGE DENSITY & 3-BAND ADAA CLIPPER",
                        "Upward Low-Level Density Lifter + 2x Oversampled Anti-Derivative Anti-Aliased Soft Clipper",
                        &telemetry,
                        None,
                    );
                    let param_refs = [
                        ("density", ParamRef::Float(&params.upward_density)),
                        ("clip", ParamRef::Float(&params.clip_drive_db)),
                        ("ceil", ParamRef::Float(&params.ceiling_db)),
                    ];
                    draw_ab_and_presets_bar(
                        ui,
                        setter,
                        &param_refs,
                        ab_state,
                        DENSITY_CLIPPER_PRESETS,
                    );
                    egui::Grid::new("density_grid")
                        .num_columns(2)
                        .spacing([16.0, 8.0])
                        .show(ui, |ui| {
                            ui.label("Upward Density Amount");
                            ui.add(widgets::ParamSlider::for_param(&params.upward_density, setter));
                            ui.end_row();
                            ui.label("3-Band ADAA Clip Drive");
                            ui.add(widgets::ParamSlider::for_param(&params.clip_drive_db, setter));
                            ui.end_row();
                            ui.label("Output Ceiling");
                            ui.add(widgets::ParamSlider::for_param(&params.ceiling_db, setter));
                            ui.end_row();
                        });
                });
            },
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        let sr = f64::from(buffer_config.sample_rate);
        self.dsp = RealtimeDensityAndClipper::new(sr);
        self.analyzer.reset(sr);
        true
    }

    fn reset(&mut self) {
        self.analyzer.clear_state();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let ceil = f64::from(self.params.ceiling_db.value());
        for mut channel_samples in buffer.iter_samples() {
            let density = f64::from(self.params.upward_density.smoothed.next());
            let drive = f64::from(self.params.clip_drive_db.smoothed.next());
            let in_l = *channel_samples.get_mut(0).unwrap_or(&mut 0.0);
            let in_r = *channel_samples.get_mut(1).unwrap_or(&mut 0.0);
            let (out_l, out_r) = self
                .dsp
                .process_stereo_sample(in_l, in_r, density, drive, ceil);
            self.analyzer.push_stereo_sample(out_l, out_r);
            self.telemetry.update_peaks(in_l, in_r, out_l, out_r, 0.0);
            if let Some(s0) = channel_samples.get_mut(0) {
                *s0 = out_l;
            }
            if let Some(s1) = channel_samples.get_mut(1) {
                *s1 = out_r;
            }
        }
        self.telemetry
            .update_spectrum_and_gr(&self.analyzer.snapshot_db(), &[0.0; 32]);
        ProcessStatus::Normal
    }
}

impl ClapPlugin for ForgeDensityClipperPlug {
    const CLAP_ID: &'static str = "org.the40thieves.forge-audio.density-adaa-clipper";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("Upward Low-Level Density Compressor & 3-Band 2x-Oversampled ADAA Soft Clipper");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Compressor,
        ClapFeature::Distortion,
        ClapFeature::Mastering,
        ClapFeature::Stereo,
    ];
}

impl Vst3Plugin for ForgeDensityClipperPlug {
    const VST3_CLASS_ID: [u8; 16] = *b"ForgeDensClip001";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Dynamics,
        Vst3SubCategory::Distortion,
        Vst3SubCategory::Mastering,
    ];
}

// ============================================================================
// 10. FORGE SPACEVERB FDN & NEURAL PREAMP (Valhalla + Neural DSP / NAM Rival)
// ============================================================================

pub struct ForgeSpaceVerbAndPreampPlug {
    params: Arc<SpaceVerbAndPreampParams>,
    telemetry: Arc<GuiTelemetry>,
    analyzer: RealtimeSpectrumAnalyzer32,
    preamp_dsp: RealtimeNeuralPreamp,
    reverb_dsp: RealtimeFdnReverb,
}

#[derive(Params)]
pub struct SpaceVerbAndPreampParams {
    #[persist = "editor-state"]
    pub editor_state: Arc<EguiState>,
    #[id = "model"]
    pub neural_model_idx: IntParam,
    #[id = "pdrive"]
    pub preamp_drive: FloatParam,
    #[id = "decay"]
    pub reverb_decay: FloatParam,
    #[id = "damp"]
    pub reverb_damping: FloatParam,
    #[id = "shimmer"]
    pub shimmer_amount: FloatParam,
    #[id = "duck"]
    pub vocal_ducking: FloatParam,
    #[id = "mix"]
    pub wet_mix: FloatParam,
    #[id = "room"]
    pub room_geometry: IntParam,
    #[id = "declo"]
    pub decay_low_mult: FloatParam,
    #[id = "dechi"]
    pub decay_high_mult: FloatParam,
    #[id = "moddep"]
    pub mod_depth: FloatParam,
    #[persist = "ab-state"]
    pub ab_state: Arc<Mutex<String>>,
}

/// Bottom of the stored `decay` knob range. The stored range is kept as it was so saved state
/// and automation stay compatible; [`reverb_decay_knob_to_rt60_secs`] gives it its meaning.
const REVERB_DECAY_KNOB_MIN: f32 = 0.05;
/// Top of the stored `decay` knob range.
const REVERB_DECAY_KNOB_MAX: f32 = 0.98;
/// RT60 at the bottom of the decay knob. Matches the floor `RealtimeFdnReverb` clamps to.
pub const REVERB_RT60_MIN_SECS: f64 = 0.2;
/// RT60 at the top of the decay knob. Matches the ceiling `RealtimeFdnReverb` clamps to.
pub const REVERB_RT60_MAX_SECS: f64 = 20.0;

/// Position of a stored decay knob value along its range, from 0 at the bottom to 1 at the top.
fn reverb_decay_knob_position(knob: f32) -> f64 {
    let span = f64::from(REVERB_DECAY_KNOB_MAX - REVERB_DECAY_KNOB_MIN);
    let pos = (f64::from(knob) - f64::from(REVERB_DECAY_KNOB_MIN)) / span;
    if pos.is_finite() {
        pos.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Maps the stored SpaceVerb decay knob (0.05 to 0.98) onto an RT60 in seconds.
///
/// The curve is logarithmic: equal knob travel multiplies the decay time by the same factor,
/// running from [`REVERB_RT60_MIN_SECS`] at the bottom to [`REVERB_RT60_MAX_SECS`] at the top.
pub fn reverb_decay_knob_to_rt60_secs(knob: f32) -> f64 {
    let ratio = REVERB_RT60_MAX_SECS / REVERB_RT60_MIN_SECS;
    REVERB_RT60_MIN_SECS * ratio.powf(reverb_decay_knob_position(knob))
}

/// Inverse of [`reverb_decay_knob_to_rt60_secs`]; RT60 values outside the range are clamped.
///
/// Zero, negative and NaN inputs map to the bottom of the knob and positive infinity to the
/// top, so the logarithm never sees a value it cannot take.
pub fn rt60_secs_to_reverb_decay_knob(rt60_secs: f64) -> f32 {
    let pos = if rt60_secs.is_nan() || rt60_secs <= REVERB_RT60_MIN_SECS {
        0.0
    } else if rt60_secs >= REVERB_RT60_MAX_SECS {
        1.0
    } else {
        let ratio = REVERB_RT60_MAX_SECS / REVERB_RT60_MIN_SECS;
        (rt60_secs / REVERB_RT60_MIN_SECS).ln() / ratio.ln()
    };
    let span = f64::from(REVERB_DECAY_KNOB_MAX - REVERB_DECAY_KNOB_MIN);
    (f64::from(REVERB_DECAY_KNOB_MIN) + pos * span) as f32
}

/// Shows the decay knob as the RT60 it produces, for example `3.90 s`.
fn reverb_decay_knob_to_string(knob: f32) -> String {
    format!("{:.2} s", reverb_decay_knob_to_rt60_secs(knob))
}

/// Parses a decay typed in seconds (`4`, `4 s`, `4.0s`, `4 sec`) back to the stored knob value.
fn string_to_reverb_decay_knob(text: &str) -> Option<f32> {
    let trimmed = text.trim();
    let number = trimmed
        .strip_suffix("sec")
        .or_else(|| trimmed.strip_suffix("s"))
        .unwrap_or(trimmed)
        .trim();
    let secs: f64 = number.parse().ok()?;
    secs.is_finite()
        .then(|| rt60_secs_to_reverb_decay_knob(secs))
}

impl Default for SpaceVerbAndPreampParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(680, 430),
            neural_model_idx: IntParam::new(
                "Neural Preamp (0=Neve1073,1=1176AllBtn,2=V76Tube,3=StuderA800)",
                0,
                IntRange::Linear { min: 0, max: 3 },
            ),
            preamp_drive: FloatParam::new(
                "Neural WaveNet Drive",
                0.30,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(3))
            .with_smoother(SmoothingStyle::Linear(15.0)),
            reverb_decay: FloatParam::new(
                "FDN Reverb Decay",
                0.65,
                FloatRange::Linear {
                    min: REVERB_DECAY_KNOB_MIN,
                    max: REVERB_DECAY_KNOB_MAX,
                },
            )
            .with_value_to_string(Arc::new(reverb_decay_knob_to_string))
            .with_string_to_value(Arc::new(string_to_reverb_decay_knob)),
            reverb_damping: FloatParam::new(
                "High-Frequency Damping",
                0.40,
                FloatRange::Linear {
                    min: 0.0,
                    max: 0.95,
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(3)),
            shimmer_amount: FloatParam::new(
                "+12 st Octave Shimmer",
                0.15,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(3)),
            vocal_ducking: FloatParam::new(
                "Vocal Sidechain Ducking",
                0.55,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(3)),
            wet_mix: FloatParam::new(
                "Reverb Wet Mix",
                0.22,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(3)),
            room_geometry: IntParam::new("Room Geometry", 1, IntRange::Linear { min: 0, max: 4 })
                .with_value_to_string(Arc::new(|v| match v {
                    0 => "Plate".to_string(),
                    2 => "Chamber".to_string(),
                    3 => "Room".to_string(),
                    4 => "Ambient".to_string(),
                    _ => "Hall".to_string(),
                }))
                .with_string_to_value(Arc::new(|s| match s.trim().to_ascii_lowercase().as_str() {
                    "0" | "plate" => Some(0),
                    "1" | "hall" => Some(1),
                    "2" | "chamber" => Some(2),
                    "3" | "room" => Some(3),
                    "4" | "ambient" => Some(4),
                    _ => None,
                })),
            decay_low_mult: FloatParam::new(
                "Low Decay EQ",
                1.0,
                FloatRange::Linear {
                    min: 0.25,
                    max: 2.0,
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(2))
            .with_unit("x"),
            decay_high_mult: FloatParam::new(
                "High Decay EQ",
                1.0,
                FloatRange::Linear {
                    min: 0.25,
                    max: 2.0,
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(2))
            .with_unit("x"),
            mod_depth: FloatParam::new(
                "Chorus Mod Depth",
                1.0,
                FloatRange::Linear { min: 0.0, max: 2.0 },
            )
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),
            ab_state: Arc::new(Mutex::new(String::new())),
        }
    }
}

impl Default for ForgeSpaceVerbAndPreampPlug {
    fn default() -> Self {
        Self {
            params: Arc::new(SpaceVerbAndPreampParams::default()),
            telemetry: Arc::new(GuiTelemetry::default()),
            analyzer: RealtimeSpectrumAnalyzer32::new(48_000.0),
            preamp_dsp: RealtimeNeuralPreamp::new(AnalogHardwareModel::Neve1073ClassA),
            reverb_dsp: RealtimeFdnReverb::new(48_000.0),
        }
    }
}

impl Plugin for ForgeSpaceVerbAndPreampPlug {
    const NAME: &'static str = "Forge SpaceVerb FDN & Neural Preamp";
    const VENDOR: &'static str = "The 40 Thieves (ForgeAudio)";
    const URL: &'static str = "https://github.com/The-40-Thieves/audio-forge";
    const EMAIL: &'static str = "audio@the40thieves.org";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),
        ..AudioIOLayout::const_default()
    }];

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;
    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        let params = self.params.clone();
        let telemetry = self.telemetry.clone();
        let initial_ab = AbBankState::load_from_persisted(&self.params.ab_state);
        create_egui_editor(
            self.params.editor_state.clone(),
            initial_ab,
            |_, _| {},
            move |egui_ctx, setter, ab_state| {
                egui::CentralPanel::default().show(egui_ctx, |ui| {
                    draw_forge_studio_header_and_spectrum(
                        ui,
                        "FORGE SPACEVERB FDN & NEURAL WAVENET PREAMP",
                        "Dilated Causal WaveNet Preamp (.nam Ready) + Moorer ER / Allpass Diffuser / 8x8 Hadamard FDN Reverb",
                        &telemetry,
                        None,
                    );
                    let param_refs = [
                        ("model", ParamRef::Int(&params.neural_model_idx)),
                        ("pdrive", ParamRef::Float(&params.preamp_drive)),
                        ("decay", ParamRef::Float(&params.reverb_decay)),
                        ("damp", ParamRef::Float(&params.reverb_damping)),
                        ("shimmer", ParamRef::Float(&params.shimmer_amount)),
                        ("duck", ParamRef::Float(&params.vocal_ducking)),
                        ("mix", ParamRef::Float(&params.wet_mix)),
                        ("room", ParamRef::Int(&params.room_geometry)),
                        ("declo", ParamRef::Float(&params.decay_low_mult)),
                        ("dechi", ParamRef::Float(&params.decay_high_mult)),
                        ("moddep", ParamRef::Float(&params.mod_depth)),
                    ];
                    draw_ab_and_presets_bar_persisted(
                        ui,
                        setter,
                        &param_refs,
                        ab_state,
                        SPACEVERB_PRESETS,
                        Some(&params.ab_state),
                    );
                    egui::ScrollArea::vertical()
                        .auto_shrink([false; 2])
                        .show(ui, |ui| {
                            egui::Grid::new("spaceverb_grid")
                                .num_columns(2)
                                .spacing([16.0, 8.0])
                                .show(ui, |ui| {
                                    ui.label("Neural Hardware Model (0..3)");
                                    ui.add(widgets::ParamSlider::for_param(&params.neural_model_idx, setter));
                                    ui.end_row();
                                    ui.label("Neural WaveNet Drive");
                                    ui.add(widgets::ParamSlider::for_param(&params.preamp_drive, setter));
                                    ui.end_row();
                                    ui.label("8x8 FDN Reverb Decay");
                                    ui.add(widgets::ParamSlider::for_param(&params.reverb_decay, setter));
                                    ui.end_row();
                                    ui.label("High Damping");
                                    ui.add(widgets::ParamSlider::for_param(&params.reverb_damping, setter));
                                    ui.end_row();
                                    ui.label("+12 st Octave Shimmer");
                                    ui.add(widgets::ParamSlider::for_param(&params.shimmer_amount, setter));
                                    ui.end_row();
                                    ui.label("Vocal Auto-Ducking");
                                    ui.add(widgets::ParamSlider::for_param(&params.vocal_ducking, setter));
                                    ui.end_row();
                                    ui.label("Reverb Wet Mix");
                                    ui.add(widgets::ParamSlider::for_param(&params.wet_mix, setter));
                                    ui.end_row();
                                    ui.label("Room Geometry");
                                    ui.add(widgets::ParamSlider::for_param(&params.room_geometry, setter));
                                    ui.end_row();
                                    ui.label("Low Decay Rate EQ (<350 Hz)");
                                    ui.add(widgets::ParamSlider::for_param(&params.decay_low_mult, setter));
                                    ui.end_row();
                                    ui.label("High Decay Rate EQ (>4 kHz)");
                                    ui.add(widgets::ParamSlider::for_param(&params.decay_high_mult, setter));
                                    ui.end_row();
                                    ui.label("Chorus Modulation Depth");
                                    ui.add(widgets::ParamSlider::for_param(&params.mod_depth, setter));
                                    ui.end_row();
                                });
                        });
                });
            },
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        let sr = f64::from(buffer_config.sample_rate);
        self.preamp_dsp = RealtimeNeuralPreamp::new(AnalogHardwareModel::Neve1073ClassA);
        self.preamp_dsp.set_sample_rate(sr);
        self.reverb_dsp.reset(sr);
        self.analyzer.reset(sr);
        true
    }

    fn reset(&mut self) {
        let sr = self.reverb_dsp.sample_rate();
        self.preamp_dsp.reset();
        self.reverb_dsp.reset(sr);
        self.analyzer.clear_state();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let model = match self.params.neural_model_idx.value() {
            1 => AnalogHardwareModel::Fet1176AllButtons,
            2 => AnalogHardwareModel::TubeV76Triode,
            3 => AnalogHardwareModel::StuderA800Tape,
            _ => AnalogHardwareModel::Neve1073ClassA,
        };
        self.preamp_dsp.set_model(model);
        let room_geometry = match self.params.room_geometry.value() {
            0 => ReverbRoomGeometry::Plate,
            2 => ReverbRoomGeometry::Chamber,
            3 => ReverbRoomGeometry::Room,
            4 => ReverbRoomGeometry::Ambient,
            _ => ReverbRoomGeometry::Hall,
        };
        self.reverb_dsp.set_room_geometry(room_geometry);
        self.reverb_dsp.set_decay_eq(
            f64::from(self.params.decay_low_mult.value()),
            f64::from(self.params.decay_high_mult.value()),
        );
        self.reverb_dsp
            .set_mod_depth(f64::from(self.params.mod_depth.value()));
        let decay = reverb_decay_knob_to_rt60_secs(self.params.reverb_decay.value());
        let damp = f64::from(self.params.reverb_damping.value());
        let shimmer = f64::from(self.params.shimmer_amount.value());
        let duck = f64::from(self.params.vocal_ducking.value());
        let mix = f64::from(self.params.wet_mix.value());

        for mut channel_samples in buffer.iter_samples() {
            let pdrive = f64::from(self.params.preamp_drive.smoothed.next());
            let in_l = *channel_samples.get_mut(0).unwrap_or(&mut 0.0);
            let in_r = *channel_samples.get_mut(1).unwrap_or(&mut 0.0);
            let (pr_l, pr_r) =
                self.preamp_dsp
                    .process_stereo_sample(in_l, in_r, pdrive * 12.0, 0.0);
            let (out_l, out_r) = self
                .reverb_dsp
                .process_stereo_sample(pr_l, pr_r, decay, damp, shimmer, duck, mix);
            self.analyzer.push_stereo_sample(out_l, out_r);
            self.telemetry.update_peaks(in_l, in_r, out_l, out_r, 0.0);
            if let Some(s0) = channel_samples.get_mut(0) {
                *s0 = out_l;
            }
            if let Some(s1) = channel_samples.get_mut(1) {
                *s1 = out_r;
            }
        }
        self.telemetry
            .update_spectrum_and_gr(&self.analyzer.snapshot_db(), &[0.0; 32]);
        ProcessStatus::Normal
    }
}

impl ClapPlugin for ForgeSpaceVerbAndPreampPlug {
    const CLAP_ID: &'static str = "org.the40thieves.forge-audio.spaceverb-neural-preamp";
    const CLAP_DESCRIPTION: Option<&'static str> = Some(
        "Dilated Causal WaveNet Preamp Modeler (1073/1176/V76/A800) & 8x8 Hadamard FDN Shimmer/Ducking Reverb",
    );
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Reverb,
        ClapFeature::Distortion,
        ClapFeature::Stereo,
    ];
}

impl Vst3Plugin for ForgeSpaceVerbAndPreampPlug {
    const VST3_CLASS_ID: [u8; 16] = *b"ForgeSpaceVerb01";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Reverb,
        Vst3SubCategory::Distortion,
    ];
}

// ============================================================================
// 11. FORGE VOCAL TUNE (REAL-TIME LOW-LATENCY YIN + PSOLA PITCH CORRECTOR)
// ============================================================================

/// Lock-free pitch tracking readout shared between `ForgeVocalTunePlug`'s audio thread and GUI.
#[derive(Default)]
struct VocalPitchTelemetry {
    detected_hz: AtomicU32,
    target_hz: AtomicU32,
    correction_cents: AtomicU32,
}

impl VocalPitchTelemetry {
    #[inline]
    fn update(&self, detected: f64, target: f64, cents: f64) {
        self.detected_hz
            .store((detected as f32).to_bits(), Ordering::Relaxed);
        self.target_hz
            .store((target as f32).to_bits(), Ordering::Relaxed);
        self.correction_cents
            .store((cents as f32).to_bits(), Ordering::Relaxed);
    }

    #[inline]
    fn snapshot(&self) -> (f32, f32, f32) {
        (
            f32::from_bits(self.detected_hz.load(Ordering::Relaxed)),
            f32::from_bits(self.target_hz.load(Ordering::Relaxed)),
            f32::from_bits(self.correction_cents.load(Ordering::Relaxed)),
        )
    }
}

pub struct ForgeVocalTunePlug {
    params: Arc<VocalTuneParams>,
    telemetry: Arc<GuiTelemetry>,
    pitch_telemetry: Arc<VocalPitchTelemetry>,
    analyzer: RealtimeSpectrumAnalyzer32,
    dsp: RealtimeVocalTune,
    sample_rate: f64,
}

#[derive(Params)]
pub struct VocalTuneParams {
    #[persist = "editor-state"]
    pub editor_state: Arc<EguiState>,
    #[id = "key"]
    pub key_root: IntParam,
    #[id = "scale"]
    pub scale_mode: IntParam,
    #[id = "speed"]
    pub retune_speed_ms: FloatParam,
    #[id = "amt"]
    pub retune_amount: FloatParam,
    #[id = "human"]
    pub humanize: FloatParam,
    #[id = "formant"]
    pub formant_shift_st: FloatParam,
    #[id = "mix"]
    pub wet_mix: FloatParam,
}

const KEY_ROOT_NAMES: [&str; 12] = [
    "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B",
];

impl Default for VocalTuneParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(680, 380),
            key_root: IntParam::new("Key Root", 0, IntRange::Linear { min: 0, max: 11 })
                .with_value_to_string(Arc::new(|v| {
                    KEY_ROOT_NAMES[(v.clamp(0, 11)) as usize].to_string()
                }))
                .with_string_to_value(Arc::new(|s| {
                    let t = s.trim();
                    if let Ok(n) = t.parse::<i32>() {
                        return (0..=11).contains(&n).then_some(n);
                    }
                    KEY_ROOT_NAMES
                        .iter()
                        .position(|k| k.eq_ignore_ascii_case(t))
                        .map(|i| i as i32)
                })),
            scale_mode: IntParam::new("Scale Mode", 0, IntRange::Linear { min: 0, max: 4 })
                .with_value_to_string(Arc::new(|v| match v {
                    1 => "Major".to_string(),
                    2 => "Minor".to_string(),
                    3 => "Pentatonic Minor".to_string(),
                    4 => "Pentatonic Major".to_string(),
                    _ => "Chromatic".to_string(),
                }))
                .with_string_to_value(Arc::new(|s| match s.trim().to_ascii_lowercase().as_str() {
                    "0" | "chromatic" => Some(0),
                    "1" | "major" => Some(1),
                    "2" | "minor" => Some(2),
                    "3" | "pentatonic minor" | "pentmin" => Some(3),
                    "4" | "pentatonic major" | "pentmaj" => Some(4),
                    _ => None,
                })),
            retune_speed_ms: FloatParam::new(
                "Retune Speed",
                25.0,
                FloatRange::Linear {
                    min: 0.0,
                    max: 100.0,
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(1))
            .with_unit(" ms"),
            retune_amount: FloatParam::new(
                "Correction Amount",
                0.85,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),
            humanize: FloatParam::new(
                "Humanize (Vibrato)",
                0.15,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),
            formant_shift_st: FloatParam::new(
                "Formant Shift",
                0.0,
                FloatRange::Linear {
                    min: -12.0,
                    max: 12.0,
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(1))
            .with_unit(" st"),
            wet_mix: FloatParam::new("Wet Mix", 1.0, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_value_to_string(formatters::v2s_f32_percentage(0))
                .with_string_to_value(formatters::s2v_f32_percentage()),
        }
    }
}

impl Default for ForgeVocalTunePlug {
    fn default() -> Self {
        Self {
            params: Arc::new(VocalTuneParams::default()),
            telemetry: Arc::new(GuiTelemetry::default()),
            pitch_telemetry: Arc::new(VocalPitchTelemetry::default()),
            analyzer: RealtimeSpectrumAnalyzer32::new(48_000.0),
            dsp: RealtimeVocalTune::new(48_000.0),
            sample_rate: 48_000.0,
        }
    }
}

impl Plugin for ForgeVocalTunePlug {
    const NAME: &'static str = "Forge Vocal Tune";
    const VENDOR: &'static str = "The 40 Thieves (ForgeAudio)";
    const URL: &'static str = "https://github.com/The-40-Thieves/audio-forge";
    const EMAIL: &'static str = "audio@the40thieves.org";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(1),
            ..AudioIOLayout::const_default()
        },
    ];

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;
    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        let params = self.params.clone();
        let telemetry = self.telemetry.clone();
        let pitch_tel = self.pitch_telemetry.clone();
        create_egui_editor(
            self.params.editor_state.clone(),
            AbBankState::default(),
            |_, _| {},
            move |egui_ctx, setter, ab_state| {
                egui::CentralPanel::default().show(egui_ctx, |ui| {
                    draw_forge_studio_header_and_spectrum(
                        ui,
                        "FORGE VOCAL TUNE",
                        "Low-Latency Streaming YIN F0 + Scale-Locked Dual-Grain PSOLA Pitch Corrector",
                        &telemetry,
                        None,
                    );
                    let (det_hz, tgt_hz, cents) = pitch_tel.snapshot();
                    let pitch_text = if det_hz > 20.0 {
                        format!(
                            "LIVE PITCH   Detected: {:6.1} Hz  ->  Target: {:6.1} Hz  ({:+5.1} ct)",
                            det_hz, tgt_hz, cents
                        )
                    } else {
                        "LIVE PITCH   Unvoiced / Silence".to_string()
                    };
                    ui.label(
                        egui::RichText::new(pitch_text)
                            .monospace()
                            .color(egui::Color32::from_rgb(56, 189, 248)),
                    );
                    ui.add_space(4.0);
                    let param_refs = [
                        ("key", ParamRef::Int(&params.key_root)),
                        ("scale", ParamRef::Int(&params.scale_mode)),
                        ("speed", ParamRef::Float(&params.retune_speed_ms)),
                        ("amt", ParamRef::Float(&params.retune_amount)),
                        ("human", ParamRef::Float(&params.humanize)),
                        ("formant", ParamRef::Float(&params.formant_shift_st)),
                        ("mix", ParamRef::Float(&params.wet_mix)),
                    ];
                    draw_ab_and_presets_bar(ui, setter, &param_refs, ab_state, VOCAL_TUNE_PRESETS);
                    egui::Grid::new("vocal_tune_grid")
                        .num_columns(2)
                        .spacing([16.0, 8.0])
                        .show(ui, |ui| {
                            ui.label("Key Root");
                            ui.add(widgets::ParamSlider::for_param(&params.key_root, setter));
                            ui.end_row();
                            ui.label("Scale Mode");
                            ui.add(widgets::ParamSlider::for_param(&params.scale_mode, setter));
                            ui.end_row();
                            ui.label("Retune Speed (ms)");
                            ui.add(widgets::ParamSlider::for_param(
                                &params.retune_speed_ms,
                                setter,
                            ));
                            ui.end_row();
                            ui.label("Correction Amount");
                            ui.add(widgets::ParamSlider::for_param(
                                &params.retune_amount,
                                setter,
                            ));
                            ui.end_row();
                            ui.label("Humanize (Vibrato)");
                            ui.add(widgets::ParamSlider::for_param(&params.humanize, setter));
                            ui.end_row();
                            ui.label("Formant Shift (st)");
                            ui.add(widgets::ParamSlider::for_param(
                                &params.formant_shift_st,
                                setter,
                            ));
                            ui.end_row();
                            ui.label("Wet Mix");
                            ui.add(widgets::ParamSlider::for_param(&params.wet_mix, setter));
                            ui.end_row();
                        });
                });
            },
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        self.sample_rate = f64::from(buffer_config.sample_rate);
        self.dsp.reset(self.sample_rate);
        self.analyzer.reset(self.sample_rate);
        context.set_latency_samples(self.dsp.latency_samples());
        true
    }

    fn reset(&mut self) {
        self.dsp.reset(self.sample_rate);
        self.analyzer.clear_state();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let key_root = self.params.key_root.value().clamp(0, 11) as u8;
        let scale_mode = match self.params.scale_mode.value() {
            1 => VocalTuneScaleMode::Major,
            2 => VocalTuneScaleMode::Minor,
            3 => VocalTuneScaleMode::PentatonicMinor,
            4 => VocalTuneScaleMode::PentatonicMajor,
            _ => VocalTuneScaleMode::Chromatic,
        };
        self.dsp.set_key_and_scale(key_root, scale_mode);
        self.dsp
            .set_retune_speed_ms(f64::from(self.params.retune_speed_ms.value()));
        self.dsp
            .set_humanize(f64::from(self.params.humanize.value()));
        self.dsp
            .set_formant_shift_st(f64::from(self.params.formant_shift_st.value()));

        for mut channel_samples in buffer.iter_samples() {
            self.dsp
                .set_retune_amount(f64::from(self.params.retune_amount.smoothed.next()));
            self.dsp
                .set_wet_mix(f64::from(self.params.wet_mix.smoothed.next()));
            let in_l = *channel_samples.get_mut(0).unwrap_or(&mut 0.0);
            let in_r = channel_samples.get_mut(1).map_or(in_l, |s| *s);
            let (out_l, out_r) = self.dsp.process_stereo_sample(in_l, in_r);
            self.analyzer.push_stereo_sample(out_l, out_r);
            let corr_vis = (self.dsp.correction_cents() / 100.0) as f32;
            self.telemetry
                .update_peaks(in_l, in_r, out_l, out_r, corr_vis);
            if let Some(s0) = channel_samples.get_mut(0) {
                *s0 = out_l;
            }
            if let Some(s1) = channel_samples.get_mut(1) {
                *s1 = out_r;
            }
        }
        self.pitch_telemetry.update(
            self.dsp.detected_hz(),
            self.dsp.target_hz(),
            self.dsp.correction_cents(),
        );
        let corr_vis = (self.dsp.correction_cents() / 25.0) as f32;
        self.telemetry
            .update_spectrum_and_gr(&self.analyzer.snapshot_db(), &[corr_vis; 32]);
        ProcessStatus::Normal
    }
}

impl ClapPlugin for ForgeVocalTunePlug {
    const CLAP_ID: &'static str = "org.the40thieves.forge-audio.vocal-tune";
    const CLAP_DESCRIPTION: Option<&'static str> = Some(
        "Real-Time Low-Latency Streaming YIN F0 + Scale-Locked Dual-Grain PSOLA Vocal Pitch Corrector",
    );
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::PitchShifter,
        ClapFeature::Stereo,
    ];
}

impl Vst3Plugin for ForgeVocalTunePlug {
    const VST3_CLASS_ID: [u8; 16] = *b"ForgeVocalTune01";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::PitchShift];
}

// Export all 11 plugins into the all-in-one CLAP/VST3 bundle when `all-in-one-export` is enabled.
// Individual standalone VST3/CLAP plugins live in `plugins/forge-*` and export one plugin per bundle.
#[cfg(feature = "all-in-one-export")]
nih_export_clap!(
    ForgeOpenSoothePlug,
    ForgeOpenSpacerPlug,
    ForgeVocalSurgicalPlug,
    ForgeSubAndTapePlug,
    ForgeMasterLimiterPlug,
    ForgeDrumTriggerPlug,
    ForgePhaseAndMatchPlug,
    ForgeSplitEqPlug,
    ForgeDensityClipperPlug,
    ForgeSpaceVerbAndPreampPlug,
    ForgeVocalTunePlug
);
#[cfg(feature = "all-in-one-export")]
nih_export_vst3!(
    ForgeOpenSoothePlug,
    ForgeOpenSpacerPlug,
    ForgeVocalSurgicalPlug,
    ForgeSubAndTapePlug,
    ForgeMasterLimiterPlug,
    ForgeDrumTriggerPlug,
    ForgePhaseAndMatchPlug,
    ForgeSplitEqPlug,
    ForgeDensityClipperPlug,
    ForgeSpaceVerbAndPreampPlug,
    ForgeVocalTunePlug
);

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_gui_telemetry_atomic_meters_self_heal_from_nan_inf(
            in_l in prop::num::f32::ANY,
            in_r in prop::num::f32::ANY,
            out_l in prop::num::f32::ANY,
            out_r in prop::num::f32::ANY,
            gr_db in prop::num::f32::ANY,
        ) {
            let tel = GuiTelemetry::default();
            // Poison atomic slots first
            tel.peak_in_db.store(f32::NAN.to_bits(), Ordering::Relaxed);
            tel.peak_out_db.store(f32::INFINITY.to_bits(), Ordering::Relaxed);
            tel.gr_db.store(f32::NEG_INFINITY.to_bits(), Ordering::Relaxed);

            tel.update_peaks(in_l, in_r, out_l, out_r, gr_db);

            let pk_in = f32::from_bits(tel.peak_in_db.load(Ordering::Relaxed));
            let pk_out = f32::from_bits(tel.peak_out_db.load(Ordering::Relaxed));
            let gr = f32::from_bits(tel.gr_db.load(Ordering::Relaxed));

            prop_assert!(pk_in.is_finite() && (-100.0..=40.0).contains(&pk_in));
            prop_assert!(pk_out.is_finite() && (-100.0..=40.0).contains(&pk_out));
            prop_assert!(gr.is_finite() && (-90.0..=24.0).contains(&gr));
        }
    }

    #[test]
    fn test_all_10_plugins_instantiate_and_expose_valid_metadata() {
        let p1 = ForgeOpenSoothePlug::default();
        let p2 = ForgeOpenSpacerPlug::default();
        let p3 = ForgeVocalSurgicalPlug::default();
        let p4 = ForgeSubAndTapePlug::default();
        let p5 = ForgeMasterLimiterPlug::default();
        let p6 = ForgeDrumTriggerPlug::default();
        let p7 = ForgePhaseAndMatchPlug::default();
        let p8 = ForgeSplitEqPlug::default();
        let p9 = ForgeDensityClipperPlug::default();
        let p10 = ForgeSpaceVerbAndPreampPlug::default();
        let p11 = ForgeVocalTunePlug::default();

        assert!(!p1.params().param_map().is_empty());
        assert!(!p2.params().param_map().is_empty());
        assert!(!p3.params().param_map().is_empty());
        assert!(!p4.params().param_map().is_empty());
        assert!(!p5.params().param_map().is_empty());
        assert!(!p6.params().param_map().is_empty());
        assert!(!p7.params().param_map().is_empty());
        assert!(!p8.params().param_map().is_empty());
        assert!(!p9.params().param_map().is_empty());
        assert!(!p10.params().param_map().is_empty());
        assert!(!p11.params().param_map().is_empty());
    }

    /// The shared spectrum bus is process-wide and tests run on parallel threads, so every test
    /// that publishes to or scans the bus holds this lock to keep the others out of its slots.
    static SHARED_BUS_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn lock_shared_bus() -> std::sync::MutexGuard<'static, ()> {
        SHARED_BUS_TESTS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[test]
    fn open_soothe_bus_slot_zero_is_off_and_a_removed_instance_clears_its_slot() {
        let _bus = lock_shared_bus();
        let silent = [forge_dsp::SPECTRUM_FLOOR_DB; 32];
        let bands = [-20.0_f32; 32];
        let mut plug = ForgeOpenSoothePlug::default();

        // Off publishes nowhere: every slot keeps its silent default.
        plug.publish_to_bus(0, &bands);
        for slot in 0..8 {
            assert_eq!(
                read_shared_spectrum(slot),
                silent,
                "Off wrote to slot {slot}"
            );
        }

        // Slot C is index 2; moving to slot D clears slot C.
        plug.publish_to_bus(3, &bands);
        assert_eq!(read_shared_spectrum(2), bands);
        plug.publish_to_bus(4, &bands);
        assert_eq!(read_shared_spectrum(2), silent);
        assert_eq!(read_shared_spectrum(3), bands);

        drop(plug);
        assert_eq!(
            read_shared_spectrum(3),
            silent,
            "a dropped instance left its frame"
        );
    }

    const RT60_TEST_SAMPLE_RATE: f64 = 24_000.0;

    /// RT60 measured from a Schroeder backward-integrated decay curve: the time the curve takes
    /// to fall from -5 dB to -25 dB (T20), scaled up to 60 dB.
    fn schroeder_t20_rt60_secs(energy: &[f64], sample_rate: f64) -> f64 {
        let total: f64 = energy.iter().sum();
        let mut remaining = total;
        let mut t5 = None;
        for (n, e) in energy.iter().enumerate() {
            let level_db = 10.0 * (remaining / total).log10();
            if t5.is_none() && level_db <= -5.0 {
                t5 = Some(n);
            }
            if level_db <= -25.0 {
                let start = t5.unwrap_or(n);
                return 3.0 * (n - start) as f64 / sample_rate;
            }
            remaining -= e;
        }
        panic!("the reverb tail never fell 25 dB");
    }

    /// Fires an impulse through the SpaceVerb FDN at the plug's default damping, 100 % wet, and
    /// returns the RT60 measured from its tail. The dry impulse itself is left out of the curve.
    fn measured_spaceverb_rt60_secs(decay_rt60_secs: f64) -> f64 {
        let defaults = SpaceVerbAndPreampParams::default();
        let damp = f64::from(defaults.reverb_damping.value());
        let mut reverb = RealtimeFdnReverb::new(RT60_TEST_SAMPLE_RATE);
        let len = (RT60_TEST_SAMPLE_RATE * 25.0) as usize;
        let mut energy = Vec::with_capacity(len);
        for n in 0..len {
            let x = if n == 0 { 1.0 } else { 0.0 };
            let (l, r) = reverb.process_stereo_sample(x, x, decay_rt60_secs, damp, 0.0, 0.0, 1.0);
            let e = f64::from(l).powi(2) + f64::from(r).powi(2);
            energy.push(if n == 0 { 0.0 } else { e });
        }
        schroeder_t20_rt60_secs(&energy, RT60_TEST_SAMPLE_RATE)
    }

    #[test]
    fn spaceverb_decay_knob_sweeps_the_measured_rt60_well_past_one_second() {
        let params = SpaceVerbAndPreampParams::default();
        let measured: Vec<f64> = [0.0_f32, 0.33, 0.66, 1.0]
            .iter()
            .map(|&normalized| {
                let knob = params.reverb_decay.preview_plain(normalized);
                measured_spaceverb_rt60_secs(reverb_decay_knob_to_rt60_secs(knob))
            })
            .collect();
        for pair in measured.windows(2) {
            assert!(
                pair[1] > pair[0] * 1.5,
                "measured RT60 did not rise with the knob: {measured:?}"
            );
        }
        assert!(
            measured[0] < 0.6,
            "bottom of the knob is too long: {measured:?}"
        );
        assert!(
            measured[3] > 5.0,
            "top of the knob never got well past 1 s: {measured:?}"
        );
    }

    #[test]
    fn spaceverb_decay_shows_seconds_and_keeps_its_parameter_ids() {
        let params = SpaceVerbAndPreampParams::default();
        let decay = &params.reverb_decay;
        assert_eq!(decay.normalized_value_to_string(0.0, true), "0.20 s");
        assert_eq!(decay.normalized_value_to_string(1.0, true), "20.00 s");
        let normalized = decay
            .string_to_normalized_value("4 s")
            .expect("a decay typed in seconds parses");
        let rt60 = reverb_decay_knob_to_rt60_secs(decay.preview_plain(normalized));
        assert!((rt60 - 4.0).abs() < 1e-3, "4 s round-tripped to {rt60} s");
        assert!(decay.string_to_normalized_value("long").is_none());

        let ids: Vec<String> = params.param_map().into_iter().map(|p| p.0).collect();
        for id in ["model", "pdrive", "decay", "damp", "shimmer", "duck", "mix"] {
            assert!(ids.iter().any(|i| i == id), "parameter id {id} is missing");
        }
    }

    #[test]
    fn reverb_decay_mapping_round_trips_and_clamps_bad_inputs() {
        let bottom = REVERB_DECAY_KNOB_MIN;
        let top = REVERB_DECAY_KNOB_MAX;
        for bad in [0.0, -1.0, f64::NEG_INFINITY, f64::NAN, 0.1] {
            let knob = rt60_secs_to_reverb_decay_knob(bad);
            assert!(
                knob.is_finite() && knob == bottom,
                "{bad} s mapped to {knob}"
            );
        }
        for big in [20.0, 1e9, f64::INFINITY] {
            assert_eq!(rt60_secs_to_reverb_decay_knob(big), top, "{big} s");
        }
        assert!((reverb_decay_knob_to_rt60_secs(bottom) - REVERB_RT60_MIN_SECS).abs() < 1e-6);
        assert!((reverb_decay_knob_to_rt60_secs(top) - REVERB_RT60_MAX_SECS).abs() < 1e-4);
        for secs in [0.25, 1.0, 2.0, 5.0, 12.5, 19.0] {
            let back = reverb_decay_knob_to_rt60_secs(rt60_secs_to_reverb_decay_knob(secs));
            assert!(
                (back / secs - 1.0).abs() < 1e-4,
                "{secs} s round-tripped to {back} s"
            );
        }
        assert_eq!(string_to_reverb_decay_knob("0 s"), Some(bottom));
        assert_eq!(string_to_reverb_decay_knob("inf"), None);
    }

    /// Reads one Vocal Surgical control out of a [`VocalSurgicalControls`] snapshot.
    type VocalControlField = fn(&VocalSurgicalControls) -> f64;

    /// Each Vocal Surgical control, its smoother, and a full-range jump to automate it through.
    fn vocal_surgical_jumps(
        params: &VocalSurgicalParams,
    ) -> [(&'static str, &FloatParam, VocalControlField, f32, f32); 4] {
        [
            (
                "declick",
                &params.declick_sensitivity,
                |c| c.declick,
                0.0,
                1.0,
            ),
            (
                "deplosive",
                &params.deplosive_strength,
                |c| c.deplosive,
                0.0,
                1.0,
            ),
            (
                "deess",
                &params.deess_threshold_db,
                |c| c.deess_db,
                -45.0,
                0.0,
            ),
            ("gate", &params.breath_gate_db, |c| c.gate_db, -70.0, -20.0),
        ]
    }

    #[test]
    fn vocal_surgical_controls_ramp_across_a_block_instead_of_stepping() {
        const BLOCK: usize = 64;
        let params = VocalSurgicalParams::default();
        let jumps = vocal_surgical_jumps(&params);
        for (_, param, _, from, to) in jumps {
            param.smoothed.reset(from);
            param.smoothed.set_target(48_000.0, to);
        }
        let block: Vec<VocalSurgicalControls> = (0..BLOCK)
            .map(|_| next_vocal_surgical_controls(&params))
            .collect();
        for (name, _, field, from, to) in jumps {
            let span = f64::from(to - from);
            let values: Vec<f64> = block.iter().map(field).collect();
            let largest_step = values
                .windows(2)
                .map(|w| (w[1] - w[0]).abs())
                .fold((values[0] - f64::from(from)).abs(), f64::max);
            assert!(
                largest_step < span * 0.01,
                "{name} stepped by {largest_step} of a {span} jump within one block"
            );
            let moved = values[BLOCK - 1] - f64::from(from);
            assert!(
                moved * span > 0.0 && moved.abs() < span.abs() * 0.5,
                "{name} moved {moved} of {span} by the end of the block"
            );
        }
    }

    #[test]
    fn bus_slot_names_match_the_slot_letters() {
        assert_eq!(bus_slot_to_string(0), "Off");
        assert_eq!(bus_slot_to_string(1), "Slot A");
        assert_eq!(bus_slot_to_string(8), "Slot H");
        assert_eq!(bus_slot_to_string(-1), "Off");
        assert_eq!(bus_slot_to_string(9), "Off");
    }

    /// Plugs A and B share Slot F (index 5) and B published last, so B owns it. A switching
    /// away or being dropped must leave B's frame; B dropping clears it.
    #[test]
    fn open_soothe_instances_sharing_a_slot_never_silence_each_other() {
        let _bus = lock_shared_bus();
        const PARAM: usize = 6;
        const SLOT: usize = 5;
        let silent = [forge_dsp::SPECTRUM_FLOOR_DB; 32];
        let (a_bands, b_bands, c_bands) = ([-30.0_f32; 32], [-10.0_f32; 32], [-40.0_f32; 32]);
        let mut a = ForgeOpenSoothePlug::default();
        let mut b = ForgeOpenSoothePlug::default();
        a.publish_to_bus(PARAM, &a_bands);
        b.publish_to_bus(PARAM, &b_bands);

        // A moves to Slot G (index 6): its clear of Slot F must not touch B's frame.
        a.publish_to_bus(PARAM + 1, &a_bands);
        assert_eq!(read_shared_spectrum(SLOT), b_bands, "A's switch silenced B");
        assert_eq!(read_shared_spectrum(SLOT + 1), a_bands);

        // C joins Slot F and is removed after B publishes again: B's frame stays.
        let mut c = ForgeOpenSoothePlug::default();
        c.publish_to_bus(PARAM, &c_bands);
        b.publish_to_bus(PARAM, &b_bands);
        drop(c);
        assert_eq!(read_shared_spectrum(SLOT), b_bands, "C's drop silenced B");

        drop(a);
        assert_eq!(read_shared_spectrum(SLOT + 1), silent, "A owned Slot G");
        drop(b);
        assert_eq!(read_shared_spectrum(SLOT), silent, "B owned Slot F");
    }

    /// clap-validator's `param-conversions` check: every value's text must parse back to the
    /// same value, through the plugin's own parameter, and re-format to the same text.
    #[test]
    fn bus_slot_text_round_trips_for_every_value() {
        let params = OpenSootheParams::default();
        let param = &params.shared_bus_slot;
        for value in 0..=8 {
            let text = bus_slot_to_string(value);
            assert_eq!(bus_slot_from_string(&text), Some(value), "{text:?}");
            let normalized = param.preview_normalized(value);
            let shown = param.normalized_value_to_string(normalized, false);
            assert_eq!(shown, text);
            let parsed = param.string_to_normalized_value(&shown);
            assert_eq!(parsed, Some(normalized), "{shown:?} did not parse back");
        }
    }

    #[test]
    fn bus_slot_text_accepts_letters_numbers_and_any_case() {
        for (text, value) in [
            ("off", 0),
            (" OFF ", 0),
            ("slot a", 1),
            ("SLOT H", 8),
            ("Slot  c", 3),
            ("d", 4),
            ("G", 7),
            ("0", 0),
            ("5", 5),
            ("8", 8),
        ] {
            assert_eq!(bus_slot_from_string(text), Some(value), "{text:?}");
        }
        for text in ["", "Slot", "Slot I", "I", "9", "-1", "Slot AB", "on"] {
            assert_eq!(bus_slot_from_string(text), None, "{text:?}");
        }
    }
}
