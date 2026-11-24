//! Forge OpenSoothe De-Fizz & Forge OpenSpacer Unmask plugins.

use crate::chassis::*;
use crate::*;

define_presets! {
    const OPENSOOTHE_PRESETS {
        "Vocal De-Harsh" => { "sens": 0.25, "depth": 0.35, "sharp": 0.25, "delta": 0.0, "submono": 0.0, "tape": 0.0, "limit": 0.0, "ceil": 0.85 };
        "AI Fizz Rescue" => { "sens": 0.15, "depth": 0.60, "sharp": 0.35, "delta": 0.0, "submono": 1.0, "tape": 0.15, "limit": 1.0, "ceil": 0.85 };
        "Gentle Master" => { "sens": 0.40, "depth": 0.20, "sharp": 0.50, "delta": 0.0, "submono": 0.0, "tape": 0.10, "limit": 1.0, "ceil": 0.85 };
    }
    const OPENSPACER_PRESETS {
        "Vocal Pocket" => { "amt": 0.45, "maxcut": 0.35, "lowhz": 0.18, "highhz": 0.45, "kickmode": 0.0 };
        "Kick -> Sub" => { "amt": 0.65, "maxcut": 0.50, "lowhz": 0.02, "highhz": 0.08, "kickmode": 1.0 };
    }
}
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
    /// Last latency sample count reported to the host so `process` can update PDC on toggle.
    last_reported_latency: u32,
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
        let dsp = RealtimeOpenSoothe::new(48_000.0);
        let limiter_dsp = RealtimeMasterLimiter::new(48_000.0);
        let last_reported_latency = dsp.latency_samples() + limiter_dsp.latency_samples();
        Self {
            params: Arc::new(OpenSootheParams::default()),
            telemetry: Arc::new(GuiTelemetry::default()),
            dsp,
            sub_tape_dsp: RealtimeSubAndTapeBus::new(48_000.0, 160.0),
            limiter_dsp,
            sample_rate: 48_000.0,
            last_reported_latency,
            published_slot: None,
            bus_owner: SpectrumBusOwner::unique(),
        }
    }
}

/// Slot letters for bus-slot parameter values `1..=8`.
const BUS_SLOT_LETTERS: &str = "ABCDEFGH";

/// Bus-slot parameter display: `0` is `Off`, `1..=8` are `Slot A..H`.
pub(crate) fn bus_slot_to_string(value: i32) -> String {
    usize::try_from(value)
        .ok()
        .and_then(|v| v.checked_sub(1))
        .and_then(|i| BUS_SLOT_LETTERS.get(i..=i))
        .map_or_else(|| String::from("Off"), |letter| format!("Slot {letter}"))
}

/// Parses bus-slot text back to its parameter value, ignoring case and surrounding spaces:
/// `Off`, `Slot A..H`, a bare letter `A..H`, or the plain value `0..8`.
pub(crate) fn bus_slot_from_string(text: &str) -> Option<i32> {
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
    pub(crate) fn publish_to_bus(&mut self, bus_param: usize, bands: &[f32; 32]) {
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
    forge_plugin_common!("Forge OpenSoothe De-Fizz", crate::chassis::STEREO_LAYOUTS);

    forge_ab_editor!(
        persisted_scroll,
        title: "FORGE OPENSOOTHE DE-FIZZ",
        subtitle: "2048-pt OLA STFT Dynamic Spectral Resonance & AI Metallic Whistle Suppressor",
        slot: (p => shared_spectrum_slot_index(p.shared_bus_slot.value() as usize)),
        drag: ("SENS", sensitivity_db, egui::Color32::from_rgb(250, 204, 21)),
        presets: OPENSOOTHE_PRESETS,
        grid_id: "opensoothe_grid",
        params: [
            ("sens", Float, sensitivity_db, "Sensitivity (dB)"),
            ("depth", Float, max_depth_db, "Max Suppression (dB)"),
            ("sharp", Int, sharpness_bins, "Sharpness Radius"),
            ("delta", Bool, delta_listen, "Delta Listen (Fizz Only)"),
            ("submono", Bool, enable_sub_tape, "SubMono (<160 Hz) + Tape Bus"),
            ("tape", Float, tape_drive, "Console8 / ChowTape Drive"),
            ("limit", Bool, enable_limiter, "True-Peak -1.0 dBTP Brickwall"),
            ("ceil", Float, ceiling_dbtp, "Limiter Ceiling (dBTP)"),
            ("bus_slot", Int, shared_bus_slot, "Cross-Instance Spectrum Slot"),
            ("tilt", Float, tilt_db_per_oct, "Spectral Tilt (dB/oct)"),
            ("slink", Float, stereo_link, "Stereo Link"),
            ("msbal", Float, mid_side_balance, "Mid/Side Focus (-1 M .. +1 S)"),
            ("lowlat", Bool, low_latency_mode, "Low-Latency Tracking (~5.3 ms)"),
            ("atk", Float, attack_ms, "Attack (ms)"),
            ("rel", Float, release_ms, "Release (ms)"),
            ("hard", Bool, hard_mode, "Hard Mode (Surgical vs Soft)"),
            ("foc_lo", Float, focus_low_db, "Focus Low (<400 Hz)"),
            ("foc_mid", Float, focus_mid_db, "Focus Mid (0.4-4 kHz)"),
            ("foc_hi", Float, focus_high_db, "Focus High (>4 kHz)"),
        ]
    );

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
        let latency = self.latency_samples();
        self.last_reported_latency = latency;
        context.set_latency_samples(latency);
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
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let low_lat = self.params.low_latency_mode.value();
        self.dsp.set_low_latency(low_lat);
        let latency = self.latency_samples();
        if latency != self.last_reported_latency {
            self.last_reported_latency = latency;
            context.set_latency_samples(latency);
        }
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

            let (in_l, in_r) = read_stereo(&mut channel_samples);

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

            let gr_est = estimate_gr_db(in_l, in_r, out_l, out_r);
            self.telemetry
                .update_peaks(in_l, in_r, out_l, out_r, gr_est);
            write_stereo(&mut channel_samples, out_l, out_r);
        }

        // Publish true 2048-pt STFT 32-band spectral energy & dynamic GR curve to GUI & Cross-Instance Bus
        let bands = self.dsp.last_spectrum_32_db();
        let gr_curve = self.dsp.last_band_gr_32_db();
        self.telemetry.update_spectrum_and_gr(&bands, &gr_curve);
        self.publish_to_bus(bus_slot, &bands);

        ProcessStatus::Normal
    }
}

impl_forge_formats!(
    ForgeOpenSoothePlug,
    clap_id: "org.the40thieves.forge-audio.opensoothe-defizz",
    clap_desc: "Real-Time 2048-pt OLA STFT Dynamic Spectral Resonance & AI Metallic Fizz Suppressor",
    clap_features: &[
        ClapFeature::AudioEffect,
        ClapFeature::Restoration,
        ClapFeature::Equalizer,
        ClapFeature::Mastering,
        ClapFeature::Stereo,
    ],
    vst3_id: *b"ForgeOpenSoothe1",
    vst3_cats: &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Restoration,
        Vst3SubCategory::Mastering,
    ],
);

// ============================================================================
// 2. FORGE OPENSPACER SIDECHAIN UNMASKER (4-Channel Sidechain CLAP Plugin)
// ============================================================================

pub struct ForgeOpenSpacerPlug {
    params: Arc<OpenSpacerParams>,
    telemetry: Arc<GuiTelemetry>,
    analyzer: RealtimeSpectrumAnalyzer32,
    dsp: RealtimeOpenSpacer,
    sample_rate: f64,
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

impl_standard_plugin_default!(ForgeOpenSpacerPlug, OpenSpacerParams, RealtimeOpenSpacer);

impl Plugin for ForgeOpenSpacerPlug {
    forge_plugin_common!(
        "Forge OpenSpacer Unmask",
        crate::chassis::STEREO_SIDECHAIN_LAYOUTS
    );

    forge_ab_editor!(
        title: "FORGE OPENSPACER UNMASK",
        subtitle: "4-Channel Sidechain Dynamic Spectral Unmasker (Vocal -> Music & Kick -> Sub-Bass)",
        presets: OPENSPACER_PRESETS,
        grid_id: "openspacer_grid",
        params: [
            ("amt", Float, amount, "Unmask Amount"),
            ("maxcut", Float, max_cut_db, "Max Carve Depth (dB)"),
            ("lowhz", Float, low_hz, "Low Crossover (Hz)"),
            ("highhz", Float, high_hz, "High Crossover (Hz)"),
            ("kickmode", Bool, kick_bass_mode, "Kick -> Sub-Bass Mode"),
            ("delta", Bool, delta_listen, "Delta Listen (Carved Only)"),
            ("msbal", Float, mid_side_balance, "Mid/Side Focus (-1 M .. +1 S)"),
        ]
    );

    impl_standard_plugin_lifecycle!(latency);

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
            let gr_est = estimate_gr_db(tgt_l, tgt_r, out_l, out_r);
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
        let carve_db = self.dsp.last_carve_bands_db().map(|v| -v.abs());
        self.telemetry
            .update_spectrum_and_gr(&self.analyzer.snapshot_db(), &carve_db);

        ProcessStatus::Normal
    }
}

impl_forge_formats!(
    ForgeOpenSpacerPlug,
    clap_id: "org.the40thieves.forge-audio.openspacer-unmask",
    clap_desc: "Sidechain Dynamic Spectral Unmasker (Vocal -> Music & Kick -> Bass)",
    clap_features: &[
        ClapFeature::AudioEffect,
        ClapFeature::Equalizer,
        ClapFeature::Mixing,
        ClapFeature::Stereo,
    ],
    vst3_id: *b"ForgeOpenSpacer1",
    vst3_cats: &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Eq,
        Vst3SubCategory::Dynamics,
    ],
);
