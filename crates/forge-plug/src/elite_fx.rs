//! Forge SplitEQ, Forge Density & Clipper & Forge SpaceVerb & Preamp plugins.

use crate::chassis::*;
use crate::*;

define_presets! {
    const SPLITEQ_PRESETS {
        "Drum Punch" => { "tr_low": 0.62, "tr_mid": 0.55, "tr_high": 0.60, "tn_low": 0.48, "tn_mid": 0.45, "tn_high": 0.50 };
        "Vocal Air" => { "tr_low": 0.50, "tr_mid": 0.45, "tr_high": 0.45, "tn_low": 0.50, "tn_mid": 0.54, "tn_high": 0.62 };
    }
    const DENSITY_CLIPPER_PRESETS {
        "Glue Bus" => { "density": 0.25, "clip": 0.20, "ceil": 0.85 };
        "Loud Master Clip" => { "density": 0.40, "clip": 0.45, "ceil": 0.90 };
    }
    const SPACEVERB_PRESETS {
        "Vocal Hall" => { "model": 0.0, "pdrive": 0.25, "decay": 0.60, "damp": 0.45, "shimmer": 0.12, "duck": 0.60, "mix": 0.22 };
        "Lush Shimmer" => { "model": 0.67, "pdrive": 0.35, "decay": 0.82, "damp": 0.30, "shimmer": 0.45, "duck": 0.40, "mix": 0.38 };
    }
}
// ============================================================================
// 8. FORGE SPLITEQ TRANSIENT & TONAL SHAPER (Eventide SplitEQ / Spiff Rival)
// ============================================================================

pub struct ForgeSplitEqPlug {
    params: Arc<SplitEqParams>,
    telemetry: Arc<GuiTelemetry>,
    analyzer: RealtimeSpectrumAnalyzer32,
    dsp: RealtimeTransientTonalSplit,
    sample_rate: f64,
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

impl_standard_plugin_default!(ForgeSplitEqPlug, SplitEqParams, RealtimeTransientTonalSplit);

impl Plugin for ForgeSplitEqPlug {
    forge_plugin_common!(
        "Forge SplitEQ Transient & Tonal",
        crate::chassis::STEREO_LAYOUTS
    );

    forge_ab_editor!(
        title: "FORGE SPLITEQ — TRANSIENT vs. TONAL",
        subtitle: "Structural 3-Band Transient Attack vs. Sustained Harmonic Body Shaper",
        presets: SPLITEQ_PRESETS,
        grid_id: "spliteq_grid",
        params: [
            ("tr_low", Float, trans_low_db, "Low Transient (<250 Hz)"),
            ("tr_mid", Float, trans_mid_db, "Mid Transient (250-3.5kHz)"),
            ("tr_high", Float, trans_high_db, "High Transient (>3.5kHz)"),
            ("tn_low", Float, tonal_low_db, "Low Tonal Body (<250 Hz)"),
            ("tn_mid", Float, tonal_mid_db, "Mid Tonal Sustain (250-3.5kHz)"),
            ("tn_high", Float, tonal_high_db, "High Tonal Air (>3.5kHz)"),
        ]
    );

    impl_standard_plugin_lifecycle!();

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
            let (in_l, in_r) = read_stereo(&mut channel_samples);
            let (out_l, out_r) = self.dsp.process_stereo_sample(in_l, in_r, tr, tn, 1.5, 0);
            finish_stereo_sample(
                &mut self.analyzer,
                &self.telemetry,
                &mut channel_samples,
                in_l,
                in_r,
                out_l,
                out_r,
                0.0,
            );
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

impl_forge_formats!(
    ForgeSplitEqPlug,
    clap_id: "org.the40thieves.forge-audio.spliteq-transient-tonal",
    clap_desc: "Structural 3-Band Transient vs. Tonal Equalizer & Attack/Sustain Shaper",
    clap_features: &[
        ClapFeature::AudioEffect,
        ClapFeature::Equalizer,
        ClapFeature::TransientShaper,
        ClapFeature::Stereo,
    ],
    vst3_id: *b"ForgeSplitEq0001",
    vst3_cats: &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Eq,
        Vst3SubCategory::Dynamics,
    ],
);

// ============================================================================
// 9. FORGE DENSITY & 3-BAND ADAA CLIPPER (oeksound Bloom / Neutron 5 Rival)
// ============================================================================

pub struct ForgeDensityClipperPlug {
    params: Arc<DensityClipperParams>,
    telemetry: Arc<GuiTelemetry>,
    analyzer: RealtimeSpectrumAnalyzer32,
    dsp: RealtimeDensityAndClipper,
    sample_rate: f64,
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

impl_standard_plugin_default!(
    ForgeDensityClipperPlug,
    DensityClipperParams,
    RealtimeDensityAndClipper
);

impl Plugin for ForgeDensityClipperPlug {
    forge_plugin_common!(
        "Forge Density & 3-Band ADAA Clipper",
        crate::chassis::STEREO_LAYOUTS
    );

    forge_ab_editor!(
        title: "FORGE DENSITY & 3-BAND ADAA CLIPPER",
        subtitle: "Upward Low-Level Density Lifter + 2x Oversampled Anti-Derivative Anti-Aliased Soft Clipper",
        presets: DENSITY_CLIPPER_PRESETS,
        grid_id: "density_grid",
        params: [
            ("density", Float, upward_density, "Upward Density Amount"),
            ("clip", Float, clip_drive_db, "3-Band ADAA Clip Drive"),
            ("ceil", Float, ceiling_db, "Output Ceiling"),
        ]
    );

    impl_standard_plugin_lifecycle!(self, sr => RealtimeDensityAndClipper::new(sr));

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
            let (in_l, in_r) = read_stereo(&mut channel_samples);
            let (out_l, out_r) = self
                .dsp
                .process_stereo_sample(in_l, in_r, density, drive, ceil);
            finish_stereo_sample(
                &mut self.analyzer,
                &self.telemetry,
                &mut channel_samples,
                in_l,
                in_r,
                out_l,
                out_r,
                0.0,
            );
        }
        finish_block_spectrum(&self.analyzer, &self.telemetry);
        ProcessStatus::Normal
    }
}

impl_forge_formats!(
    ForgeDensityClipperPlug,
    clap_id: "org.the40thieves.forge-audio.density-adaa-clipper",
    clap_desc: "Upward Low-Level Density Compressor & 3-Band 2x-Oversampled ADAA Soft Clipper",
    clap_features: &[
        ClapFeature::AudioEffect,
        ClapFeature::Compressor,
        ClapFeature::Distortion,
        ClapFeature::Mastering,
        ClapFeature::Stereo,
    ],
    vst3_id: *b"ForgeDensClip001",
    vst3_cats: &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Dynamics,
        Vst3SubCategory::Distortion,
        Vst3SubCategory::Mastering,
    ],
);

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
pub(crate) const REVERB_DECAY_KNOB_MIN: f32 = 0.05;
/// Top of the stored `decay` knob range.
pub(crate) const REVERB_DECAY_KNOB_MAX: f32 = 0.98;
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
pub(crate) fn string_to_reverb_decay_knob(text: &str) -> Option<f32> {
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
            room_geometry: enum_int_param(
                "Room Geometry",
                1,
                &["Plate", "Hall", "Chamber", "Room", "Ambient"],
                &[],
            ),
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
    forge_plugin_common!(
        "Forge SpaceVerb FDN & Neural Preamp",
        crate::chassis::STEREO_LAYOUTS
    );

    forge_ab_editor!(
        persisted_scroll,
        title: "FORGE SPACEVERB FDN & NEURAL WAVENET PREAMP",
        subtitle: "Dilated Causal WaveNet Preamp (.nam Ready) + Moorer ER / Allpass Diffuser / 8x8 Hadamard FDN Reverb",
        presets: SPACEVERB_PRESETS,
        grid_id: "spaceverb_grid",
        params: [
            ("model", Int, neural_model_idx, "Neural Hardware Model (0..3)"),
            ("pdrive", Float, preamp_drive, "Neural WaveNet Drive"),
            ("decay", Float, reverb_decay, "8x8 FDN Reverb Decay"),
            ("damp", Float, reverb_damping, "High Damping"),
            ("shimmer", Float, shimmer_amount, "+12 st Octave Shimmer"),
            ("duck", Float, vocal_ducking, "Vocal Auto-Ducking"),
            ("mix", Float, wet_mix, "Reverb Wet Mix"),
            ("room", Int, room_geometry, "Room Geometry"),
            ("declo", Float, decay_low_mult, "Low Decay Rate EQ (<350 Hz)"),
            ("dechi", Float, decay_high_mult, "High Decay Rate EQ (>4 kHz)"),
            ("moddep", Float, mod_depth, "Chorus Modulation Depth"),
        ]
    );

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
            let (in_l, in_r) = read_stereo(&mut channel_samples);
            let (pr_l, pr_r) =
                self.preamp_dsp
                    .process_stereo_sample(in_l, in_r, pdrive * 12.0, 0.0);
            let (out_l, out_r) = self
                .reverb_dsp
                .process_stereo_sample(pr_l, pr_r, decay, damp, shimmer, duck, mix);
            finish_stereo_sample(
                &mut self.analyzer,
                &self.telemetry,
                &mut channel_samples,
                in_l,
                in_r,
                out_l,
                out_r,
                0.0,
            );
        }
        finish_block_spectrum(&self.analyzer, &self.telemetry);
        ProcessStatus::Normal
    }
}

impl_forge_formats!(
    ForgeSpaceVerbAndPreampPlug,
    clap_id: "org.the40thieves.forge-audio.spaceverb-neural-preamp",
    clap_desc: "Dilated Causal WaveNet Preamp Modeler (1073/1176/V76/A800) & 8x8 Hadamard FDN Shimmer/Ducking Reverb",
    clap_features: &[
        ClapFeature::AudioEffect,
        ClapFeature::Reverb,
        ClapFeature::Distortion,
        ClapFeature::Stereo,
    ],
    vst3_id: *b"ForgeSpaceVerb01",
    vst3_cats: &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Reverb,
        Vst3SubCategory::Distortion,
    ],
);
