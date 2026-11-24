//! Forge Vocal Surgical, Forge Drum Trigger & Forge Vocal Tune plugins.

use crate::chassis::*;
use crate::*;

define_presets! {
    const VOCAL_SURGICAL_PRESETS {
        "Clean Lead" => { "declick": 0.40, "deplosive": 0.45, "deess": 0.55, "gate": 0.35 };
        "Broadcast Dry" => { "declick": 0.60, "deplosive": 0.70, "deess": 0.45, "gate": 0.50 };
    }
    const DRUM_TRIGGER_PRESETS {
        "Tight Kit" => { "thresh": 0.55, "neosoul": 0.0 };
        "Neo-Soul Chords" => { "thresh": 0.50, "neosoul": 1.0 };
    }
    const VOCAL_TUNE_PRESETS {
        "Natural Pop" => { "speed": 0.25, "amt": 0.78, "human": 0.30, "mix": 1.0 };
        "Hard Trap Tune" => { "speed": 0.0, "amt": 1.0, "human": 0.0, "mix": 1.0 };
        "Subtle Polish" => { "speed": 0.45, "amt": 0.55, "human": 0.45, "mix": 0.85 };
    }
}
// ============================================================================
// 3. FORGE VOCAL SURGICAL STRIP (Zero-Latency Vocal Restoration)
// ============================================================================

pub struct ForgeVocalSurgicalPlug {
    params: Arc<VocalSurgicalParams>,
    telemetry: Arc<GuiTelemetry>,
    analyzer: RealtimeSpectrumAnalyzer32,
    dsp: RealtimeVocalSurgical,
    sample_rate: f64,
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
pub(crate) struct VocalSurgicalControls {
    pub(crate) declick: f64,
    pub(crate) deplosive: f64,
    pub(crate) deess_db: f64,
    pub(crate) gate_db: f64,
}

/// Advances every Vocal Surgical smoother by one sample and returns the new values.
pub(crate) fn next_vocal_surgical_controls(params: &VocalSurgicalParams) -> VocalSurgicalControls {
    VocalSurgicalControls {
        declick: f64::from(params.declick_sensitivity.smoothed.next()),
        deplosive: f64::from(params.deplosive_strength.smoothed.next()),
        deess_db: f64::from(params.deess_threshold_db.smoothed.next()),
        gate_db: f64::from(params.breath_gate_db.smoothed.next()),
    }
}

impl_standard_plugin_default!(
    ForgeVocalSurgicalPlug,
    VocalSurgicalParams,
    RealtimeVocalSurgical
);

impl Plugin for ForgeVocalSurgicalPlug {
    forge_plugin_common!("Forge Vocal Surgical Strip", crate::chassis::STEREO_LAYOUTS);

    forge_ab_editor!(
        title: "FORGE VOCAL SURGICAL STRIP",
        subtitle: "Zero-Latency Mouth De-Click, Dynamic Plosive Suppressor, Split-Band De-Esser & Breath Gate",
        presets: VOCAL_SURGICAL_PRESETS,
        grid_id: "vocal_surgical_grid",
        params: [
            ("declick", Float, declick_sensitivity, "Mouth De-Click Sensitivity"),
            ("deplosive", Float, deplosive_strength, "Plosive Duck (<115 Hz)"),
            ("deess", Float, deess_threshold_db, "Split-Band De-Esser (dB)"),
            ("gate", Float, breath_gate_db, "Breath Gate Threshold (dB)"),
        ]
    );

    impl_standard_plugin_lifecycle!();

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        for mut channel_samples in buffer.iter_samples() {
            let ctl = next_vocal_surgical_controls(&self.params);
            let (in_l, in_r) = read_stereo(&mut channel_samples);
            let (out_l, out_r) = self.dsp.process_stereo_sample(
                in_l,
                in_r,
                ctl.declick,
                ctl.deplosive,
                ctl.deess_db,
                ctl.gate_db,
            );
            let gr_est = estimate_gr_db(in_l, in_r, out_l, out_r);
            finish_stereo_sample(
                &mut self.analyzer,
                &self.telemetry,
                &mut channel_samples,
                in_l,
                in_r,
                out_l,
                out_r,
                gr_est,
            );
        }
        finish_block_spectrum(&self.analyzer, &self.telemetry);
        ProcessStatus::Normal
    }
}

impl_forge_formats!(
    ForgeVocalSurgicalPlug,
    clap_id: "org.the40thieves.forge-audio.vocal-surgical",
    clap_desc: "Zero-Latency Vocal Mouth De-Click, Dynamic Plosive Suppressor, Split-Band De-Esser & Breath Gate",
    clap_features: &[
        ClapFeature::AudioEffect,
        ClapFeature::Restoration,
        ClapFeature::Compressor,
        ClapFeature::Stereo,
    ],
    vst3_id: *b"ForgeVocalSurg01",
    vst3_cats: &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Restoration,
        Vst3SubCategory::Dynamics,
    ],
);

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
    sample_rate: f64,
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
            sample_rate: 48_000.0,
        }
    }
}

impl Plugin for ForgeDrumTriggerPlug {
    forge_plugin_common!(
        "Forge Drum Trigger & Theory MIDI",
        crate::chassis::STEREO_LAYOUTS,
        midi_in = MidiConfig::Basic,
        midi_out = MidiConfig::Basic
    );

    forge_ab_editor!(
        title: "FORGE DRUM TRIGGER & THEORY MIDI",
        subtitle: "Real-Time Audio-to-MIDI 3-Band Drum Transient Replacer (C1/D1/F#1) & Neo-Soul Chord Voicer",
        presets: DRUM_TRIGGER_PRESETS,
        grid_id: "drum_trigger_grid",
        params: [
            ("thresh", Float, threshold_db, "Transient Trigger Threshold (dB)"),
            ("neosoul", Bool, expand_midi_chords, "1-Finger MIDI -> Neo-Soul 9th Voicing"),
        ]
    );

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        let sr = f64::from(buffer_config.sample_rate);
        self.sample_rate = sr;
        self.dsp.reset(sr);
        self.analyzer.reset(sr);
        self.notes.set_sample_rate(sr);
        true
    }

    /// Notes cannot be sent from here, so held notes are released at the start of the next block.
    fn reset(&mut self) {
        self.dsp.reset(self.sample_rate);
        self.notes.set_sample_rate(self.sample_rate);
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

            let (in_l, in_r) = read_stereo(&mut channel_samples);
            self.analyzer.push_stereo_sample(in_l, in_r);
            self.telemetry.update_peaks(in_l, in_r, in_l, in_r, 0.0);
            self.trigger_drums(context, sample_idx as u32, in_l, in_r, thresh);
        }
        // Events timed at or past the end of the block (should not happen, but never drop them).
        while let Some(event) = pending {
            self.forward_event(context, event);
            pending = context.next_event();
        }
        finish_block_spectrum(&self.analyzer, &self.telemetry);

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

impl_forge_formats!(
    ForgeDrumTriggerPlug,
    clap_id: "org.the40thieves.forge-audio.drum-trigger-theory",
    clap_desc: "Real-Time Audio-to-MIDI 3-Band Drum Transient Replacer (C1/D1/F#1) & Neo-Soul Chord Voicer",
    clap_features: &[
        ClapFeature::AudioEffect,
        ClapFeature::NoteEffect,
        ClapFeature::Drum,
        ClapFeature::Utility,
        ClapFeature::Stereo,
    ],
    vst3_id: *b"ForgeDrumTrig001",
    vst3_cats: &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Drum,
        Vst3SubCategory::Tools,
    ],
);

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
            scale_mode: enum_int_param(
                "Scale Mode",
                0,
                &[
                    "Chromatic",
                    "Major",
                    "Minor",
                    "Pentatonic Minor",
                    "Pentatonic Major",
                ],
                &[("pentmin", 3), ("pentmaj", 4)],
            ),
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
    forge_plugin_common!("Forge Vocal Tune", crate::chassis::STEREO_OR_MONO_LAYOUTS);

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        let params = self.params.clone();
        let telemetry = self.telemetry.clone();
        let pitch_tel = self.pitch_telemetry.clone();
        forge_editor(
            self.params.editor_state.clone(),
            AbBankState::default(),
            move |ui, setter, ab_state| {
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
                param_grid(ui, "vocal_tune_grid", |ui| {
                    param_row(ui, setter, "Key Root", &params.key_root);
                    param_row(ui, setter, "Scale Mode", &params.scale_mode);
                    param_row(ui, setter, "Retune Speed (ms)", &params.retune_speed_ms);
                    param_row(ui, setter, "Correction Amount", &params.retune_amount);
                    param_row(ui, setter, "Humanize (Vibrato)", &params.humanize);
                    param_row(ui, setter, "Formant Shift (st)", &params.formant_shift_st);
                    param_row(ui, setter, "Wet Mix", &params.wet_mix);
                });
            },
        )
    }

    impl_standard_plugin_lifecycle!(latency);

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
            let (in_l, in_r) = read_stereo_or_mono(&mut channel_samples);
            let (out_l, out_r) = self.dsp.process_stereo_sample(in_l, in_r);
            let corr_vis = (self.dsp.correction_cents() / 100.0) as f32;
            finish_stereo_sample(
                &mut self.analyzer,
                &self.telemetry,
                &mut channel_samples,
                in_l,
                in_r,
                out_l,
                out_r,
                corr_vis,
            );
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

impl_forge_formats!(
    ForgeVocalTunePlug,
    clap_id: "org.the40thieves.forge-audio.vocal-tune",
    clap_desc: "Real-Time Low-Latency Streaming YIN F0 + Scale-Locked Dual-Grain PSOLA Vocal Pitch Corrector",
    clap_features: &[
        ClapFeature::AudioEffect,
        ClapFeature::PitchShifter,
        ClapFeature::Stereo,
    ],
    vst3_id: *b"ForgeVocalTune01",
    vst3_cats: &[Vst3SubCategory::Fx, Vst3SubCategory::PitchShift],
);

// Export all 11 plugins into the all-in-one CLAP/VST3 bundle when `all-in-one-export` is enabled.
// Individual standalone VST3/CLAP plugins live in `plugins/forge-*` and export one plugin per bundle.
