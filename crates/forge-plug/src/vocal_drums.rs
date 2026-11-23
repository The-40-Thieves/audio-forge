//! Forge Vocal Surgical, Forge Drum Trigger & Forge Vocal Tune plugins.

use crate::*;
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
