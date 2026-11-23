//! Forge SubMono & Tape, Forge Master Limiter & Forge Phase/Match EQ plugins.

use crate::*;
// ============================================================================
// 4. FORGE SUBMONO & CONSOLE TAPE BUS
// ============================================================================

pub struct ForgeSubAndTapePlug {
    params: Arc<SubAndTapeParams>,
    telemetry: Arc<GuiTelemetry>,
    analyzer: RealtimeSpectrumAnalyzer32,
    dsp: RealtimeSubAndTapeBus,
    sample_rate: f64,
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
            sample_rate: 48_000.0,
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
        self.sample_rate = sr;
        self.dsp = RealtimeSubAndTapeBus::new(sr, f64::from(self.params.mono_cutoff_hz.value()));
        self.analyzer.reset(sr);
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
