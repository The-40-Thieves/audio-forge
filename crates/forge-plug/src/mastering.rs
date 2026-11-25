//! Forge SubMono & Tape, Forge Master Limiter & Forge Phase/Match EQ plugins.

use crate::chassis::*;
use crate::*;

define_presets! {
    const SUB_TAPE_PRESETS {
        "Tight Club Sub" => { "cutoff": 0.38, "sub": 0.35, "width": 0.55, "drive": 0.25 };
        "Warm 15 IPS" => { "cutoff": 0.30, "sub": 0.20, "width": 0.50, "drive": 0.55 };
    }
    const LIMITER_PRESETS {
        "Spotify -1 dBTP" => { "drive": 0.45, "ceil": 0.85, "rel": 0.18 };
        "Club Loud" => { "drive": 0.65, "ceil": 0.90, "rel": 0.08 };
        "Transparent Master" => { "drive": 0.40, "ceil": 0.85, "rel": 0.35 };
    }
    const PHASE_EQ_PRESETS {
        "AI De-Harsh Flat" => { "inv": 0.0, "eqamt": 0.50, "genre": 0.0, "autogain": 1.0 };
        "Modern Hip-Hop" => { "inv": 0.0, "eqamt": 0.70, "genre": 0.25, "autogain": 1.0 };
    }
}
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
    forge_plugin_common!(
        "Forge SubMono & Console Tape",
        crate::chassis::STEREO_LAYOUTS
    );

    forge_ab_editor!(
        title: "FORGE SUBMONO & CONSOLE TAPE (JILES-ATHERTON RK4)",
        subtitle: "Linkwitz-Riley <160Hz Mono-Maker, Allpass Disperser, Jiles-Atherton RK4 Hysteresis & Head-Bump",
        presets: SUB_TAPE_PRESETS,
        grid_id: "subtape_grid",
        params: [
            ("cutoff", Float, mono_cutoff_hz, "Sub Mono Cutoff (Hz)"),
            ("sub", Float, sub_reinforce, "Sub Harmonic + Disperser Reinforce"),
            ("width", Float, stereo_width, "High Stereo Width"),
            ("drive", Float, tape_drive, "Jiles-Atherton RK4 + Console8 Drive"),
        ]
    );

    impl_standard_plugin_lifecycle!(
        self,
        sr => RealtimeSubAndTapeBus::new(sr, f64::from(self.params.mono_cutoff_hz.value()))
    );

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
            let (in_l, in_r) = read_stereo(&mut channel_samples);
            let (out_l, out_r) = self
                .dsp
                .process_stereo_sample(in_l, in_r, sub, width, drive);
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
    ForgeSubAndTapePlug,
    clap_id: "org.the40thieves.forge-audio.submono-tape",
    clap_desc: "Linkwitz-Riley <160Hz Sub Mono-Maker, Harmonic Bass Reinforcer & Console8/ChowTape Saturation",
    clap_features: &[
        ClapFeature::AudioEffect,
        ClapFeature::Mastering,
        ClapFeature::Distortion,
        ClapFeature::Stereo,
    ],
    vst3_id: *b"ForgeSubAndTape1",
    vst3_cats: &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Mastering,
        Vst3SubCategory::Distortion,
    ],
);

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
            limiter_style: enum_int_param(
                "Limiter Style",
                0,
                &["Transparent", "Punchy", "Dynamic", "Aggressive"],
                &[],
            ),
            channel_link: FloatParam::new(
                "Channel Link",
                1.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),
            dither_mode: enum_int_param(
                "Output Dither",
                0,
                &["Off", "TPDF 24-bit", "Noise-Shaped 16-bit"],
                &[("none", 0), ("tpdf", 1), ("noise-shaped", 2), ("shaped", 2)],
            ),
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
    forge_plugin_common!(
        "Forge Master True-Peak Limiter",
        crate::chassis::STEREO_LAYOUTS
    );

    forge_ab_editor!(
        persisted_scroll,
        title: "FORGE MASTER TRUE-PEAK BRICKWALL LIMITER",
        subtitle: "Lookahead 4x/8x Inter-Sample True-Peak Brickwall Limiter (-1.0 dBTP) & ITU-R BS.1770-4 Meter",
        drag: ("CEIL", ceiling_dbtp, egui::Color32::from_rgb(248, 113, 113)),
        presets: LIMITER_PRESETS,
        grid_id: "limiter_grid",
        params: [
            ("drive", Float, drive_db, "Input Gain / Drive (dB)"),
            ("ceil", Float, ceiling_dbtp, "True-Peak Ceiling (dBTP)"),
            ("rel", Float, release_ms, "Release Time (ms)"),
            ("style", Int, limiter_style, "Limiter Style"),
            ("clink", Float, channel_link, "Channel Link"),
            ("dither", Int, dither_mode, "Output Dither"),
            ("isp8x", Bool, isp_8x, "8x ISP Oversampling"),
            ("unity", Bool, unity_audition, "1:1 Unity Gain Audition"),
        ]
    );

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
            let (in_l, in_r) = read_stereo(&mut channel_samples);
            let (mut out_l, mut out_r) =
                self.dsp.process_stereo_sample(in_l, in_r, drive, ceil, rel);
            if unity_audition && drive.abs() > 0.01 {
                let comp = 10.0_f32.powf((-drive as f32) / 20.0);
                out_l *= comp;
                out_r *= comp;
            }
            self.loudness_meter.push_output_sample(out_l, out_r);
            let (_lufs, _dbtp, gr) = self.dsp.telemetry();
            finish_stereo_sample(
                &mut self.analyzer,
                &self.telemetry,
                &mut channel_samples,
                in_l,
                in_r,
                out_l,
                out_r,
                -gr,
            );
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

impl_forge_formats!(
    ForgeMasterLimiterPlug,
    clap_id: "org.the40thieves.forge-audio.master-limiter",
    clap_desc: "Lookahead 4x Inter-Sample True-Peak Brickwall Limiter (-1.0 dBTP) & ITU-R BS.1770-4 Meter",
    clap_features: &[
        ClapFeature::AudioEffect,
        ClapFeature::Limiter,
        ClapFeature::Mastering,
        ClapFeature::Stereo,
    ],
    vst3_id: *b"ForgeMasterLim01",
    vst3_cats: &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Mastering,
        Vst3SubCategory::Dynamics,
    ],
);

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
    main_cap: [f64; 4096],
    sc_cap: [f64; 4096],
    cap_pos: usize,
    auto_delay: usize,
    auto_invert: bool,
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
    #[id = "autoal"]
    pub auto_align_sidechain: BoolParam,
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
            .with_value_to_string(formatters::v2s_f32_rounded(3))
            .with_smoother(SmoothingStyle::Linear(15.0)),
            genre_target_idx: IntParam::new(
                "Genre Target (0=Pop,1=Trap,2=RnB,3=EDM,4=Rock,5=Jazz)",
                0,
                IntRange::Linear { min: 0, max: 5 },
            ),
            auto_gain_comp: BoolParam::new("ZLEqualizer Auto-Gain Compensation", true),
            auto_align_sidechain: BoolParam::new("Auto-Align to Sidechain", false),
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
            main_cap: [0.0; 4096],
            sc_cap: [0.0; 4096],
            cap_pos: 0,
            auto_delay: 0,
            auto_invert: false,
        }
    }
}

fn estimate_lag_and_polarity(
    main_cap: &[f64; 4096],
    sc_cap: &[f64; 4096],
) -> Option<(usize, bool)> {
    const WIN_START: usize = 512;
    const WIN_END: usize = 2048;
    let mut e_main = 0.0_f64;
    let mut e_sc = 0.0_f64;
    for t in (WIN_START..WIN_END).step_by(8) {
        e_main += main_cap[t] * main_cap[t];
        e_sc += sc_cap[t] * sc_cap[t];
    }
    let denom = (e_main * e_sc).sqrt();
    if denom <= 1e-12 {
        return None;
    }
    let mut best_delay = 0_usize;
    let mut best_dot = 0.0_f64;
    for d in 0..=512_usize {
        let mut dot = 0.0_f64;
        for t in (WIN_START..WIN_END).step_by(8) {
            dot += sc_cap[t] * main_cap[t - d];
        }
        if dot.abs() > best_dot.abs() {
            best_dot = dot;
            best_delay = d;
        }
    }
    (best_dot.abs() > 0.25 * denom).then_some((best_delay, best_dot < 0.0))
}

impl ForgePhaseAndMatchPlug {
    /// Returns the currently measured `(delay_samples, invert_polarity)` from sidechain auto-align.
    pub fn detected_sidechain_alignment(&self) -> (usize, bool) {
        (self.auto_delay, self.auto_invert)
    }

    #[inline]
    fn push_sidechain_align_sample(&mut self, main_m: f64, sc_m: f64) {
        self.main_cap[self.cap_pos] = main_m;
        self.sc_cap[self.cap_pos] = sc_m;
        self.cap_pos += 1;
        if self.cap_pos >= 2048 {
            if let Some((d, inv)) = estimate_lag_and_polarity(&self.main_cap, &self.sc_cap) {
                self.auto_delay = d;
                self.auto_invert = inv;
            }
            self.cap_pos = 0;
        }
    }

    fn build_combined_eq_and_bode(&mut self, eq_amt: f64) -> ([f64; 18], [f32; 32]) {
        let genre = match self.params.genre_target_idx.value() {
            1 => GenreTonalTarget::HipHopTrap,
            2 => GenreTonalTarget::NeoSoulRnB,
            3 => GenreTonalTarget::EdmClub,
            4 => GenreTonalTarget::RockMetal,
            5 => GenreTonalTarget::AcousticJazz,
            _ => GenreTonalTarget::ModernPop,
        };
        let target_curve = genre.target_curve_18_db();
        const AI_WARMTH_CURVE_18: [f64; 18] = [
            0.4, 0.8, 1.1, 0.6, 0.0, -0.5, -0.8, -0.4, 0.0, -0.3, -1.2, -2.2, -2.5, -1.8, -0.6,
            0.5, 1.1, 1.3,
        ];
        let mut combined = [0.0_f64; 18];
        for i in 0..18 {
            combined[i] = AI_WARMTH_CURVE_18[i] * 0.65 + target_curve[i] * 0.15;
        }
        let mut bode_32 = [0.0_f32; 32];
        for (b, val) in bode_32.iter_mut().enumerate() {
            *val = (combined[(b * 17) / 31] * eq_amt) as f32;
        }
        (combined, bode_32)
    }

    pub(crate) fn process_slices(
        &mut self,
        samples_len: usize,
        main_slices: &mut [&mut [f32]],
        sc_pair: Option<(&[f32], &[f32])>,
        auto_align: bool,
    ) {
        let (delay, inv) = if auto_align && sc_pair.is_some() {
            (self.auto_delay, self.auto_invert)
        } else {
            (
                self.params.delay_samples.value().max(0) as usize,
                self.params.invert_polarity.value(),
            )
        };
        let eq_amt = f64::from(self.params.ai_warmth_eq_amount.smoothed.next());
        for _ in 1..samples_len {
            let _ = self.params.ai_warmth_eq_amount.smoothed.next();
        }
        let use_agc = self.params.auto_gain_comp.value();
        let (combined, bode_32) = self.build_combined_eq_and_bode(eq_amt);
        self.phase_dsp.set_alignment(delay, inv);
        self.eq_dsp.set_gains(&combined, eq_amt);

        for i in 0..samples_len {
            let in_l = main_slices
                .first()
                .and_then(|c| c.get(i).copied())
                .unwrap_or(0.0);
            let in_r = main_slices
                .get(1)
                .and_then(|c| c.get(i).copied())
                .unwrap_or(in_l);
            if auto_align && let Some((sc_l_ch, sc_r_ch)) = sc_pair {
                let sc_l = sc_l_ch.get(i).copied().unwrap_or(0.0);
                let sc_r = sc_r_ch.get(i).copied().unwrap_or(sc_l);
                self.push_sidechain_align_sample(
                    0.5 * f64::from(in_l + in_r),
                    0.5 * f64::from(sc_l + sc_r),
                );
            }
            let (al_l, al_r) = self.phase_dsp.process_stereo_sample(in_l, in_r);
            let (eq_l, eq_r) = self.eq_dsp.process_stereo_sample(al_l, al_r);
            let (out_l, out_r) = self
                .auto_gain
                .process_stereo(in_l, in_r, eq_l, eq_r, use_agc);
            self.analyzer.push_stereo_sample(out_l, out_r);
            self.telemetry.update_peaks(in_l, in_r, out_l, out_r, 0.0);
            if let Some(c0) = main_slices.first_mut() {
                c0[i] = out_l;
            }
            if let Some(c1) = main_slices.get_mut(1) {
                c1[i] = out_r;
            }
        }
        self.telemetry
            .update_spectrum_and_gr(&self.analyzer.snapshot_db(), &bode_32);
    }
}

impl Plugin for ForgePhaseAndMatchPlug {
    forge_plugin_common!(
        "Forge PhaseAlign & 18-Band EQ",
        crate::chassis::STEREO_SIDECHAIN_OR_STEREO_LAYOUTS
    );

    forge_ab_editor!(
        title: "FORGE PHASE-ALIGN & 18-BAND MASTERING EQ",
        subtitle: "Sample-Accurate Phase Aligner, Nyquist De-Cramped 18-Band AI De-Harsh EQ, Bode Overlay & Auto-Gain",
        presets: PHASE_EQ_PRESETS,
        grid_id: "phase_eq_grid",
        params: [
            ("delay", Int, delay_samples, "Phase Delay Offset (spl)"),
            ("inv", Bool, invert_polarity, "Invert Polarity (180 deg)"),
            ("eqamt", Float, ai_warmth_eq_amount, "18-Band AI De-Harsh & Genre Amount"),
            ("genre", Int, genre_target_idx, "Genre Target (0=Pop..5=Jazz)"),
            ("autogain", Bool, auto_gain_comp, "Auto-Gain Compensation (AGC)"),
            ("autoal", Bool, auto_align_sidechain, "Auto-Align to Sidechain"),
        ]
    );

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
        self.main_cap.fill(0.0);
        self.sc_cap.fill(0.0);
        self.cap_pos = 0;
        self.auto_delay = 0;
        self.auto_invert = false;
        true
    }

    fn reset(&mut self) {
        self.phase_dsp.reset();
        self.eq_dsp.reset();
        self.auto_gain.reset();
        self.analyzer.clear_state();
        self.main_cap.fill(0.0);
        self.sc_cap.fill(0.0);
        self.cap_pos = 0;
        self.auto_delay = 0;
        self.auto_invert = false;
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let samples_len = buffer.samples();
        let sc_buf = aux.inputs.first_mut();
        let sc_pair = sc_buf.as_ref().map(|b| {
            let s = b.as_slice_immutable();
            let l: &[f32] = s.first().map_or(&[], |c| &**c);
            let r: &[f32] = s.get(1).map_or(l, |c| &**c);
            (l, r)
        });
        let main_slices = buffer.as_slice();
        let mut slice_refs: [&mut [f32]; 2] = [&mut [], &mut []];
        let count = main_slices.len().min(2);
        for (dst, src) in slice_refs
            .iter_mut()
            .zip(main_slices.iter_mut())
            .take(count)
        {
            *dst = src;
        }
        let auto_align = self.params.auto_align_sidechain.value();
        self.process_slices(samples_len, &mut slice_refs[..count], sc_pair, auto_align);
        ProcessStatus::Normal
    }
}

impl_forge_formats!(
    ForgePhaseAndMatchPlug,
    clap_id: "org.the40thieves.forge-audio.phase-and-matcheq",
    clap_desc: "Sample-Accurate Phase/Polarity Aligner, 18-Band AI De-Harsh EQ & Genre Tonal Balance Target",
    clap_features: &[
        ClapFeature::AudioEffect,
        ClapFeature::Equalizer,
        ClapFeature::Mastering,
        ClapFeature::PhaseVocoder,
        ClapFeature::Stereo,
    ],
    vst3_id: *b"ForgePhaseEq0001",
    vst3_cats: &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Eq,
        Vst3SubCategory::Mastering,
    ],
);
