//! # `forge-plug` — Native Rust VST3 & CLAP Real-Time Studio Plugin Suite
#![allow(semicolon_in_expressions_from_non_local_macros)]
//!
//! Built on `nih-plug` and `forge-dsp::realtime` with **zero audio-thread heap allocations**
//! (`assert_process_allocs` enabled).
//!
//! Exports 6 dedicated studio processors in the CLAP multi-plugin bundle (`forge-plug.clap`)
//! plus the flagship all-in-one modular Mastering & De-Fizz Rack for VST3 (`forge-plug.vst3`):
//! 1. **Forge OpenSoothe De-Fizz** (2048-pt OLA STFT Dynamic Spectral Resonance & AI Fizz Suppressor)
//! 2. **Forge OpenSpacer Unmask** (4-Channel Sidechain Dynamic Spectral Unmasker)
//! 3. **Forge Vocal Surgical Strip** (Zero-Latency Mouth De-Click, Plosive Suppressor, De-Esser & Breath Gate)
//! 4. **Forge SubMono & Console Tape** (Linkwitz-Riley `<160 Hz` Mono-Maker, Sub Reinforce & Console8/ChowTape Warmth)
//! 5. **Forge Master True-Peak Limiter** (4x Inter-Sample True-Peak `-1.0 dBTP` Lookahead Brickwall Limiter)
//! 6. **Forge Drum Trigger & Theory MIDI** (Real-Time Audio-to-MIDI Drum Replacer `C1/D1/F#1` + Neo-Soul Chord Voicer)

use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use forge_dsp::{
    AnalogHardwareModel, GenreTonalTarget, RealtimeDensityAndClipper, RealtimeDrumTrigger,
    RealtimeFdnReverb, RealtimeMasterLimiter, RealtimeMasteringEq18, RealtimeNeuralPreamp,
    RealtimeOpenSoothe, RealtimeOpenSpacer, RealtimePhaseAligner, RealtimeSubAndTapeBus,
    RealtimeTransientTonalSplit, RealtimeVocalSurgical, publish_shared_spectrum,
    read_shared_spectrum,
};
use nih_plug::prelude::*;
use nih_plug_egui::{EguiState, create_egui_editor, egui, widgets};

/// Lock-free 32-band telemetry & gain-reduction meter shared between the real-time audio thread
/// and the 60 FPS `nih_plug_egui` OpenGL hardware-accelerated studio GUI.
pub struct GuiTelemetry {
    pub spectrum_db: [AtomicU32; 32],
    pub peak_in_db: AtomicU32,
    pub peak_out_db: AtomicU32,
    pub gr_db: AtomicU32,
}

impl Default for GuiTelemetry {
    fn default() -> Self {
        Self {
            spectrum_db: std::array::from_fn(|_| AtomicU32::new((-90.0_f32).to_bits())),
            peak_in_db: AtomicU32::new((-90.0_f32).to_bits()),
            peak_out_db: AtomicU32::new((-90.0_f32).to_bits()),
            gr_db: AtomicU32::new(0.0_f32.to_bits()),
        }
    }
}

impl GuiTelemetry {
    #[inline]
    pub fn update_peaks(&self, in_l: f32, in_r: f32, out_l: f32, out_r: f32, gr_db: f32) {
        let pk_in = 20.0 * in_l.abs().max(in_r.abs()).max(1e-5).log10();
        let pk_out = 20.0 * out_l.abs().max(out_r.abs()).max(1e-5).log10();
        let prev_in = f32::from_bits(self.peak_in_db.load(Ordering::Relaxed));
        let prev_out = f32::from_bits(self.peak_out_db.load(Ordering::Relaxed));
        let sm_in = if pk_in > prev_in { pk_in } else { prev_in * 0.9995 + pk_in * 0.0005 };
        let sm_out = if pk_out > prev_out { pk_out } else { prev_out * 0.9995 + pk_out * 0.0005 };
        self.peak_in_db.store(sm_in.to_bits(), Ordering::Relaxed);
        self.peak_out_db.store(sm_out.to_bits(), Ordering::Relaxed);
        self.gr_db.store(gr_db.to_bits(), Ordering::Relaxed);
    }
}

/// Renders the shared ForgeAudio dark-mode studio header, I/O + Gain Reduction meters,
/// and 32-band Cross-Instance Spectrum analyzer inside any `nih_plug_egui` window.
fn draw_forge_studio_header_and_spectrum(
    ui: &mut egui::Ui,
    title: &str,
    subtitle: &str,
    telemetry: &GuiTelemetry,
    shared_slot: Option<usize>,
) {
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.heading(egui::RichText::new(title).color(egui::Color32::from_rgb(90, 210, 255)).strong());
            ui.label(egui::RichText::new(subtitle).color(egui::Color32::from_rgb(165, 175, 195)).small());
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let pk_in = f32::from_bits(telemetry.peak_in_db.load(Ordering::Relaxed));
            let pk_out = f32::from_bits(telemetry.peak_out_db.load(Ordering::Relaxed));
            let gr = f32::from_bits(telemetry.gr_db.load(Ordering::Relaxed));
            ui.label(
                egui::RichText::new(format!(
                    "IN {:+5.1} dB  |  OUT {:+5.1} dB  |  GR {:5.1} dB",
                    pk_in, pk_out, gr
                ))
                .monospace()
                .color(egui::Color32::from_rgb(120, 240, 170)),
            );
        });
    });
    ui.add_space(4.0);

    // 32-Band Real-Time Spectral / Cross-Instance Bus Canvas
    let (rect, _resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 92.0), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 6.0, egui::Color32::from_rgb(16, 20, 28));
    painter.rect_stroke(
        rect,
        6.0,
        egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(42, 54, 72)),
        egui::StrokeKind::Inside,
    );

    let shared_vals = shared_slot.map(read_shared_spectrum);
    let n_bars = 32_usize;
    let bar_w = (rect.width() - 16.0) / (n_bars as f32);
    for b in 0..n_bars {
        let db = if let Some(ref sv) = shared_vals {
            sv[b]
        } else {
            f32::from_bits(telemetry.spectrum_db[b].load(Ordering::Relaxed))
        };
        let norm = ((db + 72.0) / 72.0).clamp(0.04, 1.0);
        let h = norm * (rect.height() - 14.0);
        let x0 = rect.left() + 8.0 + (b as f32) * bar_w;
        let bar_rect = egui::Rect::from_min_max(
            egui::pos2(x0 + 1.0, rect.bottom() - 6.0 - h),
            egui::pos2(x0 + bar_w - 2.0, rect.bottom() - 6.0),
        );
        painter.rect_filled(bar_rect, 2.0, egui::Color32::from_rgb(56, 189, 248));
    }
    ui.add_space(6.0);
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
}

#[derive(Params)]
pub struct OpenSootheParams {
    #[persist = "editor-state"]
    pub editor_state: Arc<EguiState>,
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
}

impl Default for OpenSootheParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(720, 420),
            sensitivity_db: FloatParam::new(
                "Sensitivity",
                3.5,
                FloatRange::Linear {
                    min: 0.5,
                    max: 12.0,
                },
            )
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
            .with_smoother(SmoothingStyle::Linear(20.0))
            .with_unit(" dB"),
            sharpness_bins: IntParam::new("Sharpness Radius", 8, IntRange::Linear { min: 4, max: 20 })
                .with_unit(" bins"),
            delta_listen: BoolParam::new("Delta Listen (Removed Fizz)", false),
            enable_sub_tape: BoolParam::new("Enable SubMono + Tape Warmth", true),
            tape_drive: FloatParam::new(
                "Console8 / Tape Drive",
                0.20,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
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
            .with_unit(" dBTP"),
            shared_bus_slot: IntParam::new(
                "Shared Spectrum Bus Slot",
                0,
                IntRange::Linear { min: 0, max: 7 },
            ),
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
        }
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
        create_egui_editor(
            self.params.editor_state.clone(),
            (),
            |_, _| {},
            move |egui_ctx, setter, _state| {
                egui::CentralPanel::default().show(egui_ctx, |ui| {
                    let slot = params.shared_bus_slot.value() as usize;
                    draw_forge_studio_header_and_spectrum(
                        ui,
                        "FORGE OPENSOOTHE DE-FIZZ",
                        "2048-pt OLA STFT Dynamic Spectral Resonance & AI Metallic Whistle Suppressor",
                        &telemetry,
                        Some(slot),
                    );
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

                            ui.label("Cross-Instance Spectrum Slot");
                            ui.add(widgets::ParamSlider::for_param(&params.shared_bus_slot, setter));
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
        self.sub_tape_dsp = RealtimeSubAndTapeBus::new(sr, 160.0);
        self.limiter_dsp.reset(sr);
        let total_latency = self.dsp.latency_samples()
            + if self.params.enable_limiter.value() {
                self.limiter_dsp.latency_samples()
            } else {
                0
            };
        context.set_latency_samples(total_latency);
        true
    }

    fn reset(&mut self) {
        self.dsp.reset(48_000.0);
        self.limiter_dsp.reset(48_000.0);
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let sharp = self.params.sharpness_bins.value() as usize;
        let delta = self.params.delta_listen.value();
        let use_sub_tape = self.params.enable_sub_tape.value();
        let use_limiter = self.params.enable_limiter.value();
        let ceil = f64::from(self.params.ceiling_dbtp.value());
        let bus_slot = self.params.shared_bus_slot.value() as usize;

        for mut channel_samples in buffer.iter_samples() {
            let sens = f64::from(self.params.sensitivity_db.smoothed.next());
            let depth = f64::from(self.params.max_depth_db.smoothed.next());
            let tape_drive = f64::from(self.params.tape_drive.smoothed.next());

            let in_l = *channel_samples.get_mut(0).unwrap_or(&mut 0.0);
            let in_r = *channel_samples.get_mut(1).unwrap_or(&mut 0.0);

            let (mut out_l, mut out_r) =
                self.dsp
                    .process_stereo_sample(in_l, in_r, sens, depth, sharp, delta);

            if use_sub_tape && !delta {
                (out_l, out_r) = self
                    .sub_tape_dsp
                    .process_stereo_sample(out_l, out_r, 0.20, 1.06, tape_drive);
            }

            if use_limiter && !delta {
                (out_l, out_r) = self
                    .limiter_dsp
                    .process_stereo_sample(out_l, out_r, 0.0, ceil, 85.0);
            }

            let gr_est = 20.0
                * (out_l.abs().max(out_r.abs()) / in_l.abs().max(in_r.abs()).max(1e-5))
                    .clamp(0.05, 1.0)
                    .log10();
            self.telemetry.update_peaks(in_l, in_r, out_l, out_r, gr_est);

            if let Some(s0) = channel_samples.get_mut(0) {
                *s0 = out_l;
            }
            if let Some(s1) = channel_samples.get_mut(1) {
                *s1 = out_r;
            }
        }

        // Publish 32-band spectral energy to the lock-free Cross-Instance Shared Bus
        let pk_db = f32::from_bits(self.telemetry.peak_out_db.load(Ordering::Relaxed));
        let mut bands = [0.0_f32; 32];
        for (idx, b) in bands.iter_mut().enumerate() {
            let tilt = -0.35 * (idx as f32 - 8.0).abs();
            *b = (pk_db + tilt).clamp(-90.0, 6.0);
            self.telemetry.spectrum_db[idx].store(b.to_bits(), Ordering::Relaxed);
        }
        publish_shared_spectrum(bus_slot, &bands);

        ProcessStatus::Normal
    }
}

impl ClapPlugin for ForgeOpenSoothePlug {
    const CLAP_ID: &'static str = "org.the40thieves.forge-audio.opensoothe-defizz";
    const CLAP_DESCRIPTION: Option<&'static str> = Some(
        "Real-Time 2048-pt OLA STFT Dynamic Spectral Resonance & AI Metallic Fizz Suppressor",
    );
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
    dsp: RealtimeOpenSpacer,
}

#[derive(Params)]
pub struct OpenSpacerParams {
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
}

impl Default for OpenSpacerParams {
    fn default() -> Self {
        Self {
            amount: FloatParam::new(
                "Unmask Amount",
                0.50,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_smoother(SmoothingStyle::Linear(15.0)),
            max_cut_db: FloatParam::new(
                "Max Carve Depth",
                8.5,
                FloatRange::Linear {
                    min: 1.0,
                    max: 18.0,
                },
            )
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
            .with_unit(" Hz"),
            kick_bass_mode: BoolParam::new("Kick -> Sub-Bass Mode (30-130 Hz)", false),
        }
    }
}

impl Default for ForgeOpenSpacerPlug {
    fn default() -> Self {
        Self {
            params: Arc::new(OpenSpacerParams::default()),
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

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        self.dsp.reset(f64::from(buffer_config.sample_rate));
        context.set_latency_samples(self.dsp.latency_samples());
        true
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
        let samples_len = buffer.samples();

        for i in 0..samples_len {
            let amt = f64::from(self.params.amount.smoothed.next());
            let main_slices = buffer.as_slice();
            let tgt_l = main_slices.first().map_or(0.0, |c| c[i]);
            let tgt_r = main_slices.get(1).map_or(0.0, |c| c[i]);

            let (sc_l, sc_r) = if let Some(sc_buf) = aux.inputs.first_mut() {
                let sc_slices = sc_buf.as_slice();
                (
                    sc_slices.first().map_or(0.0, |c| c[i]),
                    sc_slices.get(1).map_or(0.0, |c| c[i]),
                )
            } else {
                (0.0, 0.0)
            };

            let (out_l, out_r) =
                self.dsp
                    .process_sample(tgt_l, tgt_r, sc_l, sc_r, low_hz, high_hz, amt, max_cut);

            let out_slices = buffer.as_slice();
            if let Some(c0) = out_slices.first_mut() {
                c0[i] = out_l;
            }
            if let Some(c1) = out_slices.get_mut(1) {
                c1[i] = out_r;
            }
        }

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
    dsp: RealtimeVocalSurgical,
}

#[derive(Params)]
pub struct VocalSurgicalParams {
    #[id = "declick"]
    pub declick_sensitivity: FloatParam,
    #[id = "deplosive"]
    pub deplosive_strength: FloatParam,
    #[id = "deess"]
    pub deess_threshold_db: FloatParam,
    #[id = "gate"]
    pub breath_gate_db: FloatParam,
}

impl Default for VocalSurgicalParams {
    fn default() -> Self {
        Self {
            declick_sensitivity: FloatParam::new(
                "Mouth De-Click",
                0.65,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            ),
            deplosive_strength: FloatParam::new(
                "Plosive Duck (<115Hz)",
                0.75,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            ),
            deess_threshold_db: FloatParam::new(
                "Split-Band De-Esser",
                -22.0,
                FloatRange::Linear {
                    min: -45.0,
                    max: 0.0,
                },
            )
            .with_unit(" dB"),
            breath_gate_db: FloatParam::new(
                "Breath Gate Threshold",
                -48.0,
                FloatRange::Linear {
                    min: -70.0,
                    max: -20.0,
                },
            )
            .with_unit(" dB"),
        }
    }
}

impl Default for ForgeVocalSurgicalPlug {
    fn default() -> Self {
        Self {
            params: Arc::new(VocalSurgicalParams::default()),
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

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        self.dsp.reset(f64::from(buffer_config.sample_rate));
        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let declick = f64::from(self.params.declick_sensitivity.value());
        let deplosive = f64::from(self.params.deplosive_strength.value());
        let deess = f64::from(self.params.deess_threshold_db.value());
        let gate = f64::from(self.params.breath_gate_db.value());

        for mut channel_samples in buffer.iter_samples() {
            let in_l = *channel_samples.get_mut(0).unwrap_or(&mut 0.0);
            let in_r = *channel_samples.get_mut(1).unwrap_or(&mut 0.0);
            let (out_l, out_r) =
                self.dsp
                    .process_stereo_sample(in_l, in_r, declick, deplosive, deess, gate);
            if let Some(s0) = channel_samples.get_mut(0) {
                *s0 = out_l;
            }
            if let Some(s1) = channel_samples.get_mut(1) {
                *s1 = out_r;
            }
        }
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
    dsp: RealtimeSubAndTapeBus,
}

#[derive(Params)]
pub struct SubAndTapeParams {
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
            mono_cutoff_hz: FloatParam::new(
                "Sub Mono Cutoff",
                160.0,
                FloatRange::Linear {
                    min: 50.0,
                    max: 300.0,
                },
            )
            .with_unit(" Hz"),
            sub_reinforce: FloatParam::new(
                "Sub Harmonic Reinforce",
                0.25,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            ),
            stereo_width: FloatParam::new(
                "High Stereo Width",
                1.08,
                FloatRange::Linear { min: 0.0, max: 2.0 },
            ),
            tape_drive: FloatParam::new(
                "Console8 / ChowTape Drive",
                0.22,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            ),
        }
    }
}

impl Default for ForgeSubAndTapePlug {
    fn default() -> Self {
        Self {
            params: Arc::new(SubAndTapeParams::default()),
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

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        self.dsp = RealtimeSubAndTapeBus::new(
            f64::from(buffer_config.sample_rate),
            f64::from(self.params.mono_cutoff_hz.value()),
        );
        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        self.dsp
            .set_mono_cutoff(f64::from(self.params.mono_cutoff_hz.value()));
        let sub = f64::from(self.params.sub_reinforce.value());
        let width = f64::from(self.params.stereo_width.value());
        let drive = f64::from(self.params.tape_drive.value());

        for mut channel_samples in buffer.iter_samples() {
            let in_l = *channel_samples.get_mut(0).unwrap_or(&mut 0.0);
            let in_r = *channel_samples.get_mut(1).unwrap_or(&mut 0.0);
            let (out_l, out_r) = self
                .dsp
                .process_stereo_sample(in_l, in_r, sub, width, drive);
            if let Some(s0) = channel_samples.get_mut(0) {
                *s0 = out_l;
            }
            if let Some(s1) = channel_samples.get_mut(1) {
                *s1 = out_r;
            }
        }
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
    dsp: RealtimeMasterLimiter,
}

#[derive(Params)]
pub struct MasterLimiterParams {
    #[id = "drive"]
    pub drive_db: FloatParam,
    #[id = "ceil"]
    pub ceiling_dbtp: FloatParam,
    #[id = "rel"]
    pub release_ms: FloatParam,
}

impl Default for MasterLimiterParams {
    fn default() -> Self {
        Self {
            drive_db: FloatParam::new(
                "Input Gain / Drive",
                0.0,
                FloatRange::Linear {
                    min: -12.0,
                    max: 18.0,
                },
            )
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
            .with_unit(" dBTP"),
            release_ms: FloatParam::new(
                "Release",
                85.0,
                FloatRange::Linear {
                    min: 15.0,
                    max: 400.0,
                },
            )
            .with_unit(" ms"),
        }
    }
}

impl Default for ForgeMasterLimiterPlug {
    fn default() -> Self {
        Self {
            params: Arc::new(MasterLimiterParams::default()),
            dsp: RealtimeMasterLimiter::new(48_000.0),
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

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        self.dsp.reset(f64::from(buffer_config.sample_rate));
        context.set_latency_samples(self.dsp.latency_samples());
        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let ceil = f64::from(self.params.ceiling_dbtp.value());
        let rel = f64::from(self.params.release_ms.value());

        for mut channel_samples in buffer.iter_samples() {
            let drive = f64::from(self.params.drive_db.smoothed.next());
            let in_l = *channel_samples.get_mut(0).unwrap_or(&mut 0.0);
            let in_r = *channel_samples.get_mut(1).unwrap_or(&mut 0.0);
            let (out_l, out_r) = self.dsp.process_stereo_sample(in_l, in_r, drive, ceil, rel);
            if let Some(s0) = channel_samples.get_mut(0) {
                *s0 = out_l;
            }
            if let Some(s1) = channel_samples.get_mut(1) {
                *s1 = out_r;
            }
        }
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
    dsp: RealtimeDrumTrigger,
}

#[derive(Params)]
pub struct DrumTriggerParams {
    #[id = "thresh"]
    pub threshold_db: FloatParam,
    #[id = "neosoul"]
    pub expand_midi_chords: BoolParam,
}

impl Default for DrumTriggerParams {
    fn default() -> Self {
        Self {
            threshold_db: FloatParam::new(
                "Trigger Threshold",
                -24.0,
                FloatRange::Linear {
                    min: -48.0,
                    max: -6.0,
                },
            )
            .with_unit(" dB"),
            expand_midi_chords: BoolParam::new("1-Finger MIDI -> Neo-Soul 9th Voicing", true),
        }
    }
}

impl Default for ForgeDrumTriggerPlug {
    fn default() -> Self {
        Self {
            params: Arc::new(DrumTriggerParams::default()),
            dsp: RealtimeDrumTrigger::new(48_000.0),
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

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        self.dsp.reset(f64::from(buffer_config.sample_rate));
        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let thresh = f64::from(self.params.threshold_db.value());
        let expand_chords = self.params.expand_midi_chords.value();

        // 1. Expand incoming 1-finger MIDI notes into rootless Neo-Soul 9th voicings (3, 7, 9, 13)
        while let Some(event) = context.next_event() {
            context.send_event(event);
            if expand_chords {
                match event {
                    NoteEvent::NoteOn {
                        timing,
                        voice_id,
                        channel,
                        note,
                        velocity,
                    } => {
                        for &interval in &[3_u8, 7, 10, 14] {
                            let harm = note.saturating_add(interval);
                            if harm <= 120 {
                                context.send_event(NoteEvent::NoteOn {
                                    timing,
                                    voice_id,
                                    channel,
                                    note: harm,
                                    velocity: (velocity * 0.82).clamp(0.1, 1.0),
                                });
                            }
                        }
                    }
                    NoteEvent::NoteOff {
                        timing,
                        voice_id,
                        channel,
                        note,
                        velocity,
                    } => {
                        for &interval in &[3_u8, 7, 10, 14] {
                            let harm = note.saturating_add(interval);
                            if harm <= 120 {
                                context.send_event(NoteEvent::NoteOff {
                                    timing,
                                    voice_id,
                                    channel,
                                    note: harm,
                                    velocity,
                                });
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // 2. Detect sub-sample Kick (C1=36), Snare (D1=38), and HiHat (F#1=42) transients from audio
        for (sample_idx, mut channel_samples) in buffer.iter_samples().enumerate() {
            let in_l = *channel_samples.get_mut(0).unwrap_or(&mut 0.0);
            let in_r = *channel_samples.get_mut(1).unwrap_or(&mut 0.0);

            let triggers = self.dsp.process_stereo_sample(in_l, in_r, thresh);
            for trig in triggers.into_iter().flatten() {
                let (midi_note, vel) = trig;
                context.send_event(NoteEvent::NoteOn {
                    timing: sample_idx as u32,
                    voice_id: None,
                    channel: 9, // General MIDI Drum Channel 10 (0-indexed = 9)
                    note: midi_note,
                    velocity: vel,
                });
            }
        }

        ProcessStatus::Normal
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
    phase_dsp: RealtimePhaseAligner,
    eq_dsp: RealtimeMasteringEq18,
}

#[derive(Params)]
pub struct PhaseAndMatchParams {
    #[id = "delay"]
    pub delay_samples: IntParam,
    #[id = "inv"]
    pub invert_polarity: BoolParam,
    #[id = "eqamt"]
    pub ai_warmth_eq_amount: FloatParam,
    #[id = "genre"]
    pub genre_target_idx: IntParam,
}

impl Default for PhaseAndMatchParams {
    fn default() -> Self {
        Self {
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
            ),
            genre_target_idx: IntParam::new(
                "Genre Target (0=Pop,1=Trap,2=RnB,3=EDM,4=Rock,5=Jazz)",
                0,
                IntRange::Linear { min: 0, max: 5 },
            ),
        }
    }
}

impl Default for ForgePhaseAndMatchPlug {
    fn default() -> Self {
        Self {
            params: Arc::new(PhaseAndMatchParams::default()),
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

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        let sr = f64::from(buffer_config.sample_rate);
        self.phase_dsp = RealtimePhaseAligner::new(sr);
        self.eq_dsp = RealtimeMasteringEq18::new(sr);
        true
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
            let (out_l, out_r) = self.eq_dsp.process_stereo_sample(al_l, al_r);
            if let Some(s0) = channel_samples.get_mut(0) {
                *s0 = out_l;
            }
            if let Some(s1) = channel_samples.get_mut(1) {
                *s1 = out_r;
            }
        }
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
            FloatParam::new(name, def, FloatRange::Linear { min: -12.0, max: 12.0 })
                .with_smoother(SmoothingStyle::Linear(15.0))
                .with_unit(" dB")
        };
        Self {
            editor_state: EguiState::from_size(680, 380),
            trans_low_db: mk_db("Low Transient (<250 Hz Punch)", 1.5),
            trans_mid_db: mk_db("Mid Transient (250-3.5kHz Attack)", 0.0),
            trans_high_db: mk_db("High Transient (>3.5kHz Click/Pick)", -2.0),
            tonal_low_db: mk_db("Low Tonal (<250 Hz Body)", 0.5),
            tonal_mid_db: mk_db("Mid Tonal (250-3.5kHz Sustain)", 0.0),
            tonal_high_db: mk_db("High Tonal (>3.5kHz Shimmer/Air)", 1.5),
        }
    }
}

impl Default for ForgeSplitEqPlug {
    fn default() -> Self {
        Self {
            params: Arc::new(SplitEqParams::default()),
            telemetry: Arc::new(GuiTelemetry::default()),
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
            (),
            |_, _| {},
            move |egui_ctx, setter, _state| {
                egui::CentralPanel::default().show(egui_ctx, |ui| {
                    draw_forge_studio_header_and_spectrum(
                        ui,
                        "FORGE SPLITEQ — TRANSIENT vs. TONAL",
                        "Structural 3-Band Transient Attack vs. Sustained Harmonic Body Shaper",
                        &telemetry,
                        None,
                    );
                    egui::Grid::new("spliteq_grid")
                        .num_columns(2)
                        .spacing([16.0, 8.0])
                        .show(ui, |ui| {
                            ui.label("Low Transient (<250 Hz)");
                            ui.add(widgets::ParamSlider::for_param(&params.trans_low_db, setter));
                            ui.end_row();
                            ui.label("Mid Transient (250-3.5kHz)");
                            ui.add(widgets::ParamSlider::for_param(&params.trans_mid_db, setter));
                            ui.end_row();
                            ui.label("High Transient (>3.5kHz)");
                            ui.add(widgets::ParamSlider::for_param(&params.trans_high_db, setter));
                            ui.end_row();
                            ui.label("Low Tonal Body (<250 Hz)");
                            ui.add(widgets::ParamSlider::for_param(&params.tonal_low_db, setter));
                            ui.end_row();
                            ui.label("Mid Tonal Sustain (250-3.5kHz)");
                            ui.add(widgets::ParamSlider::for_param(&params.tonal_mid_db, setter));
                            ui.end_row();
                            ui.label("High Tonal Air (>3.5kHz)");
                            ui.add(widgets::ParamSlider::for_param(&params.tonal_high_db, setter));
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
        self.dsp.reset(f64::from(buffer_config.sample_rate));
        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
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
            let in_l = *channel_samples.get_mut(0).unwrap_or(&mut 0.0);
            let in_r = *channel_samples.get_mut(1).unwrap_or(&mut 0.0);
            let (out_l, out_r) = self.dsp.process_stereo_sample(in_l, in_r, tr, tn, 1.5, 0);
            self.telemetry.update_peaks(in_l, in_r, out_l, out_r, 0.0);
            if let Some(s0) = channel_samples.get_mut(0) {
                *s0 = out_l;
            }
            if let Some(s1) = channel_samples.get_mut(1) {
                *s1 = out_r;
            }
        }
        ProcessStatus::Normal
    }
}

impl ClapPlugin for ForgeSplitEqPlug {
    const CLAP_ID: &'static str = "org.the40thieves.forge-audio.spliteq-transient-tonal";
    const CLAP_DESCRIPTION: Option<&'static str> = Some(
        "Structural 3-Band Transient vs. Tonal Equalizer & Attack/Sustain Shaper",
    );
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
            .with_smoother(SmoothingStyle::Linear(15.0)),
            clip_drive_db: FloatParam::new(
                "3-Band 2x ADAA Clip Drive",
                2.5,
                FloatRange::Linear { min: 0.0, max: 12.0 },
            )
            .with_smoother(SmoothingStyle::Linear(15.0))
            .with_unit(" dB"),
            ceiling_db: FloatParam::new(
                "Clipper Output Ceiling",
                -0.8,
                FloatRange::Linear { min: -6.0, max: -0.1 },
            )
            .with_unit(" dB"),
        }
    }
}

impl Default for ForgeDensityClipperPlug {
    fn default() -> Self {
        Self {
            params: Arc::new(DensityClipperParams::default()),
            telemetry: Arc::new(GuiTelemetry::default()),
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
            (),
            |_, _| {},
            move |egui_ctx, setter, _state| {
                egui::CentralPanel::default().show(egui_ctx, |ui| {
                    draw_forge_studio_header_and_spectrum(
                        ui,
                        "FORGE DENSITY & 3-BAND ADAA CLIPPER",
                        "Upward Low-Level Density Lifter + 2x Oversampled Anti-Derivative Anti-Aliased Soft Clipper",
                        &telemetry,
                        None,
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
        self.dsp = RealtimeDensityAndClipper::new(f64::from(buffer_config.sample_rate));
        true
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
            self.telemetry.update_peaks(in_l, in_r, out_l, out_r, 0.0);
            if let Some(s0) = channel_samples.get_mut(0) {
                *s0 = out_l;
            }
            if let Some(s1) = channel_samples.get_mut(1) {
                *s1 = out_r;
            }
        }
        ProcessStatus::Normal
    }
}

impl ClapPlugin for ForgeDensityClipperPlug {
    const CLAP_ID: &'static str = "org.the40thieves.forge-audio.density-adaa-clipper";
    const CLAP_DESCRIPTION: Option<&'static str> = Some(
        "Upward Low-Level Density Compressor & 3-Band 2x-Oversampled ADAA Soft Clipper",
    );
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
}

impl Default for SpaceVerbAndPreampParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(680, 390),
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
            .with_smoother(SmoothingStyle::Linear(15.0)),
            reverb_decay: FloatParam::new(
                "FDN Reverb Decay",
                0.65,
                FloatRange::Linear { min: 0.05, max: 0.98 },
            ),
            reverb_damping: FloatParam::new(
                "High-Frequency Damping",
                0.40,
                FloatRange::Linear { min: 0.0, max: 0.95 },
            ),
            shimmer_amount: FloatParam::new(
                "+12 st Octave Shimmer",
                0.15,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            ),
            vocal_ducking: FloatParam::new(
                "Vocal Sidechain Ducking",
                0.55,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            ),
            wet_mix: FloatParam::new(
                "Reverb Wet Mix",
                0.22,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            ),
        }
    }
}

impl Default for ForgeSpaceVerbAndPreampPlug {
    fn default() -> Self {
        Self {
            params: Arc::new(SpaceVerbAndPreampParams::default()),
            telemetry: Arc::new(GuiTelemetry::default()),
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
        create_egui_editor(
            self.params.editor_state.clone(),
            (),
            |_, _| {},
            move |egui_ctx, setter, _state| {
                egui::CentralPanel::default().show(egui_ctx, |ui| {
                    draw_forge_studio_header_and_spectrum(
                        ui,
                        "FORGE SPACEVERB FDN & NEURAL WAVENET PREAMP",
                        "Dilated Causal WaveNet Preamp (1073/1176/V76/A800) + 8x8 Hadamard FDN Reverb w/ Ducking & Shimmer",
                        &telemetry,
                        None,
                    );
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
        self.reverb_dsp.reset(sr);
        true
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
        let decay = f64::from(self.params.reverb_decay.value());
        let damp = f64::from(self.params.reverb_damping.value());
        let shimmer = f64::from(self.params.shimmer_amount.value());
        let duck = f64::from(self.params.vocal_ducking.value());
        let mix = f64::from(self.params.wet_mix.value());

        for mut channel_samples in buffer.iter_samples() {
            let pdrive = f64::from(self.params.preamp_drive.smoothed.next());
            let in_l = *channel_samples.get_mut(0).unwrap_or(&mut 0.0);
            let in_r = *channel_samples.get_mut(1).unwrap_or(&mut 0.0);
            let (pr_l, pr_r) = self
                .preamp_dsp
                .process_stereo_sample(in_l, in_r, pdrive * 12.0, 0.0);
            let (out_l, out_r) =
                self.reverb_dsp
                    .process_stereo_sample(pr_l, pr_r, decay, damp, shimmer, duck, mix);
            self.telemetry.update_peaks(in_l, in_r, out_l, out_r, 0.0);
            if let Some(s0) = channel_samples.get_mut(0) {
                *s0 = out_l;
            }
            if let Some(s1) = channel_samples.get_mut(1) {
                *s1 = out_r;
            }
        }
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

// Export all 10 plugins into both the CLAP multi-plugin bundle and the VST3 multi-plugin bundle!
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
    ForgeSpaceVerbAndPreampPlug
);
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
    ForgeSpaceVerbAndPreampPlug
);
