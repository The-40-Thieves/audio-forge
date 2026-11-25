//! Shared plugin metadata, IO layouts, declarative boilerplate macros, parameter helpers,
//! and zero-allocation audio-block utilities used across all 11 `forge-plug` processors.

use std::num::NonZeroU32;
use std::sync::Arc;

use forge_dsp::RealtimeSpectrumAnalyzer32;
use nih_plug::buffer::ChannelSamples;
use nih_plug::prelude::{
    AudioIOLayout, BufferConfig, Editor, IntParam, IntRange, Param, ParamSetter, new_nonzero_u32,
};
use nih_plug_egui::{EguiState, create_egui_editor, egui, widgets};

use crate::ui::GuiTelemetry;

pub(crate) const FORGE_VENDOR: &str = "The 40 Thieves (ForgeAudio)";
pub(crate) const FORGE_URL: &str = "https://github.com/The-40-Thieves/audio-forge";
pub(crate) const FORGE_EMAIL: &str = "audio@the40thieves.org";

pub(crate) const STEREO_LAYOUTS: &[AudioIOLayout] = &[AudioIOLayout {
    main_input_channels: NonZeroU32::new(2),
    main_output_channels: NonZeroU32::new(2),
    ..AudioIOLayout::const_default()
}];

pub(crate) const STEREO_OR_MONO_LAYOUTS: &[AudioIOLayout] = &[
    STEREO_LAYOUTS[0],
    AudioIOLayout {
        main_input_channels: NonZeroU32::new(1),
        main_output_channels: NonZeroU32::new(1),
        ..AudioIOLayout::const_default()
    },
];

pub(crate) const STEREO_SIDECHAIN_LAYOUTS: &[AudioIOLayout] = &[AudioIOLayout {
    main_input_channels: NonZeroU32::new(2),
    main_output_channels: NonZeroU32::new(2),
    aux_input_ports: &[new_nonzero_u32(2)],
    ..AudioIOLayout::const_default()
}];

pub(crate) const STEREO_SIDECHAIN_OR_STEREO_LAYOUTS: &[AudioIOLayout] =
    &[STEREO_SIDECHAIN_LAYOUTS[0], STEREO_LAYOUTS[0]];

/// Defines compact curated [`crate::ui::FactoryPreset`] slices without repeating multi-line
/// struct boilerplate.
macro_rules! define_presets {
    ($(const $name:ident { $($pname:literal => { $($pid:literal : $pval:expr),* $(,)? };)* })*) => {
        $(
            const $name: &[crate::ui::FactoryPreset] = &[
                $(
                    crate::ui::FactoryPreset {
                        name: $pname,
                        values: &[$(($pid, $pval)),*],
                    },
                )*
            ];
        )*
    };
}

/// Generates the shared `Plugin` associated constants and `params()` implementation.
macro_rules! forge_plugin_common {
    ($name:expr, $layouts:expr $(, midi_in = $midi_in:expr, midi_out = $midi_out:expr)?) => {
        const NAME: &'static str = $name;
        const VENDOR: &'static str = crate::chassis::FORGE_VENDOR;
        const URL: &'static str = crate::chassis::FORGE_URL;
        const EMAIL: &'static str = crate::chassis::FORGE_EMAIL;
        const VERSION: &'static str = env!("CARGO_PKG_VERSION");
        const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = $layouts;
        $(
            const MIDI_INPUT: MidiConfig = $midi_in;
            const MIDI_OUTPUT: MidiConfig = $midi_out;
        )?
        const SAMPLE_ACCURATE_AUTOMATION: bool = true;
        type SysExMessage = ();
        type BackgroundTask = ();

        fn params(&self) -> Arc<dyn Params> {
            self.params.clone()
        }
    };
}

/// Generates both `ClapPlugin` and `Vst3Plugin` trait implementations for a `forge-plug` processor.
macro_rules! impl_forge_formats {
    (
        $plug:ty,
        clap_id: $clap_id:expr,
        clap_desc: $clap_desc:expr,
        clap_features: $clap_features:expr,
        vst3_id: $vst3_id:expr,
        vst3_cats: $vst3_cats:expr $(,)?
    ) => {
        impl ClapPlugin for $plug {
            const CLAP_ID: &'static str = $clap_id;
            const CLAP_DESCRIPTION: Option<&'static str> = Some($clap_desc);
            const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
            const CLAP_SUPPORT_URL: Option<&'static str> = Some(Self::URL);
            const CLAP_FEATURES: &'static [ClapFeature] = $clap_features;
        }

        impl Vst3Plugin for $plug {
            const VST3_CLASS_ID: [u8; 16] = $vst3_id;
            const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = $vst3_cats;
        }
    };
}

/// Generates `Default` for standard 5-field `{ params, telemetry, analyzer, dsp, sample_rate }`
/// plugins.
macro_rules! impl_standard_plugin_default {
    ($plug:ty, $params:ty, $dsp:ty) => {
        impl Default for $plug {
            fn default() -> Self {
                Self {
                    params: Arc::new(<$params>::default()),
                    telemetry: Arc::new(GuiTelemetry::default()),
                    analyzer: RealtimeSpectrumAnalyzer32::new(48_000.0),
                    dsp: <$dsp>::new(48_000.0),
                    sample_rate: 48_000.0,
                }
            }
        }
    };
}

/// Generates standard `initialize` and `reset` methods for 5-field DSP plugins.
#[inline]
pub(crate) fn init_sample_rate(
    sample_rate_slot: &mut f64,
    buffer_config: &BufferConfig,
    analyzer: &mut RealtimeSpectrumAnalyzer32,
) -> f64 {
    let sr = f64::from(buffer_config.sample_rate);
    *sample_rate_slot = sr;
    analyzer.reset(sr);
    sr
}

macro_rules! impl_standard_plugin_lifecycle {
    (@reset) => {
        fn reset(&mut self) {
            self.dsp.reset(self.sample_rate);
            self.analyzer.clear_state();
        }
    };
    () => {
        fn initialize(
            &mut self,
            _audio_io_layout: &AudioIOLayout,
            buffer_config: &BufferConfig,
            _context: &mut impl InitContext<Self>,
        ) -> bool {
            let sr = crate::chassis::init_sample_rate(
                &mut self.sample_rate,
                buffer_config,
                &mut self.analyzer,
            );
            self.dsp.reset(sr);
            true
        }
        impl_standard_plugin_lifecycle!(@reset);
    };
    (latency) => {
        fn initialize(
            &mut self,
            _audio_io_layout: &AudioIOLayout,
            buffer_config: &BufferConfig,
            context: &mut impl InitContext<Self>,
        ) -> bool {
            let sr = crate::chassis::init_sample_rate(
                &mut self.sample_rate,
                buffer_config,
                &mut self.analyzer,
            );
            self.dsp.reset(sr);
            context.set_latency_samples(self.dsp.latency_samples());
            true
        }
        impl_standard_plugin_lifecycle!(@reset);
    };
    ($self_id:ident, $sr_id:ident => $build_dsp:expr) => {
        fn initialize(
            &mut $self_id,
            _audio_io_layout: &AudioIOLayout,
            buffer_config: &BufferConfig,
            _context: &mut impl InitContext<Self>,
        ) -> bool {
            let $sr_id = crate::chassis::init_sample_rate(
                &mut $self_id.sample_rate,
                buffer_config,
                &mut $self_id.analyzer,
            );
            $self_id.dsp = $build_dsp;
            true
        }
        impl_standard_plugin_lifecycle!(@reset);
    };
}

/// Generates a standard 2-column A/B studio `editor()` method.
macro_rules! forge_ab_editor {
    (
        title: $title:expr,
        subtitle: $sub:expr,
        presets: $presets:expr,
        grid_id: $grid_id:expr,
        params: [$(($pid:literal, $pkind:ident, $pfield:ident, $plabel:literal)),* $(,)?]
    ) => {
        fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
            let params = self.params.clone();
            let telemetry = self.telemetry.clone();
            forge_editor(
                self.params.editor_state.clone(),
                AbBankState::default(),
                move |ui, setter, ab_state| {
                    draw_forge_studio_header_and_spectrum(ui, $title, $sub, &telemetry, None);
                    let param_refs = [$(($pid, ParamRef::$pkind(&params.$pfield))),*];
                    draw_ab_and_presets_bar(ui, setter, &param_refs, ab_state, $presets);
                    param_grid(ui, $grid_id, |ui| {
                        $(param_row(ui, setter, $plabel, &params.$pfield);)*
                    });
                },
            )
        }
    };
    (
        persisted_scroll,
        title: $title:expr,
        subtitle: $sub:expr,
        $(slot: ($sparams:ident => $slot_expr:expr),)?
        $(drag: ($dlabel:expr, $dfield:ident, $dcolor:expr),)?
        presets: $presets:expr,
        grid_id: $grid_id:expr,
        params: [$(($pid:literal, $pkind:ident, $pfield:ident, $plabel:literal)),* $(,)?]
    ) => {
        fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
            let params = self.params.clone();
            let telemetry = self.telemetry.clone();
            let initial_ab = AbBankState::load_from_persisted(&self.params.ab_state);
            forge_editor(
                self.params.editor_state.clone(),
                initial_ab,
                move |ui, setter, ab_state| {
                    let slot = forge_ab_editor!(@slot params $(, $sparams => $slot_expr)?);
                    let (drag_setter, drag) =
                        forge_ab_editor!(@drag params, setter $(, $dlabel, $dfield, $dcolor)?);
                    draw_forge_studio_header_and_spectrum_interactive(
                        ui,
                        $title,
                        $sub,
                        &telemetry,
                        slot,
                        drag_setter,
                        drag,
                    );
                    let param_refs = [$(($pid, ParamRef::$pkind(&params.$pfield))),*];
                    draw_ab_and_presets_bar_persisted(
                        ui,
                        setter,
                        &param_refs,
                        ab_state,
                        $presets,
                        Some(&params.ab_state),
                    );
                    scroll_param_grid(ui, $grid_id, |ui| {
                        $(param_row(ui, setter, $plabel, &params.$pfield);)*
                    });
                },
            )
        }
    };
    (@slot $params:ident) => {
        None
    };
    (@slot $params:ident, $sparams:ident => $slot_expr:expr) => {{
        let $sparams = &$params;
        $slot_expr
    }};
    (@drag $params:ident, $setter:ident) => {
        (None, None)
    };
    (@drag $params:ident, $setter:ident, $dlabel:expr, $dfield:ident, $dcolor:expr) => {
        (
            Some($setter),
            Some(CanvasDragHandle {
                label: $dlabel,
                param: &$params.$dfield,
                color: $dcolor,
            }),
        )
    };
}

/// Builds an enumerated `IntParam` with display labels and case-insensitive aliases.
pub(crate) fn enum_int_param(
    name: &'static str,
    default: i32,
    labels: &'static [&'static str],
    aliases: &'static [(&'static str, i32)],
) -> IntParam {
    let max_idx = (labels.len() as i32).saturating_sub(1);
    IntParam::new(
        name,
        default,
        IntRange::Linear {
            min: 0,
            max: max_idx,
        },
    )
    .with_value_to_string(Arc::new(move |v| {
        labels
            .get(v.clamp(0, max_idx) as usize)
            .copied()
            .unwrap_or("")
            .to_string()
    }))
    .with_string_to_value(Arc::new(move |s| {
        let clean = s.trim();
        if let Ok(n) = clean.parse::<i32>()
            && (0..=max_idx).contains(&n)
        {
            return Some(n);
        }
        for (idx, label) in labels.iter().enumerate() {
            if label.eq_ignore_ascii_case(clean) {
                return Some(idx as i32);
            }
        }
        for &(alias, val) in aliases {
            if alias.eq_ignore_ascii_case(clean) {
                return Some(val);
            }
        }
        None
    }))
}

/// Creates a `nih_plug_egui` studio editor inside a `CentralPanel`.
pub(crate) fn forge_editor<S: Send + Sync + 'static>(
    editor_state: Arc<EguiState>,
    initial_state: S,
    draw: impl Fn(&mut egui::Ui, &ParamSetter, &mut S) + Send + Sync + 'static,
) -> Option<Box<dyn Editor>> {
    create_egui_editor(
        editor_state,
        initial_state,
        |_, _| {},
        move |egui_ctx, setter, state| {
            egui::CentralPanel::default().show(egui_ctx, |ui| {
                draw(ui, setter, state);
            });
        },
    )
}

/// Renders a labeled 2-column studio parameter slider grid.
pub(crate) fn param_grid(ui: &mut egui::Ui, id_source: &str, add_rows: impl FnOnce(&mut egui::Ui)) {
    egui::Grid::new(id_source)
        .num_columns(2)
        .spacing([16.0, 8.0])
        .show(ui, add_rows);
}

/// Renders a scrollable labeled 2-column studio parameter slider grid.
pub(crate) fn scroll_param_grid(
    ui: &mut egui::Ui,
    id_source: &str,
    add_rows: impl FnOnce(&mut egui::Ui),
) {
    egui::ScrollArea::vertical()
        .auto_shrink([false; 2])
        .show(ui, |ui| {
            param_grid(ui, id_source, add_rows);
        });
}

/// Renders one label + `ParamSlider` row inside an `egui::Grid`.
pub(crate) fn param_row<P: Param>(ui: &mut egui::Ui, setter: &ParamSetter, label: &str, param: &P) {
    ui.label(label);
    ui.add(widgets::ParamSlider::for_param(param, setter));
    ui.end_row();
}

/// Reads stereo samples `(in_l, in_r)` from a `nih_plug` channel frame without allocating.
#[inline]
pub(crate) fn read_stereo(channel_samples: &mut ChannelSamples<'_, '_>) -> (f32, f32) {
    let in_l = *channel_samples.get_mut(0).unwrap_or(&mut 0.0);
    let in_r = *channel_samples.get_mut(1).unwrap_or(&mut 0.0);
    (in_l, in_r)
}

/// Reads stereo samples or duplicates mono left into right when the track has only 1 channel.
#[inline]
pub(crate) fn read_stereo_or_mono(channel_samples: &mut ChannelSamples<'_, '_>) -> (f32, f32) {
    let in_l = *channel_samples.get_mut(0).unwrap_or(&mut 0.0);
    let in_r = channel_samples.get_mut(1).map_or(in_l, |s| *s);
    (in_l, in_r)
}

/// Writes stereo samples `(out_l, out_r)` back into a `nih_plug` channel frame without allocating.
#[inline]
pub(crate) fn write_stereo(channel_samples: &mut ChannelSamples<'_, '_>, out_l: f32, out_r: f32) {
    if let Some(s0) = channel_samples.get_mut(0) {
        *s0 = out_l;
    }
    if let Some(s1) = channel_samples.get_mut(1) {
        *s1 = out_r;
    }
}

/// Estimates instantaneous stereo attenuation in negative dB (`<= 0.0`) from I/O peak ratio.
#[inline]
pub(crate) fn estimate_gr_db(in_l: f32, in_r: f32, out_l: f32, out_r: f32) -> f32 {
    20.0 * (out_l.abs().max(out_r.abs()) / in_l.abs().max(in_r.abs()).max(1e-5))
        .clamp(0.05, 1.0)
        .log10()
}

/// Updates the analyzer, peak meters, and output frame for one processed stereo sample.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_stereo_sample(
    analyzer: &mut RealtimeSpectrumAnalyzer32,
    telemetry: &GuiTelemetry,
    channel_samples: &mut ChannelSamples<'_, '_>,
    in_l: f32,
    in_r: f32,
    out_l: f32,
    out_r: f32,
    gr_db: f32,
) {
    analyzer.push_stereo_sample(out_l, out_r);
    telemetry.update_peaks(in_l, in_r, out_l, out_r, gr_db);
    write_stereo(channel_samples, out_l, out_r);
}

/// Publishes the 32-band analyzer snapshot with flat (`0 dB`) per-band gain reduction.
#[inline]
pub(crate) fn finish_block_spectrum(
    analyzer: &RealtimeSpectrumAnalyzer32,
    telemetry: &GuiTelemetry,
) {
    telemetry.update_spectrum_and_gr(&analyzer.snapshot_db(), &[0.0; 32]);
}
