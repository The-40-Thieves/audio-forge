//! # `forge-plug` — Native Rust VST3 & CLAP Real-Time Studio Plugin Suite
#![allow(semicolon_in_expressions_from_non_local_macros)]
//!
//! Built on `nih-plug` and `forge-dsp::realtime` with **zero audio-thread heap allocations**
//! (`assert_process_allocs` enabled).
//!
//! Implements 11 VST3/CLAP studio processors (exported individually from `plugins/forge-*`,
//! or together as one bundle when `all-in-one-export` is enabled):
//! 1. **Forge OpenSoothe De-Fizz** (2048-pt OLA STFT Dynamic Spectral Resonance & AI Fizz Suppressor)
//! 2. **Forge OpenSpacer Unmask** (4-Channel Sidechain Dynamic Spectral Unmasker)
//! 3. **Forge Vocal Surgical Strip** (Zero-Latency Mouth De-Click, Plosive Suppressor, De-Esser & Breath Gate)
//! 4. **Forge SubMono & Console Tape** (Linkwitz-Riley `<160 Hz` Mono-Maker, Sub Reinforce & Console8/ChowTape Warmth)
//! 5. **Forge Master True-Peak Limiter** (4x Inter-Sample True-Peak `-1.0 dBTP` Lookahead Brickwall Limiter)
//! 6. **Forge Drum Trigger & Theory MIDI** (Real-Time Audio-to-MIDI Drum Replacer `C1/D1/F#1` + Neo-Soul Chord Voicer)
//! 7. **Forge Phase & Match EQ** (Phase/Polarity Aligner + 18-Band Mastering EQ & Genre Tonal Balance)
//! 8. **Forge SplitEQ** (3-Band Structural Transient/Tonal Equalizer & Shaper)
//! 9. **Forge Density & Clipper** (3-Band Upward Density Compressor & 4x ADAA Soft Clipper)
//! 10. **Forge SpaceVerb & Preamp** (8x8 Hadamard FDN Reverb with Ducking/Shimmer + Neural WaveNet Preamp)
//! 11. **Forge Vocal Tune** (Low-Latency FFT-YIN F0 + Scale-Locked Dual-Grain PSOLA Pitch Corrector)

#[macro_use]
mod chassis;
mod elite_fx;
mod mastering;
mod soothe_spacer;
mod ui;
mod vocal_drums;

pub use elite_fx::*;
pub use mastering::*;
pub use soothe_spacer::*;
pub use vocal_drums::*;

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
use nih_plug_egui::{EguiState, egui};

pub use ui::{
    AbBankState, FactoryPreset, GuiTelemetry, OutputLoudnessMeter, ParamRef, UserPreset,
    export_user_preset_json, import_user_preset_json,
};
use ui::{
    CanvasDragHandle, draw_ab_and_presets_bar, draw_ab_and_presets_bar_persisted,
    draw_forge_studio_header_and_spectrum, draw_forge_studio_header_and_spectrum_interactive,
};

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
    use crate::elite_fx::{
        REVERB_DECAY_KNOB_MAX, REVERB_DECAY_KNOB_MIN, string_to_reverb_decay_knob,
    };
    use crate::soothe_spacer::{bus_slot_from_string, bus_slot_to_string};
    use crate::vocal_drums::{
        VocalSurgicalControls, VocalSurgicalParams, next_vocal_surgical_controls,
    };
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

        #[test]
        fn prop_gui_ai_telemetry_self_heals_from_nan_inf(
            corr in prop::num::f32::ANY,
            mid_db in prop::num::f32::ANY,
            side_db in prop::num::f32::ANY,
        ) {
            let tel = GuiTelemetry::default();
            tel.high_mid_corr.store(f32::NAN.to_bits(), Ordering::Relaxed);
            tel.mid_fizz_db.store(f32::INFINITY.to_bits(), Ordering::Relaxed);
            tel.side_fizz_db.store(f32::NEG_INFINITY.to_bits(), Ordering::Relaxed);

            tel.update_ai_telemetry(corr, mid_db, side_db);

            let c = f32::from_bits(tel.high_mid_corr.load(Ordering::Relaxed));
            let m = f32::from_bits(tel.mid_fizz_db.load(Ordering::Relaxed));
            let s = f32::from_bits(tel.side_fizz_db.load(Ordering::Relaxed));

            prop_assert!(c.is_finite() && (-1.0..=1.0).contains(&c));
            prop_assert!(m.is_finite() && (-90.0..=12.0).contains(&m));
            prop_assert!(s.is_finite() && (-90.0..=12.0).contains(&s));
        }
    }

    #[test]
    fn open_soothe_params_keep_legacy_ids_and_add_ulpf() {
        let plug = ForgeOpenSoothePlug::default();
        let params = plug.params();
        let ids: Vec<String> = params
            .param_map()
            .into_iter()
            .map(|(id, _, _)| id)
            .collect();
        for expected in [
            "sens", "depth", "sharp", "delta", "submono", "tape", "limit", "ceil", "bus_slot",
            "tilt", "slink", "msbal", "lowlat", "atk", "rel", "hard", "foc_lo", "foc_mid",
            "foc_hi", "ulpf",
        ] {
            assert!(
                ids.iter().any(|id| id == expected),
                "missing OpenSoothe param id '{expected}'"
            );
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
        ] {
            assert_eq!(bus_slot_from_string(text), Some(value), "{text:?}");
        }
        for (text, value) in [("d", 4), ("G", 7), ("0", 0), ("5", 5), ("8", 8)] {
            assert_eq!(bus_slot_from_string(text), Some(value), "{text:?}");
        }
        for text in ["", "Slot", "Slot I", "I", "9", "-1", "Slot AB", "on"] {
            assert_eq!(bus_slot_from_string(text), None, "{text:?}");
        }
    }

    struct MockInitContext {
        latency_samples: std::cell::Cell<u32>,
        calls: std::cell::Cell<usize>,
    }

    impl MockInitContext {
        fn new() -> Self {
            Self {
                latency_samples: std::cell::Cell::new(0),
                calls: std::cell::Cell::new(0),
            }
        }
    }

    impl<P: Plugin> InitContext<P> for MockInitContext {
        fn plugin_api(&self) -> PluginApi {
            PluginApi::Clap
        }

        fn execute(&self, _task: P::BackgroundTask) {}

        fn set_latency_samples(&self, samples: u32) {
            self.latency_samples.set(samples);
            self.calls.set(self.calls.get() + 1);
        }

        fn set_current_voice_capacity(&self, _capacity: u32) {}
    }

    fn verify_plugin_lifecycle<P: Plugin + Default>() -> (u32, usize) {
        let mut plug = P::default();
        let layout = P::AUDIO_IO_LAYOUTS[0];
        let cfg = BufferConfig {
            sample_rate: 48_000.0,
            min_buffer_size: Some(64),
            max_buffer_size: 1024,
            process_mode: ProcessMode::Realtime,
        };
        let mut ctx = MockInitContext::new();
        assert!(plug.initialize(&layout, &cfg, &mut ctx));
        plug.reset();
        // Verify re-initializing at 96 kHz + resetting succeeds without panic.
        let cfg_96k = BufferConfig {
            sample_rate: 96_000.0,
            ..cfg
        };
        assert!(plug.initialize(&layout, &cfg_96k, &mut ctx));
        plug.reset();
        (ctx.latency_samples.get(), ctx.calls.get())
    }

    #[test]
    fn all_11_plugins_initialize_and_reset_cleanly() {
        let (soothe_lat, soothe_calls) = verify_plugin_lifecycle::<ForgeOpenSoothePlug>();
        assert!(soothe_calls >= 2 && soothe_lat > 0);
        let (spacer_lat, _) = verify_plugin_lifecycle::<ForgeOpenSpacerPlug>();
        assert!(spacer_lat > 0);
        verify_plugin_lifecycle::<ForgeVocalSurgicalPlug>();
        verify_plugin_lifecycle::<ForgeSubAndTapePlug>();
        let (lim_lat, _) = verify_plugin_lifecycle::<ForgeMasterLimiterPlug>();
        assert!(lim_lat > 0);
        verify_plugin_lifecycle::<ForgeDrumTriggerPlug>();
        verify_plugin_lifecycle::<ForgePhaseAndMatchPlug>();
        verify_plugin_lifecycle::<ForgeSplitEqPlug>();
        verify_plugin_lifecycle::<ForgeDensityClipperPlug>();
        verify_plugin_lifecycle::<ForgeSpaceVerbAndPreampPlug>();
        let (tune_lat, _) = verify_plugin_lifecycle::<ForgeVocalTunePlug>();
        assert!(tune_lat > 0);
    }

    #[test]
    fn spaceverb_preamp_loads_persisted_wavenet_nam_json_across_reinitialize() {
        let mut plug = ForgeSpaceVerbAndPreampPlug::default();
        let wavenet_json = serde_json::json!({
            "version": "0.5.4",
            "architecture": "WaveNet",
            "config": {
                "layers": [{
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": 1,
                    "channels": 1,
                    "kernel_size": 2,
                    "dilations": [1, 2],
                    "activation": "Tanh",
                    "gated": false,
                    "head_bias": true
                }],
                "head_scale": 0.5
            },
            "weights": vec![0.25_f64; 18]
        })
        .to_string();

        plug.load_nam_profile_json(&wavenet_json)
            .expect("valid WaveNet .nam JSON");
        assert!(plug.is_nam_wavenet_loaded());

        let layout = ForgeSpaceVerbAndPreampPlug::AUDIO_IO_LAYOUTS[0];
        let cfg = BufferConfig {
            sample_rate: 48_000.0,
            min_buffer_size: Some(64),
            max_buffer_size: 256,
            process_mode: ProcessMode::Realtime,
        };
        let mut ctx = MockInitContext::new();
        assert!(plug.initialize(&layout, &cfg, &mut ctx));
        assert!(plug.is_nam_wavenet_loaded());

        let mut left: Vec<f32> = (0..128).map(|i| 0.3 * (i as f32 * 0.08).sin()).collect();
        let mut right = left.clone();
        plug.process_slices(&mut [&mut left[..], &mut right[..]]);
        assert!(left.iter().chain(&right).all(|s| s.is_finite()));
        plug.reset();
    }

    #[test]
    fn phase_and_match_auto_aligns_sidechain_delay_and_polarity() {
        let mut plug = ForgePhaseAndMatchPlug::default();
        let layout = ForgePhaseAndMatchPlug::AUDIO_IO_LAYOUTS[0];
        let cfg = BufferConfig {
            sample_rate: 48_000.0,
            min_buffer_size: Some(64),
            max_buffer_size: 2048,
            process_mode: ProcessMode::Realtime,
        };
        let mut ctx = MockInitContext::new();
        assert!(plug.initialize(&layout, &cfg, &mut ctx));

        let ids: Vec<String> = plug
            .params()
            .param_map()
            .into_iter()
            .map(|(id, _, _)| id)
            .collect();
        assert!(ids.iter().any(|id| id == "autoal"), "missing autoal id");

        let n = 2048_usize;
        let delay = 24_usize;
        let mut main_l: Vec<f32> = (0..n)
            .map(|i| 0.45 * (2.0 * std::f32::consts::PI * 220.0 * (i as f32) / 48_000.0).sin())
            .collect();
        let mut main_r = main_l.clone();
        let sc_l: Vec<f32> = (0..n)
            .map(|i| if i >= delay { -main_l[i - delay] } else { 0.0 })
            .collect();
        let sc_r = sc_l.clone();

        plug.process_slices(
            n,
            &mut [&mut main_l[..], &mut main_r[..]],
            Some((&sc_l, &sc_r)),
            true,
        );
        let (detected_delay, detected_invert) = plug.detected_sidechain_alignment();
        assert_eq!(detected_delay, delay);
        assert!(detected_invert, "expected inverted sidechain polarity");
        assert!(main_l.iter().chain(&main_r).all(|s| s.is_finite()));
    }

    #[test]
    fn mono_layout_process_succeeds_for_vocal_surgical_spliteq_and_clipper() {
        let cfg = BufferConfig {
            sample_rate: 48_000.0,
            min_buffer_size: Some(32),
            max_buffer_size: 128,
            process_mode: ProcessMode::Realtime,
        };
        let mut ctx = MockInitContext::new();

        let mut voc = ForgeVocalSurgicalPlug::default();
        assert!(voc.initialize(&ForgeVocalSurgicalPlug::AUDIO_IO_LAYOUTS[1], &cfg, &mut ctx));
        let mut mono = vec![0.25_f32; 64];
        voc.process_slices(&mut [&mut mono[..]]);
        assert!(mono.iter().all(|s| s.is_finite()));

        let mut split = ForgeSplitEqPlug::default();
        assert!(split.initialize(&ForgeSplitEqPlug::AUDIO_IO_LAYOUTS[1], &cfg, &mut ctx));
        split.process_slices(&mut [&mut mono[..]]);
        assert!(mono.iter().all(|s| s.is_finite()));

        let mut clip = ForgeDensityClipperPlug::default();
        assert!(clip.initialize(
            &ForgeDensityClipperPlug::AUDIO_IO_LAYOUTS[1],
            &cfg,
            &mut ctx
        ));
        clip.process_slices(&mut [&mut mono[..]]);
        assert!(mono.iter().all(|s| s.is_finite()));
    }
}
