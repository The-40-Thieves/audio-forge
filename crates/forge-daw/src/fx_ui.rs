//! Parameter editors for every native ForgeAudio FX slot in ForgeStudio DAW.

use eframe::egui;
use forge_engine::FxSlotConfig;

pub(crate) fn draw_fx_slot_controls(ui: &mut egui::Ui, fx: &mut FxSlotConfig) -> bool {
    let mut changed = false;
    match fx {
        FxSlotConfig::GainTrim { gain_db, .. } => {
            changed |= ui
                .add(egui::Slider::new(gain_db, -24.0..=24.0).text("Gain (dB)"))
                .changed();
        }
        FxSlotConfig::OpenSoothe {
            sensitivity_db,
            max_depth_db,
            low_latency,
            ..
        } => {
            changed |= ui
                .add(egui::Slider::new(sensitivity_db, 0.5..=12.0).text("Sens (dB)"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(max_depth_db, 0.5..=18.0).text("Depth (dB)"))
                .changed();
            changed |= ui.checkbox(low_latency, "Low-Latency (5.3 ms)").changed();
        }
        FxSlotConfig::VocalSurgical {
            declick,
            deplosive,
            deess_thresh_db,
            ..
        } => {
            ui.horizontal(|ui| {
                changed |= ui.checkbox(declick, "De-Click").changed();
                changed |= ui.checkbox(deplosive, "De-Plosive").changed();
            });
            changed |= ui
                .add(egui::Slider::new(deess_thresh_db, -36.0..=-4.0).text("De-Ess (dB)"))
                .changed();
        }
        FxSlotConfig::SubMonoTape {
            mono_cutoff_hz,
            tape_drive,
            ..
        } => {
            changed |= ui
                .add(egui::Slider::new(mono_cutoff_hz, 60.0..=250.0).text("Mono Hz"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(tape_drive, 0.0..=1.0).text("Tape Drive"))
                .changed();
        }
        FxSlotConfig::MasterLimiter {
            drive_db,
            ceiling_dbtp,
            isp_8x,
            ..
        } => {
            changed |= ui
                .add(egui::Slider::new(drive_db, -6.0..=18.0).text("Drive (dB)"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(ceiling_dbtp, -6.0..=-0.1).text("Ceil (dBTP)"))
                .changed();
            changed |= ui.checkbox(isp_8x, "8x ISP Oversampling").changed();
        }
        FxSlotConfig::SplitEq {
            tr_low_db,
            tr_mid_db,
            tr_high_db,
            tn_low_db,
            tn_mid_db,
            tn_high_db,
            ..
        } => {
            changed |= ui
                .add(egui::Slider::new(tr_low_db, -12.0..=12.0).text("Trans Low"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(tr_mid_db, -12.0..=12.0).text("Trans Mid"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(tr_high_db, -12.0..=12.0).text("Trans High"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(tn_low_db, -12.0..=12.0).text("Tonal Low"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(tn_mid_db, -12.0..=12.0).text("Tonal Mid"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(tn_high_db, -12.0..=12.0).text("Tonal High"))
                .changed();
        }
        FxSlotConfig::DensityClipper {
            density,
            clip_drive_db,
            ceiling_db,
            ..
        } => {
            changed |= ui
                .add(egui::Slider::new(density, 0.0..=1.0).text("Density"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(clip_drive_db, 0.0..=12.0).text("4x ADAA Clip"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(ceiling_db, -12.0..=0.0).text("Ceil (dB)"))
                .changed();
        }
        FxSlotConfig::SpaceVerb {
            decay_rt60_secs,
            damping,
            shimmer,
            ducking,
            wet_mix,
            ..
        } => {
            changed |= ui
                .add(egui::Slider::new(decay_rt60_secs, 0.2..=12.0).text("Decay (s)"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(damping, 0.05..=0.90).text("Damping"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(shimmer, 0.0..=1.0).text("Shimmer"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(ducking, 0.0..=1.0).text("Ducking"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(wet_mix, 0.0..=1.0).text("Wet Mix"))
                .changed();
        }
        FxSlotConfig::VocalTune {
            key_root,
            scale_mode,
            retune_speed_ms,
            retune_amount,
            humanize,
            formant_shift_st,
            wet_mix,
            ..
        } => {
            const ROOTS: [&str; 12] = [
                "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
            ];
            const SCALES: [&str; 5] = [
                "Chromatic",
                "Major",
                "Minor",
                "Pentatonic Maj",
                "Pentatonic Min",
            ];
            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt("vt_root")
                    .width(46.0)
                    .selected_text(ROOTS[usize::from(*key_root % 12)])
                    .show_ui(ui, |ui| {
                        for (i, name) in ROOTS.iter().enumerate() {
                            if ui.selectable_value(key_root, i as u8, *name).changed() {
                                changed = true;
                            }
                        }
                    });
                egui::ComboBox::from_id_salt("vt_scale")
                    .width(108.0)
                    .selected_text(SCALES[usize::from((*scale_mode).min(4))])
                    .show_ui(ui, |ui| {
                        for (i, name) in SCALES.iter().enumerate() {
                            if ui.selectable_value(scale_mode, i as u8, *name).changed() {
                                changed = true;
                            }
                        }
                    });
            });
            changed |= ui
                .add(egui::Slider::new(retune_speed_ms, 0.0..=100.0).text("Speed (ms)"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(retune_amount, 0.0..=1.0).text("Amount"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(humanize, 0.0..=1.0).text("Humanize"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(formant_shift_st, -12.0..=12.0).text("Formant (st)"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(wet_mix, 0.0..=1.0).text("Wet Mix"))
                .changed();
        }
    }
    changed
}
