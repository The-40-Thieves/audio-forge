//! Parameter editors for every native ForgeAudio FX slot in ForgeStudio DAW.

use eframe::egui;
use forge_engine::FxSlotConfig;

fn slider(
    ui: &mut egui::Ui,
    val: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    text: &str,
) -> bool {
    ui.add(egui::Slider::new(val, range).text(text)).changed()
}

pub(crate) fn draw_fx_slot_controls(
    ui: &mut egui::Ui,
    fx: &mut FxSlotConfig,
    sidechain_tracks: &[(u64, String)],
) -> bool {
    let mut changed = false;
    match fx {
        FxSlotConfig::GainTrim { gain_db, .. } => {
            changed |= slider(ui, gain_db, -24.0..=24.0, "Gain (dB)");
        }
        FxSlotConfig::OpenSoothe {
            sensitivity_db,
            max_depth_db,
            sharpness_bins,
            low_latency,
            tilt_db_per_oct,
            ultrasonic_lpf_hz,
            mid_side_balance,
            stereo_link,
            hard_mode,
            ..
        } => {
            changed |= slider(ui, sensitivity_db, 0.5..=12.0, "Sens (dB)");
            changed |= slider(ui, max_depth_db, 0.5..=18.0, "Depth (dB)");
            changed |= ui
                .add(egui::Slider::new(sharpness_bins, 2..=24).text("Sharp (bins)"))
                .changed();
            changed |= slider(ui, tilt_db_per_oct, -6.0..=6.0, "Tilt (dB/oct)");
            changed |= slider(ui, ultrasonic_lpf_hz, 0.0..=22_000.0, "Ultra LPF (0=Off)");
            changed |= slider(ui, mid_side_balance, -1.0..=1.0, "M/S Focus");
            changed |= slider(ui, stereo_link, 0.0..=1.0, "Stereo Link");
            ui.horizontal(|ui| {
                changed |= ui.checkbox(low_latency, "Low-Lat (5.3ms)").changed();
                changed |= ui.checkbox(hard_mode, "Hard Mode").changed();
            });
        }
        FxSlotConfig::VocalSurgical {
            declick,
            deplosive,
            deess_thresh_db,
            gate_thresh_db,
            ..
        } => {
            ui.horizontal(|ui| {
                changed |= ui.checkbox(declick, "De-Click").changed();
                changed |= ui.checkbox(deplosive, "De-Plosive").changed();
            });
            changed |= ui
                .add(egui::Slider::new(deess_thresh_db, -36.0..=-4.0).text("De-Ess (dB)"))
                .changed();
            changed |= slider(ui, gate_thresh_db, -80.0..=-12.0, "Gate (dB)");
        }
        FxSlotConfig::SubMonoTape {
            mono_cutoff_hz,
            sub_reinforce,
            stereo_width,
            tape_drive,
            ..
        } => {
            changed |= ui
                .add(egui::Slider::new(mono_cutoff_hz, 60.0..=250.0).text("Mono Hz"))
                .changed();
            changed |= slider(ui, sub_reinforce, 0.0..=1.0, "Sub Boost");
            changed |= slider(ui, stereo_width, 0.0..=2.0, "Width");
            changed |= ui
                .add(egui::Slider::new(tape_drive, 0.0..=1.0).text("Tape Drive"))
                .changed();
        }
        FxSlotConfig::MasterLimiter {
            drive_db,
            ceiling_dbtp,
            release_ms,
            isp_8x,
            ..
        } => {
            changed |= ui
                .add(egui::Slider::new(drive_db, -6.0..=18.0).text("Drive (dB)"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(ceiling_dbtp, -6.0..=-0.1).text("Ceil (dBTP)"))
                .changed();
            changed |= slider(ui, release_ms, 10.0..=500.0, "Release (ms)");
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
                "Pentatonic Min",
                "Pentatonic Maj",
            ];
            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt(ui.scope_id().with("vt_root"))
                    .width(46.0)
                    .selected_text(ROOTS[usize::from(*key_root % 12)])
                    .show_ui(ui, |ui| {
                        for (i, name) in ROOTS.iter().enumerate() {
                            if ui.selectable_value(key_root, i as u8, *name).changed() {
                                changed = true;
                            }
                        }
                    });
                egui::ComboBox::from_id_salt(ui.scope_id().with("vt_scale"))
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
            changed |= slider(ui, wet_mix, 0.0..=1.0, "Wet Mix");
        }
        FxSlotConfig::OpenSpacer {
            amount,
            max_cut_db,
            low_hz,
            high_hz,
            mid_side_balance,
            delta_listen,
            sidechain_track_id,
            ..
        } => {
            let selected_label = sidechain_track_id
                .and_then(|id| {
                    sidechain_tracks
                        .iter()
                        .find(|(tid, _)| *tid == id)
                        .map(|(tid, name)| format!("#{tid}: {name}"))
                        .or_else(|| Some(format!("#{id} (Missing)")))
                })
                .unwrap_or_else(|| "Off (Bypassed)".to_string());
            ui.horizontal(|ui| {
                ui.label("Sidechain:");
                egui::ComboBox::from_id_salt(ui.scope_id().with("spacer_sc"))
                    .width(150.0)
                    .selected_text(selected_label)
                    .show_ui(ui, |ui| {
                        if ui
                            .selectable_value(sidechain_track_id, None, "Off (Bypassed)")
                            .changed()
                        {
                            changed = true;
                        }
                        for (tid, tname) in sidechain_tracks {
                            if ui
                                .selectable_value(
                                    sidechain_track_id,
                                    Some(*tid),
                                    format!("#{tid}: {tname}"),
                                )
                                .changed()
                            {
                                changed = true;
                            }
                        }
                    });
            });
            changed |= slider(ui, amount, 0.0..=1.0, "Unmask Amt");
            changed |= slider(ui, max_cut_db, 0.5..=24.0, "Max Cut (dB)");
            changed |= slider(ui, low_hz, 20.0..=4_000.0, "Low Hz");
            changed |= slider(ui, high_hz, 1_000.0..=20_000.0, "High Hz");
            changed |= slider(ui, mid_side_balance, -1.0..=1.0, "M/S Focus");
            changed |= ui.checkbox(delta_listen, "Delta Listen").changed();
        }
        FxSlotConfig::PhaseMatchEq {
            delay_samples,
            invert_polarity,
            low_shelf_db,
            mid_dip_db,
            air_shelf_db,
            ..
        } => {
            changed |= ui
                .add(egui::Slider::new(delay_samples, 0..=4096).text("Delay (smp)"))
                .changed();
            changed |= ui.checkbox(invert_polarity, "Invert Polarity").changed();
            changed |= slider(ui, low_shelf_db, -12.0..=12.0, "Low Shelf (dB)");
            changed |= slider(ui, mid_dip_db, -12.0..=12.0, "Mid Dip (dB)");
            changed |= slider(ui, air_shelf_db, -12.0..=12.0, "Air Shelf (dB)");
        }
        FxSlotConfig::NeuralPreamp {
            model,
            input_drive_db,
            output_trim_db,
            ..
        } => {
            const MODELS: [&str; 4] = ["Neve 1073", "FET 1176", "Tube V76", "Studer A800"];
            egui::ComboBox::from_id_salt(ui.scope_id().with("nam_model"))
                .width(120.0)
                .selected_text(MODELS[usize::from((*model).min(3))])
                .show_ui(ui, |ui| {
                    for (i, label) in MODELS.iter().enumerate() {
                        changed |= ui.selectable_value(model, i as u8, *label).changed();
                    }
                });
            changed |= slider(ui, input_drive_db, -12.0..=24.0, "Drive (dB)");
            changed |= slider(ui, output_trim_db, -24.0..=12.0, "Trim (dB)");
        }
        FxSlotConfig::DrumTrigger {
            threshold_db, mix, ..
        } => {
            changed |= slider(ui, threshold_db, -60.0..=0.0, "Thresh (dB)");
            changed |= slider(ui, mix, 0.0..=1.0, "Punch Mix");
        }
    }
    changed
}
