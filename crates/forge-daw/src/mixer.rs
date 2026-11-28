//! Bottom Mixing Console dock with channel strips, Aux sends, Master strip, and 32-band RTA.

use eframe::egui;
use forge_engine::{LiveAudioEngine, TrackSend};

use crate::ForgeDawApp;

impl ForgeDawApp {
    pub(crate) fn draw_mixer_dock(&mut self, ui: &mut egui::Ui) {
        if !self.show_mixer {
            return;
        }
        let meters = self
            .audio
            .as_ref()
            .map(LiveAudioEngine::meters_snapshot)
            .unwrap_or_default();

        egui::Panel::bottom("mixer_dock")
            .min_size(175.0)
            .default_size(190.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("MIXER & MASTER SPECTRUM")
                            .strong()
                            .color(egui::Color32::from_rgb(56, 189, 248)),
                    );
                    ui.separator();
                    ui.label(
                        egui::RichText::new(format!(
                            "Master L {:+5.1} dB | R {:+5.1} dB | PDC: {} smp | GPU: {}",
                            meters.master_peak_l_db,
                            meters.master_peak_r_db,
                            meters.total_pdc_samples,
                            self.gpu_backend_label
                        ))
                        .monospace()
                        .small()
                        .color(egui::Color32::from_rgb(120, 240, 170)),
                    );
                });
                ui.separator();

                let prev_snapshot = self.project.clone();
                let mut changed = false;

                ui.horizontal(|ui| {
                    egui::ScrollArea::horizontal()
                        .max_width((ui.available_width() - 320.0).max(200.0))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                let aux_meta: Vec<(u64, String)> = self
                                    .project
                                    .aux_buses
                                    .iter()
                                    .enumerate()
                                    .map(|(i, b)| (b.id, format!("Aux{}", i + 1)))
                                    .collect();
                                for (ti, tr) in self.project.tracks.iter_mut().enumerate() {
                                    ui.group(|ui| {
                                        ui.set_width(120.0);
                                        let col = egui::Color32::from_rgb(
                                            tr.color_rgb[0],
                                            tr.color_rgb[1],
                                            tr.color_rgb[2],
                                        );
                                        ui.label(egui::RichText::new(&tr.name).small().color(col));
                                        ui.horizontal(|ui| {
                                            if ui.selectable_label(tr.mute, "M").clicked() {
                                                tr.mute = !tr.mute;
                                                changed = true;
                                            }
                                            if ui.selectable_label(tr.solo, "S").clicked() {
                                                tr.solo = !tr.solo;
                                                changed = true;
                                            }
                                            let pk = meters
                                                .track_peaks_db
                                                .get(ti)
                                                .copied()
                                                .unwrap_or(-90.0);
                                            ui.label(
                                                egui::RichText::new(format!("{:4.0}", pk))
                                                    .monospace()
                                                    .small(),
                                            );
                                        });
                                        changed |= ui
                                            .add(
                                                egui::Slider::new(&mut tr.gain_db, -48.0..=12.0)
                                                    .text("dB"),
                                            )
                                            .changed();
                                        changed |= ui
                                            .add(
                                                egui::Slider::new(&mut tr.pan, -1.0..=1.0)
                                                    .text("Pan"),
                                            )
                                            .changed();
                                        for (aux_id, aux_lbl) in &aux_meta {
                                            let mut send_db = tr
                                                .sends
                                                .iter()
                                                .find(|s| s.bus_id == *aux_id)
                                                .map_or(-60.0, |s| s.gain_db);
                                            if ui
                                                .add(
                                                    egui::Slider::new(&mut send_db, -60.0..=6.0)
                                                        .text(aux_lbl.as_str()),
                                                )
                                                .changed()
                                            {
                                                if let Some(s) = tr
                                                    .sends
                                                    .iter_mut()
                                                    .find(|s| s.bus_id == *aux_id)
                                                {
                                                    s.gain_db = send_db;
                                                } else {
                                                    tr.sends.push(TrackSend {
                                                        bus_id: *aux_id,
                                                        gain_db: send_db,
                                                        pre_fader: false,
                                                        mute: false,
                                                    });
                                                }
                                                changed = true;
                                            }
                                        }
                                    });
                                }

                                // Aux / FX Return Bus Strips
                                for bus in &mut self.project.aux_buses {
                                    ui.group(|ui| {
                                        ui.set_width(114.0);
                                        ui.label(
                                            egui::RichText::new(&bus.name)
                                                .small()
                                                .strong()
                                                .color(egui::Color32::from_rgb(192, 132, 252)),
                                        );
                                        ui.horizontal(|ui| {
                                            if ui.selectable_label(bus.mute, "M").clicked() {
                                                bus.mute = !bus.mute;
                                                changed = true;
                                            }
                                            ui.label(
                                                egui::RichText::new(format!(
                                                    "FX:{}",
                                                    bus.inserts.len()
                                                ))
                                                .monospace()
                                                .small(),
                                            );
                                        });
                                        changed |= ui
                                            .add(
                                                egui::Slider::new(&mut bus.gain_db, -48.0..=12.0)
                                                    .text("Ret"),
                                            )
                                            .changed();
                                        changed |= ui
                                            .add(
                                                egui::Slider::new(&mut bus.pan, -1.0..=1.0)
                                                    .text("Pan"),
                                            )
                                            .changed();
                                    });
                                }
                            });
                        });

                    ui.separator();
                    // Master Bus 32-Band Real-Time Spectrum
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("MASTER").strong());
                            changed |= ui
                                .add(
                                    egui::Slider::new(
                                        &mut self.project.master_gain_db,
                                        -24.0..=6.0,
                                    )
                                    .text("dB"),
                                )
                                .changed();
                        });
                        let (rect, _) = ui.allocate_exact_size(
                            egui::vec2(ui.available_width().max(260.0), 92.0),
                            egui::Sense::hover(),
                        );
                        let p = ui.painter_at(rect);
                        p.rect_filled(rect, 4.0, egui::Color32::from_rgb(12, 16, 24));
                        let bar_w = (rect.width() - 8.0) / 32.0;
                        for (b, &db) in meters.spectrum_32_db.iter().enumerate() {
                            let norm = ((db + 72.0) / 72.0).clamp(0.03, 1.0);
                            let h = norm * (rect.height() - 8.0);
                            let x0 = rect.left() + 4.0 + (b as f32) * bar_w;
                            let bar = egui::Rect::from_min_max(
                                egui::pos2(x0 + 1.0, rect.bottom() - 4.0 - h),
                                egui::pos2(x0 + bar_w - 1.5, rect.bottom() - 4.0),
                            );
                            p.rect_filled(bar, 1.5, egui::Color32::from_rgb(56, 189, 248));
                        }
                    });
                });

                if changed {
                    self.commit_project_change(prev_snapshot);
                }
            });
    }
}
