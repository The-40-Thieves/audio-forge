//! Right-hand Object Editor, Multi-Octave Piano Roll, and Track/Aux/Master FX Inspector.

use eframe::egui;
use forge_engine::{FadeCurve, FxSlotConfig, MidiNote, SynthPreset};

use crate::fx_ui::draw_fx_slot_controls;
use crate::{ForgeDawApp, InspectorTab};

impl ForgeDawApp {
    pub(crate) fn draw_object_editor(&mut self, ui: &mut egui::Ui) {
        egui::Panel::right("object_editor")
            .min_size(310.0)
            .default_size(340.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.inspector_tab, InspectorTab::Object, "Object");
                    ui.selectable_value(
                        &mut self.inspector_tab,
                        InspectorTab::PianoRoll,
                        "Piano Roll",
                    );
                    ui.selectable_value(
                        &mut self.inspector_tab,
                        InspectorTab::TrackBusFx,
                        "Track/Bus FX",
                    );
                });
                ui.separator();

                match self.inspector_tab {
                    InspectorTab::PianoRoll => {
                        self.draw_piano_roll_inspector(ui);
                        return;
                    }
                    InspectorTab::TrackBusFx => {
                        self.draw_track_bus_fx_inspector(ui);
                        return;
                    }
                    InspectorTab::Object => {}
                }

                ui.heading(
                    egui::RichText::new("OBJECT EDITOR")
                        .color(egui::Color32::from_rgb(251, 146, 60)),
                );
                ui.label(
                    egui::RichText::new("Samplitude-Style Per-Object Elastic, EQ, Fades & FX")
                        .small()
                        .color(egui::Color32::from_rgb(148, 163, 184)),
                );
                ui.separator();

                let Some(obj_id) = self.selected_object_id else {
                    ui.add_space(16.0);
                    ui.label("Select an Audio Object on the timeline (or click '+ Demo Clip') to edit its fades, 3-band EQ, and per-object FX chain.");
                    return;
                };

                let mut changed = false;
                let mut delete_obj = false;
                let mut split_obj = false;
                let mut split_stems = false;
                let splitting_now = self.stem_split_rx.is_some();
                let prev_snapshot = self.project.clone();
                let sr = f64::from(self.project.sample_rate.max(1));

                if let Some(obj) = self.project.object_mut(obj_id) {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Name:");
                            if ui.text_edit_singleline(&mut obj.name).changed() {
                                changed = true;
                            }
                        });
                        ui.horizontal(|ui| {
                            if ui.button("Split @ Playhead").clicked() {
                                split_obj = true;
                            }
                            if ui
                                .add_enabled(
                                    !splitting_now,
                                    egui::Button::new(if splitting_now {
                                        "Splitting..."
                                    } else {
                                        "Split Stems (Studio)"
                                    }),
                                )
                                .clicked()
                            {
                                split_stems = true;
                            }
                            if ui.button("Delete Object").clicked() {
                                delete_obj = true;
                            }
                        });
                        ui.separator();

                        // Gain / Pan / Polarity / Mute
                        egui::Grid::new("obj_core_grid")
                            .num_columns(2)
                            .spacing([12.0, 6.0])
                            .show(ui, |ui| {
                                ui.label("Gain (dB)");
                                changed |= ui
                                    .add(egui::Slider::new(&mut obj.gain_db, -36.0..=18.0))
                                    .changed();
                                ui.end_row();

                                ui.label("Pan");
                                changed |= ui
                                    .add(egui::Slider::new(&mut obj.pan, -1.0..=1.0))
                                    .changed();
                                ui.end_row();

                                ui.label("Phase / Mute");
                                ui.horizontal(|ui| {
                                    changed |= ui.checkbox(&mut obj.phase_invert, "Invert").changed();
                                    changed |= ui.checkbox(&mut obj.mute, "Mute").changed();
                                });
                                ui.end_row();

                                ui.label("Time Stretch");
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut obj.time_stretch, 0.5..=2.0)
                                            .suffix("x"),
                                    )
                                    .changed();
                                ui.end_row();

                                ui.label("Pitch Shift");
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut obj.pitch_semitones, -12.0..=12.0)
                                            .suffix(" st"),
                                    )
                                    .changed();
                                ui.end_row();
                            });

                        ui.separator();
                        ui.label(egui::RichText::new("Fades").strong());
                        let mut fi_ms = (obj.fade_in_samples as f64 / sr * 1000.0) as f32;
                        let mut fo_ms = (obj.fade_out_samples as f64 / sr * 1000.0) as f32;
                        let max_ms = (obj.length_samples as f64 / sr * 500.0).max(1.0) as f32;
                        if ui
                            .add(egui::Slider::new(&mut fi_ms, 0.0..=max_ms).text("Fade In (ms)"))
                            .changed()
                        {
                            obj.fade_in_samples = ((f64::from(fi_ms) * 0.001 * sr).round() as u64)
                                .min(obj.length_samples);
                            changed = true;
                        }
                        if ui
                            .add(egui::Slider::new(&mut fo_ms, 0.0..=max_ms).text("Fade Out (ms)"))
                            .changed()
                        {
                            obj.fade_out_samples = ((f64::from(fo_ms) * 0.001 * sr).round() as u64)
                                .min(obj.length_samples);
                            changed = true;
                        }
                        ui.horizontal(|ui| {
                            ui.label("Curve:");
                            for (lbl, curve) in [
                                ("Lin", FadeCurve::Linear),
                                ("EqPwr", FadeCurve::EqualPower),
                                ("Exp", FadeCurve::Exponential),
                                ("Log", FadeCurve::Logarithmic),
                            ] {
                                if ui.selectable_label(obj.fade_curve == curve, lbl).clicked() {
                                    obj.fade_curve = curve;
                                    changed = true;
                                }
                            }
                        });

                        ui.separator();
                        ui.label(egui::RichText::new("Per-Object 3-Band EQ").strong());
                        changed |= ui
                            .add(egui::Slider::new(&mut obj.eq.low_db, -12.0..=12.0).text("Low 120Hz"))
                            .changed();
                        changed |= ui
                            .add(egui::Slider::new(&mut obj.eq.mid_db, -12.0..=12.0).text("Mid 1.2kHz"))
                            .changed();
                        changed |= ui
                            .add(egui::Slider::new(&mut obj.eq.high_db, -12.0..=12.0).text("High 6.5kHz"))
                            .changed();

                        ui.separator();
                        ui.label(
                            egui::RichText::new(format!(
                                "Per-Object FX Chain ({})",
                                obj.fx_chain.len()
                            ))
                            .strong()
                            .color(egui::Color32::from_rgb(56, 189, 248)),
                        );

                        egui::ComboBox::from_id_salt("add_obj_fx")
                            .selected_text("+ Add ForgeAudio FX...")
                            .show_ui(ui, |ui| {
                                type FxPreset = (&'static str, fn() -> FxSlotConfig);
                                let presets: [FxPreset; 8] = [
                                    ("Forge OpenSoothe De-Fizz", FxSlotConfig::default_open_soothe),
                                    ("Forge Vocal Surgical", FxSlotConfig::default_vocal_surgical),
                                    ("Forge Vocal Tune", FxSlotConfig::default_vocal_tune),
                                    ("Forge SubMono & Tape", FxSlotConfig::default_sub_mono_tape),
                                    ("Forge SplitEQ", FxSlotConfig::default_split_eq),
                                    ("Forge Density & Clipper", FxSlotConfig::default_density_clipper),
                                    ("Forge SpaceVerb FDN", FxSlotConfig::default_space_verb),
                                    ("Forge Master Limiter", FxSlotConfig::default_master_limiter),
                                ];
                                for (label, make_fn) in presets {
                                    if ui.selectable_label(false, label).clicked() {
                                        obj.fx_chain.push(make_fn());
                                        changed = true;
                                    }
                                }
                            });

                        let mut remove_idx = None;
                        for (idx, fx) in obj.fx_chain.iter_mut().enumerate() {
                            ui.group(|ui| {
                                ui.horizontal(|ui| {
                                    let mut active = !fx.is_bypassed();
                                    if ui.checkbox(&mut active, fx.display_name()).changed() {
                                        fx.set_bypass(!active);
                                        changed = true;
                                    }
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            if ui.small_button("X").clicked() {
                                                remove_idx = Some(idx);
                                            }
                                        },
                                    );
                                });
                                changed |= draw_fx_slot_controls(ui, fx);
                            });
                        }
                        if let Some(rm) = remove_idx {
                            obj.fx_chain.remove(rm);
                            changed = true;
                        }
                    });
                }

                if split_obj {
                    let split_pos = self.playhead_samples();
                    if let Some(right_id) = self.project.split_object(obj_id, split_pos) {
                        self.selected_object_id = Some(right_id);
                        self.commit_project_change(prev_snapshot);
                        self.status_line = "Split object at playhead".to_string();
                    } else {
                        self.status_line =
                            "Move playhead inside the selected object to split".to_string();
                    }
                } else if split_stems {
                    self.start_stem_split_for_object(obj_id);
                } else if delete_obj {
                    if let Some((ti, oi)) = self.project.find_object(obj_id) {
                        self.project.tracks[ti].objects.remove(oi);
                        self.selected_object_id = None;
                        self.commit_project_change(prev_snapshot);
                    }
                } else if changed {
                    self.commit_project_change(prev_snapshot);
                }
            });
    }

    fn draw_piano_roll_inspector(&mut self, ui: &mut egui::Ui) {
        ui.heading(
            egui::RichText::new("MIDI PIANO ROLL & SYNTH")
                .color(egui::Color32::from_rgb(192, 132, 252)),
        );
        if self.project.tracks.is_empty() {
            ui.label("No tracks in project. Click '+ Track' to add one.");
            return;
        }
        let tr_idx = self
            .selected_track_idx
            .min(self.project.tracks.len().saturating_sub(1));
        ui.label(
            egui::RichText::new(format!("Track: {}", self.project.tracks[tr_idx].name))
                .small()
                .color(egui::Color32::from_rgb(148, 163, 184)),
        );
        ui.separator();

        let prev = self.project.clone();
        let mut changed = false;

        ui.horizontal(|ui| {
            ui.label("Synth:");
            egui::ComboBox::from_id_salt("track_synth_preset")
                .selected_text(self.project.tracks[tr_idx].synth_preset.label())
                .show_ui(ui, |ui| {
                    for preset in SynthPreset::ALL {
                        if ui
                            .selectable_value(
                                &mut self.project.tracks[tr_idx].synth_preset,
                                preset,
                                preset.label(),
                            )
                            .changed()
                        {
                            changed = true;
                        }
                    }
                });
        });

        if self.project.tracks[tr_idx].midi_clips.is_empty() {
            ui.add_space(12.0);
            if ui.button("+ Create 4-Bar Am7 MIDI Clip").clicked() {
                self.add_demo_midi_clip();
            }
            if changed {
                self.commit_project_change(prev);
            }
            return;
        }

        // Resolve the specifically selected MIDI clip on this track (fallback to first clip)
        let clip_idx = self
            .selected_midi_clip_id
            .and_then(|cid| {
                self.project.tracks[tr_idx]
                    .midi_clips
                    .iter()
                    .position(|c| c.id == cid)
            })
            .unwrap_or(0);

        // Interactive Multi-Octave (2 octaves visible, C1..C7 scrollable) × Multi-Bar Piano Roll Grid
        let sr = f64::from(self.project.sample_rate.max(8_000));
        let step_smp = (((60.0 / self.project.bpm.max(30.0)) * 0.25) * sr).round() as u64;
        let bar_steps = (u64::from(self.project.time_sig_num.clamp(1, 16)) * 16
            / u64::from(self.project.time_sig_den.clamp(1, 16)))
        .max(1);
        let bar_smp = step_smp.saturating_mul(bar_steps).max(1);
        let total_bars = self.project.tracks[tr_idx].midi_clips[clip_idx]
            .length_samples
            .div_ceil(bar_smp)
            .max(1);
        if self.piano_roll_bar >= total_bars {
            self.piano_roll_bar = total_bars - 1;
        }
        let bar_offset_smp = self.piano_roll_bar.saturating_mul(bar_smp);

        let mut delete_clip = false;
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(format!(
                    "'{}' | Bar {}/{} | Oct C{}–C{}",
                    self.project.tracks[tr_idx].midi_clips[clip_idx].name,
                    self.piano_roll_bar + 1,
                    total_bars,
                    self.piano_roll_base_octave,
                    self.piano_roll_base_octave + 2
                ))
                .small()
                .color(egui::Color32::from_rgb(148, 163, 184)),
            );
            if ui.small_button("< Bar").clicked() && self.piano_roll_bar > 0 {
                self.piano_roll_bar -= 1;
            }
            if ui.small_button("Bar >").clicked() && self.piano_roll_bar + 1 < total_bars {
                self.piano_roll_bar += 1;
            }
            if ui.small_button("Oct -").clicked() && self.piano_roll_base_octave > 1 {
                self.piano_roll_base_octave -= 1;
            }
            if ui.small_button("Oct +").clicked() && self.piano_roll_base_octave < 5 {
                self.piano_roll_base_octave += 1;
            }
            if ui
                .small_button(egui::RichText::new("Del Clip").color(egui::Color32::LIGHT_RED))
                .clicked()
            {
                delete_clip = true;
            }
        });

        if delete_clip {
            self.project.tracks[tr_idx].midi_clips.remove(clip_idx);
            self.selected_midi_clip_id = None;
            self.commit_project_change(prev);
            return;
        }

        const PITCH_CLASSES: [&str; 12] = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];
        let low_pitch = (self.piano_roll_base_octave + 1).saturating_mul(12);
        let high_pitch = low_pitch.saturating_add(24).min(108);
        let mut add_note: Option<(u64, u8)> = None;
        let mut remove_note_idx: Option<usize> = None;

        egui::ScrollArea::vertical().show(ui, |ui| {
            let clip = &self.project.tracks[tr_idx].midi_clips[clip_idx];
            for pitch in (low_pitch..=high_pitch).rev() {
                let pc = PITCH_CLASSES[usize::from(pitch % 12)];
                let oct = (pitch / 12).saturating_sub(1);
                let is_black = pc.ends_with('#');
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 2.0;
                    ui.add_sized(
                        [30.0, 15.0],
                        egui::Label::new(
                            egui::RichText::new(format!("{pc}{oct}"))
                                .monospace()
                                .small()
                                .color(if is_black {
                                    egui::Color32::from_rgb(148, 163, 184)
                                } else {
                                    egui::Color32::from_rgb(226, 232, 240)
                                }),
                        ),
                    );
                    for step in 0..bar_steps {
                        let start = bar_offset_smp.saturating_add(step * step_smp);
                        if start >= clip.length_samples {
                            break;
                        }
                        let existing_idx = clip.notes.iter().position(|n| {
                            n.pitch == pitch && n.start_samples.abs_diff(start) < step_smp / 2
                        });
                        let active = existing_idx.is_some();
                        let col = if active {
                            egui::Color32::from_rgb(192, 132, 252)
                        } else if step % 4 == 0 {
                            egui::Color32::from_rgb(38, 46, 62)
                        } else if is_black {
                            egui::Color32::from_rgb(18, 23, 34)
                        } else {
                            egui::Color32::from_rgb(24, 30, 42)
                        };
                        let btn = egui::Button::new("")
                            .fill(col)
                            .min_size(egui::vec2(14.0, 14.0));
                        if ui.add(btn).clicked() {
                            if let Some(idx) = existing_idx {
                                remove_note_idx = Some(idx);
                            } else {
                                add_note = Some((start, pitch));
                            }
                        }
                    }
                });
            }
        });

        if let Some(idx) = remove_note_idx {
            self.project.tracks[tr_idx].midi_clips[clip_idx]
                .notes
                .remove(idx);
            changed = true;
        } else if let Some((start, pitch)) = add_note {
            let unique_id = self.project.alloc_id();
            self.project.tracks[tr_idx].midi_clips[clip_idx]
                .notes
                .push(MidiNote {
                    id: unique_id,
                    start_samples: start,
                    length_samples: step_smp.saturating_mul(9) / 10,
                    pitch,
                    velocity: 0.82,
                });
            changed = true;
        }

        if changed {
            self.commit_project_change(prev);
        }
    }

    fn draw_track_bus_fx_inspector(&mut self, ui: &mut egui::Ui) {
        ui.heading(
            egui::RichText::new("TRACK, AUX & MASTER FX")
                .color(egui::Color32::from_rgb(56, 189, 248)),
        );
        if self.project.tracks.is_empty() {
            ui.label("No tracks in project. Click '+ Track' to add one.");
            return;
        }
        let tr_idx = self
            .selected_track_idx
            .min(self.project.tracks.len().saturating_sub(1));
        let prev = self.project.clone();
        let mut changed = false;

        egui::ScrollArea::vertical().show(ui, |ui| {
            type FxPreset = (&'static str, fn() -> FxSlotConfig);
            let presets: [FxPreset; 8] = [
                (
                    "Forge OpenSoothe De-Fizz",
                    FxSlotConfig::default_open_soothe,
                ),
                ("Forge Vocal Surgical", FxSlotConfig::default_vocal_surgical),
                ("Forge Vocal Tune", FxSlotConfig::default_vocal_tune),
                ("Forge SubMono & Tape", FxSlotConfig::default_sub_mono_tape),
                ("Forge SplitEQ", FxSlotConfig::default_split_eq),
                (
                    "Forge Density & Clipper",
                    FxSlotConfig::default_density_clipper,
                ),
                ("Forge SpaceVerb FDN", FxSlotConfig::default_space_verb),
                ("Forge Master Limiter", FxSlotConfig::default_master_limiter),
            ];

            let mut delete_track = false;
            ui.horizontal(|ui| {
                ui.label("Track:");
                if ui
                    .text_edit_singleline(&mut self.project.tracks[tr_idx].name)
                    .changed()
                {
                    changed = true;
                }
            });
            ui.horizontal(|ui| {
                changed |= ui
                    .checkbox(
                        &mut self.project.tracks[tr_idx].phase_invert,
                        "Phase Invert",
                    )
                    .changed();
                if self.project.tracks.len() > 1 && ui.small_button("Delete Track").clicked() {
                    delete_track = true;
                }
            });
            if delete_track {
                self.project.tracks.remove(tr_idx);
                self.selected_track_idx = tr_idx.min(self.project.tracks.len().saturating_sub(1));
                self.selected_object_id = None;
                changed = true;
                return;
            }
            ui.separator();

            ui.label(
                egui::RichText::new(format!(
                    "Track Inserts: {}",
                    self.project.tracks[tr_idx].name
                ))
                .strong(),
            );
            egui::ComboBox::from_id_salt("add_track_fx")
                .selected_text("+ Add Track Insert...")
                .show_ui(ui, |ui| {
                    for (label, make_fn) in presets {
                        if ui.selectable_label(false, label).clicked() {
                            self.project.tracks[tr_idx].inserts.push(make_fn());
                            changed = true;
                        }
                    }
                });
            let mut rm_tr = None;
            for (idx, fx) in self.project.tracks[tr_idx].inserts.iter_mut().enumerate() {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.strong(fx.display_name());
                        if ui.small_button("X").clicked() {
                            rm_tr = Some(idx);
                        }
                    });
                    changed |= draw_fx_slot_controls(ui, fx);
                });
            }
            if let Some(rm) = rm_tr {
                self.project.tracks[tr_idx].inserts.remove(rm);
                changed = true;
            }

            ui.separator();
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Aux Bus Inserts").strong());
                if ui.small_button("+ Aux Bus").clicked() {
                    let n = self.project.aux_buses.len() + 1;
                    self.project.add_aux_bus(format!("Aux {n}: FX Bus"));
                    changed = true;
                }
            });
            for (bi, bus) in self.project.aux_buses.iter_mut().enumerate() {
                ui.label(
                    egui::RichText::new(&bus.name)
                        .small()
                        .strong()
                        .color(egui::Color32::from_rgb(192, 132, 252)),
                );
                egui::ComboBox::from_id_salt(("add_aux_fx", bi))
                    .selected_text("+ Add Aux Insert...")
                    .show_ui(ui, |ui| {
                        for (label, make_fn) in presets {
                            if ui.selectable_label(false, label).clicked() {
                                bus.inserts.push(make_fn());
                                changed = true;
                            }
                        }
                    });
                let mut rm_aux = None;
                for (idx, fx) in bus.inserts.iter_mut().enumerate() {
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            ui.strong(fx.display_name());
                            if ui.small_button("X").clicked() {
                                rm_aux = Some(idx);
                            }
                        });
                        changed |= draw_fx_slot_controls(ui, fx);
                    });
                }
                if let Some(rm) = rm_aux {
                    bus.inserts.remove(rm);
                    changed = true;
                }
            }

            ui.separator();
            ui.label(egui::RichText::new("Master Bus Inserts").strong());
            egui::ComboBox::from_id_salt("add_master_fx")
                .selected_text("+ Add Master Insert...")
                .show_ui(ui, |ui| {
                    for (label, make_fn) in presets {
                        if ui.selectable_label(false, label).clicked() {
                            self.project.master_inserts.push(make_fn());
                            changed = true;
                        }
                    }
                });
            let mut rm_m = None;
            for (idx, fx) in self.project.master_inserts.iter_mut().enumerate() {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.strong(fx.display_name());
                        if ui.small_button("X").clicked() {
                            rm_m = Some(idx);
                        }
                    });
                    changed |= draw_fx_slot_controls(ui, fx);
                });
            }
            if let Some(rm) = rm_m {
                self.project.master_inserts.remove(rm);
                changed = true;
            }
        });

        if changed {
            self.commit_project_change(prev);
        }
    }
}
