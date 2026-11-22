//! # `forge-daw` — ForgeStudio Object-Oriented Mixing & Mastering DAW
//!
//! Samplitude-inspired native Rust DAW built on `forge-engine`, `forge-dsp`, and `eframe`/`egui`:
//! - Multitrack arrange timeline with real waveform peaks, drag-to-move, split-at-playhead,
//!   fades, and zoom/scrub ruler.
//! - Dedicated **Object Editor** inspector with per-object gain, pan, polarity, fade curves,
//!   3-band EQ, and **per-object FX chains** hosting all native ForgeAudio processors.
//! - Live mixer console with 32-band Master Spectrum, automatic PDC, hardware `cpal` playback,
//!   `.forgeproj` save/load, and 24-bit WAV mastering bounce with ITU-R BS.1770-4 audit.

use std::f64::consts::PI;
use std::path::PathBuf;

use eframe::egui;
use forge_core::AudioBuffer;
use forge_engine::{
    AudioObject, ClipCache, DawProject, FadeCurve, FxSlotConfig, LiveAudioEngine,
    ObjectOrientedEngine, UndoStack,
};

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1360.0, 840.0])
            .with_min_inner_size([960.0, 600.0])
            .with_title("ForgeStudio DAW — Object-Oriented Audio & Mastering"),
        ..Default::default()
    };
    eframe::run_native(
        "ForgeStudio DAW",
        options,
        Box::new(|cc| {
            configure_studio_theme(&cc.egui_ctx);
            Ok(Box::new(ForgeDawApp::new()))
        }),
    )
}

fn configure_studio_theme(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.window_fill = egui::Color32::from_rgb(15, 18, 25);
    visuals.panel_fill = egui::Color32::from_rgb(18, 22, 31);
    visuals.faint_bg_color = egui::Color32::from_rgb(24, 30, 42);
    visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(26, 32, 44);
    visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(33, 41, 56);
    visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(45, 58, 79);
    visuals.widgets.active.bg_fill = egui::Color32::from_rgb(56, 189, 248);
    ctx.set_visuals(visuals);
}

struct ForgeDawApp {
    project: DawProject,
    undo: UndoStack,
    cache: ClipCache,
    audio: Option<LiveAudioEngine>,
    selected_track_idx: usize,
    selected_object_id: Option<u64>,
    pixels_per_sec: f32,
    scroll_secs: f32,
    import_path_buf: String,
    project_path_buf: String,
    bounce_path_buf: String,
    status_line: String,
    show_mixer: bool,
}

impl ForgeDawApp {
    fn new() -> Self {
        let project = DawProject::default();
        let cache = ClipCache::default();
        let (audio, status_line) = match LiveAudioEngine::start(&project, &cache) {
            Ok(eng) => {
                let msg = format!(
                    "Ready — Audio: {} ({} Hz)",
                    eng.device_name(),
                    eng.sample_rate()
                );
                (Some(eng), msg)
            }
            Err(e) => (
                None,
                format!("Audio output unavailable ({e}) — Offline mode"),
            ),
        };
        Self {
            project,
            undo: UndoStack::default(),
            cache,
            audio,
            selected_track_idx: 0,
            selected_object_id: None,
            pixels_per_sec: 120.0,
            scroll_secs: 0.0,
            import_path_buf: String::new(),
            project_path_buf: "session.forgeproj".to_string(),
            bounce_path_buf: "master_bounce.wav".to_string(),
            status_line,
            show_mixer: true,
        }
    }

    fn commit_project_change(&mut self, prev: DawProject) {
        self.undo.push(prev);
        self.sync_audio_engine();
    }

    fn sync_audio_engine(&self) {
        if let Some(ref eng) = self.audio {
            eng.reload_project(&self.project, &self.cache);
        }
    }

    fn playhead_samples(&self) -> u64 {
        self.audio
            .as_ref()
            .map_or(0, LiveAudioEngine::playhead_samples)
    }

    fn set_playhead_samples(&self, pos: u64) {
        if let Some(ref eng) = self.audio {
            eng.seek(pos);
        }
    }

    fn add_demo_clip(&mut self) {
        let sr = self.project.sample_rate;
        let frames = (sr as usize) * 4;
        let mut ch_l = vec![0.0_f64; frames];
        let mut ch_r = vec![0.0_f64; frames];
        let chord_freqs = [220.0_f64, 261.63, 329.63, 392.0]; // Am7
        for i in 0..frames {
            let t = (i as f64) / f64::from(sr);
            let beat_env = (-((t * 2.0).fract()) * 6.0).exp();
            let mut sum = 0.0;
            for &f in &chord_freqs {
                sum += 0.12 * (2.0 * PI * f * t).sin();
            }
            let kick = 0.35 * (2.0 * PI * (55.0 + 90.0 * beat_env) * t).sin() * beat_env;
            ch_l[i] = (sum + kick) * 0.7;
            ch_r[i] = (sum * 0.95 + kick) * 0.7;
        }
        if let Ok(buf) = AudioBuffer::from_channels(vec![ch_l, ch_r], sr) {
            let prev = self.project.clone();
            let path = PathBuf::from(format!("mem://demo_am7_{}.wav", self.project.tracks.len()));
            self.cache.insert_memory_clip(path.clone(), buf);
            let id = self.project.alloc_id();
            let mut obj = AudioObject::new(id, "Demo Am7 Groove", path, frames as u64);
            obj.timeline_start_samples = self.playhead_samples();
            obj.fade_in_samples = u64::from(sr) / 20;
            obj.fade_out_samples = u64::from(sr) / 10;
            let tr_idx = self
                .selected_track_idx
                .min(self.project.tracks.len().saturating_sub(1));
            self.project.tracks[tr_idx].objects.push(obj);
            self.selected_object_id = Some(id);
            self.commit_project_change(prev);
            self.status_line = "Inserted 4s Demo Am7 Audio Object".to_string();
        }
    }

    fn import_wav_to_selected_track(&mut self) {
        let raw = self.import_path_buf.trim().to_string();
        if raw.is_empty() {
            self.status_line = "Enter a WAV/audio file path to import".to_string();
            return;
        }
        let path = PathBuf::from(&raw);
        match self.cache.load_clip(&path) {
            Ok(buf) => {
                let prev = self.project.clone();
                let id = self.project.alloc_id();
                let name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("Clip")
                    .to_string();
                let mut obj = AudioObject::new(id, name, path, buf.num_frames() as u64);
                obj.timeline_start_samples = self.playhead_samples();
                let tr_idx = self
                    .selected_track_idx
                    .min(self.project.tracks.len().saturating_sub(1));
                self.project.tracks[tr_idx].objects.push(obj);
                self.selected_object_id = Some(id);
                self.commit_project_change(prev);
                self.status_line = format!("Imported '{}' onto track {}", raw, tr_idx + 1);
            }
            Err(e) => {
                self.status_line = format!("Import failed: {e}");
            }
        }
    }

    fn bounce_master(&mut self) {
        let out_path = PathBuf::from(self.bounce_path_buf.trim());
        match ObjectOrientedEngine::bounce_to_wav(&self.project, &self.cache, &out_path) {
            Ok(report) => {
                self.status_line = format!(
                    "Bounced '{}' | {:.2} LUFS | {:.2} dBTP | DistroKid Ready: {}",
                    out_path.display(),
                    report.integrated_lufs,
                    report.true_peak_dbtp,
                    if report.distrokid_ready { "YES" } else { "NO" }
                );
            }
            Err(e) => {
                self.status_line = format!("Bounce failed: {e}");
            }
        }
    }

    fn draw_transport_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("transport_bar").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                ui.label(
                    egui::RichText::new("FORGESTUDIO DAW")
                        .strong()
                        .color(egui::Color32::from_rgb(56, 189, 248)),
                );
                ui.separator();

                let is_playing = self.audio.as_ref().is_some_and(LiveAudioEngine::is_playing);
                if ui
                    .button(
                        egui::RichText::new(if is_playing { "Pause" } else { "Play" })
                            .strong()
                            .color(if is_playing {
                                egui::Color32::from_rgb(74, 222, 128)
                            } else {
                                egui::Color32::from_rgb(226, 232, 240)
                            }),
                    )
                    .clicked()
                    && let Some(ref eng) = self.audio
                {
                    eng.toggle_play();
                }
                if ui.button("Stop").clicked()
                    && let Some(ref eng) = self.audio
                {
                    eng.stop();
                }
                if ui
                    .selectable_label(self.project.loop_enabled, "Loop")
                    .clicked()
                {
                    let prev = self.project.clone();
                    self.project.loop_enabled = !self.project.loop_enabled;
                    self.commit_project_change(prev);
                }

                ui.separator();
                let pos = self.playhead_samples();
                let secs = (pos as f64) / f64::from(self.project.sample_rate.max(1));
                let mins = (secs / 60.0).floor() as u64;
                let rem_secs = secs - (mins as f64) * 60.0;
                let beats = secs * (self.project.bpm / 60.0);
                let bar = (beats / f64::from(self.project.time_sig_num)).floor() as u64 + 1;
                let beat = (beats % f64::from(self.project.time_sig_num)).floor() as u64 + 1;
                ui.label(
                    egui::RichText::new(format!(
                        "{:02}:{:06.3}  |  Bar {:03}.{}",
                        mins, rem_secs, bar, beat
                    ))
                    .monospace()
                    .strong()
                    .color(egui::Color32::from_rgb(120, 240, 170)),
                );

                ui.separator();
                ui.label("BPM");
                let mut bpm = self.project.bpm;
                if ui
                    .add(
                        egui::DragValue::new(&mut bpm)
                            .speed(0.5)
                            .range(30.0..=300.0),
                    )
                    .changed()
                {
                    self.project.bpm = bpm;
                }

                ui.separator();
                if ui
                    .add_enabled(self.undo.can_undo(), egui::Button::new("Undo"))
                    .clicked()
                    && let Some(prev) = self.undo.undo(self.project.clone())
                {
                    self.project = prev;
                    self.sync_audio_engine();
                }
                if ui
                    .add_enabled(self.undo.can_redo(), egui::Button::new("Redo"))
                    .clicked()
                    && let Some(next) = self.undo.redo(self.project.clone())
                {
                    self.project = next;
                    self.sync_audio_engine();
                }

                ui.separator();
                if ui.button("+ Demo Clip").clicked() {
                    self.add_demo_clip();
                }
                if ui.button("+ Track").clicked() {
                    let prev = self.project.clone();
                    let n = self.project.tracks.len() + 1;
                    self.project.add_track(format!("{n:02} Audio Track"));
                    self.commit_project_change(prev);
                }
                if ui.button("Bounce Master").clicked() {
                    self.bounce_master();
                }
                ui.checkbox(&mut self.show_mixer, "Mixer");
            });

            // Second compact I/O bar: Import WAV + Save/Load .forgeproj + Bounce path
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("WAV:").small());
                ui.add(
                    egui::TextEdit::singleline(&mut self.import_path_buf)
                        .desired_width(210.0)
                        .hint_text("path/to/audio.wav"),
                );
                if ui.small_button("Import").clicked() {
                    self.import_wav_to_selected_track();
                }
                ui.separator();
                ui.label(egui::RichText::new("Project:").small());
                ui.add(egui::TextEdit::singleline(&mut self.project_path_buf).desired_width(150.0));
                if ui.small_button("Save").clicked() {
                    let p = PathBuf::from(self.project_path_buf.trim());
                    match self.project.save_to_file(&p) {
                        Ok(()) => self.status_line = format!("Saved '{}'", p.display()),
                        Err(e) => self.status_line = format!("Save failed: {e}"),
                    }
                }
                if ui.small_button("Load").clicked() {
                    let p = PathBuf::from(self.project_path_buf.trim());
                    match DawProject::load_from_file(&p) {
                        Ok(loaded) => {
                            let prev = self.project.clone();
                            self.project = loaded;
                            for tr in &self.project.tracks {
                                for obj in &tr.objects {
                                    let _ = self.cache.load_clip(&obj.source_path);
                                }
                            }
                            self.commit_project_change(prev);
                            self.status_line = format!("Loaded '{}'", p.display());
                        }
                        Err(e) => self.status_line = format!("Load failed: {e}"),
                    }
                }
                ui.separator();
                ui.label(egui::RichText::new("Out:").small());
                ui.add(egui::TextEdit::singleline(&mut self.bounce_path_buf).desired_width(140.0));
                ui.separator();
                ui.label(
                    egui::RichText::new(&self.status_line)
                        .small()
                        .color(egui::Color32::from_rgb(165, 175, 195)),
                );
            });
            ui.add_space(4.0);
        });
    }

    fn draw_object_editor(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("object_editor")
            .min_width(310.0)
            .default_width(340.0)
            .show(ctx, |ui| {
                ui.heading(
                    egui::RichText::new("OBJECT EDITOR")
                        .color(egui::Color32::from_rgb(251, 146, 60)),
                );
                ui.label(
                    egui::RichText::new("Samplitude-Style Per-Object EQ, Fades & FX Chain")
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

    fn draw_mixer_dock(&mut self, ctx: &egui::Context) {
        if !self.show_mixer {
            return;
        }
        let meters = self
            .audio
            .as_ref()
            .map(LiveAudioEngine::meters_snapshot)
            .unwrap_or_default();

        egui::TopBottomPanel::bottom("mixer_dock")
            .min_height(175.0)
            .default_height(190.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("MIXER & MASTER SPECTRUM")
                            .strong()
                            .color(egui::Color32::from_rgb(56, 189, 248)),
                    );
                    ui.separator();
                    ui.label(
                        egui::RichText::new(format!(
                            "Master L {:+5.1} dB | R {:+5.1} dB | PDC: {} smp",
                            meters.master_peak_l_db,
                            meters.master_peak_r_db,
                            meters.total_pdc_samples
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
                                for (ti, tr) in self.project.tracks.iter_mut().enumerate() {
                                    ui.group(|ui| {
                                        ui.set_width(110.0);
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

    fn draw_arrange_timeline(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("ARRANGE TIMELINE").strong());
                ui.separator();
                ui.add(
                    egui::Slider::new(&mut self.pixels_per_sec, 30.0..=400.0).text("Zoom (px/s)"),
                );
                ui.add(egui::Slider::new(&mut self.scroll_secs, 0.0..=120.0).text("Scroll (s)"));
            });
            ui.separator();

            let sr = f64::from(self.project.sample_rate.max(1));
            let playhead_secs = (self.playhead_samples() as f64 / sr) as f32;
            let header_w = 175.0_f32;
            let lane_h = 84.0_f32;

            // Clickable Time Ruler
            ui.horizontal(|ui| {
                ui.allocate_exact_size(egui::vec2(header_w, 22.0), egui::Sense::hover());
                let (ruler_rect, ruler_resp) = ui.allocate_exact_size(
                    egui::vec2(ui.available_width().max(200.0), 22.0),
                    egui::Sense::click_and_drag(),
                );
                let rp = ui.painter_at(ruler_rect);
                rp.rect_filled(ruler_rect, 3.0, egui::Color32::from_rgb(22, 28, 40));

                if (ruler_resp.clicked() || ruler_resp.dragged())
                    && let Some(pos) = ruler_resp.interact_pointer_pos()
                {
                    let rel_x = (pos.x - ruler_rect.left()).max(0.0);
                    let target_secs = f64::from(self.scroll_secs + rel_x / self.pixels_per_sec);
                    self.set_playhead_samples((target_secs * sr).round() as u64);
                }

                let start_sec = self.scroll_secs.floor() as i32;
                let vis_secs = (ruler_rect.width() / self.pixels_per_sec).ceil() as i32 + 2;
                for s in start_sec..(start_sec + vis_secs) {
                    if s < 0 {
                        continue;
                    }
                    let x =
                        ruler_rect.left() + ((s as f32) - self.scroll_secs) * self.pixels_per_sec;
                    if x >= ruler_rect.left() && x <= ruler_rect.right() {
                        rp.line_segment(
                            [
                                egui::pos2(x, ruler_rect.top()),
                                egui::pos2(x, ruler_rect.bottom()),
                            ],
                            egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(60, 74, 96)),
                        );
                        rp.text(
                            egui::pos2(x + 3.0, ruler_rect.center().y),
                            egui::Align2::LEFT_CENTER,
                            format!("{s}s"),
                            egui::FontId::monospace(10.0),
                            egui::Color32::from_rgb(165, 175, 195),
                        );
                    }
                }
            });

            let prev_snapshot = self.project.clone();
            let mut changed = false;

            egui::ScrollArea::vertical().show(ui, |ui| {
                for ti in 0..self.project.tracks.len() {
                    let tr_color = egui::Color32::from_rgb(
                        self.project.tracks[ti].color_rgb[0],
                        self.project.tracks[ti].color_rgb[1],
                        self.project.tracks[ti].color_rgb[2],
                    );

                    ui.horizontal(|ui| {
                        // Track Header
                        let selected_tr = self.selected_track_idx == ti;
                        egui::Frame::group(ui.style())
                            .fill(if selected_tr {
                                egui::Color32::from_rgb(30, 41, 59)
                            } else {
                                egui::Color32::from_rgb(20, 26, 37)
                            })
                            .show(ui, |ui| {
                                ui.set_min_size(egui::vec2(header_w - 14.0, lane_h - 10.0));
                                ui.set_max_size(egui::vec2(header_w - 14.0, lane_h - 10.0));
                                ui.vertical(|ui| {
                                    if ui
                                        .selectable_label(
                                            selected_tr,
                                            egui::RichText::new(&self.project.tracks[ti].name)
                                                .strong()
                                                .color(tr_color),
                                        )
                                        .clicked()
                                    {
                                        self.selected_track_idx = ti;
                                    }
                                    ui.horizontal(|ui| {
                                        if ui
                                            .selectable_label(self.project.tracks[ti].mute, "M")
                                            .clicked()
                                        {
                                            self.project.tracks[ti].mute =
                                                !self.project.tracks[ti].mute;
                                            changed = true;
                                        }
                                        if ui
                                            .selectable_label(self.project.tracks[ti].solo, "S")
                                            .clicked()
                                        {
                                            self.project.tracks[ti].solo =
                                                !self.project.tracks[ti].solo;
                                            changed = true;
                                        }
                                        ui.label(
                                            egui::RichText::new(format!(
                                                "{:+.1}dB",
                                                self.project.tracks[ti].gain_db
                                            ))
                                            .monospace()
                                            .small(),
                                        );
                                    });
                                });
                            });

                        // Track Lane Canvas
                        let (lane_rect, lane_resp) = ui.allocate_exact_size(
                            egui::vec2(ui.available_width().max(200.0), lane_h),
                            egui::Sense::click_and_drag(),
                        );
                        let p = ui.painter_at(lane_rect);
                        p.rect_filled(lane_rect, 4.0, egui::Color32::from_rgb(14, 18, 26));
                        p.rect_stroke(
                            lane_rect,
                            4.0,
                            egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(35, 45, 62)),
                            egui::StrokeKind::Inside,
                        );

                        // Draw Audio Objects on this track
                        for oi in 0..self.project.tracks[ti].objects.len() {
                            let obj_id = self.project.tracks[ti].objects[oi].id;
                            let start_s =
                                (self.project.tracks[ti].objects[oi].timeline_start_samples as f64
                                    / sr) as f32;
                            let len_s = (self.project.tracks[ti].objects[oi].length_samples as f64
                                / sr) as f32;
                            let x0 = lane_rect.left()
                                + (start_s - self.scroll_secs) * self.pixels_per_sec;
                            let w = (len_s * self.pixels_per_sec).max(16.0);
                            let obj_rect = egui::Rect::from_min_size(
                                egui::pos2(x0, lane_rect.top() + 4.0),
                                egui::vec2(w, lane_rect.height() - 8.0),
                            );

                            let is_sel = self.selected_object_id == Some(obj_id);
                            let fill = egui::Color32::from_rgba_unmultiplied(
                                tr_color.r(),
                                tr_color.g(),
                                tr_color.b(),
                                if is_sel { 80 } else { 42 },
                            );
                            p.rect_filled(obj_rect, 4.0, fill);
                            p.rect_stroke(
                                obj_rect,
                                4.0,
                                egui::Stroke::new(
                                    if is_sel { 2.0_f32 } else { 1.0_f32 },
                                    if is_sel {
                                        egui::Color32::from_rgb(250, 204, 21)
                                    } else {
                                        tr_color
                                    },
                                ),
                                egui::StrokeKind::Inside,
                            );

                            // Waveform peaks inside the object
                            if let Some(peaks) = self
                                .cache
                                .get_peaks(&self.project.tracks[ti].objects[oi].source_path)
                            {
                                let mid_y = obj_rect.center().y;
                                let half_h = (obj_rect.height() * 0.42).max(4.0);
                                let step_px = 2.0_f32;
                                let n_cols = ((obj_rect.width() / step_px) as usize).max(1);
                                let obj_len =
                                    self.project.tracks[ti].objects[oi].length_samples.max(1);
                                let src_off =
                                    self.project.tracks[ti].objects[oi].source_offset_samples;
                                for c in 0..n_cols {
                                    let frac = (c as f64) / (n_cols as f64);
                                    let smp = src_off + ((frac * (obj_len as f64)) as u64);
                                    let bucket = (smp as usize) / peaks.samples_per_bucket;
                                    if let Some(&(mn, mx)) = peaks.min_max.get(bucket) {
                                        let cx = obj_rect.left() + (c as f32) * step_px;
                                        if cx >= lane_rect.left() && cx <= lane_rect.right() {
                                            p.line_segment(
                                                [
                                                    egui::pos2(cx, mid_y - mx * half_h),
                                                    egui::pos2(cx, mid_y - mn * half_h),
                                                ],
                                                egui::Stroke::new(1.0_f32, tr_color),
                                            );
                                        }
                                    }
                                }
                            }

                            // Object header label + FX count badge
                            let fx_count = self.project.tracks[ti].objects[oi].fx_chain.len();
                            let label = if fx_count > 0 {
                                format!(
                                    "{}  [FX:{}]",
                                    self.project.tracks[ti].objects[oi].name, fx_count
                                )
                            } else {
                                self.project.tracks[ti].objects[oi].name.clone()
                            };
                            p.text(
                                egui::pos2(obj_rect.left() + 6.0, obj_rect.top() + 10.0),
                                egui::Align2::LEFT_CENTER,
                                label,
                                egui::FontId::proportional(11.0),
                                egui::Color32::WHITE,
                            );

                            // Click / drag interaction on object
                            if lane_resp.clicked()
                                && let Some(pt) = lane_resp.interact_pointer_pos()
                                && obj_rect.contains(pt)
                            {
                                self.selected_track_idx = ti;
                                self.selected_object_id = Some(obj_id);
                            }
                            if is_sel && lane_resp.dragged() {
                                let dx = lane_resp.drag_delta().x;
                                if dx.abs() > 0.1 {
                                    let delta_smp =
                                        (f64::from(dx / self.pixels_per_sec) * sr).round() as i64;
                                    let cur = self.project.tracks[ti].objects[oi]
                                        .timeline_start_samples
                                        as i64;
                                    self.project.tracks[ti].objects[oi].timeline_start_samples =
                                        (cur + delta_smp).max(0) as u64;
                                    changed = true;
                                }
                            }
                        }

                        // Playhead line
                        let ph_x = lane_rect.left()
                            + (playhead_secs - self.scroll_secs) * self.pixels_per_sec;
                        if ph_x >= lane_rect.left() && ph_x <= lane_rect.right() {
                            p.line_segment(
                                [
                                    egui::pos2(ph_x, lane_rect.top()),
                                    egui::pos2(ph_x, lane_rect.bottom()),
                                ],
                                egui::Stroke::new(1.75_f32, egui::Color32::from_rgb(248, 113, 113)),
                            );
                        }
                    });
                    ui.add_space(4.0);
                }
            });

            if changed {
                self.commit_project_change(prev_snapshot);
            }
        });
    }
}

fn draw_fx_slot_controls(ui: &mut egui::Ui, fx: &mut FxSlotConfig) -> bool {
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
            tr_high_db,
            tn_high_db,
            ..
        } => {
            changed |= ui
                .add(egui::Slider::new(tr_high_db, -12.0..=12.0).text("Trans High"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(tn_high_db, -12.0..=12.0).text("Tonal High"))
                .changed();
        }
        FxSlotConfig::DensityClipper {
            density,
            clip_drive_db,
            ..
        } => {
            changed |= ui
                .add(egui::Slider::new(density, 0.0..=1.0).text("Density"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(clip_drive_db, 0.0..=12.0).text("4x ADAA Clip"))
                .changed();
        }
        FxSlotConfig::SpaceVerb {
            decay_rt60_secs,
            wet_mix,
            ..
        } => {
            changed |= ui
                .add(egui::Slider::new(decay_rt60_secs, 0.2..=12.0).text("Decay (s)"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(wet_mix, 0.0..=1.0).text("Wet Mix"))
                .changed();
        }
        FxSlotConfig::VocalTune {
            retune_speed_ms,
            retune_amount,
            formant_shift_st,
            ..
        } => {
            changed |= ui
                .add(egui::Slider::new(retune_speed_ms, 0.0..=100.0).text("Speed (ms)"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(retune_amount, 0.0..=1.0).text("Amount"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(formant_shift_st, -12.0..=12.0).text("Formant (st)"))
                .changed();
        }
    }
    changed
}

impl eframe::App for ForgeDawApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if ctx.input(|i| i.key_pressed(egui::Key::Space))
            && !ctx.wants_keyboard_input()
            && let Some(ref eng) = self.audio
        {
            eng.toggle_play();
        }

        self.draw_transport_bar(ctx);
        self.draw_object_editor(ctx);
        self.draw_mixer_dock(ctx);
        self.draw_arrange_timeline(ctx);

        if self.audio.as_ref().is_some_and(LiveAudioEngine::is_playing) {
            ctx.request_repaint();
        }
    }
}
