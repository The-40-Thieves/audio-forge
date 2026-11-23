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
    AudioObject, ClipCache, DawProject, FadeCurve, FxSlotConfig, LiveAudioEngine, LiveRecorder,
    MidiClip, MidiNote, ObjectOrientedEngine, SynthPreset, TimelineMarker, TrackSend, UndoStack,
};

fn build_native_options(renderer: eframe::Renderer) -> eframe::NativeOptions {
    let wgpu_setup = eframe::egui_wgpu::WgpuSetupCreateNew {
        power_preference: eframe::wgpu::PowerPreference::HighPerformance,
        instance_descriptor: eframe::wgpu::InstanceDescriptor {
            backends: eframe::wgpu::Backends::PRIMARY,
            ..Default::default()
        },
        ..Default::default()
    };

    let wgpu_options = eframe::egui_wgpu::WgpuConfiguration {
        present_mode: eframe::wgpu::PresentMode::AutoVsync,
        desired_maximum_frame_latency: Some(2),
        wgpu_setup: eframe::egui_wgpu::WgpuSetup::CreateNew(wgpu_setup),
        ..Default::default()
    };

    eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1360.0, 840.0])
            .with_min_inner_size([960.0, 600.0])
            .with_title("ForgeStudio DAW — Object-Oriented Audio & Mastering"),
        renderer,
        hardware_acceleration: eframe::HardwareAcceleration::Preferred,
        vsync: true,
        multisampling: 4,
        wgpu_options,
        ..Default::default()
    }
}

fn detect_gpu_label(cc: &eframe::CreationContext<'_>) -> String {
    if let Some(ref rs) = cc.wgpu_render_state {
        let info = rs.adapter.get_info();
        return format!(
            "{} [{:?} | {:?}]",
            info.name, info.backend, info.device_type
        );
    }
    "OpenGL (glow fallback)".to_string()
}

fn main() -> eframe::Result {
    let primary = build_native_options(eframe::Renderer::Wgpu);
    match eframe::run_native(
        "ForgeStudio DAW",
        primary,
        Box::new(|cc| {
            configure_studio_theme(&cc.egui_ctx);
            let gpu_label = detect_gpu_label(cc);
            Ok(Box::new(ForgeDawApp::new(gpu_label)))
        }),
    ) {
        Ok(()) => Ok(()),
        Err(wgpu_err) => {
            eprintln!("wgpu init failed ({wgpu_err}), falling back to OpenGL (glow)");
            let fallback = build_native_options(eframe::Renderer::Glow);
            eframe::run_native(
                "ForgeStudio DAW",
                fallback,
                Box::new(|cc| {
                    configure_studio_theme(&cc.egui_ctx);
                    let gpu_label = detect_gpu_label(cc);
                    Ok(Box::new(ForgeDawApp::new(gpu_label)))
                }),
            )
        }
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SnapGrid {
    Off,
    Bar,
    Quarter,
    Eighth,
    Sixteenth,
}

impl SnapGrid {
    fn label(self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::Bar => "Bar",
            Self::Quarter => "1/4",
            Self::Eighth => "1/8",
            Self::Sixteenth => "1/16",
        }
    }

    fn snap_samples(self, raw_samples: u64, sample_rate: u32, bpm: f64, time_sig_num: u8) -> u64 {
        let beats_per_step = match self {
            Self::Off => return raw_samples,
            Self::Bar => f64::from(time_sig_num.max(1)),
            Self::Quarter => 1.0,
            Self::Eighth => 0.5,
            Self::Sixteenth => 0.25,
        };
        let step_secs = (60.0 / bpm.max(20.0)) * beats_per_step;
        let step_smp = (step_secs * f64::from(sample_rate.max(8_000))).max(1.0);
        (((raw_samples as f64) / step_smp).round() * step_smp).max(0.0) as u64
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DragMode {
    Move,
    TrimLeft,
    TrimRight,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InspectorTab {
    Object,
    PianoRoll,
    TrackBusFx,
}

struct ForgeDawApp {
    project: DawProject,
    undo: UndoStack,
    cache: ClipCache,
    audio: Option<LiveAudioEngine>,
    recorder: Option<LiveRecorder>,
    selected_track_idx: usize,
    selected_object_id: Option<u64>,
    selected_midi_clip_id: Option<u64>,
    inspector_tab: InspectorTab,
    active_drag: Option<(usize, u64, DragMode)>,
    drag_anchor_samples: u64,
    drag_anchor_x: f32,
    snap_grid: SnapGrid,
    auto_crossfade: bool,
    pixels_per_sec: f32,
    scroll_secs: f32,
    import_path_buf: String,
    project_path_buf: String,
    bounce_path_buf: String,
    gpu_backend_label: String,
    status_line: String,
    show_mixer: bool,
    pointer_down: bool,
    undo_gesture_active: bool,
}

impl ForgeDawApp {
    fn new(gpu_backend_label: String) -> Self {
        let project = DawProject::default();
        let cache = ClipCache::default();
        let (audio, status_line) = match LiveAudioEngine::start(&project, &cache) {
            Ok(eng) => {
                let msg = format!(
                    "Ready — Audio: {} ({} Hz) | GPU: {}",
                    eng.device_name(),
                    eng.sample_rate(),
                    gpu_backend_label
                );
                (Some(eng), msg)
            }
            Err(e) => (
                None,
                format!("Audio output unavailable ({e}) — Offline mode | GPU: {gpu_backend_label}"),
            ),
        };
        Self {
            project,
            undo: UndoStack::default(),
            cache,
            audio,
            recorder: None,
            selected_track_idx: 0,
            selected_object_id: None,
            selected_midi_clip_id: None,
            inspector_tab: InspectorTab::Object,
            active_drag: None,
            drag_anchor_samples: 0,
            drag_anchor_x: 0.0,
            snap_grid: SnapGrid::Quarter,
            auto_crossfade: true,
            pixels_per_sec: 120.0,
            scroll_secs: 0.0,
            import_path_buf: String::new(),
            project_path_buf: "session.forgeproj".to_string(),
            bounce_path_buf: "master_bounce.wav".to_string(),
            gpu_backend_label,
            status_line,
            show_mixer: true,
            pointer_down: false,
            undo_gesture_active: false,
        }
    }

    fn commit_project_change(&mut self, prev: DawProject) {
        if self.pointer_down {
            if !self.undo_gesture_active {
                self.undo.push(prev);
                self.undo_gesture_active = true;
            }
        } else {
            self.undo.push(prev);
            self.undo_gesture_active = false;
        }
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

    fn project_media_dir(&self) -> PathBuf {
        let proj = PathBuf::from(self.project_path_buf.trim());
        let parent = proj
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        let stem = proj
            .file_stem()
            .and_then(|s| s.to_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("forgestudio");
        parent.join(format!("{stem}_media"))
    }

    fn add_demo_clip(&mut self) {
        if self.project.tracks.is_empty() {
            self.project.add_track("01 Audio Track");
        }
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
            let id = self.project.alloc_id();
            let media_dir = self.project_media_dir();
            let _ = std::fs::create_dir_all(&media_dir);
            let path = media_dir.join(format!("demo_am7_{id}.wav"));
            if buf.save_wav(&path, 24).is_ok() {
                let _ = self.cache.load_clip_for_rate(&path, sr);
            } else {
                self.cache.insert_memory_clip(path.clone(), buf);
            }
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
            self.status_line = "Inserted 4s Demo Am7 Audio Object (saved to WAV)".to_string();
        }
    }

    fn import_wav_to_selected_track(&mut self) {
        let raw = self.import_path_buf.trim().to_string();
        if raw.is_empty() {
            self.status_line = "Enter a WAV/audio file path to import".to_string();
            return;
        }
        let path = PathBuf::from(&raw);
        match self
            .cache
            .load_clip_for_rate(&path, self.project.sample_rate)
        {
            Ok(buf) => {
                let prev = self.project.clone();
                if self.project.tracks.is_empty() {
                    self.project.add_track("01 Audio Track");
                }
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
                self.status_line = format!(
                    "Imported '{}' ({} Hz) onto track {}",
                    raw,
                    self.project.sample_rate,
                    tr_idx + 1
                );
            }
            Err(e) => {
                self.status_line = format!("Import failed: {e}");
            }
        }
    }

    fn toggle_record(&mut self) {
        if let Some(rec) = self.recorder.take() {
            let tr_idx = rec
                .track_idx()
                .min(self.project.tracks.len().saturating_sub(1));
            let start_pos = rec.start_timeline_samples();
            let id = self.project.alloc_id();
            let media_dir = self.project_media_dir();
            let out_wav = media_dir.join(format!("recording_{id}.wav"));
            match rec.stop_and_save(&out_wav, self.project.sample_rate) {
                Ok(buf) => {
                    let prev = self.project.clone();
                    let len = buf.num_frames() as u64;
                    self.cache.insert_memory_clip(out_wav.clone(), buf);
                    let mut obj = AudioObject::new(id, format!("Take {id}"), out_wav, len);
                    obj.timeline_start_samples = start_pos;
                    self.project.tracks[tr_idx].objects.push(obj);
                    self.selected_object_id = Some(id);
                    self.commit_project_change(prev);
                    self.status_line = format!(
                        "Recorded {:.2}s onto track {}",
                        (len as f64) / f64::from(self.project.sample_rate.max(1)),
                        tr_idx + 1
                    );
                }
                Err(e) => {
                    self.status_line = format!("Recording failed: {e}");
                }
            }
            if let Some(ref eng) = self.audio {
                eng.set_playing(false);
            }
        } else {
            let armed_idx = self
                .project
                .tracks
                .iter()
                .position(|t| t.record_armed)
                .unwrap_or(
                    self.selected_track_idx
                        .min(self.project.tracks.len().saturating_sub(1)),
                );
            let start_pos = self.playhead_samples();
            match LiveRecorder::start(armed_idx, start_pos) {
                Ok(rec) => {
                    self.recorder = Some(rec);
                    if let Some(ref eng) = self.audio {
                        eng.set_playing(true);
                    }
                    self.status_line = format!("Recording on track {}...", armed_idx + 1);
                }
                Err(e) => {
                    self.status_line = format!("Cannot start recording: {e}");
                }
            }
        }
    }

    fn split_selected_at_playhead(&mut self) {
        let Some(sel_id) = self.selected_object_id else {
            return;
        };
        let ph = self.playhead_samples();
        let prev = self.project.clone();
        let new_id = self.project.alloc_id();
        for tr in &mut self.project.tracks {
            if let Some(oi) = tr.objects.iter().position(|o| o.id == sel_id)
                && let Some((left, right)) = tr.objects[oi].split_at(ph, new_id)
            {
                tr.objects[oi] = left;
                tr.objects.insert(oi + 1, right);
                self.selected_object_id = Some(new_id);
                self.commit_project_change(prev);
                self.status_line = "Split object at playhead".to_string();
                return;
            }
        }
    }

    fn delete_selected_object(&mut self) {
        let Some(sel_id) = self.selected_object_id else {
            return;
        };
        let prev = self.project.clone();
        for tr in &mut self.project.tracks {
            if let Some(oi) = tr.objects.iter().position(|o| o.id == sel_id) {
                tr.objects.remove(oi);
                self.selected_object_id = None;
                self.commit_project_change(prev);
                self.status_line = "Deleted selected object".to_string();
                return;
            }
        }
    }

    fn duplicate_selected_object(&mut self) {
        let Some(sel_id) = self.selected_object_id else {
            return;
        };
        let prev = self.project.clone();
        let new_id = self.project.alloc_id();
        for tr in &mut self.project.tracks {
            if let Some(obj) = tr.objects.iter().find(|o| o.id == sel_id).cloned() {
                let mut dup = obj;
                dup.id = new_id;
                dup.name = format!("{} (Copy)", dup.name);
                dup.timeline_start_samples = dup.timeline_end_samples();
                tr.objects.push(dup);
                self.selected_object_id = Some(new_id);
                self.commit_project_change(prev);
                self.status_line = "Duplicated selected object".to_string();
                return;
            }
        }
    }

    fn add_marker_at_playhead(&mut self) {
        let prev = self.project.clone();
        let pos = self.playhead_samples();
        let idx = self.project.markers.len() + 1;
        self.project.markers.push(TimelineMarker {
            name: format!("M{idx}"),
            sample_pos: pos,
        });
        self.project.markers.sort_by_key(|m| m.sample_pos);
        self.commit_project_change(prev);
        self.status_line = format!("Added marker M{idx} at sample {pos}");
    }

    fn add_demo_midi_clip(&mut self) {
        let prev = self.project.clone();
        if self.project.tracks.is_empty() {
            self.project.add_track("01 Synth Track");
        }
        let id = self.project.alloc_id();
        let mut clip =
            MidiClip::demo_am7_progression(id, self.project.sample_rate, self.project.bpm);
        clip.timeline_start_samples = self.playhead_samples();
        let tr_idx = self
            .selected_track_idx
            .min(self.project.tracks.len().saturating_sub(1));
        self.project.tracks[tr_idx].midi_clips.push(clip);
        self.selected_midi_clip_id = Some(id);
        self.inspector_tab = InspectorTab::PianoRoll;
        self.commit_project_change(prev);
        self.status_line = format!(
            "Inserted 4-bar Am7 MIDI Progression onto track {} ({})",
            tr_idx + 1,
            self.project.tracks[tr_idx].synth_preset.label()
        );
    }

    fn export_stems(&mut self) {
        let out_dir = PathBuf::from("stems_export");
        match ObjectOrientedEngine::bounce_stems(&self.project, &self.cache, &out_dir) {
            Ok(paths) => {
                self.status_line = format!(
                    "Exported {} PDC-aligned 24-bit WAV stems into '{}'",
                    paths.len(),
                    out_dir.display()
                );
            }
            Err(e) => {
                self.status_line = format!("Stem export failed: {e}");
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
                if ui.button("Stop").clicked() {
                    if self.recorder.is_some() {
                        self.toggle_record();
                    }
                    if let Some(ref eng) = self.audio {
                        eng.stop();
                    }
                }
                let is_recording = self.recorder.is_some();
                if ui
                    .button(
                        egui::RichText::new(if is_recording { "Stop Rec" } else { "Rec" })
                            .strong()
                            .color(if is_recording {
                                egui::Color32::from_rgb(248, 113, 113)
                            } else {
                                egui::Color32::from_rgb(252, 165, 165)
                            }),
                    )
                    .clicked()
                {
                    self.toggle_record();
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
                egui::ComboBox::from_id_salt("snap_grid")
                    .width(56.0)
                    .selected_text(format!("Snap: {}", self.snap_grid.label()))
                    .show_ui(ui, |ui| {
                        for g in [
                            SnapGrid::Off,
                            SnapGrid::Bar,
                            SnapGrid::Quarter,
                            SnapGrid::Eighth,
                            SnapGrid::Sixteenth,
                        ] {
                            ui.selectable_value(&mut self.snap_grid, g, g.label());
                        }
                    });

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
                if ui.button("+ MIDI").clicked() {
                    self.add_demo_midi_clip();
                }
                if ui.button("+ Track").clicked() {
                    let prev = self.project.clone();
                    let n = self.project.tracks.len() + 1;
                    self.project.add_track(format!("{n:02} Audio Track"));
                    self.commit_project_change(prev);
                }
                if ui.button("+ Marker").clicked() {
                    self.add_marker_at_playhead();
                }
                ui.checkbox(&mut self.auto_crossfade, "XFade");
                if ui.button("Bounce Master").clicked() {
                    self.bounce_master();
                }
                if ui.button("Export Stems").clicked() {
                    self.export_stems();
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
                            let target_sr = self.project.sample_rate;
                            for tr in &self.project.tracks {
                                for obj in &tr.objects {
                                    let _ =
                                        self.cache.load_clip_for_rate(&obj.source_path, target_sr);
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

        // Interactive 16-step × 13-note (C4..=C5) Piano Roll Grid
        let sr = f64::from(self.project.sample_rate.max(8_000));
        let step_smp = (((60.0 / self.project.bpm.max(30.0)) * 0.25) * sr).round() as u64;

        ui.label(
            egui::RichText::new(format!(
                "Editing '{}' — click cells to toggle 16th-note events (C4–C5):",
                self.project.tracks[tr_idx].midi_clips[clip_idx].name
            ))
            .small()
            .color(egui::Color32::from_rgb(148, 163, 184)),
        );

        let note_names = [
            "C5", "B4", "A#4", "A4", "G#4", "G4", "F#4", "F4", "E4", "D#4", "D4", "C#4", "C4",
        ];
        let mut add_note: Option<(u64, u8)> = None;
        let mut remove_note_idx: Option<usize> = None;

        egui::ScrollArea::vertical().show(ui, |ui| {
            let clip = &self.project.tracks[tr_idx].midi_clips[clip_idx];
            for (row, name) in note_names.iter().enumerate() {
                let pitch = (72 - row) as u8;
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 2.0;
                    ui.add_sized(
                        [28.0, 16.0],
                        egui::Label::new(egui::RichText::new(*name).monospace().small()),
                    );
                    for step in 0..16_u64 {
                        let start = step * step_smp;
                        let existing_idx = clip.notes.iter().position(|n| {
                            n.pitch == pitch && n.start_samples.abs_diff(start) < step_smp / 2
                        });
                        let active = existing_idx.is_some();
                        let col = if active {
                            egui::Color32::from_rgb(192, 132, 252)
                        } else if step % 4 == 0 {
                            egui::Color32::from_rgb(38, 46, 62)
                        } else {
                            egui::Color32::from_rgb(24, 30, 42)
                        };
                        let btn = egui::Button::new("")
                            .fill(col)
                            .min_size(egui::vec2(15.0, 15.0));
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
                                let first_aux_id = self.project.aux_buses.first().map(|b| b.id);
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
                                        if let Some(aux_id) = first_aux_id {
                                            let mut send_db = tr
                                                .sends
                                                .iter()
                                                .find(|s| s.bus_id == aux_id)
                                                .map_or(-60.0, |s| s.gain_db);
                                            if ui
                                                .add(
                                                    egui::Slider::new(&mut send_db, -60.0..=6.0)
                                                        .text("Aux1"),
                                                )
                                                .changed()
                                            {
                                                if let Some(s) =
                                                    tr.sends.iter_mut().find(|s| s.bus_id == aux_id)
                                                {
                                                    s.gain_db = send_db;
                                                } else {
                                                    tr.sends.push(TrackSend {
                                                        bus_id: aux_id,
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
                let sr = f64::from(self.project.sample_rate.max(1));
                let max_scroll_secs =
                    ((self.project.total_length_samples() as f64 / sr) as f32 + 30.0).max(120.0);
                ui.add(
                    egui::Slider::new(&mut self.scroll_secs, 0.0..=max_scroll_secs)
                        .text("Scroll (s)"),
                );
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

                // Draw clickable Timeline Marker flags on the ruler
                for mk in &self.project.markers {
                    let mk_secs = (mk.sample_pos as f64 / sr) as f32;
                    let mx = ruler_rect.left() + (mk_secs - self.scroll_secs) * self.pixels_per_sec;
                    if mx >= ruler_rect.left() && mx <= ruler_rect.right() {
                        let flag_rect = egui::Rect::from_min_size(
                            egui::pos2(mx, ruler_rect.top() + 1.0),
                            egui::vec2(38.0, 14.0),
                        );
                        rp.rect_filled(flag_rect, 2.0, egui::Color32::from_rgb(14, 116, 144));
                        rp.text(
                            egui::pos2(mx + 3.0, ruler_rect.top() + 8.0),
                            egui::Align2::LEFT_CENTER,
                            &mk.name,
                            egui::FontId::monospace(9.5),
                            egui::Color32::WHITE,
                        );
                        if ruler_resp.clicked()
                            && let Some(pt) = ruler_resp.interact_pointer_pos()
                            && flag_rect.contains(pt)
                        {
                            self.set_playhead_samples(mk.sample_pos);
                        }
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
                                        if ui
                                            .selectable_label(
                                                self.project.tracks[ti].record_armed,
                                                egui::RichText::new("R").color(
                                                    if self.project.tracks[ti].record_armed {
                                                        egui::Color32::from_rgb(248, 113, 113)
                                                    } else {
                                                        egui::Color32::from_rgb(160, 170, 188)
                                                    },
                                                ),
                                            )
                                            .clicked()
                                        {
                                            self.project.tracks[ti].record_armed =
                                                !self.project.tracks[ti].record_armed;
                                        }
                                        if ui
                                            .selectable_label(
                                                self.project.tracks[ti].show_automation,
                                                egui::RichText::new("A").color(
                                                    if self.project.tracks[ti].show_automation {
                                                        egui::Color32::from_rgb(250, 204, 21)
                                                    } else {
                                                        egui::Color32::from_rgb(160, 170, 188)
                                                    },
                                                ),
                                            )
                                            .clicked()
                                        {
                                            self.project.tracks[ti].show_automation =
                                                !self.project.tracks[ti].show_automation;
                                            if self.project.tracks[ti].show_automation {
                                                self.project.tracks[ti].volume_auto.enabled = true;
                                                changed = true;
                                            }
                                        }
                                    });
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "{:+.1} dB",
                                            self.project.tracks[ti].gain_db
                                        ))
                                        .monospace()
                                        .small(),
                                    );
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

                            // Viewport-culled waveform peaks inside the object (O(visible width))
                            if let Some(peaks) = self
                                .cache
                                .get_peaks(&self.project.tracks[ti].objects[oi].source_path)
                            {
                                let mid_y = obj_rect.center().y;
                                let half_h = (obj_rect.height() * 0.42).max(4.0);
                                let step_px = 2.0_f32;
                                let n_cols = ((obj_rect.width() / step_px) as usize).max(1);
                                let c_start = (((lane_rect.left() - obj_rect.left()) / step_px)
                                    .floor()
                                    .max(0.0)
                                    as usize)
                                    .min(n_cols);
                                let c_end = (((lane_rect.right() - obj_rect.left()) / step_px)
                                    .ceil()
                                    .max(0.0)
                                    as usize)
                                    .min(n_cols);
                                let obj_len =
                                    self.project.tracks[ti].objects[oi].length_samples.max(1);
                                let src_off =
                                    self.project.tracks[ti].objects[oi].source_offset_samples;
                                for c in c_start..c_end {
                                    let frac = (c as f64) / (n_cols as f64);
                                    let smp =
                                        src_off.saturating_add((frac * (obj_len as f64)) as u64);
                                    let bucket = (smp as usize) / peaks.samples_per_bucket;
                                    if let Some(&(mn, mx)) = peaks.min_max.get(bucket) {
                                        let cx = obj_rect.left() + (c as f32) * step_px;
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

                            // Visual Fade-In / Fade-Out curve overlays
                            let fade_in_s = (self.project.tracks[ti].objects[oi].fade_in_samples
                                as f64
                                / sr) as f32;
                            if fade_in_s > 0.001 {
                                let fi_w = (fade_in_s * self.pixels_per_sec).min(obj_rect.width());
                                p.line_segment(
                                    [
                                        egui::pos2(obj_rect.left(), obj_rect.bottom()),
                                        egui::pos2(obj_rect.left() + fi_w, obj_rect.top()),
                                    ],
                                    egui::Stroke::new(
                                        1.5_f32,
                                        egui::Color32::from_rgba_unmultiplied(250, 204, 21, 200),
                                    ),
                                );
                            }
                            let fade_out_s = (self.project.tracks[ti].objects[oi].fade_out_samples
                                as f64
                                / sr) as f32;
                            if fade_out_s > 0.001 {
                                let fo_w = (fade_out_s * self.pixels_per_sec).min(obj_rect.width());
                                p.line_segment(
                                    [
                                        egui::pos2(obj_rect.right() - fo_w, obj_rect.top()),
                                        egui::pos2(obj_rect.right(), obj_rect.bottom()),
                                    ],
                                    egui::Stroke::new(
                                        1.5_f32,
                                        egui::Color32::from_rgba_unmultiplied(250, 204, 21, 200),
                                    ),
                                );
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

                            // Lock onto the grabbed object on drag_started (Move / TrimLeft / TrimRight)
                            if lane_resp.drag_started()
                                && !self.project.tracks[ti].show_automation
                                && let Some(pt) = lane_resp.interact_pointer_pos()
                                && obj_rect.contains(pt)
                            {
                                self.selected_track_idx = ti;
                                self.selected_object_id = Some(obj_id);
                                let mode = if (pt.x - obj_rect.left()).abs() <= 8.0 {
                                    DragMode::TrimLeft
                                } else if (obj_rect.right() - pt.x).abs() <= 8.0 {
                                    DragMode::TrimRight
                                } else {
                                    DragMode::Move
                                };
                                self.active_drag = Some((ti, obj_id, mode));
                                self.drag_anchor_samples =
                                    self.project.tracks[ti].objects[oi].timeline_start_samples;
                                self.drag_anchor_x = pt.x;
                            }
                            if lane_resp.clicked()
                                && !self.project.tracks[ti].show_automation
                                && let Some(pt) = lane_resp.interact_pointer_pos()
                                && obj_rect.contains(pt)
                            {
                                self.selected_track_idx = ti;
                                self.selected_object_id = Some(obj_id);
                            }
                            if let Some((drag_ti, drag_id, drag_mode)) = self.active_drag
                                && drag_ti == ti
                                && drag_id == obj_id
                                && lane_resp.dragged()
                            {
                                let dx = lane_resp.drag_delta().x;
                                let cum_dx = lane_resp
                                    .interact_pointer_pos()
                                    .map_or(dx, |pt| pt.x - self.drag_anchor_x);
                                if dx.abs() > 0.1 || matches!(drag_mode, DragMode::Move) {
                                    let delta_smp =
                                        (f64::from(dx / self.pixels_per_sec) * sr).round() as i64;
                                    let obj = &mut self.project.tracks[ti].objects[oi];
                                    match drag_mode {
                                        DragMode::Move => {
                                            let cum_smp = (f64::from(cum_dx / self.pixels_per_sec)
                                                * sr)
                                                .round()
                                                as i64;
                                            let raw = (self.drag_anchor_samples as i64 + cum_smp)
                                                .max(0)
                                                as u64;
                                            let snapped = self.snap_grid.snap_samples(
                                                raw,
                                                self.project.sample_rate,
                                                self.project.bpm,
                                                self.project.time_sig_num,
                                            );
                                            if snapped != obj.timeline_start_samples {
                                                obj.timeline_start_samples = snapped;
                                                changed = true;
                                            }
                                        }
                                        DragMode::TrimLeft => {
                                            let max_shift =
                                                obj.length_samples.saturating_sub(256) as i64;
                                            let min_shift = -(obj.source_offset_samples as i64)
                                                .min(obj.timeline_start_samples as i64);
                                            let shift = delta_smp.clamp(min_shift, max_shift);
                                            if shift >= 0 {
                                                let s = shift as u64;
                                                obj.timeline_start_samples =
                                                    obj.timeline_start_samples.saturating_add(s);
                                                obj.source_offset_samples =
                                                    obj.source_offset_samples.saturating_add(s);
                                                obj.length_samples =
                                                    obj.length_samples.saturating_sub(s);
                                            } else {
                                                let s = (-shift) as u64;
                                                obj.timeline_start_samples =
                                                    obj.timeline_start_samples.saturating_sub(s);
                                                obj.source_offset_samples =
                                                    obj.source_offset_samples.saturating_sub(s);
                                                obj.length_samples =
                                                    obj.length_samples.saturating_add(s);
                                            }
                                            changed = true;
                                        }
                                        DragMode::TrimRight => {
                                            let new_len = (obj.length_samples as i64 + delta_smp)
                                                .max(256)
                                                as u64;
                                            obj.length_samples = new_len;
                                            changed = true;
                                        }
                                    }
                                }
                            }
                        }

                        if lane_resp.drag_stopped() {
                            self.active_drag = None;
                        }

                        // Render MIDI Clips with miniature piano-roll note bars
                        for mc in &self.project.tracks[ti].midi_clips {
                            let ms = (mc.timeline_start_samples as f64 / sr) as f32;
                            let ml = (mc.length_samples as f64 / sr) as f32;
                            let mx0 =
                                lane_rect.left() + (ms - self.scroll_secs) * self.pixels_per_sec;
                            let mw = (ml * self.pixels_per_sec).max(20.0);
                            let mrect = egui::Rect::from_min_size(
                                egui::pos2(mx0, lane_rect.top() + 4.0),
                                egui::vec2(mw, lane_rect.height() - 8.0),
                            );
                            p.rect_filled(
                                mrect,
                                4.0,
                                egui::Color32::from_rgba_unmultiplied(168, 85, 247, 55),
                            );
                            p.rect_stroke(
                                mrect,
                                4.0,
                                egui::Stroke::new(1.25_f32, egui::Color32::from_rgb(192, 132, 252)),
                                egui::StrokeKind::Inside,
                            );
                            p.text(
                                egui::pos2(mrect.left() + 6.0, mrect.top() + 10.0),
                                egui::Align2::LEFT_CENTER,
                                format!("MIDI: {}", mc.name),
                                egui::FontId::proportional(10.5),
                                egui::Color32::from_rgb(233, 213, 255),
                            );
                            for note in &mc.notes {
                                let nx0 = mrect.left()
                                    + ((note.start_samples as f64 / sr) as f32)
                                        * self.pixels_per_sec;
                                let nw = (((note.length_samples as f64 / sr) as f32)
                                    * self.pixels_per_sec)
                                    .max(3.0);
                                let pitch_norm =
                                    ((f32::from(note.pitch) - 44.0) / 32.0).clamp(0.0, 1.0);
                                let ny =
                                    mrect.bottom() - 6.0 - pitch_norm * (mrect.height() - 22.0);
                                if nx0 + nw >= lane_rect.left() && nx0 <= lane_rect.right() {
                                    p.rect_filled(
                                        egui::Rect::from_min_size(
                                            egui::pos2(nx0, ny),
                                            egui::vec2(nw, 3.5),
                                        ),
                                        1.0,
                                        egui::Color32::from_rgb(216, 180, 254),
                                    );
                                }
                            }
                            if lane_resp.clicked()
                                && let Some(pt) = lane_resp.interact_pointer_pos()
                                && mrect.contains(pt)
                            {
                                self.selected_track_idx = ti;
                                self.selected_midi_clip_id = Some(mc.id);
                                self.inspector_tab = InspectorTab::PianoRoll;
                            }
                        }

                        // Highlight automatic Equal-Power Crossfade overlap zones
                        for (ov_start, ov_end) in self.project.tracks[ti].overlap_regions() {
                            let ox0 = lane_rect.left()
                                + (((ov_start as f64 / sr) as f32) - self.scroll_secs)
                                    * self.pixels_per_sec;
                            let ox1 = lane_rect.left()
                                + (((ov_end as f64 / sr) as f32) - self.scroll_secs)
                                    * self.pixels_per_sec;
                            if ox1 > lane_rect.left() && ox0 < lane_rect.right() {
                                let xrect = egui::Rect::from_min_max(
                                    egui::pos2(ox0.max(lane_rect.left()), lane_rect.top() + 4.0),
                                    egui::pos2(
                                        ox1.min(lane_rect.right()),
                                        lane_rect.bottom() - 4.0,
                                    ),
                                );
                                p.rect_filled(
                                    xrect,
                                    2.0,
                                    egui::Color32::from_rgba_unmultiplied(250, 204, 21, 45),
                                );
                                p.line_segment(
                                    [xrect.left_top(), xrect.right_bottom()],
                                    egui::Stroke::new(
                                        1.25_f32,
                                        egui::Color32::from_rgb(250, 204, 21),
                                    ),
                                );
                                p.line_segment(
                                    [xrect.left_bottom(), xrect.right_top()],
                                    egui::Stroke::new(
                                        1.25_f32,
                                        egui::Color32::from_rgb(250, 204, 21),
                                    ),
                                );
                            }
                        }

                        // Interactive Volume Automation Lane Overlay
                        if self.project.tracks[ti].show_automation {
                            let db_to_y = |db: f32| -> f32 {
                                let norm = ((db.clamp(-48.0, 12.0) + 48.0) / 60.0).clamp(0.0, 1.0);
                                lane_rect.bottom() - 6.0 - norm * (lane_rect.height() - 12.0)
                            };
                            let y_to_db = |y: f32| -> f32 {
                                let norm = ((lane_rect.bottom() - 6.0 - y)
                                    / (lane_rect.height() - 12.0).max(1.0))
                                .clamp(0.0, 1.0);
                                norm * 60.0 - 48.0
                            };

                            if (lane_resp.clicked() || lane_resp.dragged())
                                && let Some(pt) = lane_resp.interact_pointer_pos()
                                && lane_rect.contains(pt)
                            {
                                let secs = ((pt.x - lane_rect.left()) / self.pixels_per_sec)
                                    + self.scroll_secs;
                                let raw_smp = (f64::from(secs.max(0.0)) * sr).round() as u64;
                                let smp = self.snap_grid.snap_samples(
                                    raw_smp,
                                    self.project.sample_rate,
                                    self.project.bpm,
                                    self.project.time_sig_num,
                                );
                                let db = y_to_db(pt.y);
                                self.project.tracks[ti].volume_auto.enabled = true;
                                self.project.tracks[ti].volume_auto.upsert_point(smp, db);
                                changed = true;
                            }

                            let pts = &self.project.tracks[ti].volume_auto.points;
                            let mut screen_pts = Vec::with_capacity(pts.len());
                            for ap in pts {
                                let ax = lane_rect.left()
                                    + (((ap.sample_pos as f64 / sr) as f32) - self.scroll_secs)
                                        * self.pixels_per_sec;
                                let ay = db_to_y(ap.value);
                                screen_pts.push(egui::pos2(ax, ay));
                            }
                            if screen_pts.len() >= 2 {
                                p.add(egui::Shape::line(
                                    screen_pts.clone(),
                                    egui::Stroke::new(
                                        2.0_f32,
                                        egui::Color32::from_rgb(250, 204, 21),
                                    ),
                                ));
                            }
                            for sp in screen_pts {
                                if sp.x >= lane_rect.left() && sp.x <= lane_rect.right() {
                                    p.circle_filled(sp, 4.0, egui::Color32::from_rgb(250, 204, 21));
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
                if self.auto_crossfade {
                    for tr in &mut self.project.tracks {
                        tr.auto_crossfade_overlaps();
                    }
                }
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

impl eframe::App for ForgeDawApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.pointer_down = ctx.input(|i| i.pointer.primary_down());
        if !self.pointer_down {
            self.undo_gesture_active = false;
        }
        // Drag-and-drop audio file import directly onto the DAW window
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });
        for path in dropped {
            self.import_path_buf = path.to_string_lossy().into_owned();
            self.import_wav_to_selected_track();
        }

        if !ctx.wants_keyboard_input() {
            let (space, split, marker, del, undo, redo, dup) = ctx.input(|i| {
                (
                    i.key_pressed(egui::Key::Space),
                    i.key_pressed(egui::Key::S) && !i.modifiers.ctrl && !i.modifiers.command,
                    i.key_pressed(egui::Key::M) && !i.modifiers.ctrl && !i.modifiers.command,
                    i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace),
                    i.key_pressed(egui::Key::Z) && (i.modifiers.ctrl || i.modifiers.command),
                    i.key_pressed(egui::Key::Y) && (i.modifiers.ctrl || i.modifiers.command),
                    i.key_pressed(egui::Key::D) && (i.modifiers.ctrl || i.modifiers.command),
                )
            });
            if space && let Some(ref eng) = self.audio {
                eng.toggle_play();
            }
            if split {
                self.split_selected_at_playhead();
            }
            if marker {
                self.add_marker_at_playhead();
            }
            if del {
                self.delete_selected_object();
            }
            if undo && let Some(prev) = self.undo.undo(self.project.clone()) {
                self.project = prev;
                self.sync_audio_engine();
            }
            if redo && let Some(next) = self.undo.redo(self.project.clone()) {
                self.project = next;
                self.sync_audio_engine();
            }
            if dup {
                self.duplicate_selected_object();
            }
        }

        self.draw_transport_bar(ctx);
        self.draw_object_editor(ctx);
        self.draw_mixer_dock(ctx);
        self.draw_arrange_timeline(ctx);

        if self.audio.as_ref().is_some_and(LiveAudioEngine::is_playing) || self.recorder.is_some() {
            ctx.request_repaint();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snap_grid_quantizes_beats_and_bars_accurately() {
        // 120 BPM @ 48 kHz -> 1 quarter beat = 24,000 samples, 1 bar (4/4) = 96,000 samples
        assert_eq!(SnapGrid::Off.snap_samples(13_333, 48_000, 120.0, 4), 13_333);
        assert_eq!(
            SnapGrid::Quarter.snap_samples(25_500, 48_000, 120.0, 4),
            24_000
        );
        assert_eq!(
            SnapGrid::Eighth.snap_samples(35_000, 48_000, 120.0, 4),
            36_000
        );
        assert_eq!(SnapGrid::Bar.snap_samples(90_000, 48_000, 120.0, 4), 96_000);
    }

    #[test]
    fn native_gpu_options_enable_wgpu_msaa_and_vsync() {
        let opts = build_native_options(eframe::Renderer::Wgpu);
        assert_eq!(opts.renderer, eframe::Renderer::Wgpu);
        assert_eq!(opts.multisampling, 4);
        assert!(opts.vsync);
        assert_eq!(
            opts.hardware_acceleration,
            eframe::HardwareAcceleration::Preferred
        );
    }

    #[test]
    fn gesture_coalesced_undo_records_single_snapshot_per_drag() {
        let mut app = ForgeDawApp {
            project: DawProject::default(),
            undo: UndoStack::default(),
            cache: ClipCache::default(),
            audio: None,
            recorder: None,
            selected_track_idx: 0,
            selected_object_id: None,
            selected_midi_clip_id: None,
            inspector_tab: InspectorTab::Object,
            active_drag: None,
            drag_anchor_samples: 0,
            drag_anchor_x: 0.0,
            snap_grid: SnapGrid::Quarter,
            auto_crossfade: true,
            pixels_per_sec: 120.0,
            scroll_secs: 0.0,
            import_path_buf: String::new(),
            project_path_buf: "session.forgeproj".to_string(),
            bounce_path_buf: "master_bounce.wav".to_string(),
            gpu_backend_label: "Test".to_string(),
            status_line: String::new(),
            show_mixer: true,
            pointer_down: true,
            undo_gesture_active: false,
        };
        let initial_gain = app.project.master_gain_db;
        // Simulate 60 frames of continuous slider drag
        for step in 1..=60 {
            let prev = app.project.clone();
            app.project.master_gain_db = initial_gain - (step as f32) * 0.1;
            app.commit_project_change(prev);
        }
        // Release pointer
        app.pointer_down = false;
        app.undo_gesture_active = false;

        // Single undo restores the exact pre-drag gain, and stack is now empty
        let restored = app
            .undo
            .undo(app.project.clone())
            .expect("should have 1 undo snapshot");
        assert!((restored.master_gain_db - initial_gain).abs() < 1e-6);
        assert!(app.undo.undo(restored).is_none());
    }
}
