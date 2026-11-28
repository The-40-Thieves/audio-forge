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

mod fx_ui;
mod gpu;
mod inspector;
mod io_jobs;
mod mixer;
mod timeline;

use eframe::egui;
use forge_core::AudioBuffer;
use forge_engine::{
    AudioObject, BeatGridReport, ClipCache, DawProject, FxSlotConfig, LiveAudioEngine,
    LiveRecorder, MidiClip, TimelineMarker, UndoStack,
};
use gpu::{build_native_options, configure_studio_theme, detect_gpu_label};
use io_jobs::{IoJobOutcome, unique_job_media_dir};
use timeline::{AutoLaneMode, DragTarget};

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

    #[cfg(test)]
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

    fn snap_samples_for_project(self, raw_samples: u64, project: &DawProject) -> u64 {
        let step_beats = match self {
            Self::Off => return raw_samples,
            Self::Bar => f64::from(project.time_sig_num.max(1)),
            Self::Quarter => 1.0,
            Self::Eighth => 0.5,
            Self::Sixteenth => 0.25,
        };
        let sr = f64::from(project.sample_rate.max(8_000));
        let raw_secs = (raw_samples as f64) / sr;
        let raw_beat = project.secs_to_beat(raw_secs);
        let snapped_beat = (raw_beat / step_beats).round().max(0.0) * step_beats;
        (project.beat_to_secs(snapped_beat) * sr).round().max(0.0) as u64
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
    AiStudio,
}

pub(crate) enum AiJobOutcome {
    Summary(String),
    TempoMap(BeatGridReport),
    ProducedStem {
        track_title: String,
        obj_name: String,
        path: PathBuf,
        start_samples: u64,
        length_samples: u64,
        time_stretch: f32,
        pitch_semitones: f32,
        summary: String,
    },
}

pub(crate) fn stage_selected_object_wav(
    src_path: &std::path::Path,
    out_dir: &std::path::Path,
    project_sr: u32,
    source_offset_samples: u64,
    length_samples: u64,
    time_stretch: f32,
) -> Result<PathBuf, String> {
    let raw = AudioBuffer::load(src_path).map_err(|e| format!("Failed to read audio: {e}"))?;
    let full_frames = raw.num_frames();
    let sr_ratio = f64::from(raw.sample_rate.max(1)) / f64::from(project_sr.max(8_000));
    let stretch = f64::from(time_stretch.clamp(0.5, 2.0));
    let start_src = ((source_offset_samples as f64) * sr_ratio).round().max(0.0) as usize;
    let needed_src = ((length_samples.max(1) as f64) * stretch * sr_ratio)
        .round()
        .max(1.0) as usize;
    if source_offset_samples == 0 && needed_src >= full_frames {
        return Ok(src_path.to_path_buf());
    }
    let clamped_start = start_src.min(full_frames);
    let end_src = clamped_start.saturating_add(needed_src).min(full_frames);
    if end_src <= clamped_start {
        return Err("Selected audio object slice has zero source frames".to_string());
    }
    let sliced_ch: Vec<Vec<f64>> = raw
        .channels
        .iter()
        .map(|ch| ch[clamped_start..end_src].to_vec())
        .collect();
    let sliced = AudioBuffer::from_channels(sliced_ch, raw.sample_rate)
        .map_err(|e| format!("Failed to slice audio object: {e}"))?;
    std::fs::create_dir_all(out_dir).map_err(|e| format!("Failed to create AI dir: {e}"))?;
    let staged_path = out_dir.join("staged_selection.wav");
    sliced
        .save_wav(&staged_path, 24)
        .map_err(|e| format!("Failed to save staged slice: {e}"))?;
    Ok(staged_path)
}

struct ForgeDawApp {
    pub(crate) project: DawProject,
    pub(crate) undo: UndoStack,
    pub(crate) cache: ClipCache,
    pub(crate) audio: Option<LiveAudioEngine>,
    pub(crate) recorder: Option<LiveRecorder>,
    pub(crate) selected_track_idx: usize,
    pub(crate) selected_object_id: Option<u64>,
    pub(crate) selected_midi_clip_id: Option<u64>,
    pub(crate) inspector_tab: InspectorTab,
    pub(crate) active_drag: Option<(usize, DragTarget, DragMode)>,
    pub(crate) drag_anchor_samples: u64,
    pub(crate) drag_anchor_x: f32,
    pub(crate) snap_grid: SnapGrid,
    pub(crate) auto_crossfade: bool,
    pub(crate) auto_lane_mode: AutoLaneMode,
    pub(crate) pixels_per_sec: f32,
    pub(crate) scroll_secs: f32,
    pub(crate) import_path_buf: String,
    pub(crate) project_path_buf: String,
    pub(crate) bounce_path_buf: String,
    pub(crate) gpu_backend_label: String,
    pub(crate) status_line: String,
    pub(crate) show_mixer: bool,
    pub(crate) pointer_down: bool,
    pub(crate) undo_gesture_active: bool,
    pub(crate) piano_roll_base_octave: u8,
    pub(crate) piano_roll_bar: u64,
    pub(crate) stem_split_rx:
        Option<std::sync::mpsc::Receiver<Result<(u64, forge_worker::SeparationReport), String>>>,
    pub(crate) ai_job_rx: Option<std::sync::mpsc::Receiver<Result<AiJobOutcome, String>>>,
    pub(crate) ai_job_label: String,
    pub(crate) ai_audit_summary: String,
    pub(crate) io_job_rx: Option<std::sync::mpsc::Receiver<Result<IoJobOutcome, String>>>,
    pub(crate) io_job_label: String,
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
        Self::from_parts(project, cache, audio, gpu_backend_label, status_line)
    }

    fn from_parts(
        project: DawProject,
        cache: ClipCache,
        audio: Option<LiveAudioEngine>,
        gpu_backend_label: String,
        status_line: String,
    ) -> Self {
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
            auto_lane_mode: AutoLaneMode::Volume,
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
            piano_roll_base_octave: 3,
            piano_roll_bar: 0,
            stem_split_rx: None,
            ai_job_rx: None,
            ai_job_label: String::new(),
            ai_audit_summary: String::new(),
            io_job_rx: None,
            io_job_label: String::new(),
        }
    }

    pub(crate) fn commit_project_change(&mut self, prev: DawProject) {
        if self.pointer_down || self.undo_gesture_active {
            if !self.undo_gesture_active {
                self.undo.push(prev);
                self.undo_gesture_active = true;
            }
        } else {
            self.undo.push(prev);
        }
        self.sync_audio_engine();
    }

    fn sync_audio_engine(&self) {
        if let Some(ref eng) = self.audio {
            eng.reload_project(&self.project, &self.cache);
        }
    }

    pub(crate) fn playhead_samples(&self) -> u64 {
        self.audio
            .as_ref()
            .map_or(0, LiveAudioEngine::playhead_samples)
    }

    pub(crate) fn set_playhead_samples(&self, pos: u64) {
        if let Some(ref eng) = self.audio {
            eng.seek(pos);
        }
    }

    pub(crate) fn project_media_dir(&self) -> PathBuf {
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
        self.start_import_audio_job();
    }

    fn toggle_record(&mut self) {
        if self.recorder.is_some() {
            if self.is_io_busy() {
                self.status_line = format!(
                    "Wait for '{}' to finish before stopping recording (take preserved)",
                    self.io_job_label
                );
                return;
            }
            if let Some(rec) = self.recorder.take()
                && let Err(rec) = self.start_save_recording_job(rec)
            {
                self.recorder = Some(*rec);
            }
        } else {
            if self.project.tracks.is_empty() {
                self.status_line = "No tracks available for recording".to_string();
                return;
            }
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

    pub(crate) fn start_stem_split_for_object(&mut self, obj_id: u64) {
        if self.stem_split_rx.is_some() {
            self.status_line = "Stem separation already running in background...".to_string();
            return;
        }
        let Some((src_path, start_pos, src_offset, obj_len, time_stretch, obj_name)) = self
            .project
            .tracks
            .iter()
            .flat_map(|t| t.objects.iter())
            .find(|o| o.id == obj_id)
            .map(|o| {
                (
                    o.source_path.clone(),
                    o.timeline_start_samples,
                    o.source_offset_samples,
                    o.length_samples,
                    o.time_stretch,
                    o.name.clone(),
                )
            })
        else {
            return;
        };
        let job_id = self.project.alloc_id();
        let out_dir = unique_job_media_dir(&self.project_media_dir(), "stems", obj_id, job_id);
        let proj_sr = self.project.sample_rate;
        let (tx, rx) = std::sync::mpsc::channel();
        self.stem_split_rx = Some(rx);
        self.status_line = format!("Splitting '{obj_name}' (6-Stem Studio RoFormer cascade)...");

        std::thread::spawn(move || {
            let mut cfg = forge_worker::SeparationConfig::from_env();
            cfg.stems = 6;
            cfg.quality = forge_worker::SeparationQuality::Studio;
            cfg.dereverb = true;
            let res = stage_selected_object_wav(
                &src_path,
                &out_dir,
                proj_sr,
                src_offset,
                obj_len,
                time_stretch,
            )
            .and_then(|staged| {
                forge_worker::separate_stems(&staged, &out_dir, &cfg)
                    .map(|report| (start_pos, report))
                    .map_err(|e| e.to_string())
            });
            let _ = tx.send(res);
        });
    }

    fn poll_stem_split_result(&mut self) {
        let Some(ref rx) = self.stem_split_rx else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok((start_pos, report))) => {
                self.stem_split_rx = None;
                let prev = self.project.clone();
                let target_sr = self.project.sample_rate;
                let mut added = 0_usize;
                let mut failed = 0_usize;
                for (label, maybe_path) in report.bundle.audio_stems() {
                    if let Some(stem_path) = maybe_path {
                        match self.cache.load_clip_for_rate(stem_path, target_sr) {
                            Ok(buf) => {
                                let len = buf.num_frames() as u64;
                                let tr_idx = self.project.tracks.len();
                                self.project.add_track(format!("Stem: {label}"));
                                let id = self.project.alloc_id();
                                let mut obj = AudioObject::new(id, label, stem_path.clone(), len);
                                obj.timeline_start_samples = start_pos;
                                self.project.tracks[tr_idx].objects.push(obj);
                                added += 1;
                            }
                            Err(_) => {
                                failed += 1;
                            }
                        }
                    }
                }
                if added > 0 {
                    self.commit_project_change(prev);
                }
                self.status_line = if failed > 0 {
                    format!(
                        "Stem split complete ({}, {} stems imported, {} failed to load)",
                        report.backend.as_str(),
                        added,
                        failed
                    )
                } else {
                    format!(
                        "Stem split complete ({}, {} stems imported)",
                        report.backend.as_str(),
                        added
                    )
                };
            }
            Ok(Err(err)) => {
                self.stem_split_rx = None;
                self.status_line = format!("Stem split failed: {err}");
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.stem_split_rx = None;
                self.status_line = "Stem split worker terminated unexpectedly".to_string();
            }
        }
    }

    pub(crate) fn start_ai_studio_job<F>(&mut self, job_label: &str, work: F)
    where
        F: FnOnce(PathBuf, PathBuf, u64, f64) -> Result<AiJobOutcome, String> + Send + 'static,
    {
        if self.ai_job_rx.is_some() {
            return;
        }
        let Some(obj_id) = self.selected_object_id else {
            self.status_line = "Select an audio object before running AI Track Studio".to_string();
            return;
        };
        let Some(found) = self
            .project
            .tracks
            .iter()
            .flat_map(|t| t.objects.iter())
            .find(|o| o.id == obj_id)
            .map(|o| {
                (
                    o.source_path.clone(),
                    o.timeline_start_samples,
                    o.source_offset_samples,
                    o.length_samples,
                    o.time_stretch,
                    o.pitch_semitones,
                )
            })
        else {
            return;
        };
        let (src_path, start_smp, src_offset, obj_len, time_stretch, pitch_st) = found;
        let job_id = self.project.alloc_id();
        let out_dir = unique_job_media_dir(&self.project_media_dir(), "ai_studio", obj_id, job_id);
        let bpm_hint = self.project.bpm;
        let proj_sr = self.project.sample_rate;
        let (tx, rx) = std::sync::mpsc::channel();
        self.ai_job_rx = Some(rx);
        self.ai_job_label = job_label.to_string();
        self.status_line = format!("Running AI Studio: {job_label}...");
        std::thread::spawn(move || {
            let outcome = stage_selected_object_wav(
                &src_path,
                &out_dir,
                proj_sr,
                src_offset,
                obj_len,
                time_stretch,
            )
            .and_then(|staged| work(staged, out_dir, start_smp, bpm_hint))
            .map(|mut res| {
                if let AiJobOutcome::ProducedStem {
                    length_samples,
                    time_stretch: ts,
                    pitch_semitones: ps,
                    ..
                } = &mut res
                {
                    *length_samples = obj_len;
                    *ts = time_stretch;
                    *ps = pitch_st;
                }
                res
            });
            let _ = tx.send(outcome);
        });
    }

    pub(crate) fn apply_ai_defizz_preset(&mut self) {
        let Some(obj_id) = self.selected_object_id else {
            self.status_line = "Select an audio object to apply AI De-Fizz".to_string();
            return;
        };
        let prev = self.project.clone();
        let defizz = FxSlotConfig::OpenSoothe {
            bypass: false,
            sensitivity_db: 2.5,
            max_depth_db: 12.0,
            sharpness_bins: 12,
            low_latency: true,
            tilt_db_per_oct: -0.5,
            ultrasonic_lpf_hz: 16_500.0,
            mid_side_balance: 0.25,
            stereo_link: 0.75,
            hard_mode: true,
        };
        if let Some(obj) = self.project.object_mut(obj_id) {
            if let Some(slot) = obj
                .fx_chain
                .iter_mut()
                .find(|fx| matches!(fx, FxSlotConfig::OpenSoothe { .. }))
            {
                *slot = defizz;
            } else {
                obj.fx_chain.push(defizz);
            }
        }
        if !self
            .project
            .master_inserts
            .iter()
            .any(|fx| matches!(fx, FxSlotConfig::MasterLimiter { .. }))
        {
            self.project
                .master_inserts
                .push(FxSlotConfig::default_master_limiter());
        }
        self.commit_project_change(prev);
        self.ai_audit_summary =
            "Applied AI De-Fizz (2.5 dB sens, 12 dB depth, 16.5 kHz ULPF, Hard Mode) + Master Limiter"
                .to_string();
        self.status_line = "Applied AI De-Fizz preset to selected object".to_string();
    }

    fn poll_ai_job_result(&mut self) {
        let Some(ref rx) = self.ai_job_rx else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(outcome)) => {
                self.ai_job_rx = None;
                self.ai_job_label.clear();
                match outcome {
                    AiJobOutcome::Summary(msg) => {
                        self.ai_audit_summary = msg.clone();
                        self.status_line = msg;
                    }
                    AiJobOutcome::TempoMap(rep) => {
                        let prev = self.project.clone();
                        self.project.bpm = rep.median_bpm.clamp(30.0, 300.0);
                        self.project.tempo_map = rep.tempo_map.clone();
                        let nodes = self.project.tempo_map.len();
                        let summary = format!(
                            "Tempo map ({}): {:.2} BPM, drift {:.2} BPM, {} nodes",
                            rep.backend, rep.median_bpm, rep.max_bpm_drift, nodes
                        );
                        self.commit_project_change(prev);
                        self.ai_audit_summary = summary.clone();
                        self.status_line = summary;
                    }
                    AiJobOutcome::ProducedStem {
                        track_title,
                        obj_name,
                        path,
                        start_samples,
                        length_samples,
                        time_stretch,
                        pitch_semitones,
                        summary,
                    } => {
                        let target_sr = self.project.sample_rate;
                        match self.cache.load_clip_for_rate(&path, target_sr) {
                            Ok(buf) => {
                                let prev = self.project.clone();
                                let tr_idx = self.project.tracks.len();
                                self.project.add_track(track_title);
                                let id = self.project.alloc_id();
                                let eff_stretch = f64::from(time_stretch.clamp(0.5, 2.0));
                                let max_timeline =
                                    ((buf.num_frames() as f64) / eff_stretch).round().max(1.0)
                                        as u64;
                                let bounded_len = length_samples.max(1).min(max_timeline);
                                let mut obj = AudioObject::new(id, obj_name, path, bounded_len);
                                obj.timeline_start_samples = start_samples;
                                obj.source_offset_samples = 0;
                                obj.time_stretch = time_stretch;
                                obj.pitch_semitones = pitch_semitones;
                                self.project.tracks[tr_idx].objects.push(obj);
                                self.selected_track_idx = tr_idx;
                                self.selected_object_id = Some(id);
                                self.commit_project_change(prev);
                                self.ai_audit_summary = summary.clone();
                                self.status_line = summary;
                            }
                            Err(err) => {
                                self.status_line = format!("Failed to import AI result: {err}");
                            }
                        }
                    }
                }
            }
            Ok(Err(err)) => {
                self.ai_job_rx = None;
                self.ai_job_label.clear();
                self.status_line = format!("AI Studio job failed: {err}");
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.ai_job_rx = None;
                self.ai_job_label.clear();
                self.status_line =
                    "AI Studio background thread terminated unexpectedly".to_string();
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

    pub(crate) fn add_demo_midi_clip(&mut self) {
        let prev = self.project.clone();
        if self.project.tracks.is_empty() {
            self.project.add_track("01 Synth Track");
        }
        let id = self.project.alloc_id();
        let mut clip =
            MidiClip::demo_am7_progression(id, self.project.sample_rate, self.project.bpm);
        let start_smp = self.playhead_samples();
        clip.timeline_start_samples = start_smp;
        let sr = f64::from(self.project.sample_rate.max(8_000));
        clip.timeline_start_beats = Some(self.project.secs_to_beat((start_smp as f64) / sr));
        let tr_idx = self
            .selected_track_idx
            .min(self.project.tracks.len().saturating_sub(1));
        self.project.tracks[tr_idx].midi_clips.push(clip);
        self.selected_object_id = None;
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
        self.start_export_stems_job();
    }

    fn bounce_master(&mut self) {
        self.start_bounce_master_job();
    }

    fn delete_selected_midi_clip(&mut self) {
        let Some(sel_id) = self.selected_midi_clip_id else {
            return;
        };
        let prev = self.project.clone();
        for tr in &mut self.project.tracks {
            if let Some(mi) = tr.midi_clips.iter().position(|c| c.id == sel_id) {
                tr.midi_clips.remove(mi);
                self.selected_midi_clip_id = None;
                self.commit_project_change(prev);
                self.status_line = "Deleted selected MIDI clip".to_string();
                return;
            }
        }
    }

    fn draw_transport_bar(&mut self, ui: &mut egui::Ui) {
        egui::Panel::top("transport_bar").show(ui, |ui| {
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
                    if self.recorder.is_none()
                        && let Some(ref eng) = self.audio
                    {
                        eng.stop();
                    }
                }
                let io_busy = self.is_io_busy();
                let is_recording = self.recorder.is_some();
                if ui
                    .add_enabled(
                        !io_busy,
                        egui::Button::new(
                            egui::RichText::new(if is_recording { "Stop Rec" } else { "Rec" })
                                .strong()
                                .color(if is_recording {
                                    egui::Color32::from_rgb(248, 113, 113)
                                } else {
                                    egui::Color32::from_rgb(252, 165, 165)
                                }),
                        ),
                    )
                    .on_disabled_hover_text("Wait for background I/O to finish before saving take")
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
                if self.project.loop_enabled {
                    let sr = f64::from(self.project.sample_rate.max(8_000));
                    let mut start_s = (self.project.loop_start_samples as f64) / sr;
                    let mut end_s = (self.project.loop_end_samples as f64) / sr;
                    let s_changed = ui
                        .add(
                            egui::DragValue::new(&mut start_s)
                                .speed(0.1)
                                .range(0.0..=3599.75)
                                .suffix("s"),
                        )
                        .changed();
                    ui.label("–");
                    let e_changed = ui
                        .add(
                            egui::DragValue::new(&mut end_s)
                                .speed(0.1)
                                .range((start_s + 0.25)..=3600.0)
                                .suffix("s"),
                        )
                        .changed();
                    if s_changed || e_changed {
                        let prev = self.project.clone();
                        self.project.loop_start_samples = (start_s.max(0.0) * sr).round() as u64;
                        self.project.loop_end_samples =
                            (end_s.max(start_s + 0.25) * sr).round() as u64;
                        self.commit_project_change(prev);
                    }
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
                let beats = self.project.secs_to_beat(secs);
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
                    let prev = self.project.clone();
                    self.project.bpm = bpm;
                    self.project.rescale_musical_midi_clips();
                    self.commit_project_change(prev);
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
                if ui
                    .add_enabled(!io_busy, egui::Button::new("Bounce Master"))
                    .clicked()
                {
                    self.bounce_master();
                }
                if ui
                    .add_enabled(!io_busy, egui::Button::new("Export Stems"))
                    .clicked()
                {
                    self.export_stems();
                }
                ui.checkbox(&mut self.show_mixer, "Mixer");
            });

            // Second compact I/O bar: Import WAV + Save/Load .forgeproj + Bounce path
            ui.horizontal(|ui| {
                let io_busy = self.is_io_busy();
                ui.label(egui::RichText::new("WAV:").small());
                ui.add(
                    egui::TextEdit::singleline(&mut self.import_path_buf)
                        .desired_width(185.0)
                        .hint_text("path/to/audio.wav"),
                );
                if ui
                    .add_enabled(!io_busy, egui::Button::new("…").small())
                    .on_hover_text("Browse audio file")
                    .clicked()
                    && let Some(path) = rfd::FileDialog::new()
                        .add_filter("Audio", &["wav", "flac", "mp3", "ogg", "m4a"])
                        .pick_file()
                {
                    self.import_path_buf = path.display().to_string();
                    self.import_wav_to_selected_track();
                }
                if ui
                    .add_enabled(!io_busy, egui::Button::new("Import").small())
                    .clicked()
                {
                    self.import_wav_to_selected_track();
                }
                ui.separator();
                ui.label(egui::RichText::new("Project:").small());
                ui.add(egui::TextEdit::singleline(&mut self.project_path_buf).desired_width(135.0));
                if ui
                    .add_enabled(!io_busy, egui::Button::new("…").small())
                    .on_hover_text("Open .forgeproj")
                    .clicked()
                    && let Some(path) = rfd::FileDialog::new()
                        .add_filter("ForgeStudio Project", &["forgeproj"])
                        .pick_file()
                {
                    self.project_path_buf = path.display().to_string();
                }
                if ui
                    .add_enabled(!io_busy, egui::Button::new("Save").small())
                    .clicked()
                {
                    self.start_save_project_job();
                }
                if ui
                    .add_enabled(!io_busy, egui::Button::new("Load").small())
                    .clicked()
                {
                    self.start_load_project_job();
                }
                ui.separator();
                ui.label(egui::RichText::new("Out:").small());
                ui.add(egui::TextEdit::singleline(&mut self.bounce_path_buf).desired_width(130.0));
                if ui
                    .small_button("…")
                    .on_hover_text("Choose master bounce WAV")
                    .clicked()
                    && let Some(path) = rfd::FileDialog::new()
                        .add_filter("WAV Audio", &["wav"])
                        .set_file_name("master_bounce.wav")
                        .save_file()
                {
                    self.bounce_path_buf = path.display().to_string();
                }
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
}

impl eframe::App for ForgeDawApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.poll_stem_split_result();
        self.poll_ai_job_result();
        self.poll_io_job_result();
        self.pointer_down = ui.input(|i| i.pointer.primary_down());
        // Drag-and-drop audio file import directly onto the DAW window
        let dropped: Vec<PathBuf> = ui.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .map(|f| f.path().to_path_buf())
                .collect()
        });
        for path in dropped {
            self.import_path_buf = path.to_string_lossy().into_owned();
            self.import_wav_to_selected_track();
        }

        if !ui.egui_wants_keyboard_input() {
            let (space, split, marker, del, undo, redo, dup) = ui.input(|i| {
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
                if self.selected_object_id.is_some() {
                    self.delete_selected_object();
                } else if self.selected_midi_clip_id.is_some() {
                    self.delete_selected_midi_clip();
                }
            }
            if undo && let Some(prev) = self.undo.undo(self.project.clone()) {
                self.project = prev;
                self.undo_gesture_active = false;
                self.sync_audio_engine();
            }
            if redo && let Some(next) = self.undo.redo(self.project.clone()) {
                self.project = next;
                self.undo_gesture_active = false;
                self.sync_audio_engine();
            }
            if dup {
                self.duplicate_selected_object();
            }
        }

        self.draw_transport_bar(ui);
        self.draw_object_editor(ui);
        self.draw_mixer_dock(ui);
        self.draw_arrange_timeline(ui);

        // Reset drag-coalescing AFTER widgets finish so a slider release-frame `.changed()`
        // stays coalesced with the drag gesture instead of pushing a second undo snapshot.
        if !self.pointer_down {
            self.undo_gesture_active = false;
        }

        if self.audio.as_ref().is_some_and(LiveAudioEngine::is_playing) || self.recorder.is_some() {
            ui.request_repaint();
        } else if self.stem_split_rx.is_some() || self.ai_job_rx.is_some() || self.is_io_busy() {
            ui.request_repaint_after(std::time::Duration::from_millis(100));
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
        assert_eq!(
            opts.wgpu_options.surface.present_mode,
            eframe::wgpu::PresentMode::AutoVsync
        );
        assert_eq!(
            opts.wgpu_options.surface.desired_maximum_frame_latency,
            Some(2)
        );
        assert!(opts.glow_options.vsync);
        assert_eq!(
            opts.glow_options.hardware_acceleration,
            eframe::egui_glow::HardwareAcceleration::Preferred
        );
    }

    #[test]
    fn gesture_coalesced_undo_records_single_snapshot_per_drag() {
        let mut app = ForgeDawApp::from_parts(
            DawProject::default(),
            ClipCache::default(),
            None,
            "Test".to_string(),
            String::new(),
        );
        app.pointer_down = true;
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

    #[test]
    fn variable_tempo_map_snap_quantizes_across_bpm_changes() {
        let mut proj = DawProject::new("TempoSnap", 48_000, 120.0);
        proj.tempo_map = vec![
            forge_engine::TempoPoint {
                time_secs: 0.0,
                bpm: 120.0,
                time_sig_num: 4,
                time_sig_den: 4,
            },
            forge_engine::TempoPoint {
                time_secs: 2.0,
                bpm: 60.0,
                time_sig_num: 4,
                time_sig_den: 4,
            },
        ];
        // At 2.0s (4 beats @ 120 BPM) + 1.0s @ 60 BPM = beat 5 at 3.0s (144,000 samples).
        let snapped = SnapGrid::Quarter.snap_samples_for_project(146_000, &proj);
        assert_eq!(snapped, 144_000);
    }

    #[test]
    fn stage_selected_object_wav_slices_split_bounds_without_leaking_prefix() {
        let dir = tempfile::tempdir().expect("create temp test dir");
        let src_path = dir.path().join("full.wav");
        // Left half: silence (0.0); right half: 0.4 amplitude DC-ish pulse
        let mut ch = vec![0.0_f64; 4_800];
        ch.extend(std::iter::repeat_n(0.4_f64, 4_800));
        AudioBuffer::from_channels(vec![ch.clone(), ch], 48_000)
            .expect("buf")
            .save_wav(&src_path, 24)
            .expect("save");

        let out_dir = dir.path().join("ai_obj");
        let staged = stage_selected_object_wav(&src_path, &out_dir, 48_000, 4_800, 2_400, 1.0)
            .expect("stage split");
        assert_ne!(staged, src_path);
        let loaded = AudioBuffer::load(&staged).expect("load staged");
        assert_eq!(loaded.num_frames(), 2_400);
        assert!((loaded.channels[0][0] - 0.4).abs() < 1e-4);
    }

    #[test]
    fn load_project_and_clips_reports_missing_audio_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let proj_path = dir.path().join("missing_clips.forgeproj");
        let mut proj = DawProject::new("MissingTest", 48_000, 120.0);
        proj.tracks[0].objects.push(AudioObject::new(
            1,
            "Missing",
            dir.path().join("does_not_exist.wav"),
            4800,
        ));
        proj.save_to_file(&proj_path).expect("save");

        let outcome = io_jobs::load_project_and_clips(&proj_path).expect("load outcome");
        match outcome {
            IoJobOutcome::LoadedProject { missing_clips, .. } => {
                assert_eq!(missing_clips.len(), 1);
                assert!(missing_clips[0].ends_with("does_not_exist.wav"));
            }
            _ => panic!("expected LoadedProject outcome"),
        }
    }

    #[test]
    fn unique_job_media_dir_produces_distinct_paths_and_midi_rescales_with_bpm() {
        let base = PathBuf::from("session_media");
        assert_ne!(
            unique_job_media_dir(&base, "ai_studio", 7, 100),
            unique_job_media_dir(&base, "ai_studio", 7, 101)
        );
        let mut proj = DawProject::new("MidiRescale", 48_000, 120.0);
        let id = proj.alloc_id();
        proj.tracks[0]
            .midi_clips
            .push(MidiClip::demo_am7_progression(id, 48_000, 120.0));
        let len_120 = proj.tracks[0].midi_clips[0].length_samples;
        proj.bpm = 240.0;
        proj.rescale_musical_midi_clips();
        assert_eq!(proj.tracks[0].midi_clips[0].length_samples, len_120 / 2);
    }

    #[test]
    fn busy_io_preserves_live_recording_instead_of_dropping_take() {
        let mut app = ForgeDawApp::from_parts(
            DawProject::new("RecGuard", 48_000, 120.0),
            ClipCache::default(),
            None,
            "Test".to_string(),
            "Ready".to_string(),
        );
        let (_tx, rx) = std::sync::mpsc::channel();
        app.io_job_rx = Some(rx);
        app.io_job_label = "Bouncing master WAV".to_string();
        app.recorder = Some(LiveRecorder::silent_stub_for_test(0, 96_000));

        app.toggle_record();
        assert!(
            app.recorder.is_some(),
            "toggle_record must not drop active recording while I/O is busy"
        );
        assert!(app.status_line.contains("take preserved"));

        let rec = app.recorder.take().unwrap();
        let preserved = app
            .start_save_recording_job(rec)
            .expect_err("busy I/O must return Err(rec)");
        assert_eq!(preserved.track_idx(), 0);
        assert_eq!(preserved.start_timeline_samples(), 96_000);
    }
}
