//! Non-blocking background I/O workers for ForgeStudio DAW (`bounce`, `export_stems`,
//! audio `import`, `.forgeproj` `save`/`load`, and recording save).

use std::path::{Path, PathBuf};

use forge_core::{AudioBuffer, AuditReport};
use forge_engine::{AudioObject, ClipCache, DawProject, LiveRecorder, ObjectOrientedEngine, Track};

use crate::ForgeDawApp;

pub(crate) enum IoJobOutcome {
    ImportedAudio {
        path: PathBuf,
        name: String,
        buf: AudioBuffer,
        target_track: usize,
        start_samples: u64,
    },
    LoadedProject {
        path: PathBuf,
        project: DawProject,
        clips: Vec<(PathBuf, AudioBuffer)>,
        missing_clips: Vec<PathBuf>,
    },
    SavedProject {
        path: PathBuf,
    },
    BouncedMaster {
        path: PathBuf,
        report: AuditReport,
    },
    ExportedStems {
        out_dir: PathBuf,
        count: usize,
    },
    SavedRecording {
        path: PathBuf,
        buf: AudioBuffer,
        track_idx: usize,
        start_samples: u64,
        obj_id: u64,
    },
}

pub(crate) fn unique_job_media_dir(
    media_dir: &Path,
    prefix: &str,
    obj_id: u64,
    job_id: u64,
) -> PathBuf {
    media_dir.join(format!("{prefix}_{obj_id}_{job_id}"))
}

pub(crate) fn load_project_and_clips(path: &Path) -> Result<IoJobOutcome, String> {
    let loaded = DawProject::load_from_file(path).map_err(|e| format!("Load failed: {e}"))?;
    let mut cache = ClipCache::default();
    let mut clips = Vec::new();
    let mut missing_clips = Vec::new();
    for tr in &loaded.tracks {
        for obj in &tr.objects {
            if cache.get_clip(&obj.source_path).is_some() {
                continue;
            }
            match cache.load_clip_for_rate(&obj.source_path, loaded.sample_rate) {
                Ok(buf) => clips.push((obj.source_path.clone(), (*buf).clone())),
                Err(_) => missing_clips.push(obj.source_path.clone()),
            }
        }
    }
    Ok(IoJobOutcome::LoadedProject {
        path: path.to_path_buf(),
        project: loaded,
        clips,
        missing_clips,
    })
}

fn format_missing_clip_warning(missing: &[PathBuf]) -> String {
    let preview: Vec<String> = missing
        .iter()
        .take(3)
        .map(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("clip")
                .to_string()
        })
        .collect();
    if missing.len() > 3 {
        format!(
            "WARNING: {} missing clips ({}, ...)",
            missing.len(),
            preview.join(", ")
        )
    } else {
        format!(
            "WARNING: {} missing clip(s): {}",
            missing.len(),
            preview.join(", ")
        )
    }
}

impl ForgeDawApp {
    pub(crate) fn is_io_busy(&self) -> bool {
        self.io_job_rx.is_some()
    }

    pub(crate) fn start_io_job<F>(&mut self, label: &str, work: F)
    where
        F: FnOnce() -> Result<IoJobOutcome, String> + Send + 'static,
    {
        if self.io_job_rx.is_some() {
            self.status_line = format!("Background I/O '{}' already running...", self.io_job_label);
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        self.io_job_rx = Some(rx);
        self.io_job_label = label.to_string();
        self.status_line = format!("{label}...");
        std::thread::spawn(move || {
            let _ = tx.send(work());
        });
    }

    pub(crate) fn start_import_audio_job(&mut self) {
        let raw = self.import_path_buf.trim().to_string();
        if raw.is_empty() {
            self.status_line = "Enter a WAV/audio file path to import".to_string();
            return;
        }
        let path = PathBuf::from(&raw);
        let target_sr = self.project.sample_rate;
        let target_track = self
            .selected_track_idx
            .min(self.project.tracks.len().saturating_sub(1));
        let start_samples = self.playhead_samples();
        self.start_io_job("Importing audio clip", move || {
            let mut cache = ClipCache::default();
            let buf = cache
                .load_clip_for_rate(&path, target_sr)
                .map_err(|e| format!("Import failed: {e}"))?;
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("Clip")
                .to_string();
            Ok(IoJobOutcome::ImportedAudio {
                path,
                name,
                buf: (*buf).clone(),
                target_track,
                start_samples,
            })
        });
    }

    pub(crate) fn start_save_project_job(&mut self) {
        let p = PathBuf::from(self.project_path_buf.trim());
        let snapshot = self.project.clone();
        self.start_io_job("Saving project", move || {
            snapshot
                .save_to_file(&p)
                .map_err(|e| format!("Save failed: {e}"))?;
            Ok(IoJobOutcome::SavedProject { path: p })
        });
    }

    pub(crate) fn start_load_project_job(&mut self) {
        let p = PathBuf::from(self.project_path_buf.trim());
        self.start_io_job("Loading project", move || load_project_and_clips(&p));
    }

    pub(crate) fn start_bounce_master_job(&mut self) {
        let out_path = PathBuf::from(self.bounce_path_buf.trim());
        let proj = self.project.clone();
        let cache = self.cache.clone();
        self.start_io_job("Bouncing master WAV", move || {
            let report = ObjectOrientedEngine::bounce_to_wav(&proj, &cache, &out_path)
                .map_err(|e| format!("Bounce failed: {e}"))?;
            Ok(IoJobOutcome::BouncedMaster {
                path: out_path,
                report,
            })
        });
    }

    pub(crate) fn start_export_stems_job(&mut self) {
        let out_dir = PathBuf::from("stems_export");
        let proj = self.project.clone();
        let cache = self.cache.clone();
        self.start_io_job("Exporting track stems", move || {
            let paths = ObjectOrientedEngine::bounce_stems(&proj, &cache, &out_dir)
                .map_err(|e| format!("Stem export failed: {e}"))?;
            Ok(IoJobOutcome::ExportedStems {
                out_dir,
                count: paths.len(),
            })
        });
    }

    pub(crate) fn start_save_recording_job(&mut self, rec: LiveRecorder) {
        let track_idx = rec.track_idx();
        let start_samples = rec.start_timeline_samples();
        let obj_id = self.project.alloc_id();
        let out_wav = self
            .project_media_dir()
            .join(format!("recording_{obj_id}.wav"));
        let target_sr = self.project.sample_rate;
        if let Some(ref eng) = self.audio {
            eng.set_playing(false);
        }
        self.start_io_job("Saving live recording", move || {
            let buf = rec
                .stop_and_save(&out_wav, target_sr)
                .map_err(|e| format!("Recording failed: {e}"))?;
            Ok(IoJobOutcome::SavedRecording {
                path: out_wav,
                buf,
                track_idx,
                start_samples,
                obj_id,
            })
        });
    }

    fn apply_io_outcome(&mut self, outcome: IoJobOutcome) {
        match outcome {
            IoJobOutcome::ImportedAudio {
                path,
                name,
                buf,
                target_track,
                start_samples,
            } => {
                let prev = self.project.clone();
                if self.project.tracks.is_empty() {
                    self.project.add_track("01 Audio Track");
                }
                let tr_idx = target_track.min(self.project.tracks.len().saturating_sub(1));
                let len = buf.num_frames() as u64;
                self.cache.insert_memory_clip(path.clone(), buf);
                let id = self.project.alloc_id();
                let mut obj = AudioObject::new(id, name, path.clone(), len);
                obj.timeline_start_samples = start_samples;
                self.project.tracks[tr_idx].objects.push(obj);
                self.selected_object_id = Some(id);
                self.commit_project_change(prev);
                self.status_line = format!(
                    "Imported '{}' ({} Hz) onto track {}",
                    path.display(),
                    self.project.sample_rate,
                    tr_idx + 1
                );
            }
            IoJobOutcome::LoadedProject {
                path,
                project,
                clips,
                missing_clips,
            } => {
                let prev = self.project.clone();
                self.project = project;
                for (clip_path, buf) in clips {
                    self.cache.insert_memory_clip(clip_path, buf);
                }
                self.commit_project_change(prev);
                self.status_line = if missing_clips.is_empty() {
                    format!("Loaded '{}'", path.display())
                } else {
                    format!(
                        "Loaded '{}' — {}",
                        path.display(),
                        format_missing_clip_warning(&missing_clips)
                    )
                };
            }
            IoJobOutcome::SavedProject { path } => {
                self.status_line = format!("Saved '{}'", path.display());
            }
            IoJobOutcome::BouncedMaster { path, report } => {
                self.status_line = format!(
                    "Bounced '{}' | {:.2} LUFS | {:.2} dBTP | DistroKid Ready: {}",
                    path.display(),
                    report.integrated_lufs,
                    report.true_peak_dbtp,
                    if report.distrokid_ready { "YES" } else { "NO" }
                );
            }
            IoJobOutcome::ExportedStems { out_dir, count } => {
                self.status_line = format!(
                    "Exported {count} PDC-aligned 24-bit WAV stems into '{}'",
                    out_dir.display()
                );
            }
            IoJobOutcome::SavedRecording {
                path,
                buf,
                track_idx,
                start_samples,
                obj_id,
            } => {
                let prev = self.project.clone();
                if self.project.tracks.is_empty() {
                    self.project.tracks.push(Track::new(
                        obj_id.saturating_add(1),
                        "Recorded Track",
                        [56, 189, 248],
                    ));
                }
                let tr_idx = track_idx.min(self.project.tracks.len().saturating_sub(1));
                let len = buf.num_frames() as u64;
                self.cache.insert_memory_clip(path.clone(), buf);
                let mut obj = AudioObject::new(obj_id, format!("Take {obj_id}"), path, len);
                obj.timeline_start_samples = start_samples;
                self.project.tracks[tr_idx].objects.push(obj);
                self.selected_object_id = Some(obj_id);
                self.commit_project_change(prev);
                self.status_line = format!(
                    "Recorded {:.2}s onto track {}",
                    (len as f64) / f64::from(self.project.sample_rate.max(1)),
                    tr_idx + 1
                );
            }
        }
    }

    pub(crate) fn poll_io_job_result(&mut self) {
        let Some(ref rx) = self.io_job_rx else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(outcome)) => {
                self.io_job_rx = None;
                self.io_job_label.clear();
                self.apply_io_outcome(outcome);
            }
            Ok(Err(err)) => {
                self.io_job_rx = None;
                self.io_job_label.clear();
                self.status_line = err;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.io_job_rx = None;
                self.io_job_label.clear();
                self.status_line = "Background I/O worker terminated unexpectedly".to_string();
            }
        }
    }
}
