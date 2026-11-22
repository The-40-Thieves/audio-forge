//! Low-latency hardware audio output via `cpal` (WASAPI/ASIO on Windows, CoreAudio on macOS,
//! ALSA/PipeWire on Linux) with atomic transport controls and graceful headless fallback.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use forge_core::{ForgeError, Result};

use crate::project::DawProject;
use crate::renderer::{ClipCache, EngineMeters, ObjectOrientedEngine};

/// Thread-safe handle driving live hardware playback of an [`ObjectOrientedEngine`].
pub struct LiveAudioEngine {
    engine: Arc<Mutex<ObjectOrientedEngine>>,
    playing: Arc<AtomicBool>,
    playhead: Arc<AtomicU64>,
    meters: Arc<Mutex<EngineMeters>>,
    _stream: Option<cpal::Stream>,
    device_name: String,
    sample_rate: u32,
}

impl LiveAudioEngine {
    /// Opens the default system audio output device (or runs in silent/headless mode if none
    /// is available, such as in CI containers).
    pub fn start(project: &DawProject, cache: &ClipCache) -> Result<Self> {
        let playing = Arc::new(AtomicBool::new(false));
        let playhead = Arc::new(AtomicU64::new(0));
        let meters = Arc::new(Mutex::new(EngineMeters::default()));

        let host = cpal::default_host();
        let Some(device) = host.default_output_device() else {
            let compiled = ObjectOrientedEngine::compile(project, cache);
            return Ok(Self {
                engine: Arc::new(Mutex::new(compiled)),
                playing,
                playhead,
                meters,
                _stream: None,
                device_name: "Offline / Headless (No Audio Device)".to_string(),
                sample_rate: project.sample_rate,
            });
        };

        let device_name = device
            .name()
            .unwrap_or_else(|_| "Default Audio Output".to_string());
        let supported = device
            .default_output_config()
            .map_err(|e| ForgeError::AudioIo {
                path: device_name.clone(),
                message: format!("Failed to query default audio output config: {e}"),
            })?;
        let sr = supported.sample_rate().0;
        let channels = supported.channels() as usize;

        let mut synced_project = project.clone();
        synced_project.sample_rate = sr;
        let compiled = ObjectOrientedEngine::compile(&synced_project, cache);
        let engine = Arc::new(Mutex::new(compiled));

        let eng_cb = Arc::clone(&engine);
        let play_cb = Arc::clone(&playing);
        let pos_cb = Arc::clone(&playhead);
        let met_cb = Arc::clone(&meters);

        let mut scratch_l = vec![0.0_f32; 4096];
        let mut scratch_r = vec![0.0_f32; 4096];

        let stream_config: cpal::StreamConfig = supported.into();
        let stream = device
            .build_output_stream(
                &stream_config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    let ch = channels.max(1);
                    let frames = data.len() / ch;
                    if !play_cb.load(Ordering::Relaxed) {
                        data.fill(0.0);
                        return;
                    }
                    if let Ok(mut guard) = eng_cb.try_lock() {
                        let mut offset = 0_usize;
                        while offset < frames {
                            let chunk = (frames - offset).min(scratch_l.len());
                            guard.render_block(&mut scratch_l[..chunk], &mut scratch_r[..chunk]);
                            for i in 0..chunk {
                                let base = (offset + i) * ch;
                                data[base] = scratch_l[i];
                                if ch > 1 {
                                    data[base + 1] = scratch_r[i];
                                }
                                for extra in 2..ch {
                                    data[base + extra] = 0.0;
                                }
                            }
                            offset += chunk;
                        }
                        pos_cb.store(guard.playhead_samples(), Ordering::Relaxed);
                        if let Ok(mut m) = met_cb.try_lock() {
                            *m = guard.meters().clone();
                        }
                    } else {
                        data.fill(0.0);
                    }
                },
                |err| {
                    eprintln!("forge-engine audio stream error: {err}");
                },
                None,
            )
            .map_err(|e| ForgeError::AudioIo {
                path: device_name.clone(),
                message: format!("Failed to open audio output stream: {e}"),
            })?;

        stream.play().map_err(|e| ForgeError::AudioIo {
            path: device_name.clone(),
            message: format!("Failed to start audio output stream: {e}"),
        })?;

        Ok(Self {
            engine,
            playing,
            playhead,
            meters,
            _stream: Some(stream),
            device_name,
            sample_rate: sr,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn is_playing(&self) -> bool {
        self.playing.load(Ordering::Relaxed)
    }

    pub fn set_playing(&self, play: bool) {
        self.playing.store(play, Ordering::Relaxed);
    }

    pub fn toggle_play(&self) -> bool {
        let next = !self.is_playing();
        self.set_playing(next);
        next
    }

    pub fn stop(&self) {
        self.set_playing(false);
        self.seek(0);
    }

    pub fn playhead_samples(&self) -> u64 {
        self.playhead.load(Ordering::Relaxed)
    }

    pub fn seek(&self, sample_pos: u64) {
        self.playhead.store(sample_pos, Ordering::Relaxed);
        if let Ok(mut guard) = self.engine.lock() {
            guard.set_playhead_samples(sample_pos);
        }
    }

    /// Recompiles the live DSP graph from an updated [`DawProject`] off-audio-thread and swaps
    /// it into the audio engine while preserving the current playhead position.
    pub fn reload_project(&self, project: &DawProject, cache: &ClipCache) {
        let mut synced = project.clone();
        synced.sample_rate = self.sample_rate;
        let mut next = ObjectOrientedEngine::compile(&synced, cache);
        let pos = self.playhead_samples();
        next.set_playhead_samples(pos);
        if let Ok(mut guard) = self.engine.lock() {
            *guard = next;
        }
    }

    pub fn meters_snapshot(&self) -> EngineMeters {
        self.meters.lock().map(|m| m.clone()).unwrap_or_default()
    }
}
