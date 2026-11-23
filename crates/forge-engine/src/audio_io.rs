//! Low-latency hardware audio output & input recording via `cpal` (WASAPI/ASIO on Windows,
//! CoreAudio on macOS, ALSA/PipeWire on Linux) with atomic transport controls, real-time
//! device/project sample-rate conversion, zero-alloc meter publication, and headless fallback.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use forge_core::{AudioBuffer, ForgeError, Result};

use crate::project::DawProject;
use crate::renderer::{ClipCache, EngineMeters, ObjectOrientedEngine, resample_buffer_linear};

/// Pre-allocated zero-allocation real-time linear resampler bridging `project_sr` to `device_sr`
/// inside the hardware audio output callback when the OS audio device runs at a different rate.
struct RealtimeOutputResampler {
    src_l: Vec<f32>,
    src_r: Vec<f32>,
    src_len: usize,
    src_pos: f64,
    prev_l: f32,
    prev_r: f32,
}

impl RealtimeOutputResampler {
    fn new(block_cap: usize) -> Self {
        Self {
            src_l: vec![0.0_f32; block_cap.max(256)],
            src_r: vec![0.0_f32; block_cap.max(256)],
            src_len: 0,
            src_pos: 0.0,
            prev_l: 0.0,
            prev_r: 0.0,
        }
    }

    fn reset(&mut self) {
        self.src_len = 0;
        self.src_pos = 0.0;
        self.prev_l = 0.0;
        self.prev_r = 0.0;
    }

    /// Renders `out_frames` at `device_sr` by pulling project-rate blocks from `engine` with
    /// zero heap allocations.
    fn render_interleaved(
        &mut self,
        engine: &mut ObjectOrientedEngine,
        data: &mut [f32],
        channels: usize,
        device_sr: f64,
    ) {
        let ch = channels.max(1);
        let out_frames = data.len() / ch;
        let project_sr = engine.sample_rate().max(8_000.0);

        if (project_sr - device_sr).abs() < 1.0 {
            let mut offset = 0_usize;
            while offset < out_frames {
                let chunk = (out_frames - offset).min(self.src_l.len());
                engine.render_block(&mut self.src_l[..chunk], &mut self.src_r[..chunk]);
                for i in 0..chunk {
                    let base = (offset + i) * ch;
                    data[base] = self.src_l[i];
                    if ch > 1 {
                        data[base + 1] = self.src_r[i];
                    }
                    for extra in 2..ch {
                        data[base + extra] = 0.0;
                    }
                }
                offset += chunk;
            }
            return;
        }

        let step = project_sr / device_sr.max(8_000.0);
        let cap = self.src_l.len();

        for frame in 0..out_frames {
            while self.src_pos >= (self.src_len as f64) {
                if self.src_len > 0 {
                    self.prev_l = self.src_l[self.src_len - 1];
                    self.prev_r = self.src_r[self.src_len - 1];
                    self.src_pos -= self.src_len as f64;
                }
                engine.render_block(&mut self.src_l[..cap], &mut self.src_r[..cap]);
                self.src_len = cap;
            }

            let idx1 = (self.src_pos.floor() as usize).min(self.src_len - 1);
            let frac = (self.src_pos - self.src_pos.floor()) as f32;
            let (s0_l, s0_r) = if idx1 == 0 {
                (self.prev_l, self.prev_r)
            } else {
                (self.src_l[idx1 - 1], self.src_r[idx1 - 1])
            };
            let s1_l = self.src_l[idx1];
            let s1_r = self.src_r[idx1];
            let l = s0_l + (s1_l - s0_l) * frac;
            let r = s0_r + (s1_r - s0_r) * frac;

            let base = frame * ch;
            data[base] = l;
            if ch > 1 {
                data[base + 1] = r;
            }
            for extra in 2..ch {
                data[base + extra] = 0.0;
            }
            self.src_pos += step;
        }
    }
}

/// Thread-safe handle driving live hardware playback of an [`ObjectOrientedEngine`].
pub struct LiveAudioEngine {
    engine: Arc<Mutex<ObjectOrientedEngine>>,
    playing: Arc<AtomicBool>,
    playhead: Arc<AtomicU64>,
    seek_gen: Arc<AtomicU64>,
    seek_target: Arc<AtomicU64>,
    meters: Arc<Mutex<EngineMeters>>,
    _stream: Option<cpal::Stream>,
    device_name: String,
    device_sample_rate: u32,
}

impl LiveAudioEngine {
    /// Opens the default system audio output device (or runs in silent/headless mode if none
    /// is available, such as in CI containers).
    pub fn start(project: &DawProject, cache: &ClipCache) -> Result<Self> {
        let playing = Arc::new(AtomicBool::new(false));
        let playhead = Arc::new(AtomicU64::new(0));
        let seek_gen = Arc::new(AtomicU64::new(0));
        let seek_target = Arc::new(AtomicU64::new(0));
        let init_meters = EngineMeters {
            track_peaks_db: vec![-90.0; project.tracks.len().max(16)],
            ..EngineMeters::default()
        };
        let meters = Arc::new(Mutex::new(init_meters));

        let compiled = ObjectOrientedEngine::compile(project, cache);

        let host = cpal::default_host();
        let Some(device) = host.default_output_device() else {
            return Ok(Self {
                engine: Arc::new(Mutex::new(compiled)),
                playing,
                playhead,
                seek_gen,
                seek_target,
                meters,
                _stream: None,
                device_name: "Offline / Headless (No Audio Device)".to_string(),
                device_sample_rate: project.sample_rate,
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
        let device_sr = supported.sample_rate().0;
        let channels = supported.channels() as usize;
        let sample_format = supported.sample_format();

        let engine = Arc::new(Mutex::new(compiled));
        let stream_config: cpal::StreamConfig = supported.into();

        let build_stream = |fmt: cpal::SampleFormat| -> Result<cpal::Stream> {
            let eng_cb = Arc::clone(&engine);
            let play_cb = Arc::clone(&playing);
            let pos_cb = Arc::clone(&playhead);
            let gen_cb = Arc::clone(&seek_gen);
            let tgt_cb = Arc::clone(&seek_target);
            let met_cb = Arc::clone(&meters);

            let mut resampler = RealtimeOutputResampler::new(1024);
            let mut last_seen_gen = 0_u64;
            let mut f32_buf = vec![0.0_f32; 8192];

            let mut render_f32 = move |out_f32: &mut [f32]| {
                let cur_gen = gen_cb.load(Ordering::Acquire);
                if !play_cb.load(Ordering::Relaxed) {
                    out_f32.fill(0.0);
                    if cur_gen != last_seen_gen {
                        resampler.reset();
                        last_seen_gen = cur_gen;
                    }
                    return;
                }
                if let Ok(mut guard) = eng_cb.try_lock() {
                    if cur_gen != last_seen_gen {
                        let tgt = tgt_cb.load(Ordering::Acquire);
                        guard.set_playhead_samples(tgt);
                        resampler.reset();
                        last_seen_gen = cur_gen;
                    }
                    resampler.render_interleaved(
                        &mut guard,
                        out_f32,
                        channels,
                        f64::from(device_sr),
                    );
                    if gen_cb.load(Ordering::Acquire) == cur_gen {
                        pos_cb.store(guard.playhead_samples(), Ordering::Release);
                    }
                    if let Ok(mut m) = met_cb.try_lock() {
                        m.copy_from_no_alloc(guard.meters());
                    }
                } else {
                    out_f32.fill(0.0);
                }
            };

            let err_cb = |err| {
                eprintln!("forge-engine audio stream error: {err}");
            };

            let res = match fmt {
                cpal::SampleFormat::F32 => device.build_output_stream(
                    &stream_config,
                    move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                        render_f32(data);
                    },
                    err_cb,
                    None,
                ),
                cpal::SampleFormat::I16 => device.build_output_stream(
                    &stream_config,
                    move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                        let ch = channels.max(1);
                        let mut offset = 0_usize;
                        while offset < data.len() {
                            let max_chunk = (f32_buf.len() / ch) * ch;
                            let chunk = (data.len() - offset).min(max_chunk.max(ch));
                            let slice = &mut f32_buf[..chunk];
                            render_f32(slice);
                            for (dst, &s) in
                                data[offset..offset + chunk].iter_mut().zip(slice.iter())
                            {
                                *dst = cpal::Sample::from_sample(s.clamp(-1.0, 1.0));
                            }
                            offset += chunk;
                        }
                    },
                    err_cb,
                    None,
                ),
                cpal::SampleFormat::U16 => device.build_output_stream(
                    &stream_config,
                    move |data: &mut [u16], _: &cpal::OutputCallbackInfo| {
                        let ch = channels.max(1);
                        let mut offset = 0_usize;
                        while offset < data.len() {
                            let max_chunk = (f32_buf.len() / ch) * ch;
                            let chunk = (data.len() - offset).min(max_chunk.max(ch));
                            let slice = &mut f32_buf[..chunk];
                            render_f32(slice);
                            for (dst, &s) in
                                data[offset..offset + chunk].iter_mut().zip(slice.iter())
                            {
                                *dst = cpal::Sample::from_sample(s.clamp(-1.0, 1.0));
                            }
                            offset += chunk;
                        }
                    },
                    err_cb,
                    None,
                ),
                other => {
                    return Err(ForgeError::AudioIo {
                        path: device_name.clone(),
                        message: format!("Unsupported audio output sample format: {other:?}"),
                    });
                }
            };
            res.map_err(|e| ForgeError::AudioIo {
                path: device_name.clone(),
                message: format!("Failed to open audio output stream: {e}"),
            })
        };

        let stream = build_stream(sample_format)?;

        stream.play().map_err(|e| ForgeError::AudioIo {
            path: device_name.clone(),
            message: format!("Failed to start audio output stream: {e}"),
        })?;

        Ok(Self {
            engine,
            playing,
            playhead,
            seek_gen,
            seek_target,
            meters,
            _stream: Some(stream),
            device_name,
            device_sample_rate: device_sr,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn sample_rate(&self) -> u32 {
        self.device_sample_rate
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
        self.playhead.load(Ordering::Acquire)
    }

    pub fn seek(&self, sample_pos: u64) {
        self.seek_target.store(sample_pos, Ordering::Release);
        self.seek_gen.fetch_add(1, Ordering::AcqRel);
        self.playhead.store(sample_pos, Ordering::Release);
        if let Ok(mut guard) = self.engine.lock() {
            guard.set_playhead_samples(sample_pos);
        }
    }

    /// Updates the live DSP graph from an edited [`DawProject`].
    ///
    /// Performs a glitch-free in-place parameter update whenever the track/object/FX topology is
    /// unchanged; only recompiles off-audio-thread when tracks, objects, or FX slots are added,
    /// removed, or reordered.
    pub fn reload_project(&self, project: &DawProject, cache: &ClipCache) {
        if let Ok(mut m) = self.meters.lock() {
            if m.track_peaks_db.capacity() < project.tracks.len() {
                m.track_peaks_db.reserve(project.tracks.len());
            }
        }
        if let Ok(mut guard) = self.engine.lock() {
            if guard.sync_mix_params(project) {
                return;
            }
        }
        let mut next = ObjectOrientedEngine::compile(project, cache);
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

/// Real-time stereo audio recorder capturing from the default system input device via `cpal`.
pub struct LiveRecorder {
    _stream: cpal::Stream,
    captured_l: Arc<Mutex<Vec<f64>>>,
    captured_r: Arc<Mutex<Vec<f64>>>,
    input_sr: u32,
    start_timeline_samples: u64,
    track_idx: usize,
}

impl LiveRecorder {
    /// Starts capturing live audio from the default system input device.
    pub fn start(track_idx: usize, start_timeline_samples: u64) -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| ForgeError::AudioIo {
                path: "default_input".to_string(),
                message: "No audio input device available for recording".to_string(),
            })?;
        let device_name = device
            .name()
            .unwrap_or_else(|_| "Default Audio Input".to_string());
        let supported = device
            .default_input_config()
            .map_err(|e| ForgeError::AudioIo {
                path: device_name.clone(),
                message: format!("Failed to query input config: {e}"),
            })?;
        let input_sr = supported.sample_rate().0;
        let channels = supported.channels() as usize;

        let captured_l = Arc::new(Mutex::new(Vec::<f64>::with_capacity(
            (input_sr as usize) * 30,
        )));
        let captured_r = Arc::new(Mutex::new(Vec::<f64>::with_capacity(
            (input_sr as usize) * 30,
        )));
        let cap_l_cb = Arc::clone(&captured_l);
        let cap_r_cb = Arc::clone(&captured_r);

        let stream_config: cpal::StreamConfig = supported.into();
        let stream = device
            .build_input_stream(
                &stream_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let ch = channels.max(1);
                    let frames = data.len() / ch;
                    if let (Ok(mut l), Ok(mut r)) = (cap_l_cb.try_lock(), cap_r_cb.try_lock()) {
                        for i in 0..frames {
                            let s0 = f64::from(data[i * ch]);
                            let s1 = if ch > 1 {
                                f64::from(data[i * ch + 1])
                            } else {
                                s0
                            };
                            l.push(s0);
                            r.push(s1);
                        }
                    }
                },
                |err| {
                    eprintln!("forge-engine input stream error: {err}");
                },
                None,
            )
            .map_err(|e| ForgeError::AudioIo {
                path: device_name.clone(),
                message: format!("Failed to open audio input stream: {e}"),
            })?;

        stream.play().map_err(|e| ForgeError::AudioIo {
            path: device_name,
            message: format!("Failed to start audio input stream: {e}"),
        })?;

        Ok(Self {
            _stream: stream,
            captured_l,
            captured_r,
            input_sr,
            start_timeline_samples,
            track_idx,
        })
    }

    pub fn track_idx(&self) -> usize {
        self.track_idx
    }

    pub fn start_timeline_samples(&self) -> u64 {
        self.start_timeline_samples
    }

    /// Stops recording, resamples to `target_sr` if needed, saves a 24-bit WAV to `output_wav`,
    /// and returns the resulting [`AudioBuffer`].
    pub fn stop_and_save(self, output_wav: &Path, target_sr: u32) -> Result<AudioBuffer> {
        drop(self._stream);
        let l = self
            .captured_l
            .lock()
            .map(|v| v.clone())
            .unwrap_or_default();
        let r = self
            .captured_r
            .lock()
            .map(|v| v.clone())
            .unwrap_or_default();
        if l.is_empty() {
            return Err(ForgeError::AudioIo {
                path: output_wav.display().to_string(),
                message: "No audio frames were captured".to_string(),
            });
        }
        let raw = AudioBuffer::from_channels(vec![l, r], self.input_sr)?;
        let converted = resample_buffer_linear(&raw, target_sr)?;
        if let Some(parent) = output_wav.parent() {
            std::fs::create_dir_all(parent)?;
        }
        converted.save_wav(output_wav, 24)?;
        Ok(converted)
    }
}
