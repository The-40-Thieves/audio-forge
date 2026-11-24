//! Low-latency hardware audio output & input recording via `cpal` (WASAPI/ASIO on Windows,
//! CoreAudio on macOS, ALSA/PipeWire on Linux) with lock-free engine swaps, atomic transport
//! controls, lock-free SPSC recording ring buffer, real-time device/project sample-rate
//! conversion, zero-alloc meter publication, and headless fallback.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use forge_core::{AudioBuffer, ForgeError, Result};
use forge_dsp::{RealtimeSincResampler, resample_buffer_sinc};

use crate::project::DawProject;
use crate::renderer::{ClipCache, EngineMeters, ObjectOrientedEngine};

/// Commands sent from the UI/control thread to the audio output thread without locking the engine.
enum EngineCommand {
    /// Zero-alloc in-place parameter swap; the swapped-out project is retired back to the UI thread.
    SyncMixParams(Box<DawProject>),
    /// Full off-audio-thread recompile swap; the previous engine is retired back to the UI thread.
    ReplaceEngine(Box<ObjectOrientedEngine>),
}

/// Heap-owning resources sent back from the audio thread so `Drop` never runs on the RT thread.
enum RetiredResource {
    Project(Box<DawProject>),
    Engine(Box<ObjectOrientedEngine>),
}

/// Pre-allocated zero-allocation real-time polyphase sinc resampler bridging `project_sr` to
/// `device_sr` inside the hardware audio output callback when rates differ.
struct RealtimeOutputResampler {
    src_l: Vec<f32>,
    src_r: Vec<f32>,
    src_idx: usize,
    src_len: usize,
    sinc: RealtimeSincResampler,
    cached_project_sr: f64,
    cached_device_sr: f64,
    q_l: [f32; 64],
    q_r: [f32; 64],
    q_idx: usize,
    q_len: usize,
}

impl RealtimeOutputResampler {
    fn new(block_cap: usize) -> Self {
        Self {
            src_l: vec![0.0_f32; block_cap.max(256)],
            src_r: vec![0.0_f32; block_cap.max(256)],
            src_idx: 0,
            src_len: 0,
            sinc: RealtimeSincResampler::new(48_000.0, 48_000.0),
            cached_project_sr: 48_000.0,
            cached_device_sr: 48_000.0,
            q_l: [0.0; 64],
            q_r: [0.0; 64],
            q_idx: 0,
            q_len: 0,
        }
    }

    fn reset(&mut self) {
        self.src_idx = 0;
        self.src_len = 0;
        self.q_idx = 0;
        self.q_len = 0;
        self.sinc
            .reset(self.cached_project_sr, self.cached_device_sr);
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

        if (project_sr - self.cached_project_sr).abs() > 0.5
            || (device_sr - self.cached_device_sr).abs() > 0.5
        {
            self.cached_project_sr = project_sr;
            self.cached_device_sr = device_sr;
            self.sinc.reset(project_sr, device_sr);
            self.src_idx = 0;
            self.src_len = 0;
            self.q_idx = 0;
            self.q_len = 0;
        }

        let cap = self.src_l.len();
        for frame in 0..out_frames {
            while self.q_idx >= self.q_len {
                if self.src_idx >= self.src_len {
                    engine.render_block(&mut self.src_l[..cap], &mut self.src_r[..cap]);
                    self.src_len = cap;
                    self.src_idx = 0;
                }
                let in_l = self.src_l[self.src_idx];
                let in_r = self.src_r[self.src_idx];
                self.src_idx += 1;
                self.q_idx = 0;
                self.q_len =
                    self.sinc
                        .process_stereo_sample(in_l, in_r, &mut self.q_l, &mut self.q_r);
            }

            let l = self.q_l[self.q_idx];
            let r = self.q_r[self.q_idx];
            self.q_idx += 1;

            let base = frame * ch;
            data[base] = l;
            if ch > 1 {
                data[base + 1] = r;
            }
            for extra in 2..ch {
                data[base + extra] = 0.0;
            }
        }
    }
}

/// Safe, lock-free single-producer / single-consumer (SPSC) stereo ring buffer.
///
/// Pre-allocated once up front; the real-time audio input callback only performs atomic
/// stores (`push_stereo`) with zero heap allocations and zero mutexes, while a background
/// collector thread drains frames (`drain_into`) onto normal heap storage.
struct StereoSpscRingBuffer {
    left: Box<[AtomicU32]>,
    right: Box<[AtomicU32]>,
    mask: usize,
    write_pos: AtomicUsize,
    read_pos: AtomicUsize,
    dropped_frames: AtomicU64,
}

impl StereoSpscRingBuffer {
    fn new(min_capacity: usize) -> Self {
        let cap = min_capacity.max(1024).next_power_of_two();
        let left: Vec<AtomicU32> = (0..cap).map(|_| AtomicU32::new(0)).collect();
        let right: Vec<AtomicU32> = (0..cap).map(|_| AtomicU32::new(0)).collect();
        Self {
            left: left.into_boxed_slice(),
            right: right.into_boxed_slice(),
            mask: cap - 1,
            write_pos: AtomicUsize::new(0),
            read_pos: AtomicUsize::new(0),
            dropped_frames: AtomicU64::new(0),
        }
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.mask + 1
    }

    /// Wait-free, zero-allocation push from the real-time audio input thread.
    #[inline]
    fn push_stereo(&self, l: f32, r: f32) -> bool {
        let w = self.write_pos.load(Ordering::Relaxed);
        let r_pos = self.read_pos.load(Ordering::Acquire);
        if w.wrapping_sub(r_pos) >= self.capacity() {
            self.dropped_frames.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        let slot = w & self.mask;
        self.left[slot].store(l.to_bits(), Ordering::Relaxed);
        self.right[slot].store(r.to_bits(), Ordering::Relaxed);
        self.write_pos.store(w.wrapping_add(1), Ordering::Release);
        true
    }

    /// Drains all currently available frames from the ring buffer into `dst_l` / `dst_r`.
    fn drain_into(&self, dst_l: &mut Vec<f64>, dst_r: &mut Vec<f64>) -> usize {
        let r_pos = self.read_pos.load(Ordering::Relaxed);
        let w = self.write_pos.load(Ordering::Acquire);
        let avail = w.wrapping_sub(r_pos).min(self.capacity());
        if avail == 0 {
            return 0;
        }
        dst_l.reserve(avail);
        dst_r.reserve(avail);
        for i in 0..avail {
            let slot = r_pos.wrapping_add(i) & self.mask;
            let l = f32::from_bits(self.left[slot].load(Ordering::Relaxed));
            let r = f32::from_bits(self.right[slot].load(Ordering::Relaxed));
            dst_l.push(f64::from(l));
            dst_r.push(f64::from(r));
        }
        self.read_pos
            .store(r_pos.wrapping_add(avail), Ordering::Release);
        avail
    }

    #[cfg(test)]
    fn dropped_frames(&self) -> u64 {
        self.dropped_frames.load(Ordering::Relaxed)
    }
}

/// Thread-safe handle driving live hardware playback of an [`ObjectOrientedEngine`].
///
/// The audio output callback exclusively owns `Box<ObjectOrientedEngine>` with zero
/// mutex contention, receiving parameter updates and off-thread recompiles via a latest-wins
/// command mailbox and retiring replaced heap objects back to the UI thread for deallocation.
pub struct LiveAudioEngine {
    cmd_mailbox: Option<Arc<Mutex<Option<EngineCommand>>>>,
    retire_rx: Option<Mutex<Receiver<RetiredResource>>>,
    headless_engine: Option<Mutex<ObjectOrientedEngine>>,
    last_project: Mutex<DawProject>,
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
                cmd_mailbox: None,
                retire_rx: None,
                headless_engine: Some(Mutex::new(compiled)),
                last_project: Mutex::new(project.clone()),
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
            .description()
            .map(|desc| desc.name().to_string())
            .unwrap_or_else(|_| "Default Audio Output".to_string());
        let supported = device
            .default_output_config()
            .map_err(|e| ForgeError::AudioIo {
                path: device_name.clone(),
                message: format!("Failed to query default audio output config: {e}"),
            })?;
        let device_sr = supported.sample_rate();
        let channels = supported.channels() as usize;
        let sample_format = supported.sample_format();

        let cmd_mailbox: Arc<Mutex<Option<EngineCommand>>> = Arc::new(Mutex::new(None));
        let cmd_mailbox_cb = Arc::clone(&cmd_mailbox);
        let (retire_tx, retire_rx) = sync_channel::<RetiredResource>(64);
        let stream_config: cpal::StreamConfig = supported.into();

        let play_cb = Arc::clone(&playing);
        let pos_cb = Arc::clone(&playhead);
        let gen_cb = Arc::clone(&seek_gen);
        let tgt_cb = Arc::clone(&seek_target);
        let met_cb = Arc::clone(&meters);

        let mut engine = Box::new(compiled);
        let mut resampler = RealtimeOutputResampler::new(1024);
        let mut last_seen_gen = 0_u64;
        let mut f32_buf = vec![0.0_f32; 8192];

        let mut render_f32 = move |out_f32: &mut [f32]| {
            // Pull latest UI command without blocking; pure Box pointer swap avoids RT alloc/drop.
            if let Ok(mut slot) = cmd_mailbox_cb.try_lock()
                && let Some(cmd) = slot.take()
            {
                drop(slot);
                match cmd {
                    EngineCommand::SyncMixParams(mut proj) => {
                        let _ = engine.sync_mix_params_swap(&mut proj);
                        let _ = retire_tx.try_send(RetiredResource::Project(proj));
                    }
                    EngineCommand::ReplaceEngine(mut next) => {
                        next.set_playhead_samples(engine.playhead_samples());
                        let old = std::mem::replace(&mut engine, next);
                        let _ = retire_tx.try_send(RetiredResource::Engine(old));
                        resampler.reset();
                    }
                }
            }

            let cur_gen = gen_cb.load(Ordering::Acquire);
            if cur_gen != last_seen_gen {
                let tgt = tgt_cb.load(Ordering::Acquire);
                engine.set_playhead_samples(tgt);
                resampler.reset();
                last_seen_gen = cur_gen;
            }

            if !play_cb.load(Ordering::Relaxed) {
                out_f32.fill(0.0);
                return;
            }

            resampler.render_interleaved(&mut engine, out_f32, channels, f64::from(device_sr));
            if gen_cb.load(Ordering::Acquire) == cur_gen {
                pos_cb.store(engine.playhead_samples(), Ordering::Release);
            }
            if let Ok(mut m) = met_cb.try_lock() {
                m.copy_from_no_alloc(engine.meters());
            }
        };

        let err_cb = |err| {
            eprintln!("forge-engine audio stream error: {err}");
        };

        let stream = match sample_format {
            cpal::SampleFormat::F32 => device.build_output_stream(
                stream_config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    render_f32(data);
                },
                err_cb,
                None,
            ),
            cpal::SampleFormat::I16 => device.build_output_stream(
                stream_config,
                move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                    render_converted_output(data, channels, &mut f32_buf, &mut render_f32);
                },
                err_cb,
                None,
            ),
            cpal::SampleFormat::U16 => device.build_output_stream(
                stream_config,
                move |data: &mut [u16], _: &cpal::OutputCallbackInfo| {
                    render_converted_output(data, channels, &mut f32_buf, &mut render_f32);
                },
                err_cb,
                None,
            ),
            cpal::SampleFormat::I32 => device.build_output_stream(
                stream_config,
                move |data: &mut [i32], _: &cpal::OutputCallbackInfo| {
                    render_converted_output(data, channels, &mut f32_buf, &mut render_f32);
                },
                err_cb,
                None,
            ),
            other => {
                return Err(ForgeError::AudioIo {
                    path: device_name,
                    message: format!("Unsupported audio output sample format: {other:?}"),
                });
            }
        }
        .map_err(|e| ForgeError::AudioIo {
            path: device_name.clone(),
            message: format!("Failed to open audio output stream: {e}"),
        })?;

        stream.play().map_err(|e| ForgeError::AudioIo {
            path: device_name.clone(),
            message: format!("Failed to start audio output stream: {e}"),
        })?;

        Ok(Self {
            cmd_mailbox: Some(cmd_mailbox),
            retire_rx: Some(Mutex::new(retire_rx)),
            headless_engine: None,
            last_project: Mutex::new(project.clone()),
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

    fn drain_retired(&self) {
        if let Some(rx_lock) = &self.retire_rx
            && let Ok(rx) = rx_lock.lock()
        {
            while let Ok(old) = rx.try_recv() {
                match old {
                    RetiredResource::Project(p) => drop(p),
                    RetiredResource::Engine(e) => drop(e),
                }
            }
        }
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
        if let Some(headless) = &self.headless_engine
            && let Ok(mut guard) = headless.lock()
        {
            guard.set_playhead_samples(sample_pos);
        }
        self.drain_retired();
    }

    /// Updates the live DSP graph from an edited [`DawProject`] without locking the audio thread.
    ///
    /// Sends a zero-alloc in-place parameter swap when topology is unchanged; compiles a fresh
    /// [`ObjectOrientedEngine`] off-audio-thread and swaps it atomically when tracks, objects,
    /// or FX slots change.
    pub fn reload_project(&self, project: &DawProject, cache: &ClipCache) {
        self.drain_retired();
        if let Ok(mut m) = self.meters.lock()
            && m.track_peaks_db.capacity() < project.tracks.len()
        {
            m.track_peaks_db.reserve(project.tracks.len());
        }

        if let Some(headless) = &self.headless_engine {
            if let Ok(mut guard) = headless.lock() {
                if !guard.sync_mix_params(project) {
                    let mut next = ObjectOrientedEngine::compile(project, cache);
                    next.set_playhead_samples(self.playhead_samples());
                    *guard = next;
                }
            }
            if let Ok(mut prev) = self.last_project.lock() {
                *prev = project.clone();
            }
            return;
        }

        let topology_same = self
            .last_project
            .lock()
            .map(|prev| ObjectOrientedEngine::topology_compatible(&prev, project))
            .unwrap_or(false);

        if let Some(mailbox) = &self.cmd_mailbox
            && let Ok(mut slot) = mailbox.lock()
        {
            // If an unconsumed ReplaceEngine is already queued, keep replacing with a fresh
            // compiled engine so we never overwrite a topology change with SyncMixParams.
            let must_replace =
                !topology_same || matches!(*slot, Some(EngineCommand::ReplaceEngine(_)));
            let cmd = if must_replace {
                let mut next = ObjectOrientedEngine::compile(project, cache);
                next.set_playhead_samples(self.playhead_samples());
                EngineCommand::ReplaceEngine(Box::new(next))
            } else {
                EngineCommand::SyncMixParams(Box::new(project.clone()))
            };
            // Replacing any unconsumed command drops it safely right here on the UI thread.
            *slot = Some(cmd);
            drop(slot);
            if let Ok(mut prev) = self.last_project.lock() {
                *prev = project.clone();
            }
        }
    }

    pub fn meters_snapshot(&self) -> EngineMeters {
        self.drain_retired();
        self.meters.lock().map(|m| m.clone()).unwrap_or_default()
    }
}

impl Drop for LiveAudioEngine {
    fn drop(&mut self) {
        self._stream = None;
        self.drain_retired();
    }
}

fn render_converted_output<T, F>(
    data: &mut [T],
    channels: usize,
    f32_buf: &mut [f32],
    render_f32: &mut F,
) where
    T: cpal::Sample + cpal::FromSample<f32>,
    F: FnMut(&mut [f32]),
{
    let ch = channels.max(1);
    let mut offset = 0_usize;
    while offset < data.len() {
        let max_chunk = (f32_buf.len() / ch) * ch;
        let chunk = (data.len() - offset).min(max_chunk.max(ch));
        let slice = &mut f32_buf[..chunk];
        render_f32(slice);
        for (dst, &s) in data[offset..offset + chunk].iter_mut().zip(slice.iter()) {
            *dst = T::from_sample(s.clamp(-1.0, 1.0));
        }
        offset += chunk;
    }
}

/// Real-time stereo audio recorder capturing from the default system input device via `cpal`.
///
/// Uses a lock-free SPSC ring buffer between the real-time input callback and a background
/// collector thread so recordings of arbitrary length never allocate or lock on the audio thread.
pub struct LiveRecorder {
    stream: Option<cpal::Stream>,
    stop_flag: Arc<AtomicBool>,
    collector: Option<JoinHandle<(Vec<f64>, Vec<f64>)>>,
    input_sr: u32,
    start_timeline_samples: u64,
    track_idx: usize,
}

impl LiveRecorder {
    /// Starts capturing live audio from the default system input device (supports `F32`, `I16`,
    /// `U16`, and `I32` hardware input sample formats).
    pub fn start(track_idx: usize, start_timeline_samples: u64) -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| ForgeError::AudioIo {
                path: "default_input".to_string(),
                message: "No audio input device available for recording".to_string(),
            })?;
        let device_name = device
            .description()
            .map(|desc| desc.name().to_string())
            .unwrap_or_else(|_| "Default Audio Input".to_string());
        let supported = device
            .default_input_config()
            .map_err(|e| ForgeError::AudioIo {
                path: device_name.clone(),
                message: format!("Failed to query input config: {e}"),
            })?;
        let input_sr = supported.sample_rate();
        let channels = supported.channels() as usize;
        let sample_format = supported.sample_format();

        // Pre-allocate at least 3 seconds of lock-free SPSC stereo ring buffer (rounded up to a
        // power of two, ~2 MiB at 48 kHz = 262,144 stereo frames / ~5.46 seconds).
        let ring = Arc::new(StereoSpscRingBuffer::new(
            (input_sr as usize).saturating_mul(3).max(131_072),
        ));
        let stop_flag = Arc::new(AtomicBool::new(false));

        let ring_consumer = Arc::clone(&ring);
        let stop_consumer = Arc::clone(&stop_flag);
        let init_cap = (input_sr as usize).saturating_mul(10);
        let collector = thread::Builder::new()
            .name("forge-recorder-collector".to_string())
            .spawn(move || {
                let mut l = Vec::<f64>::with_capacity(init_cap);
                let mut r = Vec::<f64>::with_capacity(init_cap);
                while !stop_consumer.load(Ordering::Acquire) {
                    if ring_consumer.drain_into(&mut l, &mut r) == 0 {
                        thread::sleep(Duration::from_millis(8));
                    }
                }
                ring_consumer.drain_into(&mut l, &mut r);
                (l, r)
            })
            .map_err(|e| ForgeError::AudioIo {
                path: device_name.clone(),
                message: format!("Failed to spawn audio recorder collector thread: {e}"),
            })?;

        let ring_cb = Arc::clone(&ring);
        let stream_config: cpal::StreamConfig = supported.into();
        let err_cb = |err| {
            eprintln!("forge-engine input stream error: {err}");
        };

        let stream_res = match sample_format {
            cpal::SampleFormat::F32 => device.build_input_stream(
                stream_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    capture_input_frames(data, channels, &ring_cb);
                },
                err_cb,
                None,
            ),
            cpal::SampleFormat::I16 => device.build_input_stream(
                stream_config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    capture_input_frames(data, channels, &ring_cb);
                },
                err_cb,
                None,
            ),
            cpal::SampleFormat::U16 => device.build_input_stream(
                stream_config,
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    capture_input_frames(data, channels, &ring_cb);
                },
                err_cb,
                None,
            ),
            cpal::SampleFormat::I32 => device.build_input_stream(
                stream_config,
                move |data: &[i32], _: &cpal::InputCallbackInfo| {
                    capture_input_frames(data, channels, &ring_cb);
                },
                err_cb,
                None,
            ),
            other => {
                stop_flag.store(true, Ordering::Release);
                let _ = collector.join();
                return Err(ForgeError::AudioIo {
                    path: device_name,
                    message: format!("Unsupported audio input sample format: {other:?}"),
                });
            }
        };
        let stream = match stream_res {
            Ok(s) => s,
            Err(e) => {
                stop_flag.store(true, Ordering::Release);
                let _ = collector.join();
                return Err(ForgeError::AudioIo {
                    path: device_name,
                    message: format!("Failed to open audio input stream: {e}"),
                });
            }
        };

        if let Err(e) = stream.play() {
            drop(stream);
            stop_flag.store(true, Ordering::Release);
            let _ = collector.join();
            return Err(ForgeError::AudioIo {
                path: device_name,
                message: format!("Failed to start audio input stream: {e}"),
            });
        }

        Ok(Self {
            stream: Some(stream),
            stop_flag,
            collector: Some(collector),
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
    pub fn stop_and_save(mut self, output_wav: &Path, target_sr: u32) -> Result<AudioBuffer> {
        self.stream = None;
        self.stop_flag.store(true, Ordering::Release);
        let (l, r) = self
            .collector
            .take()
            .map(|h| {
                h.join().map_err(|_| ForgeError::AudioIo {
                    path: output_wav.display().to_string(),
                    message: "Recording collector thread panicked".to_string(),
                })
            })
            .transpose()?
            .unwrap_or_default();
        if l.is_empty() {
            return Err(ForgeError::AudioIo {
                path: output_wav.display().to_string(),
                message: "No audio frames were captured".to_string(),
            });
        }
        let raw = AudioBuffer::from_channels(vec![l, r], self.input_sr)?;
        let converted = resample_buffer_sinc(&raw, target_sr)?;
        if let Some(parent) = output_wav.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        converted.save_wav(output_wav, 24)?;
        Ok(converted)
    }
}

impl Drop for LiveRecorder {
    fn drop(&mut self) {
        self.stream = None;
        self.stop_flag.store(true, Ordering::Release);
        if let Some(handle) = self.collector.take() {
            let _ = handle.join();
        }
    }
}

#[inline]
fn capture_input_frames<T>(data: &[T], channels: usize, ring: &StereoSpscRingBuffer)
where
    T: cpal::Sample,
    f32: cpal::FromSample<T>,
{
    let ch = channels.max(1);
    let frames = data.len() / ch;
    for i in 0..frames {
        let s0 = cpal::Sample::to_sample::<f32>(data[i * ch]);
        let s1 = if ch > 1 {
            cpal::Sample::to_sample::<f32>(data[i * ch + 1])
        } else {
            s0
        };
        let _ = ring.push_stereo(s0, s1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spsc_ring_buffer_preserves_stereo_order_across_wraparound() {
        let ring = StereoSpscRingBuffer::new(1024);
        assert_eq!(ring.capacity(), 1024);
        let mut out_l = Vec::new();
        let mut out_r = Vec::new();

        // Push & drain across 3 full wraparounds.
        let mut seq = 0_usize;
        for _ in 0..6 {
            for _ in 0..600 {
                let l = (seq as f32) * 0.0001;
                let r = -(seq as f32) * 0.0001;
                assert!(ring.push_stereo(l, r));
                seq += 1;
            }
            let drained = ring.drain_into(&mut out_l, &mut out_r);
            assert_eq!(drained, 600);
        }
        assert_eq!(out_l.len(), 3600);
        assert_eq!(ring.dropped_frames(), 0);
        for i in 0..3600 {
            let expected = f64::from((i as f32) * 0.0001);
            assert!((out_l[i] - expected).abs() < 1e-6);
            assert!((out_r[i] + expected).abs() < 1e-6);
        }
    }

    #[test]
    fn spsc_ring_buffer_counts_overflow_without_panicking() {
        let ring = StereoSpscRingBuffer::new(1024);
        for i in 0..1024 {
            assert!(ring.push_stereo(i as f32, 0.0));
        }
        assert!(!ring.push_stereo(999.0, 999.0));
        assert_eq!(ring.dropped_frames(), 1);
    }

    #[test]
    fn realtime_output_resampler_renders_same_and_different_rates() {
        let proj = DawProject::new("ResampleTest", 48_000, 120.0);
        let cache = ClipCache::default();
        let mut engine = ObjectOrientedEngine::compile(&proj, &cache);
        let mut resampler = RealtimeOutputResampler::new(512);

        let mut buf_same = vec![1.0_f32; 512 * 2];
        resampler.render_interleaved(&mut engine, &mut buf_same, 2, 48_000.0);
        assert!(buf_same.iter().all(|s| s.is_finite()));

        resampler.reset();
        let mut buf_44k = vec![1.0_f32; 441 * 2];
        resampler.render_interleaved(&mut engine, &mut buf_44k, 2, 44_100.0);
        assert!(buf_44k.iter().all(|s| s.is_finite()));
    }
}
