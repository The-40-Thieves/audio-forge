//! Real-time & offline Object-Oriented multitrack renderer with per-object FX chains,
//! 3-band per-object EQ, Plugin Delay Compensation (PDC), and BS.1770-4 master metering.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use forge_core::{AudioBuffer, AuditReport, Result};
use forge_dsp::{Biquad, RealtimeSpectrumAnalyzer32, audit_audio, flush_denormal_f64};

use crate::fx::FxProcessor;
use crate::midi::PolySynth;
use crate::project::{AudioObject, AuxBus, DawProject, ObjectEq3, Track, WaveformPeaks};

/// Simple pre-allocated stereo delay line for sample-accurate Plugin Delay Compensation (PDC).
struct PdcDelayLine {
    buf_l: Vec<f32>,
    buf_r: Vec<f32>,
    write_pos: usize,
    delay_samples: usize,
}

impl PdcDelayLine {
    fn new(delay_samples: usize) -> Self {
        let len = delay_samples.max(1);
        Self {
            buf_l: vec![0.0; len],
            buf_r: vec![0.0; len],
            write_pos: 0,
            delay_samples,
        }
    }

    #[inline]
    fn process(&mut self, in_l: f32, in_r: f32) -> (f32, f32) {
        if self.delay_samples == 0 {
            return (in_l, in_r);
        }
        let out_l = self.buf_l[self.write_pos];
        let out_r = self.buf_r[self.write_pos];
        self.buf_l[self.write_pos] = in_l;
        self.buf_r[self.write_pos] = in_r;
        self.write_pos = (self.write_pos + 1) % self.buf_l.len();
        (out_l, out_r)
    }

    fn reset(&mut self) {
        self.buf_l.fill(0.0);
        self.buf_r.fill(0.0);
        self.write_pos = 0;
    }
}

/// Stereo 3-band per-object EQ state (120 Hz low, 1.2 kHz mid, 6.5 kHz high).
struct ObjectEqRuntime {
    low_l: Biquad,
    low_r: Biquad,
    mid_l: Biquad,
    mid_r: Biquad,
    high_l: Biquad,
    high_r: Biquad,
    flat: bool,
}

impl ObjectEqRuntime {
    fn new(sr: f64, eq: ObjectEq3) -> Self {
        Self {
            low_l: Biquad::peaking_eq(sr, 120.0, 0.7, f64::from(eq.low_db.clamp(-18.0, 18.0))),
            low_r: Biquad::peaking_eq(sr, 120.0, 0.7, f64::from(eq.low_db.clamp(-18.0, 18.0))),
            mid_l: Biquad::peaking_eq(sr, 1200.0, 0.9, f64::from(eq.mid_db.clamp(-18.0, 18.0))),
            mid_r: Biquad::peaking_eq(sr, 1200.0, 0.9, f64::from(eq.mid_db.clamp(-18.0, 18.0))),
            high_l: Biquad::peaking_eq(sr, 6500.0, 0.7, f64::from(eq.high_db.clamp(-18.0, 18.0))),
            high_r: Biquad::peaking_eq(sr, 6500.0, 0.7, f64::from(eq.high_db.clamp(-18.0, 18.0))),
            flat: eq.is_flat(),
        }
    }

    #[inline]
    fn process(&mut self, l: f32, r: f32) -> (f32, f32) {
        if self.flat {
            return (l, r);
        }
        let mut xl = f64::from(l);
        let mut xr = f64::from(r);
        xl = self.low_l.process_sample(xl);
        xr = self.low_r.process_sample(xr);
        xl = self.mid_l.process_sample(xl);
        xr = self.mid_r.process_sample(xr);
        xl = self.high_l.process_sample(xl);
        xr = self.high_r.process_sample(xr);
        (flush_denormal_f64(xl) as f32, flush_denormal_f64(xr) as f32)
    }

    fn reset(&mut self) {
        self.low_l.reset();
        self.low_r.reset();
        self.mid_l.reset();
        self.mid_r.reset();
        self.high_l.reset();
        self.high_r.reset();
    }
}

struct ObjectRuntime {
    config: AudioObject,
    clip: Option<Arc<AudioBuffer>>,
    eq: ObjectEqRuntime,
    fx: Vec<FxProcessor>,
    pdc: PdcDelayLine,
    fx_latency: usize,
}

struct TrackRuntime {
    config: Track,
    objects: Vec<ObjectRuntime>,
    synth: PolySynth,
    synth_pdc: PdcDelayLine,
    inserts: Vec<FxProcessor>,
    pdc: PdcDelayLine,
    total_latency: usize,
    peak_db: f32,
}

struct AuxBusRuntime {
    config: AuxBus,
    inserts: Vec<FxProcessor>,
    pdc: PdcDelayLine,
    fx_latency: usize,
    acc_l: f32,
    acc_r: f32,
}

/// Live meter snapshot published from the audio engine to the DAW GUI.
#[derive(Debug, Clone)]
pub struct EngineMeters {
    pub master_peak_l_db: f32,
    pub master_peak_r_db: f32,
    pub spectrum_32_db: [f32; 32],
    pub track_peaks_db: Vec<f32>,
    pub total_pdc_samples: usize,
}

impl Default for EngineMeters {
    fn default() -> Self {
        Self {
            master_peak_l_db: -90.0,
            master_peak_r_db: -90.0,
            spectrum_32_db: [-90.0; 32],
            track_peaks_db: Vec::new(),
            total_pdc_samples: 0,
        }
    }
}

impl EngineMeters {
    /// Copies `src` into `self` without heap allocation when `self.track_peaks_db` already has
    /// sufficient capacity (pre-sized off the audio thread).
    #[inline]
    pub fn copy_from_no_alloc(&mut self, src: &Self) {
        self.master_peak_l_db = src.master_peak_l_db;
        self.master_peak_r_db = src.master_peak_r_db;
        self.spectrum_32_db = src.spectrum_32_db;
        self.total_pdc_samples = src.total_pdc_samples;
        let copy_len = src.track_peaks_db.len().min(self.track_peaks_db.capacity());
        self.track_peaks_db.truncate(copy_len);
        for (i, &val) in src.track_peaks_db.iter().take(copy_len).enumerate() {
            if i < self.track_peaks_db.len() {
                self.track_peaks_db[i] = val;
            } else {
                self.track_peaks_db.push(val);
            }
        }
    }
}

/// Resamples `buf` to `target_sr` using linear interpolation (used when importing clips whose
/// sample rate differs from the project rate).
pub fn resample_buffer_linear(buf: &AudioBuffer, target_sr: u32) -> Result<AudioBuffer> {
    if buf.sample_rate == target_sr || buf.num_frames() == 0 || target_sr == 0 {
        return Ok(buf.clone());
    }
    let src_len = buf.num_frames();
    let ratio = f64::from(buf.sample_rate) / f64::from(target_sr);
    let dst_len = ((src_len as f64) / ratio).round().max(1.0) as usize;
    let mut channels = Vec::with_capacity(buf.channels.len());
    for ch in &buf.channels {
        let mut out = Vec::with_capacity(dst_len);
        for i in 0..dst_len {
            let src_pos = (i as f64) * ratio;
            let idx0 = (src_pos.floor() as usize).min(src_len - 1);
            let idx1 = (idx0 + 1).min(src_len - 1);
            let frac = src_pos - (idx0 as f64);
            let s = ch[idx0] + (ch[idx1] - ch[idx0]) * frac;
            out.push(s);
        }
        channels.push(out);
    }
    AudioBuffer::from_channels(channels, target_sr)
}

/// Pre-loaded audio clip cache shared between the UI (for waveform peaks) and the audio engine.
#[derive(Default, Clone)]
pub struct ClipCache {
    clips: HashMap<PathBuf, Arc<AudioBuffer>>,
    peaks: HashMap<PathBuf, Arc<WaveformPeaks>>,
}

impl ClipCache {
    /// Loads a WAV/audio file from disk (or returns cached buffer) and computes waveform peaks.
    pub fn load_clip(&mut self, path: &Path) -> Result<Arc<AudioBuffer>> {
        if let Some(existing) = self.clips.get(path) {
            return Ok(Arc::clone(existing));
        }
        let buf = Arc::new(AudioBuffer::load(path)?);
        let peaks = Arc::new(WaveformPeaks::from_buffer(&buf, 256));
        self.clips.insert(path.to_path_buf(), Arc::clone(&buf));
        self.peaks.insert(path.to_path_buf(), peaks);
        Ok(buf)
    }

    /// Loads a WAV/audio file from disk and resamples it to `target_sr` if its native rate differs,
    /// guaranteeing pitch, playback speed, and timeline object duration match the project clock.
    pub fn load_clip_for_rate(&mut self, path: &Path, target_sr: u32) -> Result<Arc<AudioBuffer>> {
        if let Some(existing) = self.clips.get(path) {
            if existing.sample_rate == target_sr {
                return Ok(Arc::clone(existing));
            }
        }
        let raw = AudioBuffer::load(path)?;
        let converted = resample_buffer_linear(&raw, target_sr)?;
        let buf = Arc::new(converted);
        let peaks = Arc::new(WaveformPeaks::from_buffer(&buf, 256));
        self.clips.insert(path.to_path_buf(), Arc::clone(&buf));
        self.peaks.insert(path.to_path_buf(), peaks);
        Ok(buf)
    }

    /// Inserts an already-constructed in-memory `AudioBuffer` (ideal for tests & synthetic stems).
    pub fn insert_memory_clip(&mut self, path: PathBuf, buf: AudioBuffer) -> Arc<AudioBuffer> {
        let arc = Arc::new(buf);
        let peaks = Arc::new(WaveformPeaks::from_buffer(&arc, 256));
        self.clips.insert(path.clone(), Arc::clone(&arc));
        self.peaks.insert(path, peaks);
        arc
    }

    pub fn get_clip(&self, path: &Path) -> Option<Arc<AudioBuffer>> {
        self.clips.get(path).cloned()
    }

    pub fn get_peaks(&self, path: &Path) -> Option<Arc<WaveformPeaks>> {
        self.peaks.get(path).cloned()
    }
}

/// Samplitude-inspired **Object-Oriented Multitrack Audio Engine** with automatic PDC.
///
/// Compiles a [`DawProject`] + [`ClipCache`] into a zero-allocation per-sample execution graph
/// supporting per-object FX chains, per-object fades/EQ, track inserts, solo/mute, and master bus.
pub struct ObjectOrientedEngine {
    sample_rate: f64,
    playhead_samples: u64,
    loop_enabled: bool,
    loop_start: u64,
    loop_end: u64,
    master_gain_lin: f32,
    master_insert_cfgs: Vec<crate::project::FxSlotConfig>,
    master_inserts: Vec<FxProcessor>,
    tracks: Vec<TrackRuntime>,
    aux_buses: Vec<AuxBusRuntime>,
    main_bus_pdc: PdcDelayLine,
    analyzer: RealtimeSpectrumAnalyzer32,
    total_pdc_samples: usize,
    meters: EngineMeters,
}

impl ObjectOrientedEngine {
    /// Compiles `project` into an executable real-time audio graph (runs off-audio-thread).
    pub fn compile(project: &DawProject, cache: &ClipCache) -> Self {
        let sr = f64::from(project.sample_rate.max(8_000));
        let mut tracks = Vec::with_capacity(project.tracks.len());
        let mut max_track_latency = 0_usize;

        for track in &project.tracks {
            let mut obj_runtimes = Vec::with_capacity(track.objects.len());
            let mut max_obj_latency = 0_usize;

            for obj in &track.objects {
                let fx: Vec<FxProcessor> = obj
                    .fx_chain
                    .iter()
                    .map(|c| FxProcessor::from_config(c, sr))
                    .collect();
                let fx_latency: usize = fx
                    .iter()
                    .zip(&obj.fx_chain)
                    .map(|(p, c)| p.latency_samples(c))
                    .sum();
                max_obj_latency = max_obj_latency.max(fx_latency);
                obj_runtimes.push(ObjectRuntime {
                    config: obj.clone(),
                    clip: cache.get_clip(&obj.source_path),
                    eq: ObjectEqRuntime::new(sr, obj.eq),
                    fx,
                    pdc: PdcDelayLine::new(0),
                    fx_latency,
                });
            }

            // Align all objects within the same track to `max_obj_latency`
            for obj_rt in &mut obj_runtimes {
                let pad = max_obj_latency.saturating_sub(obj_rt.fx_latency);
                obj_rt.pdc = PdcDelayLine::new(pad);
            }

            let inserts: Vec<FxProcessor> = track
                .inserts
                .iter()
                .map(|c| FxProcessor::from_config(c, sr))
                .collect();
            let track_fx_lat: usize = inserts
                .iter()
                .zip(&track.inserts)
                .map(|(p, c)| p.latency_samples(c))
                .sum();
            let total_latency = max_obj_latency + track_fx_lat;
            max_track_latency = max_track_latency.max(total_latency);

            tracks.push(TrackRuntime {
                config: track.clone(),
                objects: obj_runtimes,
                synth: PolySynth::new(sr),
                synth_pdc: PdcDelayLine::new(max_obj_latency),
                inserts,
                pdc: PdcDelayLine::new(0),
                total_latency,
                peak_db: -90.0,
            });
        }

        // Align all parallel tracks to `max_track_latency` for sample-accurate PDC
        for tr in &mut tracks {
            let pad = max_track_latency.saturating_sub(tr.total_latency);
            tr.pdc = PdcDelayLine::new(pad);
        }

        // Compile Aux / FX Return Buses & align Main Mix vs Aux Bus PDC
        let mut aux_buses = Vec::with_capacity(project.aux_buses.len());
        let mut max_aux_latency = 0_usize;
        for bus in &project.aux_buses {
            let inserts: Vec<FxProcessor> = bus
                .inserts
                .iter()
                .map(|c| FxProcessor::from_config(c, sr))
                .collect();
            let fx_latency: usize = inserts
                .iter()
                .zip(&bus.inserts)
                .map(|(p, c)| p.latency_samples(c))
                .sum();
            max_aux_latency = max_aux_latency.max(fx_latency);
            aux_buses.push(AuxBusRuntime {
                config: bus.clone(),
                inserts,
                pdc: PdcDelayLine::new(0),
                fx_latency,
                acc_l: 0.0,
                acc_r: 0.0,
            });
        }
        for bus in &mut aux_buses {
            let pad = max_aux_latency.saturating_sub(bus.fx_latency);
            bus.pdc = PdcDelayLine::new(pad);
        }
        let main_bus_pdc = PdcDelayLine::new(max_aux_latency);

        let master_inserts: Vec<FxProcessor> = project
            .master_inserts
            .iter()
            .map(|c| FxProcessor::from_config(c, sr))
            .collect();
        let master_lat: usize = master_inserts
            .iter()
            .zip(&project.master_inserts)
            .map(|(p, c)| p.latency_samples(c))
            .sum();
        let total_pdc_samples = max_track_latency + max_aux_latency + master_lat;

        let track_peaks_db = vec![-90.0; tracks.len()];
        Self {
            sample_rate: sr,
            playhead_samples: 0,
            loop_enabled: project.loop_enabled,
            loop_start: project.loop_start_samples,
            loop_end: project.loop_end_samples,
            master_gain_lin: 10.0_f32.powf(project.master_gain_db.clamp(-48.0, 12.0) / 20.0),
            master_insert_cfgs: project.master_inserts.clone(),
            master_inserts,
            tracks,
            aux_buses,
            main_bus_pdc,
            analyzer: RealtimeSpectrumAnalyzer32::new(sr),
            total_pdc_samples,
            meters: EngineMeters {
                track_peaks_db,
                total_pdc_samples,
                ..EngineMeters::default()
            },
        }
    }

    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    pub fn playhead_samples(&self) -> u64 {
        self.playhead_samples
    }

    pub fn set_playhead_samples(&mut self, pos: u64) {
        self.playhead_samples = pos;
        self.reset_dsp_state();
    }

    pub fn total_pdc_samples(&self) -> usize {
        self.total_pdc_samples
    }

    pub fn meters(&self) -> &EngineMeters {
        &self.meters
    }

    fn fx_latency_changed(
        a: &crate::project::FxSlotConfig,
        b: &crate::project::FxSlotConfig,
    ) -> bool {
        if std::mem::discriminant(a) != std::mem::discriminant(b)
            || a.is_bypassed() != b.is_bypassed()
        {
            return true;
        }
        matches!(
            (a, b),
            (
                crate::project::FxSlotConfig::OpenSoothe {
                    low_latency: la,
                    ..
                },
                crate::project::FxSlotConfig::OpenSoothe {
                    low_latency: lb,
                    ..
                },
            ) if la != lb
        )
    }

    /// Attempts to update mix, loop, automation, object EQ/fades, and FX parameters in-place
    /// without resetting DSP state or interrupting live playback.
    ///
    /// Returns `true` if updated in-place, or `false` if the project topology (track count,
    /// object IDs/sources, FX slot variants, latency mode, or sample rate) changed and a full
    /// recompile is needed.
    pub fn sync_mix_params(&mut self, project: &DawProject) -> bool {
        let target_sr = f64::from(project.sample_rate.max(8_000));
        if (self.sample_rate - target_sr).abs() > 1e-3 {
            return false;
        }
        if self.tracks.len() != project.tracks.len()
            || self.aux_buses.len() != project.aux_buses.len()
            || self.master_insert_cfgs.len() != project.master_inserts.len()
        {
            return false;
        }
        for (bus_rt, bus_cfg) in self.aux_buses.iter().zip(project.aux_buses.iter()) {
            if bus_rt.config.id != bus_cfg.id || bus_rt.inserts.len() != bus_cfg.inserts.len() {
                return false;
            }
            for (a, b) in bus_rt.config.inserts.iter().zip(bus_cfg.inserts.iter()) {
                if Self::fx_latency_changed(a, b) {
                    return false;
                }
            }
        }
        for (a, b) in self
            .master_insert_cfgs
            .iter()
            .zip(project.master_inserts.iter())
        {
            if Self::fx_latency_changed(a, b) {
                return false;
            }
        }
        for (tr_rt, tr_cfg) in self.tracks.iter().zip(project.tracks.iter()) {
            if tr_rt.config.id != tr_cfg.id
                || tr_rt.objects.len() != tr_cfg.objects.len()
                || tr_rt.inserts.len() != tr_cfg.inserts.len()
            {
                return false;
            }
            for (a, b) in tr_rt.config.inserts.iter().zip(tr_cfg.inserts.iter()) {
                if Self::fx_latency_changed(a, b) {
                    return false;
                }
            }
            for (obj_rt, obj_cfg) in tr_rt.objects.iter().zip(tr_cfg.objects.iter()) {
                if obj_rt.config.id != obj_cfg.id
                    || obj_rt.config.source_path != obj_cfg.source_path
                    || obj_rt.fx.len() != obj_cfg.fx_chain.len()
                {
                    return false;
                }
                for (a, b) in obj_rt.config.fx_chain.iter().zip(obj_cfg.fx_chain.iter()) {
                    if Self::fx_latency_changed(a, b) {
                        return false;
                    }
                }
            }
        }

        // Safe to update parameters in-place without clearing DSP delay/reverb/FFT state!
        self.loop_enabled = project.loop_enabled;
        self.loop_start = project.loop_start_samples;
        self.loop_end = project.loop_end_samples;
        self.master_gain_lin = 10.0_f32.powf(project.master_gain_db.clamp(-48.0, 12.0) / 20.0);
        self.master_insert_cfgs = project.master_inserts.clone();

        for (tr_rt, tr_cfg) in self.tracks.iter_mut().zip(project.tracks.iter()) {
            tr_rt.config = tr_cfg.clone();
            for (obj_rt, obj_cfg) in tr_rt.objects.iter_mut().zip(tr_cfg.objects.iter()) {
                if obj_rt.config.eq != obj_cfg.eq {
                    obj_rt.eq = ObjectEqRuntime::new(self.sample_rate, obj_cfg.eq);
                }
                obj_rt.config = obj_cfg.clone();
            }
        }
        for (bus_rt, bus_cfg) in self.aux_buses.iter_mut().zip(project.aux_buses.iter()) {
            bus_rt.config = bus_cfg.clone();
        }
        true
    }

    pub fn reset_dsp_state(&mut self) {
        let sr = self.sample_rate;
        for tr in &mut self.tracks {
            tr.pdc.reset();
            tr.synth.reset(sr);
            tr.synth_pdc.reset();
            for ins in &mut tr.inserts {
                ins.reset(sr);
            }
            for obj in &mut tr.objects {
                obj.pdc.reset();
                obj.eq.reset();
                for fx in &mut obj.fx {
                    fx.reset(sr);
                }
            }
        }
        for bus in &mut self.aux_buses {
            bus.pdc.reset();
            bus.acc_l = 0.0;
            bus.acc_r = 0.0;
            for ins in &mut bus.inserts {
                ins.reset(sr);
            }
        }
        self.main_bus_pdc.reset();
        for ins in &mut self.master_inserts {
            ins.reset(sr);
        }
        self.analyzer.reset(sr);
    }

    #[inline]
    fn sample_clip_interp(clip: &AudioBuffer, src_pos: f64) -> (f32, f32, bool) {
        let clip_frames = clip.num_frames();
        if clip_frames == 0 || src_pos < 0.0 || !src_pos.is_finite() {
            return (0.0, 0.0, false);
        }
        let idx0 = src_pos.floor() as usize;
        if idx0 >= clip_frames {
            return (0.0, 0.0, false);
        }
        let idx1 = (idx0 + 1).min(clip_frames - 1);
        let frac = src_pos - (idx0 as f64);
        let c0 = (clip.channels[0][idx0] + (clip.channels[0][idx1] - clip.channels[0][idx0]) * frac)
            as f32;
        let c1 = if clip.channels.len() > 1 {
            (clip.channels[1][idx0] + (clip.channels[1][idx1] - clip.channels[1][idx0]) * frac)
                as f32
        } else {
            c0
        };
        (c0, c1, true)
    }

    #[inline]
    fn read_object_elastic_stereo(
        clip: &AudioBuffer,
        obj: &AudioObject,
        local: u64,
        engine_sr: f64,
    ) -> (f32, f32, bool) {
        let sr_ratio = f64::from(clip.sample_rate) / engine_sr.max(8_000.0);
        let stretch = f64::from(obj.time_stretch.clamp(0.5, 2.0));
        let pitch_st = f64::from(obj.pitch_semitones.clamp(-12.0, 12.0));
        let is_elastic = (stretch - 1.0).abs() > 1e-4 || pitch_st.abs() > 1e-3;

        if !is_elastic {
            if (sr_ratio - 1.0).abs() < 1e-6 {
                let src_idx = (obj.source_offset_samples + local) as usize;
                let clip_frames = clip.num_frames();
                if src_idx < clip_frames {
                    let c0 = clip.channels[0][src_idx] as f32;
                    let c1 = if clip.channels.len() > 1 {
                        clip.channels[1][src_idx] as f32
                    } else {
                        c0
                    };
                    return (c0, c1, true);
                }
                return (0.0, 0.0, false);
            }
            let src_pos = ((obj.source_offset_samples + local) as f64) * sr_ratio;
            return Self::sample_clip_interp(clip, src_pos);
        }

        // Zero-alloc Dual-Grain Hann Overlap-Add Elastic Time-Stretch & Pitch-Shift
        const GRAIN_LEN: f64 = 2048.0;
        const HALF_GRAIN: f64 = 1024.0;
        let pitch_ratio = 2.0_f64.powf(pitch_st / 12.0);
        let local_f = local as f64;
        let src_off_f = (obj.source_offset_samples as f64) * sr_ratio;

        let mut out_l = 0.0_f32;
        let mut out_r = 0.0_f32;
        let mut any_in = false;

        for g in 0..2 {
            let shifted = local_f + (g as f64) * HALF_GRAIN;
            let phase = shifted.rem_euclid(GRAIN_LEN);
            let anchor = shifted - phase;
            let u = phase / GRAIN_LEN;
            let win = (std::f64::consts::PI * u).sin();
            let w = (win * win) as f32;
            let src_pos = src_off_f
                + anchor * stretch * sr_ratio
                + (phase - HALF_GRAIN) * pitch_ratio * sr_ratio;
            let (gl, gr, ok) = Self::sample_clip_interp(clip, src_pos);
            if ok {
                out_l += gl * w;
                out_r += gr * w;
                any_in = true;
            }
        }
        (out_l, out_r, any_in)
    }

    #[inline]
    fn equal_power_pan(pan: f32) -> (f32, f32) {
        let p = (pan.clamp(-1.0, 1.0) + 1.0) * 0.5;
        let angle = p * std::f32::consts::FRAC_PI_2;
        (
            angle.cos() * std::f32::consts::SQRT_2,
            angle.sin() * std::f32::consts::SQRT_2,
        )
    }

    /// Renders `out_l.len()` stereo samples starting at `self.playhead_samples` with zero
    /// heap allocations and advances `self.playhead_samples`.
    pub fn render_block(&mut self, out_l: &mut [f32], out_r: &mut [f32]) {
        let frames = out_l.len().min(out_r.len());
        let any_solo = self.tracks.iter().any(|t| t.config.solo);
        let mut master_pk_l = 1e-6_f32;
        let mut master_pk_r = 1e-6_f32;

        for tr in &mut self.tracks {
            tr.peak_db = -90.0;
        }

        for i in 0..frames {
            let pos = self.playhead_samples;
            let mut mix_l = 0.0_f32;
            let mut mix_r = 0.0_f32;

            for bus in &mut self.aux_buses {
                bus.acc_l = 0.0;
                bus.acc_r = 0.0;
            }

            for tr in &mut self.tracks {
                let audible = if any_solo {
                    tr.config.solo
                } else {
                    !tr.config.mute
                };
                if !audible {
                    continue;
                }

                let mut tr_l = 0.0_f32;
                let mut tr_r = 0.0_f32;

                for obj in &mut tr.objects {
                    if obj.config.mute {
                        continue;
                    }
                    let start = obj.config.timeline_start_samples;
                    let end = obj.config.timeline_end_samples();
                    let (mut ol, mut or_) = if pos >= start && pos < end {
                        let local = pos - start;
                        if let Some(ref clip) = obj.clip {
                            let (s0, s1, in_bounds) = Self::read_object_elastic_stereo(
                                clip,
                                &obj.config,
                                local,
                                self.sample_rate,
                            );
                            if in_bounds {
                                let fade = obj.config.fade_gain_at(local) as f32;
                                let g = 10.0_f32.powf(obj.config.gain_db.clamp(-48.0, 24.0) / 20.0)
                                    * fade
                                    * if obj.config.phase_invert { -1.0 } else { 1.0 };
                                let (pl, pr) = Self::equal_power_pan(obj.config.pan);
                                (s0 * g * pl, s1 * g * pr)
                            } else {
                                (0.0, 0.0)
                            }
                        } else {
                            (0.0, 0.0)
                        }
                    } else {
                        (0.0, 0.0)
                    };

                    // Run per-object 3-band EQ & per-object FX chain + intra-track PDC
                    (ol, or_) = obj.eq.process(ol, or_);
                    for (proc_fx, cfg) in obj.fx.iter_mut().zip(&obj.config.fx_chain) {
                        (ol, or_) = proc_fx.process_stereo(cfg, ol, or_);
                    }
                    (ol, or_) = obj.pdc.process(ol, or_);

                    tr_l += ol;
                    tr_r += or_;
                }

                // Render built-in zero-alloc 16-voice PolySynth for any MIDI clips on this track,
                // PDC-aligned to match latency-compensated audio objects on the same track.
                if !tr.config.midi_clips.is_empty() {
                    let (mut ml, mut mr) =
                        tr.synth
                            .render_sample(&tr.config.midi_clips, tr.config.synth_preset, pos);
                    (ml, mr) = tr.synth_pdc.process(ml, mr);
                    tr_l += ml;
                    tr_r += mr;
                }

                // Track-level FX inserts + track fader/pan/automation/polarity + inter-track PDC
                for (proc_fx, cfg) in tr.inserts.iter_mut().zip(&tr.config.inserts) {
                    (tr_l, tr_r) = proc_fx.process_stereo(cfg, tr_l, tr_r);
                }
                (tr_l, tr_r) = tr.pdc.process(tr_l, tr_r);

                let pre_l = tr_l;
                let pre_r = tr_r;

                let gain_db = tr.config.volume_auto.evaluate(pos, tr.config.gain_db);
                let pan = tr.config.pan_auto.evaluate(pos, tr.config.pan);
                let tg = 10.0_f32.powf(gain_db.clamp(-60.0, 12.0) / 20.0)
                    * if tr.config.phase_invert { -1.0 } else { 1.0 };
                let (tpl, tpr) = Self::equal_power_pan(pan);
                tr_l *= tg * tpl;
                tr_r *= tg * tpr;

                // Route pre-fader and post-fader track sends to Aux Buses (zero alloc)
                for send in &tr.config.sends {
                    if send.mute || send.gain_db <= -59.5 {
                        continue;
                    }
                    let sg = 10.0_f32.powf(send.gain_db.clamp(-60.0, 6.0) / 20.0);
                    let (sl, sr_) = if send.pre_fader {
                        (pre_l * sg, pre_r * sg)
                    } else {
                        (tr_l * sg, tr_r * sg)
                    };
                    for bus in &mut self.aux_buses {
                        if bus.config.id == send.bus_id && !bus.config.mute {
                            bus.acc_l += sl;
                            bus.acc_r += sr_;
                            break;
                        }
                    }
                }

                let pk = tr_l.abs().max(tr_r.abs());
                let pk_db = 20.0 * pk.max(1e-5).log10();
                if pk_db > tr.peak_db {
                    tr.peak_db = pk_db;
                }

                mix_l += tr_l;
                mix_r += tr_r;
            }

            // Align Main Mix bus with parallel Aux bus PDC, then sum processed Aux returns
            (mix_l, mix_r) = self.main_bus_pdc.process(mix_l, mix_r);
            for bus in &mut self.aux_buses {
                if bus.config.mute {
                    continue;
                }
                let mut bl = bus.acc_l;
                let mut br = bus.acc_r;
                for (proc_fx, cfg) in bus.inserts.iter_mut().zip(&bus.config.inserts) {
                    (bl, br) = proc_fx.process_stereo(cfg, bl, br);
                }
                (bl, br) = bus.pdc.process(bl, br);
                let bg = 10.0_f32.powf(bus.config.gain_db.clamp(-60.0, 12.0) / 20.0);
                let (bpl, bpr) = Self::equal_power_pan(bus.config.pan);
                mix_l += bl * bg * bpl;
                mix_r += br * bg * bpr;
            }

            // Master bus inserts & gain
            for (proc_fx, cfg) in self.master_inserts.iter_mut().zip(&self.master_insert_cfgs) {
                (mix_l, mix_r) = proc_fx.process_stereo(cfg, mix_l, mix_r);
            }
            mix_l *= self.master_gain_lin;
            mix_r *= self.master_gain_lin;

            if !mix_l.is_finite() {
                mix_l = 0.0;
            }
            if !mix_r.is_finite() {
                mix_r = 0.0;
            }

            self.analyzer.push_stereo_sample(mix_l, mix_r);
            master_pk_l = master_pk_l.max(mix_l.abs());
            master_pk_r = master_pk_r.max(mix_r.abs());

            out_l[i] = mix_l;
            out_r[i] = mix_r;

            let next_pos = pos.saturating_add(1);
            if self.loop_enabled && self.loop_end > self.loop_start && next_pos >= self.loop_end {
                self.playhead_samples = self.loop_start;
            } else {
                self.playhead_samples = next_pos;
            }
        }

        self.meters.master_peak_l_db = 20.0 * master_pk_l.max(1e-5).log10();
        self.meters.master_peak_r_db = 20.0 * master_pk_r.max(1e-5).log10();
        self.meters.spectrum_32_db = self.analyzer.snapshot_db();
        self.meters.track_peaks_db.clear();
        for tr in &self.tracks {
            self.meters.track_peaks_db.push(tr.peak_db);
        }
    }

    /// Renders the full project offline from sample `0` to `total_frames` (compensating for PDC)
    /// into a stereo 64-bit [`AudioBuffer`].
    pub fn bounce_offline(
        project: &DawProject,
        cache: &ClipCache,
        total_frames: usize,
    ) -> Result<AudioBuffer> {
        let mut engine = Self::compile(project, cache);
        engine.loop_enabled = false;
        let pdc = engine.total_pdc_samples();
        let render_len = total_frames.saturating_add(pdc);

        let mut ch_l = Vec::with_capacity(total_frames);
        let mut ch_r = Vec::with_capacity(total_frames);
        let mut blk_l = vec![0.0_f32; 512];
        let mut blk_r = vec![0.0_f32; 512];

        let mut rendered = 0_usize;
        while rendered < render_len {
            let chunk = (render_len - rendered).min(512);
            engine.render_block(&mut blk_l[..chunk], &mut blk_r[..chunk]);
            for i in 0..chunk {
                let global_idx = rendered + i;
                if global_idx >= pdc && ch_l.len() < total_frames {
                    ch_l.push(f64::from(blk_l[i]));
                    ch_r.push(f64::from(blk_r[i]));
                }
            }
            rendered += chunk;
        }

        AudioBuffer::from_channels(vec![ch_l, ch_r], project.sample_rate)
    }

    fn compute_reverb_tail_frames(project: &DawProject) -> usize {
        let mut tail_secs = 0.0_f32;
        let inspect_fx = |fx: &crate::project::FxSlotConfig, acc: &mut f32| {
            if !fx.is_bypassed()
                && let crate::project::FxSlotConfig::SpaceVerb {
                    decay_rt60_secs, ..
                } = fx
            {
                *acc = acc.max((*decay_rt60_secs * 1.5).clamp(0.5, 8.0));
            }
        };
        for fx in &project.master_inserts {
            inspect_fx(fx, &mut tail_secs);
        }
        let any_solo = project.tracks.iter().any(|t| t.solo);
        for tr in &project.tracks {
            let audible = if any_solo { tr.solo } else { !tr.mute };
            if !audible {
                continue;
            }
            for fx in &tr.inserts {
                inspect_fx(fx, &mut tail_secs);
            }
            for obj in &tr.objects {
                if !obj.mute {
                    for fx in &obj.fx_chain {
                        inspect_fx(fx, &mut tail_secs);
                    }
                }
            }
        }
        for bus in &project.aux_buses {
            if bus.mute {
                continue;
            }
            let has_active_send = project.tracks.iter().any(|t| {
                let audible = if any_solo { t.solo } else { !t.mute };
                audible
                    && t.sends
                        .iter()
                        .any(|s| s.bus_id == bus.id && !s.mute && s.gain_db > -59.5)
            });
            if has_active_send {
                for fx in &bus.inserts {
                    inspect_fx(fx, &mut tail_secs);
                }
            }
        }
        (f64::from(tail_secs) * f64::from(project.sample_rate)).round() as usize
    }

    /// Bounces the project to a 24-bit WAV file and returns its ITU-R BS.1770-4 [`AuditReport`].
    ///
    /// Automatically appends an FX decay tail beyond the last object boundary so reverb and delay
    /// tails are never hard-truncated.
    pub fn bounce_to_wav(
        project: &DawProject,
        cache: &ClipCache,
        output_wav: &Path,
    ) -> Result<AuditReport> {
        let tail_frames = Self::compute_reverb_tail_frames(project);
        let total = (project.total_length_samples() as usize).saturating_add(tail_frames);
        let buf = Self::bounce_offline(project, cache, total)?;
        buf.save_wav(output_wav, 24)?;
        Ok(audit_audio(&buf))
    }

    /// Samplitude-style **Batch Multi-Track Stem Export**: renders every unmuted track as its own
    /// PDC-aligned 24-bit stereo WAV stem (with unique track-index prefix and FX tail preservation)
    /// inside `output_dir` and returns the written file paths.
    pub fn bounce_stems(
        project: &DawProject,
        cache: &ClipCache,
        output_dir: &Path,
    ) -> Result<Vec<PathBuf>> {
        std::fs::create_dir_all(output_dir)?;
        let mut written = Vec::new();

        for (ti, tr) in project.tracks.iter().enumerate() {
            if tr.mute || (tr.objects.is_empty() && tr.midi_clips.is_empty()) {
                continue;
            }
            let mut stem_proj = project.clone();
            for (j, t) in stem_proj.tracks.iter_mut().enumerate() {
                t.solo = j == ti;
                t.mute = j != ti;
            }
            let tail_frames = Self::compute_reverb_tail_frames(&stem_proj);
            let total = (stem_proj.total_length_samples() as usize).saturating_add(tail_frames);
            let safe_name: String = tr
                .name
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                        c
                    } else {
                        '_'
                    }
                })
                .collect();
            let stem_path = output_dir.join(format!("{:02}_{safe_name}.wav", ti + 1));
            let buf = Self::bounce_offline(&stem_proj, cache, total)?;
            buf.save_wav(&stem_path, 24)?;
            written.push(stem_path);
        }

        Ok(written)
    }
}
