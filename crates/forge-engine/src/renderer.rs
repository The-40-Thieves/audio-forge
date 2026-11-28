//! Real-time & offline Object-Oriented multitrack renderer with per-object FX chains,
//! 3-band per-object EQ, Plugin Delay Compensation (PDC), and BS.1770-4 master metering.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use forge_core::{AudioBuffer, AuditReport, ForgeError, Result};
use forge_dsp::{
    Biquad, ParamSmoother, RealtimeSpectrumAnalyzer32, audit_audio, flush_denormal_f64,
};

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

    fn update_coeffs(&mut self, sr: f64, eq: ObjectEq3) {
        let copy = |dst: &mut Biquad, src: Biquad| {
            dst.b0 = src.b0;
            dst.b1 = src.b1;
            dst.b2 = src.b2;
            dst.a1 = src.a1;
            dst.a2 = src.a2;
        };
        let low = Biquad::peaking_eq(sr, 120.0, 0.7, f64::from(eq.low_db.clamp(-18.0, 18.0)));
        let mid = Biquad::peaking_eq(sr, 1200.0, 0.9, f64::from(eq.mid_db.clamp(-18.0, 18.0)));
        let high = Biquad::peaking_eq(sr, 6500.0, 0.7, f64::from(eq.high_db.clamp(-18.0, 18.0)));
        copy(&mut self.low_l, low);
        copy(&mut self.low_r, low);
        copy(&mut self.mid_l, mid);
        copy(&mut self.mid_r, mid);
        copy(&mut self.high_l, high);
        copy(&mut self.high_r, high);
        self.flat = eq.is_flat();
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
    fx_audio_pdc: Vec<PdcDelayLine>,
    fx_sc_pdc: Vec<PdcDelayLine>,
    pdc: PdcDelayLine,
    fx_latency: usize,
    gain_l_smooth: ParamSmoother,
    gain_r_smooth: ParamSmoother,
}

struct TrackRuntime {
    config: Track,
    objects: Vec<ObjectRuntime>,
    synth: PolySynth,
    synth_pdc: PdcDelayLine,
    inserts: Vec<FxProcessor>,
    insert_audio_pdc: Vec<PdcDelayLine>,
    insert_sc_pdc: Vec<PdcDelayLine>,
    pdc: PdcDelayLine,
    total_latency: usize,
    peak_db: f32,
    gain_l_smooth: ParamSmoother,
    gain_r_smooth: ParamSmoother,
}

struct AuxBusRuntime {
    config: AuxBus,
    inserts: Vec<FxProcessor>,
    pdc: PdcDelayLine,
    fx_latency: usize,
    acc_l: f32,
    acc_r: f32,
    gain_l_smooth: ParamSmoother,
    gain_r_smooth: ParamSmoother,
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

/// Resamples `buf` to `target_sr` using normalized Kaiser-windowed sinc interpolation.
pub fn resample_buffer_linear(buf: &AudioBuffer, target_sr: u32) -> Result<AudioBuffer> {
    forge_dsp::resample_buffer_sinc(buf, target_sr)
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
        if let Some(existing) = self.clips.get(path)
            && existing.sample_rate == target_sr
        {
            return Ok(Arc::clone(existing));
        }
        let raw = AudioBuffer::load(path)?;
        let converted = forge_dsp::resample_buffer_sinc(&raw, target_sr)?;
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

const LOOP_FADE_SAMPLES: usize = 64;

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
    loop_fade_remaining: usize,
    master_gain_lin: f32,
    master_gain_smooth: ParamSmoother,
    master_insert_cfgs: Vec<crate::project::FxSlotConfig>,
    master_inserts: Vec<FxProcessor>,
    tracks: Vec<TrackRuntime>,
    aux_buses: Vec<AuxBusRuntime>,
    main_bus_pdc: PdcDelayLine,
    analyzer: RealtimeSpectrumAnalyzer32,
    total_pdc_samples: usize,
    meters: EngineMeters,
    track_ids: Vec<u64>,
    track_pre_l: Vec<f32>,
    track_pre_r: Vec<f32>,
    track_prev_l: Vec<f32>,
    track_prev_r: Vec<f32>,
    track_post_pdc_l: Vec<f32>,
    track_post_pdc_r: Vec<f32>,
    track_master_sc_pdc: Vec<PdcDelayLine>,
    track_master_sc_l: Vec<f32>,
    track_master_sc_r: Vec<f32>,
}

#[inline]
fn resolve_sidechain_idx(
    track_ids: &[u64],
    self_idx: Option<usize>,
    cfg: &crate::project::FxSlotConfig,
) -> Option<usize> {
    let crate::project::FxSlotConfig::OpenSpacer {
        bypass: false,
        sidechain_track_id: Some(sc_id),
        ..
    } = cfg
    else {
        return None;
    };
    track_ids
        .iter()
        .enumerate()
        .find(|&(idx, &id)| self_idx != Some(idx) && id == *sc_id)
        .map(|(idx, _)| idx)
}

#[inline]
fn resolve_sidechain_sample(
    track_ids: &[u64],
    self_idx: Option<usize>,
    buf_l: &[f32],
    buf_r: &[f32],
    cfg: &crate::project::FxSlotConfig,
) -> Option<(f32, f32)> {
    resolve_sidechain_idx(track_ids, self_idx, cfg).map(|idx| (buf_l[idx], buf_r[idx]))
}

#[inline]
fn db_to_lin_f32(db: f32, min_db: f32, max_db: f32) -> f32 {
    let clean = if db.is_finite() { db } else { 0.0 };
    10.0_f32.powf(clean.clamp(min_db, max_db) / 20.0)
}

#[inline]
fn object_target_gains(obj: &AudioObject) -> (f64, f64) {
    if obj.mute {
        return (0.0, 0.0);
    }
    let g = db_to_lin_f32(obj.gain_db, -48.0, 24.0) * if obj.phase_invert { -1.0 } else { 1.0 };
    let (pl, pr) = ObjectOrientedEngine::equal_power_pan(obj.pan);
    (f64::from(g * pl), f64::from(g * pr))
}

#[inline]
fn track_target_gains(track: &Track, pos: u64, audible: bool) -> (f64, f64) {
    if !audible {
        return (0.0, 0.0);
    }
    let gain_db = track.volume_auto.evaluate(pos, track.gain_db);
    let pan = track.pan_auto.evaluate(pos, track.pan);
    let tg = db_to_lin_f32(gain_db, -60.0, 12.0) * if track.phase_invert { -1.0 } else { 1.0 };
    let (pl, pr) = ObjectOrientedEngine::equal_power_pan(pan);
    (f64::from(tg * pl), f64::from(tg * pr))
}

#[inline]
fn aux_target_gains(bus: &AuxBus) -> (f64, f64) {
    if bus.mute {
        return (0.0, 0.0);
    }
    let bg = db_to_lin_f32(bus.gain_db, -60.0, 12.0);
    let (bpl, bpr) = ObjectOrientedEngine::equal_power_pan(bus.pan);
    (f64::from(bg * bpl), f64::from(bg * bpr))
}

fn compile_fx_chain(
    slots: &[crate::project::FxSlotConfig],
    sr: f64,
    self_track_idx: Option<usize>,
    track_ids: &[u64],
) -> (Vec<FxProcessor>, usize) {
    let procs: Vec<FxProcessor> = slots
        .iter()
        .map(|c| FxProcessor::from_config(c, sr))
        .collect();
    let latency: usize = procs
        .iter()
        .zip(slots)
        .map(|(p, c)| p.latency_samples_for(c, self_track_idx, track_ids))
        .sum();
    (procs, latency)
}

fn align_object_sidechains(
    tracks: &mut [TrackRuntime],
    raw_obj_latencies: &[usize],
    track_ids: &[u64],
) {
    for (tr_idx, tr) in tracks.iter_mut().enumerate() {
        let mut max_obj_lat = 0_usize;
        for obj_rt in &mut tr.objects {
            let mut cur_lat = 0_usize;
            for (k, (proc_fx, cfg)) in obj_rt.fx.iter().zip(&obj_rt.config.fx_chain).enumerate() {
                if let Some(sc_idx) = resolve_sidechain_idx(track_ids, Some(tr_idx), cfg) {
                    let sc_lat = raw_obj_latencies[sc_idx].saturating_add(1);
                    let audio_pad = sc_lat.saturating_sub(cur_lat);
                    let sc_pad = cur_lat.saturating_sub(sc_lat);
                    obj_rt.fx_audio_pdc[k] = PdcDelayLine::new(audio_pad);
                    obj_rt.fx_sc_pdc[k] = PdcDelayLine::new(sc_pad);
                    cur_lat = cur_lat.saturating_add(audio_pad);
                }
                cur_lat = cur_lat.saturating_add(proc_fx.latency_samples_for(
                    cfg,
                    Some(tr_idx),
                    track_ids,
                ));
            }
            obj_rt.fx_latency = cur_lat;
            max_obj_lat = max_obj_lat.max(cur_lat);
        }
        for obj_rt in &mut tr.objects {
            obj_rt.pdc = PdcDelayLine::new(max_obj_lat.saturating_sub(obj_rt.fx_latency));
        }
        tr.synth_pdc = PdcDelayLine::new(max_obj_lat);
        tr.total_latency = max_obj_lat;
    }
}

fn align_track_insert_sidechains(tracks: &mut [TrackRuntime], track_ids: &[u64]) -> usize {
    let base_obj_lats: Vec<usize> = tracks.iter().map(|t| t.total_latency).collect();
    let mut max_track_latency = 0_usize;
    for (tr_idx, tr) in tracks.iter_mut().enumerate() {
        let mut cur_lat = base_obj_lats[tr_idx];
        for (k, (proc_fx, cfg)) in tr.inserts.iter().zip(&tr.config.inserts).enumerate() {
            if let Some(sc_idx) = resolve_sidechain_idx(track_ids, Some(tr_idx), cfg) {
                let sc_lat = base_obj_lats[sc_idx];
                let audio_pad = sc_lat.saturating_sub(cur_lat);
                let sc_pad = cur_lat.saturating_sub(sc_lat);
                tr.insert_audio_pdc[k] = PdcDelayLine::new(audio_pad);
                tr.insert_sc_pdc[k] = PdcDelayLine::new(sc_pad);
                cur_lat = cur_lat.saturating_add(audio_pad);
            }
            cur_lat =
                cur_lat.saturating_add(proc_fx.latency_samples_for(cfg, Some(tr_idx), track_ids));
        }
        tr.total_latency = cur_lat;
        max_track_latency = max_track_latency.max(cur_lat);
    }
    for tr in tracks {
        tr.pdc = PdcDelayLine::new(max_track_latency.saturating_sub(tr.total_latency));
    }
    max_track_latency
}

fn compile_tracks(
    project: &DawProject,
    cache: &ClipCache,
    sr: f64,
    track_ids: &[u64],
) -> (Vec<TrackRuntime>, usize) {
    let mut tracks = Vec::with_capacity(project.tracks.len());
    let mut raw_obj_latencies = Vec::with_capacity(project.tracks.len());
    for (tr_idx, track) in project.tracks.iter().enumerate() {
        let mut obj_runtimes = Vec::with_capacity(track.objects.len());
        let mut max_obj_latency = 0_usize;
        for obj in &track.objects {
            let (fx, fx_latency) = compile_fx_chain(&obj.fx_chain, sr, Some(tr_idx), track_ids);
            max_obj_latency = max_obj_latency.max(fx_latency);
            let fx_len = fx.len();
            let (ogl, ogr) = object_target_gains(obj);
            obj_runtimes.push(ObjectRuntime {
                config: obj.clone(),
                clip: cache.get_clip(&obj.source_path),
                eq: ObjectEqRuntime::new(sr, obj.eq),
                fx,
                fx_audio_pdc: (0..fx_len).map(|_| PdcDelayLine::new(0)).collect(),
                fx_sc_pdc: (0..fx_len).map(|_| PdcDelayLine::new(0)).collect(),
                pdc: PdcDelayLine::new(0),
                fx_latency,
                gain_l_smooth: ParamSmoother::new(ogl, sr, 5.0),
                gain_r_smooth: ParamSmoother::new(ogr, sr, 5.0),
            });
        }
        raw_obj_latencies.push(max_obj_latency);
        let (inserts, _) = compile_fx_chain(&track.inserts, sr, Some(tr_idx), track_ids);
        let ins_len = inserts.len();
        let any_solo = project.tracks.iter().any(|t| t.solo);
        let audible = if any_solo { track.solo } else { !track.mute };
        let (tgl, tgr) = track_target_gains(track, 0, audible);
        tracks.push(TrackRuntime {
            config: track.clone(),
            objects: obj_runtimes,
            synth: PolySynth::new(sr),
            synth_pdc: PdcDelayLine::new(max_obj_latency),
            inserts,
            insert_audio_pdc: (0..ins_len).map(|_| PdcDelayLine::new(0)).collect(),
            insert_sc_pdc: (0..ins_len).map(|_| PdcDelayLine::new(0)).collect(),
            pdc: PdcDelayLine::new(0),
            total_latency: max_obj_latency,
            peak_db: -90.0,
            gain_l_smooth: ParamSmoother::new(tgl, sr, 5.0),
            gain_r_smooth: ParamSmoother::new(tgr, sr, 5.0),
        });
    }
    align_object_sidechains(&mut tracks, &raw_obj_latencies, track_ids);
    let max_track_latency = align_track_insert_sidechains(&mut tracks, track_ids);
    (tracks, max_track_latency)
}

impl ObjectOrientedEngine {
    /// Compiles `project` into an executable real-time audio graph (runs off-audio-thread).
    pub fn compile(project: &DawProject, cache: &ClipCache) -> Self {
        let sr = f64::from(project.sample_rate.max(8_000));
        let track_ids: Vec<u64> = project.tracks.iter().map(|t| t.id).collect();
        let (tracks, max_track_latency) = compile_tracks(project, cache, sr, &track_ids);

        let mut aux_buses = Vec::with_capacity(project.aux_buses.len());
        let mut max_aux_latency = 0_usize;
        for bus in &project.aux_buses {
            let (inserts, fx_latency) = compile_fx_chain(&bus.inserts, sr, None, &track_ids);
            max_aux_latency = max_aux_latency.max(fx_latency);
            let (bgl, bgr) = aux_target_gains(bus);
            aux_buses.push(AuxBusRuntime {
                config: bus.clone(),
                inserts,
                pdc: PdcDelayLine::new(0),
                fx_latency,
                acc_l: 0.0,
                acc_r: 0.0,
                gain_l_smooth: ParamSmoother::new(bgl, sr, 5.0),
                gain_r_smooth: ParamSmoother::new(bgr, sr, 5.0),
            });
        }
        for bus in &mut aux_buses {
            bus.pdc = PdcDelayLine::new(max_aux_latency.saturating_sub(bus.fx_latency));
        }
        let main_bus_pdc = PdcDelayLine::new(max_aux_latency);

        let (master_inserts, master_lat) =
            compile_fx_chain(&project.master_inserts, sr, None, &track_ids);
        let total_pdc_samples = max_track_latency + max_aux_latency + master_lat;

        let num_tracks = tracks.len();
        let track_peaks_db = vec![-90.0; num_tracks];
        let master_gain_lin = db_to_lin_f32(project.master_gain_db, -48.0, 12.0);
        Self {
            sample_rate: sr,
            playhead_samples: 0,
            loop_enabled: project.loop_enabled,
            loop_start: project.loop_start_samples,
            loop_end: project.loop_end_samples,
            loop_fade_remaining: 0,
            master_gain_lin,
            master_gain_smooth: ParamSmoother::new(f64::from(master_gain_lin), sr, 5.0),
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
            track_ids,
            track_pre_l: vec![0.0_f32; num_tracks],
            track_pre_r: vec![0.0_f32; num_tracks],
            track_prev_l: vec![0.0_f32; num_tracks],
            track_prev_r: vec![0.0_f32; num_tracks],
            track_post_pdc_l: vec![0.0_f32; num_tracks],
            track_post_pdc_r: vec![0.0_f32; num_tracks],
            track_master_sc_pdc: (0..num_tracks)
                .map(|_| PdcDelayLine::new(max_aux_latency))
                .collect(),
            track_master_sc_l: vec![0.0_f32; num_tracks],
            track_master_sc_r: vec![0.0_f32; num_tracks],
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
        self_track_idx: Option<usize>,
        track_ids: &[u64],
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
        ) || matches!(
            (a, b),
            (
                crate::project::FxSlotConfig::MasterLimiter { isp_8x: ia, .. },
                crate::project::FxSlotConfig::MasterLimiter { isp_8x: ib, .. },
            ) if ia != ib
        ) || (matches!(a, crate::project::FxSlotConfig::OpenSpacer { .. })
            && crate::fx::open_spacer_sidechain_active(a, self_track_idx, track_ids)
                != crate::fx::open_spacer_sidechain_active(b, self_track_idx, track_ids))
    }

    fn fx_chain_latency_changed(
        prev: &[crate::project::FxSlotConfig],
        next: &[crate::project::FxSlotConfig],
        self_track_idx: Option<usize>,
        track_ids: &[u64],
    ) -> bool {
        prev.len() != next.len()
            || prev
                .iter()
                .zip(next.iter())
                .any(|(a, b)| Self::fx_latency_changed(a, b, self_track_idx, track_ids))
    }

    /// Checks off-audio-thread whether `next` has the exact same track, object, aux, and FX-latency
    /// topology as `prev` so it can be applied via [`Self::sync_mix_params_swap`] without recompiling.
    pub fn topology_compatible(prev: &DawProject, next: &DawProject) -> bool {
        if prev.sample_rate != next.sample_rate
            || prev.tracks.len() != next.tracks.len()
            || prev.aux_buses.len() != next.aux_buses.len()
            || prev
                .tracks
                .iter()
                .zip(next.tracks.iter())
                .any(|(a, b)| a.id != b.id)
        {
            return false;
        }
        let track_ids: Vec<u64> = prev.tracks.iter().map(|t| t.id).collect();
        if prev
            .aux_buses
            .iter()
            .zip(next.aux_buses.iter())
            .any(|(a, b)| {
                a.id != b.id
                    || Self::fx_chain_latency_changed(&a.inserts, &b.inserts, None, &track_ids)
            })
        {
            return false;
        }
        if Self::fx_chain_latency_changed(
            &prev.master_inserts,
            &next.master_inserts,
            None,
            &track_ids,
        ) {
            return false;
        }
        for (tr_idx, (a, b)) in prev.tracks.iter().zip(next.tracks.iter()).enumerate() {
            if a.objects.len() != b.objects.len()
                || Self::fx_chain_latency_changed(&a.inserts, &b.inserts, Some(tr_idx), &track_ids)
            {
                return false;
            }
            if a.objects.iter().zip(b.objects.iter()).any(|(oa, ob)| {
                oa.id != ob.id
                    || oa.source_path != ob.source_path
                    || Self::fx_chain_latency_changed(
                        &oa.fx_chain,
                        &ob.fx_chain,
                        Some(tr_idx),
                        &track_ids,
                    )
            }) {
                return false;
            }
        }
        true
    }

    /// Zero-allocation, zero-deallocation in-place parameter swap for real-time audio callbacks.
    /// Swaps owned config containers out of `project` so the caller can retire and drop the
    /// previous containers off the audio thread.
    pub fn sync_mix_params_swap(&mut self, project: &mut DawProject) -> bool {
        let target_sr = f64::from(project.sample_rate.max(8_000));
        if (self.sample_rate - target_sr).abs() > 1e-3 {
            return false;
        }
        if self.tracks.len() != project.tracks.len()
            || self.aux_buses.len() != project.aux_buses.len()
            || self.master_insert_cfgs.len() != project.master_inserts.len()
            || self
                .tracks
                .iter()
                .zip(project.tracks.iter())
                .any(|(tr_rt, tr_cfg)| tr_rt.config.id != tr_cfg.id)
        {
            return false;
        }
        for (bus_rt, bus_cfg) in self.aux_buses.iter().zip(project.aux_buses.iter()) {
            if bus_rt.config.id != bus_cfg.id
                || bus_rt.inserts.len() != bus_cfg.inserts.len()
                || Self::fx_chain_latency_changed(
                    &bus_rt.config.inserts,
                    &bus_cfg.inserts,
                    None,
                    &self.track_ids,
                )
            {
                return false;
            }
        }
        if Self::fx_chain_latency_changed(
            &self.master_insert_cfgs,
            &project.master_inserts,
            None,
            &self.track_ids,
        ) {
            return false;
        }
        for (tr_idx, (tr_rt, tr_cfg)) in self.tracks.iter().zip(project.tracks.iter()).enumerate() {
            if tr_rt.objects.len() != tr_cfg.objects.len()
                || tr_rt.inserts.len() != tr_cfg.inserts.len()
                || Self::fx_chain_latency_changed(
                    &tr_rt.config.inserts,
                    &tr_cfg.inserts,
                    Some(tr_idx),
                    &self.track_ids,
                )
            {
                return false;
            }
            for (obj_rt, obj_cfg) in tr_rt.objects.iter().zip(tr_cfg.objects.iter()) {
                if obj_rt.config.id != obj_cfg.id
                    || obj_rt.config.source_path != obj_cfg.source_path
                    || obj_rt.fx.len() != obj_cfg.fx_chain.len()
                    || Self::fx_chain_latency_changed(
                        &obj_rt.config.fx_chain,
                        &obj_cfg.fx_chain,
                        Some(tr_idx),
                        &self.track_ids,
                    )
                {
                    return false;
                }
            }
        }

        // Safe to update parameters in-place without clearing DSP delay/reverb/FFT state!
        self.loop_enabled = project.loop_enabled;
        self.loop_start = project.loop_start_samples;
        self.loop_end = project.loop_end_samples;
        self.master_gain_lin = 10.0_f32.powf(project.master_gain_db.clamp(-48.0, 12.0) / 20.0);
        self.master_gain_smooth
            .set_target(f64::from(self.master_gain_lin));
        std::mem::swap(&mut self.master_insert_cfgs, &mut project.master_inserts);

        for (tr_rt, tr_cfg) in self.tracks.iter_mut().zip(project.tracks.iter_mut()) {
            for (obj_rt, obj_cfg) in tr_rt.objects.iter_mut().zip(tr_cfg.objects.iter_mut()) {
                if obj_rt.config.eq != obj_cfg.eq {
                    obj_rt.eq.update_coeffs(self.sample_rate, obj_cfg.eq);
                }
                std::mem::swap(&mut obj_rt.config, obj_cfg);
            }
            std::mem::swap(&mut tr_rt.config, tr_cfg);
        }
        for (bus_rt, bus_cfg) in self.aux_buses.iter_mut().zip(project.aux_buses.iter_mut()) {
            std::mem::swap(&mut bus_rt.config, bus_cfg);
        }
        true
    }

    /// Attempts to update mix, loop, automation, object EQ/fades, and FX parameters in-place
    /// without resetting DSP state or interrupting live playback.
    ///
    /// Returns `true` if updated in-place, or `false` if the project topology (track count,
    /// object IDs/sources, FX slot variants, latency mode, or sample rate) changed and a full
    /// recompile is needed.
    pub fn sync_mix_params(&mut self, project: &DawProject) -> bool {
        let mut owned = project.clone();
        self.sync_mix_params_swap(&mut owned)
    }

    pub fn reset_dsp_state(&mut self) {
        let sr = self.sample_rate;
        let pos = self.playhead_samples;
        let any_solo = self.tracks.iter().any(|t| t.config.solo);
        self.loop_fade_remaining = 0;
        self.master_gain_smooth
            .reset(f64::from(self.master_gain_lin));
        for tr in &mut self.tracks {
            let audible = if any_solo {
                tr.config.solo
            } else {
                !tr.config.mute
            };
            let (tgl, tgr) = track_target_gains(&tr.config, pos, audible);
            tr.gain_l_smooth.reset(tgl);
            tr.gain_r_smooth.reset(tgr);
            tr.pdc.reset();
            tr.synth.reset(sr);
            tr.synth_pdc.reset();
            for p in &mut tr.insert_audio_pdc {
                p.reset();
            }
            for p in &mut tr.insert_sc_pdc {
                p.reset();
            }
            for ins in &mut tr.inserts {
                ins.reset(sr);
            }
            for obj in &mut tr.objects {
                let (ogl, ogr) = object_target_gains(&obj.config);
                obj.gain_l_smooth.reset(ogl);
                obj.gain_r_smooth.reset(ogr);
                obj.pdc.reset();
                obj.eq.reset();
                for p in &mut obj.fx_audio_pdc {
                    p.reset();
                }
                for p in &mut obj.fx_sc_pdc {
                    p.reset();
                }
                for fx in &mut obj.fx {
                    fx.reset(sr);
                }
            }
        }
        for bus in &mut self.aux_buses {
            let (bgl, bgr) = aux_target_gains(&bus.config);
            bus.gain_l_smooth.reset(bgl);
            bus.gain_r_smooth.reset(bgr);
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
        self.track_pre_l.fill(0.0);
        self.track_pre_r.fill(0.0);
        self.track_prev_l.fill(0.0);
        self.track_prev_r.fill(0.0);
        self.track_post_pdc_l.fill(0.0);
        self.track_post_pdc_r.fill(0.0);
        self.track_master_sc_l.fill(0.0);
        self.track_master_sc_r.fill(0.0);
        for pdc in &mut self.track_master_sc_pdc {
            pdc.reset();
        }
    }

    #[inline]
    fn catmull_rom(p0: f64, p1: f64, p2: f64, p3: f64, t: f64) -> f32 {
        let t2 = t * t;
        let t3 = t2 * t;
        let out = 0.5
            * ((2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
        out as f32
    }

    #[inline]
    fn sample_clip_interp(clip: &AudioBuffer, src_pos: f64) -> (f32, f32, bool) {
        let clip_frames = clip.num_frames();
        if clip_frames == 0 || src_pos < 0.0 || !src_pos.is_finite() {
            return (0.0, 0.0, false);
        }
        let i1 = src_pos.floor() as usize;
        if i1 >= clip_frames {
            return (0.0, 0.0, false);
        }
        let i0 = i1.saturating_sub(1);
        let i2 = (i1 + 1).min(clip_frames - 1);
        let i3 = (i1 + 2).min(clip_frames - 1);
        let frac = src_pos - (i1 as f64);
        let ch0 = &clip.channels[0];
        let c0 = Self::catmull_rom(ch0[i0], ch0[i1], ch0[i2], ch0[i3], frac);
        let c1 = if clip.channels.len() > 1 {
            let ch1 = &clip.channels[1];
            Self::catmull_rom(ch1[i0], ch1[i1], ch1[i2], ch1[i3], frac)
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
            let src_u64 = obj.source_offset_samples.saturating_add(local);
            if (sr_ratio - 1.0).abs() < 1e-6 {
                let src_idx = usize::try_from(src_u64).unwrap_or(usize::MAX);
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
            let src_pos = (src_u64 as f64) * sr_ratio;
            return Self::sample_clip_interp(clip, src_pos);
        }

        // Zero-alloc Dual-Grain Hann Overlap-Add Elastic Time-Stretch & Pitch-Shift
        // Scale grain duration (~42 ms) with sample rate so 96/192 kHz sessions stay smooth.
        let grain_len = (0.04266 * engine_sr.max(8_000.0))
            .round()
            .clamp(512.0, 8192.0);
        let half_grain = grain_len * 0.5;
        let pitch_ratio = 2.0_f64.powf(pitch_st / 12.0);
        let local_f = local as f64;
        let src_off_f = (obj.source_offset_samples as f64) * sr_ratio;

        let mut out_l = 0.0_f32;
        let mut out_r = 0.0_f32;
        let mut any_in = false;

        for g in 0..2 {
            let shifted = local_f + (g as f64) * half_grain;
            let phase = shifted.rem_euclid(grain_len);
            let anchor = shifted - phase;
            let u = phase / grain_len;
            let win = (std::f64::consts::PI * u).sin();
            let w = (win * win) as f32;
            let src_pos = src_off_f
                + anchor * stretch * sr_ratio
                + (phase - half_grain) * pitch_ratio * sr_ratio;
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

    #[inline]
    fn object_tail_pad(obj: &ObjectRuntime, sr: f64) -> u64 {
        if obj.fx.is_empty() {
            return (obj.pdc.delay_samples as u64).saturating_add(64);
        }
        let mut tail_secs = 4.0_f64;
        for fx in &obj.config.fx_chain {
            if !fx.is_bypassed()
                && let crate::project::FxSlotConfig::SpaceVerb {
                    decay_rt60_secs, ..
                } = fx
            {
                tail_secs = tail_secs.max(f64::from(*decay_rt60_secs) * 1.5);
            }
        }
        (tail_secs.clamp(4.0, 20.0) * sr).round() as u64
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn render_object_sample(
        obj: &mut ObjectRuntime,
        pos: u64,
        sr: f64,
        tr_idx: usize,
        track_ids: &[u64],
        track_prev_l: &[f32],
        track_prev_r: &[f32],
    ) -> (f32, f32) {
        if obj.config.mute
            && obj.gain_l_smooth.current().abs() < 1e-5
            && obj.gain_r_smooth.current().abs() < 1e-5
        {
            return (0.0, 0.0);
        }
        let start = obj.config.timeline_start_samples;
        let end = obj.config.timeline_end_samples();
        let tail_pad = Self::object_tail_pad(obj, sr);
        if pos < start || pos >= end.saturating_add(tail_pad) {
            return (0.0, 0.0);
        }
        let (tgt_l, tgt_r) = object_target_gains(&obj.config);
        obj.gain_l_smooth.set_target(tgt_l);
        obj.gain_r_smooth.set_target(tgt_r);
        let gl_sm = obj.gain_l_smooth.next_val() as f32;
        let gr_sm = obj.gain_r_smooth.next_val() as f32;

        let (mut ol, mut or_) = if pos >= start
            && pos < end
            && let Some(ref clip) = obj.clip
        {
            let local = pos - start;
            let (s0, s1, in_bounds) =
                Self::read_object_elastic_stereo(clip, &obj.config, local, sr);
            if in_bounds {
                let fade = obj.config.fade_gain_at(local) as f32;
                (s0 * gl_sm * fade, s1 * gr_sm * fade)
            } else {
                (0.0, 0.0)
            }
        } else {
            (0.0, 0.0)
        };

        (ol, or_) = obj.eq.process(ol, or_);
        for (k, (proc_fx, cfg)) in obj.fx.iter_mut().zip(&obj.config.fx_chain).enumerate() {
            (ol, or_) = obj.fx_audio_pdc[k].process(ol, or_);
            let sc =
                resolve_sidechain_sample(track_ids, Some(tr_idx), track_prev_l, track_prev_r, cfg)
                    .map(|(sl, sr_)| obj.fx_sc_pdc[k].process(sl, sr_));
            (ol, or_) = proc_fx.process_stereo_with_sidechain(cfg, ol, or_, sc);
        }
        obj.pdc.process(ol, or_)
    }

    #[inline]
    fn render_track_pre_stage(&mut self, pos: u64, any_solo: bool) {
        let sr = self.sample_rate;
        for (tr_idx, tr) in self.tracks.iter_mut().enumerate() {
            let audible = if any_solo {
                tr.config.solo
            } else {
                !tr.config.mute
            };
            if !audible
                && tr.gain_l_smooth.current().abs() < 1e-5
                && tr.gain_r_smooth.current().abs() < 1e-5
            {
                self.track_pre_l[tr_idx] = 0.0;
                self.track_pre_r[tr_idx] = 0.0;
                continue;
            }
            let (mut tr_l, mut tr_r) = (0.0_f32, 0.0_f32);
            for obj in &mut tr.objects {
                let (ol, or_) = Self::render_object_sample(
                    obj,
                    pos,
                    sr,
                    tr_idx,
                    &self.track_ids,
                    &self.track_prev_l,
                    &self.track_prev_r,
                );
                tr_l += ol;
                tr_r += or_;
            }
            if !tr.config.midi_clips.is_empty() {
                let (mut ml, mut mr) =
                    tr.synth
                        .render_sample(&tr.config.midi_clips, tr.config.synth_preset, pos);
                (ml, mr) = tr.synth_pdc.process(ml, mr);
                tr_l += ml;
                tr_r += mr;
            }
            self.track_pre_l[tr_idx] = tr_l;
            self.track_pre_r[tr_idx] = tr_r;
        }
    }

    #[inline]
    fn render_tracks_post_stage(&mut self, pos: u64, any_solo: bool) -> (f32, f32) {
        let (mut mix_l, mut mix_r) = (0.0_f32, 0.0_f32);
        for (tr_idx, tr) in self.tracks.iter_mut().enumerate() {
            let audible = if any_solo {
                tr.config.solo
            } else {
                !tr.config.mute
            };
            if !audible
                && tr.gain_l_smooth.current().abs() < 1e-5
                && tr.gain_r_smooth.current().abs() < 1e-5
            {
                self.track_post_pdc_l[tr_idx] = 0.0;
                self.track_post_pdc_r[tr_idx] = 0.0;
                continue;
            }
            let mut tr_l = self.track_pre_l[tr_idx];
            let mut tr_r = self.track_pre_r[tr_idx];
            for (k, (proc_fx, cfg)) in tr.inserts.iter_mut().zip(&tr.config.inserts).enumerate() {
                (tr_l, tr_r) = tr.insert_audio_pdc[k].process(tr_l, tr_r);
                let sc = resolve_sidechain_sample(
                    &self.track_ids,
                    Some(tr_idx),
                    &self.track_pre_l,
                    &self.track_pre_r,
                    cfg,
                )
                .map(|(sl, sr_)| tr.insert_sc_pdc[k].process(sl, sr_));
                (tr_l, tr_r) = proc_fx.process_stereo_with_sidechain(cfg, tr_l, tr_r, sc);
            }
            (tr_l, tr_r) = tr.pdc.process(tr_l, tr_r);
            let (pre_l, pre_r) = (tr_l, tr_r);
            self.track_post_pdc_l[tr_idx] = pre_l;
            self.track_post_pdc_r[tr_idx] = pre_r;

            let (tpl, tpr) = track_target_gains(&tr.config, pos, audible);
            tr.gain_l_smooth.set_target(tpl);
            tr.gain_r_smooth.set_target(tpr);
            tr_l *= tr.gain_l_smooth.next_val() as f32;
            tr_r *= tr.gain_r_smooth.next_val() as f32;

            Self::accumulate_track_sends(
                &tr.config.sends,
                &mut self.aux_buses,
                (pre_l, pre_r),
                (tr_l, tr_r),
            );
            let pk_db = 20.0 * tr_l.abs().max(tr_r.abs()).max(1e-5).log10();
            if pk_db > tr.peak_db {
                tr.peak_db = pk_db;
            }
            mix_l += tr_l;
            mix_r += tr_r;
        }
        (mix_l, mix_r)
    }

    #[inline]
    fn accumulate_track_sends(
        sends: &[crate::project::TrackSend],
        aux_buses: &mut [AuxBusRuntime],
        pre: (f32, f32),
        post: (f32, f32),
    ) {
        for send in sends {
            if send.mute || send.gain_db <= -59.5 {
                continue;
            }
            let sg = db_to_lin_f32(send.gain_db, -60.0, 6.0);
            let (sl, sr_) = if send.pre_fader {
                (pre.0 * sg, pre.1 * sg)
            } else {
                (post.0 * sg, post.1 * sg)
            };
            for bus in aux_buses.iter_mut() {
                if bus.config.id == send.bus_id && !bus.config.mute {
                    bus.acc_l += sl;
                    bus.acc_r += sr_;
                    break;
                }
            }
        }
    }

    #[inline]
    fn render_aux_and_master_sample(&mut self, mix_l: f32, mix_r: f32) -> (f32, f32) {
        let (mut out_l, mut out_r) = self.main_bus_pdc.process(mix_l, mix_r);
        for bus in &mut self.aux_buses {
            if bus.config.mute
                && bus.gain_l_smooth.current().abs() < 1e-5
                && bus.gain_r_smooth.current().abs() < 1e-5
            {
                continue;
            }
            let (mut bl, mut br) = (bus.acc_l, bus.acc_r);
            for (proc_fx, cfg) in bus.inserts.iter_mut().zip(&bus.config.inserts) {
                let sc = resolve_sidechain_sample(
                    &self.track_ids,
                    None,
                    &self.track_post_pdc_l,
                    &self.track_post_pdc_r,
                    cfg,
                );
                (bl, br) = proc_fx.process_stereo_with_sidechain(cfg, bl, br, sc);
            }
            (bl, br) = bus.pdc.process(bl, br);
            let (bgl, bgr) = aux_target_gains(&bus.config);
            bus.gain_l_smooth.set_target(bgl);
            bus.gain_r_smooth.set_target(bgr);
            out_l += bl * (bus.gain_l_smooth.next_val() as f32);
            out_r += br * (bus.gain_r_smooth.next_val() as f32);
        }

        for (proc_fx, cfg) in self.master_inserts.iter_mut().zip(&self.master_insert_cfgs) {
            let sc = resolve_sidechain_sample(
                &self.track_ids,
                None,
                &self.track_master_sc_l,
                &self.track_master_sc_r,
                cfg,
            );
            (out_l, out_r) = proc_fx.process_stereo_with_sidechain(cfg, out_l, out_r, sc);
        }
        let mg = self.master_gain_smooth.next_val() as f32;
        out_l *= mg;
        out_r *= mg;
        if self.loop_fade_remaining > 0 {
            let ramp =
                (LOOP_FADE_SAMPLES - self.loop_fade_remaining) as f32 / (LOOP_FADE_SAMPLES as f32);
            out_l *= ramp;
            out_r *= ramp;
            self.loop_fade_remaining -= 1;
        }
        (
            if out_l.is_finite() { out_l } else { 0.0 },
            if out_r.is_finite() { out_r } else { 0.0 },
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
            for bus in &mut self.aux_buses {
                bus.acc_l = 0.0;
                bus.acc_r = 0.0;
            }
            self.render_track_pre_stage(pos, any_solo);
            let (raw_mix_l, raw_mix_r) = self.render_tracks_post_stage(pos, any_solo);
            self.track_prev_l.copy_from_slice(&self.track_pre_l);
            self.track_prev_r.copy_from_slice(&self.track_pre_r);
            for (tr_idx, pdc) in self.track_master_sc_pdc.iter_mut().enumerate() {
                let (ml, mr) =
                    pdc.process(self.track_post_pdc_l[tr_idx], self.track_post_pdc_r[tr_idx]);
                self.track_master_sc_l[tr_idx] = ml;
                self.track_master_sc_r[tr_idx] = mr;
            }
            let (mix_l, mix_r) = self.render_aux_and_master_sample(raw_mix_l, raw_mix_r);
            self.analyzer.push_stereo_sample(mix_l, mix_r);
            master_pk_l = master_pk_l.max(mix_l.abs());
            master_pk_r = master_pk_r.max(mix_r.abs());
            out_l[i] = mix_l;
            out_r[i] = mix_r;

            let next_pos = pos.saturating_add(1);
            if self.loop_enabled && self.loop_end > self.loop_start && next_pos >= self.loop_end {
                self.playhead_samples = self.loop_start;
                self.loop_fade_remaining = LOOP_FADE_SAMPLES;
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
        if total_frames > crate::project::MAX_BOUNCE_FRAMES {
            return Err(ForgeError::InvalidParameter(format!(
                "Bounce frame count ({total_frames}) exceeds safety limit ({})",
                crate::project::MAX_BOUNCE_FRAMES
            )));
        }
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

    pub fn compute_reverb_tail_frames(project: &DawProject) -> usize {
        let mut tail_secs = 0.0_f32;
        let inspect_fx = |fx: &crate::project::FxSlotConfig, acc: &mut f32| {
            if !fx.is_bypassed()
                && let crate::project::FxSlotConfig::SpaceVerb {
                    decay_rt60_secs, ..
                } = fx
            {
                *acc = acc.max((*decay_rt60_secs * 1.5).clamp(0.5, 20.0));
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
        let base_frames = usize::try_from(project.total_length_samples()).map_err(|_| {
            ForgeError::InvalidParameter("Project timeline length overflows usize".to_string())
        })?;
        let total = base_frames.saturating_add(tail_frames);
        let buf = Self::bounce_offline(project, cache, total)?;
        buf.save_wav(output_wav, 24)?;
        Ok(audit_audio(&buf))
    }

    /// Renders every unmuted track as a soloed, PDC-aligned stereo [`AudioBuffer`] with FX tail
    /// preservation and invokes `on_stem(track_index, sanitized_track_name, &buffer)`.
    pub fn bounce_stems_with<F>(
        project: &DawProject,
        cache: &ClipCache,
        mut on_stem: F,
    ) -> Result<usize>
    where
        F: FnMut(usize, &str, &AudioBuffer) -> Result<()>,
    {
        let mut count = 0_usize;
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
            let base_frames = usize::try_from(stem_proj.total_length_samples()).map_err(|_| {
                ForgeError::InvalidParameter("Project timeline length overflows usize".to_string())
            })?;
            let total = base_frames.saturating_add(tail_frames);
            let mut safe_name: String = tr
                .name
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                        c
                    } else {
                        '_'
                    }
                })
                .take(64)
                .collect();
            if safe_name.trim_matches('_').is_empty() {
                safe_name = "track".to_string();
            }
            let buf = Self::bounce_offline(&stem_proj, cache, total)?;
            on_stem(ti, &safe_name, &buf)?;
            count += 1;
        }
        Ok(count)
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
        Self::bounce_stems_with(project, cache, |ti, safe_name, buf| {
            let stem_path = output_dir.join(format!("{:02}_{safe_name}.wav", ti + 1));
            buf.save_wav(&stem_path, 24)?;
            written.push(stem_path);
            Ok(())
        })?;
        Ok(written)
    }
}
