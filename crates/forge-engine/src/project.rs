//! Object-Oriented DAW Project Model (`.forgeproj`), Samplitude-style `AudioObject`,
//! multi-resolution waveform peak summaries, and bounded undo/redo command history.

use std::path::{Path, PathBuf};

use forge_core::{AudioBuffer, ForgeError, Result};
use serde::{Deserialize, Serialize};

/// Current `.forgeproj` JSON schema version.
pub const PROJECT_SCHEMA_VERSION: u32 = 1;
/// Maximum `.forgeproj` JSON file size accepted (16 MiB) to prevent memory exhaustion.
pub const MAX_PROJECT_JSON_BYTES: usize = 16 * 1024 * 1024;
/// Maximum undo history depth kept in memory.
pub const MAX_UNDO_DEPTH: usize = 64;

/// Fade-in / fade-out envelope shape for an [`AudioObject`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum FadeCurve {
    Linear,
    #[default]
    EqualPower,
    Exponential,
    Logarithmic,
}

impl FadeCurve {
    /// Evaluates fade gain in `0.0..=1.0` for normalized progress `t` in `0.0..=1.0`.
    #[inline]
    pub fn gain(self, t: f64) -> f64 {
        let u = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => u,
            Self::EqualPower => (u * std::f64::consts::FRAC_PI_2).sin(),
            Self::Exponential => u * u,
            Self::Logarithmic => u.sqrt(),
        }
    }
}

/// Built-in 3-band per-object studio EQ (Low 120 Hz, Mid 1.2 kHz, High 6.5 kHz, in dB).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ObjectEq3 {
    pub low_db: f32,
    pub mid_db: f32,
    pub high_db: f32,
}

impl Default for ObjectEq3 {
    fn default() -> Self {
        Self {
            low_db: 0.0,
            mid_db: 0.0,
            high_db: 0.0,
        }
    }
}

impl ObjectEq3 {
    #[inline]
    pub fn is_flat(&self) -> bool {
        self.low_db.abs() < 0.05 && self.mid_db.abs() < 0.05 && self.high_db.abs() < 0.05
    }
}

/// Native ForgeAudio processor slot attachable to any **Object**, **Track**, or **Master Bus**.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FxSlotConfig {
    GainTrim {
        bypass: bool,
        gain_db: f32,
    },
    OpenSoothe {
        bypass: bool,
        sensitivity_db: f32,
        max_depth_db: f32,
        sharpness_bins: usize,
        low_latency: bool,
        tilt_db_per_oct: f32,
    },
    VocalSurgical {
        bypass: bool,
        declick: bool,
        deplosive: bool,
        deess_thresh_db: f32,
        gate_thresh_db: f32,
    },
    SubMonoTape {
        bypass: bool,
        mono_cutoff_hz: f32,
        sub_reinforce: f32,
        stereo_width: f32,
        tape_drive: f32,
    },
    MasterLimiter {
        bypass: bool,
        drive_db: f32,
        ceiling_dbtp: f32,
        release_ms: f32,
        isp_8x: bool,
    },
    SplitEq {
        bypass: bool,
        tr_low_db: f32,
        tr_mid_db: f32,
        tr_high_db: f32,
        tn_low_db: f32,
        tn_mid_db: f32,
        tn_high_db: f32,
    },
    DensityClipper {
        bypass: bool,
        density: f32,
        clip_drive_db: f32,
        ceiling_db: f32,
    },
    SpaceVerb {
        bypass: bool,
        decay_rt60_secs: f32,
        damping: f32,
        shimmer: f32,
        ducking: f32,
        wet_mix: f32,
    },
    VocalTune {
        bypass: bool,
        key_root: u8,
        scale_mode: u8,
        retune_speed_ms: f32,
        retune_amount: f32,
        humanize: f32,
        formant_shift_st: f32,
        wet_mix: f32,
    },
}

impl FxSlotConfig {
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::GainTrim { .. } => "Gain Trim",
            Self::OpenSoothe { .. } => "Forge OpenSoothe",
            Self::VocalSurgical { .. } => "Forge Vocal Surgical",
            Self::SubMonoTape { .. } => "Forge SubMono & Tape",
            Self::MasterLimiter { .. } => "Forge Master Limiter",
            Self::SplitEq { .. } => "Forge SplitEQ",
            Self::DensityClipper { .. } => "Forge Density & Clipper",
            Self::SpaceVerb { .. } => "Forge SpaceVerb FDN",
            Self::VocalTune { .. } => "Forge Vocal Tune",
        }
    }

    pub fn is_bypassed(&self) -> bool {
        match self {
            Self::GainTrim { bypass, .. }
            | Self::OpenSoothe { bypass, .. }
            | Self::VocalSurgical { bypass, .. }
            | Self::SubMonoTape { bypass, .. }
            | Self::MasterLimiter { bypass, .. }
            | Self::SplitEq { bypass, .. }
            | Self::DensityClipper { bypass, .. }
            | Self::SpaceVerb { bypass, .. }
            | Self::VocalTune { bypass, .. } => *bypass,
        }
    }

    pub fn set_bypass(&mut self, value: bool) {
        match self {
            Self::GainTrim { bypass, .. }
            | Self::OpenSoothe { bypass, .. }
            | Self::VocalSurgical { bypass, .. }
            | Self::SubMonoTape { bypass, .. }
            | Self::MasterLimiter { bypass, .. }
            | Self::SplitEq { bypass, .. }
            | Self::DensityClipper { bypass, .. }
            | Self::SpaceVerb { bypass, .. }
            | Self::VocalTune { bypass, .. } => *bypass = value,
        }
    }

    pub fn default_open_soothe() -> Self {
        Self::OpenSoothe {
            bypass: false,
            sensitivity_db: 3.5,
            max_depth_db: 9.0,
            sharpness_bins: 10,
            low_latency: true,
            tilt_db_per_oct: 0.0,
        }
    }

    pub fn default_master_limiter() -> Self {
        Self::MasterLimiter {
            bypass: false,
            drive_db: 0.0,
            ceiling_dbtp: -1.0,
            release_ms: 85.0,
            isp_8x: false,
        }
    }

    pub fn default_space_verb() -> Self {
        Self::SpaceVerb {
            bypass: false,
            decay_rt60_secs: 2.2,
            damping: 0.4,
            shimmer: 0.1,
            ducking: 0.45,
            wet_mix: 0.22,
        }
    }

    pub fn default_vocal_tune() -> Self {
        Self::VocalTune {
            bypass: false,
            key_root: 0,
            scale_mode: 1,
            retune_speed_ms: 25.0,
            retune_amount: 0.85,
            humanize: 0.15,
            formant_shift_st: 0.0,
            wet_mix: 1.0,
        }
    }

    pub fn default_vocal_surgical() -> Self {
        Self::VocalSurgical {
            bypass: false,
            declick: true,
            deplosive: true,
            deess_thresh_db: -18.0,
            gate_thresh_db: -48.0,
        }
    }

    pub fn default_sub_mono_tape() -> Self {
        Self::SubMonoTape {
            bypass: false,
            mono_cutoff_hz: 150.0,
            sub_reinforce: 0.2,
            stereo_width: 1.0,
            tape_drive: 0.25,
        }
    }

    pub fn default_split_eq() -> Self {
        Self::SplitEq {
            bypass: false,
            tr_low_db: 0.0,
            tr_mid_db: 0.0,
            tr_high_db: 0.0,
            tn_low_db: 0.0,
            tn_mid_db: 0.0,
            tn_high_db: 0.0,
        }
    }

    pub fn default_density_clipper() -> Self {
        Self::DensityClipper {
            bypass: false,
            density: 0.3,
            clip_drive_db: 2.0,
            ceiling_db: -0.5,
        }
    }
}

/// Samplitude-style **Audio Object** living on a track timeline.
///
/// Every object carries its own non-destructive clip bounds, fades, gain/pan/polarity,
/// 3-band EQ, and **per-object FX chain** independent of the host track.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioObject {
    pub id: u64,
    pub name: String,
    pub source_path: PathBuf,
    pub timeline_start_samples: u64,
    pub source_offset_samples: u64,
    pub length_samples: u64,
    pub gain_db: f32,
    pub pan: f32,
    pub phase_invert: bool,
    pub mute: bool,
    pub fade_in_samples: u64,
    #[serde(default)]
    pub fade_in_offset_samples: u64,
    pub fade_out_samples: u64,
    #[serde(default)]
    pub fade_out_offset_samples: u64,
    pub fade_curve: FadeCurve,
    pub eq: ObjectEq3,
    pub fx_chain: Vec<FxSlotConfig>,
}

impl AudioObject {
    pub fn new(
        id: u64,
        name: impl Into<String>,
        source_path: PathBuf,
        length_samples: u64,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            source_path,
            timeline_start_samples: 0,
            source_offset_samples: 0,
            length_samples,
            gain_db: 0.0,
            pan: 0.0,
            phase_invert: false,
            mute: false,
            fade_in_samples: 0,
            fade_in_offset_samples: 0,
            fade_out_samples: 0,
            fade_out_offset_samples: 0,
            fade_curve: FadeCurve::EqualPower,
            eq: ObjectEq3::default(),
            fx_chain: Vec::new(),
        }
    }

    #[inline]
    pub fn timeline_end_samples(&self) -> u64 {
        self.timeline_start_samples
            .saturating_add(self.length_samples)
    }

    /// Evaluates the combined fade-in / fade-out gain at `local_sample` (`0..length_samples`),
    /// honoring split fade offsets so splitting inside an active fade preserves the exact envelope.
    #[inline]
    pub fn fade_gain_at(&self, local_sample: u64) -> f64 {
        let mut g = 1.0_f64;
        if self.fade_in_samples > 0 {
            let eff_pos = local_sample.saturating_add(self.fade_in_offset_samples);
            if eff_pos < self.fade_in_samples {
                let t = (eff_pos as f64) / (self.fade_in_samples as f64);
                g *= self.fade_curve.gain(t);
            }
        }
        if self.fade_out_samples > 0 && self.length_samples > 0 {
            let remaining = self
                .length_samples
                .saturating_sub(local_sample)
                .saturating_add(self.fade_out_offset_samples);
            if remaining <= self.fade_out_samples {
                let t = (remaining as f64) / (self.fade_out_samples as f64);
                g *= self.fade_curve.gain(t);
            }
        }
        g
    }

    /// Splits this object non-destructively at `split_timeline_sample`, returning `(left, right)`
    /// or `None` if `split_timeline_sample` is not strictly inside the object.
    ///
    /// Preserves the exact original fade-in and fade-out envelopes even when splitting inside an
    /// active fade region.
    pub fn split_at(&self, split_timeline_sample: u64, new_right_id: u64) -> Option<(Self, Self)> {
        if split_timeline_sample <= self.timeline_start_samples
            || split_timeline_sample >= self.timeline_end_samples()
        {
            return None;
        }
        let left_len = split_timeline_sample - self.timeline_start_samples;
        let right_len = self.length_samples - left_len;

        let mut left = self.clone();
        left.length_samples = left_len;
        let left_out_offset = self.fade_out_offset_samples.saturating_add(right_len);
        if self.fade_out_samples > 0 && left_out_offset < self.fade_out_samples {
            left.fade_out_samples = self.fade_out_samples;
            left.fade_out_offset_samples = left_out_offset;
        } else {
            left.fade_out_samples = 0;
            left.fade_out_offset_samples = 0;
        }

        let mut right = self.clone();
        right.id = new_right_id;
        right.name = format!("{} (R)", self.name);
        right.timeline_start_samples = split_timeline_sample;
        right.source_offset_samples = self.source_offset_samples.saturating_add(left_len);
        right.length_samples = right_len;
        let right_in_offset = self.fade_in_offset_samples.saturating_add(left_len);
        if self.fade_in_samples > 0 && right_in_offset < self.fade_in_samples {
            right.fade_in_samples = self.fade_in_samples;
            right.fade_in_offset_samples = right_in_offset;
        } else {
            right.fade_in_samples = 0;
            right.fade_in_offset_samples = 0;
        }

        Some((left, right))
    }
}

/// A single time-stamped automation breakpoint on a track envelope lane.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AutomationPoint {
    pub sample_pos: u64,
    pub value: f32,
}

/// Piecewise-linear automation lane evaluated with zero heap allocations on the audio thread.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct AutomationLane {
    pub enabled: bool,
    pub points: Vec<AutomationPoint>,
}

impl AutomationLane {
    /// Inserts or updates a breakpoint at `sample_pos` and keeps `points` sorted by time.
    pub fn upsert_point(&mut self, sample_pos: u64, value: f32) {
        if let Some(existing) = self
            .points
            .iter_mut()
            .find(|p| p.sample_pos.abs_diff(sample_pos) <= 128)
        {
            existing.sample_pos = sample_pos;
            existing.value = value;
        } else {
            self.points.push(AutomationPoint { sample_pos, value });
        }
        self.points.sort_by_key(|p| p.sample_pos);
    }

    /// Evaluates the lane at `sample_pos` using binary search + linear interpolation (zero alloc).
    #[inline]
    pub fn evaluate(&self, sample_pos: u64, fallback: f32) -> f32 {
        if !self.enabled || self.points.is_empty() {
            return fallback;
        }
        if self.points.len() == 1 || sample_pos <= self.points[0].sample_pos {
            return self.points[0].value;
        }
        let last = self.points[self.points.len() - 1];
        if sample_pos >= last.sample_pos {
            return last.value;
        }
        let idx = self.points.partition_point(|p| p.sample_pos <= sample_pos);
        let p0 = self.points[idx.saturating_sub(1)];
        let p1 = self.points[idx];
        let span = p1.sample_pos.saturating_sub(p0.sample_pos);
        if span == 0 {
            return p1.value;
        }
        let t = (sample_pos - p0.sample_pos) as f32 / span as f32;
        p0.value + (p1.value - p0.value) * t
    }
}

/// A multitrack audio lane containing zero or more [`AudioObject`]s and track-level FX inserts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Track {
    pub id: u64,
    pub name: String,
    pub color_rgb: [u8; 3],
    pub gain_db: f32,
    pub pan: f32,
    pub mute: bool,
    pub solo: bool,
    pub phase_invert: bool,
    #[serde(default)]
    pub record_armed: bool,
    #[serde(default)]
    pub show_automation: bool,
    #[serde(default)]
    pub volume_auto: AutomationLane,
    #[serde(default)]
    pub pan_auto: AutomationLane,
    pub objects: Vec<AudioObject>,
    pub inserts: Vec<FxSlotConfig>,
}

impl Track {
    pub fn new(id: u64, name: impl Into<String>, color_rgb: [u8; 3]) -> Self {
        Self {
            id,
            name: name.into(),
            color_rgb,
            gain_db: 0.0,
            pan: 0.0,
            mute: false,
            solo: false,
            phase_invert: false,
            record_armed: false,
            show_automation: false,
            volume_auto: AutomationLane::default(),
            pan_auto: AutomationLane::default(),
            objects: Vec::new(),
            inserts: Vec::new(),
        }
    }
}

/// Timeline arrangement marker (e.g., `[Intro]`, `[Chorus]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimelineMarker {
    pub name: String,
    pub sample_pos: u64,
}

/// Complete ForgeStudio DAW project document (`.forgeproj`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DawProject {
    pub schema_version: u32,
    pub name: String,
    pub sample_rate: u32,
    pub bpm: f64,
    pub time_sig_num: u8,
    pub time_sig_den: u8,
    pub loop_enabled: bool,
    pub loop_start_samples: u64,
    pub loop_end_samples: u64,
    pub master_gain_db: f32,
    pub master_inserts: Vec<FxSlotConfig>,
    pub tracks: Vec<Track>,
    pub markers: Vec<TimelineMarker>,
    next_id: u64,
}

const TRACK_PALETTE: [[u8; 3]; 6] = [
    [56, 189, 248],  // Cyan
    [74, 222, 128],  // Emerald
    [251, 146, 60],  // Amber
    [192, 132, 252], // Violet
    [248, 113, 113], // Coral
    [250, 204, 21],  // Gold
];

impl Default for DawProject {
    fn default() -> Self {
        Self::new("Untitled ForgeStudio Session", 48_000, 120.0)
    }
}

impl DawProject {
    pub fn new(name: impl Into<String>, sample_rate: u32, bpm: f64) -> Self {
        let sr = sample_rate.clamp(8_000, 384_000);
        let clean_bpm = if bpm.is_finite() {
            bpm.clamp(30.0, 300.0)
        } else {
            120.0
        };
        Self {
            schema_version: PROJECT_SCHEMA_VERSION,
            name: name.into(),
            sample_rate: sr,
            bpm: clean_bpm,
            time_sig_num: 4,
            time_sig_den: 4,
            loop_enabled: false,
            loop_start_samples: 0,
            loop_end_samples: u64::from(sr) * 8,
            master_gain_db: 0.0,
            master_inserts: vec![FxSlotConfig::default_master_limiter()],
            tracks: vec![
                Track::new(1, "01 Lead Vocal", TRACK_PALETTE[0]),
                Track::new(2, "02 Instrumental / Beat", TRACK_PALETTE[1]),
            ],
            markers: vec![TimelineMarker {
                name: "Start".to_string(),
                sample_pos: 0,
            }],
            next_id: 10,
        }
    }

    pub fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        id
    }

    pub fn add_track(&mut self, name: impl Into<String>) -> u64 {
        let id = self.alloc_id();
        let color = TRACK_PALETTE[self.tracks.len() % TRACK_PALETTE.len()];
        self.tracks.push(Track::new(id, name, color));
        id
    }

    /// Total timeline length in samples across all objects (minimum 10 seconds).
    pub fn total_length_samples(&self) -> u64 {
        let max_end = self
            .tracks
            .iter()
            .flat_map(|t| t.objects.iter())
            .map(AudioObject::timeline_end_samples)
            .max()
            .unwrap_or(0);
        max_end.max(u64::from(self.sample_rate) * 10)
    }

    /// Finds `(track_index, object_index)` by `object_id`.
    pub fn find_object(&self, object_id: u64) -> Option<(usize, usize)> {
        for (ti, track) in self.tracks.iter().enumerate() {
            if let Some(oi) = track.objects.iter().position(|o| o.id == object_id) {
                return Some((ti, oi));
            }
        }
        None
    }

    pub fn object_mut(&mut self, object_id: u64) -> Option<&mut AudioObject> {
        let (ti, oi) = self.find_object(object_id)?;
        Some(&mut self.tracks[ti].objects[oi])
    }

    /// Splits the object with `object_id` at `split_sample` in place, returning the new right ID.
    pub fn split_object(&mut self, object_id: u64, split_sample: u64) -> Option<u64> {
        let (ti, oi) = self.find_object(object_id)?;
        let right_id = self.alloc_id();
        let (left, right) = self.tracks[ti].objects[oi].split_at(split_sample, right_id)?;
        self.tracks[ti].objects[oi] = left;
        self.tracks[ti].objects.insert(oi + 1, right);
        Some(right_id)
    }

    /// Serializes the project to pretty-printed `.forgeproj` JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(ForgeError::from)
    }

    /// Deserializes a `.forgeproj` JSON string with strict size & parameter validation.
    pub fn from_json(raw: &str) -> Result<Self> {
        if raw.len() > MAX_PROJECT_JSON_BYTES {
            return Err(ForgeError::InvalidParameter(format!(
                "Project JSON exceeds maximum size ({MAX_PROJECT_JSON_BYTES} bytes)"
            )));
        }
        let mut proj: Self = serde_json::from_str(raw)?;
        proj.sample_rate = proj.sample_rate.clamp(8_000, 384_000);
        if !proj.bpm.is_finite() {
            proj.bpm = 120.0;
        }
        proj.bpm = proj.bpm.clamp(30.0, 300.0);
        Ok(proj)
    }

    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let json = self.to_json()?;
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load_from_file(path: &Path) -> Result<Self> {
        let meta = std::fs::metadata(path)?;
        if meta.len() as usize > MAX_PROJECT_JSON_BYTES {
            return Err(ForgeError::InvalidParameter(format!(
                "Project file '{}' exceeds maximum size ({MAX_PROJECT_JSON_BYTES} bytes)",
                path.display()
            )));
        }
        let raw = std::fs::read_to_string(path)?;
        Self::from_json(&raw)
    }
}

/// Bounded undo/redo snapshot stack for non-destructive DAW editing.
#[derive(Debug, Clone, Default)]
pub struct UndoStack {
    past: Vec<DawProject>,
    future: Vec<DawProject>,
}

impl UndoStack {
    pub fn push(&mut self, snapshot: DawProject) {
        self.future.clear();
        if self.past.len() >= MAX_UNDO_DEPTH {
            self.past.remove(0);
        }
        self.past.push(snapshot);
    }

    pub fn undo(&mut self, current: DawProject) -> Option<DawProject> {
        let prev = self.past.pop()?;
        self.future.push(current);
        Some(prev)
    }

    pub fn redo(&mut self, current: DawProject) -> Option<DawProject> {
        let next = self.future.pop()?;
        self.past.push(current);
        Some(next)
    }

    pub fn can_undo(&self) -> bool {
        !self.past.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.future.is_empty()
    }
}

/// Precomputed min/max waveform envelope buckets for fast 60 FPS GPU timeline rendering.
#[derive(Debug, Clone)]
pub struct WaveformPeaks {
    pub samples_per_bucket: usize,
    pub min_max: Vec<(f32, f32)>,
}

impl WaveformPeaks {
    pub fn from_buffer(buf: &AudioBuffer, samples_per_bucket: usize) -> Self {
        let step = samples_per_bucket.max(1);
        let frames = buf.num_frames();
        let n_buckets = frames.div_ceil(step);
        let mut min_max = Vec::with_capacity(n_buckets);
        let ch0 = buf.channels.first();
        let ch1 = buf.channels.get(1);
        for b in 0..n_buckets {
            let start = b * step;
            let end = (start + step).min(frames);
            let mut mn = 0.0_f32;
            let mut mx = 0.0_f32;
            for i in start..end {
                let s0 = ch0.map_or(0.0, |c| c[i]) as f32;
                let s1 = ch1.map_or(s0, |c| c[i] as f32);
                let s = 0.5 * (s0 + s1);
                if s.is_finite() {
                    mn = mn.min(s);
                    mx = mx.max(s);
                }
            }
            min_max.push((mn.clamp(-1.0, 1.0), mx.clamp(-1.0, 1.0)));
        }
        Self {
            samples_per_bucket: step,
            min_max,
        }
    }
}
