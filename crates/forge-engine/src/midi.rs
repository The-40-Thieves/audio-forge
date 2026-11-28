//! MIDI clip model and zero-allocation 16-voice stereo PolyBLEP synthesizer for ForgeStudio DAW.

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// A single MIDI note event inside a [`MidiClip`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MidiNote {
    pub id: u64,
    /// Offset in samples relative to the start of the enclosing [`MidiClip`].
    pub start_samples: u64,
    pub length_samples: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_beats: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub length_beats: Option<f64>,
    /// MIDI note number (`0..=127`, where `60` = Middle C / C4).
    pub pitch: u8,
    /// Normalized note velocity (`0.0..=1.0`).
    pub velocity: f32,
}

impl MidiNote {
    #[inline]
    pub fn end_samples(&self) -> u64 {
        self.start_samples.saturating_add(self.length_samples)
    }
}

/// Non-destructive MIDI clip placed on a [`crate::Track`] timeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MidiClip {
    pub id: u64,
    pub name: String,
    pub timeline_start_samples: u64,
    pub length_samples: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeline_start_beats: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub length_beats: Option<f64>,
    pub notes: Vec<MidiNote>,
}

impl MidiClip {
    pub fn new(id: u64, name: impl Into<String>, length_samples: u64) -> Self {
        Self {
            id,
            name: name.into(),
            timeline_start_samples: 0,
            length_samples,
            timeline_start_beats: None,
            length_beats: None,
            notes: Vec::new(),
        }
    }

    #[inline]
    pub fn timeline_end_samples(&self) -> u64 {
        self.timeline_start_samples
            .saturating_add(self.length_samples)
    }

    /// Recomputes sample positions/durations from optional musical-beat anchors (`*_beats`).
    pub fn sync_samples_from_beats(&mut self, sample_rate: u32, bpm: f64) {
        let beat_smp = (60.0 / bpm.clamp(20.0, 300.0)) * f64::from(sample_rate.max(8_000));
        if let Some(sb) = self.timeline_start_beats
            && sb.is_finite()
        {
            self.timeline_start_samples = (sb.max(0.0) * beat_smp).round() as u64;
        }
        if let Some(lb) = self.length_beats
            && lb.is_finite()
        {
            self.length_samples = (lb.max(0.01) * beat_smp).round().max(1.0) as u64;
        }
        for n in &mut self.notes {
            if let Some(sb) = n.start_beats
                && sb.is_finite()
            {
                n.start_samples = (sb.max(0.0) * beat_smp).round() as u64;
            }
            if let Some(lb) = n.length_beats
                && lb.is_finite()
            {
                n.length_samples = (lb.max(0.01) * beat_smp).round().max(1.0) as u64;
            }
        }
    }

    /// Creates a 4-bar musical Am7 chord progression MIDI clip at `sample_rate` and `bpm`.
    pub fn demo_am7_progression(id: u64, sample_rate: u32, bpm: f64) -> Self {
        let chords: [&[u8]; 4] = [
            &[57, 60, 64, 67], // Am7
            &[53, 57, 60, 64], // Fmaj7
            &[48, 52, 55, 60], // C
            &[55, 59, 62, 67], // G
        ];
        let mut notes = Vec::with_capacity(16);
        let mut nid = 1_u64;
        for (bi, chord) in chords.iter().enumerate() {
            let start_b = (bi as f64) * 4.0;
            for &pitch in *chord {
                notes.push(MidiNote {
                    id: nid,
                    start_samples: 0,
                    length_samples: 1,
                    start_beats: Some(start_b),
                    length_beats: Some(3.75),
                    pitch,
                    velocity: 0.78,
                });
                nid += 1;
            }
        }
        let mut clip = Self {
            id,
            name: "Am7 Progression".to_string(),
            timeline_start_samples: 0,
            length_samples: 1,
            timeline_start_beats: Some(0.0),
            length_beats: Some(16.0),
            notes,
        };
        clip.sync_samples_from_beats(sample_rate, bpm);
        clip
    }
}

/// Built-in zero-allocation studio instrument presets for [`PolySynth`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SynthPreset {
    #[default]
    WarmSawPad,
    ElectricPiano,
    Sub808Bass,
    PluckLead,
}

impl SynthPreset {
    pub fn label(self) -> &'static str {
        match self {
            Self::WarmSawPad => "Warm Saw Pad",
            Self::ElectricPiano => "Electric Piano",
            Self::Sub808Bass => "Sub 808 Bass",
            Self::PluckLead => "Pluck Lead",
        }
    }

    pub const ALL: [Self; 4] = [
        Self::WarmSawPad,
        Self::ElectricPiano,
        Self::Sub808Bass,
        Self::PluckLead,
    ];
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum EnvStage {
    #[default]
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

#[derive(Clone, Copy)]
struct SynthVoice {
    active: bool,
    gate: bool,
    prev_gate: bool,
    stage: EnvStage,
    note_id: u64,
    pitch: u8,
    velocity: f32,
    phase_a: f64,
    phase_b: f64,
    env: f64,
    lp_l: f64,
    lp_r: f64,
}

impl Default for SynthVoice {
    fn default() -> Self {
        Self {
            active: false,
            gate: false,
            prev_gate: false,
            stage: EnvStage::Idle,
            note_id: 0,
            pitch: 60,
            velocity: 0.0,
            phase_a: 0.0,
            phase_b: 0.0,
            env: 0.0,
            lp_l: 0.0,
            lp_r: 0.0,
        }
    }
}

const MAX_POLY_VOICES: usize = 16;

fn build_midi_freq_table() -> [f64; 128] {
    let mut tbl = [440.0_f64; 128];
    let mut pitch = 0_usize;
    while pitch < 128 {
        tbl[pitch] = 440.0 * 2.0_f64.powf(((pitch as f64) - 69.0) / 12.0);
        pitch += 1;
    }
    tbl
}

/// Zero-allocation 16-voice stereo PolyBLEP synthesizer with ADSR envelopes and LUT frequencies.
pub struct PolySynth {
    sample_rate: f64,
    freq_table: [f64; 128],
    voices: [SynthVoice; MAX_POLY_VOICES],
    cached_preset: Option<SynthPreset>,
    atk_step: f64,
    dec_mul: f64,
    sus_level: f64,
    rel_mul: f64,
    lp_alpha: f64,
    detune_ratio: f64,
}

impl PolySynth {
    pub fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate: sample_rate.max(8_000.0),
            freq_table: build_midi_freq_table(),
            voices: [SynthVoice::default(); MAX_POLY_VOICES],
            cached_preset: None,
            atk_step: 0.001,
            dec_mul: 0.999,
            sus_level: 0.75,
            rel_mul: 0.9999,
            lp_alpha: 0.25,
            detune_ratio: 1.0,
        }
    }

    pub fn reset(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate.max(8_000.0);
        self.voices = [SynthVoice::default(); MAX_POLY_VOICES];
        self.cached_preset = None;
    }

    #[inline]
    fn poly_blep(t: f64, dt: f64) -> f64 {
        if t < dt {
            let x = t / dt;
            x + x - x * x - 1.0
        } else if t > 1.0 - dt {
            let x = (t - 1.0) / dt;
            x * x + x + x + 1.0
        } else {
            0.0
        }
    }

    #[inline]
    fn trigger_note(&mut self, note_id: u64, pitch: u8, velocity: f32) {
        // Keep envelope and oscillator phase continuous when retriggering a releasing voice (click-free).
        for v in &mut self.voices {
            if v.active && v.note_id == note_id {
                v.gate = true;
                if !v.prev_gate {
                    v.velocity = velocity.clamp(0.05, 1.0);
                    v.stage = EnvStage::Attack;
                }
                return;
            }
        }
        let slot = self
            .voices
            .iter()
            .position(|v| !v.active)
            .unwrap_or_else(|| {
                self.voices
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.env.total_cmp(&b.1.env))
                    .map_or(0, |(i, _)| i)
            });
        self.voices[slot] = SynthVoice {
            active: true,
            gate: true,
            prev_gate: false,
            stage: EnvStage::Attack,
            note_id,
            pitch,
            velocity: velocity.clamp(0.05, 1.0),
            phase_a: 0.0,
            phase_b: 0.25,
            env: 0.0,
            lp_l: 0.0,
            lp_r: 0.0,
        };
    }

    #[inline]
    fn ensure_preset_coeffs(&mut self, preset: SynthPreset) {
        if self.cached_preset == Some(preset) {
            return;
        }
        let sr = self.sample_rate;
        let (atk_s, dec_s, sus, rel_s, cutoff_hz, detune_ratio) = match preset {
            SynthPreset::WarmSawPad => (0.045, 0.25, 0.80, 0.35, 2200.0, 1.0045),
            SynthPreset::ElectricPiano => (0.004, 0.45, 0.45, 0.28, 3400.0, 1.0012),
            SynthPreset::Sub808Bass => (0.003, 0.20, 0.65, 0.18, 420.0, 1.0),
            SynthPreset::PluckLead => (0.002, 0.18, 0.30, 0.14, 4800.0, 1.003),
        };
        self.atk_step = 1.0 / (atk_s * sr).max(1.0);
        self.dec_mul = (-1.0 / (dec_s * sr).max(1.0)).exp();
        self.sus_level = sus;
        self.rel_mul = (-1.0 / (rel_s * sr).max(1.0)).exp();
        self.lp_alpha = (2.0 * PI * cutoff_hz / sr).clamp(0.002, 0.92);
        self.detune_ratio = detune_ratio;
        self.cached_preset = Some(preset);
    }

    #[inline]
    fn advance_adsr(&self, v: &mut SynthVoice) -> bool {
        if !v.gate {
            v.stage = EnvStage::Release;
            v.env *= self.rel_mul;
            if v.env < 1e-4 {
                v.active = false;
                v.stage = EnvStage::Idle;
                v.env = 0.0;
                return false;
            }
            return true;
        }
        match v.stage {
            EnvStage::Idle | EnvStage::Attack => {
                v.env = (v.env + self.atk_step).min(1.0);
                if v.env >= 1.0 - 1e-6 {
                    v.env = 1.0;
                    v.stage = EnvStage::Decay;
                }
            }
            EnvStage::Decay => {
                v.env = self.sus_level + (v.env - self.sus_level) * self.dec_mul;
                if (v.env - self.sus_level).abs() < 1e-4 {
                    v.env = self.sus_level;
                    v.stage = EnvStage::Sustain;
                }
            }
            EnvStage::Sustain | EnvStage::Release => {
                v.stage = EnvStage::Sustain;
                v.env = self.sus_level;
            }
        }
        true
    }

    #[inline]
    fn synth_voice_sample(
        v: &mut SynthVoice,
        preset: SynthPreset,
        dt_a: f64,
        dt_b: f64,
    ) -> (f64, f64) {
        let (raw_l, raw_r) = match preset {
            SynthPreset::WarmSawPad | SynthPreset::PluckLead => {
                let saw_a = 2.0 * v.phase_a - 1.0 - Self::poly_blep(v.phase_a, dt_a);
                let saw_b = 2.0 * v.phase_b - 1.0 - Self::poly_blep(v.phase_b, dt_b);
                (saw_a * 0.7 + saw_b * 0.3, saw_b * 0.7 + saw_a * 0.3)
            }
            SynthPreset::ElectricPiano => {
                let s1 = (2.0 * PI * v.phase_a).sin();
                let s2 = 0.35 * (4.0 * PI * v.phase_a).sin() * v.env;
                let s3 = 0.15 * (6.0 * PI * v.phase_b).sin();
                let sig = s1 + s2 + s3;
                (sig, sig * 0.96)
            }
            SynthPreset::Sub808Bass => {
                let sub = (2.0 * PI * v.phase_a).sin();
                let sat = (sub * 1.6).tanh();
                (sat, sat)
            }
        };
        v.phase_a += dt_a;
        if v.phase_a >= 1.0 {
            v.phase_a -= 1.0;
        }
        v.phase_b += dt_b;
        if v.phase_b >= 1.0 {
            v.phase_b -= 1.0;
        }
        (raw_l, raw_r)
    }

    /// Evaluates active notes in `clips` at `timeline_pos` and renders one stereo sample
    /// with zero heap allocations.
    #[inline]
    pub fn render_sample(
        &mut self,
        clips: &[MidiClip],
        preset: SynthPreset,
        timeline_pos: u64,
    ) -> (f32, f32) {
        if clips.is_empty() && !self.voices.iter().any(|v| v.active) {
            return (0.0, 0.0);
        }
        for v in &mut self.voices {
            v.prev_gate = v.gate;
            v.gate = false;
        }
        for clip in clips {
            let c_start = clip.timeline_start_samples;
            let c_end = clip.timeline_end_samples();
            if timeline_pos >= c_start && timeline_pos < c_end {
                let local = timeline_pos - c_start;
                for note in &clip.notes {
                    if local >= note.start_samples && local < note.end_samples() {
                        let key = (clip.id << 32) ^ note.id;
                        self.trigger_note(key, note.pitch, note.velocity);
                    }
                }
            }
        }
        if !self.voices.iter().any(|v| v.active) {
            return (0.0, 0.0);
        }

        self.ensure_preset_coeffs(preset);
        let sr = self.sample_rate;
        let lp_alpha = self.lp_alpha;
        let detune_ratio = self.detune_ratio;
        let mut out_l = 0.0_f64;
        let mut out_r = 0.0_f64;

        for idx in 0..MAX_POLY_VOICES {
            let mut voice = self.voices[idx];
            if !voice.active || !self.advance_adsr(&mut voice) {
                self.voices[idx] = voice;
                continue;
            }
            let freq = self.freq_table[usize::from(voice.pitch.min(127))];
            let dt_a = (freq / sr).clamp(1e-6, 0.45);
            let dt_b = ((freq * detune_ratio) / sr).clamp(1e-6, 0.45);
            let (raw_l, raw_r) = Self::synth_voice_sample(&mut voice, preset, dt_a, dt_b);
            voice.lp_l += lp_alpha * (raw_l - voice.lp_l);
            voice.lp_r += lp_alpha * (raw_r - voice.lp_r);
            let amp = 0.22 * f64::from(voice.velocity) * voice.env;
            out_l += voice.lp_l * amp;
            out_r += voice.lp_r * amp;
            self.voices[idx] = voice;
        }
        (out_l as f32, out_r as f32)
    }
}
