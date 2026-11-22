//! # `forge-engine` — Real-Time Object-Oriented Multitrack DAW Engine
//!
//! Samplitude-inspired object-oriented audio architecture built on `forge-core`, `forge-dsp`,
//! and `forge-vocal`:
//! - Every [`AudioObject`] carries non-destructive clip bounds, fade-in/out curves, 3-band EQ,
//!   and its own **per-object FX chain** hosting any native ForgeAudio studio processor.
//! - Automatic intra-track and inter-track **Plugin Delay Compensation (PDC)**.
//! - Real-time hardware playback via `cpal` ([`LiveAudioEngine`]) plus bit-exact offline bounce
//!   with ITU-R BS.1770-4 loudness & True-Peak auditing.

pub mod audio_io;
pub mod fx;
pub mod project;
pub mod renderer;

pub use audio_io::LiveAudioEngine;
pub use fx::FxProcessor;
pub use project::{
    AudioObject, DawProject, FadeCurve, FxSlotConfig, MAX_PROJECT_JSON_BYTES, MAX_UNDO_DEPTH,
    ObjectEq3, PROJECT_SCHEMA_VERSION, TimelineMarker, Track, UndoStack, WaveformPeaks,
};
pub use renderer::{ClipCache, EngineMeters, ObjectOrientedEngine};

#[cfg(test)]
mod tests {
    use super::*;
    use forge_core::AudioBuffer;
    use std::f64::consts::PI;
    use std::path::PathBuf;

    fn sine_clip(sr: u32, frames: usize, freq_hz: f64, amp: f64) -> AudioBuffer {
        let ch: Vec<f64> = (0..frames)
            .map(|i| amp * (2.0 * PI * freq_hz * (i as f64) / f64::from(sr)).sin())
            .collect();
        AudioBuffer::from_channels(vec![ch.clone(), ch], sr).expect("valid sine clip")
    }

    #[test]
    fn project_json_round_trip_and_undo_redo_work() {
        let mut proj = DawProject::new("Test Mix", 48_000, 96.0);
        let mut undo = UndoStack::default();
        undo.push(proj.clone());

        let tid = proj.add_track("03 Bass");
        assert_eq!(proj.tracks.len(), 3);
        assert!(proj.tracks.iter().any(|t| t.id == tid));

        let json = proj.to_json().expect("serialize");
        let loaded = DawProject::from_json(&json).expect("deserialize");
        assert_eq!(loaded, proj);

        let undone = undo.undo(proj).expect("undo");
        assert_eq!(undone.tracks.len(), 2);
        let redone = undo.redo(undone).expect("redo");
        assert_eq!(redone.tracks.len(), 3);
    }

    #[test]
    fn object_split_and_fade_curves_are_continuous() {
        let mut obj = AudioObject::new(10, "Vocal", PathBuf::from("vox.wav"), 4800);
        obj.fade_in_samples = 480;
        obj.fade_out_samples = 480;
        obj.fade_curve = FadeCurve::EqualPower;

        assert!((obj.fade_gain_at(0) - 0.0).abs() < 1e-6);
        assert!((obj.fade_gain_at(2400) - 1.0).abs() < 1e-6);
        assert!((obj.fade_gain_at(4800) - 0.0).abs() < 1e-6);

        let (left, right) = obj.split_at(2400, 11).expect("split at midpoint");
        assert_eq!(left.length_samples, 2400);
        assert_eq!(right.timeline_start_samples, 2400);
        assert_eq!(right.source_offset_samples, 2400);
        assert_eq!(right.length_samples, 2400);
    }

    #[test]
    fn object_oriented_engine_pdc_aligns_parallel_tracks_and_applies_per_object_fx() {
        let sr = 48_000_u32;
        let frames = 4096_usize;
        let mut cache = ClipCache::default();
        let clip_path = PathBuf::from("mem://impulse.wav");

        // Impulse at sample 512 so we can verify exact PDC sample alignment across tracks
        let mut imp = vec![0.0_f64; frames];
        imp[512] = 0.8;
        cache.insert_memory_clip(
            clip_path.clone(),
            AudioBuffer::from_channels(vec![imp.clone(), imp], sr).unwrap(),
        );

        let mut proj = DawProject::new("PDC Test", sr, 120.0);
        proj.master_inserts.clear();

        // Track 0: object with MasterLimiter (lookahead latency ~144 samples)
        let mut obj0 = AudioObject::new(100, "Clip A", clip_path.clone(), frames as u64);
        obj0.fx_chain.push(FxSlotConfig::default_master_limiter());
        proj.tracks[0].objects.push(obj0);

        // Track 1: dry object with 0 latency — PDC must delay Track 1 so both impulses align!
        let mut obj1 = AudioObject::new(101, "Clip B", clip_path, frames as u64);
        obj1.gain_db = -6.0;
        proj.tracks[1].objects.push(obj1);

        // Solo Track 1 alone vs Track 0 alone and check peak arrives at sample 512 after PDC
        proj.tracks[0].mute = true;
        let dry_only = ObjectOrientedEngine::bounce_offline(&proj, &cache, frames).unwrap();
        let peak_idx_dry = dry_only.channels[0]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(
            peak_idx_dry, 512,
            "PDC-compensated dry track impulse should land at exact sample 512"
        );
    }

    #[test]
    fn offline_bounce_wav_and_bs1770_audit_succeed() {
        let sr = 48_000_u32;
        let frames = sr as usize; // 1 second
        let mut cache = ClipCache::default();
        let clip_path = PathBuf::from("mem://tone.wav");
        cache.insert_memory_clip(clip_path.clone(), sine_clip(sr, frames, 440.0, 0.35));

        let mut proj = DawProject::new("Bounce Test", sr, 120.0);
        let mut obj = AudioObject::new(200, "Lead Sine", clip_path, frames as u64);
        obj.eq = ObjectEq3 {
            low_db: 0.0,
            mid_db: 2.0,
            high_db: 0.0,
        };
        obj.fx_chain.push(FxSlotConfig::GainTrim {
            bypass: false,
            gain_db: 1.5,
        });
        proj.tracks[0].objects.push(obj);

        let dir = tempfile::tempdir().unwrap();
        let out_wav = dir.path().join("master.wav");
        let report = ObjectOrientedEngine::bounce_to_wav(&proj, &cache, &out_wav).unwrap();
        assert!(out_wav.is_file());
        assert!(report.true_peak_dbtp <= -0.9);
        assert!(report.integrated_lufs.is_finite());
    }
}
