//! # `forge-engine` — Real-Time Object-Oriented Multitrack DAW Engine
//!
//! Samplitude-inspired object-oriented audio architecture built on `forge-core`, `forge-dsp`,
//! and `forge-vocal`:
//! - Every [`AudioObject`] carries non-destructive clip bounds, fade-in/out curves (exact across
//!   splits), 3-band EQ, and its own **per-object FX chain** hosting any native ForgeAudio processor.
//! - Per-track zero-allocation **Volume & Pan Automation Lanes** ([`AutomationLane`]).
//! - Automatic intra-track and inter-track **Plugin Delay Compensation (PDC)**.
//! - Real-time hardware playback & input recording via `cpal` ([`LiveAudioEngine`],
//!   [`LiveRecorder`]) with device/project sample-rate conversion plus bit-exact offline bounce
//!   with automatic FX tail preservation and ITU-R BS.1770-4 loudness & True-Peak auditing.

pub mod audio_io;
pub mod fx;
pub mod midi;
pub mod project;
pub mod renderer;

pub use audio_io::{LiveAudioEngine, LiveRecorder};
pub use fx::FxProcessor;
pub use midi::{MidiClip, MidiNote, PolySynth, SynthPreset};
pub use project::{
    AudioObject, AutomationLane, AutomationPoint, AuxBus, DawProject, FadeCurve, FxSlotConfig,
    MAX_PROJECT_JSON_BYTES, MAX_UNDO_DEPTH, ObjectEq3, PROJECT_SCHEMA_VERSION, TimelineMarker,
    Track, TrackSend, UndoStack, WaveformPeaks,
};
pub use renderer::{ClipCache, EngineMeters, ObjectOrientedEngine, resample_buffer_linear};

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
        obj.fade_in_samples = 1200;
        obj.fade_out_samples = 1200;
        obj.fade_curve = FadeCurve::EqualPower;

        assert!((obj.fade_gain_at(0) - 0.0).abs() < 1e-6);
        assert!((obj.fade_gain_at(2400) - 1.0).abs() < 1e-6);
        assert!((obj.fade_gain_at(4800) - 0.0).abs() < 1e-6);

        // Split right in the middle of the fade-in (sample 600 < 1200)
        let (left_in, right_in) = obj.split_at(600, 11).expect("split inside fade-in");
        for s in 0..4800_u64 {
            let orig = obj.fade_gain_at(s);
            let piece = if s < 600 {
                left_in.fade_gain_at(s)
            } else {
                right_in.fade_gain_at(s - 600)
            };
            assert!(
                (orig - piece).abs() < 1e-12,
                "fade-in split mismatch at sample {s}: orig={orig}, piece={piece}"
            );
        }

        // Split right in the middle of the fade-out (sample 4200 > 3600)
        let (left_out, right_out) = obj.split_at(4200, 12).expect("split inside fade-out");
        for s in 0..4800_u64 {
            let orig = obj.fade_gain_at(s);
            let piece = if s < 4200 {
                left_out.fade_gain_at(s)
            } else {
                right_out.fade_gain_at(s - 4200)
            };
            assert!(
                (orig - piece).abs() < 1e-12,
                "fade-out split mismatch at sample {s}: orig={orig}, piece={piece}"
            );
        }
    }

    #[test]
    fn object_oriented_engine_pdc_aligns_parallel_tracks_and_applies_per_object_fx() {
        let sr = 48_000_u32;
        let frames = 4096_usize;
        let mut cache = ClipCache::default();
        let clip_path = PathBuf::from("mem://impulse.wav");

        // Impulse at sample 512 so we can verify exact PDC sample alignment across BOTH tracks
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

        // 1. Check dry track alone lands at sample 512 after PDC
        proj.tracks[0].mute = true;
        proj.tracks[1].mute = false;
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

        // 2. Check FX-processed track alone ALSO lands aligned with sample 512 (within ±1 sample)
        proj.tracks[0].mute = false;
        proj.tracks[1].mute = true;
        let fx_only = ObjectOrientedEngine::bounce_offline(&proj, &cache, frames).unwrap();
        let peak_idx_fx = fx_only.channels[0]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .map(|(i, _)| i)
            .unwrap();
        assert!(
            peak_idx_fx.abs_diff(512) <= 1,
            "FX-processed track impulse should align with sample 512 after PDC, got {peak_idx_fx}"
        );
    }

    #[test]
    fn sample_rate_conversion_preserves_duration_and_pitch() {
        let dir = tempfile::tempdir().unwrap();
        let wav_path = dir.path().join("clip_44k.wav");
        // 1.0 second at 44.1 kHz
        let src = sine_clip(44_100, 44_100, 440.0, 0.5);
        src.save_wav(&wav_path, 24).unwrap();

        let mut cache = ClipCache::default();
        let resampled = cache.load_clip_for_rate(&wav_path, 48_000).unwrap();
        assert_eq!(resampled.sample_rate, 48_000);
        // Should be 1.0 second = 48,000 samples (within ±2 samples)
        assert!(
            resampled.num_frames().abs_diff(48_000) <= 2,
            "expected ~48000 frames, got {}",
            resampled.num_frames()
        );
    }

    #[test]
    fn sync_mix_params_updates_in_place_and_automation_evaluates() {
        let sr = 48_000_u32;
        let mut cache = ClipCache::default();
        let clip_path = PathBuf::from("mem://auto.wav");
        cache.insert_memory_clip(clip_path.clone(), sine_clip(sr, 4096, 440.0, 0.5));

        let mut proj = DawProject::new("Auto Test", sr, 120.0);
        proj.master_inserts.clear();
        proj.tracks[0]
            .objects
            .push(AudioObject::new(10, "Clip", clip_path, 4096));
        proj.tracks[0].volume_auto.enabled = true;
        proj.tracks[0].volume_auto.upsert_point(0, -60.0);
        proj.tracks[0].volume_auto.upsert_point(2048, 0.0);

        let mut engine = ObjectOrientedEngine::compile(&proj, &cache);
        // Changing gain/automation without changing topology must succeed in-place!
        proj.tracks[0].gain_db = -3.0;
        assert!(engine.sync_mix_params(&proj));

        let mut l = vec![0.0_f32; 2048];
        let mut r = vec![0.0_f32; 2048];
        engine.render_block(&mut l, &mut r);
        let early_rms: f32 = l[..128].iter().map(|s| s * s).sum::<f32>() / 128.0;
        let late_rms: f32 = l[1920..].iter().map(|s| s * s).sum::<f32>() / 128.0;
        assert!(
            late_rms > early_rms * 100.0,
            "volume automation should ramp smoothly from -60 dB to 0 dB"
        );
    }

    #[test]
    fn offline_bounce_wav_and_bs1770_audit_succeed_with_reverb_tail() {
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
        obj.fx_chain.push(FxSlotConfig::default_space_verb());
        proj.tracks[0].objects.push(obj);

        let dir = tempfile::tempdir().unwrap();
        let out_wav = dir.path().join("master.wav");
        let report = ObjectOrientedEngine::bounce_to_wav(&proj, &cache, &out_wav).unwrap();
        assert!(out_wav.is_file());
        assert!(report.true_peak_dbtp <= -0.9);
        assert!(report.integrated_lufs.is_finite());

        // Verify reverb tail extended the bounced WAV beyond 1 second!
        let bounced = AudioBuffer::load(&out_wav).unwrap();
        assert!(
            bounced.num_frames() > frames + (sr as usize / 2),
            "bounce_to_wav should preserve reverb decay tail beyond clip end"
        );
    }

    #[test]
    fn meters_copy_from_no_alloc_preserves_capacity() {
        let mut dst = EngineMeters {
            track_peaks_db: Vec::with_capacity(8),
            ..EngineMeters::default()
        };
        let ptr_before = dst.track_peaks_db.as_ptr();
        let src = EngineMeters {
            master_peak_l_db: -3.2,
            master_peak_r_db: -2.8,
            track_peaks_db: vec![-12.0, -6.5, -18.0],
            total_pdc_samples: 144,
            ..EngineMeters::default()
        };
        dst.copy_from_no_alloc(&src);
        assert_eq!(dst.track_peaks_db.as_ptr(), ptr_before);
        assert_eq!(dst.track_peaks_db, vec![-12.0, -6.5, -18.0]);
        assert_eq!(dst.total_pdc_samples, 144);
    }

    #[test]
    fn aux_bus_parallel_send_routing_and_pdc_work() {
        let sr = 48_000_u32;
        let frames = 4096_usize;
        let mut cache = ClipCache::default();
        let clip_path = PathBuf::from("mem://aux_send.wav");
        cache.insert_memory_clip(clip_path.clone(), sine_clip(sr, frames, 440.0, 0.4));

        let mut proj = DawProject::new("Aux Bus Test", sr, 120.0);
        proj.master_inserts.clear();
        proj.tracks[0]
            .objects
            .push(AudioObject::new(10, "Vocal", clip_path, frames as u64));
        // Mute main track fader via -60 dB and send pre-fader at 0 dB to Aux 1 (Hall Verb)
        proj.tracks[0].gain_db = -60.0;
        let aux_id = proj.aux_buses[0].id;
        proj.tracks[0].sends.push(TrackSend {
            bus_id: aux_id,
            gain_db: 0.0,
            pre_fader: true,
            mute: false,
        });

        let wet = ObjectOrientedEngine::bounce_offline(&proj, &cache, frames).unwrap();
        let wet_energy: f64 = wet.channels[0].iter().map(|s| s * s).sum();
        assert!(
            wet_energy > 1.0,
            "pre-fader send to Aux 1 Hall Verb should produce audible wet energy, got {wet_energy}"
        );
    }

    #[test]
    fn track_auto_crossfade_overlaps_applies_equal_power_xfade() {
        let mut tr = Track::new(1, "Vocal Comp", [56, 189, 248]);
        let mut a = AudioObject::new(10, "Take 1", PathBuf::from("t1.wav"), 4800);
        a.timeline_start_samples = 0;
        let mut b = AudioObject::new(11, "Take 2", PathBuf::from("t2.wav"), 4800);
        b.timeline_start_samples = 3600; // 1200-sample overlap with Take 1
        tr.objects.push(a);
        tr.objects.push(b);

        let pairs = tr.auto_crossfade_overlaps();
        assert_eq!(pairs, 1);
        assert_eq!(tr.objects[0].fade_out_samples, 1200);
        assert_eq!(tr.objects[1].fade_in_samples, 1200);
        assert_eq!(tr.objects[0].fade_curve, FadeCurve::EqualPower);
        assert_eq!(tr.objects[1].fade_curve, FadeCurve::EqualPower);
        assert_eq!(tr.overlap_regions(), vec![(3600, 4800)]);
    }

    #[test]
    fn elastic_time_stretch_and_pitch_shift_work() {
        let sr = 48_000_u32;
        let frames = 8192_usize;
        let mut cache = ClipCache::default();
        let clip_path = PathBuf::from("mem://elastic.wav");
        cache.insert_memory_clip(clip_path.clone(), sine_clip(sr, frames, 440.0, 0.5));

        let mut proj = DawProject::new("Elastic Test", sr, 120.0);
        proj.master_inserts.clear();
        let mut obj = AudioObject::new(10, "Stretched", clip_path, frames as u64);
        obj.time_stretch = 0.75;
        obj.pitch_semitones = 5.0;
        proj.tracks[0].objects.push(obj);

        let out = ObjectOrientedEngine::bounce_offline(&proj, &cache, frames).unwrap();
        let energy: f64 = out.channels[0].iter().map(|s| s * s).sum();
        assert!(out.channels[0].iter().all(|s| s.is_finite()));
        assert!(
            energy > 10.0,
            "elastic stretched+pitched audio should produce clean audible output, got {energy}"
        );
    }

    #[test]
    fn midi_poly_synth_renders_polyphonic_chord() {
        let sr = 48_000_u32;
        let cache = ClipCache::default();
        let mut proj = DawProject::new("MIDI Synth Test", sr, 120.0);
        proj.master_inserts.clear();
        proj.tracks[0].synth_preset = SynthPreset::ElectricPiano;
        let cid = proj.alloc_id();
        proj.tracks[0]
            .midi_clips
            .push(MidiClip::demo_am7_progression(cid, sr, 120.0));

        let out = ObjectOrientedEngine::bounce_offline(&proj, &cache, 16_384).unwrap();
        let energy: f64 = out.channels[0].iter().map(|s| s * s).sum();
        assert!(out.channels[0].iter().all(|s| s.is_finite()));
        assert!(
            energy > 5.0,
            "MIDI PolySynth chord progression should produce audible stereo energy, got {energy}"
        );
    }

    #[test]
    fn batch_stem_export_writes_per_track_wavs() {
        let sr = 48_000_u32;
        let frames = 4096_usize;
        let mut cache = ClipCache::default();
        let clip_path = PathBuf::from("mem://stem.wav");
        cache.insert_memory_clip(clip_path.clone(), sine_clip(sr, frames, 220.0, 0.35));

        let mut proj = DawProject::new("Stem Export Test", sr, 120.0);
        proj.tracks[0].objects.push(AudioObject::new(
            10,
            "Vox",
            clip_path.clone(),
            frames as u64,
        ));
        let mid_id = proj.alloc_id();
        proj.tracks[1]
            .midi_clips
            .push(MidiClip::demo_am7_progression(mid_id, sr, 120.0));

        let dir = tempfile::tempdir().unwrap();
        let stems = ObjectOrientedEngine::bounce_stems(&proj, &cache, dir.path()).unwrap();
        assert_eq!(stems.len(), 2);
        for s in &stems {
            assert!(s.is_file(), "expected stem WAV at {}", s.display());
        }
    }

    #[test]
    fn edge_case_fixes_stretched_split_opensoothe_pdc_and_duplicate_stems() {
        // 1. Time-stretched split advances source_offset by left_len * time_stretch
        let mut stretched = AudioObject::new(10, "Loop", PathBuf::from("loop.wav"), 4000);
        stretched.time_stretch = 0.5;
        let (_l, r) = stretched.split_at(2000, 11).expect("split");
        assert_eq!(r.source_offset_samples, 1000);

        // 2. Toggling OpenSoothe low_latency forces sync_mix_params to return false (recompute PDC)
        let mut proj = DawProject::new("PDC Mode Test", 48_000, 120.0);
        proj.tracks[0]
            .inserts
            .push(FxSlotConfig::default_open_soothe());
        let cache = ClipCache::default();
        let mut engine = ObjectOrientedEngine::compile(&proj, &cache);
        if let FxSlotConfig::OpenSoothe {
            ref mut low_latency,
            ..
        } = proj.tracks[0].inserts[0]
        {
            *low_latency = !*low_latency;
        }
        assert!(
            !engine.sync_mix_params(&proj),
            "toggling OpenSoothe low_latency must trigger recompile for PDC"
        );

        // 3. Duplicate track names produce distinct indexed stem WAV files
        let mut dup_proj = DawProject::new("Dup Stem Test", 48_000, 120.0);
        dup_proj.tracks[0].name = "Vocal".to_string();
        dup_proj.tracks[1].name = "Vocal".to_string();
        let mut dup_cache = ClipCache::default();
        let p = PathBuf::from("mem://dup.wav");
        dup_cache.insert_memory_clip(p.clone(), sine_clip(48_000, 2048, 330.0, 0.3));
        dup_proj.tracks[0]
            .objects
            .push(AudioObject::new(1, "A", p.clone(), 2048));
        dup_proj.tracks[1]
            .objects
            .push(AudioObject::new(2, "B", p, 2048));
        let dir = tempfile::tempdir().unwrap();
        let stems = ObjectOrientedEngine::bounce_stems(&dup_proj, &dup_cache, dir.path()).unwrap();
        assert_eq!(stems.len(), 2);
        assert_ne!(stems[0], stems[1]);
        assert!(stems[0].is_file() && stems[1].is_file());
    }
}
