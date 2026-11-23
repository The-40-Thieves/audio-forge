//! Verifies that `ObjectOrientedEngine::render_block` performs zero heap allocations
//! on the real-time audio path, including per-object 3-band EQ, per-object FX chains,
//! PDC delay lines, track inserts, master bus FX, and 32-band spectrum analysis.

use assert_no_alloc::{AllocDisabler, assert_no_alloc, violation_count};
use forge_core::AudioBuffer;
use forge_engine::{
    AudioObject, ClipCache, DawProject, FxSlotConfig, MidiClip, ObjectOrientedEngine, SynthPreset,
    TrackSend,
};
use std::path::PathBuf;

#[global_allocator]
static A: AllocDisabler = AllocDisabler;

#[test]
fn object_oriented_engine_render_block_is_zero_alloc() {
    let mut project = DawProject::new("No-Alloc RT Verification", 48_000, 120.0);
    let clip_path = PathBuf::from("synthetic://rt-test.wav");

    let mut cache = ClipCache::default();
    let mut l = vec![0.0_f64; 8192];
    let mut r = vec![0.0_f64; 8192];
    for i in 0..8192 {
        let s = 0.25 * (2.0 * std::f64::consts::PI * 220.0 * i as f64 / 48_000.0).sin();
        l[i] = s;
        r[i] = s;
    }
    cache.insert_memory_clip(
        clip_path.clone(),
        AudioBuffer::from_channels(vec![l, r], 48_000).expect("valid buffer"),
    );

    let obj_id = project.alloc_id();
    let mut obj = AudioObject::new(obj_id, "Lead Vocal", clip_path.clone(), 8192);
    obj.time_stretch = 0.85;
    obj.pitch_semitones = 3.0;
    obj.eq.low_db = 1.5;
    obj.eq.mid_db = -1.0;
    obj.eq.high_db = 2.0;
    obj.fx_chain.push(FxSlotConfig::default_open_soothe());
    obj.fx_chain.push(FxSlotConfig::default_vocal_surgical());
    obj.fx_chain.push(FxSlotConfig::default_vocal_tune());
    project.tracks[0].objects.push(obj);

    let midi_id = project.alloc_id();
    project.tracks[1].synth_preset = SynthPreset::WarmSawPad;
    project.tracks[1]
        .midi_clips
        .push(MidiClip::demo_am7_progression(midi_id, 48_000, 120.0));

    let obj2_id = project.alloc_id();
    let mut obj2 = AudioObject::new(obj2_id, "Bass", clip_path, 8192);
    obj2.fx_chain.push(FxSlotConfig::default_sub_mono_tape());
    obj2.fx_chain.push(FxSlotConfig::default_split_eq());
    project.tracks[1].objects.push(obj2);

    project.tracks[0]
        .inserts
        .push(FxSlotConfig::default_space_verb());
    let aux_id = project.aux_buses[0].id;
    project.tracks[0].sends.push(TrackSend {
        bus_id: aux_id,
        gain_db: -6.0,
        pre_fader: false,
        mute: false,
    });
    project
        .master_inserts
        .push(FxSlotConfig::default_density_clipper());
    project
        .master_inserts
        .push(FxSlotConfig::default_master_limiter());

    let mut engine = ObjectOrientedEngine::compile(&project, &cache);
    let mut out_l = vec![0.0_f32; 512];
    let mut out_r = vec![0.0_f32; 512];

    let before = violation_count();
    assert_no_alloc(|| {
        for _ in 0..16 {
            engine.render_block(&mut out_l, &mut out_r);
        }
    });
    let after = violation_count();
    assert_eq!(
        after - before,
        0,
        "ObjectOrientedEngine::render_block must perform zero heap allocations on the audio thread"
    );
    assert!(out_l.iter().all(|s| s.is_finite()));
    assert!(out_r.iter().all(|s| s.is_finite()));
}
