//! CodSpeed / divan benchmarks for `forge-engine` multitrack rendering and project deserialization.

use std::f64::consts::PI;
use std::path::PathBuf;

use divan::{Bencher, black_box};
use forge_core::AudioBuffer;
use forge_engine::{AudioObject, ClipCache, DawProject, FxSlotConfig, ObjectOrientedEngine};

fn main() {
    divan::main();
}

fn bench_project_and_cache() -> (DawProject, ClipCache) {
    let sr = 48_000_u32;
    let frames = sr as usize;
    let clip_path = PathBuf::from("bench_lead.wav");
    let ch: Vec<f64> = (0..frames)
        .map(|i| {
            let t = (i as f64) / f64::from(sr);
            0.35 * (2.0 * PI * 220.0 * t).sin() + 0.15 * (2.0 * PI * 3_500.0 * t).sin()
        })
        .collect();
    let buf = AudioBuffer::from_channels(vec![ch.clone(), ch], sr).expect("valid clip");

    let mut cache = ClipCache::default();
    cache.insert_memory_clip(clip_path.clone(), buf);

    let mut proj = DawProject::new("EngineBench", sr, 120.0);
    let obj_id = proj.alloc_id();
    let mut obj = AudioObject::new(obj_id, "Lead", clip_path, frames as u64);
    obj.fx_chain.push(FxSlotConfig::default_split_eq());
    proj.tracks[0].objects.push(obj);
    proj.tracks[0]
        .inserts
        .push(FxSlotConfig::default_density_clipper());
    (proj, cache)
}

#[divan::bench(args = [256, 1024])]
fn engine_render_block(bencher: Bencher, block_size: usize) {
    let (proj, cache) = bench_project_and_cache();
    let mut engine = ObjectOrientedEngine::compile(&proj, &cache);
    let mut out_l = vec![0.0_f32; block_size];
    let mut out_r = vec![0.0_f32; block_size];
    bencher.bench_local(|| {
        if engine.playhead_samples() >= 48_000 {
            engine.set_playhead_samples(0);
        }
        engine.render_block(black_box(&mut out_l), black_box(&mut out_r));
    });
}

#[divan::bench]
fn project_json_roundtrip_parse(bencher: Bencher) {
    let (proj, _) = bench_project_and_cache();
    let json = proj.to_json().expect("serialize");
    bencher.bench_local(|| {
        let parsed = DawProject::from_json(black_box(&json)).expect("parse");
        black_box(parsed);
    });
}
