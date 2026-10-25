//! CodSpeed / divan benchmarks for `forge-vocal` pitch tracking, pitch shifting and harmony generation.

use std::f64::consts::PI;

use divan::{black_box, Bencher};
use forge_core::AudioBuffer;
use forge_vocal::{
    compile_reaper_take_pitch_envelope, estimate_lpc_formant_tilt_db, generate_vocal_harmony_stack,
    shift_vocal_pitch_formant_preserved, track_vocal_pitch, VocalScale,
};

const SR: u32 = 48_000;

fn main() {
    divan::main();
}

/// Synthetic "sung" vocal: slightly vibrato'd fundamental with a few harmonics and a sibilant burst.
fn vocal_buffer(frames: usize) -> AudioBuffer {
    let mut state = 0x9E37_79B9_u32;
    let ch: Vec<f64> = (0..frames)
        .map(|i| {
            let t = (i as f64) / f64::from(SR);
            let f0 = 218.0 * (1.0 + 0.01 * (2.0 * PI * 5.5 * t).sin());
            let phase = 2.0 * PI * f0 * t;
            let voiced = 0.4 * phase.sin() + 0.2 * (2.0 * phase).sin() + 0.1 * (3.0 * phase).sin();
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let noise = (f64::from(state) / f64::from(u32::MAX)) * 2.0 - 1.0;
            // Short sibilant section every 0.5 s
            if (t % 0.5) > 0.42 {
                0.15 * noise
            } else {
                voiced + 0.01 * noise
            }
        })
        .collect();
    AudioBuffer::from_channels(vec![ch], SR).expect("valid buffer")
}

const FRAMES: &[usize] = &[24_000, 48_000];

#[divan::bench]
fn parse_scale() -> VocalScale {
    VocalScale::parse(black_box("F# dorian")).expect("scale")
}

#[divan::bench(args = FRAMES)]
fn track_pitch_yin(bencher: Bencher, frames: usize) {
    let buf = vocal_buffer(frames);
    let scale = VocalScale::parse("A minor").expect("scale");
    bencher.bench(|| track_vocal_pitch(black_box(&buf), &scale, 0.8));
}

#[divan::bench]
fn lpc_formant_tilt(bencher: Bencher) {
    let buf = vocal_buffer(2_048);
    let frame = buf.channels[0].clone();
    bencher.bench(|| estimate_lpc_formant_tilt_db(black_box(&frame)));
}

#[divan::bench(args = FRAMES)]
fn pitch_shift_formant_preserved(bencher: Bencher, frames: usize) {
    let buf = vocal_buffer(frames);
    let curve = [(0.0, 0.0), (0.25, 2.0), (0.5, -3.0), (1.0, 4.0)];
    bencher.bench(|| {
        shift_vocal_pitch_formant_preserved(black_box(&buf), &curve, -1.0).expect("shifted")
    });
}

#[divan::bench(args = [24_000])]
fn harmony_stack(bencher: Bencher, frames: usize) {
    let buf = vocal_buffer(frames);
    let scale = VocalScale::parse("A minor").expect("scale");
    bencher.bench(|| generate_vocal_harmony_stack(black_box(&buf), &scale).expect("harmonies"));
}

#[divan::bench]
fn reaper_pitch_envelope(bencher: Bencher) {
    let buf = vocal_buffer(48_000);
    let scale = VocalScale::parse("A minor").expect("scale");
    let frames = track_vocal_pitch(&buf, &scale, 0.8);
    bencher.bench(|| compile_reaper_take_pitch_envelope(black_box(&frames)));
}
