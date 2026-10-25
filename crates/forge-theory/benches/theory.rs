//! CodSpeed / divan benchmarks for `forge-theory` chord parsing, voice leading, ABC and drum triggers.

use std::f64::consts::PI;

use divan::{black_box, Bencher};
use forge_core::AudioBuffer;
use forge_theory::{
    export_abc_score, extract_drum_midi_triggers, parse_abc_score, parse_chord_symbol,
    reharmonize_progression, solve_voice_leading, GrooveStyle, VoicingStyle,
};

fn main() {
    divan::main();
}

fn progression(len: usize) -> Vec<String> {
    const BASE: [&str; 8] = ["Dm7", "G7", "Cmaj7", "A7", "Fmaj7", "Bb13", "Em7", "Ebdim"];
    BASE.iter()
        .cycle()
        .take(len)
        .map(|s| (*s).to_string())
        .collect()
}

#[divan::bench]
fn parse_chord_symbols(bencher: Bencher) {
    let chords = progression(64);
    bencher.bench(|| {
        for c in &chords {
            let _ = black_box(parse_chord_symbol(black_box(c)));
        }
    });
}

#[divan::bench]
fn reharmonize_neosoul(bencher: Bencher) {
    let chords = progression(64);
    bencher.bench(|| reharmonize_progression(black_box(&chords), VoicingStyle::NeoSoulRootless));
}

#[divan::bench(
    consts = [8, 64],
    args = [
        (VoicingStyle::Close, GrooveStyle::Straight),
        (VoicingStyle::Drop2, GrooveStyle::MpcSwing58),
        (VoicingStyle::NeoSoulRootless, GrooveStyle::NeoSoulDilla),
        (VoicingStyle::CinematicSpread, GrooveStyle::AcousticStrum),
    ]
)]
fn voice_leading<const N: usize>(bencher: Bencher, (style, groove): (VoicingStyle, GrooveStyle)) {
    let chords = reharmonize_progression(&progression(N), style);
    bencher.bench(|| {
        solve_voice_leading(black_box(&chords), style, groove, 480, 4).expect("voice leading")
    });
}

#[divan::bench]
fn abc_export_and_parse(bencher: Bencher) {
    let chords = progression(32);
    bencher.bench(|| {
        let abc = export_abc_score("Bench", "C", 92, black_box(&chords));
        parse_abc_score(&abc, 480)
    });
}

/// Synthetic drum loop: kicks on beats, snares on backbeats, hats on 8ths (120 BPM).
fn drum_loop(seconds: f64) -> AudioBuffer {
    let sr = 48_000_u32;
    let frames = (f64::from(sr) * seconds) as usize;
    let beat = f64::from(sr) * 0.5;
    let mut state = 0x1234_5678_u32;
    let ch: Vec<f64> = (0..frames)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let noise = (f64::from(state) / f64::from(u32::MAX)) * 2.0 - 1.0;
            let pos_in_beat = (i as f64) % beat;
            let beat_idx = ((i as f64) / beat) as usize;
            let t_beat = pos_in_beat / f64::from(sr);
            let kick = (-t_beat * 30.0).exp() * (2.0 * PI * 60.0 * t_beat).sin();
            let snare = if beat_idx % 2 == 1 {
                0.6 * (-t_beat * 25.0).exp() * noise
            } else {
                0.0
            };
            let pos_in_eighth = (i as f64) % (beat / 2.0) / f64::from(sr);
            let hat = 0.2 * (-pos_in_eighth * 120.0).exp() * noise;
            0.8 * kick + snare + hat
        })
        .collect();
    AudioBuffer::from_channels(vec![ch], sr).expect("valid buffer")
}

#[divan::bench]
fn drum_midi_triggers(bencher: Bencher) {
    let buf = drum_loop(2.0);
    bencher.bench(|| extract_drum_midi_triggers(black_box(&buf), 120.0, 480));
}
