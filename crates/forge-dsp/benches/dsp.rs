//! CodSpeed / divan benchmarks for `forge-dsp` offline processing and real-time streaming processors.

use std::f64::consts::PI;

use divan::{Bencher, black_box};
use forge_core::AudioBuffer;
use forge_dsp::{
    AnalogHardwareModel, DeFizzConfig, RealtimeDensityAndClipper, RealtimeFdnReverb,
    RealtimeMasterLimiter, RealtimeMasteringEq18, RealtimeNeuralPreamp, RealtimeOpenSoothe,
    RealtimeOpenSpacer, RealtimeSubAndTapeBus, RealtimeTransientTonalSplit, RealtimeVocalSurgical,
    RestorationConfig, UnmaskConfig, align_phase_gcc_phat, align_vocal_double_dtw,
    apply_18_band_eq, apply_spectral_defizz, apply_spectral_unmask, audit_audio,
    clean_vocal_surgical, compute_true_peak, normalize_and_limit, run_full_restoration_chain,
};

const SR: u32 = 48_000;
/// Block size used for the real-time streaming processor benchmarks.
const BLOCK: usize = 4_096;

fn main() {
    divan::main();
}

/// Deterministic test signal: warm fundamental + harsh high partial + light pseudo-noise.
fn test_signal(frames: usize, seed: u32) -> Vec<f64> {
    let mut state = seed.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    (0..frames)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let noise = (f64::from(state) / f64::from(u32::MAX)) * 2.0 - 1.0;
            let t = (i as f64) / f64::from(SR);
            0.3 * (2.0 * PI * 220.0 * t).sin()
                + 0.25 * (2.0 * PI * 4_200.0 * t).sin()
                + 0.05 * noise
        })
        .collect()
}

fn stereo_buffer(frames: usize) -> AudioBuffer {
    AudioBuffer::from_channels(vec![test_signal(frames, 1), test_signal(frames, 2)], SR)
        .expect("valid buffer")
}

fn mono_buffer(frames: usize, seed: u32) -> AudioBuffer {
    AudioBuffer::from_channels(vec![test_signal(frames, seed)], SR).expect("valid buffer")
}

fn block_f32(frames: usize) -> Vec<f32> {
    test_signal(frames, 7)
        .into_iter()
        .map(|s| s as f32)
        .collect()
}

// ---------------------------------------------------------------------------
// Offline analysis & mastering
// ---------------------------------------------------------------------------

/// Durations in frames: 0.25 s and 1 s of 48 kHz stereo audio.
const OFFLINE_FRAMES: &[usize] = &[12_000, 48_000];

#[divan::bench(args = OFFLINE_FRAMES)]
fn audit_audio_bs1770(bencher: Bencher, frames: usize) {
    let buf = stereo_buffer(frames);
    bencher.bench(|| audit_audio(black_box(&buf)));
}

#[divan::bench(args = OFFLINE_FRAMES)]
fn true_peak_4x(bencher: Bencher, frames: usize) {
    let buf = stereo_buffer(frames);
    bencher.bench(|| compute_true_peak(black_box(&buf)));
}

#[divan::bench(args = OFFLINE_FRAMES)]
fn normalize_and_limit_master(bencher: Bencher, frames: usize) {
    let buf = stereo_buffer(frames);
    bencher.bench(|| normalize_and_limit(black_box(&buf), -14.0, -1.0).expect("mastered"));
}

#[divan::bench(args = OFFLINE_FRAMES)]
fn spectral_defizz(bencher: Bencher, frames: usize) {
    let buf = stereo_buffer(frames);
    let config = DeFizzConfig::default();
    // `apply_spectral_defizz` fans channels out with rayon. Pin it to a single-thread pool that
    // reuses the benchmark thread so the measurement covers the DSP work rather than thread
    // scheduling/system calls, keeping results stable under the CPU simulation instrument.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .use_current_thread()
        .build()
        .expect("rayon pool");
    bencher.bench_local(|| {
        pool.install(|| apply_spectral_defizz(black_box(&buf), &config).expect("defizzed"))
    });
}

#[divan::bench(args = OFFLINE_FRAMES)]
fn spectral_unmask_vocal_over_music(bencher: Bencher, frames: usize) {
    let target = stereo_buffer(frames);
    let sidechain =
        AudioBuffer::from_channels(vec![test_signal(frames, 11), test_signal(frames, 12)], SR)
            .expect("valid buffer");
    let config = UnmaskConfig::vocal_over_instruments();
    bencher.bench(|| {
        apply_spectral_unmask(black_box(&target), black_box(&sidechain), &config).expect("unmasked")
    });
}

// ---------------------------------------------------------------------------
// Restoration
// ---------------------------------------------------------------------------

#[divan::bench(args = OFFLINE_FRAMES)]
fn full_restoration_chain(bencher: Bencher, frames: usize) {
    let buf = stereo_buffer(frames);
    let config = RestorationConfig::default();
    bencher.bench(|| run_full_restoration_chain(black_box(&buf), &config).expect("restored"));
}

#[divan::bench(args = OFFLINE_FRAMES)]
fn vocal_surgical_cleanup(bencher: Bencher, frames: usize) {
    let buf = stereo_buffer(frames);
    bencher.bench(|| clean_vocal_surgical(black_box(&buf), true, true, -12.0).expect("cleaned"));
}

#[divan::bench(args = OFFLINE_FRAMES)]
fn mastering_eq_18_band(bencher: Bencher, frames: usize) {
    let buf = stereo_buffer(frames);
    let gains = [
        1.0, 0.5, -0.5, 0.0, -1.5, -2.0, 0.0, 0.5, 1.0, 0.0, -0.5, 0.0, 1.5, 2.0, 1.0, 0.0, -1.0,
        0.5,
    ];
    bencher
        .with_inputs(|| buf.clone())
        .bench_local_refs(|b| apply_18_band_eq(b, black_box(&gains)));
}

// ---------------------------------------------------------------------------
// Alignment
// ---------------------------------------------------------------------------

#[divan::bench(args = [16_384, 65_536])]
fn gcc_phat_phase_align(bencher: Bencher, frames: usize) {
    let reference = mono_buffer(frames, 3);
    // Target = reference delayed by 23 samples with inverted polarity.
    let mut tgt = vec![0.0; frames];
    for (t, r) in tgt[23..].iter_mut().zip(&reference.channels[0]) {
        *t = -r;
    }
    let target = AudioBuffer::from_channels(vec![tgt], SR).expect("valid buffer");
    bencher.bench(|| {
        align_phase_gcc_phat(black_box(&reference), black_box(&target), 20.0).expect("aligned")
    });
}

#[divan::bench(args = [24_000, 48_000])]
fn dtw_vocal_double_align(bencher: Bencher, frames: usize) {
    let guide = mono_buffer(frames, 4);
    // Dub = guide delayed by 5 ms and slightly attenuated.
    let mut dub_ch = vec![0.0; frames];
    for (d, g) in dub_ch[240..].iter_mut().zip(&guide.channels[0]) {
        *d = 0.9 * g;
    }
    let dub = AudioBuffer::from_channels(vec![dub_ch], SR).expect("valid buffer");
    bencher.bench(|| {
        align_vocal_double_dtw(black_box(&guide), black_box(&dub), 60.0).expect("aligned")
    });
}

// ---------------------------------------------------------------------------
// Real-time streaming processors (one block of BLOCK stereo samples)
// ---------------------------------------------------------------------------

mod realtime {
    use super::*;

    #[divan::bench]
    fn open_soothe(bencher: Bencher) {
        let input = block_f32(BLOCK);
        let mut proc = RealtimeOpenSoothe::new(f64::from(SR));
        bencher.bench_local(|| {
            for &s in &input {
                black_box(proc.process_stereo_sample(s, s, 3.5, 9.0, 8, false));
            }
        });
    }

    #[divan::bench]
    fn open_spacer(bencher: Bencher) {
        let input = block_f32(BLOCK);
        let side = test_signal(BLOCK, 9)
            .into_iter()
            .map(|s| s as f32)
            .collect::<Vec<_>>();
        let mut proc = RealtimeOpenSpacer::new(f64::from(SR));
        bencher.bench_local(|| {
            for (&s, &sc) in input.iter().zip(&side) {
                black_box(proc.process_sample(s, s, sc, sc, 200.0, 7_500.0, 1.0, 4.5));
            }
        });
    }

    #[divan::bench]
    fn vocal_surgical(bencher: Bencher) {
        let input = block_f32(BLOCK);
        let mut proc = RealtimeVocalSurgical::new(f64::from(SR));
        bencher.bench_local(|| {
            for &s in &input {
                black_box(proc.process_stereo_sample(s, s, 0.7, 0.7, -20.0, -50.0));
            }
        });
    }

    #[divan::bench]
    fn sub_and_tape_bus(bencher: Bencher) {
        let input = block_f32(BLOCK);
        let mut proc = RealtimeSubAndTapeBus::new(f64::from(SR), 160.0);
        bencher.bench_local(|| {
            for &s in &input {
                black_box(proc.process_stereo_sample(s, s, 0.3, 1.05, 0.25));
            }
        });
    }

    #[divan::bench]
    fn master_limiter(bencher: Bencher) {
        let input = block_f32(BLOCK);
        let mut proc = RealtimeMasterLimiter::new(f64::from(SR));
        bencher.bench_local(|| {
            for &s in &input {
                black_box(proc.process_stereo_sample(s, s, 3.0, -1.0, 85.0));
            }
        });
    }

    #[divan::bench]
    fn mastering_eq_18(bencher: Bencher) {
        let input = block_f32(BLOCK);
        let mut proc = RealtimeMasteringEq18::new(f64::from(SR));
        let gains = [
            1.0, 0.5, -0.5, 0.0, -1.5, -2.0, 0.0, 0.5, 1.0, 0.0, -0.5, 0.0, 1.5, 2.0, 1.0, 0.0,
            -1.0, 0.5,
        ];
        proc.set_gains(&gains, 1.0);
        bencher.bench_local(|| {
            for &s in &input {
                black_box(proc.process_stereo_sample(s, s));
            }
        });
    }

    #[divan::bench]
    fn transient_tonal_split(bencher: Bencher) {
        let input = block_f32(BLOCK);
        let mut proc = RealtimeTransientTonalSplit::new(f64::from(SR));
        bencher.bench_local(|| {
            for &s in &input {
                black_box(proc.process_stereo_sample(
                    s,
                    s,
                    [2.0, 0.0, -2.0],
                    [0.5, 0.0, 1.5],
                    1.5,
                    0,
                ));
            }
        });
    }

    #[divan::bench]
    fn density_and_clipper(bencher: Bencher) {
        let input = block_f32(BLOCK);
        let mut proc = RealtimeDensityAndClipper::new(f64::from(SR));
        bencher.bench_local(|| {
            for &s in &input {
                black_box(proc.process_stereo_sample(s, s, 0.4, 3.0, -0.8));
            }
        });
    }

    #[divan::bench]
    fn fdn_reverb(bencher: Bencher) {
        let input = block_f32(BLOCK);
        let mut proc = RealtimeFdnReverb::new(f64::from(SR));
        bencher.bench_local(|| {
            for &s in &input {
                black_box(proc.process_stereo_sample(s, s, 0.65, 0.4, 0.15, 0.5, 0.25));
            }
        });
    }

    #[divan::bench]
    fn neural_preamp(bencher: Bencher) {
        let input = block_f32(BLOCK);
        let mut proc = RealtimeNeuralPreamp::new(AnalogHardwareModel::Neve1073ClassA);
        bencher.bench_local(|| {
            for &s in &input {
                black_box(proc.process_stereo_sample(s, s, 4.0, 0.0));
            }
        });
    }
}
