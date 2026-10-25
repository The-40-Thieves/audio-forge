//! Allocation guard for the real-time processors the plugins run on the audio thread.
//!
//! `assert_no_alloc`'s allocator (the one nih-plug uses for `assert_process_allocs`) counts
//! every allocation and deallocation the current thread makes inside `assert_no_alloc`. Each
//! processor is built first (construction may allocate), then its per-sample `process` path
//! and its `reset` are run under the guard, which must see zero violations.
//!
//! Debug builds only: in release, `assert_no_alloc`'s default `disable_release` feature (also
//! used by nih-plug) compiles the allocator out. CI runs `cargo test` in debug.
#![cfg(debug_assertions)]

use assert_no_alloc::{AllocDisabler, assert_no_alloc, reset_violation_count, violation_count};
use forge_dsp::{
    RealtimeDrumTrigger, RealtimeMasterLimiter, RealtimeMasteringEq18, RealtimeOpenSoothe,
    RealtimeOpenSpacer, RealtimePhaseAligner, RealtimeSubAndTapeBus, RealtimeVocalSurgical,
};

#[global_allocator]
static GLOBAL: AllocDisabler = AllocDisabler;

/// Runs `f` under the guard and returns how many allocator calls it made on this thread.
fn allocations_during(f: impl FnOnce()) -> u32 {
    reset_violation_count();
    assert_no_alloc(f);
    violation_count()
}

/// Enough frames to run several 512-sample STFT hops.
const FRAMES: usize = 4096;

fn signal(i: usize) -> (f32, f32) {
    let t = i as f32 / 48_000.0;
    let l = 0.6 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
    let r = 0.6 * (2.0 * std::f32::consts::PI * 5_300.0 * t).sin();
    (l, r)
}

fn assert_no_allocs(name: &str, f: impl FnOnce()) {
    let n = allocations_during(f);
    assert_eq!(n, 0, "{name} made {n} allocator call(s) on the audio path");
}

#[test]
fn guard_detects_allocations() {
    let n = allocations_during(|| {
        std::hint::black_box(vec![0_u8; 64]);
    });
    assert!(n > 0, "the guard should see a Vec allocation");
}

#[test]
fn open_soothe_process_and_reset_do_not_allocate() {
    let mut dsp = RealtimeOpenSoothe::new(48_000.0);
    assert_no_allocs("RealtimeOpenSoothe", || {
        for i in 0..FRAMES {
            let (l, r) = signal(i);
            std::hint::black_box(dsp.process_stereo_sample(l, r, 3.5, 9.0, 8, i % 2 == 0));
        }
        dsp.reset(44_100.0);
        dsp.reset(96_000.0);
    });
}

#[test]
fn open_spacer_process_and_reset_do_not_allocate() {
    let mut dsp = RealtimeOpenSpacer::new(48_000.0);
    assert_no_allocs("RealtimeOpenSpacer", || {
        for i in 0..FRAMES {
            let (l, r) = signal(i);
            std::hint::black_box(dsp.process_sample(l, r, r, l, 180.0, 8_000.0, 0.7, 6.0));
        }
        dsp.reset(96_000.0);
    });
}

#[test]
fn master_limiter_process_and_reset_do_not_allocate() {
    let mut dsp = RealtimeMasterLimiter::new(48_000.0);
    assert_no_allocs("RealtimeMasterLimiter", || {
        for i in 0..FRAMES {
            let (l, r) = signal(i);
            std::hint::black_box(dsp.process_stereo_sample(2.0 * l, 2.0 * r, 6.0, -1.0, 85.0));
            std::hint::black_box(dsp.process_delay_only(l, r));
        }
        // Changing sample rate in either direction must reuse the existing delay lines.
        dsp.reset(44_100.0);
        dsp.reset(192_000.0);
        dsp.reset(8_000.0);
    });
}

#[test]
fn other_realtime_processors_do_not_allocate() {
    let mut vocal = RealtimeVocalSurgical::new(48_000.0);
    let mut tape = RealtimeSubAndTapeBus::new(48_000.0, 160.0);
    let mut eq = RealtimeMasteringEq18::new(48_000.0);
    let mut phase = RealtimePhaseAligner::new(48_000.0);
    let mut drums = RealtimeDrumTrigger::new(48_000.0);
    assert_no_allocs("vocal/tape/eq/phase/drum processors", || {
        for i in 0..FRAMES {
            let (l, r) = signal(i);
            std::hint::black_box(vocal.process_stereo_sample(l, r, 0.5, 0.5, -24.0, -60.0));
            std::hint::black_box(tape.process_stereo_sample(l, r, 0.2, 1.06, 0.5));
            std::hint::black_box(eq.process_stereo_sample(l, r));
            std::hint::black_box(phase.process_stereo_sample(l, r));
            std::hint::black_box(drums.process_stereo_sample(l, r, -30.0));
        }
        vocal.reset(96_000.0);
        tape.reset(96_000.0);
        drums.reset(96_000.0);
    });
}

#[test]
fn retuning_and_bypass_paths_do_not_allocate() {
    let mut tape = RealtimeSubAndTapeBus::new(48_000.0, 160.0);
    let mut eq = RealtimeMasteringEq18::new(48_000.0);
    let gains = [3.0; 18];
    assert_no_allocs("tape/eq retuning and bypass", || {
        for i in 0..FRAMES {
            if i % 64 == 0 {
                // Cutoff glide, EQ gain glide, and tape and disperser bypass then re-engage.
                let block = i / 64;
                tape.set_mono_cutoff(60.0 + 4.0 * block as f64);
                eq.set_gains(&gains, if block % 2 == 0 { 1.5 } else { 0.0 });
            }
            let on = (i / 1024) % 2 == 0;
            let (l, r) = signal(i);
            let (sub, drive) = if on { (0.4, 0.5) } else { (0.0, 0.0) };
            std::hint::black_box(tape.process_stereo_sample(l, r, sub, 1.06, drive));
            std::hint::black_box(eq.process_stereo_sample(l, r));
        }
        tape.reset(96_000.0);
    });
}
