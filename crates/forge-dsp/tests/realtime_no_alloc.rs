//! Allocation guard for the real-time processors the plugins run on the audio thread.
//!
//! A counting global allocator records heap allocations made by the current thread while a
//! guard is armed. Each processor is built first (construction may allocate), then its
//! per-sample `process` path and its `reset` are run under the guard, which must see zero
//! allocations.

use forge_dsp::{
    RealtimeDrumTrigger, RealtimeMasterLimiter, RealtimeMasteringEq18, RealtimeOpenSoothe,
    RealtimeOpenSpacer, RealtimePhaseAligner, RealtimeSubAndTapeBus, RealtimeVocalSurgical,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct CountingAlloc;

thread_local! {
    static ARMED: Cell<bool> = const { Cell::new(false) };
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

fn record() {
    // `try_with` so allocations during thread-local teardown are ignored instead of panicking.
    let _ = ARMED.try_with(|armed| {
        if armed.get() {
            let _ = ALLOCS.try_with(|n| n.set(n.get() + 1));
        }
    });
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record();
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

/// Runs `f` with the guard armed and returns how many allocations it made on this thread.
fn allocations_during(f: impl FnOnce()) -> usize {
    ALLOCS.with(|n| n.set(0));
    ARMED.with(|a| a.set(true));
    f();
    ARMED.with(|a| a.set(false));
    ALLOCS.with(Cell::get)
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
    assert_eq!(n, 0, "{name} allocated {n} time(s) on the audio path");
}

#[test]
fn guard_detects_allocations() {
    let n = allocations_during(|| {
        std::hint::black_box(vec![0_u8; 64]);
    });
    assert!(n > 0, "the counting allocator should see a Vec allocation");
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
