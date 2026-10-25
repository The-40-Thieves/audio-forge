//! Cross-instance shared spectrum bus (THE-1596): slot mapping, silent defaults, clearing, and
//! frame consistency under concurrent publish and read.
//!
//! Each test uses its own slot: tests in this binary run in parallel and share the statics.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use forge_dsp::{
    SPECTRUM_FLOOR_DB, clear_shared_spectrum, publish_shared_spectrum, read_shared_spectrum,
    shared_spectrum_slot_index,
};

#[test]
fn bus_param_zero_is_off_and_one_to_eight_are_slots_a_to_h() {
    assert_eq!(shared_spectrum_slot_index(0), None);
    for p in 1..=8 {
        assert_eq!(shared_spectrum_slot_index(p), Some(p - 1));
    }
    assert_eq!(shared_spectrum_slot_index(9), None);
}

#[test]
fn unpublished_slots_read_as_silence() {
    // Slot index 7 is never published to in this binary.
    assert_eq!(read_shared_spectrum(7), [SPECTRUM_FLOOR_DB; 32]);
}

#[test]
fn clearing_a_slot_returns_it_to_silence() {
    let mut bands = [-12.0_f32; 32];
    bands[3] = f32::NAN;
    assert!(publish_shared_spectrum(1, &bands));
    let read = read_shared_spectrum(1);
    assert_eq!(read[0], -12.0);
    assert_eq!(
        read[3], SPECTRUM_FLOOR_DB,
        "NaN must read as silence, not 0 dB"
    );
    assert!(clear_shared_spectrum(1));
    assert_eq!(read_shared_spectrum(1), [SPECTRUM_FLOOR_DB; 32]);
}

/// A writer publishes frames whose 32 bands all hold the same value while a reader checks
/// that no frame it gets mixes bands from two publishes.
#[test]
fn concurrent_reads_never_see_a_torn_frame() {
    const SLOT: usize = 4;
    let stop = Arc::new(AtomicBool::new(false));
    let writer = {
        let stop = Arc::clone(&stop);
        thread::spawn(move || {
            let mut k = 0_u32;
            while !stop.load(Ordering::Relaxed) {
                k = (k + 1) % 4096;
                publish_shared_spectrum(SLOT, &[-(k as f32) * 0.01; 32]);
            }
        })
    };
    let mut torn = 0_usize;
    let mut distinct = std::collections::HashSet::new();
    for _ in 0..200_000 {
        let frame = read_shared_spectrum(SLOT);
        if frame.iter().any(|&v| v != frame[0]) {
            torn += 1;
        }
        distinct.insert(frame[0].to_bits());
    }
    stop.store(true, Ordering::Relaxed);
    writer.join().expect("writer thread");
    assert_eq!(torn, 0, "{torn} of 200000 reads mixed two published frames");
    assert!(
        distinct.len() > 10,
        "the reader should see the writer's frames change"
    );
}
