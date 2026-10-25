//! Cross-instance shared spectrum bus (THE-1596): slot mapping, silent defaults, clearing, and
//! frame consistency under concurrent publish and read.
//!
//! Each test uses its own slot: tests in this binary run in parallel and share the statics.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use forge_dsp::{
    SPECTRUM_FLOOR_DB, SpectrumBusOwner, clear_shared_spectrum, publish_shared_spectrum,
    read_shared_spectrum, shared_spectrum_owner, shared_spectrum_slot_index,
};

const SILENT: [f32; 32] = [SPECTRUM_FLOOR_DB; 32];

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
    assert_eq!(read_shared_spectrum(7), SILENT);
    assert_eq!(shared_spectrum_owner(7), None);
}

#[test]
fn clearing_a_slot_returns_it_to_silence() {
    let owner = SpectrumBusOwner::unique();
    let mut bands = [-12.0_f32; 32];
    bands[3] = f32::NAN;
    assert!(publish_shared_spectrum(owner, 1, &bands));
    assert_eq!(shared_spectrum_owner(1), Some(owner));
    let read = read_shared_spectrum(1);
    assert_eq!(read[0], -12.0);
    assert_eq!(
        read[3], SPECTRUM_FLOOR_DB,
        "NaN must read as silence, not 0 dB"
    );
    assert!(clear_shared_spectrum(owner, 1));
    assert_eq!(read_shared_spectrum(1), SILENT);
    assert_eq!(shared_spectrum_owner(1), None);
}

#[test]
fn owner_ids_are_unique() {
    let ids: std::collections::HashSet<_> = (0..1000).map(|_| SpectrumBusOwner::unique()).collect();
    assert_eq!(ids.len(), 1000);
}

/// Instances A and B share slot index 2 and B published last, so B owns it: A moving away or
/// being removed must not silence B, and only B's own clear empties the slot.
#[test]
fn only_the_last_publisher_can_clear_a_shared_slot() {
    const SLOT: usize = 2;
    let (a, b) = (SpectrumBusOwner::unique(), SpectrumBusOwner::unique());
    let (a_bands, b_bands) = ([-30.0_f32; 32], [-10.0_f32; 32]);
    assert!(publish_shared_spectrum(a, SLOT, &a_bands));
    assert!(publish_shared_spectrum(b, SLOT, &b_bands));
    assert_eq!(shared_spectrum_owner(SLOT), Some(b));

    assert!(!clear_shared_spectrum(a, SLOT), "A no longer owns the slot");
    assert_eq!(read_shared_spectrum(SLOT), b_bands, "A's clear silenced B");
    assert_eq!(shared_spectrum_owner(SLOT), Some(b));

    assert!(clear_shared_spectrum(b, SLOT));
    assert_eq!(read_shared_spectrum(SLOT), SILENT);
    // Once cleared, nobody owns it, so a second clear does nothing.
    assert!(!clear_shared_spectrum(b, SLOT));

    // Ownership follows the last write: A publishing again takes it back.
    assert!(publish_shared_spectrum(b, SLOT, &b_bands));
    assert!(publish_shared_spectrum(a, SLOT, &a_bands));
    assert!(!clear_shared_spectrum(b, SLOT));
    assert_eq!(read_shared_spectrum(SLOT), a_bands);
    assert!(clear_shared_spectrum(a, SLOT));
    assert_eq!(read_shared_spectrum(SLOT), SILENT);
}

#[test]
fn out_of_range_slots_are_ignored() {
    let owner = SpectrumBusOwner::unique();
    assert!(!publish_shared_spectrum(owner, 8, &[-1.0; 32]));
    assert!(!clear_shared_spectrum(owner, 8));
    assert_eq!(shared_spectrum_owner(8), None);
    assert_eq!(read_shared_spectrum(8), SILENT);
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
            let owner = SpectrumBusOwner::unique();
            let mut k = 0_u32;
            while !stop.load(Ordering::Relaxed) {
                k = (k + 1) % 4096;
                publish_shared_spectrum(owner, SLOT, &[-(k as f32) * 0.01; 32]);
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
