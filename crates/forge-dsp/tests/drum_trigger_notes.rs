//! Every note the Drum Trigger plugin generates must be released, exactly once.
//!
//! A [`Host`] stands in for the receiving instrument: it fails on a NoteOn for a key that is
//! already sounding, a NoteOff for a key that is not, an out-of-range velocity, or an event
//! timed outside its block or out of order. Each test ends by checking nothing is left sounding.

use std::collections::HashMap;
use std::f64::consts::PI;

use forge_dsp::drum_midi::{CHORD_INTERVALS, DRUM_GATE_MS};
use forge_dsp::{DrumTriggerNotes, NoteMsg, RealtimeDrumTrigger};

const SR: f64 = 48_000.0;
const BLOCK: u32 = 64;

/// Sounding keys and when each started, in absolute samples.
#[derive(Default)]
struct Host {
    sounding: HashMap<(u8, u8), u64>,
    /// Absolute start of the block being delivered.
    block_start: u64,
    last: u64,
    ons: usize,
    /// `(note, length in samples)` of every released note.
    released: Vec<(u8, u64)>,
}

impl Host {
    fn receive(&mut self, msg: NoteMsg) {
        let (timing, key) = match msg {
            NoteMsg::On {
                timing,
                channel,
                note,
                velocity,
            } => {
                assert!((0.0..=1.0).contains(&velocity), "velocity {velocity}");
                (timing, (channel, note))
            }
            NoteMsg::Off {
                timing,
                channel,
                note,
            } => (timing, (channel, note)),
        };
        assert!(timing < BLOCK, "timing {timing} outside the block");
        let at = self.block_start + u64::from(timing);
        assert!(at >= self.last, "event at {at} after one at {}", self.last);
        self.last = at;
        match msg {
            NoteMsg::On { .. } => {
                let previous = self.sounding.insert(key, at);
                assert!(previous.is_none(), "NoteOn {key:?} at {at} while it sounds");
                self.ons += 1;
            }
            NoteMsg::Off { .. } => {
                let start = self.sounding.remove(&key);
                let start = start.unwrap_or_else(|| panic!("NoteOff {key:?} at {at} not on"));
                self.released.push((key.1, at - start));
            }
        }
    }

    fn assert_all_released(&self) {
        assert!(self.sounding.is_empty(), "stuck notes: {:?}", self.sounding);
    }
}

/// Kick, snare and hat hits every 250 ms, cycling through the three.
fn drum_pattern(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let hit = i / 12_000;
            let t = (i % 12_000) as f64 / SR;
            let freq = [55.0, 900.0, 11_000.0][hit % 3];
            (0.8 * (-t / 0.05).exp() * (2.0 * PI * freq * t).sin()) as f32
        })
        .collect()
}

#[test]
fn audio_drum_hits_each_get_a_note_off_one_gate_later() {
    let gate = (SR * DRUM_GATE_MS / 1000.0).round() as u64;
    let mut trigger = RealtimeDrumTrigger::new(SR);
    let mut notes = DrumTriggerNotes::new(SR);
    let mut host = Host::default();
    let audio = drum_pattern(144_000);
    for (block, samples) in audio.chunks(BLOCK as usize).enumerate() {
        host.block_start = block as u64 * u64::from(BLOCK);
        let emit = &mut |msg| host.receive(msg);
        for (timing, &s) in (0_u32..).zip(samples) {
            notes.tick(timing, emit);
            let hits = trigger.process_stereo_sample(s, s, -24.0);
            for (slot, hit) in hits.into_iter().enumerate() {
                if let Some((note, velocity)) = hit {
                    notes.drum_hit(timing, slot, note, velocity, emit);
                }
            }
        }
    }
    assert!(host.ons >= 12, "only {} drum notes", host.ons);
    host.assert_all_released();
    assert!(notes.is_idle());
    for &(note, length) in &host.released {
        assert_eq!(length, gate, "note {note} held {length} samples");
    }
}

#[test]
fn chord_is_released_when_expansion_is_switched_off_mid_note() {
    let mut notes = DrumTriggerNotes::new(SR);
    let mut host = Host::default();
    let emit = &mut |msg| host.receive(msg);
    notes.set_chord_expansion(true);
    notes.root_on(0, 0, 60, 0.9, emit);
    notes.set_chord_expansion(false);
    notes.root_off(10, 0, 60, emit);
    assert_eq!(host.ons, CHORD_INTERVALS.len());
    host.assert_all_released();
}

#[test]
fn no_stray_note_offs_when_expansion_is_switched_on_mid_note() {
    let mut notes = DrumTriggerNotes::new(SR);
    let mut host = Host::default();
    let emit = &mut |msg| host.receive(msg);
    notes.set_chord_expansion(false);
    notes.root_on(0, 0, 60, 0.9, emit);
    notes.set_chord_expansion(true);
    notes.root_off(10, 0, 60, emit);
    assert_eq!(host.ons, 0);
    assert!(host.released.is_empty(), "{:?}", host.released);
}

#[test]
fn roots_sharing_a_chord_tone_never_stack_it() {
    // 60 voices 63, 67, 70, 74 and 63 voices 66, 70, 73, 77: both hold 70.
    let mut notes = DrumTriggerNotes::new(SR);
    let mut host = Host::default();
    let emit = &mut |msg| host.receive(msg);
    notes.root_on(0, 0, 60, 0.9, emit);
    notes.root_on(1, 0, 63, 0.9, emit);
    notes.root_off(2, 0, 60, emit);
    assert!(
        host.sounding.contains_key(&(0, 70)),
        "70 cut while 63 holds it"
    );
    notes.root_off(3, 0, 63, &mut |msg| host.receive(msg));
    host.assert_all_released();
    assert!(notes.is_idle());
}

#[test]
fn repeated_root_note_on_retriggers_its_chord() {
    let mut notes = DrumTriggerNotes::new(SR);
    let mut host = Host::default();
    let emit = &mut |msg| host.receive(msg);
    notes.root_on(0, 0, 60, 0.9, emit);
    notes.root_on(5, 0, 60, 0.9, emit);
    notes.root_off(9, 0, 60, emit);
    assert_eq!(host.ons, 2 * CHORD_INTERVALS.len());
    host.assert_all_released();
}

#[test]
fn release_all_closes_held_chords_and_drum_gates() {
    let mut notes = DrumTriggerNotes::new(SR);
    let mut host = Host::default();
    let emit = &mut |msg| host.receive(msg);
    notes.root_on(0, 3, 48, 0.7, emit);
    notes.drum_hit(1, 0, 36, 0.8, emit);
    notes.drum_hit(1, 2, 42, 0.8, emit);
    notes.release_all(2, emit);
    host.assert_all_released();
    // The root's own NoteOff and later gate ticks must not release anything twice.
    let emit = &mut |msg| host.receive(msg);
    notes.root_off(3, 3, 48, emit);
    for t in 4..BLOCK {
        notes.tick(t, emit);
    }
    assert_eq!(host.released.len(), host.ons);
    assert!(notes.is_idle());
}

#[test]
fn out_of_range_velocities_are_clamped() {
    let mut notes = DrumTriggerNotes::new(SR);
    let mut host = Host::default();
    let emit = &mut |msg| host.receive(msg);
    notes.root_on(0, 0, 60, 7.5, emit);
    notes.root_on(0, 1, 60, f32::NAN, emit);
    notes.drum_hit(0, 1, 38, -3.0, emit);
    notes.drum_hit(0, 2, 42, f32::INFINITY, emit);
    notes.release_all(1, emit);
    host.assert_all_released();
}
