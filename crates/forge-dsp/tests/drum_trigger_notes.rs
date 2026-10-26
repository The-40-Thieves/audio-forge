//! Every note the Drum Trigger plugin forwards or generates must be released, exactly once.
//!
//! A [`Host`] stands in for the receiving instrument: it fails on a NoteOn for a key that is
//! already sounding, a NoteOff for a key that is not or with another voice ID, an out-of-range
//! velocity, or an event timed outside its block or out of order. Each test ends by checking
//! nothing is left sounding.

use std::collections::HashMap;
use std::f64::consts::PI;

use forge_dsp::drum_midi::{CHORD_INTERVALS, DRUM_GATE_MS, MAX_ROOT_INSTANCES};
use forge_dsp::{DrumTriggerNotes, NoteMsg, RealtimeDrumTrigger};

const SR: f64 = 48_000.0;
const BLOCK: u32 = 64;
/// Chord tones of root 60, all below the chord ceiling.
const CHORD_60: [u8; 4] = [63, 67, 70, 74];

/// Sounding keys, when each started in absolute samples, and its voice ID.
#[derive(Default)]
struct Host {
    sounding: HashMap<(u8, u8), (u64, Option<i32>)>,
    /// Absolute start of the block being delivered.
    block_start: u64,
    last: u64,
    ons: usize,
    /// `(note, length in samples)` of every released note.
    released: Vec<(u8, u64)>,
}

impl Host {
    fn receive(&mut self, msg: NoteMsg) {
        let (timing, key, voice) = match msg {
            NoteMsg::On {
                timing,
                channel,
                note,
                velocity,
                voice_id,
            } => {
                assert!((0.0..=1.0).contains(&velocity), "velocity {velocity}");
                (timing, (channel, note), voice_id)
            }
            NoteMsg::Off {
                timing,
                channel,
                note,
                voice_id,
            } => (timing, (channel, note), voice_id),
        };
        assert!(timing < BLOCK, "timing {timing} outside the block");
        let at = self.block_start + u64::from(timing);
        assert!(at >= self.last, "event at {at} after one at {}", self.last);
        self.last = at;
        match msg {
            NoteMsg::On { .. } => {
                let previous = self.sounding.insert(key, (at, voice));
                assert!(previous.is_none(), "NoteOn {key:?} at {at} while it sounds");
                self.ons += 1;
            }
            NoteMsg::Off { .. } => {
                let start = self.sounding.remove(&key);
                let (start, on_voice) =
                    start.unwrap_or_else(|| panic!("NoteOff {key:?} at {at} not on"));
                assert_eq!(voice, on_voice, "NoteOff {key:?} for another voice");
                self.released.push((key.1, at - start));
            }
        }
    }

    fn sounds(&self, channel: u8, note: u8) -> bool {
        self.sounding.contains_key(&(channel, note))
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
    notes.root_on(0, 0, 60, 0.9, None, emit);
    notes.set_chord_expansion(false);
    notes.root_off(10, 0, 60, None, emit);
    // The forwarded root and its chord tones.
    assert_eq!(host.ons, 1 + CHORD_INTERVALS.len());
    host.assert_all_released();
}

#[test]
fn no_stray_note_offs_when_expansion_is_switched_on_mid_note() {
    let mut notes = DrumTriggerNotes::new(SR);
    let mut host = Host::default();
    let emit = &mut |msg| host.receive(msg);
    notes.set_chord_expansion(false);
    notes.root_on(0, 0, 60, 0.9, None, emit);
    notes.set_chord_expansion(true);
    notes.root_off(10, 0, 60, None, emit);
    // Only the forwarded root itself sounded and was released.
    assert_eq!(host.ons, 1);
    assert_eq!(host.released, [(60, 10)]);
}

#[test]
fn roots_sharing_a_chord_tone_never_stack_it() {
    // 60 voices 63, 67, 70, 74 and 63 voices 66, 70, 73, 77: both hold 70.
    let mut notes = DrumTriggerNotes::new(SR);
    let mut host = Host::default();
    let emit = &mut |msg| host.receive(msg);
    notes.root_on(0, 0, 60, 0.9, None, emit);
    notes.root_on(1, 0, 63, 0.9, None, emit);
    notes.root_off(2, 0, 60, None, emit);
    assert!(host.sounds(0, 70), "70 cut while 63 holds it");
    assert!(host.sounds(0, 63), "root 63 cut by 60's chord tone 63");
    notes.root_off(3, 0, 63, None, &mut |msg| host.receive(msg));
    host.assert_all_released();
    assert!(notes.is_idle());
}

#[test]
fn repeated_root_note_on_retriggers_its_chord() {
    let mut notes = DrumTriggerNotes::new(SR);
    let mut host = Host::default();
    let emit = &mut |msg| host.receive(msg);
    notes.root_on(0, 0, 60, 0.9, None, emit);
    notes.root_on(5, 0, 60, 0.9, None, emit);
    notes.root_off(9, 0, 60, None, emit);
    assert_eq!(host.ons, 2 * (1 + CHORD_INTERVALS.len()));
    host.assert_all_released();
}

#[test]
fn chord_tone_never_cuts_off_a_held_root_on_the_same_key() {
    // Root 57 voices 60, 64, 67, 71, so its chord tone 60 shares the held root 60's key.
    let mut notes = DrumTriggerNotes::new(SR);
    let mut host = Host::default();
    notes.root_on(0, 0, 60, 0.9, Some(1), &mut |m| host.receive(m));
    notes.root_on(1, 0, 57, 0.9, Some(2), &mut |m| host.receive(m));
    notes.root_off(2, 0, 57, Some(2), &mut |m| host.receive(m));
    assert!(host.sounds(0, 60), "root 60 cut off by root 57's chord");
    // And the other way round: root 60 released while 57's chord tone still holds 60.
    notes.root_on(3, 0, 57, 0.9, Some(3), &mut |m| host.receive(m));
    notes.root_off(4, 0, 60, Some(1), &mut |m| host.receive(m));
    assert!(
        host.sounds(0, 60),
        "57's chord tone 60 cut off by root 60's NoteOff"
    );
    notes.root_off(5, 0, 57, Some(3), &mut |m| host.receive(m));
    host.assert_all_released();
    assert!(notes.is_idle());
}

#[test]
fn older_voice_note_off_keeps_the_newer_voice_chord() {
    let mut notes = DrumTriggerNotes::new(SR);
    let mut host = Host::default();
    notes.root_on(0, 0, 60, 0.9, Some(10), &mut |m| host.receive(m));
    notes.root_on(1, 0, 60, 0.9, Some(11), &mut |m| host.receive(m));
    notes.root_off(2, 0, 60, Some(10), &mut |m| host.receive(m));
    for note in CHORD_60 {
        assert!(
            host.sounds(0, note),
            "voice 10's NoteOff cut voice 11's tone {note}"
        );
    }
    assert!(host.sounds(0, 60), "voice 10's NoteOff cut voice 11's root");
    notes.root_off(3, 0, 60, Some(11), &mut |m| host.receive(m));
    host.assert_all_released();
    assert!(notes.is_idle());
}

#[test]
fn wildcard_note_off_releases_every_voice_on_its_key() {
    let mut notes = DrumTriggerNotes::new(SR);
    let mut host = Host::default();
    notes.root_on(0, 0, 60, 0.9, Some(10), &mut |m| host.receive(m));
    notes.root_on(1, 0, 60, 0.9, Some(11), &mut |m| host.receive(m));
    notes.root_off(2, 0, 60, None, &mut |m| host.receive(m));
    host.assert_all_released();
    assert!(notes.is_idle());
}

#[test]
fn full_root_table_releases_its_oldest_instance() {
    // Every voice gets its own channel and key, so each instance sounds on its own.
    let key = |voice: usize| ((voice % 16) as u8, 36 + (voice / 16) as u8);
    let mut notes = DrumTriggerNotes::new(SR);
    let mut host = Host::default();
    notes.set_chord_expansion(false);
    for voice in 0..=MAX_ROOT_INSTANCES {
        let (channel, root) = key(voice);
        notes.root_on(0, channel, root, 0.9, Some(voice as i32), &mut |m| {
            host.receive(m)
        });
    }
    // Voice 0 was released to make room; its own NoteOff then matches nothing and is dropped.
    let (channel, root) = key(0);
    assert!(!host.sounds(channel, root), "oldest voice not released");
    assert_eq!(host.sounding.len(), MAX_ROOT_INSTANCES);
    for voice in 0..=MAX_ROOT_INSTANCES {
        let (channel, root) = key(voice);
        notes.root_off(1, channel, root, Some(voice as i32), &mut |m| {
            host.receive(m)
        });
    }
    host.assert_all_released();
    assert!(notes.is_idle());
}

#[test]
fn release_all_closes_held_chords_and_drum_gates() {
    let mut notes = DrumTriggerNotes::new(SR);
    let mut host = Host::default();
    let emit = &mut |msg| host.receive(msg);
    notes.root_on(0, 3, 48, 0.7, None, emit);
    notes.drum_hit(1, 0, 36, 0.8, emit);
    notes.drum_hit(1, 2, 42, 0.8, emit);
    notes.release_all(2, emit);
    host.assert_all_released();
    // The root's own NoteOff and later gate ticks must not release anything twice.
    let emit = &mut |msg| host.receive(msg);
    notes.root_off(3, 3, 48, None, emit);
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
    notes.root_on(0, 0, 60, 7.5, None, emit);
    notes.root_on(0, 1, 60, f32::NAN, None, emit);
    notes.drum_hit(0, 1, 38, -3.0, emit);
    notes.drum_hit(0, 2, 42, f32::INFINITY, emit);
    notes.release_all(1, emit);
    host.assert_all_released();
}
