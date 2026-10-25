//! Note bookkeeping for the Drum Trigger & Theory MIDI plugin.
//!
//! Every NoteOn this module emits gets exactly one matching NoteOff:
//!
//! - audio-detected drum hits are gated, with a NoteOff a fixed length after each NoteOn;
//! - incoming roots are forwarded through this module too, and each root instance remembers the
//!   chord tones it added, so its NoteOff releases them even if chord expansion was switched off
//!   while it was held;
//! - every channel and key has one occupancy count over all of its holders: forwarded roots,
//!   chord tones and drum slots alike. A NoteOn for a key that is already sounding re-articulates
//!   it (a NoteOff, then the NoteOn, at the same timing) and adds a holder. A NoteOff from any
//!   holder only drops its count, and the key's NoteOff reaches the host when the last holder
//!   lets go. So a chord tone never cuts off a held root on the same key, two roots that share a
//!   chord tone never stack it, and a hat retrigger never stacks its own note;
//! - [`DrumTriggerNotes::release_all`] closes everything still held, for reset and transport stop.
//!
//! Incoming root instances are identified by channel, key and voice ID (see
//! [`DrumTriggerNotes::root_on`]), so overlapping voices on one key each release only the chord
//! they started. They live in a fixed table of [`MAX_ROOT_INSTANCES`] entries; when it is full the
//! oldest instance is released early to make room.
//!
//! The module is allocation-free and knows nothing about the plugin framework: messages go out
//! through a caller-supplied `emit` closure in the order they must reach the host.

/// A note message for the host. Only one instance of a channel and key sounds at a time, so a
/// NoteOff matches it by channel and key. A forwarded root keeps its incoming voice ID and
/// generated notes carry `None`; a NoteOff carries the voice ID of the instance it closes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NoteMsg {
    /// Starts `note` on `channel` with a velocity in `[0, 1]`.
    On {
        timing: u32,
        channel: u8,
        note: u8,
        velocity: f32,
        voice_id: Option<i32>,
    },
    /// Releases `note` on `channel`.
    Off {
        timing: u32,
        channel: u8,
        note: u8,
        voice_id: Option<i32>,
    },
}

/// General MIDI drum channel 10, zero-based.
pub const DRUM_CHANNEL: u8 = 9;
/// Length of an audio-triggered drum note. Shorter than the shortest hat holdoff (22.5 ms), so
/// a drum slot never retriggers while its previous note is still on.
pub const DRUM_GATE_MS: f64 = 20.0;
/// Rootless Neo-Soul 9th voicing: minor 3rd, 5th, minor 7th and 9th above the root.
pub const CHORD_INTERVALS: [u8; 4] = [3, 7, 10, 14];
/// Incoming root instances held at once, over all channels and keys. When the table is full, a
/// new root first releases the oldest instance, so no note is ever left untracked.
pub const MAX_ROOT_INSTANCES: usize = 64;
/// Chord tones above this note are dropped.
const MAX_CHORD_NOTE: u8 = 120;
/// Velocity scale for chord tones relative to their root.
const CHORD_VELOCITY_SCALE: f32 = 0.82;

/// One held incoming root and whether it was expanded into chord tones.
#[derive(Clone, Copy)]
struct RootInstance {
    channel: u8,
    root: u8,
    voice_id: Option<i32>,
    expanded: bool,
    /// Start order, so a full table releases its oldest instance.
    serial: u64,
}

impl RootInstance {
    fn is(&self, channel: u8, root: u8) -> bool {
        self.channel == channel && self.root == root
    }
}

/// Tracks which notes are sounding and emits the NoteOffs that close them.
pub struct DrumTriggerNotes {
    /// Number of holders (forwarded roots, chord tones or drum slots) of each channel and key.
    holders: [[u8; 128]; 16],
    /// Voice ID of the instance sounding on each channel and key.
    voice: [[Option<i32>; 128]; 16],
    roots: [Option<RootInstance>; MAX_ROOT_INSTANCES],
    next_serial: u64,
    /// Samples left before each drum slot's note is released; 0 when idle.
    gate_left: [u32; 3],
    gate_note: [u8; 3],
    gate_len: u32,
    expand_chords: bool,
}

impl DrumTriggerNotes {
    pub fn new(sample_rate: f64) -> Self {
        Self {
            holders: [[0; 128]; 16],
            voice: [[None; 128]; 16],
            roots: [None; MAX_ROOT_INSTANCES],
            next_serial: 0,
            gate_left: [0; 3],
            gate_note: [0; 3],
            gate_len: gate_samples(sample_rate),
            expand_chords: true,
        }
    }

    /// Changes the drum gate length. Held notes stay tracked so they can still be released.
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.gate_len = gate_samples(sample_rate);
    }

    /// Sets whether roots that start from now on are expanded into chord tones. Roots already
    /// held keep the chord they started with.
    pub fn set_chord_expansion(&mut self, expand: bool) {
        self.expand_chords = expand;
    }

    /// True when no note, forwarded or generated, is sounding.
    pub fn is_idle(&self) -> bool {
        self.holders.iter().flatten().all(|&h| h == 0)
    }

    /// An incoming root NoteOn. Forwards it with its `voice_id`, then adds its chord tones when
    /// chord expansion is on. An instance is identified by channel, key and voice ID: a NoteOn
    /// for an instance that is already held (the same ID, or `None` again) first releases it,
    /// while instances with different voice IDs on one key are held side by side.
    pub fn root_on(
        &mut self,
        timing: u32,
        channel: u8,
        root: u8,
        velocity: f32,
        voice_id: Option<i32>,
        emit: &mut impl FnMut(NoteMsg),
    ) {
        if key_index(channel, root).is_none() {
            return;
        }
        let held = self
            .roots
            .iter()
            .position(|r| r.is_some_and(|r| r.is(channel, root) && r.voice_id == voice_id));
        if let Some(index) = held {
            self.end_root(timing, index, emit);
        }
        let index = self.free_root_slot(timing, emit);
        let expanded = self.expand_chords;
        self.roots[index] = Some(RootInstance {
            channel,
            root,
            voice_id,
            expanded,
            serial: self.next_serial,
        });
        self.next_serial += 1;
        self.hold(
            timing,
            channel,
            root,
            clean_velocity(velocity),
            voice_id,
            emit,
        );
        if expanded {
            let velocity = chord_velocity(velocity);
            for note in chord_tones(root) {
                self.hold(timing, channel, note, velocity, None, emit);
            }
        }
    }

    /// An incoming root NoteOff. Releases the matching instance, its root and the chord tones it
    /// started, whatever the current expansion setting. `Some(id)` releases only the instance
    /// started with that voice ID; `None` is a wildcard and releases every instance on the
    /// channel and key. A NoteOff that matches no held instance is dropped.
    pub fn root_off(
        &mut self,
        timing: u32,
        channel: u8,
        root: u8,
        voice_id: Option<i32>,
        emit: &mut impl FnMut(NoteMsg),
    ) {
        for index in 0..self.roots.len() {
            let matches = self.roots[index].is_some_and(|r| {
                r.is(channel, root) && (voice_id.is_none() || r.voice_id == voice_id)
            });
            if matches {
                self.end_root(timing, index, emit);
            }
        }
    }

    /// Advances the drum gates by one sample, releasing any whose time is up at `timing`. Call
    /// once per sample, before that sample's [`Self::drum_hit`]s.
    pub fn tick(&mut self, timing: u32, emit: &mut impl FnMut(NoteMsg)) {
        for slot in 0..self.gate_left.len() {
            if self.gate_left[slot] == 0 {
                continue;
            }
            self.gate_left[slot] -= 1;
            if self.gate_left[slot] == 0 {
                self.release(timing, DRUM_CHANNEL, self.gate_note[slot], emit);
            }
        }
    }

    /// An audio-detected hit in drum `slot` (0 kick, 1 snare, 2 hat), gated for
    /// [`DRUM_GATE_MS`]. A hit on a slot whose note is still on closes that note first.
    pub fn drum_hit(
        &mut self,
        timing: u32,
        slot: usize,
        note: u8,
        velocity: f32,
        emit: &mut impl FnMut(NoteMsg),
    ) {
        if slot >= self.gate_left.len() || note > 127 {
            return;
        }
        if self.gate_left[slot] > 0 {
            self.release(timing, DRUM_CHANNEL, self.gate_note[slot], emit);
        }
        self.hold(
            timing,
            DRUM_CHANNEL,
            note,
            clean_velocity(velocity),
            None,
            emit,
        );
        self.gate_note[slot] = note;
        self.gate_left[slot] = self.gate_len;
    }

    /// Releases every note still sounding, forwarded roots included, at `timing`. Later root
    /// NoteOffs for the instances it released are dropped.
    pub fn release_all(&mut self, timing: u32, emit: &mut impl FnMut(NoteMsg)) {
        let keys = self.holders.iter_mut().zip(self.voice.iter_mut());
        for (channel, (holders, voices)) in (0_u8..).zip(keys) {
            for (note, (held, voice_id)) in (0_u8..).zip(holders.iter_mut().zip(voices)) {
                if std::mem::take(held) > 0 {
                    emit(NoteMsg::Off {
                        timing,
                        channel,
                        note,
                        voice_id: voice_id.take(),
                    });
                }
            }
        }
        self.roots = [None; MAX_ROOT_INSTANCES];
        self.gate_left = [0; 3];
    }

    /// Releases a held root instance: its root key and, if it was expanded, its chord tones.
    fn end_root(&mut self, timing: u32, index: usize, emit: &mut impl FnMut(NoteMsg)) {
        let Some(instance) = self.roots[index].take() else {
            return;
        };
        let (channel, root) = (instance.channel, instance.root);
        self.release(timing, channel, root, emit);
        if instance.expanded {
            for note in chord_tones(root) {
                self.release(timing, channel, note, emit);
            }
        }
    }

    /// A free root slot, releasing the oldest instance to make one when the table is full.
    fn free_root_slot(&mut self, timing: u32, emit: &mut impl FnMut(NoteMsg)) -> usize {
        if let Some(index) = self.roots.iter().position(Option::is_none) {
            return index;
        }
        let oldest = (0..self.roots.len())
            .min_by_key(|&i| self.roots[i].map_or(u64::MAX, |r| r.serial))
            .unwrap_or(0);
        self.end_root(timing, oldest, emit);
        oldest
    }

    /// Adds a holder to a key and starts it, re-articulating the sounding instance (a NoteOff
    /// first) so two instances of one key never stack.
    fn hold(
        &mut self,
        timing: u32,
        channel: u8,
        note: u8,
        velocity: f32,
        voice_id: Option<i32>,
        emit: &mut impl FnMut(NoteMsg),
    ) {
        let Some((c, n)) = key_index(channel, note) else {
            return;
        };
        if self.holders[c][n] > 0 {
            emit(NoteMsg::Off {
                timing,
                channel,
                note,
                voice_id: self.voice[c][n],
            });
        }
        self.holders[c][n] = self.holders[c][n].saturating_add(1);
        self.voice[c][n] = voice_id;
        emit(NoteMsg::On {
            timing,
            channel,
            note,
            velocity,
            voice_id,
        });
    }

    /// Drops a holder from a key and releases it when the last holder lets go.
    fn release(&mut self, timing: u32, channel: u8, note: u8, emit: &mut impl FnMut(NoteMsg)) {
        let Some((c, n)) = key_index(channel, note) else {
            return;
        };
        if self.holders[c][n] == 0 {
            return;
        }
        self.holders[c][n] -= 1;
        if self.holders[c][n] == 0 {
            emit(NoteMsg::Off {
                timing,
                channel,
                note,
                voice_id: self.voice[c][n].take(),
            });
        }
    }
}

fn gate_samples(sample_rate: f64) -> u32 {
    let sr = crate::sanitize_param_f64(sample_rate, 48_000.0, 8_000.0, 768_000.0);
    ((sr * DRUM_GATE_MS / 1000.0).round() as u32).max(1)
}

fn key_index(channel: u8, note: u8) -> Option<(usize, usize)> {
    (channel < 16 && note < 128).then_some((usize::from(channel), usize::from(note)))
}

/// The chord tones added above `root`, skipping any above [`MAX_CHORD_NOTE`].
fn chord_tones(root: u8) -> impl Iterator<Item = u8> {
    CHORD_INTERVALS
        .into_iter()
        .map(move |interval| root.saturating_add(interval))
        .filter(|&note| note <= MAX_CHORD_NOTE)
}

/// Clamps a velocity into `[0, 1]`, mapping NaN to 0.
fn clean_velocity(velocity: f32) -> f32 {
    if velocity.is_finite() {
        velocity.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn chord_velocity(root_velocity: f32) -> f32 {
    (clean_velocity(root_velocity) * CHORD_VELOCITY_SCALE).clamp(0.1, 1.0)
}
