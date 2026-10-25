//! Note bookkeeping for the Drum Trigger & Theory MIDI plugin.
//!
//! Every NoteOn this module emits gets exactly one matching NoteOff:
//!
//! - audio-detected drum hits are gated, with a NoteOff a fixed length after each NoteOn;
//! - chord tones added to an incoming root are remembered per root, so the root's NoteOff
//!   releases them even if chord expansion was switched off while the root was held;
//! - a note that is already sounding is closed before it sounds again, so two roots that share
//!   a chord tone (or a hat retrigger) never stack two instances of one key;
//! - [`DrumTriggerNotes::release_all`] closes everything still held, for reset and transport stop.
//!
//! The module is allocation-free and knows nothing about the plugin framework: messages go out
//! through a caller-supplied `emit` closure in the order they must reach the host.

/// A note message for the host. Emitted notes carry no voice ID: each is unique on its channel
/// and key while it sounds, so a NoteOff matches it by channel and key alone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NoteMsg {
    /// Starts `note` on `channel` with a velocity in `[0, 1]`.
    On {
        timing: u32,
        channel: u8,
        note: u8,
        velocity: f32,
    },
    /// Releases `note` on `channel`.
    Off { timing: u32, channel: u8, note: u8 },
}

/// General MIDI drum channel 10, zero-based.
pub const DRUM_CHANNEL: u8 = 9;
/// Length of an audio-triggered drum note. Shorter than the shortest hat holdoff (22.5 ms), so
/// a drum slot never retriggers while its previous note is still on.
pub const DRUM_GATE_MS: f64 = 20.0;
/// Rootless Neo-Soul 9th voicing: minor 3rd, 5th, minor 7th and 9th above the root.
pub const CHORD_INTERVALS: [u8; 4] = [3, 7, 10, 14];
/// Chord tones above this note are dropped.
const MAX_CHORD_NOTE: u8 = 120;
/// Velocity scale for chord tones relative to their root.
const CHORD_VELOCITY_SCALE: f32 = 0.82;

/// Tracks which generated notes are sounding and emits the NoteOffs that close them.
pub struct DrumTriggerNotes {
    /// Number of sources (roots or drum slots) holding each channel and key.
    holders: [[u8; 128]; 16],
    /// Whether each held root was expanded into chord tones when it started.
    expanded: [[bool; 128]; 16],
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
            expanded: [[false; 128]; 16],
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

    /// True when no generated note is sounding.
    pub fn is_idle(&self) -> bool {
        self.holders.iter().flatten().all(|&h| h == 0)
    }

    /// An incoming root NoteOn, already forwarded by the caller. Adds its chord tones when chord
    /// expansion is on; a repeated NoteOn on a held root first releases the previous chord.
    pub fn root_on(
        &mut self,
        timing: u32,
        channel: u8,
        root: u8,
        velocity: f32,
        emit: &mut impl FnMut(NoteMsg),
    ) {
        let Some(slot) = key_index(channel, root) else {
            return;
        };
        if self.expanded[slot.0][slot.1] {
            self.root_off(timing, channel, root, emit);
        }
        if !self.expand_chords {
            return;
        }
        self.expanded[slot.0][slot.1] = true;
        let velocity = chord_velocity(velocity);
        for note in chord_tones(root) {
            self.hold(timing, channel, note, velocity, emit);
        }
    }

    /// An incoming root NoteOff. Releases the chord tones the root started, whatever the current
    /// expansion setting.
    pub fn root_off(&mut self, timing: u32, channel: u8, root: u8, emit: &mut impl FnMut(NoteMsg)) {
        let Some(slot) = key_index(channel, root) else {
            return;
        };
        if !std::mem::take(&mut self.expanded[slot.0][slot.1]) {
            return;
        }
        for note in chord_tones(root) {
            self.release(timing, channel, note, emit);
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
        self.hold(timing, DRUM_CHANNEL, note, clean_velocity(velocity), emit);
        self.gate_note[slot] = note;
        self.gate_left[slot] = self.gate_len;
    }

    /// Releases every generated note still sounding, at `timing`.
    pub fn release_all(&mut self, timing: u32, emit: &mut impl FnMut(NoteMsg)) {
        for (channel, keys) in (0_u8..).zip(self.holders.iter_mut()) {
            for (note, holders) in (0_u8..).zip(keys.iter_mut()) {
                if std::mem::take(holders) > 0 {
                    emit(NoteMsg::Off {
                        timing,
                        channel,
                        note,
                    });
                }
            }
        }
        self.expanded = [[false; 128]; 16];
        self.gate_left = [0; 3];
    }

    /// Adds a holder to a key, closing the sounding instance first so it never stacks.
    fn hold(
        &mut self,
        timing: u32,
        channel: u8,
        note: u8,
        velocity: f32,
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
            });
        }
        self.holders[c][n] = self.holders[c][n].saturating_add(1);
        emit(NoteMsg::On {
            timing,
            channel,
            note,
            velocity,
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
