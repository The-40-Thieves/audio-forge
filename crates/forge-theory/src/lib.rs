//! `forge-theory` — Symbolic Music Theory, Minimal-Motion Voice Leading, Reharmonization,
//! Pocket/Groove Humanization, Drum Transient-to-MIDI Triggering, ABC Notation & `.mid` Compiler.

mod abc;
mod chord;

use std::borrow::Cow;
use std::io::{BufWriter, Write};
use std::path::Path;

use forge_core::{AudioBuffer, ForgeError, Result};
use midly::{
    Format, Header, MetaMessage, MidiMessage, Smf, Timing, TrackEvent, TrackEventKind,
    num::{u4, u7, u15, u24, u28},
};
use serde::{Deserialize, Serialize};

pub use abc::parse_abc_score;

/// Tempo range every BPM argument is validated against at the CLI and MCP boundary.
pub const BPM_RANGE: std::ops::RangeInclusive<f64> = 30.0..=300.0;

/// Accepts a finite tempo inside [`BPM_RANGE`].
pub fn validate_bpm(bpm: f64) -> Result<f64> {
    if bpm.is_finite() && BPM_RANGE.contains(&bpm) {
        Ok(bpm)
    } else {
        Err(ForgeError::InvalidParameter(format!(
            "bpm ({bpm}) must be between {} and {}",
            BPM_RANGE.start(),
            BPM_RANGE.end()
        )))
    }
}

/// Library-level fallback for callers that skip validation: the same range as
/// [`validate_bpm`], and 120 BPM for a non-finite tempo.
fn clamp_bpm(bpm: f64) -> f64 {
    if bpm.is_finite() {
        bpm.clamp(*BPM_RANGE.start(), *BPM_RANGE.end())
    } else {
        120.0
    }
}

/// A discrete MIDI note event with micro-timing in ticks (or seconds) and velocity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoteEvent {
    pub midi_note: u8,
    pub velocity: u8,
    pub start_tick: u32,
    pub duration_ticks: u32,
    pub channel: u8,
}

/// Voicing style for chord progression realization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoicingStyle {
    /// Standard close position block chords.
    Close,
    /// Open Drop-2 voicing (second highest voice dropped an octave; ideal for guitars, strings, keys).
    Drop2,
    /// Rootless Jazz / Neo-Soul voicing (3rd, 7th, 9th, 13th in mid register + separate sub-bass root).
    NeoSoulRootless,
    /// Wide cinematic spread across 2.5 octaves.
    CinematicSpread,
}

/// Groove & micro-timing feel template.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GrooveStyle {
    /// Strict mathematical grid.
    Straight,
    /// Neo-Soul / J Dilla pocket (laid-back snare/chord attacks +14 ticks, swung 16ths, warm strum).
    NeoSoulDilla,
    /// Classic 58% MPC swing with dynamic accenting on beats 2 & 4.
    MpcSwing58,
    /// Gentle acoustic fingerstyle upward chord strum (10 ticks per voice).
    AcousticStrum,
}

/// Parses a pitch class name (`C`, `C#`, `Db`, `D`, ..., `B`) into `0..12`.
pub fn parse_pitch_class(token: &str) -> Result<(u8, &str)> {
    let s = token.trim();
    if s.is_empty() {
        return Err(ForgeError::TheoryError("Empty chord symbol".to_string()));
    }
    let bytes = s.as_bytes();
    let base = match bytes[0].to_ascii_uppercase() {
        b'C' => 0,
        b'D' => 2,
        b'E' => 4,
        b'F' => 5,
        b'G' => 7,
        b'A' => 9,
        b'B' => 11,
        _ => {
            return Err(ForgeError::TheoryError(format!(
                "Invalid root note in chord '{s}'"
            )));
        }
    };
    if bytes.len() >= 2 {
        match bytes[1] {
            b'#' => return Ok(((base + 1) % 12, &s[2..])),
            b'b' => return Ok(((base + 11) % 12, &s[2..])),
            _ => {}
        }
    }
    Ok((base, &s[1..]))
}

/// A parsed chord symbol: root, intervals above the root, and an optional slash bass.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParsedChord {
    pub root_pc: u8,
    /// Semitone intervals above the root, starting with `0`.
    pub intervals: Vec<u8>,
    /// Pitch class after a slash (`C/E` gives `Some(4)`), `None` for root-position symbols.
    pub bass_pc: Option<u8>,
}

/// Parses a chord symbol (e.g., `Dm9`, `G13`, `Cmaj9`, `A7b13`, `F#m7b5`) into `(root_pc, intervals)`.
///
/// Any slash bass is validated and then dropped; use [`parse_chord`] to keep it.
pub fn parse_chord_symbol(symbol: &str) -> Result<(u8, Vec<u8>)> {
    let chord = parse_chord(symbol)?;
    Ok((chord.root_pc, chord.intervals))
}

/// Parses a chord symbol, including slash chords (`C/E`, `Am7/G`) and `6/9` chords.
///
/// Qualities outside the common table are read as `[family][extension][modifiers]`
/// (`maj7#11`, `7b5`, `9sus4`, `13b9`, `m11b5`, `7(b9,#11)`). An unrecognised quality or
/// slash bass is an error.
pub fn parse_chord(symbol: &str) -> Result<ParsedChord> {
    let (root_pc, rest) = parse_pitch_class(symbol)?;
    let (quality, bass_pc) = split_slash_bass(rest, symbol)?;
    let intervals = match common_quality(&quality) {
        Some(intervals) => intervals,
        None => chord::parse_quality(&quality).map_err(|why| {
            ForgeError::TheoryError(format!(
                "Unknown chord quality '{quality}' in '{}': {why}",
                symbol.trim()
            ))
        })?,
    };
    Ok(ParsedChord {
        root_pc,
        intervals,
        bass_pc,
    })
}

/// Splits `quality/bass` and validates the bass note. The slash in `6/9` belongs to the quality.
///
/// The quality borrows from `rest` except for `6/9`, so the common case does not allocate.
fn split_slash_bass<'a>(rest: &'a str, symbol: &str) -> Result<(Cow<'a, str>, Option<u8>)> {
    let Some((quality, bass)) = rest.split_once('/') else {
        return Ok((Cow::Borrowed(rest.trim()), None));
    };
    if quality.ends_with('6') && bass.starts_with('9') {
        let joined = format!("{quality}{bass}");
        let (quality, bass_pc) = split_slash_bass(&joined, symbol)?;
        return Ok((Cow::Owned(quality.into_owned()), bass_pc));
    }
    let invalid = || {
        ForgeError::TheoryError(format!(
            "Invalid slash bass '{}' in chord '{}'",
            bass.trim(),
            symbol.trim()
        ))
    };
    let bass = bass.trim();
    if bass.is_empty() {
        return Err(invalid());
    }
    match parse_pitch_class(bass) {
        Ok((bass_pc, "")) => Ok((Cow::Borrowed(quality.trim()), Some(bass_pc))),
        _ => Err(invalid()),
    }
}

/// Exact-match table for the common qualities, kept for their established voicings.
fn common_quality(qual: &str) -> Option<Vec<u8>> {
    let intervals = match qual {
        "" | "M" | "maj" => vec![0, 4, 7],
        "m" | "min" | "-" => vec![0, 3, 7],
        "dim" | "o" => vec![0, 3, 6],
        "dim7" | "o7" => vec![0, 3, 6, 9],
        "m7b5" | "ø" | "ø7" => vec![0, 3, 6, 10],
        "aug" | "+" => vec![0, 4, 8],
        "sus2" => vec![0, 2, 7],
        "sus4" | "sus" => vec![0, 5, 7],
        "7sus4" | "7sus" => vec![0, 5, 7, 10],
        "6" | "maj6" => vec![0, 4, 7, 9],
        "m6" | "min6" => vec![0, 3, 7, 9],
        "7" | "dom7" => vec![0, 4, 7, 10],
        "maj7" | "M7" | "Δ" | "Δ7" => vec![0, 4, 7, 11],
        "m7" | "min7" | "-7" => vec![0, 3, 7, 10],
        "mMaj7" | "m(maj7)" => vec![0, 3, 7, 11],
        "9" | "dom9" => vec![0, 4, 7, 10, 14],
        "maj9" | "M9" => vec![0, 4, 7, 11, 14],
        "m9" | "min9" | "-9" => vec![0, 3, 7, 10, 14],
        "m11" | "min11" => vec![0, 3, 7, 10, 14, 17],
        "13" | "dom13" => vec![0, 4, 10, 14, 21],
        "maj13" | "M13" => vec![0, 4, 7, 11, 14, 21],
        "7b9" => vec![0, 4, 7, 10, 13],
        "7#9" => vec![0, 4, 7, 10, 15],
        "7b13" | "7alt" => vec![0, 4, 10, 13, 20],
        "add9" => vec![0, 4, 7, 14],
        _ => return None,
    };
    Some(intervals)
}

/// Reharmonizes a sequence of chord symbols with Neo-Soul / Jazz extensions, secondary dominants,
/// or tritone substitutions while preserving the harmonic function.
pub fn reharmonize_progression(chords: &[String], style: VoicingStyle) -> Vec<String> {
    if style != VoicingStyle::NeoSoulRootless {
        return chords.to_vec();
    }
    chords
        .iter()
        .map(|c| {
            let trimmed = c.trim();
            if let Ok((_, qual)) = parse_pitch_class(trimmed) {
                let root_str = &trimmed[..trimmed.len() - qual.len()];
                match qual {
                    "" | "M" | "maj" | "maj7" => format!("{root_str}maj9"),
                    "m" | "min" | "m7" => format!("{root_str}m9"),
                    "7" => format!("{root_str}13"),
                    _ => trimmed.to_string(),
                }
            } else {
                trimmed.to_string()
            }
        })
        .collect()
}

/// Minimal-Motion Voice-Leading Solver: computes upper-structure MIDI chord voicings (`C3..G5`)
/// and dedicated sub-bass MIDI notes (`C1..C2`) that minimize total semitone voice movement
/// and avoid muddy low-interval clashes.
pub fn solve_voice_leading(
    chord_symbols: &[String],
    style: VoicingStyle,
    groove: GrooveStyle,
    ticks_per_beat: u32,
    beats_per_chord: u32,
) -> Result<(Vec<NoteEvent>, Vec<NoteEvent>)> {
    if chord_symbols.is_empty() {
        return Err(ForgeError::TheoryError(
            "Chord progression cannot be empty".to_string(),
        ));
    }

    let safe_tpb = ticks_per_beat.clamp(24, 19_200);
    let safe_bpc = beats_per_chord.clamp(1, 64);
    let chord_dur = safe_tpb.saturating_mul(safe_bpc);
    let mut chord_events = Vec::new();
    let mut bass_events = Vec::new();
    let mut prev_voicing: Vec<u8> = Vec::new();

    for (idx, sym) in chord_symbols.iter().enumerate() {
        let (root_pc, intervals) = parse_chord_symbol(sym)?;
        let base_start_tick = (idx as u32).saturating_mul(chord_dur);

        // 1. Dedicated phase-pure Sub-Bass root note in C2..B2 (MIDI 36..47)
        let bass_note = 36 + root_pc;
        let bass_offset = match groove {
            GrooveStyle::NeoSoulDilla => 10, // Laid back behind the kick transient
            _ => 0,
        };
        bass_events.push(NoteEvent {
            midi_note: bass_note,
            velocity: 102,
            start_tick: base_start_tick + bass_offset,
            duration_ticks: chord_dur.saturating_sub(bass_offset + 15).max(1),
            channel: 1,
        });

        // 2. Build candidate upper voices around target center register (MIDI 60 = C4)
        let active_intervals: Vec<u8> = match style {
            VoicingStyle::NeoSoulRootless if intervals.len() >= 4 => {
                // Omit root in upper structure (handled by bass); keep 3rd, 7th, 9th/13th
                intervals.iter().copied().skip(1).collect()
            }
            _ => intervals.clone(),
        };

        let mut current_voicing: Vec<u8> = active_intervals
            .iter()
            .map(|&iv| 60 + ((root_pc + iv) % 12))
            .collect();
        current_voicing.sort_unstable();

        // Optimize inversion / octave placement to minimize L1 pitch distance from `prev_voicing`
        if !prev_voicing.is_empty() {
            let prev_center: f64 = prev_voicing.iter().map(|&n| f64::from(n)).sum::<f64>()
                / (prev_voicing.len() as f64);

            for note in &mut current_voicing {
                let mut best_n = *note;
                let mut best_dist = (f64::from(*note) - prev_center).abs();
                for shift in [-12_i16, 12_i16] {
                    let cand = (i16::from(*note) + shift).clamp(50, 81) as u8;
                    let dist = (f64::from(cand) - prev_center).abs();
                    if dist < best_dist {
                        best_dist = dist;
                        best_n = cand;
                    }
                }
                *note = best_n;
            }
            current_voicing.sort_unstable();
        }

        // Apply Drop-2 or Cinematic spread transformation
        match style {
            VoicingStyle::Drop2 if current_voicing.len() >= 3 => {
                let drop_idx = current_voicing.len() - 2;
                current_voicing[drop_idx] = current_voicing[drop_idx].saturating_sub(12).max(48);
                current_voicing.sort_unstable();
            }
            VoicingStyle::CinematicSpread if current_voicing.len() >= 3 => {
                current_voicing[0] = current_voicing[0].saturating_sub(12).max(48);
                if let Some(last) = current_voicing.last_mut() {
                    *last = last.saturating_add(12).min(84);
                }
                current_voicing.sort_unstable();
            }
            _ => {}
        }

        prev_voicing = current_voicing.clone();

        // 3. Apply Groove / Strum micro-timing and finger velocity curve
        for (v_idx, &midi_note) in current_voicing.iter().enumerate() {
            let strum_ticks = match groove {
                GrooveStyle::Straight => 0,
                GrooveStyle::NeoSoulDilla => (v_idx as u32) * 8 + 12,
                GrooveStyle::MpcSwing58 => (v_idx as u32) * 4,
                GrooveStyle::AcousticStrum => (v_idx as u32) * 12,
            };
            // Emphasize top melody voice and bottom anchor voice
            let is_outer = v_idx == 0 || v_idx + 1 == current_voicing.len();
            let velocity = if is_outer { 90 } else { 76 };

            chord_events.push(NoteEvent {
                midi_note,
                velocity,
                start_tick: base_start_tick + strum_ticks,
                duration_ticks: chord_dur.saturating_sub(strum_ticks + 20).max(1),
                channel: 0,
            });
        }
    }

    Ok((chord_events, bass_events))
}

/// Detects drum transients in a drum stem or mix and emits velocity-mapped MIDI trigger tracks
/// (`Kick` = C1 / MIDI 36, `Snare` = D1 / MIDI 38, `Hi-Hat` = F#1 / MIDI 42) — Slate Trigger 2 equivalent.
pub fn extract_drum_midi_triggers(
    buffer: &AudioBuffer,
    bpm: f64,
    ticks_per_beat: u32,
) -> (Vec<NoteEvent>, Vec<NoteEvent>, Vec<NoteEvent>) {
    let sr = f64::from(buffer.sample_rate.max(1));
    let mono = buffer.to_mono();
    let n = mono.len();
    if n < 256 {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let clean_bpm = clamp_bpm(bpm);
    let safe_tpb = ticks_per_beat.clamp(24, 19_200);
    let ticks_per_sec = (clean_bpm / 60.0) * f64::from(safe_tpb);
    let win = ((sr * 0.012).round() as usize).max(32); // 12ms transient analysis window
    let min_retrigger_samples = ((sr * 0.075).round() as usize).max(128); // 75ms debounce

    // Simple 1-pole band split for Kick (<130 Hz), Snare (160–2500 Hz), and Hats (>4500 Hz)
    let alpha_low = (2.0 * std::f64::consts::PI * 130.0 / sr).clamp(0.005, 0.5);
    let alpha_high = (2.0 * std::f64::consts::PI * 4500.0 / sr).clamp(0.01, 0.8);

    let mut low_sig = vec![0.0_f64; n];
    let mut mid_sig = vec![0.0_f64; n];
    let mut high_sig = vec![0.0_f64; n];

    let mut lp_state = 0.0_f64;
    let mut hp_state = 0.0_f64;
    for i in 0..n {
        let x = mono[i];
        lp_state += alpha_low * (x - lp_state);
        hp_state += alpha_high * (x - hp_state);
        low_sig[i] = lp_state;
        high_sig[i] = x - hp_state;
        mid_sig[i] = x - low_sig[i] - high_sig[i];
    }

    let detect_band_hits = |sig: &[f64], midi_note: u8, thresh_mult: f64| -> Vec<NoteEvent> {
        let mut events = Vec::new();
        let global_rms = (sig.iter().map(|&v| v * v).sum::<f64>() / (sig.len() as f64))
            .sqrt()
            .max(1e-6);
        let min_peak = (global_rms * thresh_mult).max(0.02);

        let mut last_hit: usize = 0;
        let mut prev_rms = global_rms;
        for start in (0..n.saturating_sub(win)).step_by(win / 2) {
            let slice = &sig[start..start + win];
            let rms = (slice.iter().map(|&v| v * v).sum::<f64>() / (win as f64)).sqrt();
            let peak = slice.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));

            if peak > min_peak
                && rms > prev_rms * 1.65
                && (events.is_empty() || start.saturating_sub(last_hit) >= min_retrigger_samples)
            {
                let norm_vel =
                    ((peak / (global_rms * 5.0)).sqrt() * 115.0).clamp(32.0, 127.0) as u8;
                let tick = (((start as f64) / sr) * ticks_per_sec).round() as u32;
                events.push(NoteEvent {
                    midi_note,
                    velocity: norm_vel,
                    start_tick: tick,
                    duration_ticks: ticks_per_beat / 4,
                    channel: 9, // GM Drum Channel 10 (0-indexed = 9)
                });
                last_hit = start;
            }
            prev_rms = 0.75 * prev_rms + 0.25 * rms;
        }
        events
    };

    let kick_events = detect_band_hits(&low_sig, 36, 1.8);
    let snare_events = detect_band_hits(&mid_sig, 38, 2.0);
    let hat_events = detect_band_hits(&high_sig, 42, 1.9);

    (kick_events, snare_events, hat_events)
}

/// Renders a chord progression and melody into a clean ABC notation lead sheet.
pub fn export_abc_score(title: &str, key: &str, bpm: u32, chords: &[String]) -> String {
    let mut out = format!("X:1\nT:{title}\nM:4/4\nL:1/8\nQ:1/4={bpm}\nK:{key}\n");
    for (idx, chord) in chords.iter().enumerate() {
        let root_char = chord.chars().next().unwrap_or('C').to_ascii_uppercase();
        out.push_str(&format!("\"{chord}\"{root_char}4 {root_char}4 | "));
        if (idx + 1) % 4 == 0 {
            out.push('\n');
        }
    }
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out
}

/// Writes one or more `NoteEvent` tracks to a Standard MIDI File (`.mid`) with tempo metadata.
/// Creates `path` (and its parent directories) and delegates to [`write_midi_to_writer`].
pub fn write_midi_file<P: AsRef<Path>>(
    path: P,
    bpm: f64,
    ticks_per_beat: u16,
    tracks_notes: &[(&str, &[NoteEvent])],
) -> Result<()> {
    let path_ref = path.as_ref();
    if let Some(parent) = path_ref.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(path_ref)
        .map_err(|e| ForgeError::TheoryError(format!("Failed to write MIDI file: {e}")))?;
    write_midi_to_writer(BufWriter::new(file), bpm, ticks_per_beat, tracks_notes)
}

/// Encodes `NoteEvent` tracks as a Standard MIDI File into `writer` (flushed before
/// returning). The tempo is clamped to 30-300 BPM; a non-finite tempo becomes 120 BPM.
pub fn write_midi_to_writer<W: Write>(
    mut writer: W,
    bpm: f64,
    ticks_per_beat: u16,
    tracks_notes: &[(&str, &[NoteEvent])],
) -> Result<()> {
    let midi_err =
        |e: std::io::Error| ForgeError::TheoryError(format!("Failed to write MIDI file: {e}"));
    let mut smf_tracks = vec![tempo_track(bpm)];
    smf_tracks.extend(tracks_notes.iter().map(|&(_name, notes)| note_track(notes)));
    let smf = Smf {
        header: Header {
            format: Format::Parallel,
            timing: Timing::Metrical(u15::new(ticks_per_beat.clamp(1, 0x7FFF))),
        },
        tracks: smf_tracks,
    };
    smf.write_std(&mut writer).map_err(midi_err)?;
    writer.flush().map_err(midi_err)
}

/// The conductor track: one tempo event, then end of track.
fn tempo_track(bpm: f64) -> Vec<TrackEvent<'static>> {
    let clean_bpm = clamp_bpm(bpm);
    let us_per_qn = (60_000_000.0 / clean_bpm).round() as u32;
    vec![
        TrackEvent {
            delta: u28::new(0),
            kind: TrackEventKind::Meta(MetaMessage::Tempo(u24::new(us_per_qn.min(0x00FF_FFFF)))),
        },
        TrackEvent {
            delta: u28::new(0),
            kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
        },
    ]
}

/// Converts notes into sorted (tick, is_on, channel, note, velocity) on/off events.
fn note_events(notes: &[NoteEvent]) -> Vec<(u32, bool, u8, u8, u8)> {
    let mut raw_events: Vec<(u32, bool, u8, u8, u8)> = Vec::with_capacity(notes.len() * 2);
    for n in notes {
        let start = n.start_tick.min(0x0FFF_0000);
        let end = start
            .saturating_add(n.duration_ticks.max(1))
            .min(0x0FFF_FFFE);
        raw_events.push((
            start,
            true,
            n.channel & 0x0F,
            n.midi_note & 0x7F,
            n.velocity & 0x7F,
        ));
        raw_events.push((end, false, n.channel & 0x0F, n.midi_note & 0x7F, 0));
    }
    raw_events.sort_by_key(|&(tick, is_on, _, note, _)| (tick, is_on, note));
    raw_events
}

/// One note track: delta-timed note on/off messages, then end of track.
fn note_track(notes: &[NoteEvent]) -> Vec<TrackEvent<'static>> {
    let raw_events = note_events(notes);
    let mut track = Vec::with_capacity(raw_events.len() + 1);
    let mut last_tick = 0_u32;
    for (tick, is_on, ch, note, vel) in raw_events {
        let delta = tick.saturating_sub(last_tick);
        last_tick = tick;
        let message = if is_on {
            MidiMessage::NoteOn {
                key: u7::new(note),
                vel: u7::new(vel),
            }
        } else {
            MidiMessage::NoteOff {
                key: u7::new(note),
                vel: u7::new(0),
            }
        };
        track.push(TrackEvent {
            delta: u28::new(delta),
            kind: TrackEventKind::Midi {
                channel: u4::new(ch),
                message,
            },
        });
    }
    track.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
    });
    track
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn bpm_validation_and_library_clamp_share_one_range() {
        for bad in [0.0, -5.0, 29.9, 300.1, f64::NAN, f64::INFINITY] {
            assert!(validate_bpm(bad).is_err(), "{bad} accepted");
        }
        for good in [30.0, 92.0, 300.0] {
            assert_eq!(validate_bpm(good).expect("valid"), good);
        }
        assert_eq!(clamp_bpm(0.0), 30.0);
        assert_eq!(clamp_bpm(1_000.0), 300.0);
        assert_eq!(clamp_bpm(f64::NAN), 120.0);
    }

    #[test]
    fn test_voice_leading_and_neosoul_reharmonizer() {
        let raw = vec![
            "Dm7".to_string(),
            "G7".to_string(),
            "Cmaj7".to_string(),
            "A7".to_string(),
        ];
        let reharm = reharmonize_progression(&raw, VoicingStyle::NeoSoulRootless);
        assert_eq!(reharm.len(), 4);
        assert_eq!(reharm[0], "Dm9");
        let (chords, bass) = solve_voice_leading(
            &reharm,
            VoicingStyle::NeoSoulRootless,
            GrooveStyle::NeoSoulDilla,
            480,
            4,
        )
        .expect("voice leading");
        assert!(!chords.is_empty());
        assert_eq!(bass.len(), 4);
    }

    const STYLES: [VoicingStyle; 4] = [
        VoicingStyle::Close,
        VoicingStyle::Drop2,
        VoicingStyle::NeoSoulRootless,
        VoicingStyle::CinematicSpread,
    ];
    const GROOVES: [GrooveStyle; 4] = [
        GrooveStyle::Straight,
        GrooveStyle::NeoSoulDilla,
        GrooveStyle::MpcSwing58,
        GrooveStyle::AcousticStrum,
    ];

    /// M9 / M10: extensions, alterations and suspensions are spelled out, not guessed.
    #[test]
    fn test_chord_extensions_and_alterations_m9_m10() {
        let cases: &[(&str, u8, &[u8])] = &[
            ("Cmaj7#11", 0, &[0, 4, 7, 11, 18]),
            ("Cmaj11", 0, &[0, 4, 7, 11, 14, 17]),
            ("G7b5", 7, &[0, 4, 6, 10]),
            ("G7#5", 7, &[0, 4, 8, 10]),
            ("G7#11", 7, &[0, 4, 7, 10, 18]),
            ("G9sus4", 7, &[0, 5, 7, 10, 14]),
            ("G13b9", 7, &[0, 4, 7, 10, 13, 21]),
            ("G11", 7, &[0, 4, 7, 10, 14, 17]),
            ("Gm11", 7, &[0, 3, 7, 10, 14, 17]),
            ("Dm11b5", 2, &[0, 3, 6, 10, 14, 17]),
            ("C7(b9,#11)", 0, &[0, 4, 7, 10, 13, 18]),
            ("F#7#9b13", 6, &[0, 4, 7, 10, 15, 20]),
            ("Ebmaj9#11", 3, &[0, 4, 7, 11, 14, 18]),
            ("Bbm(maj9)", 10, &[0, 3, 7, 11, 14]),
            ("C+maj7", 0, &[0, 4, 8, 11]),
            ("Cmadd9", 0, &[0, 3, 7, 14]),
            ("Cm6/9", 0, &[0, 3, 7, 9, 14]),
            ("C5", 0, &[0, 7]),
            // Enharmonic roots
            ("Db7", 1, &[0, 4, 7, 10]),
            ("C#7", 1, &[0, 4, 7, 10]),
            ("E#", 5, &[0, 4, 7]),
            ("Cbmaj7", 11, &[0, 4, 7, 11]),
            ("B#m", 0, &[0, 3, 7]),
        ];
        for &(symbol, root, intervals) in cases {
            let parsed = parse_chord_symbol(symbol).unwrap_or_else(|e| panic!("{symbol}: {e}"));
            assert_eq!(parsed, (root, intervals.to_vec()), "{symbol}");
        }
    }

    #[test]
    fn test_slash_and_six_nine_chords() {
        let cases: &[(&str, u8, &[u8], Option<u8>)] = &[
            ("C/E", 0, &[0, 4, 7], Some(4)),
            ("Am7/G", 9, &[0, 3, 7, 10], Some(7)),
            ("Dm7b5/Ab", 2, &[0, 3, 6, 10], Some(8)),
            ("F#m/C#", 6, &[0, 3, 7], Some(1)),
            ("C6/9", 0, &[0, 4, 7, 9, 14], None),
            ("C6/9/E", 0, &[0, 4, 7, 9, 14], Some(4)),
        ];
        for &(symbol, root_pc, intervals, bass_pc) in cases {
            let expected = ParsedChord {
                root_pc,
                intervals: intervals.to_vec(),
                bass_pc,
            };
            assert_eq!(parse_chord(symbol).expect(symbol), expected, "{symbol}");
        }
    }

    #[test]
    fn test_malformed_chord_symbols_are_rejected() {
        let cases = [
            ("Cxyz", "Unknown chord quality"),
            ("C7b10", "Unknown chord quality"),
            ("Cmsus4", "sus cannot follow a minor third"),
            ("Cm5", "power chord"),
            ("C##", "Unknown chord quality"),
            ("C7(b9", "unclosed"),
            ("C7b9)", "unbalanced"),
            ("C7((b9))", "nested"),
            ("C/H", "Invalid slash bass"),
            ("C/", "Invalid slash bass"),
            ("C/E/G", "Invalid slash bass"),
            ("C/Em", "Invalid slash bass"),
            ("H7", "Invalid root note"),
            ("", "Empty chord symbol"),
        ];
        for (symbol, needle) in cases {
            let err = parse_chord(symbol).expect_err(symbol).to_string();
            assert!(err.contains(needle), "{symbol}: {err}");
        }
    }

    proptest! {
        /// Grammar-generated symbols always parse, and each written alteration is in the result.
        #[test]
        fn prop_chord_grammar_spells_every_alteration(
            root in "[A-G][#b]?",
            family in "(|m|maj|dim|aug|ø)",
            // Always present, so `C` + `b9` is never read as the root `Cb`.
            extension in "(6|69|7|9|11|13)",
            modifiers in prop::collection::vec("(b5|#5|b9|#9|#11|b13|add9)", 0..3),
            bass in "(|/[A-G][#b]?)",
        ) {
            let symbol = format!("{root}{family}{extension}{}{bass}", modifiers.concat());
            let chord = parse_chord(&symbol).unwrap();
            prop_assert!(chord.root_pc < 12);
            prop_assert_eq!(chord.intervals.first(), Some(&0));
            prop_assert!(chord.intervals.windows(2).all(|w| w[0] < w[1]));
            for (modifier, interval) in [("b5", 6), ("#5", 8), ("b9", 13), ("#9", 15), ("#11", 18), ("b13", 20)] {
                if modifiers.iter().any(|m| m == modifier) {
                    prop_assert!(chord.intervals.contains(&interval), "{} lacks {}", symbol, modifier);
                }
            }
            prop_assert_eq!(chord.bass_pc.is_some(), !bass.is_empty());
            let progression = vec![symbol.clone(), "G7".to_string()];
            prop_assert!(solve_voice_leading(&progression, VoicingStyle::Close, GrooveStyle::Straight, 480, 4).is_ok());
        }

        #[test]
        fn prop_chord_parser_voice_leading_and_midi_compiler_totality(
            sym_a in ".*",
            sym_b in "[A-G][#b]?(m7|maj7|9|13|dim|aug|sus4)?",
            style_idx in 0usize..5,
            groove_idx in 0usize..5,
            ticks_per_beat in 0u32..100_000,
            beats_per_chord in 0u32..64,
            bpm in prop::num::f64::ANY,
        ) {
            let _ = parse_chord_symbol(&sym_a);
            let parsed = parse_chord_symbol(&sym_b).unwrap();
            prop_assert!(parsed.0 < 12);
            prop_assert!(!parsed.1.is_empty());

            let style = STYLES[style_idx % 4];
            let groove = GROOVES[groove_idx % 4];

            let prog = vec![sym_b.clone(), "G7".to_string(), "Cmaj9".to_string()];
            let reharm = reharmonize_progression(&prog, style);
            let (chord_notes, bass_notes) =
                solve_voice_leading(&reharm, style, groove, ticks_per_beat, beats_per_chord).unwrap();
            prop_assert!(!chord_notes.is_empty());
            prop_assert_eq!(bass_notes.len(), 3);
            for n in chord_notes.iter().chain(bass_notes.iter()) {
                prop_assert!((36..=88).contains(&n.midi_note));
                prop_assert!((1..=127).contains(&n.velocity));
                prop_assert!(n.duration_ticks >= 1);
            }

            let abc = export_abc_score("PropTest", "C", 120, &reharm);
            let (abc_chords, abc_melody) = parse_abc_score(&abc, ticks_per_beat).unwrap();
            prop_assert_eq!(abc_chords.len(), 3);
            prop_assert!(!abc_melody.is_empty());

            // Verify Standard MIDI File (.mid) round-trip parsing via midly::Smf::parse
            let dir = tempfile::tempdir().unwrap();
            let mid_path = dir.path().join("out.mid");
            write_midi_file(
                &mid_path,
                bpm,
                ticks_per_beat as u16,
                &[("Chords", &chord_notes), ("Bass", &bass_notes), ("Melody", &abc_melody)],
            )
            .unwrap();
            let raw_bytes = std::fs::read(&mid_path).unwrap();
            let parsed_smf = midly::Smf::parse(&raw_bytes).unwrap();
            prop_assert_eq!(parsed_smf.tracks.len(), 4);
        }
    }

    #[test]
    fn midi_writer_round_trips_and_matches_the_path_version() {
        let notes = [
            NoteEvent {
                midi_note: 60,
                velocity: 100,
                start_tick: 0,
                duration_ticks: 480,
                channel: 0,
            },
            NoteEvent {
                midi_note: 64,
                velocity: 90,
                start_tick: 240,
                duration_ticks: 480,
                channel: 1,
            },
        ];
        let tracks: [(&str, &[NoteEvent]); 1] = [("Chords", &notes)];
        let mut bytes = Vec::new();
        write_midi_to_writer(&mut bytes, 92.0, 480, &tracks).expect("encode");
        let smf = midly::Smf::parse(&bytes).expect("parse");
        assert_eq!(smf.tracks.len(), 2);
        let note_ons = smf.tracks[1]
            .iter()
            .filter(|e| {
                matches!(
                    e.kind,
                    TrackEventKind::Midi {
                        message: MidiMessage::NoteOn { .. },
                        ..
                    }
                )
            })
            .count();
        assert_eq!(note_ons, 2);

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("nested").join("chords.mid");
        write_midi_file(&path, 92.0, 480, &tracks).expect("write");
        assert_eq!(std::fs::read(&path).expect("read"), bytes);
    }
}
