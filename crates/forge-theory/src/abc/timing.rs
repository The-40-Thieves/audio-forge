//! Timing passes: broken rhythm, tuplets, and placing notes on the tick grid with ties merged.

use forge_core::Result;

use super::{Element, line_error};
use crate::NoteEvent;

/// Applies `>`/`<` to the timed elements on either side.
pub(super) fn apply_broken_rhythm(elements: &mut [Element]) -> Result<()> {
    for idx in 0..elements.len() {
        let Element::Broken { line, steps } = elements[idx] else {
            continue;
        };
        let prev = elements[..idx].iter().rposition(Element::is_timed);
        let next = elements[idx + 1..]
            .iter()
            .position(Element::is_timed)
            .map(|offset| idx + 1 + offset);
        let (Some(prev), Some(next)) = (prev, next) else {
            return Err(line_error(line, "broken rhythm needs a note on each side"));
        };
        let short = 0.5_f64.powi(steps.abs());
        let (first, second) = if steps > 0 {
            (2.0 - short, short)
        } else {
            (short, 2.0 - short)
        };
        elements[prev].scale(first);
        elements[next].scale(second);
    }
    Ok(())
}

/// Scales the `r` timed elements after each tuplet marker by `q/p`.
pub(super) fn apply_tuplets(elements: &mut [Element]) -> Result<()> {
    for idx in 0..elements.len() {
        let Element::Tuplet { line, p, q, r } = elements[idx] else {
            continue;
        };
        let targets: Vec<usize> = (idx + 1..elements.len())
            .filter(|&i| elements[i].is_timed())
            .take(r as usize)
            .collect();
        if targets.len() < r as usize {
            let message = format!("tuplet ({p} expects {r} notes, found {}", targets.len());
            return Err(line_error(line, &message));
        }
        for target in targets {
            elements[target].scale(f64::from(q) / f64::from(p));
        }
    }
    Ok(())
}

/// Places the timed elements on the tick grid and merges tied notes.
pub(super) fn render(
    elements: &[Element],
    ticks_per_beat: u32,
) -> Result<(Vec<String>, Vec<NoteEvent>)> {
    let whole_note_ticks = 4.0 * f64::from(ticks_per_beat.max(1));
    // Float-to-int `as` saturates, so very long scores clamp to `u32::MAX` instead of wrapping.
    let to_tick = |pos: f64| (pos * whole_note_ticks).round() as u32;
    let mut chords = Vec::new();
    let mut melody: Vec<NoteEvent> = Vec::new();
    let mut last: Vec<usize> = Vec::new();
    let mut tied: Vec<usize> = Vec::new();
    let mut pos = 0.0_f64;
    for element in elements {
        match element {
            Element::Chord(symbol) => chords.push(symbol.clone()),
            Element::Tie { line } if last.is_empty() => {
                return Err(line_error(*line, "tie '-' must follow a note"));
            }
            Element::Tie { .. } => tied.clone_from(&last),
            Element::Rest { len } => {
                pos += len;
                last.clear();
                tied.clear();
            }
            Element::Notes { pitches, len } => {
                let (start, end) = (to_tick(pos), to_tick(pos + len));
                last = emit_notes(&mut melody, &tied, pitches, start, end);
                tied.clear();
                pos += len;
            }
            Element::Broken { .. } | Element::Tuplet { .. } => {}
        }
    }
    Ok((chords, melody))
}

/// Emits one event per pitch, extending a tied event of the same pitch instead of restarting it.
fn emit_notes(
    melody: &mut Vec<NoteEvent>,
    tied: &[usize],
    pitches: &[u8],
    start: u32,
    end: u32,
) -> Vec<usize> {
    let mut indices = Vec::with_capacity(pitches.len());
    for &pitch in pitches {
        if let Some(&idx) = tied.iter().find(|&&idx| melody[idx].midi_note == pitch) {
            let held = end.saturating_sub(melody[idx].start_tick);
            melody[idx].duration_ticks = articulate(held);
            indices.push(idx);
            continue;
        }
        melody.push(NoteEvent {
            midi_note: pitch,
            velocity: 92,
            start_tick: start,
            duration_ticks: articulate(end.saturating_sub(start)),
            channel: 0,
        });
        indices.push(melody.len() - 1);
    }
    indices
}

/// Leaves a short release gap before the next note, never below 20 ticks.
fn articulate(ticks: u32) -> u32 {
    ticks.saturating_sub(10).max(20)
}
