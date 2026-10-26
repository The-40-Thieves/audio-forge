//! ABC notation reader for symbolic lead sheets (`YuE2-3B`, `SheetSage2`).
//!
//! Supports the ABC 2.1 subset those generators emit: `K:` keys and modes (including inline
//! `[K:...]` changes), `L:` and `M:` (with the meter-based default unit length), bar-scoped
//! accidentals, multi-digit and fractional lengths (`A12`, `A/2`, `A3/2`, `A//`), `z`/`x` and
//! multi-bar `Z`/`X` rests, ties, broken rhythm (`>`/`<`), tuplets (`(3abc`, `(p:q:r`), chords in
//! brackets (`[CEG]2`), quoted chord symbols, and skipped decorations, annotations, grace notes,
//! comments and non-music fields. Only the first voice (`V:`) is read. Malformed input is an error.

mod body;
mod fields;
#[cfg(test)]
mod tests;
mod timing;

use forge_core::{ForgeError, Result};

use crate::NoteEvent;
use body::parse_body_line;
use fields::{apply_field, field_line};
use timing::{apply_broken_rhythm, apply_tuplets, render};

/// Largest number accepted in a length, tuplet or multi-bar rest.
const MAX_NUMBER: u32 = 9999;

type Parse<T> = std::result::Result<T, String>;

#[derive(Debug, Clone)]
enum Element {
    /// One note with its length in whole notes. Kept apart from `Notes` so a plain note does
    /// not allocate.
    Note {
        pitch: u8,
        len: f64,
    },
    /// A bracketed chord with its length in whole notes.
    Notes {
        pitches: Vec<u8>,
        len: f64,
    },
    Rest {
        len: f64,
    },
    Tie {
        line: usize,
    },
    Broken {
        line: usize,
        steps: i32,
    },
    Tuplet {
        line: usize,
        p: u32,
        q: u32,
        r: u32,
    },
    Chord(String),
}

impl Element {
    fn is_timed(&self) -> bool {
        matches!(
            self,
            Self::Note { .. } | Self::Notes { .. } | Self::Rest { .. }
        )
    }

    fn scale(&mut self, factor: f64) {
        if let Self::Note { len, .. } | Self::Notes { len, .. } | Self::Rest { len } = self {
            *len *= factor;
        }
    }
}

/// Field state that changes how body notes are read.
struct Context {
    /// `L:` unit note length in whole notes.
    unit: f64,
    unit_set: bool,
    /// Bar length in whole notes, `None` for `M:none`.
    meter: Option<f64>,
    /// Compound meter (6/8, 9/8, 12/8, ...), which changes the default tuplet `q`.
    compound: bool,
    /// Key-signature accidental per letter (`C = 0`).
    key: [i8; 7],
    /// Accidentals written earlier in the bar, keyed by the natural MIDI pitch they alter.
    bar: Vec<(i32, i8)>,
    first_voice: Option<String>,
    voice_active: bool,
}

impl Default for Context {
    fn default() -> Self {
        Self {
            unit: 1.0 / 8.0,
            unit_set: false,
            meter: Some(1.0),
            compound: false,
            key: [0; 7],
            bar: Vec::new(),
            first_voice: None,
            voice_active: true,
        }
    }
}

/// Byte cursor over one body line. It only slices at ASCII positions, so UTF-8 text is safe.
struct Lexer<'a> {
    text: &'a str,
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn peek(&self) -> Option<u8> {
        self.text.as_bytes().get(self.pos).copied()
    }

    fn peek_at(&self, offset: usize) -> Option<u8> {
        self.text.as_bytes().get(self.pos + offset).copied()
    }

    fn bump(&mut self) {
        self.pos += 1;
    }

    fn skip_while(&mut self, pred: impl Fn(u8) -> bool) {
        while self.peek().is_some_and(&pred) {
            self.bump();
        }
    }

    /// Returns the text up to the next `close` byte and moves past it.
    fn take_until(&mut self, close: u8, what: &str) -> Parse<&'a str> {
        let rest = &self.text[self.pos..];
        let end = rest
            .bytes()
            .position(|b| b == close)
            .ok_or_else(|| format!("unterminated {what}"))?;
        self.pos += end + 1;
        Ok(&rest[..end])
    }

    fn number(&mut self) -> Parse<Option<u32>> {
        let start = self.pos;
        self.skip_while(|b| b.is_ascii_digit());
        let digits = &self.text[start..self.pos];
        if digits.is_empty() {
            return Ok(None);
        }
        match digits.parse::<u32>() {
            Ok(n) if n <= MAX_NUMBER => Ok(Some(n)),
            _ => Err(format!("number '{digits}' is larger than {MAX_NUMBER}")),
        }
    }
}

/// Parses a symbolic ABC score (from `YuE2-3B` or `SheetSage2`) into chord symbols and melody `NoteEvent`s.
///
/// Returns an error naming the line for malformed input, such as an unterminated chord symbol,
/// an unknown key, a zero length or a note outside the MIDI range.
pub fn parse_abc_score(abc: &str, ticks_per_beat: u32) -> Result<(Vec<String>, Vec<NoteEvent>)> {
    let mut elements = tokenize(abc)?;
    apply_broken_rhythm(&mut elements)?;
    apply_tuplets(&mut elements)?;
    render(elements, ticks_per_beat)
}

fn line_error(line: usize, message: &str) -> ForgeError {
    ForgeError::TheoryError(format!("ABC line {line}: {message}"))
}

fn tokenize(abc: &str) -> Result<Vec<Element>> {
    let mut ctx = Context::default();
    let mut out = Vec::new();
    for (idx, raw) in abc.lines().enumerate() {
        let line = idx + 1;
        let text = raw.trim();
        if text.is_empty() || text.starts_with('%') {
            continue;
        }
        let result = match field_line(text) {
            Some((letter, value)) => apply_field(&mut ctx, letter, value),
            None => parse_body_line(&mut ctx, text, line, &mut out),
        };
        result.map_err(|message| line_error(line, &message))?;
    }
    Ok(out)
}
