//! ABC notation reader for symbolic lead sheets (`YuE2-3B`, `SheetSage2`).
//!
//! Supports the ABC 2.1 subset those generators emit: `K:` keys and modes (including inline
//! `[K:...]` changes), `L:` and `M:` (with the meter-based default unit length), bar-scoped
//! accidentals, multi-digit and fractional lengths (`A12`, `A/2`, `A3/2`, `A//`), `z`/`x` and
//! multi-bar `Z`/`X` rests, ties, broken rhythm (`>`/`<`), tuplets (`(3abc`, `(p:q:r`), chords in
//! brackets (`[CEG]2`), quoted chord symbols, and skipped decorations, annotations, grace notes,
//! comments and non-music fields. Only the first voice (`V:`) is read. Malformed input is an error.

use std::collections::HashMap;
use std::iter::Peekable;

use forge_core::{ForgeError, Result};

use crate::NoteEvent;

/// Largest number accepted in a length, tuplet or multi-bar rest.
const MAX_NUMBER: u32 = 9999;
/// Semitones above C for the letters `C D E F G A B`.
const LETTER_SEMIS: [i32; 7] = [0, 2, 4, 5, 7, 9, 11];
/// Letter indices (`C = 0`) in the order sharps and flats are added to a key signature.
const SHARP_ORDER: [usize; 7] = [3, 0, 4, 1, 5, 2, 6];
const FLAT_ORDER: [usize; 7] = [6, 2, 5, 1, 4, 0, 3];
/// Clef and voice words that may follow the key in a `K:` field.
const CLEF_WORDS: [&str; 12] = [
    "treble", "bass", "bass3", "baritone", "tenor", "alto", "alto1", "alto2", "mezzo", "soprano",
    "perc", "none",
];

type Parse<T> = std::result::Result<T, String>;

#[derive(Debug, Clone)]
enum Element {
    /// One note, or a bracketed chord, with its length in whole notes.
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
        matches!(self, Self::Notes { .. } | Self::Rest { .. })
    }

    fn scale(&mut self, factor: f64) {
        if let Self::Notes { len, .. } | Self::Rest { len } = self {
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
    bar: HashMap<i32, i8>,
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
            bar: HashMap::new(),
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
    render(&elements, ticks_per_beat)
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
            None if ctx.voice_active => parse_body_line(&mut ctx, text, line, &mut out),
            None => Ok(()),
        };
        result.map_err(|message| line_error(line, &message))?;
    }
    Ok(out)
}

/// A whole-line information field such as `K:G` or `w:lyrics`.
fn field_line(text: &str) -> Option<(u8, &str)> {
    let bytes = text.as_bytes();
    let is_field = bytes.len() >= 2
        && (bytes[0].is_ascii_alphabetic() || bytes[0] == b'+')
        && bytes[1] == b':';
    is_field.then(|| (bytes[0], &text[2..]))
}

fn apply_field(ctx: &mut Context, letter: u8, value: &str) -> Parse<()> {
    match letter {
        b'K' => {
            ctx.key = parse_key(value)?;
            ctx.bar.clear();
        }
        b'L' => {
            ctx.unit = parse_fraction(value)?;
            ctx.unit_set = true;
        }
        b'M' => {
            (ctx.meter, ctx.compound) = parse_meter(value)?;
            if !ctx.unit_set {
                let short_meter = ctx.meter.is_some_and(|m| m < 0.75);
                ctx.unit = if short_meter { 1.0 / 16.0 } else { 1.0 / 8.0 };
            }
        }
        b'V' => {
            let id = value.split_whitespace().next().unwrap_or("").to_string();
            let first = ctx.first_voice.get_or_insert_with(|| id.clone());
            ctx.voice_active = *first == id;
        }
        _ => {}
    }
    Ok(())
}

fn parse_body_line(
    ctx: &mut Context,
    text: &str,
    line: usize,
    out: &mut Vec<Element>,
) -> Parse<()> {
    let mut lx = Lexer { text, pos: 0 };
    while let Some(byte) = lx.peek() {
        // An inline `[V:...]` can switch to another voice mid-line; drop what that voice writes.
        let (active, before) = (ctx.voice_active, out.len());
        match byte {
            b'%' => break,
            b'"' => parse_quoted(&mut lx, out)?,
            b'!' | b'+' => {
                lx.bump();
                lx.take_until(byte, "decoration")?;
            }
            b'{' => {
                lx.bump();
                lx.take_until(b'}', "grace notes")?;
            }
            b'[' => parse_bracket(ctx, &mut lx, out)?,
            b'|' | b':' | b']' => parse_bar(ctx, &mut lx),
            b'(' => parse_paren(ctx, &mut lx, line, out)?,
            b'-' => {
                lx.bump();
                out.push(Element::Tie { line });
            }
            b'>' | b'<' => parse_broken(&mut lx, line, out)?,
            b'&' => return Err("voice overlay '&' is not supported".to_string()),
            b'0'..=b'9' | b'/' => {
                return Err(format!("length '{}' does not follow a note", byte as char));
            }
            _ => parse_note_or_rest(ctx, &mut lx, out)?,
        }
        if !active {
            out.truncate(before);
        }
    }
    Ok(())
}

fn parse_note_or_rest(ctx: &mut Context, lx: &mut Lexer, out: &mut Vec<Element>) -> Parse<()> {
    match lx.peek() {
        Some(b'^' | b'_' | b'=' | b'A'..=b'G' | b'a'..=b'g') => {
            let pitch = parse_pitch(ctx, lx)?;
            let len = parse_length(lx)? * ctx.unit;
            out.push(Element::Notes {
                pitches: vec![pitch],
                len,
            });
        }
        Some(b'z' | b'x') => {
            lx.bump();
            let len = parse_length(lx)? * ctx.unit;
            out.push(Element::Rest { len });
        }
        Some(b'Z' | b'X') => {
            lx.bump();
            let bars = lx.number()?.unwrap_or(1);
            let meter = ctx.meter.ok_or("multi-bar rest needs a meter (M:)")?;
            if bars == 0 {
                return Err("multi-bar rest cannot be zero bars".to_string());
            }
            out.push(Element::Rest {
                len: f64::from(bars) * meter,
            });
        }
        // Spacing, line continuation and one-letter decorations (`~`, `.`, `H`, `T`, ...)
        _ => lx.bump(),
    }
    Ok(())
}

fn parse_quoted(lx: &mut Lexer, out: &mut Vec<Element>) -> Parse<()> {
    lx.bump();
    let text = lx.take_until(b'"', "quoted chord symbol")?.trim();
    // `^_<>@` mark free-text annotations, not chord symbols.
    let is_annotation = text.starts_with(['^', '_', '<', '>', '@']);
    if !text.is_empty() && !is_annotation {
        out.push(Element::Chord(text.to_string()));
    }
    Ok(())
}

fn parse_bar(ctx: &mut Context, lx: &mut Lexer) {
    lx.skip_while(|b| matches!(b, b'|' | b':' | b']'));
    ctx.bar.clear();
    // Numbered repeat endings (`|1`, `:|2`, `|1,3`).
    if lx.peek().is_some_and(|b| b.is_ascii_digit()) {
        lx.skip_while(|b| b.is_ascii_digit() || b == b',' || b == b'-');
    }
}

/// `[K:...]` inline field, `[1` repeat ending, `[|` bar line, or a `[CEG]` chord.
fn parse_bracket(ctx: &mut Context, lx: &mut Lexer, out: &mut Vec<Element>) -> Parse<()> {
    let next = lx.peek_at(1);
    if next.is_some_and(|b| b.is_ascii_alphabetic()) && lx.peek_at(2) == Some(b':') {
        lx.bump();
        let field = lx.take_until(b']', "inline field")?;
        let (letter, value) = field_line(field).ok_or("malformed inline field")?;
        return apply_field(ctx, letter, value);
    }
    if next.is_some_and(|b| b.is_ascii_digit()) {
        lx.bump();
        lx.skip_while(|b| b.is_ascii_digit() || b == b',' || b == b'-');
        return Ok(());
    }
    if next == Some(b'|') {
        lx.bump();
        parse_bar(ctx, lx);
        return Ok(());
    }
    lx.bump();
    parse_chord_notes(ctx, lx, out)
}

/// Notes of a `[CEG]` chord. The chord lasts as long as its first note times the outer length.
fn parse_chord_notes(ctx: &mut Context, lx: &mut Lexer, out: &mut Vec<Element>) -> Parse<()> {
    let mut pitches = Vec::new();
    let mut first_len = None;
    loop {
        match lx.peek() {
            None => return Err("unterminated chord '['".to_string()),
            Some(b']') => break,
            Some(b' ' | b'-') => lx.bump(),
            Some(b'^' | b'_' | b'=' | b'A'..=b'G' | b'a'..=b'g') => {
                pitches.push(parse_pitch(ctx, lx)?);
                let len = parse_length(lx)?;
                first_len.get_or_insert(len);
            }
            Some(other) => return Err(format!("unexpected '{}' inside chord", other as char)),
        }
    }
    lx.bump();
    let first_len = first_len.ok_or("empty chord '[]'")?;
    let len = first_len * parse_length(lx)? * ctx.unit;
    out.push(Element::Notes { pitches, len });
    Ok(())
}

/// A tuplet `(p:q:r` or the start of a slur.
fn parse_paren(ctx: &Context, lx: &mut Lexer, line: usize, out: &mut Vec<Element>) -> Parse<()> {
    lx.bump();
    let Some(p) = lx.number()? else {
        return Ok(());
    };
    if !(2..=9).contains(&p) {
        return Err(format!("tuplet ({p} is not supported (use 2-9)"));
    }
    let mut q = None;
    let mut r = None;
    if lx.peek() == Some(b':') {
        lx.bump();
        q = lx.number()?;
        if lx.peek() == Some(b':') {
            lx.bump();
            r = lx.number()?;
        }
    }
    let q = q.unwrap_or_else(|| default_tuplet_q(p, ctx.compound));
    let r = r.unwrap_or(p);
    if q == 0 || r == 0 {
        return Err("tuplet values cannot be zero".to_string());
    }
    out.push(Element::Tuplet { line, p, q, r });
    Ok(())
}

/// ABC 2.1 default for `q`: 3 for duplets, quadruplets and octuplets, otherwise 2, or 3 for
/// 5, 7 and 9 in compound meter.
fn default_tuplet_q(p: u32, compound: bool) -> u32 {
    match p {
        2 | 4 | 8 => 3,
        3 | 6 => 2,
        _ if compound => 3,
        _ => 2,
    }
}

fn parse_broken(lx: &mut Lexer, line: usize, out: &mut Vec<Element>) -> Parse<()> {
    let Some(symbol) = lx.peek() else {
        return Ok(());
    };
    let start = lx.pos;
    lx.skip_while(|b| b == symbol);
    let count = lx.pos - start;
    if count > 3 {
        return Err(format!(
            "broken rhythm '{}' is longer than three",
            symbol as char
        ));
    }
    let steps = count as i32;
    out.push(Element::Broken {
        line,
        steps: if symbol == b'>' { steps } else { -steps },
    });
    Ok(())
}

/// Accidental, letter and octave marks, resolved against the bar and the key signature.
fn parse_pitch(ctx: &mut Context, lx: &mut Lexer) -> Parse<u8> {
    let accidental = parse_accidental(lx)?;
    let letter = lx.peek().ok_or("accidental without a note")?;
    let (index, mut natural) = match letter {
        b'A'..=b'G' => (letter_index(letter), 60),
        b'a'..=b'g' => (letter_index(letter.to_ascii_uppercase()), 72),
        _ => return Err("accidental must be followed by a note letter".to_string()),
    };
    natural += LETTER_SEMIS[index];
    lx.bump();
    while let Some(mark @ (b'\'' | b',')) = lx.peek() {
        natural = natural.saturating_add(if mark == b'\'' { 12 } else { -12 });
        lx.bump();
    }
    let shift = match accidental {
        Some(shift) => {
            ctx.bar.insert(natural, shift);
            shift
        }
        None => ctx.bar.get(&natural).copied().unwrap_or(ctx.key[index]),
    };
    u8::try_from(natural.saturating_add(i32::from(shift)))
        .ok()
        .filter(|&pitch| pitch <= 127)
        .ok_or_else(|| "note is outside the MIDI range 0-127".to_string())
}

fn letter_index(upper: u8) -> usize {
    match upper {
        b'C' => 0,
        b'D' => 1,
        b'E' => 2,
        b'F' => 3,
        b'G' => 4,
        b'A' => 5,
        _ => 6,
    }
}

fn parse_accidental(lx: &mut Lexer) -> Parse<Option<i8>> {
    let shift = match (lx.peek(), lx.peek_at(1)) {
        (Some(b'^'), Some(b'^')) => 2,
        (Some(b'_'), Some(b'_')) => -2,
        (Some(b'^'), _) => 1,
        (Some(b'_'), _) => -1,
        (Some(b'='), _) => 0,
        _ => return Ok(None),
    };
    lx.pos += if shift == 2 || shift == -2 { 2 } else { 1 };
    if lx.peek().is_some_and(|b| b == b'/' || b.is_ascii_digit()) {
        return Err("microtonal accidentals are not supported".to_string());
    }
    Ok(Some(shift))
}

/// Length multiplier after a note or rest: `2`, `12`, `/`, `//`, `/4`, `3/2`.
fn parse_length(lx: &mut Lexer) -> Parse<f64> {
    let numerator = lx.number()?.unwrap_or(1);
    if numerator == 0 {
        return Err("note length cannot be zero".to_string());
    }
    let mut len = f64::from(numerator);
    while lx.peek() == Some(b'/') {
        lx.bump();
        let denominator = lx.number()?.unwrap_or(2);
        if denominator == 0 {
            return Err("note length cannot divide by zero".to_string());
        }
        len /= f64::from(denominator);
    }
    Ok(len)
}

/// `L:` value such as `1/8`, `1/4` or `1`.
fn parse_fraction(value: &str) -> Parse<f64> {
    let value = value.trim();
    let invalid = || format!("invalid unit length '{value}'");
    let (num, den) = value.split_once('/').unwrap_or((value, "1"));
    let num: u32 = num.trim().parse().map_err(|_| invalid())?;
    let den: u32 = den.trim().parse().map_err(|_| invalid())?;
    if num == 0 || den == 0 || num > MAX_NUMBER || den > MAX_NUMBER {
        return Err(invalid());
    }
    Ok(f64::from(num) / f64::from(den))
}

/// `M:` value such as `4/4`, `6/8`, `C`, `C|`, `(2+3)/8` or `none`.
fn parse_meter(value: &str) -> Parse<(Option<f64>, bool)> {
    let value = value.trim();
    match value {
        "" | "none" => return Ok((None, false)),
        "C" | "C|" => return Ok((Some(1.0), false)),
        _ => {}
    }
    let invalid = || format!("invalid meter '{value}'");
    let (num, den) = value.split_once('/').ok_or_else(invalid)?;
    let mut beats = 0_u32;
    for part in num
        .trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .split('+')
    {
        let n: u32 = part.trim().parse().map_err(|_| invalid())?;
        beats = beats.saturating_add(n);
    }
    let den: u32 = den.trim().parse().map_err(|_| invalid())?;
    if beats == 0 || den == 0 || beats > MAX_NUMBER || den > MAX_NUMBER {
        return Err(invalid());
    }
    let compound = beats > 3 && beats.is_multiple_of(3);
    Ok((Some(f64::from(beats) / f64::from(den)), compound))
}

/// `K:` value: tonic and mode (`G`, `Am`, `F# dor`, `Bbmix`), `none`, `HP`/`Hp`, then optional
/// explicit accidentals (`^f`, `_b`), `exp`, clef words and `name=value` settings.
fn parse_key(value: &str) -> Parse<[i8; 7]> {
    let mut tokens = value.split_whitespace().peekable();
    let mut key = match tokens.peek().copied() {
        None => return Ok([0; 7]),
        Some("none" | "HP") => {
            tokens.next();
            [0; 7]
        }
        Some("Hp") => {
            tokens.next();
            signature_from_fifths(2)
        }
        Some(tonic) if tonic.starts_with(|c: char| ('A'..='G').contains(&c)) => {
            tokens.next();
            key_signature(tonic, &mut tokens)?
        }
        Some(_) => [0; 7],
    };
    for token in tokens {
        apply_key_token(&mut key, token)?;
    }
    Ok(key)
}

fn key_signature<'a>(
    tonic: &'a str,
    tokens: &mut Peekable<impl Iterator<Item = &'a str>>,
) -> Parse<[i8; 7]> {
    let bytes = tonic.as_bytes();
    let letter_fifths = match bytes[0] {
        b'C' => 0,
        b'D' => 2,
        b'E' => 4,
        b'F' => -1,
        b'G' => 1,
        b'A' => 3,
        _ => 5,
    };
    let (fifths, attached) = match bytes.get(1) {
        Some(b'#') => (letter_fifths + 7, &tonic[2..]),
        Some(b'b') => (letter_fifths - 7, &tonic[2..]),
        _ => (letter_fifths, &tonic[1..]),
    };
    let mode = if attached.is_empty() {
        tokens.next_if(|t| mode_offset(t).is_some()).unwrap_or("")
    } else {
        attached
    };
    let offset = match mode {
        "" => 0,
        _ => mode_offset(mode).ok_or_else(|| format!("unknown key mode '{mode}'"))?,
    };
    let total = fifths + offset;
    if !(-7..=7).contains(&total) {
        return Err(format!("key '{tonic}' needs more than 7 sharps or flats"));
    }
    Ok(signature_from_fifths(total))
}

/// Circle-of-fifths offset of a mode relative to its major key (`dor` = -2).
fn mode_offset(word: &str) -> Option<i32> {
    let lower = word.to_ascii_lowercase();
    if lower == "m" {
        return Some(-3);
    }
    if lower.len() < 3 || !lower.bytes().all(|b| b.is_ascii_alphabetic()) {
        return None;
    }
    match &lower[..3] {
        "maj" | "ion" => Some(0),
        "lyd" => Some(1),
        "mix" => Some(-1),
        "dor" => Some(-2),
        "min" | "aeo" => Some(-3),
        "phr" => Some(-4),
        "loc" => Some(-5),
        _ => None,
    }
}

fn signature_from_fifths(fifths: i32) -> [i8; 7] {
    let mut key = [0_i8; 7];
    let count = fifths.unsigned_abs() as usize;
    let (order, shift) = if fifths > 0 {
        (&SHARP_ORDER, 1)
    } else {
        (&FLAT_ORDER, -1)
    };
    for &letter in &order[..count.min(7)] {
        key[letter] = shift;
    }
    key
}

fn apply_key_token(key: &mut [i8; 7], token: &str) -> Parse<()> {
    if token == "exp" {
        *key = [0; 7];
        return Ok(());
    }
    if token.contains('=') && !token.starts_with('=') {
        return Ok(()); // `clef=bass`, `middle=d`, `transpose=-2`
    }
    let clef = token.trim_end_matches("+8").trim_end_matches("-8");
    if CLEF_WORDS.contains(&clef) {
        return Ok(());
    }
    let mut lx = Lexer {
        text: token,
        pos: 0,
    };
    let invalid = || format!("unrecognised key signature token '{token}'");
    let shift = parse_accidental(&mut lx)
        .map_err(|_| invalid())?
        .ok_or_else(invalid)?;
    match token.as_bytes()[lx.pos..] {
        [letter] if b"ABCDEFGabcdefg".contains(&letter) => {
            key[letter_index(letter.to_ascii_uppercase())] = shift;
            Ok(())
        }
        _ => Err(invalid()),
    }
}

/// Applies `>`/`<` to the timed elements on either side.
fn apply_broken_rhythm(elements: &mut [Element]) -> Result<()> {
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
fn apply_tuplets(elements: &mut [Element]) -> Result<()> {
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
fn render(elements: &[Element], ticks_per_beat: u32) -> Result<(Vec<String>, Vec<NoteEvent>)> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const TPB: u32 = 480;

    fn melody(abc: &str) -> Vec<NoteEvent> {
        parse_abc_score(abc, TPB)
            .unwrap_or_else(|e| panic!("{abc}: {e}"))
            .1
    }

    fn pitches(abc: &str) -> Vec<u8> {
        melody(abc).iter().map(|n| n.midi_note).collect()
    }

    fn starts(abc: &str) -> Vec<u32> {
        melody(abc).iter().map(|n| n.start_tick).collect()
    }

    fn summary(abc: &str) -> Vec<(u8, u32, u32)> {
        melody(abc)
            .iter()
            .map(|n| (n.midi_note, n.start_tick, n.duration_ticks))
            .collect()
    }

    #[test]
    fn test_key_signatures_modes_and_key_changes() {
        let cases: &[(&str, &[u8])] = &[
            ("K:G\nF G", &[66, 67]),
            ("K:Bb\nB E A", &[70, 63, 69]),
            ("K:Dm\nB", &[70]),
            ("K:F#m\nG c", &[68, 73]),
            ("K:A dor\nF c", &[66, 72]),
            ("K:Bbmix\nA E", &[68, 63]),
            ("K:E phrygian\nF", &[65]),
            ("K:F lydian\nB", &[71]),
            ("K:C clef=bass\nB,", &[59]),
            ("K:D exp ^g\nF G", &[65, 68]),
            ("K:C\nF [K:G] F", &[65, 66]),
            ("K:C\nF\nK:D\nF c", &[65, 66, 73]),
        ];
        for &(abc, expected) in cases {
            assert_eq!(pitches(abc), expected, "{abc}");
        }
    }

    #[test]
    fn test_accidentals_last_until_the_bar_line() {
        let cases: &[(&str, &[u8])] = &[
            ("K:C\n^F F | F", &[66, 66, 65]),
            ("K:G\n=F F | F", &[65, 65, 66]),
            ("K:C\n^^C __B =C", &[62, 69, 60]),
            ("K:C\n^c c C", &[73, 73, 60]),
            ("K:C\nC' c, C,,", &[72, 60, 36]),
        ];
        for &(abc, expected) in cases {
            assert_eq!(pitches(abc), expected, "{abc}");
        }
    }

    #[test]
    fn test_unit_length_and_duration_forms() {
        // L:1/8 at 480 ticks per beat is 240 ticks per unit.
        assert_eq!(
            starts("L:1/8\nC12 D/2 E3/2 F// G/ A"),
            [0, 2880, 3000, 3360, 3420, 3540]
        );
        assert_eq!(starts("L:1/4\nC D [L:1/16] E F"), [0, 480, 960, 1080]);
        // Without L:, a meter below 3/4 makes the unit a sixteenth.
        assert_eq!(starts("M:2/4\nK:C\nC D"), [0, 120]);
        assert_eq!(starts("M:6/8\nK:C\nC D"), [0, 240]);
        assert_eq!(summary("L:1/4\nC2 D"), [(60, 0, 950), (62, 960, 470)]);
    }

    #[test]
    fn test_rests_ties_broken_rhythm_and_tuplets() {
        let cases: &[(&str, &[u32])] = &[
            ("C z2 D x D", &[0, 720, 1200]),
            ("M:4/4\nL:1/8\nZ2 C X C", &[3840, 6000]),
            ("C>D E<F", &[0, 360, 480, 600]),
            ("C>>D E", &[0, 420, 480]),
            ("(3CDE F", &[0, 160, 320, 480]),
            ("(3:2:2C D E", &[0, 160, 320]),
            ("M:6/8\n(5CDEFG A", &[0, 144, 288, 432, 576, 720]),
        ];
        for &(abc, expected) in cases {
            assert_eq!(starts(abc), expected, "{abc}");
        }
        assert_eq!(
            summary("C2-C2 D | E-|E"),
            [(60, 0, 950), (62, 960, 230), (64, 1200, 470)]
        );
    }

    #[test]
    fn test_bracketed_chords() {
        assert_eq!(
            summary("[CEG]2 [C2E2] [^FA]/"),
            [
                (60, 0, 470),
                (64, 0, 470),
                (67, 0, 470),
                (60, 480, 470),
                (64, 480, 470),
                (66, 960, 110),
                (69, 960, 110),
            ]
        );
    }

    #[test]
    fn test_headers_decorations_annotations_and_voices_are_not_notes() {
        let abc = "X:1\nT:Test Tune\nC:Composer\nM:4/4\nL:1/8\nK:C\n\
                   %%MIDI program 1\n\"Dm9\"!f! D +p+ E \"^Verse\" {g}F .G ~A % comment B\n\
                   w: la la la\nW: Words after\n\"G13\"G |1 c :|2 d |]";
        let (chords, notes) = parse_abc_score(abc, TPB).unwrap();
        assert_eq!(chords, ["Dm9", "G13"]);
        let got: Vec<u8> = notes.iter().map(|n| n.midi_note).collect();
        assert_eq!(got, [62, 64, 65, 67, 69, 67, 72, 74]);
        assert_eq!(pitches("V:1\nC D\nV:2\nE F\nV:1\nG [V:2] A"), [60, 62, 67]);
    }

    #[test]
    fn test_malformed_abc_is_rejected_with_the_line_number() {
        let cases = [
            ("C\n\"Dm9 C", "ABC line 2: unterminated quoted chord symbol"),
            ("K:H\nC", "ABC line 1: unrecognised key signature token 'H'"),
            ("K:G#\nC", "more than 7 sharps or flats"),
            ("K:C foo\nC", "unrecognised key signature token 'foo'"),
            ("K:Cxyz\nC", "unknown key mode 'xyz'"),
            ("L:0/8\nC", "invalid unit length"),
            ("M:4/x\nC", "invalid meter"),
            ("C0", "note length cannot be zero"),
            ("C/0", "divide by zero"),
            ("C 3", "length '3' does not follow a note"),
            ("[CE", "unterminated chord"),
            ("[]", "empty chord"),
            ("(3C D", "tuplet (3 expects 3 notes, found 2"),
            ("(1C", "tuplet (1 is not supported"),
            (">C", "broken rhythm needs a note on each side"),
            ("C>>>>D", "longer than three"),
            ("-C", "tie '-' must follow a note"),
            ("^/C", "microtonal"),
            ("^ C", "accidental must be followed by a note letter"),
            ("M:none\nZ", "multi-bar rest needs a meter"),
            ("c''''''", "outside the MIDI range"),
            ("C & E", "voice overlay"),
            ("!trill C", "unterminated decoration"),
            ("{g C", "unterminated grace notes"),
            ("C99999", "larger than 9999"),
        ];
        for (abc, needle) in cases {
            let err = parse_abc_score(abc, TPB).expect_err(abc).to_string();
            assert!(err.contains(needle), "{abc:?}: {err}");
        }
    }

    proptest! {
        /// Arbitrary text, in many short lines so one- and two-byte lines come up often.
        #[test]
        fn prop_abc_parser_never_panics(abc in "(.{0,6}\n){0,16}.{0,40}", tpb in any::<u32>()) {
            let _ = parse_abc_score(&abc, tpb);
        }

        #[test]
        fn prop_abc_parser_never_panics_on_abc_like_text(
            abc in r#"([A-Ga-gzxZX^_=,'/0-9 |:\[\]()<>\-"!+{}%&.~]|\n|K:[A-G][#b]?m?\n|L:1/[1-9]\n|M:[1-9]/[1-9]\n){0,80}"#,
        ) {
            let _ = parse_abc_score(&abc, TPB);
        }

        /// Note starts are the running sum of the written lengths.
        #[test]
        fn prop_note_lengths_accumulate(
            notes in prop::collection::vec(
                (0usize..7, 1u32..=16, prop::sample::select(vec![1u32, 2, 4])),
                1..40,
            ),
        ) {
            let mut abc = String::from("L:1/8\nK:C\n");
            let mut expected = Vec::new();
            let mut tick = 0_u32;
            for &(letter, num, den) in &notes {
                abc.push(char::from(b"CDEFGAB"[letter]));
                abc.push_str(&format!("{num}/{den} "));
                expected.push(tick);
                tick += 240 * num / den;
            }
            prop_assert_eq!(starts(&abc), expected);
        }
    }
}
