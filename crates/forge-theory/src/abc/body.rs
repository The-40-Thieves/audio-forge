//! Body tokenizer: notes, rests, chords, ties, tuplets, broken rhythm, bar lines and the
//! decorations, annotations and grace notes that are skipped.

use super::fields::{apply_field, field_line};
use super::{Context, Element, Lexer, Parse};

/// Semitones above C for the letters `C D E F G A B`.
const LETTER_SEMIS: [i32; 7] = [0, 2, 4, 5, 7, 9, 11];
/// Body bytes that carry no pitch or timing: spacing (space, tab, backquote and `y`), line
/// continuation `\`, the slur end `)`, and the one-letter decorations `.~HLMOPSTuv`.
/// Any other unrecognised character is an error.
const SKIPPED: &[u8] = b" \t`y\\).~HLMOPSTuv";

pub(super) fn parse_body_line(
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
        Some(byte) if SKIPPED.contains(&byte) => lx.bump(),
        _ => return Err(unexpected_character(lx)),
    }
    Ok(())
}

/// Names the character at the cursor, which may start a multi-byte UTF-8 sequence.
fn unexpected_character(lx: &Lexer) -> String {
    let found = lx.text[lx.pos..].chars().next().unwrap_or(' ');
    format!("unexpected character '{found}' in the music")
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

pub(super) fn letter_index(upper: u8) -> usize {
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

pub(super) fn parse_accidental(lx: &mut Lexer) -> Parse<Option<i8>> {
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
