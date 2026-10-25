//! Information fields: `K:` keys and modes, `L:` unit lengths, `M:` meters and `V:` voices.

use std::iter::Peekable;

use super::body::{letter_index, parse_accidental};
use super::{Context, Lexer, MAX_NUMBER, Parse};

/// Letter indices (`C = 0`) in the order sharps and flats are added to a key signature.
const SHARP_ORDER: [usize; 7] = [3, 0, 4, 1, 5, 2, 6];
const FLAT_ORDER: [usize; 7] = [6, 2, 5, 1, 4, 0, 3];
/// Clef and voice words that may follow the key in a `K:` field.
const CLEF_WORDS: [&str; 12] = [
    "treble", "bass", "bass3", "baritone", "tenor", "alto", "alto1", "alto2", "mezzo", "soprano",
    "perc", "none",
];

/// A whole-line information field such as `K:G` or `w:lyrics`.
pub(super) fn field_line(text: &str) -> Option<(u8, &str)> {
    let bytes = text.as_bytes();
    let is_field = bytes.len() >= 2
        && (bytes[0].is_ascii_alphabetic() || bytes[0] == b'+')
        && bytes[1] == b':';
    is_field.then(|| (bytes[0], &text[2..]))
}

pub(super) fn apply_field(ctx: &mut Context, letter: u8, value: &str) -> Parse<()> {
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
