//! Structured chord-quality grammar for symbols outside the common-quality table.
//!
//! A quality reads as `[family] [extension] [modifier...]`, for example `maj` + `7` + `#11`,
//! `m` + `9` + `b5`, or no family + `13` + `b9`. Parentheses and commas group modifiers
//! (`7(b9,#11)`). Anything the grammar does not recognise is an error rather than a guess.

/// Third, fifth and seventh implied by the family prefix (`m`, `maj`, `dim`, ...).
#[derive(Debug, Clone, Copy)]
struct Family {
    third: u8,
    fifth: u8,
    seventh: u8,
    /// Half-diminished chords always carry their seventh, even without a number.
    forces_seventh: bool,
}

const DOMINANT: Family = Family {
    third: 4,
    fifth: 7,
    seventh: 10,
    forces_seventh: false,
};

/// Family prefixes, longest first so `mMaj` wins over `m` and `maj` wins over `m`.
const FAMILIES: [(&str, Family); 19] = [
    ("mMaj", minor_major()),
    ("mmaj", minor_major()),
    ("minmaj", minor_major()),
    ("mM", minor_major()),
    ("-maj", minor_major()),
    ("mΔ", minor_major()),
    ("maj", major_seventh()),
    ("Maj", major_seventh()),
    ("M", major_seventh()),
    ("Δ", major_seventh()),
    ("min", minor()),
    ("m", minor()),
    ("-", minor()),
    ("dim", diminished()),
    ("o", diminished()),
    ("°", diminished()),
    ("ø", half_diminished()),
    ("aug", augmented()),
    ("+", augmented()),
];

/// Prefixes that turn an augmented triad into an augmented major seventh (`+maj7`).
const MAJOR_SEVENTH_PREFIXES: [(&str, ()); 4] = [("maj", ()), ("Maj", ()), ("M", ()), ("Δ", ())];

const fn minor() -> Family {
    Family {
        third: 3,
        ..DOMINANT
    }
}

const fn minor_major() -> Family {
    Family {
        third: 3,
        seventh: 11,
        ..DOMINANT
    }
}

const fn major_seventh() -> Family {
    Family {
        seventh: 11,
        ..DOMINANT
    }
}

const fn diminished() -> Family {
    Family {
        third: 3,
        fifth: 6,
        seventh: 9,
        forces_seventh: false,
    }
}

const fn half_diminished() -> Family {
    Family {
        third: 3,
        fifth: 6,
        seventh: 10,
        forces_seventh: true,
    }
}

const fn augmented() -> Family {
    Family {
        fifth: 8,
        ..DOMINANT
    }
}

/// One alteration, addition or suspension after the extension number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Modifier {
    Fifth(u8),
    Ninth(u8),
    Eleventh(u8),
    Thirteenth(u8),
    Add(u8),
    Sus(u8),
}

/// Modifier spellings, longest first so `add13` wins over `add1...` and `sus4` over `sus`.
const MODIFIERS: [(&str, Modifier); 21] = [
    ("add13", Modifier::Add(21)),
    ("add11", Modifier::Add(17)),
    ("add9", Modifier::Add(14)),
    ("add6", Modifier::Add(9)),
    ("add4", Modifier::Add(5)),
    ("add2", Modifier::Add(2)),
    ("sus4", Modifier::Sus(5)),
    ("sus2", Modifier::Sus(2)),
    ("sus", Modifier::Sus(5)),
    ("b13", Modifier::Thirteenth(20)),
    ("-13", Modifier::Thirteenth(20)),
    ("#11", Modifier::Eleventh(18)),
    ("+11", Modifier::Eleventh(18)),
    ("b9", Modifier::Ninth(13)),
    ("-9", Modifier::Ninth(13)),
    ("#9", Modifier::Ninth(15)),
    ("+9", Modifier::Ninth(15)),
    ("b5", Modifier::Fifth(6)),
    ("-5", Modifier::Fifth(6)),
    ("#5", Modifier::Fifth(8)),
    ("+5", Modifier::Fifth(8)),
];

/// Extension numbers, longest first so `13` wins over `1...` and `69` over `6`.
const EXTENSIONS: [(&str, u8); 7] = [
    ("13", 13),
    ("11", 11),
    ("69", 69),
    ("9", 9),
    ("7", 7),
    ("6", 6),
    ("5", 5),
];

/// Parses a chord quality (the part after the root and before any slash bass) into
/// semitone intervals above the root, sorted and without duplicates.
pub(crate) fn parse_quality(quality: &str) -> Result<Vec<u8>, String> {
    let flat = strip_groups(quality)?;
    let (family, after_family) = split_family(&flat);
    let (extension, after_extension) = split_prefix(after_family, &EXTENSIONS);
    let modifiers = parse_modifiers(after_extension)?;
    build_intervals(family, extension, &modifiers)
}

/// Removes `(`, `)` and `,` after checking the parentheses are balanced and not nested.
fn strip_groups(quality: &str) -> Result<String, String> {
    let mut depth = 0_u8;
    let mut out = String::with_capacity(quality.len());
    for c in quality.chars() {
        match c {
            '(' if depth == 0 => depth = 1,
            '(' => return Err("nested parentheses".to_string()),
            ')' if depth == 1 => depth = 0,
            ')' => return Err("unbalanced ')'".to_string()),
            ',' if depth == 1 => {}
            ',' => return Err("',' outside parentheses".to_string()),
            _ => out.push(c),
        }
    }
    if depth != 0 {
        return Err("unclosed '('".to_string());
    }
    Ok(out)
}

fn split_family(quality: &str) -> (Family, &str) {
    let (found, rest) = split_prefix(quality, &FAMILIES);
    let family = found.unwrap_or(DOMINANT);
    // `+maj7` / `augmaj7`: augmented triad with a major seventh.
    if found.is_some() && family.fifth == 8 {
        let (major, rest_major) = split_prefix(rest, &MAJOR_SEVENTH_PREFIXES);
        if major.is_some() {
            return (
                Family {
                    seventh: 11,
                    ..family
                },
                rest_major,
            );
        }
    }
    (family, rest)
}

fn split_prefix<'a, T: Copy>(text: &'a str, table: &[(&str, T)]) -> (Option<T>, &'a str) {
    table
        .iter()
        .find_map(|&(prefix, value)| text.strip_prefix(prefix).map(|rest| (Some(value), rest)))
        .unwrap_or((None, text))
}

fn parse_modifiers(mut rest: &str) -> Result<Vec<Modifier>, String> {
    let mut modifiers = Vec::new();
    while !rest.is_empty() {
        match split_prefix(rest, &MODIFIERS) {
            (Some(modifier), tail) => {
                modifiers.push(modifier);
                rest = tail;
            }
            (None, _) => return Err(format!("unrecognised '{rest}'")),
        }
    }
    Ok(modifiers)
}

fn build_intervals(
    family: Family,
    extension: Option<u8>,
    modifiers: &[Modifier],
) -> Result<Vec<u8>, String> {
    if extension == Some(5) {
        return power_chord(family, modifiers);
    }
    let mut third = family.third;
    let mut fifths = vec![family.fifth];
    let mut upper = extension_tones(extension);
    let mut fifth_altered = false;
    for &modifier in modifiers {
        match modifier {
            Modifier::Fifth(iv) => {
                if !fifth_altered {
                    fifths.clear();
                    fifth_altered = true;
                }
                fifths.push(iv);
            }
            Modifier::Sus(iv) if family.third == 4 => third = iv,
            Modifier::Sus(_) => return Err("sus cannot follow a minor third".to_string()),
            Modifier::Ninth(iv) => replace_tone(&mut upper, 14, iv),
            Modifier::Eleventh(iv) => replace_tone(&mut upper, 17, iv),
            Modifier::Thirteenth(iv) => replace_tone(&mut upper, 21, iv),
            Modifier::Add(iv) => upper.push(iv),
        }
    }
    let has_seventh = family.forces_seventh || matches!(extension, Some(7 | 9 | 11 | 13));
    let mut intervals = vec![0, third];
    intervals.extend(fifths);
    if has_seventh {
        intervals.push(family.seventh);
    }
    intervals.extend(upper);
    intervals.sort_unstable();
    intervals.dedup();
    Ok(intervals)
}

/// `C5`: root and fifth only, which takes no quality or modifier.
fn power_chord(family: Family, modifiers: &[Modifier]) -> Result<Vec<u8>, String> {
    if family.third != DOMINANT.third || family.fifth != DOMINANT.fifth || !modifiers.is_empty() {
        return Err("a power chord (5) takes no quality or modifier".to_string());
    }
    Ok(vec![0, 7])
}

/// Upper tones (above the seventh) implied by the extension number.
fn extension_tones(extension: Option<u8>) -> Vec<u8> {
    match extension {
        Some(6) => vec![9],
        Some(69) => vec![9, 14],
        Some(9) => vec![14],
        Some(11) => vec![14, 17],
        Some(13) => vec![14, 21],
        _ => Vec::new(),
    }
}

/// An alteration replaces the natural tone it modifies (`b9` drops the natural `9`).
fn replace_tone(upper: &mut Vec<u8>, natural: u8, altered: u8) {
    upper.retain(|&iv| iv != natural);
    upper.push(altered);
}
