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
fn test_unknown_key_roots_are_rejected() {
    // A `K:` field without a tonic may still carry a clef, settings or explicit accidentals.
    let cases: &[(&str, &[u8])] = &[
        ("K:bass\nB,", &[59]),
        ("K:clef=treble\nF", &[65]),
        ("K:^f ^c\nF C", &[66, 61]),
    ];
    for &(abc, expected) in cases {
        assert_eq!(pitches(abc), expected, "{abc}");
    }
    let cases = [
        ("K:H\nC", "ABC line 1: unknown key root 'H'"),
        ("K:g\nC", "ABC line 1: unknown key root 'g'"),
        ("K:C\nC [K:Xm] D", "ABC line 2: unknown key root 'Xm'"),
    ];
    for (abc, needle) in cases {
        let err = parse_abc_score(abc, TPB).expect_err(abc).to_string();
        assert!(err.contains(needle), "{abc:?}: {err}");
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
fn test_unknown_body_characters_are_rejected() {
    // Spacing, continuation, slur ends and one-letter decorations are still skipped.
    assert_eq!(
        pitches("C `D y E \\\n.F ~G HA (B c) uC vD\tE"),
        [60, 62, 64, 65, 67, 69, 71, 72, 60, 62, 64]
    );
    let cases = [
        (
            "K:C\nC Q D",
            "ABC line 2: unexpected character 'Q' in the music",
        ),
        ("C ; D", "ABC line 1: unexpected character ';'"),
        ("C\nD é E", "ABC line 2: unexpected character 'é'"),
        ("C # D", "unexpected character '#'"),
    ];
    for (abc, needle) in cases {
        let err = parse_abc_score(abc, TPB).expect_err(abc).to_string();
        assert!(err.contains(needle), "{abc:?}: {err}");
    }
}

#[test]
fn test_malformed_abc_is_rejected_with_the_line_number() {
    let cases = [
        ("C\n\"Dm9 C", "ABC line 2: unterminated quoted chord symbol"),
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
