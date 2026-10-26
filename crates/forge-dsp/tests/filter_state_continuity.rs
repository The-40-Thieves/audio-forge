//! Filter state across parameter changes: retuning a running filter, or re-engaging a bypassed
//! branch, must not throw away state and step the output.
//!
//! Each test feeds a constant-amplitude low-frequency sine and changes a parameter at block
//! boundaries. The largest sample-to-sample step of the output is compared against the largest
//! step the same processor produces with the parameter held still, which is what a sine of that
//! amplitude and frequency moves by on its own. A click shows up as a step well above it.

use std::f64::consts::PI;

use forge_dsp::{AllpassDisperser, RealtimeMasteringEq18, RealtimeSubAndTapeBus};

const SR: f64 = 48_000.0;
/// Host-style block size; parameters change only at block boundaries.
const BLOCK: usize = 64;
/// How far above the held-parameter reference a step may go before it counts as a click.
const STEP_HEADROOM: f64 = 1.5;

fn sine(i: usize, freq_hz: f64, amp: f64) -> f64 {
    amp * (2.0 * PI * freq_hz * i as f64 / SR).sin()
}

/// Largest absolute difference between consecutive samples.
fn max_step(samples: &[f64]) -> f64 {
    samples
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0, f64::max)
}

/// Triangle wave in `[0, 1]` with a period of `period` blocks.
fn triangle(block: usize, period: usize) -> f64 {
    let phase = (block % period) as f64 / period as f64;
    1.0 - (2.0 * phase - 1.0).abs()
}

/// A pure side signal: everything the mono-maker's side high-passes see.
fn side_input(i: usize) -> (f32, f32) {
    let s = sine(i, 200.0, 0.5) as f32;
    (s, -s)
}

/// Runs a Sub & Tape bus built with `start_cutoff` over `blocks` blocks of the side sine,
/// calling `cutoff_for(block)` at the start of each block, and returns the left output.
fn run_sub_tape_cutoff(
    blocks: usize,
    start_cutoff: f64,
    cutoff_for: impl Fn(usize) -> Option<f64>,
) -> Vec<f64> {
    let mut bus = RealtimeSubAndTapeBus::new(SR, start_cutoff);
    let mut out = Vec::with_capacity(blocks * BLOCK);
    for block in 0..blocks {
        if let Some(cutoff) = cutoff_for(block) {
            bus.set_mono_cutoff(cutoff);
        }
        for n in 0..BLOCK {
            let (l, r) = side_input(block * BLOCK + n);
            out.push(f64::from(bus.process_stereo_sample(l, r, 0.0, 1.0, 0.0).0));
        }
    }
    out
}

/// H5: the plugin sets the mono cutoff every block. With an unchanged value that must leave the
/// output untouched, bit for bit, instead of rebuilding the side high-passes.
#[test]
fn unchanged_mono_cutoff_every_block_is_a_no_op() {
    let untouched = run_sub_tape_cutoff(256, 160.0, |_| None);
    let reset_each_block = run_sub_tape_cutoff(256, 160.0, |_| Some(160.0));
    for (i, (a, b)) in untouched.iter().zip(&reset_each_block).enumerate() {
        assert!(
            a.to_bits() == b.to_bits(),
            "sample {i}: setting the same cutoff every block changed the output ({a} vs {b})"
        );
    }
}

/// H5: sweeping the mono cutoff block by block must glide, not step.
#[test]
fn mono_cutoff_sweep_has_no_discontinuities() {
    // The references are built at their cutoff and never retuned.
    let held_low = max_step(&run_sub_tape_cutoff(256, 50.0, |_| None));
    let held_high = max_step(&run_sub_tape_cutoff(256, 300.0, |_| None));
    let reference = held_low.max(held_high);
    let swept = run_sub_tape_cutoff(512, 50.0, |b| Some(50.0 + 250.0 * triangle(b, 40)));
    let step = max_step(&swept);
    assert!(
        step <= STEP_HEADROOM * reference,
        "cutoff sweep stepped by {step:.5}, held-cutoff reference {reference:.5}"
    );
}

const EQ_GAINS_DB: [f64; 18] = [
    4.0, -4.0, 4.0, -4.0, 4.0, -4.0, 4.0, -4.0, 4.0, -4.0, 4.0, -4.0, 4.0, -4.0, 4.0, -4.0, 4.0,
    -4.0,
];
const EQ_ALT_GAINS_DB: [f64; 18] = [
    -3.0, 3.0, -3.0, 3.0, -3.0, 3.0, -3.0, 3.0, -3.0, 3.0, -3.0, 3.0, -3.0, 3.0, -3.0, 3.0, -3.0,
    3.0,
];

/// Runs the 18-band EQ over `blocks` blocks of a 250 Hz sine, calling `gains_for(block)` at the
/// start of each block, and returns the left output.
fn run_eq(blocks: usize, gains_for: impl Fn(usize) -> ([f64; 18], f64)) -> Vec<f64> {
    let mut eq = RealtimeMasteringEq18::new(SR);
    let mut out = Vec::with_capacity(blocks * BLOCK);
    for block in 0..blocks {
        let (gains, amount) = gains_for(block);
        eq.set_gains(&gains, amount);
        for n in 0..BLOCK {
            let s = sine(block * BLOCK + n, 250.0, 0.3) as f32;
            out.push(f64::from(eq.process_stereo_sample(s, s).0));
        }
    }
    out
}

/// Held-gain reference for the EQ: the larger step of the flat and the strongest setting, after
/// the glide from flat has settled.
fn eq_reference() -> f64 {
    let settle = 64 * BLOCK;
    let flat = run_eq(128, |_| (EQ_GAINS_DB, 0.0));
    let strong = run_eq(128, |_| (EQ_GAINS_DB, 1.5));
    let alt = run_eq(128, |_| (EQ_ALT_GAINS_DB, 1.5));
    max_step(&flat[settle..])
        .max(max_step(&strong[settle..]))
        .max(max_step(&alt[settle..]))
}

/// M2: automating the EQ amount block by block must not click.
#[test]
fn eq_amount_automation_has_no_discontinuities() {
    let reference = eq_reference();
    let swept = run_eq(512, |b| (EQ_GAINS_DB, 1.5 * triangle(b, 48)));
    let step = max_step(&swept);
    assert!(
        step <= STEP_HEADROOM * reference,
        "EQ amount sweep stepped by {step:.5}, held-gain reference {reference:.5}"
    );
}

/// M2: switching the target curve (as a genre change does) must not click.
#[test]
fn eq_curve_switch_has_no_discontinuities() {
    let reference = eq_reference();
    let switched = run_eq(512, |b| {
        let curve = if (b / 64) % 2 == 0 {
            EQ_GAINS_DB
        } else {
            EQ_ALT_GAINS_DB
        };
        (curve, 1.5)
    });
    let step = max_step(&switched);
    assert!(
        step <= STEP_HEADROOM * reference,
        "EQ curve switch stepped by {step:.5}, held-gain reference {reference:.5}"
    );
}

/// Runs the Sub & Tape bus on a 100 Hz mid sine with the tape drive `drive_for(block)`.
fn run_tape(blocks: usize, drive_for: impl Fn(usize) -> f64) -> Vec<f64> {
    let mut bus = RealtimeSubAndTapeBus::new(SR, 160.0);
    let mut out = Vec::with_capacity(blocks * BLOCK);
    for block in 0..blocks {
        let drive = drive_for(block);
        for n in 0..BLOCK {
            let i = block * BLOCK + n;
            let l = (sine(i, 100.0, 0.5) + sine(i, 230.0, 0.1)) as f32;
            let r = sine(i, 100.0, 0.5) as f32;
            out.push(f64::from(
                bus.process_stereo_sample(l, r, 0.0, 1.0, drive).0,
            ));
        }
    }
    out
}

/// Drive schedule with a 177-block period: on for 100 blocks, ramped down over 15 blocks (the
/// way the plugin's 20 ms parameter smoother moves it), bypassed, then switched straight back on.
/// The period puts each toggle at a different signal phase.
fn tape_drive_schedule(block: usize) -> f64 {
    match block % 177 {
        0..=99 => 0.6,
        b @ 100..=114 => 0.6 * (115 - b) as f64 / 16.0,
        115..=161 => 0.0,
        _ => 0.6,
    }
}

/// Low severity: bypassing the tape branch and bringing it back must not click, either from the
/// state it held when it was bypassed or from the switch between dry and wet.
#[test]
fn tape_bypass_and_reengage_has_no_transient() {
    let settle = 16 * BLOCK;
    let wet = run_tape(256, |_| 0.6);
    let dry = run_tape(256, |_| 0.0);
    let reference = max_step(&wet[settle..]).max(max_step(&dry[settle..]));
    let toggled = run_tape(540, tape_drive_schedule);
    let step = max_step(&toggled[settle..]);
    assert!(
        step <= STEP_HEADROOM * reference,
        "tape bypass toggling stepped by {step:.5}, steady reference {reference:.5}"
    );
}

/// Low severity: a disperser that re-engages after bypass must start from cleared state, exactly
/// like a freshly built one, not from the state it held when it was bypassed.
#[test]
fn disperser_reengages_from_cleared_state() {
    let input = |i: usize| sine(i, 100.0, 0.5);
    let mut used = AllpassDisperser::default();
    for i in 0..4_800 {
        used.process_sample(input(i), SR, 135.0, 0.45);
    }
    for i in 4_800..9_731 {
        used.process_sample(input(i), SR, 135.0, 0.0);
    }
    let mut fresh = AllpassDisperser::default();
    for i in 9_731..12_000 {
        let a = used.process_sample(input(i), SR, 135.0, 0.45);
        let b = fresh.process_sample(input(i), SR, 135.0, 0.45);
        assert!(
            a.to_bits() == b.to_bits(),
            "sample {i}: re-engaged disperser {a} differs from a fresh one {b}"
        );
    }
}
