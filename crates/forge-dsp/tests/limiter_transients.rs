//! Short overs through both limiters: single-sample spikes and 8-sample bursts at +6 dB and
//! +12 dB over the ceiling must come out limited, not clipped, with the output's true peak (as
//! `compute_true_peak` measures it) at the ceiling.

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, FRAC_PI_8, PI};

use forge_core::AudioBuffer;
use forge_dsp::{RealtimeMasterLimiter, audit_audio, compute_true_peak, normalize_and_limit};

const CEILING_DBTP: f64 = -1.0;
/// How far over the ceiling the measured output true peak may read.
const TP_TOLERANCE_DB: f64 = 0.1;
const OVERS_DB: [f64; 2] = [6.0, 12.0];
const SAMPLE_RATES: [u32; 3] = [44_100, 48_000, 96_000];
/// Spacing between overs, longer than the lookahead plus the release tail that matters.
const SPACING: usize = 6_000;
const OVERS_PER_SIGNAL: usize = 6;

#[derive(Clone, Copy, Debug)]
enum Over {
    /// One sample at the over level.
    Spike,
    /// Eight samples at the over level.
    Burst,
    /// Eight samples of a quarter-sample-rate sine phased so every sample sits at 0.707 of its
    /// peak. The over lives between samples, so only a true-peak detector sees it.
    InterSampleBurst,
    /// The same sine phased so its peaks fall a quarter of a sample after a sample, where a
    /// detector that only interpolates the midpoint under-reads.
    QuarterPhaseBurst,
}

const OVERS: [Over; 4] = [
    Over::Spike,
    Over::Burst,
    Over::InterSampleBurst,
    Over::QuarterPhaseBurst,
];

fn db_to_lin(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

/// A quiet 200 Hz bed with `OVERS_PER_SIGNAL` overs of `kind`, alternating in sign.
fn signal(kind: Over, over_db: f64, sample_rate: u32) -> Vec<f64> {
    let sr = f64::from(sample_rate);
    let level = db_to_lin(CEILING_DBTP + over_db);
    let mut x: Vec<f64> = (0..SPACING * (OVERS_PER_SIGNAL + 1))
        .map(|i| 0.1 * (2.0 * PI * 200.0 * i as f64 / sr).sin())
        .collect();
    for n in 0..OVERS_PER_SIGNAL {
        let start = SPACING * (n + 1);
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        let len = if matches!(kind, Over::Spike) { 1 } else { 8 };
        for k in 0..len {
            x[start + k] = sign
                * match kind {
                    Over::Spike | Over::Burst => level,
                    Over::InterSampleBurst => level * (FRAC_PI_2 * k as f64 + FRAC_PI_4).sin(),
                    Over::QuarterPhaseBurst => {
                        level * (FRAC_PI_2 * k as f64 + 3.0 * FRAC_PI_8).sin()
                    }
                };
        }
    }
    x
}

fn stereo(x: &[f64], sample_rate: u32) -> AudioBuffer {
    AudioBuffer::from_channels(vec![x.to_vec(), x.to_vec()], sample_rate).expect("valid buffer")
}

/// Runs `x` through a fresh realtime limiter and returns the latency-aligned output.
fn realtime_limit(x: &[f64], sample_rate: u32) -> Vec<f64> {
    let mut limiter = RealtimeMasterLimiter::new(f64::from(sample_rate));
    let latency = limiter.latency_samples() as usize;
    let mut out = Vec::with_capacity(x.len() + latency);
    for &s in x.iter().chain(std::iter::repeat_n(&0.0, latency)) {
        let s = s as f32;
        let (l, _) = limiter.process_stereo_sample(s, s, 0.0, CEILING_DBTP, 85.0);
        out.push(f64::from(l));
    }
    out.split_off(latency)
}

#[test]
fn realtime_limiter_limits_short_overs_without_clipping() {
    // The limiter's final safety clamp sits exactly at the ceiling: a sample there was clipped.
    let clamp = db_to_lin(CEILING_DBTP) as f32;
    for sr in SAMPLE_RATES {
        for kind in OVERS {
            for over_db in OVERS_DB {
                let out = realtime_limit(&signal(kind, over_db, sr), sr);
                let clipped = out.iter().filter(|&&y| (y as f32).abs() >= clamp).count();
                let (_, tp_db, at) = compute_true_peak(&stereo(&out, sr));
                assert_eq!(
                    clipped, 0,
                    "{kind:?} +{over_db} dB @ {sr}: {clipped} samples hard-clipped at the clamp"
                );
                assert!(
                    tp_db <= CEILING_DBTP + TP_TOLERANCE_DB,
                    "{kind:?} +{over_db} dB @ {sr}: output true peak {tp_db:.3} dBTP at {at:.4} s"
                );
            }
        }
    }
}

/// Masters `x` with the target set to its own loudness, so the makeup gain is zero and the
/// overs reach the limiter at their designed level.
fn offline_limit(x: &[f64], sample_rate: u32) -> Vec<f64> {
    let input = stereo(x, sample_rate);
    let lufs = audit_audio(&input).integrated_lufs;
    let out = normalize_and_limit(&input, lufs, CEILING_DBTP).expect("limits");
    out.channels[0].clone()
}

#[test]
fn offline_limiter_limits_short_overs_to_the_ceiling() {
    for sr in SAMPLE_RATES {
        for kind in OVERS {
            for over_db in OVERS_DB {
                let x = signal(kind, over_db, sr);
                let out = offline_limit(&x, sr);
                let (_, tp_db, at) = compute_true_peak(&stereo(&out, sr));
                assert!(
                    tp_db <= CEILING_DBTP + TP_TOLERANCE_DB,
                    "{kind:?} +{over_db} dB @ {sr}: output true peak {tp_db:.3} dBTP at {at:.4} s"
                );
                // The bed between overs keeps its level: the overs were limited locally rather
                // than pulled down by the whole-file safety trim.
                let bed_in = x[SPACING / 2..SPACING]
                    .iter()
                    .fold(0.0_f64, |m, s| m.max(s.abs()));
                let bed_out = out[SPACING / 2..SPACING]
                    .iter()
                    .fold(0.0_f64, |m, s| m.max(s.abs()));
                let bed_change_db = 20.0 * (bed_out / bed_in).log10();
                assert!(
                    bed_change_db.abs() < 0.5,
                    "{kind:?} +{over_db} dB @ {sr}: bed level moved {bed_change_db:.2} dB"
                );
            }
        }
    }
}

#[test]
fn offline_limiter_ramps_its_attack() {
    for sr in SAMPLE_RATES {
        for over_db in OVERS_DB {
            let x = signal(Over::Spike, over_db, sr);
            let out = offline_limit(&x, sr);
            // Where the input is well away from zero, out / in is the applied gain (times a
            // constant makeup and trim). Normalize by its maximum and find the largest
            // sample-to-sample change.
            let gains: Vec<Option<f64>> = x
                .iter()
                .zip(&out)
                .map(|(&i, &o)| (i.abs() > 0.05).then(|| o / i))
                .collect();
            let max_gain = gains.iter().flatten().fold(0.0_f64, |m, &g| m.max(g));
            let max_step = gains
                .windows(2)
                .filter_map(|w| Some((w[1]? - w[0]?).abs() / max_gain))
                .fold(0.0_f64, f64::max);
            assert!(
                max_step < 0.02,
                "spike +{over_db} dB @ {sr}: gain jumps {:.1}% in one sample",
                max_step * 100.0
            );
        }
    }
}

#[test]
fn offline_limiter_handles_an_over_at_the_file_start() {
    for sr in SAMPLE_RATES {
        for over_db in OVERS_DB {
            // The bed alone, with one spike on the very first frame.
            let mut x: Vec<f64> = (0..3 * SPACING)
                .map(|i| 0.1 * (2.0 * PI * 200.0 * i as f64 / f64::from(sr)).sin())
                .collect();
            x[0] = db_to_lin(CEILING_DBTP + over_db);
            let out = offline_limit(&x, sr);
            let (_, tp_db, at) = compute_true_peak(&stereo(&out, sr));
            assert!(
                tp_db <= CEILING_DBTP + TP_TOLERANCE_DB,
                "start spike +{over_db} dB @ {sr}: output true peak {tp_db:.3} dBTP at {at:.4} s"
            );
            // Well past the release, the bed keeps its level: the spike was limited where it
            // occurred, not by attenuating the whole file.
            let bed = |v: &[f64]| v[2 * SPACING..].iter().fold(0.0_f64, |m, s| m.max(s.abs()));
            let bed_change_db = 20.0 * (bed(&out) / bed(&x)).log10();
            assert!(
                bed_change_db.abs() < 0.5,
                "start spike +{over_db} dB @ {sr}: bed level moved {bed_change_db:.2} dB"
            );
        }
    }
}
