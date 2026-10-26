//! Calibration and band-placement tests for the 32-band display spectrum (THE-1596).
//!
//! Levels are dB relative to a full-scale sine, so a sine of amplitude `A` centred in a band
//! must read `20 * log10(A)` there, at 44.1, 48 and 96 kHz alike.

use std::f64::consts::PI;

use forge_dsp::spectrum::stft_band_bins;
use forge_dsp::{
    RealtimeOpenSoothe, RealtimeSpectrumAnalyzer32, SPECTRUM_BANDS, spectrum_band_center_hz,
    spectrum_band_edges_hz,
};

const RATES: [f64; 3] = [44_100.0, 48_000.0, 96_000.0];

fn amp_db(amp: f64) -> f32 {
    (20.0 * amp.log10()) as f32
}

/// Runs `secs` of a sine through a fresh filter-bank analyzer and returns its snapshot.
fn filter_bank_sine(sr: f64, freq: f64, amp: f64, secs: f64) -> [f32; SPECTRUM_BANDS] {
    let mut an = RealtimeSpectrumAnalyzer32::new(sr);
    let n = (secs * sr) as usize;
    for i in 0..n {
        let s = (amp * (2.0 * PI * freq * i as f64 / sr).sin()) as f32;
        an.push_stereo_sample(s, s);
    }
    an.snapshot_db()
}

fn loudest_band(snap: &[f32; SPECTRUM_BANDS]) -> usize {
    (0..SPECTRUM_BANDS)
        .max_by(|&a, &b| snap[a].total_cmp(&snap[b]))
        .unwrap_or(0)
}

fn band_containing(freq: f64) -> usize {
    (0..SPECTRUM_BANDS)
        .find(|&b| {
            let (lo, hi) = spectrum_band_edges_hz(b);
            (lo..hi).contains(&freq)
        })
        .expect("frequency inside 25 Hz .. 18.5 kHz")
}

#[test]
fn band_layout_spans_25_hz_to_18_5_khz() {
    let (lo, _) = spectrum_band_edges_hz(0);
    let (_, hi) = spectrum_band_edges_hz(SPECTRUM_BANDS - 1);
    assert!((lo - 25.0).abs() < 1e-9 && (hi - 18_500.0).abs() < 1e-6);
    for b in 1..SPECTRUM_BANDS {
        assert!((spectrum_band_edges_hz(b).0 - spectrum_band_edges_hz(b - 1).1).abs() < 1e-9);
    }
}

#[test]
fn filter_bank_puts_a_1_khz_sine_in_the_1_khz_band() {
    let expected = band_containing(1_000.0);
    for sr in RATES {
        let snap = filter_bank_sine(sr, 1_000.0, 0.5, 0.5);
        let top = loudest_band(&snap);
        assert_eq!(
            top, expected,
            "{sr} Hz: 1 kHz sine peaks in band {top}: {snap:?}"
        );
        // 1 kHz sits near the band's upper edge, where the band-pass is down by up to 3 dB.
        let err = snap[top] - amp_db(0.5);
        assert!(
            (-3.1..0.5).contains(&err),
            "{sr} Hz: 1 kHz at -6 dBFS reads {} dB",
            snap[top]
        );
    }
}

#[test]
fn filter_bank_reads_sines_at_band_centres_at_their_level() {
    for sr in RATES {
        for band in [0, 4, 9, 15, 20, 26, 31] {
            let fc = spectrum_band_center_hz(band);
            for amp in [1.0, 0.1] {
                let snap = filter_bank_sine(sr, fc, amp, 1.5);
                let err = snap[band] - amp_db(amp);
                assert!(
                    err.abs() < 0.5,
                    "{sr} Hz band {band} ({fc:.0} Hz) at {:.1} dBFS reads {} dB",
                    amp_db(amp),
                    snap[band]
                );
                assert_eq!(loudest_band(&snap), band, "{sr} Hz band {band}: {snap:?}");
            }
        }
    }
}

#[test]
fn filter_bank_release_is_time_based_at_every_rate() {
    for sr in RATES {
        let mut an = RealtimeSpectrumAnalyzer32::new(sr);
        let fc = spectrum_band_center_hz(20);
        for i in 0..(sr as usize) {
            let s = (2.0 * PI * fc * i as f64 / sr).sin() as f32;
            an.push_stereo_sample(s, s);
        }
        // 0.3 s of silence is one release time constant: about 4.3 dB of fall.
        for _ in 0..((0.3 * sr) as usize) {
            an.push_stereo_sample(0.0, 0.0);
        }
        let level = an.snapshot_db()[20];
        assert!(
            (-5.5..=-3.0).contains(&level),
            "{sr} Hz: after 0.3 s, {level} dB"
        );
    }
}

/// Bands whose lower edge the edge tests probe: low, middle and every one of the top bands,
/// where an unwarped bilinear band-pass is narrowest at 44.1 and 48 kHz.
const EDGE_BANDS: [usize; 8] = [1, 8, 16, 24, 28, 29, 30, 31];

/// Level of an ideal constant-Q band-pass with -3 dB points on `band`'s edges at `freq`, in
/// dB: the shape the band layout advertises, independent of the sample rate.
fn ideal_band_db(band: usize, freq: f64) -> f32 {
    let (lo, hi) = spectrum_band_edges_hz(band);
    let fc = (lo * hi).sqrt();
    let q = fc / (hi - lo);
    let x = q * (freq / fc - fc / freq);
    (-10.0 * (1.0 + x * x).log10()) as f32
}

#[test]
fn filter_bank_band_edges_are_3_db_down_in_both_neighbours_at_every_rate() {
    for sr in RATES {
        for band in EDGE_BANDS {
            // The edge is also the geometric midpoint between the two band centres.
            let edge = spectrum_band_edges_hz(band).0;
            let snap = filter_bank_sine(sr, edge, 1.0, 1.0);
            for b in [band - 1, band] {
                assert!(
                    (snap[b] + 3.01).abs() < 0.5,
                    "{sr} Hz: the {edge:.0} Hz edge reads {} dB in band {b}, not -3 dB",
                    snap[b]
                );
            }
        }
    }
}

#[test]
fn filter_bank_follows_the_band_shape_halfway_between_centre_and_edge() {
    for sr in RATES {
        for band in [0, 15, 28, 29, 30, 31] {
            let (lo, hi) = spectrum_band_edges_hz(band);
            let fc = spectrum_band_center_hz(band);
            for freq in [(lo * fc).sqrt(), (fc * hi).sqrt()] {
                let snap = filter_bank_sine(sr, freq, 1.0, 1.0);
                // About -1 dB for the ideal shape. Near Nyquist the prewarped band is flatter
                // in Hz above its centre, but it must stay well inside its -3 dB edges.
                let expected = ideal_band_db(band, freq);
                assert!(
                    (expected - 0.5..-0.25).contains(&snap[band]),
                    "{sr} Hz band {band}: {freq:.0} Hz reads {} dB, ideal {expected:.2} dB",
                    snap[band]
                );
            }
        }
    }
}

#[test]
fn filter_bank_integrates_a_steady_quiet_sine_to_its_level() {
    // -84 dBFS sits above the -90 dB display floor, but its per-sample power increments at
    // 96 kHz are far below the old -120 dB state floor.
    let amp = 10.0_f64.powf(-84.0 / 20.0);
    let fc = spectrum_band_center_hz(0);
    for sr in [48_000.0, 96_000.0] {
        let snap = filter_bank_sine(sr, fc, amp, 2.0);
        assert!(
            (snap[0] + 84.0).abs() < 1.0,
            "{sr} Hz: a -84 dBFS sine at {fc:.1} Hz reads {} dB in band 0",
            snap[0]
        );
    }
}

/// Feeds `n` frames to OpenSoothe and returns the hop-averaged band power of the last half.
fn open_soothe_mean_power(sr: f64, n: usize, frame: impl Fn(usize) -> (f32, f32)) -> [f64; 32] {
    let mut dsp = RealtimeOpenSoothe::new(sr);
    let mut acc = [0.0_f64; 32];
    let mut hops = 0.0;
    for i in 0..n {
        let (l, r) = frame(i);
        dsp.process_stereo_sample(l, r, 3.5, 9.0, 8, false);
        if i >= n / 2 && i % 512 == 0 {
            let snap = dsp.last_spectrum_32_db();
            for (a, &db) in acc.iter_mut().zip(snap.iter()) {
                *a += 10.0_f64.powf(f64::from(db) / 10.0);
            }
            hops += 1.0;
        }
    }
    acc.map(|p| p / hops)
}

fn power_db(p: f64) -> f32 {
    (10.0 * p.log10()) as f32
}

#[test]
fn stft_spectrum_reads_sines_at_their_level_in_their_band() {
    for sr in RATES {
        for band in [8, 14, 20, 26, 31] {
            let fc = spectrum_band_center_hz(band);
            let sine = |i: usize| {
                let s = (0.5 * (2.0 * PI * fc * i as f64 / sr).sin()) as f32;
                (s, s)
            };
            let pow = open_soothe_mean_power(sr, 16_384, sine);
            let level = power_db(pow[band]);
            assert!(
                (level - amp_db(0.5)).abs() < 1.0,
                "{sr} Hz band {band} ({fc:.0} Hz) at -6 dBFS reads {level} dB"
            );
            let top = (0..32).max_by(|&a, &b| pow[a].total_cmp(&pow[b]));
            assert_eq!(top, Some(band), "{sr} Hz band {band}");
        }
    }
}

#[test]
fn stft_spectrum_averages_both_channels() {
    let fc = spectrum_band_center_hz(20);
    let left_only = |i: usize| {
        let s = (0.5 * (2.0 * PI * fc * i as f64 / 48_000.0).sin()) as f32;
        (s, 0.0)
    };
    let pow = open_soothe_mean_power(48_000.0, 16_384, left_only);
    // Half the power of the same sine on both channels: -6 dB - 3 dB.
    let level = power_db(pow[20]);
    assert!((level - (amp_db(0.5) - 3.01)).abs() < 1.0, "{level} dB");
}

#[test]
fn stft_spectrum_sums_white_noise_power_across_each_band() {
    let sr = 48_000.0;
    let mut state = 0x2545_f491_u32;
    let mut noise = Vec::with_capacity(65_536);
    for _ in 0..65_536 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        noise.push((f64::from(state) / f64::from(u32::MAX) - 0.5) as f32 * 0.5);
    }
    // Uniform on [-0.25, 0.25]: variance 0.25^2 / 3.
    let var = 0.0625 / 3.0;
    let pow = open_soothe_mean_power(sr, noise.len(), |i| (noise[i], noise[i]));
    for (band, &band_pow) in pow.iter().enumerate().skip(4) {
        let (b0, b1) = stft_band_bins(band, sr / 2048.0, 1025);
        let bins = (b1 - b0 + 1) as f64;
        // Each bin carries 4 * var / N of sine-referenced power.
        let expected = power_db(4.0 * var * bins / 2048.0);
        let level = power_db(band_pow);
        assert!(
            (level - expected).abs() < 1.0,
            "band {band} ({bins} bins): {level} dB, expected {expected} dB"
        );
    }
}
