//! Band splits in the vocal strips must sum back to flat.
//!
//! The de-plosive stages split off a low band and duck it, and the de-esser removes part of a
//! sibilance band. With nothing being ducked, white noise through each stage must come out with a
//! flat magnitude response, and the de-esser must only ever attenuate its band, never boost or
//! invert it.

use std::f64::consts::PI;

use forge_core::AudioBuffer;
use forge_dsp::{Biquad, RealtimeVocalSurgical, clean_vocal_surgical};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

const SR: f64 = 48_000.0;
const FFT_SIZE: usize = 4096;
/// Samples skipped before analysis so filter start-up does not enter the estimate.
const WARMUP: usize = 8192;
/// Allowed deviation from a flat response, in dB.
const FLAT_TOL_DB: f64 = 0.5;

/// Deterministic white noise in `[-amp, amp]` from a 64-bit LCG.
fn white_noise(len: usize, amp: f64) -> Vec<f64> {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = (state >> 11) as f64 / (1_u64 << 53) as f64;
            amp * (2.0 * unit - 1.0)
        })
        .collect()
}

/// Welch H1 estimate `|Sxy| / Sxx` per FFT bin, from Hann-windowed frames at 50% overlap.
fn transfer_magnitude(x: &[f64], y: &[f64]) -> Vec<f64> {
    let fft = FftPlanner::<f64>::new().plan_fft_forward(FFT_SIZE);
    let window: Vec<f64> = (0..FFT_SIZE)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / FFT_SIZE as f64).cos())
        .collect();
    let bins = FFT_SIZE / 2 + 1;
    let mut sxx = vec![0.0; bins];
    let mut sxy = vec![Complex::new(0.0, 0.0); bins];
    let mut start = WARMUP;
    while start + FFT_SIZE <= x.len().min(y.len()) {
        let frame = |s: &[f64]| -> Vec<Complex<f64>> {
            (0..FFT_SIZE)
                .map(|i| Complex::new(s[start + i] * window[i], 0.0))
                .collect()
        };
        let mut fx = frame(x);
        let mut fy = frame(y);
        fft.process(&mut fx);
        fft.process(&mut fy);
        for k in 0..bins {
            sxx[k] += fx[k].norm_sqr();
            sxy[k] += fy[k] * fx[k].conj();
        }
        start += FFT_SIZE / 2;
    }
    sxy.iter()
        .zip(&sxx)
        .map(|(c, &p)| c.norm() / p.max(1e-30))
        .collect()
}

/// Panics with the worst bin if the response leaves `±FLAT_TOL_DB` between 20 Hz and 20 kHz.
fn assert_flat(label: &str, x: &[f64], y: &[f64]) {
    let mags = transfer_magnitude(x, y);
    let bin_hz = SR / FFT_SIZE as f64;
    let (worst_hz, worst_db) = mags
        .iter()
        .enumerate()
        .map(|(k, &m)| (k as f64 * bin_hz, 20.0 * m.max(1e-12).log10()))
        .filter(|&(hz, _)| (20.0..=20_000.0).contains(&hz))
        .fold((0.0_f64, 0.0_f64), |acc, b| {
            if b.1.abs() > acc.1.abs() { b } else { acc }
        });
    assert!(
        worst_db.abs() <= FLAT_TOL_DB,
        "{label}: response is {worst_db:.2} dB at {worst_hz:.0} Hz, outside ±{FLAT_TOL_DB} dB"
    );
}

#[test]
fn offline_deplosive_is_flat_without_plosives() {
    let x = white_noise(48_000 * 4, 0.1);
    let buf = AudioBuffer::from_channels(vec![x.clone()], SR as u32).expect("buffer");
    let out = clean_vocal_surgical(&buf, false, true, 0.0).expect("cleaned");
    assert_flat("clean_vocal_surgical de-plosive", &x, &out.channels[0]);
}

#[test]
fn realtime_vocal_strip_is_flat_without_ducking() {
    let x = white_noise(48_000 * 4, 0.1);
    let mut strip = RealtimeVocalSurgical::new(SR);
    // De-click off, default plosive strength (white noise never trips it), de-esser threshold at
    // 0 dBFS so it never engages, breath gate at its floor.
    let y: Vec<f64> = x
        .iter()
        .map(|&s| {
            let (l, _) = strip.process_stereo_sample(s as f32, s as f32, 0.0, 0.5, 0.0, -90.0);
            f64::from(l)
        })
        .collect();
    let x32: Vec<f64> = x.iter().map(|&s| f64::from(s as f32)).collect();
    assert_flat("RealtimeVocalSurgical", &x32, &y);
}

#[test]
fn sibilance_band_peaks_at_unity() {
    let bp = Biquad::bandpass(SR, 7200.0, 1.4);
    assert!(bp.magnitude_db_at(7200.0, SR).abs() < 0.01);
    for hz in [100.0, 1000.0, 5000.0, 7000.0, 7400.0, 10_000.0, 20_000.0] {
        assert!(
            bp.magnitude_db_at(hz, SR) <= 1e-6,
            "band-pass gain above 0 dB at {hz} Hz"
        );
    }
}

/// Signed gain of the de-esser on a steady sine: the projection of the output onto the input.
fn deesser_gain(freq_hz: f64, threshold_db: f64) -> (f64, f64) {
    let mut strip = RealtimeVocalSurgical::new(SR);
    let n = 24_000;
    let (mut xy, mut xx, mut yy) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let x = (0.5 * (2.0 * PI * freq_hz * i as f64 / SR).sin()) as f32;
        let (l, _) = strip.process_stereo_sample(x, x, 0.0, 0.0, threshold_db, -90.0);
        if i >= n / 2 {
            let (x, y) = (f64::from(x), f64::from(l));
            xy += x * y;
            xx += x * x;
            yy += y * y;
        }
    }
    (xy / xx, (yy / xx).sqrt())
}

#[test]
fn deesser_only_attenuates_its_band() {
    for step in 0..=30 {
        let threshold_db = -60.0 + 2.0 * f64::from(step);
        let (signed, _) = deesser_gain(7200.0, threshold_db);
        assert!(
            (0.0..=1.0 + 1e-3).contains(&signed),
            "band-centre gain {signed:.3} at threshold {threshold_db} dB is outside 0..1"
        );
        for hz in [3000.0, 5000.0, 6000.0, 7200.0, 8500.0, 10_000.0, 12_000.0] {
            let (_, rms_ratio) = deesser_gain(hz, threshold_db);
            assert!(
                rms_ratio <= 1.0 + 1e-3,
                "de-esser boosts {hz} Hz by {rms_ratio:.3}x at threshold {threshold_db} dB"
            );
        }
    }
}

/// A loud 50 Hz pop under quiet broadband content, so the plosive detectors trip.
fn plosive_signal(len: usize) -> Vec<f64> {
    white_noise(len, 0.01)
        .into_iter()
        .enumerate()
        .map(|(i, n)| n + 0.5 * (2.0 * PI * 50.0 * i as f64 / SR).sin())
        .collect()
}

fn rms(s: &[f64]) -> f64 {
    (s.iter().map(|v| v * v).sum::<f64>() / s.len() as f64).sqrt()
}

#[test]
fn deplosive_still_ducks_low_pops() {
    let x = plosive_signal(48_000);
    let buf = AudioBuffer::from_channels(vec![x.clone()], SR as u32).expect("buffer");
    let offline = clean_vocal_surgical(&buf, false, true, 0.0).expect("cleaned");
    let offline_ratio = rms(&offline.channels[0][WARMUP..]) / rms(&x[WARMUP..]);
    assert!(
        offline_ratio < 0.8,
        "offline de-plosive left {offline_ratio:.2} of the pop"
    );

    let mut strip = RealtimeVocalSurgical::new(SR);
    let y: Vec<f64> = x
        .iter()
        .map(|&s| {
            f64::from(
                strip
                    .process_stereo_sample(s as f32, s as f32, 0.0, 1.0, 0.0, -90.0)
                    .0,
            )
        })
        .collect();
    let realtime_ratio = rms(&y[WARMUP..]) / rms(&x[WARMUP..]);
    assert!(
        realtime_ratio < 0.8,
        "realtime de-plosive left {realtime_ratio:.2} of the pop"
    );
}
