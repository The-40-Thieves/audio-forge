//! 32-band display spectrum: the band layout every analyzer shares, the real-time filter-bank
//! analyzer the plugin GUIs draw, the STFT band reduction OpenSoothe uses, and the lock-free
//! cross-instance spectrum bus.
//!
//! All levels are dB relative to a full-scale sine: a sine of amplitude `A` centred in a band
//! reads `20 * log10(A)` in that band, so a full-scale sine reads 0 dB.

use std::sync::atomic::{AtomicU32, Ordering, fence};

use crate::loudness_limiter::{CytomicSvf, sanitize_param_f64, sanitize_sample_f32};

/// Number of display bands.
pub const SPECTRUM_BANDS: usize = 32;
/// Lower edge of band 0, in Hz.
pub const SPECTRUM_LOW_HZ: f64 = 25.0;
/// Upper edge of band 31, in Hz.
pub const SPECTRUM_HIGH_HZ: f64 = 18_500.0;
/// Level of a silent or unpublished band, in dB. It is finite (unlike `-inf`) so `max`, the
/// meters and the shared-bus sanitizer stay well defined, and it sits below the GUI's -72 dB floor.
pub const SPECTRUM_FLOOR_DB: f32 = -90.0;
/// Highest level a band reports, in dB.
pub const SPECTRUM_CEIL_DB: f32 = 6.0;

/// Power below which a band is treated as silent (-120 dB), which also keeps the envelopes
/// out of the subnormal range.
const POWER_FLOOR: f64 = 1.0e-12;
/// Release time constant of the display envelope (about 14 dB/s of fall).
const RELEASE_TAU_S: f64 = 0.3;
/// Bounds of the per-band power integration time: three periods of the band centre.
const MIN_AVG_TAU_S: f64 = 0.005;
const MAX_AVG_TAU_S: f64 = 0.12;

/// Lower and upper edge of `band`, in Hz. The 32 bands split 25 Hz to 18.5 kHz into equal
/// log-frequency steps (about 0.3 octave each) at every sample rate.
pub fn spectrum_band_edges_hz(band: usize) -> (f64, f64) {
    let ratio = SPECTRUM_HIGH_HZ / SPECTRUM_LOW_HZ;
    let edge = |i: usize| SPECTRUM_LOW_HZ * ratio.powf(i as f64 / SPECTRUM_BANDS as f64);
    (edge(band), edge(band + 1))
}

/// Geometric centre of `band`, in Hz.
pub fn spectrum_band_center_hz(band: usize) -> f64 {
    let (lo, hi) = spectrum_band_edges_hz(band);
    (lo * hi).sqrt()
}

/// Constant Q that puts each band-pass's -3 dB points on its band edges.
fn band_q() -> f64 {
    let step = (SPECTRUM_HIGH_HZ / SPECTRUM_LOW_HZ).powf(1.0 / SPECTRUM_BANDS as f64);
    1.0 / (step.sqrt() - 1.0 / step.sqrt())
}

/// Converts a sine-referenced band power to a clamped display level.
fn power_to_db(power: f64) -> f32 {
    if power > POWER_FLOOR {
        ((10.0 * power.log10()) as f32).clamp(SPECTRUM_FLOOR_DB, SPECTRUM_CEIL_DB)
    } else {
        SPECTRUM_FLOOR_DB
    }
}

/// One-pole coefficient for time constant `tau_s` at `sr`.
fn one_pole_coeff(tau_s: f64, sr: f64) -> f64 {
    1.0 - (-1.0 / (tau_s * sr)).exp()
}

// ============================================================================
// REAL-TIME FILTER-BANK ANALYZER
// ============================================================================

/// Zero-allocation 32-band real-time spectrum analyzer (25 Hz to 18.5 kHz).
///
/// Each band is a Cytomic TPT SVF band-pass at the band's geometric centre, run at the full
/// sample rate and scaled to unity gain at the centre, with its -3 dB points on the band
/// edges. Its output power is integrated over about three periods of the centre frequency,
/// then a peak-hold envelope with a 300 ms release sets the bar's ballistics.
pub struct RealtimeSpectrumAnalyzer32 {
    filters: [CytomicSvf; SPECTRUM_BANDS],
    /// `1/Q`: the SVF band-pass output has gain `Q` at its centre.
    bp_gain: f64,
    avg_coeff: [f64; SPECTRUM_BANDS],
    release_coeff: f64,
    avg_pow: [f64; SPECTRUM_BANDS],
    env_pow: [f64; SPECTRUM_BANDS],
}

impl RealtimeSpectrumAnalyzer32 {
    pub fn new(sample_rate: f64) -> Self {
        let sr = sanitize_param_f64(sample_rate, 48_000.0, 8_000.0, 768_000.0);
        let q = band_q();
        let centre = |b: usize| spectrum_band_center_hz(b).min(sr * 0.45);
        Self {
            filters: std::array::from_fn(|b| CytomicSvf::lowpass(sr, centre(b), q)),
            bp_gain: 1.0 / q,
            avg_coeff: std::array::from_fn(|b| {
                let tau = (3.0 / centre(b)).clamp(MIN_AVG_TAU_S, MAX_AVG_TAU_S);
                one_pole_coeff(tau, sr)
            }),
            release_coeff: one_pole_coeff(RELEASE_TAU_S, sr),
            avg_pow: [0.0; SPECTRUM_BANDS],
            env_pow: [0.0; SPECTRUM_BANDS],
        }
    }

    pub fn reset(&mut self, sample_rate: f64) {
        *self = Self::new(sample_rate);
    }

    /// Feeds one stereo frame (analysed as the mid signal `(l + r) / 2`).
    #[inline]
    pub fn push_stereo_sample(&mut self, l: f32, r: f32) {
        let mono = 0.5 * (sanitize_sample_f32(l) + sanitize_sample_f32(r));
        for b in 0..SPECTRUM_BANDS {
            let (_lp, bp, _hp) = self.filters[b].process_multimode(mono);
            let band = bp * self.bp_gain;
            // Twice the mean square, so a sine of amplitude A integrates to A^2.
            let avg = self.avg_pow[b] + self.avg_coeff[b] * (2.0 * band * band - self.avg_pow[b]);
            self.avg_pow[b] = if avg > POWER_FLOOR { avg } else { 0.0 };
            let env = self.env_pow[b];
            self.env_pow[b] = if avg >= env {
                avg
            } else {
                let fall = env + self.release_coeff * (avg - env);
                if fall > POWER_FLOOR { fall } else { 0.0 }
            };
        }
    }

    /// Current 32-band levels in dB re full-scale sine, clamped to `[-90, 6]`.
    #[inline]
    pub fn snapshot_db(&self) -> [f32; SPECTRUM_BANDS] {
        std::array::from_fn(|b| power_to_db(self.env_pow[b]))
    }
}

// ============================================================================
// STFT BAND REDUCTION
// ============================================================================

/// Power normalization for an `n`-point STFT frame windowed by `window`: `4 / (n * sum(w^2))`.
///
/// It corrects for the window's coherent gain and its equivalent noise bandwidth together, so
/// the sum of one-sided bin powers `|X[k]|^2` across a sine's main lobe comes to `A^2`.
pub fn stft_band_power_norm(window: &[f64]) -> f64 {
    let sum_sq: f64 = window.iter().map(|w| w * w).sum();
    4.0 / ((window.len() as f64) * sum_sq).max(f64::MIN_POSITIVE)
}

/// First and last bin (inclusive) of `band` for an STFT with `bin_hz` spacing and `half_bins`
/// one-sided bins. It always spans at least three bins around the band centre, enough for a
/// Hann main lobe, and never includes DC or Nyquist.
pub fn stft_band_bins(band: usize, bin_hz: f64, half_bins: usize) -> (usize, usize) {
    let (lo, hi) = spectrum_band_edges_hz(band);
    let centre = (spectrum_band_center_hz(band) / bin_hz).round() as usize;
    let last_bin = half_bins.saturating_sub(2).max(1);
    let b0 = ((lo / bin_hz).floor() as usize).min(centre.saturating_sub(1));
    let b1 = ((hi / bin_hz).ceil() as usize).max(centre + 1);
    (b0.clamp(1, last_bin), b1.clamp(1, last_bin))
}

/// Adds the power of the one-sided magnitudes `mags` in each band to `band_pow`, scaled by
/// `norm` from [`stft_band_power_norm`].
pub fn accumulate_stft_band_power(
    mags: &[f64],
    bin_hz: f64,
    norm: f64,
    band_pow: &mut [f64; SPECTRUM_BANDS],
) {
    for (band, acc) in band_pow.iter_mut().enumerate() {
        let (b0, b1) = stft_band_bins(band, bin_hz, mags.len());
        let sum: f64 = mags[b0..=b1].iter().map(|m| m * m).sum();
        *acc += sum * norm;
    }
}

/// Converts summed band powers (already averaged over channels) to display levels.
pub fn band_power_to_db(band_pow: &[f64; SPECTRUM_BANDS]) -> [f32; SPECTRUM_BANDS] {
    std::array::from_fn(|b| power_to_db(band_pow[b]))
}

// ============================================================================
// LOCK-FREE CROSS-INSTANCE SHARED SPECTRUM BUS
// ============================================================================

/// Number of shared slots (`Slot A..H`).
pub const SHARED_SPECTRUM_SLOT_COUNT: usize = 8;
const FLOOR_BITS: u32 = SPECTRUM_FLOOR_DB.to_bits();
/// How many times a reader retries a slot a writer is updating before giving up.
const READ_RETRIES: usize = 64;

/// Process-wide shared spectrum slots, holding 32 band levels in dB each. Every band starts at
/// [`SPECTRUM_FLOOR_DB`], so a slot nobody has published to reads as silence.
///
/// Read and write them only through [`publish_shared_spectrum`] and [`read_shared_spectrum`],
/// which guard each slot with a sequence counter so a reader never sees half of one frame and
/// half of another.
pub static SHARED_SPECTRUM_SLOTS: [[AtomicU32; SPECTRUM_BANDS]; SHARED_SPECTRUM_SLOT_COUNT] =
    [const { [const { AtomicU32::new(FLOOR_BITS) }; SPECTRUM_BANDS] }; SHARED_SPECTRUM_SLOT_COUNT];

/// Per-slot seqlock counter: odd while a writer is storing a frame.
static SHARED_SPECTRUM_SEQ: [AtomicU32; SHARED_SPECTRUM_SLOT_COUNT] =
    [const { AtomicU32::new(0) }; SHARED_SPECTRUM_SLOT_COUNT];

/// Maps a plugin's bus-slot parameter (`0 = Off`, `1..=8 = Slot A..H`) to a slot index.
pub fn shared_spectrum_slot_index(bus_param: usize) -> Option<usize> {
    bus_param
        .checked_sub(1)
        .filter(|&i| i < SHARED_SPECTRUM_SLOT_COUNT)
}

/// Non-finite levels become the floor; subnormal ones flush to 0 dB.
fn sanitize_bus_db(v: f32) -> f32 {
    if !v.is_finite() {
        SPECTRUM_FLOOR_DB
    } else if v.abs() < 1e-30 {
        0.0
    } else {
        v
    }
}

/// Publishes a 32-band spectrum (dB) into slot `slot_idx` (`0..8`) as one consistent frame.
///
/// Wait-free: if another instance is publishing to the same slot at that moment, this frame is
/// skipped and `false` is returned. Out-of-range slots are ignored and also return `false`.
#[inline]
pub fn publish_shared_spectrum(slot_idx: usize, bands_32: &[f32; SPECTRUM_BANDS]) -> bool {
    let (Some(seq), Some(slot)) = (
        SHARED_SPECTRUM_SEQ.get(slot_idx),
        SHARED_SPECTRUM_SLOTS.get(slot_idx),
    ) else {
        return false;
    };
    let start = seq.load(Ordering::Relaxed);
    if start & 1 == 1
        || seq
            .compare_exchange(
                start,
                start.wrapping_add(1),
                Ordering::Acquire,
                Ordering::Relaxed,
            )
            .is_err()
    {
        return false;
    }
    fence(Ordering::Release);
    for (atom, &val) in slot.iter().zip(bands_32.iter()) {
        atom.store(sanitize_bus_db(val).to_bits(), Ordering::Relaxed);
    }
    seq.store(start.wrapping_add(2), Ordering::Release);
    true
}

/// Resets slot `slot_idx` to silence, as when its publishing instance goes away.
#[inline]
pub fn clear_shared_spectrum(slot_idx: usize) -> bool {
    publish_shared_spectrum(slot_idx, &[SPECTRUM_FLOOR_DB; SPECTRUM_BANDS])
}

/// Reads one consistent 32-band frame (dB) from slot `slot_idx` (`0..8`).
///
/// Out-of-range slots, and a slot still being written after a bounded number of retries,
/// read as silence ([`SPECTRUM_FLOOR_DB`]).
#[inline]
pub fn read_shared_spectrum(slot_idx: usize) -> [f32; SPECTRUM_BANDS] {
    let (Some(seq), Some(slot)) = (
        SHARED_SPECTRUM_SEQ.get(slot_idx),
        SHARED_SPECTRUM_SLOTS.get(slot_idx),
    ) else {
        return [SPECTRUM_FLOOR_DB; SPECTRUM_BANDS];
    };
    for _ in 0..READ_RETRIES {
        let before = seq.load(Ordering::Acquire);
        if before & 1 == 0 {
            let frame: [f32; SPECTRUM_BANDS] =
                std::array::from_fn(|b| f32::from_bits(slot[b].load(Ordering::Relaxed)));
            fence(Ordering::Acquire);
            if seq.load(Ordering::Relaxed) == before {
                return frame.map(sanitize_bus_db);
            }
        }
        std::hint::spin_loop();
    }
    [SPECTRUM_FLOOR_DB; SPECTRUM_BANDS]
}
