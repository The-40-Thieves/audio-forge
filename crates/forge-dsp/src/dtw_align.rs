//! Banded DTW Vocal Double / Harmony Time-Aligner (VocAlign equivalent).
//!
//! Moved out of [`crate::phase_align`], which keeps re-exporting the public items so existing
//! `phase_align::` paths still resolve.

use forge_core::{AudioBuffer, ForgeError, Result};
use serde::{Deserialize, Serialize};

/// A REAPER-compatible stretch marker mapping `source_pos_secs` to `timeline_pos_secs`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StretchMarker {
    pub timeline_pos_secs: f64,
    pub source_pos_secs: f64,
}

/// Time-aligns a vocal double/harmony (`dub`) to a lead vocal (`guide`) using banded Dynamic Time Warping
/// (Synchro Arts VocAlign equivalent) and emits native REAPER `SM` stretch markers.
///
/// DTW memory is linear in the input length: only the warp band of each row is stored. Output
/// past the end of the dub is silent.
pub fn align_vocal_double_dtw(
    guide: &AudioBuffer,
    dub: &AudioBuffer,
    max_warp_ms: f64,
) -> Result<(AudioBuffer, Vec<StretchMarker>)> {
    align_vocal_double_dtw_traced(guide, dub, max_warp_ms).map(|(buf, markers, _)| (buf, markers))
}

/// [`align_vocal_double_dtw`], also returning how many DTW cells it allocated (`0` when the
/// input is too short to align). Tests use the count to check the memory bound.
pub(crate) fn align_vocal_double_dtw_traced(
    guide: &AudioBuffer,
    dub: &AudioBuffer,
    max_warp_ms: f64,
) -> Result<(AudioBuffer, Vec<StretchMarker>, usize)> {
    if guide.sample_rate != dub.sample_rate {
        return Err(ForgeError::InvalidParameter(
            "Guide and dub vocals must share the same sample rate".to_string(),
        ));
    }
    let sr = f64::from(guide.sample_rate.max(1));
    let g_mono = guide.to_mono();
    let d_mono = dub.to_mono();
    let frames = g_mono.len().min(d_mono.len());
    // Extract 10ms energy + zero-crossing (sibilance/consonant) feature vectors
    let hop = ((sr * 0.010).round() as usize).max(64);
    let num_steps = frames / hop;
    if frames < 1024 || num_steps < 4 {
        return Ok((dub.clone(), Vec::new(), 0));
    }
    let g_feats = dtw_features(&g_mono, hop, num_steps);
    let d_feats = dtw_features(&d_mono, hop, num_steps);

    let clean_warp_ms =
        crate::loudness_limiter::sanitize_param_f64(max_warp_ms, 60.0, 10.0, 1000.0);
    let band = (((clean_warp_ms / 10.0).round() as usize).max(2)).min(num_steps / 2);
    let mut dtw = BandedDtw::new(num_steps, band);
    dtw.fill(&g_feats, &d_feats);
    let step_map = dtw.step_map();

    let markers = stretch_markers(&step_map, hop, sr);
    let out_frames = guide.num_frames();
    let channels = dub
        .channels
        .iter()
        .map(|ch| warp_channel(ch, &step_map, hop, out_frames))
        .collect();
    let warped = AudioBuffer {
        channels,
        sample_rate: dub.sample_rate,
    };
    Ok((warped, markers, dtw.allocated_cells()))
}

/// Per-hop `(log energy dB, zero-crossing rate × 40)` features for the first `num_steps` hops.
fn dtw_features(sig: &[f64], hop: usize, num_steps: usize) -> Vec<(f64, f64)> {
    (0..num_steps)
        .map(|s| {
            let slice = &sig[s * hop..(s + 1) * hop];
            let sum_sq: f64 = slice.iter().map(|x| x * x).sum();
            let zcr = slice
                .windows(2)
                .filter(|w| (w[1] >= 0.0) != (w[0] >= 0.0))
                .count() as f64;
            let log_e = 10.0 * (sum_sq / (slice.len() as f64) + 1e-9).log10();
            let norm_zcr = zcr / (slice.len() as f64);
            (log_e, norm_zcr * 40.0)
        })
        .collect()
}

/// Back-pointer moves stored per DTW cell.
const MOVE_NONE: u8 = 0;
const MOVE_DIAG: u8 = 1;
const MOVE_UP: u8 = 2;
const MOVE_LEFT: u8 = 3;
/// Extra cost of a horizontal or vertical DTW step over a diagonal one.
const STEP_PENALTY: f64 = 1.5;

/// Sakoe-Chiba banded DTW over `num_steps` guide and dub feature frames.
///
/// Only cells with `|i - j| <= band` can lie on the path, so each guide step keeps
/// `2 * band + 1` one-byte back-pointers and the cost lives in two rolling rows. Memory is
/// `O(num_steps * (2 * band + 1))` instead of the `O(num_steps²)` of a full matrix, and the
/// path is identical to the full-matrix banded search.
pub(crate) struct BandedDtw {
    num_steps: usize,
    band: usize,
    width: usize,
    /// Move into each banded cell, row-major by guide step.
    back: Vec<u8>,
}

impl BandedDtw {
    pub(crate) fn new(num_steps: usize, band: usize) -> Self {
        let width = 2 * band + 1;
        Self {
            num_steps,
            band,
            width,
            back: vec![MOVE_NONE; num_steps * width],
        }
    }

    /// Cells held by the back-pointer table and the two cost rows.
    pub(crate) fn allocated_cells(&self) -> usize {
        self.back.len() + 2 * self.width
    }

    fn in_band(&self, i: usize, j: usize) -> bool {
        j + self.band >= i && j <= i + self.band
    }

    /// Column of cell `(i, j)` within row `i`. Callers check [`Self::in_band`] first.
    fn slot(&self, i: usize, j: usize) -> usize {
        j + self.band - i
    }

    /// Runs the cost recurrence and records the best move into every banded cell.
    pub(crate) fn fill(&mut self, guide: &[(f64, f64)], dub: &[(f64, f64)]) {
        let mut prev = vec![f64::INFINITY; self.width];
        let mut cur = vec![f64::INFINITY; self.width];
        for (i, g) in guide.iter().enumerate().take(self.num_steps) {
            cur.fill(f64::INFINITY);
            let j_min = i.saturating_sub(self.band);
            let j_max = (i + self.band).min(self.num_steps - 1);
            for (j, d) in dub.iter().enumerate().take(j_max + 1).skip(j_min) {
                let dist = (g.0 - d.0).abs() + (g.1 - d.1).abs();
                let (best, mv) = if i == 0 && j == 0 {
                    (0.0, MOVE_NONE)
                } else {
                    self.best_predecessor(i, j, &prev, &cur)
                };
                let k = self.slot(i, j);
                cur[k] = best + dist;
                self.back[i * self.width + k] = mv;
            }
            std::mem::swap(&mut prev, &mut cur);
        }
    }

    /// Cheapest of the diagonal, vertical and horizontal predecessors of `(i, j)`; ties keep
    /// that order.
    fn best_predecessor(&self, i: usize, j: usize, prev: &[f64], cur: &[f64]) -> (f64, u8) {
        let mut best = (f64::INFINITY, MOVE_NONE);
        if i > 0 && j > 0 && self.in_band(i - 1, j - 1) && prev[self.slot(i - 1, j - 1)] < best.0 {
            best = (prev[self.slot(i - 1, j - 1)], MOVE_DIAG);
        }
        if i > 0 && self.in_band(i - 1, j) && prev[self.slot(i - 1, j)] + STEP_PENALTY < best.0 {
            best = (prev[self.slot(i - 1, j)] + STEP_PENALTY, MOVE_UP);
        }
        if j > 0 && self.in_band(i, j - 1) && cur[self.slot(i, j - 1)] + STEP_PENALTY < best.0 {
            best = (cur[self.slot(i, j - 1)] + STEP_PENALTY, MOVE_LEFT);
        }
        best
    }

    /// Backtracks from `(N-1, N-1)` and maps each guide step to the last dub step it meets.
    pub(crate) fn step_map(&self) -> Vec<f64> {
        let mut step_map = vec![0.0_f64; self.num_steps];
        let (mut i, mut j) = (self.num_steps - 1, self.num_steps - 1);
        let mut last_i = usize::MAX;
        loop {
            // Backtracking meets each guide step at its largest dub step first.
            if i != last_i {
                step_map[i] = j as f64;
                last_i = i;
            }
            match self.back[i * self.width + self.slot(i, j)] {
                MOVE_DIAG => (i, j) = (i - 1, j - 1),
                MOVE_UP => i -= 1,
                MOVE_LEFT => j -= 1,
                _ => break,
            }
        }
        step_map
    }
}

/// Emits decimated REAPER StretchMarkers every ~250ms where timing diverges
fn stretch_markers(step_map: &[f64], hop: usize, sr: f64) -> Vec<StretchMarker> {
    (0..step_map.len())
        .step_by(25)
        .map(|i| StretchMarker {
            timeline_pos_secs: (i * hop) as f64 / sr,
            source_pos_secs: (step_map[i] * (hop as f64)) / sr,
        })
        .collect()
}

/// Resamples one dub channel smoothly along the warped contour. Frames that map past the last
/// dub sample are silent.
fn warp_channel(ch: &[f64], step_map: &[f64], hop: usize, out_frames: usize) -> Vec<f64> {
    let last = ch.len().saturating_sub(1);
    (0..out_frames)
        .map(|f| {
            let src_f = warped_source_pos(f, step_map, hop);
            if ch.is_empty() || src_f > last as f64 {
                return 0.0;
            }
            let idx0 = src_f.floor() as usize;
            let idx_frac = src_f - (idx0 as f64);
            ch[idx0] * (1.0 - idx_frac) + ch[(idx0 + 1).min(last)] * idx_frac
        })
        .collect()
}

/// Dub sample position for output frame `f`. Past the last DTW step the contour continues at
/// one dub sample per output sample, so the dub tail plays out and then ends.
fn warped_source_pos(f: usize, step_map: &[f64], hop: usize) -> f64 {
    let hop_f = hop as f64;
    let last_step = step_map.len() - 1;
    let step_f = (f as f64) / hop_f;
    let s0 = step_f.floor() as usize;
    if s0 >= last_step {
        return step_map[last_step] * hop_f + (f - last_step * hop) as f64;
    }
    let frac = step_f - (s0 as f64);
    let mapped_step = step_map[s0] * (1.0 - frac) + step_map[s0 + 1] * frac;
    mapped_step * hop_f
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Deterministic xorshift noise in `-1.0..1.0`.
    fn noise(len: usize, seed: u64) -> Vec<f64> {
        let mut state = seed.max(1);
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0
            })
            .collect()
    }

    fn mono(samples: Vec<f64>, sr: u32) -> AudioBuffer {
        AudioBuffer::from_channels(vec![samples], sr).expect("valid buffer")
    }

    /// `count` 120 ms test syllables as `(amplitude, tone Hz, length in samples)`.
    fn syllables(count: usize, seed: u64, sr: f64) -> Vec<(f64, f64, usize)> {
        let draws = noise(2 * count, seed);
        let len = (0.12 * sr) as usize;
        draws
            .chunks(2)
            .map(|d| {
                let amp = 0.425 + 0.375 * d[0];
                let freq = 150.0 + (0.5 + 0.5 * d[1]) * (0.35 * sr - 150.0);
                (amp, freq, len)
            })
            .collect()
    }

    fn render(segments: &[(f64, f64, usize)], sr: f64) -> Vec<f64> {
        segments
            .iter()
            .flat_map(|&(amp, freq, len)| {
                (0..len).map(move |i| amp * (2.0 * PI * freq * (i as f64) / sr).sin())
            })
            .collect()
    }

    /// The pre-fix DTW recurrence over a full `N x N` matrix, kept as the reference result.
    fn full_matrix_step_map(g: &[(f64, f64)], d: &[(f64, f64)], band: usize) -> Vec<f64> {
        let n = g.len();
        let mut cost = vec![vec![f64::INFINITY; n]; n];
        let mut back = vec![vec![(0_usize, 0_usize); n]; n];
        for i in 0..n {
            for j in i.saturating_sub(band)..=(i + band).min(n - 1) {
                let dist = (g[i].0 - d[j].0).abs() + (g[i].1 - d[j].1).abs();
                if i == 0 && j == 0 {
                    cost[0][0] = dist;
                    continue;
                }
                let mut best = (f64::INFINITY, (0, 0));
                if i > 0 && j > 0 && cost[i - 1][j - 1] < best.0 {
                    best = (cost[i - 1][j - 1], (i - 1, j - 1));
                }
                if i > 0 && cost[i - 1][j] + STEP_PENALTY < best.0 {
                    best = (cost[i - 1][j] + STEP_PENALTY, (i - 1, j));
                }
                if j > 0 && cost[i][j - 1] + STEP_PENALTY < best.0 {
                    best = (cost[i][j - 1] + STEP_PENALTY, (i, j - 1));
                }
                cost[i][j] = best.0 + dist;
                back[i][j] = best.1;
            }
        }
        let mut path = vec![(n - 1, n - 1)];
        while path[path.len() - 1] != (0, 0) {
            let (ci, cj) = path[path.len() - 1];
            path.push(back[ci][cj]);
        }
        let mut step_map = vec![0.0; n];
        for &(i, j) in path.iter().rev() {
            step_map[i] = j as f64;
        }
        step_map
    }

    #[test]
    fn banded_dtw_matches_full_matrix_reference() {
        let n = 400;
        let feats = |seed| {
            noise(2 * n, seed)
                .chunks(2)
                .map(|c| (30.0 * c[0], 5.0 + 5.0 * c[1]))
                .collect::<Vec<_>>()
        };
        let (g, d) = (feats(5), feats(9));
        for band in [2, 12, 60] {
            let mut dtw = BandedDtw::new(n, band);
            dtw.fill(&g, &d);
            assert_eq!(
                dtw.step_map(),
                full_matrix_step_map(&g, &d, band),
                "band {band}"
            );
        }
    }

    #[test]
    fn ten_minute_dtw_alignment_stays_within_memory_budget() {
        // 10 minutes at 8 kHz gives 60_000 steps of 10 ms, the same step count as at 48 kHz.
        let sr = 8_000.0;
        let guide = render(&syllables(5_000, 21, sr), sr);
        let frames = guide.len();
        assert_eq!(frames, 10 * 60 * 8_000);
        let mut dub = vec![0.0; frames];
        dub[400..].copy_from_slice(&guide[..frames - 400]);
        let (warped, markers, cells) =
            align_vocal_double_dtw_traced(&mono(guide, 8_000), &mono(dub, 8_000), 120.0)
                .expect("aligned");
        let (num_steps, width) = (frames / 80, 2 * 12 + 1);
        // One byte per banded cell, about 1.5 MB. The old full matrices needed about 86 GB.
        assert_eq!(cells, num_steps * width + 2 * width, "cells {cells}");
        assert!(cells <= 64 << 20, "cells {cells}");
        assert_eq!(warped.num_frames(), frames);
        assert_eq!(markers.len(), num_steps.div_ceil(25));
    }

    #[test]
    fn dtw_recovers_a_known_stretch() {
        let sr = 16_000.0;
        let guide_segs = syllables(40, 3, sr);
        let mut dub_segs = guide_segs.clone();
        // Syllable 20 (2.40 s to 2.52 s in the guide) is 40 ms longer in the dub.
        dub_segs[20].2 += 640;
        let guide = mono(render(&guide_segs, sr), 16_000);
        let dub = mono(render(&dub_segs, sr), 16_000);
        let (_, markers) = align_vocal_double_dtw(&guide, &dub, 120.0).expect("aligned");
        let offsets = |from: f64, to: f64| {
            markers
                .iter()
                .filter(|m| (from..to).contains(&m.timeline_pos_secs))
                .map(|m| m.source_pos_secs - m.timeline_pos_secs)
                .collect::<Vec<_>>()
        };
        let before = offsets(0.25, 2.3);
        let after = offsets(2.6, 4.5);
        assert_eq!((before.len(), after.len()), (9, 7));
        assert!(before.iter().all(|o| o.abs() < 0.011), "{before:?}");
        assert!(after.iter().all(|o| (o - 0.04).abs() < 0.011), "{after:?}");
    }

    #[test]
    fn dtw_output_is_silent_past_the_end_of_a_shorter_dub() {
        let sr = 16_000_u32;
        let tone = |secs: f64| {
            (0..(secs * 16_000.0) as usize)
                .map(|i| 0.5 + 0.3 * (2.0 * PI * 220.0 * (i as f64) / 16_000.0).sin())
                .collect::<Vec<_>>()
        };
        let (guide, dub) = (mono(tone(5.0), sr), mono(tone(3.0), sr));
        let (warped, _) = align_vocal_double_dtw(&guide, &dub, 60.0).expect("aligned");
        assert_eq!(warped.num_frames(), guide.num_frames());
        let (body, tail) = warped.channels[0].split_at(dub.num_frames());
        // The dub plays to its last sample, then the output is silent rather than a held sample.
        assert!(body.iter().all(|s| *s >= 0.19), "dub body dropped out");
        let tail_peak = tail.iter().fold(0.0_f64, |m, s| m.max(s.abs()));
        assert_eq!(tail_peak, 0.0, "held sample past the end of the dub");
    }
}
