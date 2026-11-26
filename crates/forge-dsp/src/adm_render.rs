//! Pure-Rust ITU-R BS.2076 / BS.2051 3D VBAP ADM object & bed reference renderer.

use forge_core::{
    AdmBwfInfo, AdmObjectBlock, AdmProgrammeKind, AdmTrackKind, AudioBuffer, ForgeError, Result,
};
use serde::{Deserialize, Serialize};

use crate::bs1770_5_position_weight;

const RENDER_BLOCK_STEP_FRAMES: usize = 64;

/// Supported ITU-R BS.2051 reference loudspeaker rendering layouts for ADM spatial QC.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum ReferenceLoudspeakerConfig {
    /// 5.1 ITU-R BS.2051 System B (`0+5+0`: L, R, C, LFE, Ls, Rs).
    Layout5_1,
    /// 7.1 ITU-R BS.2051 (`0+7+0`: L, R, C, LFE, Lss, Rss, Lrs, Rrs).
    Layout7_1,
    /// 7.1.4 ITU-R BS.2051 System H (`4+7+0`: 7.1 bed + Ltf, Rtf, Ltr, Rtr).
    #[default]
    Layout7_1_4,
}

#[derive(Debug, Clone, Copy)]
struct SpeakerSpec {
    label: &'static str,
    azimuth_deg: f64,
    elevation_deg: f64,
    is_lfe: bool,
}

const SPEAKERS_5_1: [SpeakerSpec; 6] = [
    SpeakerSpec {
        label: "M+030",
        azimuth_deg: 30.0,
        elevation_deg: 0.0,
        is_lfe: false,
    },
    SpeakerSpec {
        label: "M-030",
        azimuth_deg: -30.0,
        elevation_deg: 0.0,
        is_lfe: false,
    },
    SpeakerSpec {
        label: "M+000",
        azimuth_deg: 0.0,
        elevation_deg: 0.0,
        is_lfe: false,
    },
    SpeakerSpec {
        label: "LFE",
        azimuth_deg: 0.0,
        elevation_deg: 0.0,
        is_lfe: true,
    },
    SpeakerSpec {
        label: "M+110",
        azimuth_deg: 110.0,
        elevation_deg: 0.0,
        is_lfe: false,
    },
    SpeakerSpec {
        label: "M-110",
        azimuth_deg: -110.0,
        elevation_deg: 0.0,
        is_lfe: false,
    },
];

const SPEAKERS_7_1: [SpeakerSpec; 8] = [
    SpeakerSpec {
        label: "M+030",
        azimuth_deg: 30.0,
        elevation_deg: 0.0,
        is_lfe: false,
    },
    SpeakerSpec {
        label: "M-030",
        azimuth_deg: -30.0,
        elevation_deg: 0.0,
        is_lfe: false,
    },
    SpeakerSpec {
        label: "M+000",
        azimuth_deg: 0.0,
        elevation_deg: 0.0,
        is_lfe: false,
    },
    SpeakerSpec {
        label: "LFE",
        azimuth_deg: 0.0,
        elevation_deg: 0.0,
        is_lfe: true,
    },
    SpeakerSpec {
        label: "M+090",
        azimuth_deg: 90.0,
        elevation_deg: 0.0,
        is_lfe: false,
    },
    SpeakerSpec {
        label: "M-090",
        azimuth_deg: -90.0,
        elevation_deg: 0.0,
        is_lfe: false,
    },
    SpeakerSpec {
        label: "M+135",
        azimuth_deg: 135.0,
        elevation_deg: 0.0,
        is_lfe: false,
    },
    SpeakerSpec {
        label: "M-135",
        azimuth_deg: -135.0,
        elevation_deg: 0.0,
        is_lfe: false,
    },
];

const SPEAKERS_7_1_4: [SpeakerSpec; 12] = [
    SpeakerSpec {
        label: "M+030",
        azimuth_deg: 30.0,
        elevation_deg: 0.0,
        is_lfe: false,
    },
    SpeakerSpec {
        label: "M-030",
        azimuth_deg: -30.0,
        elevation_deg: 0.0,
        is_lfe: false,
    },
    SpeakerSpec {
        label: "M+000",
        azimuth_deg: 0.0,
        elevation_deg: 0.0,
        is_lfe: false,
    },
    SpeakerSpec {
        label: "LFE",
        azimuth_deg: 0.0,
        elevation_deg: 0.0,
        is_lfe: true,
    },
    SpeakerSpec {
        label: "M+090",
        azimuth_deg: 90.0,
        elevation_deg: 0.0,
        is_lfe: false,
    },
    SpeakerSpec {
        label: "M-090",
        azimuth_deg: -90.0,
        elevation_deg: 0.0,
        is_lfe: false,
    },
    SpeakerSpec {
        label: "M+135",
        azimuth_deg: 135.0,
        elevation_deg: 0.0,
        is_lfe: false,
    },
    SpeakerSpec {
        label: "M-135",
        azimuth_deg: -135.0,
        elevation_deg: 0.0,
        is_lfe: false,
    },
    SpeakerSpec {
        label: "U+030",
        azimuth_deg: 30.0,
        elevation_deg: 30.0,
        is_lfe: false,
    },
    SpeakerSpec {
        label: "U-030",
        azimuth_deg: -30.0,
        elevation_deg: 30.0,
        is_lfe: false,
    },
    SpeakerSpec {
        label: "U+110",
        azimuth_deg: 110.0,
        elevation_deg: 30.0,
        is_lfe: false,
    },
    SpeakerSpec {
        label: "U-110",
        azimuth_deg: -110.0,
        elevation_deg: 30.0,
        is_lfe: false,
    },
];

const VIRTUAL_ZENITH_IDX: usize = 12;

const TRIANGLES_7_1_4: [[usize; 3]; 15] = [
    [2, 0, 8],
    [2, 8, 9],
    [2, 9, 1],
    [0, 4, 8],
    [4, 10, 8],
    [4, 6, 10],
    [6, 7, 10],
    [7, 11, 10],
    [7, 5, 11],
    [5, 1, 9],
    [5, 11, 9],
    [8, 9, VIRTUAL_ZENITH_IDX],
    [8, 10, VIRTUAL_ZENITH_IDX],
    [10, 11, VIRTUAL_ZENITH_IDX],
    [11, 9, VIRTUAL_ZENITH_IDX],
];

impl ReferenceLoudspeakerConfig {
    /// Human-readable ITU-R BS.2051 specification identifier for QC reports.
    pub fn description(self) -> &'static str {
        match self {
            Self::Layout5_1 => "ITU-R BS.2051-3 System B (0+5+0 / 5.1)",
            Self::Layout7_1 => "ITU-R BS.2051-3 (0+7+0 / 7.1)",
            Self::Layout7_1_4 => "ITU-R BS.2051-3 System H (4+7+0 / 7.1.4)",
        }
    }

    pub fn channel_count(self) -> usize {
        self.speakers().len()
    }

    pub fn speaker_labels(self) -> Vec<String> {
        self.speakers()
            .iter()
            .map(|s| s.label.to_string())
            .collect()
    }

    /// ITU-R BS.1770-5 Annex 3 speaker-position power weights `g_i` for this reference layout.
    pub fn bs1770_5_weights(self) -> Vec<f64> {
        self.speakers()
            .iter()
            .map(|s| bs1770_5_position_weight(s.azimuth_deg, s.elevation_deg, s.is_lfe))
            .collect()
    }

    fn speakers(self) -> &'static [SpeakerSpec] {
        match self {
            Self::Layout5_1 => &SPEAKERS_5_1,
            Self::Layout7_1 => &SPEAKERS_7_1,
            Self::Layout7_1_4 => &SPEAKERS_7_1_4,
        }
    }

    fn horizontal_ring(self) -> &'static [usize] {
        match self {
            Self::Layout5_1 => &[2, 0, 4, 5, 1],
            Self::Layout7_1 | Self::Layout7_1_4 => &[2, 0, 4, 6, 7, 5, 1],
        }
    }
}

fn adm_dir_vec(azimuth_deg: f64, elevation_deg: f64) -> [f64; 3] {
    let az = azimuth_deg.to_radians();
    let el = elevation_deg.clamp(-90.0, 90.0).to_radians();
    let ce = el.cos();
    [-ce * az.sin(), ce * az.cos(), el.sin()]
}

fn l2_normalize_gains(gains: &mut [f64]) {
    let sum_sq: f64 = gains.iter().map(|g| g * g).sum();
    if sum_sq > 1e-18 {
        let inv = 1.0 / sum_sq.sqrt();
        for g in gains {
            *g *= inv;
        }
    }
}

fn solve_2d_horizontal(ring: &[usize], specs: &[SpeakerSpec], azimuth_deg: f64) -> Vec<f64> {
    let mut gains = vec![0.0_f64; specs.len()];
    let [tx, ty, _] = adm_dir_vec(azimuth_deg, 0.0);
    let t_norm = (tx * tx + ty * ty).sqrt().max(1e-12);
    let ux = tx / t_norm;
    let uy = ty / t_norm;

    for win in 0..ring.len() {
        let ia = ring[win];
        let ib = ring[(win + 1) % ring.len()];
        let [ax, ay, _] = adm_dir_vec(specs[ia].azimuth_deg, 0.0);
        let [bx, by, _] = adm_dir_vec(specs[ib].azimuth_deg, 0.0);
        let det = ax * by - ay * bx;
        if det.abs() < 1e-9 {
            continue;
        }
        let ga = (ux * by - uy * bx) / det;
        let gb = (ax * uy - ay * ux) / det;
        if ga >= -1e-6 && gb >= -1e-6 {
            gains[ia] = ga.max(0.0);
            gains[ib] = gb.max(0.0);
            l2_normalize_gains(&mut gains);
            return gains;
        }
    }
    let best = *ring
        .iter()
        .max_by(|&&a, &&b| {
            let va = adm_dir_vec(specs[a].azimuth_deg, 0.0);
            let vb = adm_dir_vec(specs[b].azimuth_deg, 0.0);
            (va[0] * ux + va[1] * uy).total_cmp(&(vb[0] * ux + vb[1] * uy))
        })
        .unwrap_or(&0);
    gains[best] = 1.0;
    gains
}

fn det3(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    a[0] * (b[1] * c[2] - b[2] * c[1]) - b[0] * (a[1] * c[2] - a[2] * c[1])
        + c[0] * (a[1] * b[2] - a[2] * b[1])
}

fn solve_3d_hull_gains(specs: &[SpeakerSpec], azimuth_deg: f64, elevation_deg: f64) -> Vec<f64> {
    let mut gains = vec![0.0_f64; specs.len()];
    let u = adm_dir_vec(azimuth_deg, elevation_deg);
    let node_vec = |idx: usize| -> [f64; 3] {
        if idx == VIRTUAL_ZENITH_IDX {
            [0.0, 0.0, 1.0]
        } else {
            adm_dir_vec(specs[idx].azimuth_deg, specs[idx].elevation_deg)
        }
    };

    let mut best_slack = f64::NEG_INFINITY;
    let mut best_tri = TRIANGLES_7_1_4[0];
    let mut best_w = [0.0_f64; 3];
    for &tri in &TRIANGLES_7_1_4 {
        let (v0, v1, v2) = (node_vec(tri[0]), node_vec(tri[1]), node_vec(tri[2]));
        let d = det3(v0, v1, v2);
        if d.abs() < 1e-9 {
            continue;
        }
        let w = [
            det3(u, v1, v2) / d,
            det3(v0, u, v2) / d,
            det3(v0, v1, u) / d,
        ];
        let min_w = w[0].min(w[1]).min(w[2]);
        if min_w > best_slack {
            best_slack = min_w;
            best_tri = tri;
            best_w = w;
        }
    }
    for (idx, w) in best_tri.iter().zip(best_w) {
        let clamped = w.max(0.0);
        if *idx == VIRTUAL_ZENITH_IDX {
            for g in &mut gains[8..12] {
                *g += 0.5 * clamped;
            }
        } else {
            gains[*idx] += clamped;
        }
    }
    l2_normalize_gains(&mut gains);
    gains
}

/// Computes energy-preserving (`sum(g^2) == 1.0`, LFE `== 0.0`) VBAP loudspeaker gains.
pub fn compute_vbap_gains(
    target: ReferenceLoudspeakerConfig,
    azimuth_deg: f64,
    elevation_deg: f64,
) -> Vec<f64> {
    let specs = target.speakers();
    if matches!(target, ReferenceLoudspeakerConfig::Layout7_1_4) && elevation_deg > 0.0 {
        solve_3d_hull_gains(specs, azimuth_deg, elevation_deg)
    } else {
        solve_2d_horizontal(target.horizontal_ring(), specs, azimuth_deg)
    }
}

fn accumulate_constant_gains(src: &[f64], out: &mut [Vec<f64>], gains: &[f64]) {
    for (ch_out, &g) in out.iter_mut().zip(gains) {
        if g.abs() <= 1e-12 {
            continue;
        }
        for (dst, &s) in ch_out.iter_mut().zip(src) {
            *dst += s * g;
        }
    }
}

fn direct_speaker_gains(
    target: ReferenceLoudspeakerConfig,
    az: f64,
    el: f64,
    is_lfe: bool,
) -> Vec<f64> {
    let specs = target.speakers();
    let mut gains = vec![0.0_f64; specs.len()];
    if is_lfe {
        if let Some(lfe_idx) = specs.iter().position(|s| s.is_lfe) {
            gains[lfe_idx] = 1.0;
        }
        return gains;
    }
    if let Some((idx, _)) = specs.iter().enumerate().find(|(_, s)| {
        !s.is_lfe && (s.azimuth_deg - az).abs() <= 2.0 && (s.elevation_deg - el).abs() <= 2.0
    }) {
        gains[idx] = 1.0;
        return gains;
    }
    compute_vbap_gains(target, az, el)
}

fn eval_block_state_at(
    blocks: &[AdmObjectBlock],
    object_gain: f64,
    t_secs: f64,
    prev_state: &mut (f64, f64, f64),
) -> (f64, f64, f64) {
    let active = blocks
        .iter()
        .find(|b| {
            let end = b.start_secs + b.duration_secs.unwrap_or(f64::INFINITY);
            t_secs >= b.start_secs && t_secs < end
        })
        .or_else(|| blocks.last());
    let Some(blk) = active else {
        return *prev_state;
    };
    let (az, el, g) = if blk.jump_position && blk.interpolation_secs.is_none_or(|v| v <= 0.0) {
        (blk.azimuth_deg, blk.elevation_deg, object_gain * blk.gain)
    } else if let Some(dur) = blk.interpolation_secs.filter(|&d| d > 0.0) {
        let alpha = ((t_secs - blk.start_secs) / dur).clamp(0.0, 1.0);
        let az = prev_state.0 + alpha * (blk.azimuth_deg - prev_state.0);
        let el = prev_state.1 + alpha * (blk.elevation_deg - prev_state.1);
        let g = prev_state.2 + alpha * (object_gain * blk.gain - prev_state.2);
        (az, el, g)
    } else {
        (blk.azimuth_deg, blk.elevation_deg, object_gain * blk.gain)
    };
    *prev_state = (az, el, g);
    (az, el, g)
}

fn render_object_track(
    src: &[f64],
    out: &mut [Vec<f64>],
    target: ReferenceLoudspeakerConfig,
    sr: f64,
    object_gain: f64,
    blocks: &[AdmObjectBlock],
) {
    if blocks.is_empty() {
        let mut gains = compute_vbap_gains(target, 0.0, 0.0);
        for g in &mut gains {
            *g *= object_gain;
        }
        accumulate_constant_gains(src, out, &gains);
        return;
    }
    let mut state = (
        blocks[0].azimuth_deg,
        blocks[0].elevation_deg,
        object_gain * blocks[0].gain,
    );
    let mut start = 0usize;
    while start < src.len() {
        let end = (start + RENDER_BLOCK_STEP_FRAMES).min(src.len());
        let mid_secs = ((start + end) as f64) * 0.5 / sr;
        let (az, el, gain) = eval_block_state_at(blocks, object_gain, mid_secs, &mut state);
        let vbap = compute_vbap_gains(target, az, el);
        for (ch_out, &g) in out.iter_mut().zip(&vbap) {
            let scaled = g * gain;
            if scaled.abs() <= 1e-12 {
                continue;
            }
            for (dst, &s) in ch_out[start..end].iter_mut().zip(&src[start..end]) {
                *dst += s * scaled;
            }
        }
        start = end;
    }
}

/// Renders a validated ADM BWF programme (ChannelBedOnly, ObjectOnly, or MixedBedAndObjects)
/// onto a specified ITU-R BS.2051 `target` loudspeaker layout for BS.1770-5 Annex 4 metering.
pub fn render_adm_programme_to_layout(
    buf: &AudioBuffer,
    adm: &AdmBwfInfo,
    target: ReferenceLoudspeakerConfig,
) -> Result<AudioBuffer> {
    if !adm.metadata_errors.is_empty()
        || matches!(
            adm.programme_kind,
            AdmProgrammeKind::MissingAdm | AdmProgrammeKind::UnsupportedOrMalformed
        )
    {
        return Err(ForgeError::InvalidParameter(
            "Cannot render ADM programme with missing or malformed/unsupported metadata"
                .to_string(),
        ));
    }
    let mut out = AudioBuffer::new(target.channel_count(), buf.num_frames(), buf.sample_rate)?;
    let sr = f64::from(buf.sample_rate.max(1));
    for assoc in &adm.track_associations {
        let track_idx = usize::from(assoc.track_index.saturating_sub(1));
        let Some(src) = buf.channels.get(track_idx) else {
            return Err(ForgeError::InvalidParameter(format!(
                "ADM track_index {} out of bounds for {} WAV channels",
                assoc.track_index,
                buf.num_channels()
            )));
        };
        match &assoc.kind {
            AdmTrackKind::DirectSpeaker(pos) => {
                let gains =
                    direct_speaker_gains(target, pos.azimuth_deg, pos.elevation_deg, pos.is_lfe);
                accumulate_constant_gains(src, &mut out.channels, &gains);
            }
            AdmTrackKind::Object {
                object_gain,
                blocks,
            } => {
                render_object_track(src, &mut out.channels, target, sr, *object_gain, blocks);
            }
            AdmTrackKind::Unsupported(td) => {
                return Err(ForgeError::InvalidParameter(format!(
                    "Unsupported ADM track typeDefinition {td:?} on track {}",
                    assoc.track_index
                )));
            }
        }
    }
    Ok(out)
}
