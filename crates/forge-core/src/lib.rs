//! Core 64-bit planar audio buffer, studio telemetry contracts, and file I/O for ForgeAudio.

use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Structured domain error hierarchy for ForgeAudio operations.
#[derive(Debug, Error)]
pub enum ForgeError {
    #[error("Audio I/O failure on '{path}': {message}")]
    AudioIo { path: String, message: String },

    #[error("Invalid audio parameter: {0}")]
    InvalidParameter(String),

    #[error("Music theory / ABC score syntax error: {0}")]
    TheoryError(String),

    #[error("REAPER integration error: {0}")]
    ReaperError(String),

    #[error("GPU Worker / Model execution error: {0}")]
    WorkerError(String),

    #[error("ISO 3901 ISRC validation error: {0}")]
    IsrcError(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Filesystem error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, ForgeError>;

mod wav_io;

fn open_for_read(path: &Path) -> Result<std::fs::File> {
    std::fs::File::open(path).map_err(|e| ForgeError::AudioIo {
        path: path.display().to_string(),
        message: e.to_string(),
    })
}

/// 64-bit double-precision planar studio audio buffer (`channels[ch][frame]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioBuffer {
    pub channels: Vec<Vec<f64>>,
    pub sample_rate: u32,
}

impl AudioBuffer {
    /// Creates a silent `AudioBuffer` with `num_channels` and `num_frames`.
    pub fn new(num_channels: usize, num_frames: usize, sample_rate: u32) -> Result<Self> {
        if num_channels == 0 {
            return Err(ForgeError::InvalidParameter(
                "AudioBuffer requires at least 1 channel".to_string(),
            ));
        }
        if sample_rate == 0 {
            return Err(ForgeError::InvalidParameter(
                "Sample rate must be greater than 0 Hz".to_string(),
            ));
        }
        Ok(Self {
            channels: vec![vec![0.0; num_frames]; num_channels],
            sample_rate,
        })
    }

    /// Creates an `AudioBuffer` directly from planar channel vectors.
    /// Validates that `channels` is non-empty, `sample_rate > 0`, and all channels have identical frame length.
    pub fn from_channels(channels: Vec<Vec<f64>>, sample_rate: u32) -> Result<Self> {
        if channels.is_empty() {
            return Err(ForgeError::InvalidParameter(
                "AudioBuffer requires at least 1 channel".to_string(),
            ));
        }
        if sample_rate == 0 {
            return Err(ForgeError::InvalidParameter(
                "Sample rate must be greater than 0 Hz".to_string(),
            ));
        }
        let expected_frames = channels[0].len();
        for (idx, ch) in channels.iter().enumerate().skip(1) {
            if ch.len() != expected_frames {
                return Err(ForgeError::InvalidParameter(format!(
                    "Ragged AudioBuffer: channel 0 has {} frames, but channel {} has {} frames",
                    expected_frames,
                    idx,
                    ch.len()
                )));
            }
        }
        Ok(Self {
            channels,
            sample_rate,
        })
    }

    /// Number of channels (`1` = mono, `2` = stereo).
    #[inline]
    pub fn num_channels(&self) -> usize {
        self.channels.len()
    }

    /// Number of sample frames per channel.
    #[inline]
    pub fn num_frames(&self) -> usize {
        self.channels.first().map_or(0, Vec::len)
    }

    /// Duration in seconds.
    #[inline]
    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.num_frames() as f64 / f64::from(self.sample_rate)
        }
    }

    /// Converts mono to stereo (or returns a 2-channel clone if already stereo+).
    pub fn to_stereo(&self) -> Self {
        match self.num_channels() {
            0 => Self {
                channels: vec![Vec::new(), Vec::new()],
                sample_rate: self.sample_rate,
            },
            1 => Self {
                channels: vec![self.channels[0].clone(), self.channels[0].clone()],
                sample_rate: self.sample_rate,
            },
            _ => Self {
                channels: vec![self.channels[0].clone(), self.channels[1].clone()],
                sample_rate: self.sample_rate,
            },
        }
    }

    /// Downmixes all channels to mono with equal power scaling, sanitizing any non-finite samples.
    pub fn to_mono(&self) -> Vec<f64> {
        let frames = self.num_frames();
        let ch_count = self.num_channels();
        if ch_count == 0 || frames == 0 {
            return Vec::new();
        }
        let sanitize = |x: f64| -> f64 {
            if !x.is_finite() || x.abs() < 1e-30 {
                0.0
            } else {
                x.clamp(-1.0e6, 1.0e6)
            }
        };
        if ch_count == 1 {
            return self.channels[0].iter().map(|&x| sanitize(x)).collect();
        }
        let inv = 1.0 / (ch_count as f64);
        let mut mono = vec![0.0; frames];
        for ch in &self.channels {
            for (dst, &src) in mono.iter_mut().zip(ch.iter()) {
                *dst += sanitize(src) * inv;
            }
        }
        mono
    }

    /// Loads an `AudioBuffer` from WAV natively or decodes FLAC/MP3/OGG/M4A/AIFF via FFmpeg.
    /// Opens `path` and delegates to [`AudioBuffer::load_from_reader`].
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        if !path_ref.exists() {
            return Err(ForgeError::AudioIo {
                path: path_ref.display().to_string(),
                message: "File does not exist".to_string(),
            });
        }
        let file = open_for_read(path_ref)?;
        Self::load_from_reader(file, path_ref)
    }

    /// Native WAV loader supporting 8/16/24/32-bit PCM and 32-bit IEEE float.
    /// Opens `path` and delegates to [`AudioBuffer::load_wav_from_reader`].
    pub fn load_wav<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        let file = open_for_read(path_ref)?;
        Self::load_wav_from_reader(BufReader::new(file), &path_ref.display().to_string())
    }

    /// Writes the `AudioBuffer` as a 24-bit PCM WAV file with TPDF dithering or 32-bit float WAV.
    /// Creates `path` (and its parent directories) and delegates to
    /// [`AudioBuffer::save_wav_to_writer`].
    pub fn save_wav<P: AsRef<Path>>(&self, path: P, bits_per_sample: u16) -> Result<()> {
        let path_ref = path.as_ref();
        if let Some(parent) = path_ref.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let origin = path_ref.display().to_string();
        let file = std::fs::File::create(path_ref).map_err(|e| ForgeError::AudioIo {
            path: origin.clone(),
            message: e.to_string(),
        })?;
        self.save_wav_to_writer(BufWriter::new(file), bits_per_sample, &origin)
    }
}

/// Comprehensive studio loudness, phase, electrical, and AI-artifact telemetry report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditReport {
    pub sample_rate: u32,
    pub channels: usize,
    pub duration_secs: f64,
    pub integrated_lufs: f64,
    pub short_term_max_lufs: f64,
    pub momentary_max_lufs: f64,
    pub loudness_range_lu: f64,
    pub sample_peak_dbfs: f64,
    pub true_peak_dbtp: f64,
    pub true_peak_timestamp_secs: f64,
    pub crest_factor_db: f64,
    pub dc_offset: f64,
    pub clipped_samples: usize,
    pub stereo_correlation: f64,
    pub sub_160hz_correlation: f64,
    pub hum_50_60hz_db: f64,
    pub box_mud_200_500hz_db: f64,
    pub harshness_2_6khz_db: f64,
    pub sibilance_5_14khz_db: f64,
    /// Composite 0..100 score quantifying neural codec / diffusion whistling & bandwidth artifacts.
    pub ai_fizz_index: f64,
    pub distrokid_ready: bool,
    pub recommendations: Vec<String>,
}

/// Canonical stem bundle produced by separation or live multitrack ingest.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StemBundle {
    pub vocals: Option<PathBuf>,
    pub drums: Option<PathBuf>,
    pub bass: Option<PathBuf>,
    pub guitar: Option<PathBuf>,
    pub keys: Option<PathBuf>,
    pub other: Option<PathBuf>,
    pub kick_midi: Option<PathBuf>,
    pub snare_midi: Option<PathBuf>,
    pub hats_midi: Option<PathBuf>,
    pub chords_midi: Option<PathBuf>,
    pub bass_midi: Option<PathBuf>,
    pub melody_midi: Option<PathBuf>,
    pub abc_score: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_audiobuffer_rejects_ragged_channels_and_zero_sr(
            len_a in 1usize..512,
            len_b in 1usize..512,
            sr in 0u32..96_000,
        ) {
            let res = AudioBuffer::from_channels(vec![vec![0.0; len_a], vec![0.0; len_b]], sr);
            if sr == 0 || len_a != len_b {
                prop_assert!(res.is_err());
            } else {
                let buf = res.unwrap();
                prop_assert_eq!(buf.num_channels(), 2);
                prop_assert_eq!(buf.num_frames(), len_a);
            }
        }

        #[test]
        fn prop_wav_16_24_32_roundtrip_and_nan_sanitization(
            frames in 16usize..256,
            freq in 50.0f64..4000.0,
            amp in 0.05f64..0.95,
        ) {
            let sr = 48_000_u32;
            let mut ch0 = vec![0.0_f64; frames];
            let mut ch1 = vec![0.0_f64; frames];
            for i in 0..frames {
                let t = (i as f64) / (sr as f64);
                ch0[i] = amp * (2.0 * std::f64::consts::PI * freq * t).sin();
                ch1[i] = -ch0[i];
            }
            // Inject adversarial NaN / ±Inf at index 0 and 1 to verify save_wav & to_mono sanitize them
            ch0[0] = f64::NAN;
            ch1[0] = f64::INFINITY;
            if frames > 1 {
                ch0[1] = f64::NEG_INFINITY;
                ch1[1] = 1.0e-315; // subnormal
            }

            let buf = AudioBuffer::from_channels(vec![ch0.clone(), ch1.clone()], sr).unwrap();
            let mono = buf.to_mono();
            prop_assert_eq!(mono.len(), frames);
            for &m in &mono {
                prop_assert!(m.is_finite());
            }

            let dir = tempfile::tempdir().unwrap();
            for &bits in &[16_u16, 24, 32] {
                let path = dir.path().join(format!("test_{bits}.wav"));
                buf.save_wav(&path, bits).unwrap();
                let loaded = AudioBuffer::load_wav(&path).unwrap();
                prop_assert_eq!(loaded.num_channels(), 2);
                prop_assert_eq!(loaded.num_frames(), frames);
                prop_assert_eq!(loaded.sample_rate, sr);

                // First 2 frames were poisoned and must be sanitized to finite (~0.0)
                prop_assert!(loaded.channels[0][0].abs() < 1e-3);
                prop_assert!(loaded.channels[1][0].abs() < 1e-3);

                let max_tol = match bits {
                    32 => 1e-6,
                    24 => 1e-4,
                    _ => 2e-3,
                };
                for i in 2..frames {
                    let err0 = (loaded.channels[0][i] - ch0[i]).abs();
                    let err1 = (loaded.channels[1][i] - ch1[i]).abs();
                    prop_assert!(err0 < max_tol, "bits={} err0={}", bits, err0);
                    prop_assert!(err1 < max_tol, "bits={} err1={}", bits, err1);
                }
            }
        }
    }
}
