//! Core 64-bit planar audio buffer, studio telemetry contracts, and file I/O for ForgeAudio.

use std::path::{Path, PathBuf};
use std::process::Command;

use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
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

    /// Downmixes all channels to mono with equal power scaling.
    pub fn to_mono(&self) -> Vec<f64> {
        let frames = self.num_frames();
        let ch_count = self.num_channels();
        if ch_count == 0 || frames == 0 {
            return Vec::new();
        }
        if ch_count == 1 {
            return self.channels[0].clone();
        }
        let inv = 1.0 / (ch_count as f64);
        let mut mono = vec![0.0; frames];
        for ch in &self.channels {
            for (dst, &src) in mono.iter_mut().zip(ch.iter()) {
                *dst += src * inv;
            }
        }
        mono
    }

    /// Loads an `AudioBuffer` from WAV natively or decodes FLAC/MP3/OGG/M4A/AIFF via FFmpeg.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        if !path_ref.exists() {
            return Err(ForgeError::AudioIo {
                path: path_ref.display().to_string(),
                message: "File does not exist".to_string(),
            });
        }
        let ext = path_ref
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        if ext == "wav" {
            if let Ok(buf) = Self::load_wav(path_ref) {
                return Ok(buf);
            }
        }

        // Fallback to FFmpeg decoding into a temporary 32-bit float WAV
        let tmp = tempfile::Builder::new()
            .prefix("forge_decode_")
            .suffix(".wav")
            .tempfile()?;
        let status = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                &path_ref.to_string_lossy(),
                "-c:a",
                "pcm_f32le",
                &tmp.path().to_string_lossy(),
            ])
            .output()
            .map_err(|e| ForgeError::AudioIo {
                path: path_ref.display().to_string(),
                message: format!("Failed to invoke ffmpeg decoder: {e}"),
            })?;

        if !status.status.success() {
            return Err(ForgeError::AudioIo {
                path: path_ref.display().to_string(),
                message: String::from_utf8_lossy(&status.stderr).trim().to_string(),
            });
        }

        Self::load_wav(tmp.path())
    }

    /// Native WAV loader supporting 8/16/24/32-bit PCM and 32-bit IEEE float.
    pub fn load_wav<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        let mut reader = WavReader::open(path_ref).map_err(|e| ForgeError::AudioIo {
            path: path_ref.display().to_string(),
            message: e.to_string(),
        })?;
        let spec = reader.spec();
        let ch_count = usize::from(spec.channels.max(1));
        let total_samples = reader.len() as usize;
        let frames = total_samples / ch_count;
        let mut channels = vec![Vec::with_capacity(frames); ch_count];

        match (spec.sample_format, spec.bits_per_sample) {
            (SampleFormat::Float, _) => {
                for (idx, sample) in reader.samples::<f32>().enumerate() {
                    let val = sample.map_err(|e| ForgeError::AudioIo {
                        path: path_ref.display().to_string(),
                        message: e.to_string(),
                    })?;
                    channels[idx % ch_count].push(f64::from(val));
                }
            }
            (SampleFormat::Int, bits) => {
                let scale = 1.0 / ((1_i64 << (bits.saturating_sub(1))) as f64);
                for (idx, sample) in reader.samples::<i32>().enumerate() {
                    let val = sample.map_err(|e| ForgeError::AudioIo {
                        path: path_ref.display().to_string(),
                        message: e.to_string(),
                    })?;
                    channels[idx % ch_count].push(f64::from(val) * scale);
                }
            }
        }

        Ok(Self {
            channels,
            sample_rate: spec.sample_rate,
        })
    }

    /// Writes the `AudioBuffer` as a 24-bit PCM WAV file with TPDF dithering or 32-bit float WAV.
    pub fn save_wav<P: AsRef<Path>>(&self, path: P, bits_per_sample: u16) -> Result<()> {
        let path_ref = path.as_ref();
        if let Some(parent) = path_ref.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let ch_count = self.num_channels();
        let frames = self.num_frames();

        let (sample_format, bits) = if bits_per_sample == 32 {
            (SampleFormat::Float, 32)
        } else if bits_per_sample == 16 {
            (SampleFormat::Int, 16)
        } else {
            (SampleFormat::Int, 24)
        };

        let spec = WavSpec {
            channels: ch_count as u16,
            sample_rate: self.sample_rate,
            bits_per_sample: bits,
            sample_format,
        };

        let mut writer = WavWriter::create(path_ref, spec).map_err(|e| ForgeError::AudioIo {
            path: path_ref.display().to_string(),
            message: e.to_string(),
        })?;

        // Deterministic high-quality PRNG for TPDF dither on integer exports
        let mut rng_state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next_uniform = || -> f64 {
            rng_state ^= rng_state >> 12;
            rng_state ^= rng_state << 25;
            rng_state ^= rng_state >> 27;
            let u = rng_state.wrapping_mul(0x2545_F491_4F6C_DD1D);
            (u >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64)) - 0.5
        };

        match (sample_format, bits) {
            (SampleFormat::Float, _) => {
                for f in 0..frames {
                    for ch in 0..ch_count {
                        let s = self.channels[ch][f] as f32;
                        writer.write_sample(s).map_err(|e| ForgeError::AudioIo {
                            path: path_ref.display().to_string(),
                            message: e.to_string(),
                        })?;
                    }
                }
            }
            (SampleFormat::Int, 16) => {
                let scale = 32767.0_f64;
                for f in 0..frames {
                    for ch in 0..ch_count {
                        let tpdf = next_uniform() + next_uniform();
                        let s = (self.channels[ch][f].clamp(-1.0, 1.0) * scale + tpdf)
                            .round()
                            .clamp(-32768.0, 32767.0) as i16;
                        writer.write_sample(s).map_err(|e| ForgeError::AudioIo {
                            path: path_ref.display().to_string(),
                            message: e.to_string(),
                        })?;
                    }
                }
            }
            (SampleFormat::Int, _) => {
                let scale = 8_388_607.0_f64;
                for f in 0..frames {
                    for ch in 0..ch_count {
                        let tpdf = next_uniform() + next_uniform();
                        let s = (self.channels[ch][f].clamp(-1.0, 1.0) * scale + tpdf)
                            .round()
                            .clamp(-8_388_608.0, 8_388_607.0) as i32;
                        writer.write_sample(s).map_err(|e| ForgeError::AudioIo {
                            path: path_ref.display().to_string(),
                            message: e.to_string(),
                        })?;
                    }
                }
            }
        }

        writer.finalize().map_err(|e| ForgeError::AudioIo {
            path: path_ref.display().to_string(),
            message: e.to_string(),
        })?;
        Ok(())
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
