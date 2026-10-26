//! Reader- and writer-based audio file I/O.
//!
//! The path-based [`AudioBuffer::load`], [`AudioBuffer::load_wav`] and
//! [`AudioBuffer::save_wav`] open the file and delegate here, so there is one decode and one
//! encode path. Callers that must control how a file is opened (the MCP server opens every
//! client file through a directory handle anchored at its sandbox root) open it themselves and
//! pass the handle to these functions.

use std::io::{self, BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::process::Command;

use hound::{SampleFormat, WavReader, WavSpec, WavWriter};

use crate::{AudioBuffer, ForgeError, Result};

fn audio_io(origin: &str, message: impl std::fmt::Display) -> ForgeError {
    ForgeError::AudioIo {
        path: origin.to_string(),
        message: message.to_string(),
    }
}

impl AudioBuffer {
    /// Decodes audio from an open source. `name` gives the format hint (its extension) and
    /// the name used in errors; it is never opened.
    ///
    /// WAV is decoded natively. Other formats, and WAV files hound rejects, are copied from
    /// `reader` into a private temporary file that FFmpeg decodes, so FFmpeg reads the bytes
    /// of the already-open source rather than re-opening `name`.
    pub fn load_from_reader<R: Read + Seek>(mut reader: R, name: &Path) -> Result<Self> {
        let origin = name.display().to_string();
        if has_wav_extension(name) {
            if let Ok(buf) = Self::load_wav_from_reader(BufReader::new(&mut reader), &origin) {
                return Ok(buf);
            }
            reader.seek(SeekFrom::Start(0))?;
        }
        decode_with_ffmpeg(&mut reader, name)
    }

    /// Native WAV decoder (8/16/24/32-bit PCM and 32-bit IEEE float) over any reader.
    /// `origin` names the source in errors. Data that does not end on a whole frame is rejected
    /// rather than returned as ragged channels.
    pub fn load_wav_from_reader<R: Read>(reader: R, origin: &str) -> Result<Self> {
        let mut reader = WavReader::new(reader).map_err(|e| audio_io(origin, e))?;
        let spec = reader.spec();
        let ch_count = usize::from(spec.channels.max(1));

        let channels = match spec.sample_format {
            SampleFormat::Float => {
                let samples = reader.samples::<f32>().map(|s| s.map(f64::from));
                crate::deinterleave_whole_frames(samples, ch_count, origin)?
            }
            SampleFormat::Int => {
                let scale = 1.0 / ((1_i64 << (spec.bits_per_sample.saturating_sub(1))) as f64);
                let samples = reader
                    .samples::<i32>()
                    .map(|s| s.map(|v| f64::from(v) * scale));
                crate::deinterleave_whole_frames(samples, ch_count, origin)?
            }
        };

        Ok(Self {
            channels,
            sample_rate: spec.sample_rate,
        })
    }

    /// Encodes the buffer as WAV into `writer`: 24-bit (the default) or 16-bit PCM with TPDF
    /// dither, or 32-bit float when `bits_per_sample` is 32. The header is finalized and the
    /// writer flushed before returning. `origin` names the destination in errors.
    pub fn save_wav_to_writer<W: Write + Seek>(
        &self,
        writer: W,
        bits_per_sample: u16,
        origin: &str,
    ) -> Result<()> {
        let (sample_format, bits) = match bits_per_sample {
            32 => (SampleFormat::Float, 32),
            16 => (SampleFormat::Int, 16),
            _ => (SampleFormat::Int, 24),
        };
        let spec = WavSpec {
            channels: self.num_channels() as u16,
            sample_rate: self.sample_rate,
            bits_per_sample: bits,
            sample_format,
        };
        let mut writer = WavWriter::new(writer, spec).map_err(|e| audio_io(origin, e))?;
        match (sample_format, bits) {
            (SampleFormat::Float, _) => write_float(self, &mut writer, origin)?,
            (SampleFormat::Int, 16) => write_dithered(self, &mut writer, INT16, origin)?,
            (SampleFormat::Int, _) => write_dithered(self, &mut writer, INT24, origin)?,
        }
        writer.finalize().map_err(|e| audio_io(origin, e))
    }
}

fn has_wav_extension(name: &Path) -> bool {
    name.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("wav"))
}

/// Copies `reader` into a temporary file named with `name`'s extension (FFmpeg probes it),
/// decodes that to 32-bit float WAV with FFmpeg, and loads the result.
fn decode_with_ffmpeg<R: Read>(reader: &mut R, name: &Path) -> Result<AudioBuffer> {
    let origin = name.display().to_string();
    let suffix = name
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| format!(".{e}"))
        .unwrap_or_default();
    let mut source = tempfile::Builder::new()
        .prefix("forge_source_")
        .suffix(&suffix)
        .tempfile()?;
    io::copy(reader, source.as_file_mut())?;
    source.as_file_mut().flush()?;
    let decoded = tempfile::Builder::new()
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
            &source.path().to_string_lossy(),
            "-c:a",
            "pcm_f32le",
            &decoded.path().to_string_lossy(),
        ])
        .output()
        .map_err(|e| audio_io(&origin, format!("Failed to invoke ffmpeg decoder: {e}")))?;
    if !status.status.success() {
        return Err(audio_io(
            &origin,
            String::from_utf8_lossy(&status.stderr).trim(),
        ));
    }
    AudioBuffer::load_wav(decoded.path())
}

fn write_float<W: Write + Seek>(
    buf: &AudioBuffer,
    writer: &mut WavWriter<W>,
    origin: &str,
) -> Result<()> {
    for f in 0..buf.num_frames() {
        for ch in &buf.channels {
            let raw = ch.get(f).copied().unwrap_or(0.0);
            let s = if raw.is_finite() {
                raw.clamp(-1.0e6, 1.0e6) as f32
            } else {
                0.0_f32
            };
            writer.write_sample(s).map_err(|e| audio_io(origin, e))?;
        }
    }
    Ok(())
}

/// Integer PCM target: full-scale value and the clamp range of the quantized sample.
#[derive(Clone, Copy)]
struct IntFormat {
    scale: f64,
    min: f64,
    max: f64,
    narrow: bool,
}

const INT16: IntFormat = IntFormat {
    scale: 32767.0,
    min: -32768.0,
    max: 32767.0,
    narrow: true,
};

const INT24: IntFormat = IntFormat {
    scale: 8_388_607.0,
    min: -8_388_608.0,
    max: 8_388_607.0,
    narrow: false,
};

/// Deterministic xorshift PRNG for TPDF dither, so exports are reproducible.
struct Dither(u64);

impl Dither {
    fn next_uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let u = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (u >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64)) - 0.5
    }
}

fn clean_unit(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Writes TPDF-dithered, first-order error-shaped integer samples.
fn write_dithered<W: Write + Seek>(
    buf: &AudioBuffer,
    writer: &mut WavWriter<W>,
    fmt: IntFormat,
    origin: &str,
) -> Result<()> {
    let mut dither = Dither(0x9E37_79B9_7F4A_7C15);
    let mut err_state = vec![0.0_f64; buf.num_channels()];
    for f in 0..buf.num_frames() {
        for (ch, err_prev) in buf.channels.iter().zip(err_state.iter_mut()) {
            let raw = ch.get(f).copied().unwrap_or(0.0);
            let tpdf = dither.next_uniform() + dither.next_uniform();
            let shaped = clean_unit(raw) * fmt.scale - 0.5 * (*err_prev);
            let q = (shaped + tpdf).round().clamp(fmt.min, fmt.max);
            *err_prev = (q - shaped).clamp(-1.5, 1.5);
            let written = if fmt.narrow {
                writer.write_sample(q as i16)
            } else {
                writer.write_sample(q as i32)
            };
            written.map_err(|e| audio_io(origin, e))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn fixture() -> AudioBuffer {
        let left: Vec<f64> = (0..480)
            .map(|i| (f64::from(i) * 0.05).sin() * 0.5)
            .collect();
        let right: Vec<f64> = left.iter().map(|x| -x).collect();
        AudioBuffer::from_channels(vec![left, right], 48_000).expect("buffer")
    }

    #[test]
    fn writer_and_reader_round_trip_in_memory() {
        let buf = fixture();
        for bits in [16_u16, 24, 32] {
            let mut bytes = Cursor::new(Vec::new());
            buf.save_wav_to_writer(&mut bytes, bits, "memory")
                .expect("encode");
            let loaded = AudioBuffer::load_from_reader(
                Cursor::new(bytes.into_inner()),
                Path::new("memory.wav"),
            )
            .expect("decode");
            assert_eq!(loaded.sample_rate, 48_000);
            assert_eq!(loaded.num_channels(), 2);
            assert_eq!(loaded.num_frames(), 480);
            let max_err = buf.channels[0]
                .iter()
                .zip(&loaded.channels[0])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(max_err < 1e-3, "{bits}-bit error {max_err}");
        }
    }

    #[test]
    fn path_based_save_and_load_delegate_to_the_handle_versions() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("nested").join("out.wav");
        let buf = fixture();
        buf.save_wav(&path, 24).expect("save");

        let mut bytes = Cursor::new(Vec::new());
        buf.save_wav_to_writer(&mut bytes, 24, "memory")
            .expect("encode");
        assert_eq!(std::fs::read(&path).expect("read"), bytes.into_inner());

        let from_path = AudioBuffer::load(&path).expect("load");
        let file = std::fs::File::open(&path).expect("open");
        let from_handle = AudioBuffer::load_from_reader(file, &path).expect("load handle");
        assert_eq!(from_path.channels, from_handle.channels);
        assert_eq!(from_path.sample_rate, from_handle.sample_rate);
    }

    #[test]
    fn wav_reader_errors_name_the_origin() {
        let err = AudioBuffer::load_wav_from_reader(Cursor::new(b"not a wav".to_vec()), "x.wav")
            .expect_err("garbage");
        assert!(err.to_string().contains("x.wav"), "{err}");
    }
}
