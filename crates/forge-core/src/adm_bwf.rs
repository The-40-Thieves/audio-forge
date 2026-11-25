//! Pure-Rust RIFF/RF64 Audio Definition Model (ADM) Broadcast Wave Format (ITU-R BS.2076) parser.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{ForgeError, Result};

const MAX_ADM_CHUNKS: usize = 1024;
const MAX_CHNA_BYTES: usize = 1024 * 1024;
const MAX_AXML_BYTES: usize = 16 * 1024 * 1024;

/// Parsed container, BWF, and ITU-R BS.2076 ADM metadata from a WAV/RF64 file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AdmBwfInfo {
    pub is_rf64: bool,
    pub audio_format: u16,
    pub channels: u16,
    pub sample_rate: u32,
    pub bits_per_sample: u16,
    pub has_bext: bool,
    pub has_chna: bool,
    pub has_axml: bool,
    pub chna_tracks: u16,
    pub chna_uids: u16,
    pub adm_programmes: usize,
    pub adm_contents: usize,
    pub adm_objects: usize,
    pub adm_track_uids: usize,
    pub binaural_render_modes: Vec<String>,
}

fn adm_err(origin: &str, message: impl std::fmt::Display) -> ForgeError {
    ForgeError::AudioIo {
        path: origin.to_string(),
        message: message.to_string(),
    }
}

fn read_exact_arr<const N: usize, R: Read>(r: &mut R, origin: &str, what: &str) -> Result<[u8; N]> {
    let mut buf = [0_u8; N];
    r.read_exact(&mut buf)
        .map_err(|e| adm_err(origin, format!("Truncated {what}: {e}")))?;
    Ok(buf)
}

fn read_bounded_chunk<R: Read>(
    r: &mut R,
    size: u64,
    cap: usize,
    origin: &str,
    id: &str,
) -> Result<Vec<u8>> {
    if size > cap as u64 {
        return Err(adm_err(
            origin,
            format!("RIFF '{id}' chunk ({size} bytes) exceeds {cap}-byte cap"),
        ));
    }
    let mut buf = vec![0_u8; size as usize];
    r.read_exact(&mut buf)
        .map_err(|e| adm_err(origin, format!("Truncated RIFF '{id}' chunk: {e}")))?;
    Ok(buf)
}

fn parse_fmt_chunk(buf: &[u8], info: &mut AdmBwfInfo, origin: &str) -> Result<()> {
    if buf.len() < 16 {
        return Err(adm_err(origin, "RIFF 'fmt ' chunk shorter than 16 bytes"));
    }
    info.audio_format = u16::from_le_bytes([buf[0], buf[1]]);
    info.channels = u16::from_le_bytes([buf[2], buf[3]]);
    info.sample_rate = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
    info.bits_per_sample = u16::from_le_bytes([buf[14], buf[15]]);
    if info.audio_format == 0xFFFE && buf.len() >= 20 {
        let valid_bits = u16::from_le_bytes([buf[18], buf[19]]);
        if valid_bits > 0 {
            info.bits_per_sample = valid_bits;
        }
    }
    Ok(())
}

fn parse_ds64_data_size(buf: &[u8], origin: &str) -> Result<Option<u64>> {
    if buf.len() < 28 {
        return Err(adm_err(origin, "RF64 'ds64' chunk shorter than 28 bytes"));
    }
    let data_size = u64::from_le_bytes(buf[8..16].try_into().unwrap_or([0; 8]));
    Ok((data_size > 0).then_some(data_size))
}

fn extract_binaural_modes(xml: &str) -> Vec<String> {
    let mut modes = Vec::new();
    let lower = xml.to_ascii_lowercase();
    let mut rest = lower.as_str();
    while let Some(pos) = rest.find("binauralrendermode") {
        let after = &rest[pos + "binauralrendermode".len()..];
        let trimmed = after.trim_start();
        if let Some(after_eq) = trimmed.strip_prefix('=') {
            let quoted = after_eq.trim_start();
            if let Some(q) = quoted.chars().next().filter(|&c| c == '"' || c == '\'')
                && let Some(end) = quoted[1..].find(q)
            {
                push_unique_mode(&mut modes, &quoted[1..1 + end]);
            }
        } else if let Some(gt) = trimmed.find('>')
            && let Some(lt) = trimmed[gt + 1..].find('<')
        {
            push_unique_mode(&mut modes, &trimmed[gt + 1..gt + 1 + lt]);
        }
        rest = after;
    }
    modes
}

fn push_unique_mode(modes: &mut Vec<String>, raw: &str) {
    let val = raw.trim();
    if !val.is_empty() && !modes.iter().any(|m| m == val) {
        modes.push(val.to_string());
    }
}

fn count_tag(haystack_lower: &str, tag_lower: &str) -> usize {
    let open = format!("<{tag_lower}");
    haystack_lower
        .match_indices(&open)
        .filter(|&(idx, _)| {
            haystack_lower
                .as_bytes()
                .get(idx + open.len())
                .is_some_and(|&b| b.is_ascii_whitespace() || b == b'>' || b == b'/')
        })
        .count()
}

fn parse_axml_chunk(buf: &[u8], info: &mut AdmBwfInfo, origin: &str) -> Result<()> {
    let xml = std::str::from_utf8(buf)
        .map_err(|e| adm_err(origin, format!("Invalid UTF-8 in ADM 'axml' chunk: {e}")))?;
    let lower = xml.to_ascii_lowercase();
    info.has_axml = true;
    info.adm_programmes = count_tag(&lower, "audioprogramme");
    info.adm_contents = count_tag(&lower, "audiocontent");
    info.adm_objects = count_tag(&lower, "audioobject");
    info.adm_track_uids = count_tag(&lower, "audiotrackuid");
    info.binaural_render_modes = extract_binaural_modes(xml);
    Ok(())
}

fn parse_chna_chunk(buf: &[u8], info: &mut AdmBwfInfo, origin: &str) -> Result<()> {
    if buf.len() < 4 {
        return Err(adm_err(origin, "RIFF 'chna' chunk shorter than 4 bytes"));
    }
    info.has_chna = true;
    info.chna_tracks = u16::from_le_bytes([buf[0], buf[1]]);
    info.chna_uids = u16::from_le_bytes([buf[2], buf[3]]);
    Ok(())
}

fn handle_adm_chunk<R: Read + Seek>(
    reader: &mut R,
    chunk_id: &[u8; 4],
    chunk_size: u64,
    info: &mut AdmBwfInfo,
    ds64_data_size: &mut Option<u64>,
    origin: &str,
) -> Result<()> {
    match chunk_id {
        b"fmt " => {
            let buf = read_bounded_chunk(reader, chunk_size, 4096, origin, "fmt ")?;
            parse_fmt_chunk(&buf, info, origin)
        }
        b"ds64" => {
            let buf = read_bounded_chunk(reader, chunk_size, 65_536, origin, "ds64")?;
            *ds64_data_size = parse_ds64_data_size(&buf, origin)?;
            Ok(())
        }
        b"bext" => {
            info.has_bext = true;
            skip_chunk_payload(reader, chunk_size, origin, "bext")
        }
        b"chna" => {
            let buf = read_bounded_chunk(reader, chunk_size, MAX_CHNA_BYTES, origin, "chna")?;
            parse_chna_chunk(&buf, info, origin)
        }
        b"axml" => {
            let buf = read_bounded_chunk(reader, chunk_size, MAX_AXML_BYTES, origin, "axml")?;
            parse_axml_chunk(&buf, info, origin)
        }
        _ => skip_chunk_payload(
            reader,
            chunk_size,
            origin,
            &String::from_utf8_lossy(chunk_id),
        ),
    }
}

fn skip_chunk_payload<R: Seek>(reader: &mut R, size: u64, origin: &str, id: &str) -> Result<()> {
    let offset = i64::try_from(size).map_err(|_| {
        adm_err(
            origin,
            format!("RIFF '{id}' chunk size {size} overflows i64"),
        )
    })?;
    reader
        .seek(SeekFrom::Current(offset))
        .map_err(|e| adm_err(origin, format!("Failed seeking past RIFF '{id}': {e}")))?;
    Ok(())
}

fn empty_adm_info(is_rf64: bool) -> AdmBwfInfo {
    AdmBwfInfo {
        is_rf64,
        audio_format: 0,
        channels: 0,
        sample_rate: 0,
        bits_per_sample: 0,
        has_bext: false,
        has_chna: false,
        has_axml: false,
        chna_tracks: 0,
        chna_uids: 0,
        adm_programmes: 0,
        adm_contents: 0,
        adm_objects: 0,
        adm_track_uids: 0,
        binaural_render_modes: Vec::new(),
    }
}

fn scan_adm_chunks<R: Read + Seek>(
    reader: &mut R,
    file_len: u64,
    info: &mut AdmBwfInfo,
    origin: &str,
) -> Result<bool> {
    let mut ds64_data_size: Option<u64> = None;
    let mut saw_fmt = false;
    for _ in 0..MAX_ADM_CHUNKS {
        let pos = reader.stream_position().map_err(|e| adm_err(origin, e))?;
        if pos + 8 > file_len {
            break;
        }
        let hdr = read_exact_arr::<8, _>(reader, origin, "RIFF chunk header")?;
        let chunk_id: [u8; 4] = [hdr[0], hdr[1], hdr[2], hdr[3]];
        let raw_size = u64::from(u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]));
        let chunk_size = if info.is_rf64 && &chunk_id == b"data" && raw_size == u64::from(u32::MAX)
        {
            ds64_data_size.ok_or_else(|| adm_err(origin, "RF64 data size sentinel without ds64"))?
        } else {
            raw_size
        };
        if pos + 8 + chunk_size > file_len {
            return Err(adm_err(origin, "Truncated RIFF chunk extends past EOF"));
        }
        saw_fmt |= &chunk_id == b"fmt ";
        handle_adm_chunk(
            reader,
            &chunk_id,
            chunk_size,
            info,
            &mut ds64_data_size,
            origin,
        )?;
        if chunk_size % 2 == 1 && reader.stream_position().unwrap_or(0) < file_len {
            reader
                .seek(SeekFrom::Current(1))
                .map_err(|e| adm_err(origin, e))?;
        }
    }
    Ok(saw_fmt)
}

/// Parses WAV/RF64 container metadata and ITU-R BS.2076 ADM chunks (`bext`, `chna`, `axml`).
pub fn parse_adm_bwf_reader<R: Read + Seek>(reader: &mut R, origin: &str) -> Result<AdmBwfInfo> {
    let header = read_exact_arr::<12, _>(reader, origin, "RIFF/RF64 header")?;
    let is_rf64 = match &header[0..4] {
        b"RIFF" => false,
        b"RF64" => true,
        _ => return Err(adm_err(origin, "Expected RIFF or RF64 container signature")),
    };
    if &header[8..12] != b"WAVE" {
        return Err(adm_err(origin, "Expected WAVE RIFF form type"));
    }
    let file_len = reader
        .seek(SeekFrom::End(0))
        .map_err(|e| adm_err(origin, e))?;
    reader
        .seek(SeekFrom::Start(12))
        .map_err(|e| adm_err(origin, e))?;

    let mut info = empty_adm_info(is_rf64);
    if !scan_adm_chunks(reader, file_len, &mut info, origin)? {
        return Err(adm_err(origin, "Missing required 'fmt ' chunk"));
    }
    Ok(info)
}

/// Parses ADM BWF metadata from a WAV/RF64 file on disk.
pub fn parse_adm_bwf_file<P: AsRef<Path>>(path: P) -> Result<AdmBwfInfo> {
    let p = path.as_ref();
    let mut file = crate::open_for_read(p)?;
    parse_adm_bwf_reader(&mut file, &p.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn append_chunk(out: &mut Vec<u8>, id: &[u8; 4], payload: &[u8]) {
        out.extend_from_slice(id);
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
        if payload.len() % 2 == 1 {
            out.push(0);
        }
    }

    fn build_fmt_payload(channels: u16, sample_rate: u32, bits: u16) -> Vec<u8> {
        let block_align = channels * (bits / 8);
        let byte_rate = sample_rate * u32::from(block_align);
        let mut fmt = Vec::with_capacity(16);
        fmt.extend_from_slice(&1_u16.to_le_bytes());
        fmt.extend_from_slice(&channels.to_le_bytes());
        fmt.extend_from_slice(&sample_rate.to_le_bytes());
        fmt.extend_from_slice(&byte_rate.to_le_bytes());
        fmt.extend_from_slice(&block_align.to_le_bytes());
        fmt.extend_from_slice(&bits.to_le_bytes());
        fmt
    }

    fn finalize_riff(tag: &[u8; 4], body: Vec<u8>) -> Vec<u8> {
        let mut riff = Vec::with_capacity(12 + body.len());
        riff.extend_from_slice(tag);
        riff.extend_from_slice(&((4 + body.len()) as u32).to_le_bytes());
        riff.extend_from_slice(b"WAVE");
        riff.extend_from_slice(&body);
        riff
    }

    #[test]
    fn parses_synthetic_riff_adm_bwf_chunks_and_binaural_modes() {
        let mut body = Vec::new();
        append_chunk(&mut body, b"fmt ", &build_fmt_payload(6, 48_000, 24));
        append_chunk(&mut body, b"bext", &[0_u8; 602]);
        let mut chna = vec![0_u8; 4 + 40];
        chna[0..2].copy_from_slice(&6_u16.to_le_bytes());
        chna[2..4].copy_from_slice(&6_u16.to_le_bytes());
        append_chunk(&mut body, b"chna", &chna);
        let axml = br#"<ebuCoreMain><audioFormatExtended>
            <audioProgramme audioProgrammeID="APR_1001"/>
            <audioContent audioContentID="ACO_1001"/>
            <audioObject audioObjectID="AO_1001" binauralRenderMode="near"/>
            <audioTrackUID UID="ATU_00000001"/>
        </audioFormatExtended></ebuCoreMain>"#;
        append_chunk(&mut body, b"axml", axml);
        append_chunk(&mut body, b"data", &[0_u8; 18]);

        let bytes = finalize_riff(b"RIFF", body);
        let info = parse_adm_bwf_reader(&mut Cursor::new(bytes), "synth.wav").expect("parse");
        assert!(!info.is_rf64);
        assert_eq!(info.channels, 6);
        assert_eq!(info.sample_rate, 48_000);
        assert_eq!(info.bits_per_sample, 24);
        assert!(info.has_bext);
        assert!(info.has_chna);
        assert!(info.has_axml);
        assert_eq!(info.chna_tracks, 6);
        assert_eq!(info.chna_uids, 6);
        assert_eq!(info.adm_programmes, 1);
        assert_eq!(info.adm_contents, 1);
        assert_eq!(info.adm_objects, 1);
        assert_eq!(info.adm_track_uids, 1);
        assert_eq!(info.binaural_render_modes, vec!["near".to_string()]);
    }

    #[test]
    fn parses_rf64_with_ds64_and_rejects_truncated_containers() {
        let mut body = Vec::new();
        let mut ds64 = vec![0_u8; 28];
        ds64[8..16].copy_from_slice(&12_u64.to_le_bytes());
        append_chunk(&mut body, b"ds64", &ds64);
        append_chunk(&mut body, b"fmt ", &build_fmt_payload(12, 48_000, 24));
        body.extend_from_slice(b"data");
        body.extend_from_slice(&u32::MAX.to_le_bytes());
        body.extend_from_slice(&[0_u8; 12]);

        let rf64 = finalize_riff(b"RF64", body);
        let parsed = parse_adm_bwf_reader(&mut Cursor::new(rf64), "rf64.wav").expect("rf64 parse");
        assert!(parsed.is_rf64);
        assert_eq!(parsed.channels, 12);

        assert!(parse_adm_bwf_reader(&mut Cursor::new(b"RIFF\x00\x00"), "short.wav").is_err());
        assert!(
            parse_adm_bwf_reader(&mut Cursor::new(b"OggS\x00\x00\x00\x00WAVE"), "bad.ogg").is_err()
        );
    }
}
