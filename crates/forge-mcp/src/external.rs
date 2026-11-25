//! Steps that invoke external code taking path names: FFmpeg (ISRC tagging) and the
//! stem-separation worker.
//!
//! To close the TOCTOU window (THE-1600 / THE-1869), neither external tool is ever pointed
//! directly at a path inside `FORGE_MCP_ROOT`. Instead, `forge-mcp`:
//! 1. Opens every client input file through the `cap-std` sandbox root handle and copies it
//!    into a private process-owned temporary directory (`tempfile::tempdir()`).
//! 2. Runs FFmpeg or the stem worker exclusively inside that private staging directory.
//! 3. Copies every produced artifact back into `FORGE_MCP_ROOT` through `cap-std`-anchored
//!    `sandbox.create_file` handles, refusing any swapped symlink at write time.

use crate::sandbox::PathSandbox;
use forge_core::{AudioBuffer, ForgeError, StemBundle};
use forge_dsp::BeatGridReport;
use forge_reaper::{ReleaseMetadata, write_distrokid_metadata};
use forge_worker::{
    DrumHit, MidiTranscribeResult, NeuralDenoiseResult, NeuralRestoreResult, QualityEvalResult,
    SeparationConfig, SeparationReport, VocalRescueResult, evaluate_quality_hybrid,
    index_drum_sample_library, neural_restore_hybrid, render_drum_sample_layer,
    resolve_default_local_worker, run_local_neural_denoise, run_local_transcribe_midi,
    separate_stems, track_beats_hybrid, vocal_rescue_hybrid,
};
use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Copies a sandboxed input file (opened through the `cap-std` root handle) into `stage_dir`,
/// preserving its file name/extension so external tools detect the format normally.
fn stage_in(sandbox: &PathSandbox, src: &Path, stage_dir: &Path) -> Result<PathBuf, ForgeError> {
    let mut reader = sandbox.open_read(src)?;
    let file_name = src
        .file_name()
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| std::ffi::OsStr::new("input.wav"));
    let staged = stage_dir.join(file_name);
    let mut writer = File::create(&staged)?;
    io::copy(&mut reader, &mut writer)?;
    writer.flush()?;
    Ok(staged)
}

/// Copies a regular file produced inside `stage_dir` into `dest` strictly through the
/// `cap-std` sandbox root handle.
fn commit_staged(
    sandbox: &PathSandbox,
    stage_root: &Path,
    staged: &Path,
    dest: &Path,
) -> Result<PathBuf, ForgeError> {
    let canon_root = stage_root.canonicalize()?;
    let canon_staged = staged.canonicalize()?;
    if !canon_staged.starts_with(&canon_root) {
        return Err(ForgeError::InvalidParameter(format!(
            "Staged artifact '{}' escaped private staging directory",
            staged.display()
        )));
    }
    let mut reader = File::open(&canon_staged)?;
    if !reader.metadata()?.is_file() {
        return Err(ForgeError::InvalidParameter(format!(
            "Staged artifact '{}' is not a regular file",
            staged.display()
        )));
    }
    let checked_dest = sandbox.check(dest)?;
    let mut writer = sandbox.create_file(&checked_dest)?;
    io::copy(&mut reader, &mut writer)?;
    writer.flush()?;
    Ok(checked_dest)
}

/// Tags `tmp` with ISRC metadata into `out` with FFmpeg via a private staging directory.
/// `tmp` is removed whether or not tagging succeeds, including when a check fails before
/// FFmpeg runs.
pub fn tag_master(
    sandbox: &PathSandbox,
    tmp: &Path,
    out: &Path,
    meta: &ReleaseMetadata,
) -> Result<(), ForgeError> {
    let result = tag_into(sandbox, tmp, out, meta);
    let _ = sandbox.remove_file(tmp);
    result.map(drop)
}

/// Stages `tmp` into a private tempdir, runs FFmpeg there, and commits the tagged file to
/// `out` through the anchored sandbox root handle.
fn tag_into(
    sandbox: &PathSandbox,
    tmp: &Path,
    out: &Path,
    meta: &ReleaseMetadata,
) -> Result<PathBuf, ForgeError> {
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        sandbox.create_dir_all(&sandbox.check(parent)?)?;
    }
    let out = sandbox.check(out)?;
    let stage = tempfile::tempdir()?;
    let stage_in_dir = stage.path().join("in");
    let stage_out_dir = stage.path().join("out");
    std::fs::create_dir_all(&stage_in_dir)?;
    std::fs::create_dir_all(&stage_out_dir)?;
    let staged_in = stage_in(sandbox, tmp, &stage_in_dir)?;
    let out_ext = out
        .extension()
        .and_then(|e| e.to_str())
        .filter(|e| !e.is_empty())
        .unwrap_or("wav");
    let staged_out = stage_out_dir.join(format!("tagged_output.{out_ext}"));
    write_distrokid_metadata(&staged_in, &staged_out, meta)?;
    commit_staged(sandbox, &stage_out_dir, &staged_out, &out)
}

struct StagedWorkerIo {
    _stage: tempfile::TempDir,
    output_dir: PathBuf,
    stage_ref_dir: PathBuf,
    staged_in: PathBuf,
    staged_out: PathBuf,
}

fn prepare_staged_io(
    sandbox: &PathSandbox,
    input: &Path,
    output_dir: &Path,
) -> Result<StagedWorkerIo, ForgeError> {
    let output_dir = sandbox.check(output_dir)?;
    sandbox.create_dir_all(&output_dir)?;
    sandbox.verify_dir(&output_dir)?;

    let stage = tempfile::tempdir()?;
    let stage_in_dir = stage.path().join("in");
    let stage_ref_dir = stage.path().join("ref");
    let staged_out = stage.path().join("out");
    std::fs::create_dir_all(&stage_in_dir)?;
    std::fs::create_dir_all(&stage_ref_dir)?;
    std::fs::create_dir_all(&staged_out)?;
    let staged_in = stage_in(sandbox, input, &stage_in_dir)?;
    Ok(StagedWorkerIo {
        _stage: stage,
        output_dir,
        stage_ref_dir,
        staged_in,
        staged_out,
    })
}

fn stage_optional_ref(
    sandbox: &PathSandbox,
    ref_dir: &Path,
    reference: Option<&Path>,
) -> Result<Option<PathBuf>, ForgeError> {
    reference.map(|p| stage_in(sandbox, p, ref_dir)).transpose()
}

fn commit_staged_named(
    sandbox: &PathSandbox,
    io: &StagedWorkerIo,
    produced: &Path,
    label: &str,
) -> Result<PathBuf, ForgeError> {
    let file_name = produced.file_name().ok_or_else(|| {
        ForgeError::InvalidParameter(format!(
            "Staged {label} file '{}' has no file name",
            produced.display()
        ))
    })?;
    let dest = io.output_dir.join(file_name);
    commit_staged(sandbox, &io.staged_out, produced, &dest)
}

/// Separates `input` into a private staging directory on the backend `config` selects, then
/// copies every produced stem into `stems_dir` through `cap-std` root handles and rewrites
/// the bundle paths to their sandboxed destinations.
pub fn separate_stems_checked(
    sandbox: &PathSandbox,
    input: &Path,
    stems_dir: &Path,
    config: &SeparationConfig,
) -> Result<SeparationReport, ForgeError> {
    let io = prepare_staged_io(sandbox, input, stems_dir)?;
    let mut report = separate_stems(&io.staged_in, &io.staged_out, config)?;
    commit_bundle(sandbox, &io.staged_out, &io.output_dir, &mut report.bundle)?;
    Ok(report)
}

/// Runs neural audio denoising (`DeepFilterNet3` / `Mel-Band-RoFormer`) inside a private
/// staging directory, then commits the produced denoised WAV into `output_dir` through
/// `cap-std` root handles.
pub fn neural_denoise_checked(
    sandbox: &PathSandbox,
    input: &Path,
    output_dir: &Path,
    backend: &str,
    attenuation_limit_db: Option<f64>,
    timeout: Duration,
) -> Result<NeuralDenoiseResult, ForgeError> {
    let worker = resolve_default_local_worker()?;
    let io = prepare_staged_io(sandbox, input, output_dir)?;
    let mut res = run_local_neural_denoise(
        &worker,
        &io.staged_in,
        &io.staged_out,
        backend,
        attenuation_limit_db,
        timeout,
    )?;
    res.denoised = commit_staged_named(sandbox, &io, &res.denoised, "denoised")?;
    Ok(res)
}

/// Runs neural bandwidth restoration (`Apollo` / `FlowHigh` / `AudioSR` / `SonicMaster` / `exciter`)
/// inside a private staging directory and commits the restored file into `output_dir`.
pub fn neural_restore_checked(
    sandbox: &PathSandbox,
    input: &Path,
    output_dir: &Path,
    backend: &str,
    prompt: Option<&str>,
    chunk_secs: Option<f64>,
    timeout: Duration,
) -> Result<NeuralRestoreResult, ForgeError> {
    let io = prepare_staged_io(sandbox, input, output_dir)?;
    let mut res = neural_restore_hybrid(
        &io.staged_in,
        &io.staged_out,
        backend,
        prompt,
        chunk_secs,
        timeout,
    )?;
    res.restored = commit_staged_named(sandbox, &io, &res.restored, "restored")?;
    Ok(res)
}

/// Runs neural vocal rescue (`Seed-VC` / `Resemble Enhance` / `RoFormer`) inside a private
/// staging directory, also staging optional `reference_voice`.
#[allow(clippy::too_many_arguments)]
pub fn vocal_rescue_checked(
    sandbox: &PathSandbox,
    input: &Path,
    output_dir: &Path,
    backend: &str,
    reference_voice: Option<&Path>,
    dereverb: bool,
    de_aspiration: bool,
    timeout: Duration,
) -> Result<VocalRescueResult, ForgeError> {
    let io = prepare_staged_io(sandbox, input, output_dir)?;
    let staged_ref = stage_optional_ref(sandbox, &io.stage_ref_dir, reference_voice)?;
    let mut res = vocal_rescue_hybrid(
        &io.staged_in,
        &io.staged_out,
        backend,
        staged_ref.as_deref(),
        dereverb,
        de_aspiration,
        timeout,
    )?;
    res.rescued = commit_staged_named(sandbox, &io, &res.rescued, "vocal_rescued")?;
    Ok(res)
}

/// Tracks beats, downbeats, and tempo drift (`beat_this` / `librosa` / Rust onset) from a private
/// staged copy of `input`.
pub fn beat_track_checked(
    sandbox: &PathSandbox,
    input: &Path,
    hint_bpm: Option<f64>,
    backend: &str,
    timeout: Duration,
) -> Result<BeatGridReport, ForgeError> {
    let stage = tempfile::tempdir()?;
    let staged_in = stage_in(sandbox, input, stage.path())?;
    track_beats_hybrid(&staged_in, hint_bpm, backend, timeout)
}

/// Evaluates perceptual audio quality (`Audiobox Aesthetics` / `SingMOS` / proxy) from private
/// staged copies of `input` and optional `reference`.
pub fn evaluate_quality_checked(
    sandbox: &PathSandbox,
    input: &Path,
    reference: Option<&Path>,
    vocal: bool,
    backend: &str,
    timeout: Duration,
) -> Result<QualityEvalResult, ForgeError> {
    let stage = tempfile::tempdir()?;
    let in_dir = stage.path().join("in");
    let ref_dir = stage.path().join("ref");
    std::fs::create_dir_all(&in_dir)?;
    std::fs::create_dir_all(&ref_dir)?;
    let staged_in = stage_in(sandbox, input, &in_dir)?;
    let staged_ref = stage_optional_ref(sandbox, &ref_dir, reference)?;
    evaluate_quality_hybrid(&staged_in, staged_ref.as_deref(), vocal, backend, timeout)
}

/// Runs polyphonic audio-to-MIDI transcription (`basic-pitch`) inside a private staging
/// directory, then commits the produced MIDI file into `output_dir` through `cap-std` handles.
pub fn transcribe_midi_checked(
    sandbox: &PathSandbox,
    input: &Path,
    output_dir: &Path,
    timeout: Duration,
) -> Result<MidiTranscribeResult, ForgeError> {
    let worker = resolve_default_local_worker()?;
    let io = prepare_staged_io(sandbox, input, output_dir)?;
    let mut res = run_local_transcribe_midi(&worker, &io.staged_in, &io.staged_out, timeout)?;
    res.midi = commit_staged_named(sandbox, &io, &res.midi, "MIDI")?;
    Ok(res)
}

/// Indexes `.wav` one-shots from a sandboxed `samples_dir` inside a private staging directory
/// and renders a sample-layered drum track without unanchored directory reads in `FORGE_MCP_ROOT`.
pub fn render_drum_layer_checked(
    sandbox: &PathSandbox,
    samples_dir: &Path,
    frames: usize,
    sample_rate: u32,
    hits: &[DrumHit],
    wet_mix: f64,
    dry_drums: &AudioBuffer,
) -> Result<(AudioBuffer, usize), ForgeError> {
    let wav_paths = sandbox.list_files_with_ext(samples_dir, "wav", 256)?;
    let stage = tempfile::tempdir()?;
    for wav in wav_paths {
        let _ = stage_in(sandbox, &wav, stage.path())?;
    }
    let lib = index_drum_sample_library(stage.path())?;
    render_drum_sample_layer(frames, sample_rate, hits, &lib, wet_mix, Some(dry_drums))
}

fn commit_bundle(
    sandbox: &PathSandbox,
    staged_out: &Path,
    stems_dir: &Path,
    bundle: &mut StemBundle,
) -> Result<(), ForgeError> {
    let produced = [
        &mut bundle.vocals,
        &mut bundle.lead_vocals,
        &mut bundle.backing_vocals,
        &mut bundle.male_vocals,
        &mut bundle.female_vocals,
        &mut bundle.vocal_reverb,
        &mut bundle.drums,
        &mut bundle.bass,
        &mut bundle.guitar,
        &mut bundle.electric_guitar,
        &mut bundle.acoustic_guitar,
        &mut bundle.keys,
        &mut bundle.strings,
        &mut bundle.brass,
        &mut bundle.wind,
        &mut bundle.synth,
        &mut bundle.crowd,
        &mut bundle.other,
        &mut bundle.kick,
        &mut bundle.snare,
        &mut bundle.toms,
        &mut bundle.hihat,
        &mut bundle.cymbals,
        &mut bundle.kick_midi,
        &mut bundle.snare_midi,
        &mut bundle.hats_midi,
        &mut bundle.chords_midi,
        &mut bundle.bass_midi,
        &mut bundle.melody_midi,
    ];
    for slot in produced {
        if let Some(staged_path) = slot.as_ref() {
            let file_name = staged_path.file_name().ok_or_else(|| {
                ForgeError::InvalidParameter(format!(
                    "Staged stem '{}' has no file name",
                    staged_path.display()
                ))
            })?;
            let dest = stems_dir.join(file_name);
            let committed = commit_staged(sandbox, staged_out, staged_path, &dest)?;
            *slot = Some(committed);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pre_tag_render_is_removed_when_a_check_fails_before_ffmpeg() {
        let root = tempfile::tempdir().expect("tempdir");
        let sandbox = PathSandbox::new([root.path().to_path_buf()]).expect("sandbox");
        // Checked before it exists, as the server does, so the path matches the canonical root.
        let tmp = sandbox
            .check(&root.path().join("master.pre_tag.wav"))
            .expect("tmp");
        std::fs::write(&tmp, b"render").expect("write");
        // A regular file where the output's parent directory should be fails `create_dir_all`.
        std::fs::write(root.path().join("blocker"), b"").expect("write");
        let out = sandbox
            .check(&root.path().join("blocker"))
            .expect("out")
            .join("master.wav");

        assert!(tag_master(&sandbox, &tmp, &out, &meta()).is_err());
        assert!(!tmp.exists(), "pre-tag render was left behind");
    }

    #[cfg(unix)]
    #[test]
    fn cleanup_removes_a_swapped_in_symlink_not_its_target() {
        let root = tempfile::tempdir().expect("tempdir");
        let sandbox = PathSandbox::new([root.path().to_path_buf()]).expect("sandbox");
        let victim = root.path().join("victim.wav");
        std::fs::write(&victim, b"keep me").expect("write");
        let tmp = sandbox
            .check(&root.path().join("master.pre_tag.wav"))
            .expect("tmp");
        std::os::unix::fs::symlink(&victim, &tmp).expect("symlink");
        std::fs::write(root.path().join("blocker"), b"").expect("write");
        let out = sandbox
            .check(&root.path().join("blocker"))
            .expect("out")
            .join("master.wav");

        assert!(tag_master(&sandbox, &tmp, &out, &meta()).is_err());
        assert!(
            tmp.symlink_metadata().is_err(),
            "the swapped-in link was left behind"
        );
        assert_eq!(std::fs::read(&victim).expect("victim"), b"keep me");
    }

    fn meta() -> ReleaseMetadata {
        ReleaseMetadata {
            title: "T".into(),
            artist: "A".into(),
            album: "Single".into(),
            isrc: "USS1Z2600001".into(),
            bpm: None,
            key: None,
            year: None,
        }
    }
}
