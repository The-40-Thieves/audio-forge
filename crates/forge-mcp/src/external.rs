//! Steps that hand path names to code the server cannot anchor at a root handle: FFmpeg
//! (ISRC tagging) and the stem-separation worker, which open files by name themselves.
//!
//! These are narrowed, not closed (THE-1600). The server creates the directories through
//! the root handle, re-resolves and re-checks every path immediately before the call, and
//! after it returns opens every produced file through the root handle (by root-relative
//! path, so a swapped symlink fails) before reporting success.
//!
//! Residual window: between the final re-check and the moment the external code opens a
//! path, a local process with write access inside the root could still swap a component
//! for a symlink. The post-run verification then fails the call, but a write that already
//! went through the swapped link may have landed outside the root. Closing this needs the
//! external code to accept open handles (for FFmpeg, `pipe:` I/O; for the worker, a
//! handle-based API).

use crate::sandbox::PathSandbox;
use forge_core::{ForgeError, StemBundle};
use forge_reaper::{ReleaseMetadata, write_distrokid_metadata};
use forge_worker::{SeparationConfig, SeparationReport, separate_stems};
use std::path::{Path, PathBuf};

/// Re-checks `path` against the sandbox and confirms, through the root handle, that it is
/// an existing regular file. Returns the freshly resolved path.
fn recheck_file(sandbox: &PathSandbox, path: &Path) -> Result<PathBuf, ForgeError> {
    let path = sandbox.check(path)?;
    sandbox.verify_file(&path)?;
    Ok(path)
}

/// Tags `tmp` with ISRC metadata into `out` with FFmpeg. `tmp` is removed whether or not
/// tagging succeeds, including when a check fails before FFmpeg runs.
pub fn tag_master(
    sandbox: &PathSandbox,
    tmp: &Path,
    out: &Path,
    meta: &ReleaseMetadata,
) -> Result<(), ForgeError> {
    let tagged = tag_into(sandbox, tmp, out, meta);
    let _ = sandbox.remove_file(tmp);
    sandbox.verify_file(&tagged?)
}

/// Runs FFmpeg on the re-checked paths and returns the re-checked output path.
fn tag_into(
    sandbox: &PathSandbox,
    tmp: &Path,
    out: &Path,
    meta: &ReleaseMetadata,
) -> Result<PathBuf, ForgeError> {
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        sandbox.create_dir_all(&sandbox.check(parent)?)?;
    }
    let tmp = recheck_file(sandbox, tmp)?;
    let out = sandbox.check(out)?;
    write_distrokid_metadata(&tmp, &out, meta)?;
    Ok(out)
}

/// Separates `input` into `stems_dir` on the backend `config` selects, then verifies every
/// stem the worker reports. The report also names the backend that ran and why others were
/// passed over.
pub fn separate_stems_checked(
    sandbox: &PathSandbox,
    input: &Path,
    stems_dir: &Path,
    config: &SeparationConfig,
) -> Result<SeparationReport, ForgeError> {
    let stems_dir = sandbox.check(stems_dir)?;
    sandbox.create_dir_all(&stems_dir)?;
    sandbox.verify_dir(&stems_dir)?;
    let input = recheck_file(sandbox, input)?;
    let report = separate_stems(&input, &stems_dir, config)?;
    verify_bundle(sandbox, &report.bundle)?;
    Ok(report)
}

fn verify_bundle(sandbox: &PathSandbox, bundle: &StemBundle) -> Result<(), ForgeError> {
    let produced = [
        &bundle.vocals,
        &bundle.drums,
        &bundle.bass,
        &bundle.guitar,
        &bundle.keys,
        &bundle.other,
        &bundle.kick_midi,
        &bundle.snare_midi,
        &bundle.hats_midi,
        &bundle.chords_midi,
        &bundle.bass_midi,
        &bundle.melody_midi,
    ];
    for path in produced.into_iter().flatten() {
        recheck_file(sandbox, path)?;
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
