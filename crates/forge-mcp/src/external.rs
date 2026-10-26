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

/// Tags `tmp` with ISRC metadata into `out` with FFmpeg, then removes `tmp`.
pub fn tag_master(
    sandbox: &PathSandbox,
    tmp: &Path,
    out: &Path,
    meta: &ReleaseMetadata,
) -> Result<(), ForgeError> {
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        sandbox.create_dir_all(&sandbox.check(parent)?)?;
    }
    let tmp = recheck_file(sandbox, tmp)?;
    let out = sandbox.check(out)?;
    let tagged = write_distrokid_metadata(&tmp, &out, meta);
    let _ = sandbox.remove_file(&tmp);
    tagged?;
    sandbox.verify_file(&out)
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
