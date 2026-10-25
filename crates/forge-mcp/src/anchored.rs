//! File I/O anchored at the sandbox root handles (THE-1600).
//!
//! [`PathSandbox::resolve`] and [`PathSandbox::check`] validate a path when a request
//! arrives. Opening that path by name later would leave a window in which a local process
//! with write access inside the root could swap a directory or file for a symlink that
//! points outside it. Every open, create, `mkdir` and remove here instead goes through the
//! `cap_std::fs::Dir` of the root that contains the path, by root-relative path. `cap-std`
//! resolves each component relative to that handle and refuses `..` and symlinks that would
//! leave it, so a swapped path fails to open instead of being followed out of the root.
//!
//! The library crates receive the opened handles (`std::fs::File` from `into_std`) through
//! their reader- and writer-based APIs.

use crate::sandbox::PathSandbox;
use cap_std::fs::Dir;
use forge_core::{AudioBuffer, ForgeError};
use forge_reaper::{JSFX_PLUGINS, ReaperProjectBuilder};
use forge_theory::{NoteEvent, write_midi_to_writer};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

impl PathSandbox {
    /// Runs `op` with the root handle containing `path` and the root-relative path.
    fn anchored<T>(
        &self,
        path: &Path,
        op: impl FnOnce(&Dir, &Path) -> io::Result<T>,
    ) -> Result<T, ForgeError> {
        let (dir, rel) = self.anchor(path)?;
        test_hook::before_open(&rel);
        op(dir, &rel).map_err(|e| self.anchored_failure(path, &e))
    }

    fn anchored_failure(&self, path: &Path, err: &io::Error) -> ForgeError {
        if err.kind() == io::ErrorKind::NotFound {
            return ForgeError::AudioIo {
                path: path.display().to_string(),
                message: "File does not exist".to_string(),
            };
        }
        self.denied(
            &path.display().to_string(),
            &format!("cannot open it through the sandbox root handle ({err})"),
        )
    }

    /// Opens an existing file for reading.
    pub fn open_read(&self, path: &Path) -> Result<File, ForgeError> {
        self.anchored(path, |dir, rel| {
            dir.open(rel).map(cap_std::fs::File::into_std)
        })
    }

    /// Creates (or truncates) a file for writing, creating its parent directories first.
    pub fn create_file(&self, path: &Path) -> Result<File, ForgeError> {
        self.anchored(path, |dir, rel| {
            if let Some(parent) = rel.parent().filter(|p| !p.as_os_str().is_empty()) {
                dir.create_dir_all(parent)?;
            }
            dir.create(rel).map(cap_std::fs::File::into_std)
        })
    }

    /// Creates a directory and its missing parents. The root itself always exists.
    pub fn create_dir_all(&self, path: &Path) -> Result<(), ForgeError> {
        self.anchored(path, |dir, rel| {
            if rel.as_os_str().is_empty() {
                Ok(())
            } else {
                dir.create_dir_all(rel)
            }
        })
    }

    /// Confirms `path` is a regular file reachable inside its root.
    pub fn verify_file(&self, path: &Path) -> Result<(), ForgeError> {
        self.anchored(path, |dir, rel| {
            if dir.open(rel)?.metadata()?.is_file() {
                Ok(())
            } else {
                Err(io::Error::other("not a regular file"))
            }
        })
    }

    /// Confirms `path` is a directory reachable inside its root.
    pub fn verify_dir(&self, path: &Path) -> Result<(), ForgeError> {
        self.anchored(path, |dir, rel| {
            if rel.as_os_str().is_empty() {
                Ok(())
            } else {
                dir.open_dir(rel).map(drop)
            }
        })
    }

    /// Removes a file inside its root.
    pub fn remove_file(&self, path: &Path) -> Result<(), ForgeError> {
        self.anchored(path, |dir, rel| dir.remove_file(rel))
    }

    /// Decodes an audio file opened through the root handle. Non-WAV input is decoded by
    /// FFmpeg from a private copy of the opened file, never from `path`.
    pub fn load_audio(&self, path: &Path) -> Result<AudioBuffer, ForgeError> {
        AudioBuffer::load_from_reader(self.open_read(path)?, path)
    }

    /// Writes `buf` as a WAV file created through the root handle.
    pub fn save_wav(&self, buf: &AudioBuffer, path: &Path, bits: u16) -> Result<(), ForgeError> {
        let file = BufWriter::new(self.create_file(path)?);
        buf.save_wav_to_writer(file, bits, &path.display().to_string())
    }

    /// Writes `bytes` to a file created through the root handle.
    pub fn write_bytes(&self, path: &Path, bytes: &[u8]) -> Result<(), ForgeError> {
        let mut file = self.create_file(path)?;
        file.write_all(bytes)?;
        file.flush()?;
        Ok(())
    }

    /// Writes `value` as pretty-printed JSON to a file created through the root handle.
    pub fn write_json<T: serde::Serialize>(
        &self,
        path: &Path,
        value: &T,
    ) -> Result<(), ForgeError> {
        self.write_bytes(path, serde_json::to_string_pretty(value)?.as_bytes())
    }

    /// Writes a Standard MIDI File created through the root handle.
    pub fn write_midi(
        &self,
        path: &Path,
        bpm: f64,
        ticks_per_beat: u16,
        tracks: &[(&str, &[NoteEvent])],
    ) -> Result<(), ForgeError> {
        let file = BufWriter::new(self.create_file(path)?);
        write_midi_to_writer(file, bpm, ticks_per_beat, tracks)
    }

    /// Writes a REAPER project created through the root handle.
    pub fn save_rpp(&self, rpp: &ReaperProjectBuilder, path: &Path) -> Result<(), ForgeError> {
        rpp.write_rpp(BufWriter::new(self.create_file(path)?))
    }

    /// Installs the bundled JSFX plugins into `dir` through the root handle. Each file name
    /// is checked like any other derived destination before it is written.
    pub fn install_jsfx(&self, dir: &Path) -> Result<Vec<PathBuf>, ForgeError> {
        self.create_dir_all(dir)?;
        let mut installed = Vec::with_capacity(JSFX_PLUGINS.len());
        for (name, source) in JSFX_PLUGINS {
            let path = self.check(&dir.join(name))?;
            self.write_bytes(&path, source.as_bytes())?;
            installed.push(path);
        }
        Ok(installed)
    }
}

/// A test-only hook that runs between resolving a path and opening it, so tests can swap
/// a path for a symlink inside that window. Compiled out of non-test builds.
#[cfg(test)]
pub mod test_hook {
    use std::cell::RefCell;
    use std::path::Path;

    type Hook = Box<dyn FnMut(&Path)>;

    thread_local! {
        static BEFORE_OPEN: RefCell<Option<Hook>> = RefCell::new(None);
    }

    /// Installs `hook` for the current thread; it sees each root-relative path just
    /// before the anchored open.
    #[cfg_attr(not(unix), allow(dead_code))]
    pub fn set(hook: impl FnMut(&Path) + 'static) {
        BEFORE_OPEN.with(|h| *h.borrow_mut() = Some(Box::new(hook)));
    }

    #[cfg_attr(not(unix), allow(dead_code))]
    pub fn clear() {
        BEFORE_OPEN.with(|h| *h.borrow_mut() = None);
    }

    pub(super) fn before_open(rel: &Path) {
        BEFORE_OPEN.with(|h| {
            if let Some(hook) = h.borrow_mut().as_mut() {
                hook(rel);
            }
        });
    }
}

#[cfg(not(test))]
mod test_hook {
    #[inline]
    pub(super) fn before_open(_rel: &std::path::Path) {}
}
