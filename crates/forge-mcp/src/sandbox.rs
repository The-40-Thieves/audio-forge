//! Filesystem confinement for MCP tool arguments.
//!
//! Every path an MCP client passes is resolved against a set of allowed roots before any
//! file is read or written. Relative paths resolve against the first root; `..` components
//! are rejected outright, and the resolved path (with symlinks in its existing ancestors
//! followed) must stay inside one of the roots.

use forge_core::ForgeError;
use std::ffi::OsString;
use std::io;
use std::path::{Component, Path, PathBuf};

/// Environment variable listing the allowed roots, separated like `PATH`.
pub const ROOT_ENV: &str = "FORGE_MCP_ROOT";

#[derive(Debug, Clone)]
pub struct PathSandbox {
    roots: Vec<PathBuf>,
}

impl PathSandbox {
    /// Creates a sandbox over `roots`. Each root must exist; it is canonicalized.
    pub fn new<I: IntoIterator<Item = PathBuf>>(roots: I) -> io::Result<Self> {
        let roots = roots
            .into_iter()
            .map(|r| r.canonicalize())
            .collect::<io::Result<Vec<_>>>()?;
        if roots.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "at least one sandbox root is required",
            ));
        }
        Ok(Self { roots })
    }

    /// Roots from `FORGE_MCP_ROOT`, or the current working directory when it is unset or empty.
    pub fn from_env() -> io::Result<Self> {
        Self::from_env_value(std::env::var_os(ROOT_ENV))
    }

    fn from_env_value(value: Option<OsString>) -> io::Result<Self> {
        let roots: Vec<PathBuf> = value
            .map(|v| {
                std::env::split_paths(&v)
                    .filter(|p| !p.as_os_str().is_empty())
                    .collect()
            })
            .unwrap_or_default();
        if roots.is_empty() {
            Self::new([std::env::current_dir()?])
        } else {
            Self::new(roots)
        }
    }

    /// The canonical roots, for diagnostics.
    pub fn roots(&self) -> impl Iterator<Item = &Path> {
        self.roots.iter().map(PathBuf::as_path)
    }

    /// Resolves a client-supplied path to an absolute path inside the sandbox.
    pub fn resolve(&self, raw: &str) -> Result<PathBuf, ForgeError> {
        if raw.trim().is_empty() {
            return Err(ForgeError::InvalidParameter("Empty path".into()));
        }
        let requested = Path::new(raw);
        if requested
            .components()
            .any(|c| matches!(c, Component::ParentDir))
        {
            return Err(self.denied(raw, "'..' components are not allowed"));
        }

        let joined = if requested.is_absolute() {
            requested.to_path_buf()
        } else {
            self.roots[0].join(requested)
        };
        let resolved = canonicalize_existing_prefix(&joined).map_err(ForgeError::Io)?;

        if self.roots.iter().any(|root| resolved.starts_with(root)) {
            Ok(simplify(resolved))
        } else {
            Err(self.denied(raw, "path is outside the allowed roots"))
        }
    }

    fn denied(&self, raw: &str, reason: &str) -> ForgeError {
        let roots: Vec<String> = self
            .roots
            .iter()
            .map(|r| simplify(r.clone()).display().to_string())
            .collect();
        ForgeError::InvalidParameter(format!(
            "Path '{raw}' rejected: {reason} (allowed roots: {}; set {ROOT_ENV} to change)",
            roots.join(", ")
        ))
    }
}

/// Canonicalizes the longest existing ancestor of `path` and appends the rest unchanged.
/// Output paths usually do not exist yet, so plain `canonicalize` cannot be used.
fn canonicalize_existing_prefix(path: &Path) -> io::Result<PathBuf> {
    for ancestor in path.ancestors() {
        if ancestor.as_os_str().is_empty() || !ancestor.exists() {
            continue;
        }
        let mut resolved = ancestor.canonicalize()?;
        if let Ok(rest) = path.strip_prefix(ancestor) {
            for component in rest.components() {
                if let Component::Normal(part) = component {
                    resolved.push(part);
                }
            }
        }
        return Ok(resolved);
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!("no existing ancestor for '{}'", path.display()),
    ))
}

/// Drops the Windows verbatim prefix (`\\?\C:\...`) from canonical paths so they stay usable
/// by tools that do not understand it (REAPER, ffmpeg). UNC verbatim paths are left as is.
fn simplify(path: PathBuf) -> PathBuf {
    #[cfg(windows)]
    {
        let s = path.to_string_lossy();
        if let Some(rest) = s.strip_prefix(r"\\?\") {
            let bytes = rest.as_bytes();
            if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
                return PathBuf::from(rest);
            }
        }
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sandbox() -> (tempfile::TempDir, PathSandbox) {
        let dir = tempfile::tempdir().expect("tempdir");
        let sb = PathSandbox::new([dir.path().to_path_buf()]).expect("sandbox");
        (dir, sb)
    }

    #[test]
    fn relative_paths_resolve_inside_root() {
        let (dir, sb) = sandbox();
        let resolved = sb.resolve("mixes/out.wav").expect("allowed");
        let root = simplify(dir.path().canonicalize().unwrap());
        assert!(resolved.is_absolute());
        assert!(
            resolved.starts_with(&root),
            "{resolved:?} not under {root:?}"
        );
        assert!(resolved.ends_with("mixes/out.wav"));
    }

    #[test]
    fn absolute_paths_inside_root_are_allowed() {
        let (dir, sb) = sandbox();
        let inside = dir.path().join("a").join("b.wav");
        assert!(sb.resolve(&inside.to_string_lossy()).is_ok());
    }

    #[test]
    fn parent_dir_traversal_is_rejected() {
        let (_dir, sb) = sandbox();
        for raw in ["../escape.wav", "a/../../escape.wav", "a/.."] {
            let err = sb.resolve(raw).expect_err(raw);
            assert!(err.to_string().contains("'..'"), "{raw}: {err}");
        }
    }

    #[test]
    fn absolute_paths_outside_root_are_rejected() {
        let (_dir, sb) = sandbox();
        let other = tempfile::tempdir().expect("tempdir");
        let outside = other.path().join("victim.wav");
        let err = sb.resolve(&outside.to_string_lossy()).expect_err("outside");
        assert!(
            err.to_string().contains("outside the allowed roots"),
            "{err}"
        );
    }

    #[test]
    fn empty_path_is_rejected() {
        let (_dir, sb) = sandbox();
        assert!(sb.resolve("   ").is_err());
    }

    #[test]
    fn multiple_roots_from_env_value() {
        let a = tempfile::tempdir().expect("tempdir");
        let b = tempfile::tempdir().expect("tempdir");
        let joined = std::env::join_paths([a.path(), b.path()]).expect("join");
        let sb = PathSandbox::from_env_value(Some(joined)).expect("sandbox");
        assert!(
            sb.resolve(&b.path().join("x.wav").to_string_lossy())
                .is_ok()
        );
        assert_eq!(sb.roots().count(), 2);
    }

    #[cfg(unix)]
    #[test]
    fn symlink_escaping_root_is_rejected() {
        let (dir, sb) = sandbox();
        let other = tempfile::tempdir().expect("tempdir");
        std::os::unix::fs::symlink(other.path(), dir.path().join("link")).expect("symlink");
        assert!(sb.resolve("link/victim.wav").is_err());
    }
}
