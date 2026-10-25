//! `xtask` — Individual & Multi-Plugin VST3 / CLAP Bundler & System Installer for ForgeAudio.
//!
//! Supported commands:
//! - `cargo run --package xtask -- bundle <plugin-name> [--release]`
//!   Bundles any single standalone plugin (e.g. `forge-opensoothe`) or the all-in-one `forge-plug`.
//! - `cargo run --package xtask -- bundle-all [--release]`
//!   Bundles all 10 individual standalone VST3 & CLAP plugins into `target/bundled/`.
//! - `cargo run --package xtask -- install <plugin-name|--all> [--release] [--dest <folder>]`
//!   Bundles and installs an individual plugin (or `--all` 10 standalone plugins) into the DAW VST3 & CLAP directories.
//! - `cargo run --package xtask -- list`
//!   Lists all 10 standalone VST3/CLAP plugins and their descriptions.

use std::fmt;
use std::path::{Path, PathBuf};

pub const STANDALONE_PLUGINS: [(&str, &str); 10] = [
    (
        "forge-opensoothe",
        "Forge OpenSoothe De-Fizz (2048-pt OLA STFT Dynamic Spectral Resonance & AI Fizz Suppressor)",
    ),
    (
        "forge-openspacer",
        "Forge OpenSpacer Unmask (4-Channel Sidechain Dynamic Spectral Unmasker)",
    ),
    (
        "forge-vocal-surgical",
        "Forge Vocal Surgical Strip (Zero-Latency Mouth De-Click, Plosive Duck, De-Esser & Breath Gate)",
    ),
    (
        "forge-submono-tape",
        "Forge SubMono & Console Tape (<160 Hz Mono-Maker, Harmonic Sub Reinforcer & Console8/ChowTape)",
    ),
    (
        "forge-master-limiter",
        "Forge Master True-Peak Limiter (4x Inter-Sample True-Peak -1.0 dBTP Brickwall Limiter)",
    ),
    (
        "forge-drum-trigger",
        "Forge Drum Trigger & Theory MIDI (Audio-to-MIDI 3-Band Drum Replacer C1/D1/F#1 + Neo-Soul Voicer)",
    ),
    (
        "forge-phase-eq",
        "Forge PhaseAlign & 18-Band EQ (Sample-Accurate Phase Aligner, 18-Band AI De-Harsh EQ & Genre Targets)",
    ),
    (
        "forge-spliteq",
        "Forge SplitEQ Transient & Tonal (3-Band Structural Transient Attack vs. Tonal Sustain Shaper)",
    ),
    (
        "forge-density-clipper",
        "Forge Density & 3-Band ADAA Clipper (Upward Low-Level Density Compressor + 2x ADAA Soft Clipper)",
    ),
    (
        "forge-spaceverb",
        "Forge SpaceVerb FDN & Neural Preamp (Dilated WaveNet Preamp + 8x8 Hadamard FDN Shimmer/Ducking Reverb)",
    ),
];

fn main() -> nih_plug_xtask::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let subcommand = args.first().map(String::as_str).unwrap_or("");

    match subcommand {
        "list" => {
            println!("ForgeAudio Standalone VST3 & CLAP Plugins (10 Individual Packages):");
            for (idx, (pkg, desc)) in STANDALONE_PLUGINS.iter().enumerate() {
                println!(
                    "  {:2}. {:22} -> {}.vst3 / {}.clap\n      {}",
                    idx + 1,
                    pkg,
                    pkg,
                    pkg,
                    desc
                );
            }
            Ok(())
        }
        "bundle-all" => {
            let is_release = args.iter().any(|a| a == "--release");
            for &(pkg, _) in &STANDALONE_PLUGINS {
                bundle_single_package(pkg, is_release)?;
            }
            Ok(())
        }
        "install" => run_install(&args[1..]),
        _ => nih_plug_xtask::main_with_args("cargo xtask", args),
    }
}

const INSTALL_USAGE: &str =
    "Usage: cargo run --package xtask -- install <plugin-name|--all> [--release] [--dest <dir>]";

/// A malformed `install` command line. Returned from `main`, so it exits nonzero.
#[derive(Debug, PartialEq, Eq)]
struct UsageError(String);

impl fmt::Display for UsageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}
{INSTALL_USAGE}",
            self.0
        )
    }
}

impl std::error::Error for UsageError {}

/// Parsed arguments of the `install` subcommand.
#[derive(Debug, PartialEq, Eq)]
struct InstallArgs {
    targets: Vec<String>,
    is_release: bool,
    dest: Option<PathBuf>,
}

/// Parses the arguments that follow `install`.
///
/// Exactly one of a plugin name or `--all` is required, `--dest` must be followed
/// by a directory, and any other flag is rejected.
fn parse_install_args(args: &[String]) -> Result<InstallArgs, UsageError> {
    let mut is_release = false;
    let mut install_all = false;
    let mut dest = None;
    let mut plugin: Option<&str> = None;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--release" => is_release = true,
            "--all" => install_all = true,
            "--dest" => dest = Some(take_dest_value(iter.next())?),
            flag if flag.starts_with("--") => {
                return Err(UsageError(format!("unknown argument `{flag}`")));
            }
            name => plugin = Some(take_plugin_name(plugin, name)?),
        }
    }
    let targets = resolve_targets(install_all, plugin)?;
    Ok(InstallArgs {
        targets,
        is_release,
        dest,
    })
}

fn take_dest_value(value: Option<&String>) -> Result<PathBuf, UsageError> {
    match value {
        Some(dir) if !dir.is_empty() && !dir.starts_with("--") => Ok(PathBuf::from(dir)),
        _ => Err(UsageError("`--dest` requires a directory".to_string())),
    }
}

fn take_plugin_name<'a>(previous: Option<&str>, name: &'a str) -> Result<&'a str, UsageError> {
    match previous {
        Some(first) => Err(UsageError(format!(
            "expected one plugin name, got `{first}` and `{name}`"
        ))),
        None => Ok(name),
    }
}

fn resolve_targets(install_all: bool, plugin: Option<&str>) -> Result<Vec<String>, UsageError> {
    match (install_all, plugin) {
        (true, None) => Ok(STANDALONE_PLUGINS
            .iter()
            .map(|&(p, _)| p.to_string())
            .collect()),
        (false, Some(name)) => Ok(vec![name.to_string()]),
        (true, Some(name)) => Err(UsageError(format!(
            "`--all` cannot be combined with a plugin name (`{name}`)"
        ))),
        (false, None) => Err(UsageError("missing plugin name or `--all`".to_string())),
    }
}

fn run_install(args: &[String]) -> nih_plug_xtask::Result<()> {
    let install = parse_install_args(args)?;
    let (vst3_dir, clap_dir) = resolve_install_dirs(install.dest.as_deref());
    std::fs::create_dir_all(&vst3_dir)?;
    std::fs::create_dir_all(&clap_dir)?;

    let bundled_root = PathBuf::from("target").join("bundled");
    for pkg in &install.targets {
        bundle_single_package(pkg, install.is_release)?;
        for (ext, dir) in [("vst3", &vst3_dir), ("clap", &clap_dir)] {
            let src = bundled_root.join(format!("{pkg}.{ext}"));
            if src.exists() {
                let dst = dir.join(format!("{pkg}.{ext}"));
                copy_bundle(&src, &dst)?;
                println!("Installed {} -> {}", ext.to_uppercase(), dst.display());
            }
        }
    }
    Ok(())
}

/// Copies a plugin bundle, which is a directory on macOS (and for VST3 everywhere)
/// but a single file for CLAP on Windows and Linux.
fn copy_bundle(src: &Path, dst: &Path) -> std::io::Result<()> {
    if src.is_dir() {
        copy_dir_recursive(src, dst)
    } else {
        std::fs::copy(src, dst).map(|_| ())
    }
}

fn bundle_single_package(pkg: &str, is_release: bool) -> nih_plug_xtask::Result<()> {
    let mut bundle_args = vec!["bundle".to_string(), pkg.to_string()];
    if is_release {
        bundle_args.push("--release".to_string());
    }
    nih_plug_xtask::main_with_args("cargo xtask", bundle_args)
}

fn resolve_install_dirs(custom_dest: Option<&Path>) -> (PathBuf, PathBuf) {
    if let Some(base) = custom_dest {
        return (base.join("VST3"), base.join("CLAP"));
    }
    #[cfg(target_os = "windows")]
    {
        if let Ok(local_appdata) = std::env::var("LOCALAPPDATA") {
            let base = PathBuf::from(local_appdata).join("Programs").join("Common");
            return (base.join("VST3"), base.join("CLAP"));
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Ok(home) = std::env::var("HOME") {
            let base = PathBuf::from(home).join("Library/Audio/Plug-Ins");
            return (base.join("VST3"), base.join("CLAP"));
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        let base = PathBuf::from(home);
        return (base.join(".vst3"), base.join(".clap"));
    }
    (
        PathBuf::from("installed-plugins/VST3"),
        PathBuf::from("installed-plugins/CLAP"),
    )
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let dest_path = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_recursive(&entry.path(), &dest_path)?;
        } else {
            std::fs::copy(entry.path(), dest_path)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn valid_install_call_parses() {
        let parsed =
            parse_install_args(&args(&["forge-spliteq", "--release", "--dest", "out"])).unwrap();
        assert_eq!(
            parsed,
            InstallArgs {
                targets: vec!["forge-spliteq".to_string()],
                is_release: true,
                dest: Some(PathBuf::from("out")),
            }
        );
    }

    #[test]
    fn install_all_selects_every_standalone_plugin() {
        let parsed = parse_install_args(&args(&["--all"])).unwrap();
        assert_eq!(parsed.targets.len(), STANDALONE_PLUGINS.len());
        assert!(!parsed.is_release);
        assert_eq!(parsed.dest, None);
    }

    #[test]
    fn missing_dest_value_is_an_error() {
        let err = parse_install_args(&args(&["forge-spliteq", "--dest"])).unwrap_err();
        assert!(err.0.contains("--dest"), "{err}");
    }

    #[test]
    fn dest_followed_by_a_flag_is_an_error() {
        let err = parse_install_args(&args(&["--all", "--dest", "--release"])).unwrap_err();
        assert!(err.0.contains("--dest"), "{err}");
    }

    #[test]
    fn unknown_argument_is_an_error() {
        let err = parse_install_args(&args(&["forge-spliteq", "--relase"])).unwrap_err();
        assert!(err.0.contains("--relase"), "{err}");
    }

    #[test]
    fn missing_target_is_an_error() {
        assert!(parse_install_args(&args(&["--release"])).is_err());
        assert!(parse_install_args(&[]).is_err());
    }

    #[test]
    fn all_together_with_a_plugin_name_is_an_error() {
        assert!(parse_install_args(&args(&["--all", "forge-spliteq"])).is_err());
        assert!(parse_install_args(&args(&["forge-spliteq", "--all"])).is_err());
    }

    #[test]
    fn two_plugin_names_are_an_error() {
        let err = parse_install_args(&args(&["forge-spliteq", "forge-opensoothe"])).unwrap_err();
        assert!(
            err.0.contains("forge-spliteq") && err.0.contains("forge-opensoothe"),
            "{err}"
        );
    }

    #[test]
    fn usage_error_fails_main_with_usage_text() {
        let err: nih_plug_xtask::Result<()> = run_install(&args(&["--bogus"]));
        let text = err.unwrap_err().to_string();
        assert!(
            text.contains("--bogus") && text.contains(INSTALL_USAGE),
            "{text}"
        );
    }

    #[test]
    fn copy_bundle_copies_directory_bundles_recursively() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("forge-test.clap");
        let macos = src.join("Contents").join("MacOS");
        std::fs::create_dir_all(&macos).unwrap();
        std::fs::write(src.join("Contents").join("Info.plist"), "plist").unwrap();
        std::fs::write(macos.join("forge-test"), "binary").unwrap();

        let dst = tmp.path().join("installed").join("forge-test.clap");
        copy_bundle(&src, &dst).unwrap();

        let copied = dst.join("Contents");
        assert_eq!(
            std::fs::read_to_string(copied.join("Info.plist")).unwrap(),
            "plist"
        );
        let binary = copied.join("MacOS").join("forge-test");
        assert_eq!(std::fs::read_to_string(binary).unwrap(), "binary");
    }

    #[test]
    fn copy_bundle_copies_single_file_bundles() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("forge-test.clap");
        std::fs::write(&src, "clap").unwrap();
        let dst = tmp.path().join("copy.clap");
        copy_bundle(&src, &dst).unwrap();
        assert_eq!(std::fs::read_to_string(dst).unwrap(), "clap");
    }
}
