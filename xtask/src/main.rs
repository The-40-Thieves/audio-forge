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
        "install" => {
            let is_release = args.iter().any(|a| a == "--release");
            let install_all = args.iter().any(|a| a == "--all");
            let custom_dest = args
                .windows(2)
                .find(|w| w[0] == "--dest")
                .map(|w| PathBuf::from(&w[1]));

            let targets: Vec<&str> = if install_all {
                STANDALONE_PLUGINS.iter().map(|&(p, _)| p).collect()
            } else if let Some(pkg_arg) = args.get(1) {
                if pkg_arg.starts_with("--") {
                    eprintln!("Usage: cargo run --package xtask -- install <plugin-name|--all> [--release] [--dest <dir>]");
                    return Ok(());
                }
                vec![pkg_arg.as_str()]
            } else {
                eprintln!("Usage: cargo run --package xtask -- install <plugin-name|--all> [--release] [--dest <dir>]");
                return Ok(());
            };

            let (vst3_dir, clap_dir) = resolve_install_dirs(custom_dest.as_deref());
            std::fs::create_dir_all(&vst3_dir)?;
            std::fs::create_dir_all(&clap_dir)?;

            for pkg in targets {
                bundle_single_package(pkg, is_release)?;
                let bundled_root = PathBuf::from("target").join("bundled");
                let src_vst3 = bundled_root.join(format!("{pkg}.vst3"));
                let src_clap = bundled_root.join(format!("{pkg}.clap"));

                if src_vst3.exists() {
                    let dst_vst3 = vst3_dir.join(format!("{pkg}.vst3"));
                    copy_dir_recursive(&src_vst3, &dst_vst3)?;
                    println!("Installed VST3 -> {}", dst_vst3.display());
                }
                if src_clap.exists() {
                    let dst_clap = clap_dir.join(format!("{pkg}.clap"));
                    std::fs::copy(&src_clap, &dst_clap)?;
                    println!("Installed CLAP -> {}", dst_clap.display());
                }
            }
            Ok(())
        }
        _ => nih_plug_xtask::main_with_args("cargo xtask", args),
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
