# ForgeAudio (`audio-forge`)

**Open-Source Hybrid Music Production, AI Track Restoration, Vocal Surgery, Music Theory & REAPER 7.82 Mastering Suite**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-2021%20Edition-orange.svg)](Cargo.toml)
[![VST3 / CLAP](https://img.shields.io/badge/Plugins-VST3%20%7C%20CLAP%20%7C%20JSFX-purple.svg)](crates/forge-plug/)
[![REAPER](https://img.shields.io/badge/REAPER-7.82%2B%20Native%20JSFX%20%26%20RPP-green.svg)](reaper-jsfx/)

ForgeAudio (`forge` CLI + `forge-mcp` Model Context Protocol server + `forge-plug` native **VST3 / CLAP** plugin suite + native REAPER `JSFX` DSP plugins) is a zero-bloat **Rust + REAPER 7.82 + Hybrid GPU Worker** audio engineering ecosystem built for **both live studio musicians and AI/hybrid producers** (Suno, Udio, YuE2-3B, ACE-Step 1.5, Google Lyria 3.5).

---

## Why ForgeAudio? (Commercial vs. ForgeAudio Open-Source Stack)

| Commercial Industry Standard | Price | ForgeAudio (`audio-forge`) Equivalent | Runtime & Speed |
| :--- | :--- | :--- | :--- |
| **oeksound Soothe2** | \$209 | `forge-dsp::spectral_defizz` + `Forge_OpenSoothe_DeFizz.jsfx` (2048-pt OLA STFT dynamic spectral resonance & AI metallic fizz suppressor) | Pure Rust SIMD (`>120x` real-time) + REAPER JSFX (`0%` idle CPU) |
| **Wavesfactory Trackspacer** | \$59 | `forge-dsp::spectral_unmask` + `Forge_OpenSpacer_Unmask.jsfx` (32-band / STFT sidechain dynamic frequency carving: Vocal→Music, Kick→Bass) | Pure Rust + REAPER 4-channel sidechain JSFX |
| **Sound Radix Auto-Align 2** | \$199 | `forge-dsp::align_phase_gcc_phat` (Sub-sample Generalized Cross-Correlation Phase Transform delay + polarity solver) | Pure Rust (`<50 ms` per stem pair) |
| **Synchro Arts VocAlign Pro** | \$349 | `forge-dsp::align_vocal_double_dtw` (MFCC + FastDTW time-warping solver emitting non-destructive REAPER `SM` Stretch Markers) | Pure Rust + Native REAPER `.RPP` Stretch Markers |
| **Melodyne 5 / Auto-Tune Pro** | \$249–\$699 | `forge-vocal` (YIN F0 pitch tracker, **sibilant-aware unvoiced consonant bypass**, **Levinson-Durbin LPC formant envelope preservation**, 3-part harmony generator & REAPER `<PITCHENV` compiler) | Pure Rust + Native REAPER Take Pitch Envelopes |
| **Eventide SplitEQ / oeksound Spiff** | \$179–\$199 | `forge-dsp::RealtimeTransientTonalSplit` + **Forge SplitEQ** VST3/CLAP (3-band structural Transient attack vs. Tonal sustain splitter & shaper) | Pure Rust (`0` audio-thread allocs) |
| **oeksound Bloom / Neutron 5 Density** | \$199–\$249 | `forge-dsp::RealtimeDensityAndClipper` + **Forge Density & ADAA Clipper** VST3/CLAP (Upward low-level density compressor + 3-band 2x-oversampled 1st-order ADAA `tanh` soft clipper) | Pure Rust (`2x` polyphase oversampled ADAA) |
| **Valhalla VintageVerb / Shimmer + Neural DSP** | \$50–\$199 | `forge-dsp::RealtimeFdnReverb` + `RealtimeNeuralPreamp` + **Forge SpaceVerb & Neural Preamp** VST3/CLAP (Dilated causal WaveNet `1073/1176/V76/A800` preamp + 8x8 Walsh-Hadamard FDN reverb with vocal ducking & `+12 st` octave shimmer) | Pure Rust (`O(N log N)` Hadamard + ADAA WaveNet) |
| **Scaler 2 + Slate Trigger 2** | \$59 + \$149 | `forge-theory` (Neo-Soul/Jazz/Cinematic voice-leading solver, Dilla/MPC micro-timing humanizer, sub-sample drum transient-to-MIDI replacer, ABC notation) | Pure Rust (`midly`) |
| **iZotope RX 11 Advanced** | \$1,199 | `forge-dsp::restoration` (DC, cubic Hermite de-clip, 50/60 Hz 10-harmonic hum notch, vocal de-click/de-plosive) + `forge-worker` (`Mel-Band-RoFormer` + `Apollo` HF bandwidth restoration) | Rust DSP + Local RTX 3080 / Colab A100 Worker |
| **FabFilter Pro-Q 4 / Pro-L 2 / Ozone 11 / Tonal Balance 2** | \$199–\$499 | `forge-dsp::loudness_limiter` + `elite_dsp` (ITU-R BS.1770-4 LUFS, 4x True Peak `-1.0 dBTP` brickwall limiter, 18-band Mastering EQ, built-in **Genre Tonal Balance Targets**, **8-Slot Lock-Free Cross-Instance Spectrum Bus**, `nih_plug_egui` 60 FPS OpenGL GUI) | Pure Rust (`f64` headroom + `egui` OpenGL) |

---

## Architecture

```
audio-forge/
├── Cargo.toml                  # Rust workspace (9 crates)
├── crates/
│   ├── forge-core/             # 64-bit planar AudioBuffer, 16/24/32f WAV & FFmpeg I/O, TPDF dither
│   ├── forge-dsp/              # BS.1770-4 LUFS, 4x True Peak, OpenSoothe, OpenSpacer, SplitEQ, ADAA Clipper, 8x8 FDN SpaceVerb, Neural WaveNet Preamp, GCC-PHAT, FastDTW
│   ├── forge-plug/             # 10 Native Cross-DAW VST3 & CLAP Real-Time Plugins (`nih-plug` + `nih_plug_egui` 60 FPS OpenGL GUI + Lock-Free Cross-Instance Spectrum Bus)
│   ├── forge-theory/           # Voice-leading solver, Neo-Soul reharmonizer, Dilla/MPC groove, Drum-to-MIDI, ABC
│   ├── forge-vocal/            # YIN F0 pitch tracker, sibilant-aware LPC formant-preserving shifter, 3-part harmonies, REAPER <PITCHENV
│   ├── forge-reaper/           # Programmatic REAPER 7.82 .RPP AST builder, UDP OSC controller, ISO 3901 ISRC tagger
│   ├── forge-worker/           # Hybrid GPU dispatcher (Local RTX 3080 uv worker + Colab A100 + Gemini Lyria 3.5 API)
│   ├── forge-cli/              # `forge` CLI binary (<5ms startup)
│   └── forge-mcp/              # `forge-mcp` JSON-RPC 2.0 stdio server for AI agents
├── xtask/                      # `cargo run --package xtask -- bundle forge-plug` VST3 & CLAP bundler
├── reaper-jsfx/                # Native REAPER EEL2 JSFX real-time plugins
│   ├── Forge_OpenSoothe_DeFizz.jsfx
│   ├── Forge_OpenSpacer_Unmask.jsfx
│   ├── Forge_SubBass_Reinforce.jsfx
│   └── Forge_Vocal_Surgical.jsfx
├── python-worker/              # On-demand GPU sidecar (Mel-Band-RoFormer, Apollo, YuE2-3B, Basic-Pitch, Seed-VC)
└── notebooks/
    └── ForgeAudio_Colab_A100_Worker.ipynb
```

---

## Quick Start

### 1. Build, Bundle & Install Individual (or All 10) VST3 / CLAP Plugins

Every plugin in ForgeAudio can be built, bundled, and installed **individually on its own** (under `plugins/forge-*`) or all together via `xtask`:

```bash
# List all 10 standalone VST3 & CLAP plugins
cargo run --package xtask -- list

# Bundle a single standalone VST3 & CLAP plugin (e.g., only Forge OpenSoothe or Forge SplitEQ)
cargo run --package xtask -- bundle forge-opensoothe --release
cargo run --package xtask -- bundle forge-spliteq --release

# Bundle ALL 10 individual standalone VST3 & CLAP plugins into target/bundled/
cargo run --package xtask -- bundle-all --release

# Install a single plugin (or --all 10 standalone plugins) directly to your system VST3 & CLAP folders
cargo run --package xtask -- install forge-opensoothe --release
cargo run --package xtask -- install --all --release
```

| Standalone Package Name | Individual Bundles Generated | Studio Processor Description |
| :--- | :--- | :--- |
| **`forge-opensoothe`** | `forge-opensoothe.vst3` / `.clap` | **Forge OpenSoothe De-Fizz** — 2048-pt OLA STFT Dynamic Resonance & AI Fizz Suppressor (`1536`-sample PDC, `egui` 60 FPS GUI & 8-Slot Shared Spectrum Bus) |
| **`forge-openspacer`** | `forge-openspacer.vst3` / `.clap` | **Forge OpenSpacer Unmask** — 4-Channel Sidechain Spectral Unmasker (Vocal $\rightarrow$ Music & Kick $\rightarrow$ Bass) |
| **`forge-vocal-surgical`** | `forge-vocal-surgical.vst3` / `.clap` | **Forge Vocal Surgical Strip** — Zero-Latency Mouth De-Click, `<115 Hz` Plosive Suppressor, Split-Band De-Esser & Breath Gate |
| **`forge-submono-tape`** | `forge-submono-tape.vst3` / `.clap` | **Forge SubMono & Console Tape** — Linkwitz-Riley `<160 Hz` Sub Mono-Maker, Harmonic Bass Reinforcer & Console8/ChowTape Saturation |
| **`forge-master-limiter`** | `forge-master-limiter.vst3` / `.clap` | **Forge Master True-Peak Limiter** — Lookahead 4x Inter-Sample True-Peak `-1.0 dBTP` Brickwall Limiter & ITU-R BS.1770-4 Meter |
| **`forge-drum-trigger`** | `forge-drum-trigger.vst3` / `.clap` | **Forge Drum Trigger & Theory MIDI** — Real-Time Audio-to-MIDI 3-Band Drum Replacer (`C1/D1/F#1`) + 1-Finger MIDI-to-Neo-Soul Chord Voicer |
| **`forge-phase-eq`** | `forge-phase-eq.vst3` / `.clap` | **Forge PhaseAlign & 18-Band EQ** — Sample-Accurate Phase/Polarity Aligner, 18-Band AI De-Harsh EQ & Built-In Genre Tonal Balance Targets |
| **`forge-spliteq`** | `forge-spliteq.vst3` / `.clap` | **Forge SplitEQ Transient & Tonal** — Structural 3-Band Transient Attack vs. Tonal Sustain Equalizer & Shaper with 60 FPS `egui` GUI |
| **`forge-density-clipper`** | `forge-density-clipper.vst3` / `.clap` | **Forge Density & 3-Band ADAA Clipper** — Upward Low-Level Density Compressor + 3-Band 2x-Oversampled 1st-Order ADAA Soft Clipper with 60 FPS `egui` GUI |
| **`forge-spaceverb`** | `forge-spaceverb.vst3` / `.clap` | **Forge SpaceVerb FDN & Neural Preamp** — Dilated Causal WaveNet Hardware Preamp (`Neve 1073`, `1176`, `V76`, `Studer A800`) + 8x8 Walsh-Hadamard FDN Reverb w/ Ducking & `+12 st` Shimmer |

### 2. Run the Full End-to-End Pipeline (`forge ingest`)

Takes any raw recording or AI-generated song (Suno, Udio, YuE2-3B, Lyria), removes metallic `2.5–6.5 kHz` fizz, extracts drum trigger MIDI, generates voice-led Neo-Soul/Jazz chords, masters to DistroKid/Spotify `-14.0 LUFS / -1.0 dBTP` (24-bit WAV), installs the native REAPER JSFX plugins, and generates a ready-to-mix `.RPP` project:

```bash
cargo run --bin forge -- ingest \
  --input raw_track.wav \
  --output-dir ./session_out \
  --bpm 96 \
  --chords "Am9 Fmaj9 Cmaj9 G13"
```

### 3. Individual CLI Commands

```bash
# Audit ITU-R BS.1770-4 LUFS, 4x True Peak (dBTP), LRA, and AI Spectral Fizz Index
cargo run --bin forge -- audit --path track.wav --json

# Surgical Restoration + OpenSoothe Dynamic De-Fizz
cargo run --bin forge -- restore --input track.wav --output cleaned.wav --vocal --defizz-depth-db 9.5

# OpenSpacer Sidechain Dynamic Frequency Unmasking (Vocal -> Instrumental or Kick -> Bass)
cargo run --bin forge -- unmask --target instrumental.wav --sidechain vocal.wav --output unmasked.wav

# GCC-PHAT Multi-Mic Phase Alignment or FastDTW Vocal Double Alignment
cargo run --bin forge -- align --reference lead_vox.wav --target double_vox.wav --output double_aligned.wav --vocal-dtw

# Vocal Pitch Correction + Formant Shift + 3-Part Harmony Stack + REAPER <PITCHENV
cargo run --bin forge -- vocal --input lead_vox.wav --output-dir ./vox_out --root-pc 9 --minor --harmonies

# Music Theory Voice-Led MIDI + Dilla/MPC Groove + Drum Trigger Replacement
cargo run --bin forge -- theory --chords "Dm9 G13 Cmaj9 A7alt" --output-midi chords.mid --style neosoul --groove dilla --reharmonize

# Master for DistroKid / Apple Digital Masters (-14.0 LUFS, -1.0 dBTP, 18-Band EQ, Console8/ChowTape Warmth, ISRC)
cargo run --bin forge -- master --input mix.wav --output master_24bit.wav --target-lufs -14.0 --ceiling-dbtp -1.0 --isrc USABC2600001
```

---

## License

Licensed under the **Apache License, Version 2.0** ([`LICENSE`](LICENSE)).
