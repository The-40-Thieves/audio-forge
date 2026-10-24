# ForgeAudio (`audio-forge`)

**Open-Source Hybrid Music Production, AI Track Restoration, Vocal Surgery, Music Theory & REAPER 7.82 Mastering Suite**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-2021%20Edition-orange.svg)](Cargo.toml)
[![REAPER](https://img.shields.io/badge/REAPER-7.82%2B%20Native%20JSFX%20%26%20RPP-green.svg)](reaper-jsfx/)

ForgeAudio (`forge` CLI + `forge-mcp` Model Context Protocol server + native REAPER `JSFX` DSP plugins) is a zero-bloat **Rust + REAPER 7.82 + Hybrid GPU Worker** audio engineering ecosystem built for **both live studio musicians and AI/hybrid producers** (Suno, Udio, YuE2-3B, ACE-Step 1.5, Google Lyria 3.5).

---

## Why ForgeAudio? (Commercial vs. ForgeAudio Open-Source Stack)

| Commercial Industry Standard | Price | ForgeAudio (`audio-forge`) Equivalent | Runtime & Speed |
| :--- | :--- | :--- | :--- |
| **oeksound Soothe2** | \$209 | `forge-dsp::spectral_defizz` + `Forge_OpenSoothe_DeFizz.jsfx` (2048-pt OLA STFT dynamic spectral resonance & AI metallic fizz suppressor) | Pure Rust SIMD (`>120x` real-time) + REAPER JSFX (`0%` idle CPU) |
| **Wavesfactory Trackspacer** | \$59 | `forge-dsp::spectral_unmask` + `Forge_OpenSpacer_Unmask.jsfx` (32-band / STFT sidechain dynamic frequency carving: Vocal→Music, Kick→Bass) | Pure Rust + REAPER 4-channel sidechain JSFX |
| **Sound Radix Auto-Align 2** | \$199 | `forge-dsp::align_phase_gcc_phat` (Sub-sample Generalized Cross-Correlation Phase Transform delay + polarity solver) | Pure Rust (`<50 ms` per stem pair) |
| **Synchro Arts VocAlign Pro** | \$349 | `forge-dsp::align_vocal_double_dtw` (MFCC + FastDTW time-warping solver emitting non-destructive REAPER `SM` Stretch Markers) | Pure Rust + Native REAPER `.RPP` Stretch Markers |
| **Melodyne 5 / Auto-Tune Pro** | \$249–\$699 | `forge-vocal` (YIN F0 pitch tracker, scale quantizer, formant-preserving dual-grain pitch shifter, 3-part harmony generator & REAPER `<PITCHENV` compiler) | Pure Rust + Native REAPER Take Pitch Envelopes |
| **Scaler 2 + Slate Trigger 2** | \$59 + \$149 | `forge-theory` (Neo-Soul/Jazz/Cinematic voice-leading solver, Dilla/MPC micro-timing humanizer, sub-sample drum transient-to-MIDI replacer, ABC notation) | Pure Rust (`midly`) |
| **iZotope RX 11 Advanced** | \$1,199 | `forge-dsp::restoration` (DC, cubic Hermite de-clip, 50/60 Hz 10-harmonic hum notch, vocal de-click/de-plosive) + `forge-worker` (`Mel-Band-RoFormer` + `Apollo` HF bandwidth restoration) | Rust DSP + Local RTX 3080 / Colab A100 Worker |
| **FabFilter Pro-L2 / Ozone 11** | \$199–\$499 | `forge-dsp::loudness_limiter` (ITU-R BS.1770-4 K-weighted LUFS, 4x polyphase FIR True Peak `-1.0 dBTP` brickwall limiter, 18-band Mastering EQ, Reference Match EQ, Console8/ChowTape hysteresis saturation) | Pure Rust (`f64` headroom) |

---

## Architecture

```
audio-forge/
├── Cargo.toml                  # Rust workspace (7 crates)
├── crates/
│   ├── forge-core/             # 64-bit planar AudioBuffer, 16/24/32f WAV & FFmpeg I/O, TPDF dither
│   ├── forge-dsp/              # BS.1770-4 LUFS, 4x True Peak, OpenSoothe, OpenSpacer, GCC-PHAT, FastDTW, 18-Band EQ
│   ├── forge-theory/           # Voice-leading solver, Neo-Soul reharmonizer, Dilla/MPC groove, Drum-to-MIDI, ABC
│   ├── forge-vocal/            # YIN F0 pitch tracker, formant-preserving shifter, 3-part harmonies, REAPER <PITCHENV
│   ├── forge-reaper/           # Programmatic REAPER 7.82 .RPP AST builder, UDP OSC controller, ISO 3901 ISRC tagger
│   ├── forge-worker/           # Hybrid GPU dispatcher (Local RTX 3080 uv worker + Colab A100 + Gemini Lyria 3.5 API)
│   ├── forge-cli/              # `forge` CLI binary (<5ms startup)
│   └── forge-mcp/              # `forge-mcp` JSON-RPC 2.0 stdio server for AI agents
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

### 1. Build the Rust Workspace

```bash
cargo build --release --workspace
```

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
