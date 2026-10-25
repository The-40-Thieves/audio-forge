# ForgeAudio (`audio-forge`)

**An audio delivery pipeline that AI agents can drive.** Hand it a track and a target (Spotify, Apple Music, YouTube, broadcast), and it returns a repaired, spec-compliant master plus a measured report that says whether the file is ready to ship and why. On request it also produces separated stems, MIDI and a REAPER session.

[![CodSpeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://app.codspeed.io/The-40-Thieves/audio-forge?utm_source=badge)
[![CI](https://github.com/The-40-Thieves/audio-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/The-40-Thieves/audio-forge/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-2021%20Edition-orange.svg)](Cargo.toml)
[![VST3 / CLAP](https://img.shields.io/badge/Plugins-VST3%20%7C%20CLAP%20%7C%20JSFX-purple.svg)](crates/forge-plug/)
[![REAPER](https://img.shields.io/badge/REAPER-7.82%2B%20Native%20JSFX%20%26%20RPP-green.svg)](reaper-jsfx/)

The same Rust DSP engine ships in four forms:

- **`forge-mcp`**, a Model Context Protocol server for agents.
- **`forge`**, a command-line tool.
- **`forge-plug`**, ten VST3 and CLAP plugins.
- **Native REAPER JSFX** effects.

---

## The agent pipeline

An agent makes one tool call. The pipeline:

1. Audits the input.
2. Repairs DC offset, clipping and mains hum.
3. Suppresses harsh spectral resonances ("AI fizz").
4. Can tone-match the mix to a reference track or a genre curve.
5. Normalizes loudness and true-peak limits to the target.
6. Audits the result again.

Each requirement gets an explicit pass or fail, so an agent never has to judge from loose numbers whether a master meets spec.

```
input.wav ──► audit ──► repair ──► de-fizz ──► tone match ──► loudness + true-peak limit ──► audit ──► checks
                                               (optional)                                              │
                                               master.wav + master.delivery.json  ◄──────────────────────┘
```

### Connect an agent

Build the server, then register it with any MCP client. For example, with Claude Code:

```bash
cargo build --release -p forge-mcp
claude mcp add forge -- /path/to/audio-forge/target/release/forge-mcp
```

### Deliver a track

```json
{ "name": "forge_deliver",
  "arguments": { "input": "mix.wav", "output": "out/master.wav", "target": "apple_music" } }
```

The response, also written to `out/master.delivery.json`, holds both audits and these checks:

```
apple_music (-16.0 LUFS, -1.0 dBTP): -4.45 -> -16.00 LUFS, -0.59 -> -11.95 dBTP
  [PASS] integrated_loudness: -16.00 LUFS (target -16.0 +/- 1.0 LU)
  [PASS] true_peak: -11.95 dBTP (ceiling -1.0 dBTP)
  [PASS] no_clipping: 0 clipped samples
  [PASS] sample_rate: 48000 Hz (minimum 44100 Hz)
READY
```

### MCP tools

| Tool | What it returns |
| :--- | :--- |
| `forge_deliver` | A verified master for a target: before and after audits, pass/fail checks and the WAV, plus a `.delivery.json` report. |
| `forge_ingest_to_reaper` | The full session from one song: separated stems, cleaned vocal, drum-trigger and chord MIDI, a verified master for the target, and a REAPER `.rpp` project with JSFX inserts. |
| `forge_audit_audio` | BS.1770-4 integrated, short-term and momentary loudness, LRA, sample and true peak, crest factor, DC, clipping, stereo correlation and a fizz index. |
| `forge_restore_and_defizz` | The restoration chain and spectral de-fizz, with an optional vocal clean-up pass. |
| `forge_spectral_unmask` | Sidechain spectral ducking that clears space for a vocal or kick. |
| `forge_phase_align` | Delay and polarity alignment of two stems (GCC-PHAT). |
| `forge_vocal_suite` | Scale-locked pitch correction, formant-preserving shifting and harmonies. |
| `forge_music_theory_midi` | Voice-led chord and bass MIDI, optional reharmonization, and an ABC score. |
| `forge_master_distrokid` | Low-level mastering with an explicit LUFS and ceiling, reference-match EQ and ISRC tagging. |

### Delivery targets

| `target` | Integrated loudness | True-peak ceiling |
| :--- | ---: | ---: |
| `spotify`, `youtube`, `amazon_music`, `tidal` | -14 LUFS | -1 dBTP |
| `deezer` | -15 LUFS | -1 dBTP |
| `apple_music`, `podcast` | -16 LUFS | -1 dBTP |
| `ebu_r128` | -23 LUFS | -1 dBTP |
| `atsc_a85` | -24 LUFS | -2 dBTP |
| `custom` | `target_lufs` | `ceiling_dbtp` |

These are the reference levels each service publishes for playback normalization, not upload requirements, and they change. Check the platform's current documentation before a release.

A master passes when:

- Integrated loudness is within 1 LU of the target.
- True peak is at or below the ceiling.
- No samples are clipped.
- The sample rate is at least 44.1 kHz.

Sources:

- [Spotify loudness normalization](https://support.spotify.com/artists/article/loudness-normalization/)
- [EBU R 128](https://tech.ebu.ch/publications/r128)
- [ITU-R BS.1770](https://www.itu.int/rec/R-REC-BS.1770)
- Platform summaries from [Soundbridge](https://www.soundbridge.io/mastering-for-streaming-your-2026-platform-guide) and [MasteringBox](https://www.masteringbox.com/learn/mastering-for-streaming)

---

## Verified, not claimed

| What | Evidence |
| :--- | :--- |
| BS.1770-4 loudness, LRA and true peak meet the EBU conformance cases | [`tests/ebu_reference.rs`](crates/forge-dsp/tests/ebu_reference.rs) runs EBU Tech 3341 and 3342 reference signals. True peak uses the BS.1770-4 Annex 2 48-tap 4x polyphase FIR. |
| Delivery reaches every target it reports as passed | Unit tests in [`delivery.rs`](crates/forge-dsp/src/delivery.rs) push quiet, hot and low-sample-rate mixes through each target. |
| DSP stays finite and stable for arbitrary input | [`tests/proptest_invariants.rs`](crates/forge-dsp/tests/proptest_invariants.rs) uses property tests: Hadamard involution, ADAA continuity, filter decay, limiter and restoration totality. |
| Plugins load and behave in hosts | [CI](.github/workflows/ci.yml) runs pluginval (strictness 5) on every VST3 and clap-validator on every CLAP, on Linux, macOS and Windows. |
| Performance regressions are tracked | [CodSpeed](https://app.codspeed.io/The-40-Thieves/audio-forge) benchmarks [`forge-dsp`](crates/forge-dsp/benches/dsp.rs), [`forge-vocal`](crates/forge-vocal/benches/vocal.rs) and [`forge-theory`](crates/forge-theory/benches/theory.rs). |

Known gap: two clap-validator state tests are excluded in CI because of upstream bugs in nih-plug's state loading. See [SECURITY.md](SECURITY.md).

---

## Modules and what inspired them

These are independent implementations of well-known techniques. They are not clones of, or substitutes for, the commercial products named in the "Inspired by" column.

| Module | What it actually does | Inspired by | Source and tests |
| :--- | :--- | :--- | :--- |
| Spectral de-fizz | 2048-point overlap-add STFT. Detects narrow resonances against a smoothed spectral envelope and attenuates them dynamically, with a capped depth. | oeksound Soothe | [`spectral_defizz.rs`](crates/forge-dsp/src/spectral_defizz.rs) and its unit test (suppresses a harsh whistle) |
| Spectral unmask | Sidechain-keyed per-band attenuation of a target stem, in vocal (200 Hz to 7.5 kHz) or kick (30 to 130 Hz) mode. | Wavesfactory Trackspacer | [`spectral_unmask.rs`](crates/forge-dsp/src/spectral_unmask.rs), proptest totality |
| Phase align | GCC-PHAT cross-correlation finds the delay and polarity between two stems. | Sound Radix Auto-Align | [`phase_align.rs`](crates/forge-dsp/src/phase_align.rs) and its unit test (recovers an exact delay and an inverted polarity) |
| Vocal double align | MFCC features plus FastDTW. Emits REAPER stretch markers instead of rendering audio. | Synchro Arts VocAlign | [`phase_align.rs`](crates/forge-dsp/src/phase_align.rs) |
| Pitch and formant | YIN F0 tracking with a bypass for unvoiced consonants, LPC formant preservation, diatonic harmonies, and REAPER `<PITCHENV` output. | Melodyne, Auto-Tune | [`forge-vocal`](crates/forge-vocal/src/lib.rs): YIN unit test and proptests |
| Restoration | DC removal, cubic Hermite de-clip, a 50/60 Hz hum notch across 10 harmonics, and vocal de-click, de-plosive and breath gate. | iZotope RX | [`restoration.rs`](crates/forge-dsp/src/restoration.rs), proptest totality |
| Loudness and limiter | BS.1770-4 metering, then gain to target and a 3 ms lookahead true-peak limiter with a final true-peak trim. | FabFilter Pro-L, loudness meters | [`loudness_limiter.rs`](crates/forge-dsp/src/loudness_limiter.rs), EBU reference tests |
| Mastering EQ and tone match | 18-band EQ, plus curves matched to a reference track or to built-in genre tonal-balance targets. | Ozone, Tonal Balance Control | [`elite_dsp.rs`](crates/forge-dsp/src/elite_dsp.rs), [`restoration.rs`](crates/forge-dsp/src/restoration.rs) |
| Transient and tonal split | A 3-band transient/sustain split with independent shaping. | Eventide SplitEQ | [`elite_dsp.rs`](crates/forge-dsp/src/elite_dsp.rs), proptest partition of unity |
| Density and clipper | Upward low-level compression and a 3-band, 2x oversampled, first-order ADAA `tanh` clipper. | oeksound Bloom, multiband clippers | [`elite_dsp.rs`](crates/forge-dsp/src/elite_dsp.rs), ADAA proptests |
| SpaceVerb and preamp | An 8x8 Walsh-Hadamard FDN reverb with ducking and shimmer, plus a small dilated causal WaveNet preamp model. | Valhalla reverbs, NAM-style amp modeling | [`elite_dsp.rs`](crates/forge-dsp/src/elite_dsp.rs), FDN decay and hysteresis proptests |
| Music theory | A voice-leading solver, Neo-Soul and jazz reharmonization, Dilla/MPC micro-timing, audio-to-MIDI drum triggers, and ABC export. | Scaler, Slate Trigger | [`forge-theory`](crates/forge-theory/src/lib.rs): unit test and proptests |
| Stem separation | Dispatches to a local or Colab GPU worker that runs Mel-Band-RoFormer. Not pure Rust; it needs the Python worker. | Stem-splitting services | [`forge-worker`](crates/forge-worker/), [`python-worker/`](python-worker/) |

---

## Architecture

```
audio-forge/
├── crates/
│   ├── forge-core/      # 64-bit planar AudioBuffer, WAV and FFmpeg I/O, TPDF dither, AuditReport
│   ├── forge-dsp/       # Delivery pipeline, BS.1770-4 meter and limiter, restoration, de-fizz, unmask, alignment, real-time processors
│   ├── forge-plug/      # Shared nih-plug VST3/CLAP plugin implementations
│   ├── forge-theory/    # Voice leading, reharmonization, groove, drum-to-MIDI, ABC
│   ├── forge-vocal/     # YIN pitch, LPC formant-preserving shift, harmonies, REAPER <PITCHENV
│   ├── forge-reaper/    # .RPP builder, OSC control, ISRC validation and tagging, JSFX installer
│   ├── forge-worker/    # GPU worker dispatch (local uv worker, Colab, Lyria API)
│   ├── forge-cli/       # `forge` binary
│   └── forge-mcp/       # `forge-mcp` JSON-RPC 2.0 stdio MCP server
├── plugins/forge-*/     # One crate per standalone VST3/CLAP plugin
├── xtask/               # Plugin bundler and installer
├── reaper-jsfx/         # Native REAPER EEL2 effects
├── python-worker/       # GPU sidecar (stem separation, bandwidth restoration)
└── notebooks/           # Colab worker notebook
```

---

## Quick start

### Deliver from the command line

```bash
cargo run --release --bin forge -- deliver mix.wav -o out/master.wav --target spotify
cargo run --release --bin forge -- deliver mix.wav -o out/master.wav --lufs -10 --ceiling-dbtp -1.5
cargo run --release --bin forge -- deliver vocal.wav -o out/vocal.wav --target apple_music --vocal --no-defizz
cargo run --release --bin forge -- deliver mix.wav -o out/master.wav --reference ref.wav
```

`forge deliver` prints each check and exits with code 2 when the master is not ready, which makes it usable as a CI gate.

### Full session from one song

```bash
cargo run --release --bin forge -- ingest raw_track.wav -o ./session_out --bpm 96 --chords "Am9 Fmaj9 Cmaj9 G13"
```

### Individual commands

```bash
cargo run --release --bin forge -- audit track.wav
cargo run --release --bin forge -- restore track.wav -o cleaned.wav --defizz-depth 9.5
cargo run --release --bin forge -- unmask instrumental.wav --sidechain vocal.wav -o unmasked.wav --mode vocal
cargo run --release --bin forge -- phase-align overhead.wav snare_top.wav -o snare_aligned.wav
cargo run --release --bin forge -- vocal lead_vox.wav -o ./vox_out --scale "A minor" --harmonize
cargo run --release --bin forge -- theory --chords "Dm9 G13 Cmaj9 A7" -o chords.mid --style neo-soul --groove dilla
cargo run --release --bin forge -- master mix.wav -o master.wav --target-lufs -14 --ceiling-dbtp -1 --isrc USABC2600001
```

### Plugins

```bash
cargo run --package xtask -- list                            # list the 10 plugins
cargo run --package xtask -- bundle forge-opensoothe --release
cargo run --package xtask -- bundle-all --release            # everything into target/bundled/
cargo run --package xtask -- install --all --release         # copy into system VST3 and CLAP folders
```

| Package | What it does |
| :--- | :--- |
| `forge-opensoothe` | Dynamic spectral resonance suppressor (STFT, 1536-sample latency compensation, egui GUI) |
| `forge-openspacer` | Sidechain spectral unmasker (vocal over music, kick over bass) |
| `forge-vocal-surgical` | Zero-latency de-click, plosive suppression below 115 Hz, split-band de-esser and breath gate |
| `forge-submono-tape` | Linkwitz-Riley sub mono below 160 Hz, harmonic bass reinforcement, console and tape saturation |
| `forge-master-limiter` | Lookahead 4x true-peak limiter with a BS.1770-4 meter |
| `forge-drum-trigger` | Real-time 3-band drum-to-MIDI triggers and a one-finger chord voicer |
| `forge-phase-eq` | Phase and polarity aligner with an 18-band EQ and genre tonal-balance targets |
| `forge-spliteq` | 3-band transient/tonal split equalizer and shaper |
| `forge-density-clipper` | Upward density compressor and a 3-band ADAA soft clipper |
| `forge-spaceverb` | FDN reverb with ducking and shimmer, plus a WaveNet preamp model |

---

## License

Licensed under the **Apache License, Version 2.0** ([`LICENSE`](LICENSE)).
