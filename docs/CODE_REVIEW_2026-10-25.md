# Code Review — 2026-10-25

Reviewed at commit `c2f7cb6` (branch `chore/update-stack`). No code was changed as part of this review.

## Method

1. Ran the same gates as CI.
2. Ran three parallel read-only reviews, each over its own part of the code:
   - realtime DSP and plugins
   - offline DSP and core crates
   - CLI, MCP, worker, REAPER and tooling
3. Confirmed the highest-severity findings by hand against the source. These are marked **✅ confirmed**. All other findings come from reading the code and have not been reproduced by running it.

## Automated checks

| Check | Result |
|---|---|
| `cargo fmt --all -- --check` | Pass |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | Pass. The only warnings are in vendored `nih_plug`. |
| `cargo test --workspace --locked --no-fail-fast` | Pass, 57 tests, 0 failures |
| `cargo audit` | No vulnerabilities. 3 warnings are already allowed: `atty` is unmaintained and unsound (RUSTSEC-2024-0375, RUSTSEC-2021-0145), and `ttf-parser` is unmaintained (RUSTSEC-2026-0192). |
| `ruff check` / `ruff format --check` (python-worker, scripts) | Pass |

### Test coverage distribution

| Crate | Test fns |
|---|---|
| forge-dsp | 40 |
| forge-reaper | 5 |
| forge-vocal | 4 |
| forge-core, forge-plug, forge-theory | 3 each |
| forge-worker | 2 |
| forge-cli | 1 |
| forge-mcp, xtask | **0** |

Nothing tests the limiter on transients, the latency the plugins report, allocation on the audio thread, or the MCP tool dispatch. That is why the defects below pass CI.

---

## High severity

### H1. FFT allocates on the audio thread ✅ confirmed

- **Where:** `crates/forge-dsp/src/realtime.rs:250, 257, 326, 515-517, 553-554` (`RealtimeOpenSoothe::process_stft_hop`, `RealtimeOpenSpacer::process_unmask_hop`)
- **Defect:** The code calls `Fft::process()`, and in rustfft 6.4 that allocates `vec![..; get_inplace_scratch_len()]` on every call (rustfft `src/lib.rs:195-198`).
- **Impact:** OpenSoothe makes 4 heap allocations per hop and OpenSpacer makes 5. This contradicts the "zero allocation" docs and the `assert_process_allocs` intent, and it risks dropouts on the audio thread.
- **Fix:** Allocate a scratch `Vec` once in `new()` and call `process_with_scratch`.

### H2. Master limiter lookahead does not look ahead ✅ confirmed

- **Where:** `crates/forge-dsp/src/realtime.rs:927-954` (`RealtimeMasterLimiter`)
- **Defect:** `required_gain` is computed per sample, with no hold and no minimum over the lookahead window. A short over-peak such as a drum hit is over the ceiling for only about 4 samples. The 0.5 ms attack moves the gain only about 15% of the way in that time, and the gain releases on the next sample.
- **Impact:** The peak leaves the delay line almost unattenuated and is hard-clipped by the final `clamp(-ceiling, ceiling)` (line 954). Transients get clipped instead of limited, and true-peak compliance is lost. Only sustained overs are handled correctly.
- **Fix:** Take a sliding-window minimum of `required_gain` over the lookahead length, or ramp the gain so it reaches the target by the time the peak exits the delay line.

### H3. STFT plugins report the wrong latency

- **Where:** `crates/forge-dsp/src/realtime.rs:132-134, 432-434`; passed to the host at `crates/forge-plug/src/lib.rs:434, 693`
- **Defect:** The plugins report `fft_size - hop` = 1536 samples. A frame computed at ring position P writes into slot P+i, so input at time t comes out at t+2048.
- **Impact:** Plugin delay compensation is off by 512 samples (about 10.7 ms at 48 kHz) for OpenSoothe and OpenSpacer.
- **Fix:** Report 2048, or change the output read position to match 1536. Add a test that feeds an impulse and measures the actual delay.

### H4. `reset()` hardcodes 48 kHz and allocates ✅ confirmed (hardcode)

- **Where:** `crates/forge-plug/src/lib.rs:444-447` (`ForgeOpenSoothePlug::reset`); `crates/forge-dsp/src/realtime.rs:883`
- **Defect:** `reset()` calls `self.dsp.reset(48_000.0)` and `self.limiter_dsp.reset(48_000.0)`, whatever the session rate is. The limiter's `reset` is `*self = new()`, which allocates two `Vec`s. `sub_tape_dsp` is not cleared at all.
- **Impact:**
  - At 44.1 or 96 kHz, every coefficient and the limiter lookahead are wrong after a reset.
  - The lookahead stops matching the latency reported in `initialize`.
  - nih-plug documents `reset()` as callable on the audio thread, so the allocation also counts as a real-time violation.
- **Fix:** Store the sample rate in `initialize`. Make `reset` clear state only, with no reallocation.

### H5. Sub & Tape side high-pass rebuilt every block

- **Where:** `crates/forge-plug/src/lib.rs:1138-1139` with `crates/forge-dsp/src/realtime.rs:751-755`
- **Defect:** `set_mono_cutoff` runs unconditionally every block and rebuilds `side_hp1` and `side_hp2` with `Biquad::highpass`, which zeroes `z1` and `z2`.
- **Impact:** The side channel glitches at buffer rate whenever it has content.
- **Fix:** Update coefficients only when the cutoff changes, and keep the filter state.

### H6. DTW alignment allocates full N×N matrices ✅ confirmed

- **Where:** `crates/forge-dsp/src/phase_align.rs:191-192` (`align_vocal_double_dtw`)
- **Defect:** `cost` (`Vec<Vec<f64>>`) and `back` (`Vec<Vec<(usize, usize)>>`) are allocated at `num_steps × num_steps`, but only a band of width `2*band+1` is ever used.
- **Impact:** At a 10 ms hop, memory grows with the square of the length:

  | Vocal length | Memory needed |
  |---|---|
  | 3 min | about 7.8 GB |
  | 5 min | about 21 GB |
  | 10 min | about 86 GB |

  On ordinary user audio the result is an allocation abort or heavy swapping.
- **Fix:** Store only the band diagonal for each row.

### H7. De-plosive split carves a permanent notch ✅ confirmed (offline)

- **Where:**
  - offline: `crates/forge-dsp/src/restoration.rs:289-309`
  - realtime: `crates/forge-dsp/src/realtime.rs:660-678`
- **Defect:** The output is `hi + lo * plosive_gain`, built from a 2nd-order low-pass and high-pass at the same cutoff frequency. With gain 1.0, which is almost always the case, the sum has the numerator `(1, -2cos w0, 1)`, which is a notch at the cutoff.
- **Impact:**
  - Offline: `deliver(..., vocal=true)` always cuts a full-depth notch at 110 Hz into the vocal, even when there are no plosives.
  - Realtime: about -10 dB near 90-115 Hz and about -3 dB from 63 to 200 Hz, which removes the vocal fundamental.
- **Fix:** Derive the high band as `x - lp`, or use a Linkwitz-Riley crossover so the bands sum flat.

### H8. MCP tools accept arbitrary filesystem paths ✅ confirmed

- **Where:** `crates/forge-mcp/src/main.rs:346-784` (`dispatch_tool`)
- **Defect:** None of the path arguments are validated or confined: `input`, `output`, `output_midi`, `output_dir`, `reference`, `sidechain`, `target`. They go straight to `AudioBuffer::load`, `save_wav`, `write_midi_file` and `create_dir_all`.
- **Impact:** A prompt-injected agent could do any of the following:
  - overwrite any file the user can write, for example with `forge_restore_and_defizz` and `output=<any path>`, or `forge_deliver` plus its `.delivery.json` write
  - read any file that decodes as audio
  - create arbitrary directory trees through `forge_ingest_to_reaper` and `forge_vocal_suite`
- **Fix:**
  1. Define an allowed root, set by an environment variable or a flag.
  2. Canonicalize every path.
  3. Reject `..` and any absolute path outside the allowed root.

---

## Medium severity

### Realtime DSP and plugins

**M1. De-esser over-subtracts** (`realtime.rs:681-690`)
- `sib_l` is the residual of a +12 dB peaking EQ, about 2.98× the band.
- Subtracting `sib_l*(1-cut)` therefore removes too much. Below `cut` ≈ 0.66 the band is polarity-inverted instead of attenuated.
- The `sib_env` threshold is also measured on the boosted residual.
- Fix: scale by `(1-cut)/(A²-1)`, or use a true band-pass.

**M2. EQ automation clicks** (`realtime.rs:1016-1019`)
- `RealtimeMasteringEq18::set_gains` replaces each `Biquad`, which zeroes its state. Automating the EQ amount or switching genre clicks.
- Fix: keep `z1` and `z2` when updating the coefficients.

**M3. Drum Trigger notes hang** (`forge-plug/src/lib.rs:1565-1587, 1535-1541`)
- NoteOns triggered by audio on notes 36, 38 and 42 never get a NoteOff, so sustaining instruments hang.
- Chord expansion copies the incoming event, so all 5 notes share one `voice_id`, and NoteOff matching becomes ambiguous.
- Toggling `expand_chords` between a NoteOn and its NoteOff leaves the harmony notes stuck.

**M4. Snare detector leaks broadband signal** (`realtime.rs:1148`)
- `snare_bp(...) - mono*0.5` leaves half of the broadband signal in the snare band.
- Kick and hat transients can then pass the snare ratio test and fire spurious D1 notes.
- Fix: subtract `mono` in full.

**M5. Latency goes stale when the limiter is toggled** (`forge-plug/src/lib.rs:434-440`)
- Latency is reported once, in `initialize`, based on `enable_limiter` at that moment.
- Toggling the limiter later changes the real delay by the lookahead (144 samples at 48 kHz) without calling `set_latency_samples`.

**M6. Spectrum analyzer is off by 4×** (`elite_dsp.rs:1170-1186`)
- It processes every 4th sample with SVF coefficients designed for the full sample rate, and does no anti-alias filtering before decimating.
- Every band displays 4× too low (25 Hz-18.5 kHz shows as 6 Hz-4.6 kHz), and high-frequency content aliases in. This affects the GUI only.

**M7. Reverb decay maps to the wrong range** (`forge-plug/src/lib.rs:2472`, `elite_dsp.rs:526`)
- `reverb_decay` (0.05-0.98) is passed as `decay_rt60_secs`, which is clamped to 0.2-20 s.
- RT60 therefore never exceeds 1 s, and the lower third of the knob does nothing.
- Fix: map the normalized value to seconds.

**M8. Offline limiter attack is a step** (`loudness_limiter.rs:877-888`, `normalize_and_limit`)
- Attack is an instantaneous gain step (`current_gain = target_g`) taken L samples before the peak.
- Large reductions create a discontinuity, heard as a click.
- Fix: ramp the gain.

### Offline DSP, vocal, theory

**M9. Chord parser falls back to the wrong quality** (`forge-theory/src/lib.rs:111-117`)
- The catch-all maps any unknown quality that starts with `m` to a minor 7th, and anything else to a major 7th.
- `Cmaj7#11` and `Cmaj11` become C-Eb-G-Bb.
- `G7#11`, `G7b5`, `G7#5`, `G9sus4` and `G13b9` become G-B-D-F#.

**M10. `"11"` is voiced as minor 11** (`forge-theory/src/lib.rs:104`)
- `"11"` shares an arm with `m11` (`[0,3,7,10,14,17]`), so `G11` is voiced as `Gm11`.

**M11. DTW resampling repeats one sample** (`phase_align.rs:257-275`)
- When the guide is longer than the dub, `step_f` runs past `num_steps-1` and the mapped position stops moving.
- Example: a 5 s guide against a 3 s dub gives about 2 s of one repeated sample (a DC offset) instead of silence.

**M12. GCC-PHAT only looks at the start of the audio** (`phase_align.rs:60-63`)
- Analysis uses only the first `min(n, fft_len/2)` samples, at most 32768 (0.68 s at 48 kHz). The comment claims it picks the "highest-energy window".
- A silent or count-in intro longer than that gives lag 0 and no polarity decision.

**M13. Correlation confidence is always 1.0** (`phase_align.rs:93, 121`)
- `correlation_peak` is not normalized by signal energy. For ordinary audio it exceeds 1 and is clamped to 1.0, so the 0-1 confidence carries no information.

**M14. Scale parser drops the mode** (`forge-vocal/src/lib.rs:36-47`)
- `VocalScale::parse` discards the remainder returned by `parse_pitch_class`, so `"Am"` and `"F#m"` parse as major.
- `"A minor pentatonic"` reads only one token and returns full natural minor.
- Mixolydian, lydian, phrygian and locrian silently become major.

**M15. Pitch tracker is coarse and early** (`forge-vocal/src/lib.rs:154, 194-198`)
- YIN stops at the first tau below 0.15 instead of the local minimum, and it has no parabolic interpolation.
- Pitch is quantized to whole-sample lag, about 16 cents at 445 Hz and 48 kHz.
- `time_secs` is the window start rather than its center, so the pitch curves and the REAPER PITCHENV are about 20 ms early.

### CLI, MCP, worker, REAPER

**M16. GPU and Colab backends never run** ✅ confirmed (`forge-worker/src/lib.rs:40-44`)
- Only `LocalGpu` ever tries the Python worker. `Auto` and `RemoteColab{base_url}` always use the Rust DSP splitter.
- The MCP and CLI both pass `Auto`, so neural separation never runs, and `FORGE_COLAB_WORKER_URL` is never read.
- When `LocalGpu` fails, the error is discarded and the code silently falls back to the DSP splitter.

**M17. Python worker call can hang and depends on cwd** (`forge-worker/src/lib.rs:124-135`)
- `Command::new("uv").output()` has no timeout. A hung model download or CUDA stall blocks the single-threaded stdio MCP loop forever.
- The script path is relative to the process working directory. If the MCP host launches the server elsewhere, the worker fails and the failure is swallowed (see M16).

**M18. Invalid UTF-8 stops the MCP server** (`forge-mcp/src/main.rs:63`)
- `let Ok(raw) = line else { break };` exits the server on one invalid-UTF-8 line from stdin.
- A panic in any DSP call is not caught (no `catch_unwind`), so it also kills the server.

**M19. Failed deliveries are reported as success** (`forge-mcp/src/main.rs:374-385, 723-778`)
- `forge_deliver` and `forge_ingest_to_reaper` return `isError:false` even when `report.ready == false`.
- The CLI exits 2 in the same case (`forge-cli/src/main.rs:593`), so an agent that only checks `isError` treats a failed delivery as success.

**M20. Generated sessions open with missing FX** (`forge-mcp/src/main.rs:729-740`, `forge-reaper/src/lib.rs:285, 317`)
- The JSFX files are installed into `<output_dir>/Effects`.
- The `.rpp` references `ForgeAudio/Forge_*.jsfx`, which REAPER looks for under its own `Effects/ForgeAudio`.

**M21. Relative stem paths go offline** (`forge-reaper/src/lib.rs:322-323`)
- Track `FILE` paths are written exactly as given, and REAPER resolves relative paths against the `.rpp`'s directory, so `out/stems/x.wav` becomes `out/out/stems/x.wav`.
- `sanitize_rpp_string` also strips `"` from paths, which breaks the reference.
- Fix: make the paths absolute.

**M22. Python worker can pick the wrong stem** (`python-worker/forge_gpu_worker.py:40-46, 51-57, 82`)
- Stems are matched by substring on the output filename, and that filename includes the input name.
- `Lead_Vocal_mix.wav` matches `"Vocal"` for both stems, so the instrumental can be returned as the vocal. An input name containing `dry` triggers the de-reverb check.
- `Client.predict` has no timeout.

**M23. Codspeed CI can mint OIDC tokens from PR code** (`.github/workflows/codspeed.yml:12-14`)
- `id-token: write` is set for the whole workflow, so PR-controlled `build.rs` and benches can mint OIDC tokens on same-repo PRs.
- Fix: limit the permission to the step that needs it.

**M24. CI runs validator binaries unverified** (`.github/workflows/ci.yml:169-176`)
- The pluginval and clap-validator release zips are downloaded and run with no SHA256 check. Only the version tag pins them.

---

## Low severity

- **True-peak estimate reads low** (`realtime.rs:897-901`)
  - The estimator checks only t=0.5 and t≈0.75 within each sample interval, never t=0.25.
  - It under-reads some inter-sample peaks.
- **Spectrum slot 0 problems** (`forge-plug/src/lib.rs:198-207`, `elite_dsp.rs:870`)
  - The docs say slot 0 means "Off", but the code treats it as a real slot and every instance defaults to it.
  - Unpublished slots read 0.0 dB, so the bars pin at full height until the first publish.
  - Stale data persists after an instance is removed.
- **Bypassed effects click on re-entry** (`realtime.rs:803-826`, `elite_dsp.rs:1051`)
  - The tape and disperser buffers go stale while bypassed.
- **Vocal Surgical automation steps** (`forge-plug/src/lib.rs:1844-1853`)
  - Its parameters are unsmoothed, so automation steps are audible.
- **Delivery checks are looser than the standards** (`delivery.rs:24, 262, 79-84`)
  - A ±1.0 LU tolerance applies to every target, but EBU R128 allows ±0.5 LU, so -23.9 LUFS passes as "ready".
  - The Amazon ceiling is -1 dBTP. Amazon may recommend -2 dBTP; this has not been checked against their current documentation.
- **`remove_dc=false` still removes DC** (`restoration.rs:347-349`)
  - With `declip=true`, DC is still removed, because `remove_dc_and_declip` always subtracts the mean.
- **NaN spreads through spectral unmasking** (`spectral_unmask.rs:74, 108, 127`)
  - `max_reduction_db` and `sensitivity` are clamped, but `clamp` passes NaN through.
  - Target samples are not sanitized, unlike in `spectral_defizz.rs`, so one NaN spreads across every frame it touches.
- **ABC parser gaps** (`forge-theory/src/lib.rs:359, 392+`)
  - `K:` and `L:` are ignored, so the key signature never applies and the unit length is fixed at 1/8.
  - Fractional durations (`A/2`), multi-digit durations (`A12`) and `x` rests are not handled.
  - Header lines such as `w:` and `V:`, and decorations such as `!f!`, are scanned as notes.
- **Ragged channels from odd WAV files** (`forge-core/src/lib.rs:227-249`)
  - `load_wav` builds ragged channels if the total sample count is not divisible by the channel count, and `apply_spectral_unmask` would then panic in `copy_from_slice`.
  - Not yet checked whether `hound` rejects such files first.
- **JSFX install falls back silently** (`forge-reaper/src/lib.rs:25-32`)
  - With `APPDATA` unset (macOS/Linux), `install_jsfx_plugins` writes to `./reaper-effects-installed` and reports "Installed".
- **ffmpeg leftovers and argument injection** (`forge-reaper/src/lib.rs:115-157`, `forge-mcp/src/main.rs:621-641`)
  - When ffmpeg fails, `*.pre_tag.wav` is left behind.
  - An `output` that starts with `-` can be parsed as an ffmpeg option.
- **OSC send reports success regardless** (`forge-reaper/src/lib.rs:374-391`)
  - `send_reaper_osc` reports success for fire-and-forget UDP, accepts addresses without a leading `/`, and binds to 127.0.0.1, so non-loopback hosts fail. CLI only.
- **xtask exits 0 on usage errors** (`xtask/src/main.rs:95-107`)
  - Install usage errors return exit code 0, and `--dest` with no value is silently ignored.
- **xtask install fails on macOS** (`xtask/src/main.rs:124-127`)
  - `std::fs::copy` fails on directory-bundle `.clap` files.

---

## Areas that looked clean

- **STFT overlap-add** in `spectral_defizz.rs` and `spectral_unmask.rs`: Hann analysis and synthesis windows, normalization by Σw², 75% overlap.
- **GCC-PHAT phase alignment:** the lag sign and shift direction are correct, and so is the polarity decision.
- **`AudioBuffer` (forge-core):** constructors, `to_mono`, `to_stereo`, the WAV round trip and dithering.
- **Delivery loudness targets:** correct for Spotify, YouTube, Tidal, Deezer, Apple, Podcast, EBU and ATSC.
- **MIDI and theory helpers:**
  - the pitch-shifter crossfade
  - `snap_midi` and `diatonic_shift_semitones`
  - MIDI writer event ordering and its u28 bounds
- **Realtime indexing and math:**
  - ring masks, the FDN, flutter and preamp ring indexing
  - NaN, Inf and denormal flushing
  - Biquad, Cytomic SVF and K-weighting coefficients
  - allpass stability
  - nih-plug smoother usage
- **Lyria HTTP client:** has a 180 s timeout, and the key is sent in a header.
- **CLI exit codes**, and MCP JSON-RPC argument extraction (no unwraps on user input).
- **RPP string and numeric sanitizing**, and ISRC validation.
- **CI workflows:**
  - actions are pinned by SHA
  - `inputs.tag` is passed via `env`
  - permissions are `contents: read` by default
  - no `pull_request_target`

## Suggested fix order

1. **H8:** confine MCP paths. This is a security issue and the smallest change.
2. **H1, H4, H5:** real-time safety in the plugins.
3. **H2, H3:** limiter transients and latency reporting, which together decide whether the master limiter can be trusted.
4. **H6, H7:** DTW memory, and the de-plosive notch in both the offline and realtime paths.
5. **M16-M19:** worker routing, timeouts and MCP robustness.
6. Add regression tests alongside each fix:
   - an impulse-delay test for the reported latency
   - a single-transient test for the limiter's true peak
   - an allocation guard for the plugin `process()` calls
   - an MCP `dispatch_tool` test for path rejection
