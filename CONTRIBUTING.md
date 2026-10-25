# Contributing to ForgeAudio

Thanks for helping out! This guide covers how to get a working build, what CI checks, and
what we look for in a pull request.

For security problems, follow [SECURITY.md](SECURITY.md) and do **not** open a public issue.
Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting set up

- **Rust:** the latest stable toolchain (`rustup update stable`), with `rustfmt` and `clippy`.
- **Linux only:** the system libraries nih-plug's GUI and audio backends need:

  ```bash
  sudo apt-get install libasound2-dev libgl-dev libjack-dev libx11-xcb-dev libxcb1-dev \
    libxcb-dri2-0-dev libxcb-icccm4-dev libxcursor-dev libxkbcommon-dev \
    libxcb-shape0-dev libxcb-xfixes0-dev
  ```

- **Python (only for `python-worker/` or `scripts/`):** Python 3.10+ with `ruff` and `z3-solver`.
- **Optional:** `ffmpeg` on `PATH` for non-WAV audio I/O.

```bash
cargo build --workspace
cargo test --workspace
cargo xtask bundle forge-opensoothe --release   # one plugin -> target/bundled/
cargo xtask bundle-all --release                # all 10 plugins
```

## Before you open a pull request

Run the same checks as CI:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
```

If you touched Python:

```bash
ruff check python-worker scripts
ruff format --check python-worker scripts
python scripts/verify_dsp_invariants_z3.py
```

CI also bundles every plugin on Linux, Windows and macOS, validates the VST3 builds with
pluginval and the CLAP builds with clap-validator, and runs `cargo audit` and the CodSpeed
benchmarks.

## What we look for

- **Real-time safety.** `process()` must not allocate, lock, block, or do I/O. The workspace
  enables nih-plug's `assert_process_allocs`, so debug builds abort on audio-thread
  allocations; test your change in a debug build.
- **Robust input handling.** DSP code must stay finite on NaN, Inf and denormal input and on
  extreme parameters. Add a case to `crates/forge-dsp/tests/proptest_invariants.rs` when you
  add a processor.
- **Evidence for DSP claims.** Loudness, true-peak and other measurement code should have
  reference-signal tests (see `crates/forge-dsp/tests/ebu_reference.rs`). Back performance
  claims with a benchmark in `crates/*/benches`.
- **Plugin compatibility.** Don't change a parameter's `#[id]`, a plugin's `CLAP_ID`, or its
  `VST3_CLASS_ID`, since that breaks users' saved projects. Give new `FloatParam`s a
  `with_value_to_string` formatter so values round-trip through text.
- **Focused changes.** Keep refactors, formatting churn and dependency bumps out of feature or
  fix PRs where you can.

## Commits and pull requests

- Use [Conventional Commits](https://www.conventionalcommits.org/) style messages, for example
  `fix(dsp): …`, `feat(plug): …`, `ci: …`, `docs: …`.
- Fill in the pull request template: what changed, why, and how you tested it.
- Keep a PR to one logical change; it's fine to stack PRs.
- By contributing you agree your work is licensed under the
  [Apache License 2.0](LICENSE), the same as the project.
