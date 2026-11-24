<!-- entire-agent:begin -->
Read .entire/agent-guide.md for this repository's workflow, source inspection, and verification guidance.
@.entire/agent-guide.md
<!-- entire-agent:end -->

## Boundaries

- Leave `.entire/agent-guide.md` and the `entire-agent` blocks to `entire brain setup`, which regenerates them; to change them, re-run it rather than editing by hand.
- Don't commit secrets, tokens, `.env` files or `.entire/settings.local.json`. If one is committed by mistake, stop and tell the owner so it can be rotated.
- Session transcripts and checkpoints go only to the private `The-40-Thieves/entire-checkpoints` repo, not this origin. If `entire status` shows another destination, stop and ask before pushing.
- Merging to `main`, force-pushing or rewriting published history needs the owner's go-ahead in the current conversation.

## Architecture & Invariants

- **Workspace layering:** Keep `forge-core` pure (audio buffers, WAV/FFmpeg I/O, BS.1770 audit, shared types like `StemBundle`), `forge-dsp` pure real-time & offline DSP, `forge-vocal` / `forge-theory` / `forge-reaper` domain-specific, and `forge-engine` / `forge-worker` as orchestration layers above them; avoid back-edges or pulling GUI (`eframe`/`egui`) or `nih_plug` dependencies down into `forge-core` or `forge-dsp`.
- **Real-time audio safety:** Pre-allocate `Realtime*` processors in `forge-dsp` inside `new(sample_rate)`, avoid allocations or locks on the audio thread (`process_stereo_sample`, `Plugin::process`, `Plugin::reset`), flush denormals (`flush_denormal_f64`), and sanitize `NaN`/`±Inf` inputs and parameters (`sanitize_sample_f64`, `sanitize_param_f64`); if a test or diagnostic path intentionally allocates, guard it explicitly and verify production paths via `crates/forge-dsp/tests/realtime_no_alloc.rs`.
- **Plugin ABI & state reset discipline:** Keep `#[id = "..."]` parameter attributes stable in `crates/forge-plug/` (only rename a persisted ID with an explicit state migration and owner sign-off), match `ParamRef` UI bindings to `#[id]` strings, report latency changes on `low_latency_mode` / mode switches via `context.set_latency_samples(...)`, and reset internal DSP state at the cached `self.sample_rate` inside `Plugin::reset()`.
- **MCP path sandboxing:** Anchor `forge-mcp` file I/O through `cap-std` (`crates/forge-mcp/src/sandbox.rs` and `anchored.rs`) or stage external tools through private process temp dirs (`crates/forge-mcp/src/external.rs`), and reject unvalidated path joins, raw `..` traversal, or unbounded downloads/subprocesses unless an explicit, documented security exception is approved by the owner.
- **Documentation / CLI / MCP parity:** `docs/ARCHITECTURE_AND_ENGINEERING_GUIDE.md` and `README.md` are checked by tests (`architecture_guide_lists_every_tool` checks all MCP tool names); keep documented tool names, CLI subcommands, and `StemBundle` slot counts accurate when editing them.

## Verification Commands

### Windows PowerShell
```powershell
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:USERPROFILE\scoop\shims;$env:USERPROFILE\.local\bin;$env:PATH"
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
node scripts/check-license-coherence.mjs
uv lock --project python-worker --check
uvx ruff format --check python-worker scripts
uvx ruff check python-worker scripts
uvx mypy --ignore-missing-imports python-worker/forge_gpu_worker.py python-worker/test_forge_gpu_worker.py scripts/verify_dsp_invariants_z3.py
wsl python3 -m unittest discover -s python-worker -v
```

### Linux / CI
```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
node scripts/check-license-coherence.mjs
uv lock --project python-worker --check
ruff check python-worker scripts
ruff format --check python-worker scripts
mypy --ignore-missing-imports python-worker/forge_gpu_worker.py python-worker/test_forge_gpu_worker.py scripts/verify_dsp_invariants_z3.py
python3 -m unittest discover -s python-worker -v
python3 scripts/verify_dsp_invariants_z3.py
```
