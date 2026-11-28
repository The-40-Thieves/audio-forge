# Vendored `nih_plug`

This is the `nih_plug` crate (plus its `nih_plug_derive` path dependency) from
<https://github.com/robbert-vdh/nih-plug> at commit
`de421011f41a6d10fc8c7a6084e4f4dee0143683`, wired in through
`[patch."https://github.com/robbert-vdh/nih-plug.git"]` in the root `Cargo.toml`.
The other nih-plug crates (`nih_plug_egui`, `nih_plug_xtask`) still come from git.

Changes from upstream:

- `Cargo.toml`: dropped the upstream `[workspace]` and `[profile.*]` sections.
- `src/wrapper/clap/wrapper.rs`, `ext_state_load`:
  - Reject state whose 8-byte length prefix is over 64 MiB, and read the body in
    64 KiB chunks instead of calling `Vec::with_capacity` on the untrusted
    length. Random state previously aborted the host (clap-validator
    `state-invalid-random`).
  - After a successful load, schedule `Task::RescanParamValues` so the host gets
    `clap_host_params.rescan(CLAP_PARAM_RESCAN_VALUES)` on the main thread
    (clap-validator `state-reproducibility-basic`, `-binary`, `-buffered`).
- `src/wrapper/clap/wrapper.rs`, `is_activating` & `Task::LatencyChanged`:
  - Track `is_activating: AtomicBool` across `activate()` via `ActivatingGuard` and
    invoke `host.request_restart()` from `Task::LatencyChanged` when dynamic latency
    changes outside `activate()`.
- `src/audio_setup.rs`, `main_output_name`: read `names.main_output` instead of
  `names.main_input`, so a custom output port name is not replaced by the input's.

Upstream tracking: issue <https://github.com/robbert-vdh/nih-plug/issues/279>,
fix PR <https://github.com/robbert-vdh/nih-plug/pull/280> (covers the `ext_state_load`
fixes with a 256 MiB cap). Do NOT remove this directory or the `[patch]` entry
until #280 AND the `is_activating` / `Task::LatencyChanged` + `main_output_name`
fixes are merged upstream (upstream `robbert-vdh/nih-plug` is in maintenance mode).
