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
- `src/audio_setup.rs`, `main_output_name`: read `names.main_output` instead of
  `names.main_input`, so a custom output port name is not replaced by the input's.

Upstream tracking: issue <https://github.com/robbert-vdh/nih-plug/issues/279>,
fix PR <https://github.com/robbert-vdh/nih-plug/pull/280> (same two fixes; it
caps the length at 256 MiB instead of 64 MiB).

Remove this directory and the `[patch]` entry once #280 (or an equivalent fix)
is merged upstream, then bump the nih-plug revision in `Cargo.lock`.
