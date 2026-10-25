## What changed

<!-- A short summary of the change. Link the issue it fixes, if any: "Fixes #123". -->

## Why

<!-- The problem this solves, or the motivation behind it. -->

## How it was tested

<!-- Commands you ran, hosts or DAWs you tried, and before/after numbers for DSP or performance changes. -->

## Checklist

- [ ] `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D warnings` pass
- [ ] `cargo test --workspace` passes, and new behaviour has tests
- [ ] No allocations, locks, or I/O added on the audio thread (`process()`)
- [ ] No changes to existing parameter IDs, `CLAP_ID`, or `VST3_CLASS_ID` (or they are called out above)
- [ ] Docs and README updated if user-facing behaviour changed
