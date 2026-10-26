# Security Policy

ForgeAudio ships native code that runs inside other people's software: VST3/CLAP plugins
loaded into DAWs, a CLI, and an MCP server driven by AI agents. A memory-safety bug or a crash
on malformed input here can take down a user's session or project, so we treat those reports
as security issues.

## Supported versions

ForgeAudio is pre-1.0. Only the latest release and the `main` branch receive security fixes.

| Version        | Supported |
| -------------- | --------- |
| `main`         | Yes       |
| latest `0.x`   | Yes       |
| older releases | No        |

## Reporting a vulnerability

**Please do not open a public issue, pull request, or discussion for a security problem.**

Report it privately through GitHub:
**[Report a vulnerability](https://github.com/The-40-Thieves/audio-forge/security/advisories/new)**
(repository → *Security* → *Report a vulnerability*).

Please include:

- the affected component (plugin name and format, `forge` CLI, `forge-mcp`, `forge-worker`,
  the Python GPU worker, JSFX, or the release/CI pipeline) and the version or commit;
- your OS, and the DAW/host and version for plugin issues;
- steps to reproduce, ideally with a minimal input file, preset/state blob, or host project;
- the impact you observed (crash, hang, memory corruption, data loss, code execution, leaked
  secret, etc.).

### What to expect

- We aim to acknowledge reports within **7 days**, and to follow up with an assessment and an
  expected timeline after that.
- We will work on the fix in a private security advisory and may invite you to collaborate.
- Once a fix is released we publish the advisory, request a CVE where appropriate, and credit
  you unless you prefer to stay anonymous.
- Please give us a reasonable chance to ship a fix before you disclose publicly. We aim to
  resolve confirmed issues within **90 days**.

ForgeAudio is maintained by volunteers. These are goals, not guarantees, and there is no bug
bounty.

## Scope

In scope:

- **Plugins (`crates/forge-plug`, `plugins/forge-*`)**: crashes, hangs, or memory corruption
  in a host caused by loading plugin state or presets, parameter automation, unusual sample
  rates or block sizes, or audio input (including NaN/Inf/denormals); and real-time-thread
  violations that can stall a host.
- **`forge` CLI and `forge-dsp` / `forge-core`**: crashes or memory-safety issues on malformed
  audio, MIDI, or project files.
- **`forge-mcp`**: ways for a tool call to escape the paths it was given, run unintended
  commands, or otherwise act beyond its documented behaviour.
- **`forge-worker` and `python-worker`**: leaking API keys (such as `GEMINI_API_KEY`) or other
  credentials, or executing untrusted remote content.
- **Build and release pipeline**: anything that could let a third party modify published
  release artifacts or CI results.

Out of scope:

- Bugs in DAWs, plugin hosts, or third-party services, unless ForgeAudio triggers them in a
  way we can reasonably prevent.
- `forge-mcp` reading or writing files that the connected MCP client explicitly asked for.
  The client is trusted by design, so run it with the least privilege it needs.
- Audio-quality problems, performance issues, or crashes that need a modified build. Please
  report those as normal issues.

## Known issues

We track these openly because no fix is available from this repository yet:

- **`forge-mcp` external steps (low severity).** `forge-mcp` opens client files through
  directory handles anchored at each `FORGE_MCP_ROOT`, so a path swapped for an escaping
  symlink after it was checked fails to open. Two steps still pass path names to code that
  opens them itself: FFmpeg ISRC tagging (`forge_master_distrokid`) and stem separation
  (`forge_ingest_to_reaper`). The server re-checks those paths right before the call and
  verifies every produced file through the root handle afterwards, but a local process that
  already has write access inside the root could, within that short window, make the step
  write outside it (the call then reports an error). See the MCP file-access section of
  [`docs/ARCHITECTURE_AND_ENGINEERING_GUIDE.md`](docs/ARCHITECTURE_AND_ENGINEERING_GUIDE.md).

- **RUSTSEC-2021-0145 / GHSA-g98v-hv3f-hcfr (`atty`, low severity).** This is pulled in
  through `nih_plug → nih_log`, and no patched version exists. It only affects Windows builds
  that use a custom global allocator returning under-aligned memory, which ForgeAudio does not
  ship.

Fixed here, pending upstream:

- **nih-plug CLAP state loading.** The upstream `nih-plug` CLAP wrapper trusted the length
  prefix of incoming plugin state, so a corrupt or malicious state blob could abort the host
  process. ForgeAudio builds against a patched copy in [`vendor/nih-plug`](vendor/nih-plug/PATCHES.md)
  that bounds the length, and CI runs the full `clap-validator` suite. The copy goes away once
  [robbert-vdh/nih-plug#280](https://github.com/robbert-vdh/nih-plug/pull/280) is released.

## Security practices in this repository

- Every push and pull request runs `cargo audit`, `clippy -D warnings`, tests on Linux,
  Windows and macOS, and pluginval/clap-validator against the built plugins.
- GitHub Actions are pinned to full commit SHAs, and Dependabot keeps Rust crates, Actions
  and Python dependencies up to date.
- CodeQL code scanning and secret scanning are enabled.
