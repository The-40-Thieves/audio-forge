@AGENTS.md

## Gemini / Antigravity Execution Notes

- Read `AGENTS.md` (included above) and `.entire/agent-guide.md` before modifying files.
- Prefer exact verification commands from `AGENTS.md`; use PowerShell PATH prepending (`~/.cargo/bin`, `~/scoop/shims`, `~/.local/bin`) when running Rust, `uv`, or `node` tools on Windows, and `wsl python3` when running Python Z3/unittest checks if `z3-solver` is installed in WSL.
