---
name: principal-neuro-symbolic-engineer
description: Apex (Level 5) Principal Neuro-Symbolic Software & DSP Engineering skill for ForgeAudio. Enforces Type-Driven Design, NaN/Inf/Denormal State-Poisoning Immunity, Property-Based Testing (proptest), Adversarial Invariant Verification, Lyapunov Stability Proofs, Z3 SMT Grounding, and 3-Question Systemic Root-Cause Sweeps.
---

# Principal Neuro-Symbolic DSP & Systems Engineer Skill (`audio-forge`)

Activate this skill whenever designing, modifying, or verifying DSP kernels, VST3/CLAP plugins, or audio engines in `audio-forge`.

## 1. Non-Negotiable Real-Time Audio & Numerical Invariants
1. **Zero Audio-Thread Heap Allocations:** All `Realtime*` processors in `forge-dsp` and `forge-vocal` must pre-allocate buffers in `new(sample_rate)` and perform zero heap allocations in `process_stereo_sample()`.
2. **State-Poisoning Immunity (`sanitize_sample` & `flush_denormal`):**
   - Never allow `f32::NAN`, `f32::INFINITY`, or `f32::NEG_INFINITY` inputs or parameter values to enter recursive filter states (`Biquad::z1/z2`, `AdaaTanh::x_prev/f1_prev`, envelope followers, or FDN delay lines).
   - Flush subnormal floating-point values (`|x| < 1e-30 -> 0.0`) inside feedback loops to prevent 10x–50x x86/ARM denormal CPU spikes during silence decay.
3. **Safe Clamping & Overflow Prevention:**
   - Never call `x.clamp(min, max)` with dynamic bounds unless `min <= max` is guaranteed (use `sanitize_clamp(x, min, max)`).
   - Promote `f32` stereo inputs to `f64` *before* summing (`0.5 * (f64::from(l) + f64::from(r))`) to prevent `f32` addition overflow.

## 2. Mandatory Verification Gate
```powershell
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release
& "E:\Developer Box\Scripts\gitleaks.exe" detect --source . --no-git
```
