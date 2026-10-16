# AGENTS.md

## Project Overview

LiquidCortex.jl — GPU-accelerated sparse Liquid State Machine (LSM) for neuromorphic computing.
Julia package with CUDA acceleration, cuSPARSE Float16, STDP covariance learning.

## Setup Commands

- Install: `julia --project -e 'using Pkg; Pkg.instantiate()'`
- Run tests: `julia --project -e 'using Pkg; Pkg.test()'`
- Test with coverage: `julia --project -e 'using Pkg; Pkg.test(; coverage=true)'`
- Build docs: `julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()' && julia --project=docs docs/make.jl`

## Architecture

- `src/LiquidCortex.jl` — Module definition, exports, `__init__`
- `src/sparse_brain.jl` — SparseBrain (65k neurons/lobe), EnsembleBrain (4 lobes),
  BrainConfig, `step!()`, STDP
- `src/brain_lifecycle.jl` — `reset!(; keep_weights=true)`, `free!`
- `src/reference_lsm.jl` — 2,048-neuron reference reservoir (lazy-init, configurable dims)
- `test/runtests.jl` — Test suite (CPU + GPU tests gated by `_cuda_available`)
- `docs/make.jl` — Documenter.jl site (CPU-buildable; examples are not doctested)

## Code Style

- Julia standard formatting
- No domain-specific code in core (market/mining removed in PR #12)
- Generic inhibition interface: `step!(brain::SparseBrain, u::CuVector{Float32}; inhibition::Real=0.0, reflex_eta::Real=ETA)`
- Step kwargs: `plasticity=:readout_only` (default), `:recurrent_stdp`, `:none`; plus `sync`, `record_history`, `use_device_noise`, `recurrent_eta`
- Configurable dimensions: `SparseBrain(tau_m::Real; cfg::BrainConfig=BrainConfig(), n_in::Int=14, n_out::Int=16, name::AbstractString="default")` — `tau_m` is stored as `Float32`; `BrainConfig` carries `N`, connectivity, spectral radius, LIF/STDP knobs, and `rng`
- Compat: CUDA.jl `6` (latest); local TDD/verify on Julia 1.13.2

## Testing

- CPU tests always run (package load, API exports, config validation, **host-side
  reservoir kernels**: pair STDP, readout Hebbian, CSC, inhibition, aggregation)
- GPU tests gated by `LiquidCortex._cuda_available[]` (no `@test_skip` literals)
- Stochastic GPU cases seed host `Random` and `CUDA.seed!`
- CI workflows run Julia **1.13.2** only, requested as the `1.13` channel. Compat declares `1.10, 1.11, 1.12, 1.13`, which resolves to the range `[1.10, 2.0)`. 1.10 and 1.11 are never built.
- **CPU smoke:** `.github/workflows/ci.yml` → `ubuntu-latest`
- **Windows tests:** `.github/workflows/windows-ci.yml` → self-hosted runner labels
  `self-hosted`, `Windows`, `X64`, `liquidcortex` (`DESKTOP-4E1V5HV-liquidcortex`).
  The job installs Julia 1.13.2 with juliaup (idempotent if it is already
  installed) and records that bin directory on the runner service account PATH.
  Fork pull requests do not check out onto that machine.
  CUDA on the RTX 5080 runs when `CUDA.functional()` is true; CPU tests still run
  when it is not.
- **GPU tests:** `.github/workflows/gpu-ci.yml` → self-hosted runner labels
  `self-hosted`, `Linux`, `X64`, `gpu` (`ShipOfTheseus` only). The Windows
  runner does not have the `gpu` label and does not run this job. Fork pull
  requests do not check out onto that machine.

## PR Instructions

- Branch naming: `feature/`, `fix/`, `ci/`, `refactor/`, `docs/`
- Run tests before pushing
- All CI checks must pass (Julia 1.13.2, Codacy, CodeRabbit)
- Address all bot review threads before merge
- Pin GitHub Actions to full commit SHAs (not tags)
- Use `julia-actions/julia-processcoverage` for coverage — not Coverage.jl in Project.toml
- README must use pure markdown — no HTML elements (Codacy lints `<a>` and `<img>`)
## Cursor Cloud specific instructions

- Julia is provided via `juliaup` with default channel **1.13**, which is Julia **1.13.2** (admitted by this repo's `1.10, 1.11, 1.12, 1.13` compat, which spans [1.10, 2.0)). Standard setup applies: `julia --project -e 'using Pkg; Pkg.instantiate()'` then `julia --project -e 'using Pkg; Pkg.test()'`.
- The Cursor Cloud VM has **no CUDA GPU**. The test command loads CUDA artifacts and runs the CPU tests; the GPU test block is skipped (this is expected — the suite still passes). GPU paths (`step!` and cuSPARSE ops) require a real device and cannot be exercised here.
