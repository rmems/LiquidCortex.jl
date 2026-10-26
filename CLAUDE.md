@AGENTS.md

# CLAUDE.md

Claude-specific notes only. Setup commands, architecture, code style, testing policy, and PR rules are in
`AGENTS.md` (imported above); do not duplicate them here.

## Commands

Run from the repo root. Julia was not installed in the environment this file was written in, so no command
below was executed. Commands are taken from `AGENTS.md` and `.github/workflows/`.

| Command | Status |
|---|---|
| `julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'` | not run (no `julia` on PATH); CI runs this |
| `julia --project=. -e 'using Pkg; Pkg.test(; coverage=true)'` | not run (no `julia` on PATH); CI runs this on Julia 1.13 |
| `julia --project -e 'using Pkg; Pkg.test()'` | not run (no `julia` on PATH) |
| `julia --project=docs docs/make.jl` | not run (no `julia` on PATH); `documentation.yml` runs it |

- CI (`ci.yml`) copies `.github/ci-manifest/Manifest.toml` to the repo root before instantiating; the root
  `Manifest.toml` is gitignored.
- `.JuliaFormatter.toml` exists (indent 4, margin 92) but no CI job or script runs a formatter, so there is no
  format gate to run.
- `gpu-ci.yml` needs a self-hosted CUDA runner; Codecov upload needs a secret. Neither is a local gate.
- On a machine without a GPU, `Pkg.test()` loads CUDA and runs the CPU tests only; GPU tests are gated by
  `LiquidCortex._cuda_available[]`.

## Architecture map

- `src/LiquidCortex.jl`: module definition and exports. Includes the three files below.
- `src/sparse_brain.jl`: `SparseBrain`, `EnsembleBrain`, `BrainConfig`, `step!`, STDP (largest file).
- `src/brain_lifecycle.jl`: `reset!` and `free!` for brains.
- `src/reference_lsm.jl`: 2,048-neuron reference reservoir.
- `test/runtests.jl`: single test file; `examples/brain_standalone.jl`: usage example (not doctested).
- `docs/`: Documenter.jl site (`docs/make.jl`, `docs/src/`).

## Invariants and non-goals

- Keep domain-specific code (market, mining, crypto, hardware telemetry) out of the core; `REVIEW.md` checks this.
- Prefer generic signatures (`AbstractVector`, `Real` kwargs); no hardcoded dimensions, use `n_in`/`n_out`.
- New GPU tests must be gated by `LiquidCortex._cuda_available[]`.
- Update `CHANGELOG.md` for user-facing changes and `README.md` if public API changes (`REVIEW.md`).
- `README.md` must stay pure markdown (no HTML elements); GitHub Actions are pinned to full commit SHAs.
- Every `.jl` file (src, test, examples, docs) starts with `# SPDX-License-Identifier: MIT OR Apache-2.0`. Keep it on new files.

## Gotchas

- `Project.toml` has no publish `exclude`/`files` list, so adding this file needs no manifest change.
- Do not commit `Manifest.toml` at the root; do not add Coverage.jl to `Project.toml`.
- `.cursor/` and `.explore-notes/` are gitignored, so keep shared agent guidance in this file or `AGENTS.md`.
- Edit only what the task needs.
