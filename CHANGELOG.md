# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Under 0.x, a breaking change bumps the minor version (`0.2.0` → `0.3.0`).

The only git tag in the repository is `v0.1.0`. `Project.toml` currently
reads `0.2.0` but that string was never tagged, and later breaking PRs did not
bump it. **The next release must be `0.3.0`.** Do not land another breaking
change under `0.2.0`.

## [Unreleased]

### Added

- `CHANGELOG.md`, TagBot, and CompatHelper (Julia `[compat]` scanner).
- Dependabot `julia` ecosystem (Dependabot now supports Julia; `#64` noted the
  previous gap). Deprecated Dependabot `reviewers:` key removed in favor of
  `CODEOWNERS`.
- Documenter.jl site (`docs/make.jl`, `docs/src/`) and a docs CI workflow.
- Root `LICENSE` (dual MIT / Apache-2.0) so registry AutoMerge can find an
  OSI-approved license file. `LICENSE-MIT` and `LICENSE-APACHE` remain.
- Stdlib `[compat]` entries required by AutoMerge (`LinearAlgebra`, `Printf`,
  `Random`, `SparseArrays`, `Statistics`).
- `CODEOWNERS`, PR template (the `REVIEW.md` checklist), issue templates,
  `CONTRIBUTING.md`, `SECURITY.md`, and `.JuliaFormatter.toml`.
- CI-only Manifest at `.github/ci-manifest/Manifest.toml` (Julia 1.13). Root
  `Manifest.toml` stays gitignored so library consumers are not pinned.

### Changed

- GPU CI scopes `LIQUIDCORTEX_SENTRY_DSN` to the test step instead of the job.
- `sentry-release.yml` no longer parses a workspace `.env` into `$GITHUB_ENV`.

### Fixed

- Fake doctest rerun and coverage-artifact gitignore: `#66`.
- Folding the duplicate Codecov workflow into `ci.yml`: `#69` (landed).

## [0.2.0] - untagged

`Project.toml` was bumped `0.1.0` → `0.2.0` for the `SpikenautLSM` →
`LiquidCortex` rename (`#4`, 2026-05-17). No `v0.2.0` tag exists. The breaking
changes below also shipped under this version string.

### Breaking

- Rename `SpikenautLSM` → `LiquidCortex` (`#4`).
- `#12`: removed `MarketPulse`, `decode_market_pulse`, and `pulse_to_input`;
  `step!` / `ensemble_step!` take keyword-only `inhibition`; relicensed GPL-3.0 →
  MIT OR Apache-2.0; deleted `monte_carlo_paths!`.
- `#45`: CUDA compat `5` → `6`; experimental `step!` kwargs (`plasticity`,
  `sync`, `record_history`, `use_device_noise`, `recurrent_eta`); new
  `LiquidCortexValidationError`.

### Added

- Configurable `n_in` / `n_out` on `SparseBrain` and `EnsembleBrain`.
- Generic inhibition interface (caller-supplied stress signal).
- Opt-in Sentry capture via `LIQUIDCORTEX_SENTRY_DSN` / `enable_telemetry!`
  (`#32`, `#67` — does not read process-wide `SENTRY_DSN`).
- Public API docstrings (`#52`).
- Reference LSM test coverage (`#33`, `#50`).
- PrecompileTools workload for the SparseBrain hot path (`#51`).
- Constructor rejection of non-positive `tau_m` (`#65`).

## [0.1.0] - 2026-03-23

Initial packaging as `SpikenautLSM.jl` (tagged `v0.1.0`).
