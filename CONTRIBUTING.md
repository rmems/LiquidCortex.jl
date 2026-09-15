# Contributing

## Setup

```julia
julia --project -e 'using Pkg; Pkg.instantiate()'
```

Julia **1.13** is what CI runs. Compat admits `1.10`–`<2.0`.

## Tests

```julia
julia --project -e 'using Pkg; Pkg.test()'
```

CPU tests always run. GPU tests are gated by `LiquidCortex._cuda_available[]`
and are skipped on machines without CUDA. That skip is expected on CPU-only
hosts (including Cursor Cloud).

Coverage locally:

```julia
julia --project -e 'using Pkg; Pkg.test(; coverage=true)'
```

`*.cov` and `lcov.info` are gitignored.

## Formatting

`.JuliaFormatter.toml` records default JuliaFormatter settings (4-space
indent, 92-column margin). Match surrounding files.

## Pull requests

`.github/PULL_REQUEST_TEMPLATE.md` is the `REVIEW.md` checklist in the GitHub
UI. For user-facing changes, update `CHANGELOG.md`. Breaking changes under 0.x
require a minor bump in `Project.toml` (`0.2.0` → `0.3.0`) and migration notes.

Do not put domain-specific code (market, mining, hardware telemetry) in `src/`.

## Docs

```julia
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

`deploydocs` is a no-op unless running on CI against `main` or a tag.

## Security

See [SECURITY.md](SECURITY.md). Do not open public issues for vulnerabilities.
