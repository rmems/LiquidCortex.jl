![LiquidCortex](docs/logo.png)

# LiquidCortex.jl

GPU-accelerated sparse liquid state machine for neuromorphic computing

[![Julia](https://img.shields.io/badge/language-Julia-9558B2)](https://julialang.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](LICENSE)
[![Coverage](https://codecov.io/gh/rmems/LiquidCortex.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/rmems/LiquidCortex.jl)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://rmems.github.io/LiquidCortex.jl/dev/)

---

Production-grade CUDA-accelerated sparse Liquid State Machine (LSM) with
OU-SDE membrane dynamics, multi-lobe ensemble architecture, cuSPARSE mat-vec,
and STDP covariance learning.

## Features

- `SparseBrain` — configurable reservoir via `BrainConfig`: N neurons, connectivity, spectral radius, Float16 sparse weights on GPU
- Configurable input/output dimensions (`n_in`, `n_out`) and an explicit `rng` for reproducible topology / noise
- OU-SDE dynamics: `dV = ((V_rest - V)/τ + I_rec + I_ext)dt + σ dW`
- cuSPARSE Float16 sparse mat-vec on GPU (fits 65k neurons in 16 GB VRAM)
- STDP covariance learning with eligibility traces
- Configurable rolling spike history buffer (circular, on-GPU; default 1000 ticks)
- `EnsembleBrain` — multi-lobe: multiple reservoirs with different time constants (`taus` / `weights`)
- Generic inhibition interface — caller provides a stress signal

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/rmems/LiquidCortex.jl")
```

## Quick Start

```julia
using LiquidCortex, CUDA, Random

# Create a 65,536-neuron sparse LSM lobe
brain = SparseBrain(20.0f0)  # τ_m = 20ms, default n_in=14, n_out=16

# Or with custom dimensions:
#   brain = SparseBrain(20.0f0; n_in=8, n_out=4)

# Or a small, seeded reservoir for sweeps / tests:
#   cfg = BrainConfig(N=256, spectral_radius=0.8f0, rng=Random.Xoshiro(42))
#   brain = SparseBrain(20.0f0; cfg=cfg, n_in=8, n_out=4)

# Or create the full 4-lobe ensemble (262,144 neurons)
ensemble = EnsembleBrain()

# Step the reservoir with an input vector — length(u) must equal brain.n_in
u = CUDA.zeros(Float32, 14)
step!(brain, u; inhibition=0.3f0)

# Read the output
output = get_output(brain)
```

## Public API

| Type / Function | Description |
|-----------------|-------------|
| `BrainConfig(; N, conn_prob, spectral_radius, rng, ...)` | Reservoir hyperparameters (defaults match module constants) |
| `SparseBrain(tau_m; cfg, n_in, n_out)` | Create a sparse reservoir lobe (default `N=65_536`) |
| `EnsembleBrain(; n_in, n_out, cfg, taus, weights)` | Create a multi-lobe ensemble (default 4 × 65,536) |
| `step!(brain, u; inhibition, reflex_eta, ...)` | Execute one simulation timestep (see experimental kwargs) |
| `ensemble_step!(eb, u; inhibition, reflex_eta, reflex_signal, ...)` | Step all lobes and aggregate |
| `get_output(brain)` | Copy readout from GPU to CPU |
| `get_ensemble_output(eb)` | Copy aggregated readout |
| `reset!(brain; keep_weights=true)` / `reset!(eb)` | Rewind neuron state; keep weights for another trial |
| `free!(brain)` / `free!(eb)` | Release GPU buffers into the CUDA.jl pool |
| `compute_reservoir_covariance!(brain)` | Compute subsampled covariance matrix |
| `diagnostics(brain)` | Return diagnostic string |
| `ensemble_diagnostics(eb)` | Per-lobe diagnostic summary |
| `EnsembleDesynchronizedError` | Raised when lobe clocks disagree or a prior ensemble step failed |
| `enable_telemetry!(dsn)` | Opt in to Sentry capture (HTTPS DSN; does not read `SENTRY_DSN`) |

## Experimental step API

LiquidCortex is an experimental Julia package. Defaults are intentional:

| Keyword | Default / values | Meaning |
|---------|------------------|---------|
| `plasticity` | **default** `:readout_only` | Frozen recurrent `W`; Hebbian `W_out` every 10 ticks |
| | opt-in `:recurrent_stdp` | Experimental pair STDP on sparse edges every tick |
| | opt-in `:none` | No weight updates |
| `recurrent_eta` | default `1f-4` | Learning rate for `:recurrent_stdp` |
| `sync` | default `true` | `CUDA.synchronize()` at end of step; host spike diagnostics only when true |
| `record_history` | default `true` | Write spike history; if false, covariance helpers may see stale/incomplete history |
| `use_device_noise` | default `false` | Host Gaussian noise from `brain.cfg.rng`; `true` uses CUDA's device RNG (`CUDA.seed!`, not `Random.seed!`) with a one-shot host fallback if the device generator fails |

Recurrent reservoir weights are **not** trained under the default path.
Requires **CUDA.jl 6.x**. Local verification and CI workflows use **Julia 1.13**.

## OU-SDE Membrane Dynamics

```
dV = ((V_rest - V)/τ  +  W_rec·s(t)  +  W_in·x(t)) dt  +  σ dW
```

Discretized as Euler-Maruyama. Spike when `V > θ_dynamic` (the inhibition-shifted
threshold); reset to `V_reset` = −70 mV, which is distinct from `V_rest` = −65 mV.

*Ornstein & Uhlenbeck (1930); Maass, Natschläger & Markram (2002)*

## STDP Covariance Learning

```
ΔW_ij = η (⟨s_i s_j⟩ - ⟨s_i⟩⟨s_j⟩)
```

Computed on a subsampled window (`BrainConfig.cov_subsample`, default 8192,
clamped to `N`) to avoid O(N²) blow-up.

*Bi & Poo (1998); Hebb (1949)*

## Provenance

Extracted from [Eagle-Lander](https://github.com/rmems/Eagle-Lander), a private
neuromorphic GPU supervisor. The LSM core has been fully decoupled from
domain-specific logic so it works with any time-series application.

## License

Licensed under either of:

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.
