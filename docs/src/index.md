# LiquidCortex.jl

```@docs
LiquidCortex
```

GPU-accelerated sparse liquid state machine for neuromorphic computing.

Production-grade CUDA-accelerated sparse Liquid State Machine (LSM) with
OU-SDE membrane dynamics, multi-lobe ensemble architecture, cuSPARSE mat-vec,
and STDP covariance learning.

On CPU-only systems the module loads cleanly — types and API functions are
defined, but GPU allocations are deferred until a CUDA device is available.
Check `LiquidCortex._cuda_available[]` before constructing a reservoir.

## Installation

The package is not yet in the General registry:

```julia
using Pkg
Pkg.add(url="https://github.com/rmems/LiquidCortex.jl")
```

## Quick start

This example needs a CUDA device. On CPU-only hosts, `using LiquidCortex`
succeeds but `SparseBrain` / `step!` will fail until a GPU is present.

```julia
using LiquidCortex, CUDA

brain = SparseBrain(20.0f0)  # τ_m = 20 ms, default n_in=14, n_out=16
u = CUDA.zeros(Float32, 14)
step!(brain, u; inhibition=0.3f0)
output = get_output(brain)
```

See [API](api.md) for the exported surface. Docstrings are also available from
REPL helpmode (`?step!`).

## Features

- `SparseBrain` — 65,536-neuron sparse reservoir, Float16 weights on GPU
- Configurable input/output dimensions (`n_in`, `n_out`)
- OU-SDE dynamics: `dV = ((V_rest - V)/τ + I_rec + I_ext)dt + σ dW`
- `EnsembleBrain` — four lobes with different time constants
- Generic inhibition interface — caller provides a stress signal

Requires **CUDA.jl 6.x**. CI and local verification use **Julia 1.13**.
