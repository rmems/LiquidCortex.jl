# SPDX-License-Identifier: MIT OR Apache-2.0

"""
    LiquidCortex

GPU-accelerated sparse Liquid State Machine for neuromorphic computing.

Provides two LSM implementations:
- **EnsembleBrain** (`sparse_brain.jl`) — 4-lobe, 65,536-neuron/lobe sparse CUDA LSM
  with OU-SDE dynamics, STDP covariance learning, and rolling 1,000-tick spike history.
  Configurable input/output dimensions. Requires RTX-class GPU with ≥14 GB VRAM.

- **Reference LSM** (`reference_lsm.jl`) — 2,048-neuron dense CUDA reservoir for
  rapid prototyping. Configurable input/output dimensions.

On CPU-only systems the module loads cleanly — types and API functions are defined
without touching a device. Constructing `SparseBrain` / `EnsembleBrain` or
calling `run_lsm_step` requires a CUDA GPU and fails immediately (no multi-GB
host COO draw). The 2,048-neuron reference LSM defers its GPU allocations until
the first step. Check `LiquidCortex._cuda_available[]` to test CUDA availability.
"""
module LiquidCortex

using CUDA
using CommonSolve
using PrecompileTools

import CommonSolve: step!

# ── CUDA availability flag ────────────────────────────────────────────────
# Checked at __init__ time. All GPU allocations are deferred until this is true.
const _cuda_available = Ref{Bool}(false)

function __init__()
    _cuda_available[] = false

    if CUDA.functional()
        _cuda_available[] = true
        @info "LiquidCortex: CUDA functional — GPU kernels available on $(CUDA.name(CUDA.device()))."
    else
        @warn "LiquidCortex: No CUDA-capable GPU found. " *
              "Core types will load, but SparseBrain/EnsembleBrain construction, " *
              "step!, and GPU operations require a CUDA device."
    end
end

# Caller-facing validation (wrong kwargs / input size).
struct LiquidCortexValidationError <: Exception
    msg::String
end
Base.showerror(io::IO, e::LiquidCortexValidationError) =
    print(io, "LiquidCortexValidationError: ", e.msg)

# Fail before any host COO / device allocation. `min_vram_gb` is the
# documented floor for SparseBrain / EnsembleBrain (≥14 GB).
function _require_cuda(context::AbstractString; min_vram_gb::Union{Nothing,Real}=nothing)
    extra = min_vram_gb === nothing ? "" : " with ≥$(min_vram_gb) GB VRAM"
    if !_cuda_available[]
        error("$context requires a CUDA GPU$extra. No CUDA device available.")
    end
    min_vram_gb === nothing && return nothing
    total_gb = CUDA.total_memory() / 1e9
    total_gb >= Float64(min_vram_gb) && return nothing
    error("$context requires a CUDA GPU$extra. This device reports $(round(total_gb; digits=1)) GB.")
end

# ── GPU source files (structs defined at load; GPU allocations deferred to
#    constructors/runtime, guarded by _cuda_available[]) ─────────────────────
include("sparse_brain.jl")
include("brain_lifecycle.jl")
include("reference_lsm.jl")

# ── Public API ───────────────────────────────────────────────────────────────

export SparseBrain, EnsembleBrain, EnsembleDesynchronizedError
export step!, ensemble_step!, get_output, get_ensemble_output
export compute_reservoir_covariance, compute_reservoir_covariance!
export spikes, membrane, traces
export diagnostics, ensemble_diagnostics
export reset!, free!
export LiquidCortexValidationError, ETA, MAX_INHIBITION
export run_lsm_step, run_lsm_step_str, REF_N, REF_IN_DEFAULT, REF_OUT_DEFAULT

# Warm method inference at install time. Do **not** construct SparseBrain or
# EnsembleBrain here: each lobe is 65,536 neurons, `Pkg.test()` re-precompiles
# in-process, and `CUDA.device_reset!` is a no-op on CUDA.jl 6.3 — executing
# the constructors starved the 16 GB self-hosted GPU suite (RM-333 / #51).
# `__init__` has not run, so probe the device directly. Skip the reference LSM.
@compile_workload begin
    _validate_plasticity_kwargs(; plasticity = :readout_only, recurrent_eta = 1.0f-4)
    _validate_reset_kwargs(; keep_weights = true)
    if CUDA.functional()
        precompile(SparseBrain, (Float32,))
        precompile(SparseBrain, (Float64,))
        precompile(step!, (SparseBrain, CuVector{Float32}))
        precompile(step!, (SparseBrain, Vector{Float32}))
        precompile(get_output, (SparseBrain,))
        precompile(reset!, (SparseBrain,))
        precompile(free!, (SparseBrain,))
        precompile(EnsembleBrain, ())
        precompile(ensemble_step!, (EnsembleBrain, CuVector{Float32}))
        precompile(get_ensemble_output, (EnsembleBrain,))
        precompile(reset!, (EnsembleBrain,))
        precompile(free!, (EnsembleBrain,))
    end
end

end # module LiquidCortex
