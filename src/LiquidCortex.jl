# SPDX-License-Identifier: MIT OR Apache-2.0

"""
    LiquidCortex

GPU-accelerated sparse Liquid State Machine for neuromorphic computing.

Provides two LSM implementations:
- **EnsembleBrain** (`sparse_brain.jl`) — multi-lobe sparse CUDA LSM
  with OU-SDE dynamics, STDP covariance learning, and rolling spike history.
  Reservoir size, connectivity, LIF parameters, and RNG are set via
  `BrainConfig` (defaults: 65,536 neurons/lobe, 4 lobes).
  Configurable input/output dimensions. Default ensemble requires RTX-class
  GPU with ≥14 GB VRAM.

- **Reference LSM** (`reference_lsm.jl`) — 2,048-neuron dense CUDA reservoir for
  rapid prototyping. Configurable input/output dimensions.

On CPU-only systems, the module loads cleanly — types and API functions are defined
but GPU allocations are deferred until a CUDA device is available at runtime.
Check `LiquidCortex._cuda_available[]` to test CUDA availability.
"""
module LiquidCortex

using CUDA
using PrecompileTools

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
              "Core types will load, but step! and GPU operations require a CUDA device."
    end
end

# Caller-facing validation (wrong kwargs / input size).
struct LiquidCortexValidationError <: Exception
    msg::String
end
Base.showerror(io::IO, e::LiquidCortexValidationError) =
    print(io, "LiquidCortexValidationError: ", e.msg)

# ── GPU source files (structs defined at load; GPU allocations deferred to
#    constructors/runtime, guarded by _cuda_available[]) ─────────────────────
include("sparse_brain.jl")
include("brain_lifecycle.jl")
include("reference_lsm.jl")

# ── Public API ───────────────────────────────────────────────────────────────

export SparseBrain, EnsembleBrain, BrainConfig, EnsembleDesynchronizedError
export step!, ensemble_step!, get_output, get_ensemble_output
export compute_reservoir_covariance!, diagnostics, ensemble_diagnostics
export reset!, free!

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
        precompile(step!, (SparseBrain, CuVector{Float32}))
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
