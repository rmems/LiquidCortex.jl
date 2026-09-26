# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Explicit lifecycle for SparseBrain / EnsembleBrain. Constructors allocate
# hundreds of megabytes of device buffers with no paired release; these
# helpers restore neuron state without reallocating, or return those buffers
# to the CUDA.jl pool immediately.

"""CPU-safe `reset!` keyword checks (no GPU types)."""
function _validate_reset_kwargs(; keep_weights::Bool)
    keep_weights || throw(LiquidCortexValidationError(
        "reset!(; keep_weights=false) is unsupported: re-sampling reservoir " *
        "weights requires a constructor seed (GitHub #57)"))
    return nothing
end

"""
    reset!(brain::SparseBrain; keep_weights=true) -> SparseBrain

Restore neuron state, traces, history, and diagnostics to constructor
defaults. Recurrent, input, and readout weights are kept so the same
reservoir can be reused across trials.

# Arguments
- `brain::SparseBrain`: lobe to rewind (mutated in place)

# Keyword Arguments
- `keep_weights::Bool=true`: must be `true`. `false` would re-draw `W` /
  `W_in` / `W_out` and is rejected until a constructor seed exists
  (GitHub #57).

# Returns
- the same `brain`

# Examples
```julia
using LiquidCortex, CUDA
brain = SparseBrain(20.0f0; n_in=8, n_out=4)
step!(brain, CUDA.zeros(Float32, 8))
reset!(brain)
brain.tick_count == 0
```
"""
function reset!(brain::SparseBrain; keep_weights::Bool=true)
    _validate_reset_kwargs(; keep_weights=keep_weights)
    fill!(brain.V, V_REST)
    fill!(brain.S, 0.0f0)
    fill!(brain.refrac, Int32(0))
    fill!(brain.S_f16, Float16(0))
    fill!(brain.I_rec, 0.0f0)
    fill!(brain.I_ext, 0.0f0)
    fill!(brain.noise, 0.0f0)
    fill!(brain.trace_pre, 0.0f0)
    fill!(brain.trace_post, 0.0f0)
    fill!(brain.output, 0.0f0)
    fill!(brain.history, 0.0f0)
    brain.hist_idx = 1
    brain.hist_full = false
    brain.v_thresh_dynamic = Float32(V_THRESH)
    brain.tick_count = 0
    brain.total_spikes = 0
    brain.last_spike_rate = 0.0f0
    CUDA.synchronize()
    return brain
end

"""
    reset!(eb::EnsembleBrain; keep_weights=true) -> EnsembleBrain

[`reset!`](@ref reset!(::SparseBrain)) every lobe and zero the aggregated
readout. Per-lobe weights and aggregation weights are kept.

# Arguments
- `eb::EnsembleBrain`: ensemble to rewind (mutated in place)

# Keyword Arguments
- `keep_weights::Bool=true`: forwarded to each lobe; see
  [`reset!(::SparseBrain)`](@ref)

# Returns
- the same `eb`
"""
function reset!(eb::EnsembleBrain; keep_weights::Bool=true)
    _validate_reset_kwargs(; keep_weights=keep_weights)
    for lobe in eb.lobes
        reset!(lobe; keep_weights=keep_weights)
    end
    fill!(eb.agg_output, 0.0f0)
    CUDA.synchronize()
    return eb
end

# Eager device-buffer release. Public `free!` synchronizes first so
# pending `sync=false` kernels finish; this helper only returns storage
# to the CUDA.jl pool (`unsafe_free!` is a no-op after the first call).
function _free_device_buffers!(brain::SparseBrain)
    CUDA.unsafe_free!(brain.W)
    CUDA.unsafe_free!(brain.pre_idx)
    CUDA.unsafe_free!(brain.post_idx)
    CUDA.unsafe_free!(brain.W_in)
    CUDA.unsafe_free!(brain.W_out)
    CUDA.unsafe_free!(brain.V)
    CUDA.unsafe_free!(brain.S)
    CUDA.unsafe_free!(brain.refrac)
    CUDA.unsafe_free!(brain.S_f16)
    CUDA.unsafe_free!(brain.I_rec)
    CUDA.unsafe_free!(brain.I_ext)
    CUDA.unsafe_free!(brain.noise)
    CUDA.unsafe_free!(brain.trace_pre)
    CUDA.unsafe_free!(brain.trace_post)
    CUDA.unsafe_free!(brain.output)
    CUDA.unsafe_free!(brain.history)
    CUDA.unsafe_free!(brain.u_buf)
    return nothing
end

function _free_ensemble_agg!(eb::EnsembleBrain)
    CUDA.unsafe_free!(eb.agg_output)
    return nothing
end

function _free_ensemble_buffers!(eb::EnsembleBrain)
    for lobe in eb.lobes
        _free_device_buffers!(lobe)
    end
    _free_ensemble_agg!(eb)
    return nothing
end

"""
    free!(brain::SparseBrain)

Release every device buffer owned by `brain` into the CUDA.jl memory pool.
Safe to call more than once (`CUDA.unsafe_free!` is a no-op after the first
call). After this, `step!` / `reset!` / `get_output` must not be used on
`brain`; drop the reference so the Julia object can be collected.

Synchronizes the device first so kernels launched with `sync=false` cannot
touch buffers after they are returned to the pool.

CuArray finalizers already free buffers eventually, but they do not wait
for pending kernels and they honor extracted-field lifetimes. `free!` is the
deterministic, synchronized path when peak VRAM matters (for example before
constructing an [`EnsembleBrain`](@ref)).

# Arguments
- `brain::SparseBrain`: lobe whose GPU allocations are released

# Examples
```julia
using LiquidCortex, CUDA
brain = SparseBrain(20.0f0; n_in=8, n_out=4)
free!(brain)
```
"""
function free!(brain::SparseBrain)
    CUDA.synchronize()
    _free_device_buffers!(brain)
    return nothing
end

"""
    free!(eb::EnsembleBrain)

[`free!`](@ref free!(::SparseBrain)) every lobe and the aggregated readout
buffer. Safe to call more than once.

# Arguments
- `eb::EnsembleBrain`: ensemble whose GPU allocations are released
"""
function free!(eb::EnsembleBrain)
    CUDA.synchronize()
    _free_ensemble_buffers!(eb)
    return nothing
end
