# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Explicit lifecycle for SparseBrain / EnsembleBrain. Constructors allocate
# hundreds of megabytes of device buffers with no paired release; these
# helpers restore neuron state without reallocating, or return those buffers
# to the CUDA.jl pool immediately.

"""
    reset!(brain::SparseBrain) -> SparseBrain

Restore neuron state, traces, history, and diagnostics to constructor
defaults. Recurrent, input, and readout weights are kept so the same
reservoir can be reused across trials.

# Arguments
- `brain::SparseBrain`: lobe to rewind (mutated in place)

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
function reset!(brain::SparseBrain)
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
    reset!(eb::EnsembleBrain) -> EnsembleBrain

[`reset!`](@ref reset!(::SparseBrain)) every lobe and zero the aggregated
readout. Per-lobe weights and aggregation weights are kept.

# Arguments
- `eb::EnsembleBrain`: ensemble to rewind (mutated in place)

# Returns
- the same `eb`
"""
function reset!(eb::EnsembleBrain)
    for lobe in eb.lobes
        reset!(lobe)
    end
    fill!(eb.agg_output, 0.0f0)
    CUDA.synchronize()
    return eb
end

"""
    free!(brain::SparseBrain)

Release every device buffer owned by `brain` into the CUDA.jl memory pool.
Safe to call more than once. After this, `step!` / `reset!` / `get_output`
must not be used on `brain`; drop the reference so the Julia object can be
collected.

CuArray finalizers already free buffers eventually. This is the
deterministic path when peak VRAM matters (for example before constructing
an [`EnsembleBrain`](@ref)).

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

"""
    free!(eb::EnsembleBrain)

[`free!`](@ref free!(::SparseBrain)) every lobe and the aggregated readout
buffer. Safe to call more than once.

# Arguments
- `eb::EnsembleBrain`: ensemble whose GPU allocations are released
"""
function free!(eb::EnsembleBrain)
    for lobe in eb.lobes
        free!(lobe)
    end
    CUDA.unsafe_free!(eb.agg_output)
    return nothing
end
