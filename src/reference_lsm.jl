# SPDX-License-Identifier: MIT OR Apache-2.0
#
# reference_lsm.jl — Reference Liquid State Machine (2,048 Neurons)
#
# A simple 2,048-neuron dense CUDA reservoir for rapid prototyping.
# Configurable input/output dimensions, tanh activation, generic inhibition.
#
# GPU state is lazily initialized on first use to avoid wasted allocation.

using CUDA
using Statistics
using LinearAlgebra

# Reservoir parameters
const REF_N = 2048
const REF_IN_DEFAULT = 16
const REF_OUT_DEFAULT = 16

# ── Lazy-initialized GPU state ─────────────────────────────────────────────────
# These globals start as `nothing` and are allocated on first use.
# This allows `using LiquidCortex` to succeed on CPU-only machines.

const _ref_W = Ref{Union{Nothing, CuMatrix{Float32}}}(nothing)
const _ref_Win = Ref{Union{Nothing, CuMatrix{Float32}}}(nothing)
const _ref_Wout = Ref{Union{Nothing, CuMatrix{Float32}}}(nothing)
const _ref_x = Ref{Union{Nothing, CuVector{Float32}}}(nothing)
const _ref_n_in = Ref{Int}(REF_IN_DEFAULT)
const _ref_n_out = Ref{Int}(REF_OUT_DEFAULT)
const _ref_initialized = Ref{Bool}(false)

"""
    _init_ref_lsm!(; n_in=16, n_out=16) -> Nothing

Allocate the 2,048-neuron reference LSM reservoir on GPU.

Call this **manually** when you need to pin `n_in` / `n_out` before the first
step, or to pre-allocate so the first [`run_lsm_step`](@ref) is not the
allocation. Otherwise the first `run_lsm_step` / [`run_lsm_step_str`](@ref)
initializes lazily from `length(inputs_vec)` and the `n_out` keyword.

Re-calling after the reservoir is already initialized overwrites the global
`_ref_*` state (weights and membrane vector are replaced). `n_in` and `n_out`
must be positive.

# Keyword Arguments
- `n_in::Int=16`: input dimension (`REF_IN_DEFAULT`)
- `n_out::Int=16`: readout dimension (`REF_OUT_DEFAULT`)

# Returns
- `Nothing`: module-global GPU buffers are filled (`_ref_W`, `_ref_Win`,
  `_ref_Wout`, `_ref_x`) and `_ref_initialized` is set

# Examples
```julia
using LiquidCortex
LiquidCortex._init_ref_lsm!(; n_in=8, n_out=4)   # pre-allocate custom dims
y = LiquidCortex.run_lsm_step(zeros(Float32, 8), 0.0f0)
length(y) == 4
```
"""
function _init_ref_lsm!(; n_in::Int=REF_IN_DEFAULT, n_out::Int=REF_OUT_DEFAULT)
    n_in > 0 || throw(ArgumentError("n_in must be positive, got $n_in"))
    n_out > 0 || throw(ArgumentError("n_out must be positive, got $n_out"))
    _ref_W[] = cpu_randn_cu(REF_N, REF_N) .* 0.02f0
    _ref_Win[] = cpu_randn_cu(REF_N, n_in) .* 0.5f0
    _ref_Wout[] = cpu_randn_cu(n_out, REF_N) .* 0.1f0
    _ref_x[] = CUDA.zeros(Float32, REF_N)
    _ref_n_in[] = n_in
    _ref_n_out[] = n_out
    _ref_initialized[] = true
    return nothing
end

_ref_is_initialized() = _ref_initialized[]

"""
    run_lsm_step(inputs_vec, inhibit_val; n_out=16) -> Vector{Float32}

Advance the 2,048-neuron dense reference reservoir by one tick.

On the first call (when `_ref_initialized` is false) this lazily allocates
the GPU state: `n_in = length(inputs_vec)` and `n_out` from the keyword.
Later calls require `length(inputs_vec) == _ref_n_in[]` or they throw
`DimensionMismatch`. Requires a CUDA GPU.

# Arguments
- `inputs_vec::Vector{Float32}`: host input; length sets `n_in` on lazy init
- `inhibit_val::Float32`: inhibition in `[0, 1]`; recurrent gain is
  `1 - 0.4 * inhibit_val`

# Keyword Arguments
- `n_out::Int=16`: readout size used **only** on first-call lazy init.
  Ignored once the reservoir exists — call [`_init_ref_lsm!`](@ref) first
  to set both dims explicitly.

# Returns
- `Vector{Float32}`: host copy of the `n_out`-element readout

# Examples
```julia
using LiquidCortex
# Custom readout width on first call (also sets n_in from the vector)
y = LiquidCortex.run_lsm_step(zeros(Float32, 8), 0.25f0; n_out=4)
length(y) == 4

# Default 16-in / 16-out lazy init
y16 = LiquidCortex.run_lsm_step(zeros(Float32, 16), 0.0f0)
```
"""
function run_lsm_step(inputs_vec::Vector{Float32}, inhibit_val::Float32;
    n_out::Int=REF_OUT_DEFAULT)
    # Lazy initialization on first call
    if !_ref_is_initialized()
        LiquidCortex._cuda_available[] || error(
            "Reference LSM requires a CUDA GPU. No CUDA device available.")
        _init_ref_lsm!(; n_in=length(inputs_vec), n_out=n_out)
    end

    W = _ref_W[]
    Win = _ref_Win[]
    Wout = _ref_Wout[]
    x_local = _ref_x[]

    length(inputs_vec) == _ref_n_in[] || throw(DimensionMismatch(
        "input has length $(length(inputs_vec)), expected $(_ref_n_in[])"))

    # Move inputs to GPU
    u = cu(inputs_vec)

    # Reservoir dynamics: x(t+1) = tanh(W*x(t) + Win*u(t))
    # High inhibit_val reduces the gain of the recurrent connections.
    gain = 1.0f0 - (inhibit_val * 0.4f0)

    # Step the reservoir
    x_local .= tanh.(gain .* (W * x_local) .+ (Win * u))
    _ref_x[] = x_local

    # Readout Layer
    y = Wout * x_local

    return Array(y)
end

"""
    run_lsm_step_str(inputs_vec, inhibit_val; n_out=16) -> String

Same as [`run_lsm_step`](@ref), but joins the readout with commas.

Useful for FFI / logging paths that want a single string rather than a
`Vector{Float32}`. Lazy-init rules and `n_out` semantics match
`run_lsm_step`.

# Arguments
- `inputs_vec::Vector{Float32}`: host input
- `inhibit_val::Float32`: inhibition in `[0, 1]`

# Keyword Arguments
- `n_out::Int=16`: readout size used only on first-call lazy init

# Returns
- `String`: comma-separated readout, e.g. `"0.12,-0.03,0.45"`

# Examples
```julia
using LiquidCortex
s = LiquidCortex.run_lsm_step_str(zeros(Float32, 8), 0.0f0; n_out=4)
s isa String
occursin(",", s)
```
"""
function run_lsm_step_str(inputs_vec::Vector{Float32}, inhibit_val::Float32;
    n_out::Int=REF_OUT_DEFAULT)
    y = run_lsm_step(inputs_vec, inhibit_val; n_out=n_out)
    return join(y, ",")
end
