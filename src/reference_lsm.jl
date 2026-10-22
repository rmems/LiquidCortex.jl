# SPDX-License-Identifier: MIT OR Apache-2.0
#
# reference_lsm.jl — Reference Liquid State Machine (2,048 Neurons)
#
# A simple 2,048-neuron dense CUDA reservoir for rapid prototyping.
# Configurable input/output dimensions, tanh activation, generic inhibition.
#
# Prefer `ReferenceLSM` so two reservoirs do not share weights or membrane
# state. `run_lsm_step` / `run_lsm_step_str` still lazily initialize a
# process-wide default instance (GPU only).

using CUDA
using Statistics
using LinearAlgebra
using Random

# Reservoir parameters
const REF_N = 2048
const REF_IN_DEFAULT = 16
const REF_OUT_DEFAULT = 16

"""
    ReferenceLSM

Dense tanh reservoir used by the 2,048-neuron reference LSM.

Each instance owns its recurrent, input, and readout weights plus the
membrane vector, so two reservoirs cannot interfere. Construction is
deterministic given `rng`. Host arrays are used when `gpu=false` (CPU
tests); CUDA arrays require a device.

# Fields
- `W`: square recurrent matrix (`N × N`, `Float32`)
- `Win`: input weights (`N × n_in`)
- `Wout`: readout weights (`n_out × N`)
- `x`: membrane / reservoir state (`N`)
- `n_in`: input dimension
- `n_out`: readout dimension
"""
mutable struct ReferenceLSM{TW<:AbstractMatrix{Float32},
                            TWin<:AbstractMatrix{Float32},
                            TWout<:AbstractMatrix{Float32},
                            TX<:AbstractVector{Float32}}
    W::TW
    Win::TWin
    Wout::TWout
    x::TX
    n_in::Int
    n_out::Int

    function ReferenceLSM(W::TW, Win::TWin, Wout::TWout, x::TX) where {
        TW<:AbstractMatrix{Float32},
        TWin<:AbstractMatrix{Float32},
        TWout<:AbstractMatrix{Float32},
        TX<:AbstractVector{Float32},
    }
        size(W, 1) == size(W, 2) || throw(ArgumentError(
            "W must be square, got $(size(W))"))
        size(Win, 1) == size(W, 1) || throw(ArgumentError(
            "Win first dim must match W, got $(size(Win, 1)) vs $(size(W, 1))"))
        size(Wout, 2) == size(W, 1) || throw(ArgumentError(
            "Wout second dim must match W, got $(size(Wout, 2)) vs $(size(W, 1))"))
        length(x) == size(W, 1) || throw(ArgumentError(
            "x length must match W, got $(length(x)) vs $(size(W, 1))"))
        size(Win, 2) > 0 || throw(ArgumentError(
            "n_in must be positive, got $(size(Win, 2))"))
        size(Wout, 1) > 0 || throw(ArgumentError(
            "n_out must be positive, got $(size(Wout, 1))"))
        return new{TW, TWin, TWout, TX}(
            W, Win, Wout, x, size(Win, 2), size(Wout, 1))
    end
end

function Base.show(io::IO, lsm::ReferenceLSM)
    print(io, "ReferenceLSM(N=", size(lsm.W, 1),
          ", n_in=", lsm.n_in, ", n_out=", lsm.n_out, ")")
end

"""
    ReferenceLSM(; n_in=16, n_out=16, N=2048, rng=default_rng(), gpu=...)

Allocate a dense reference reservoir. `gpu` defaults to
`LiquidCortex._cuda_available[]`. When `gpu=true`, weights are drawn on
the host from `rng` and uploaded (`cpu_randn_cu`); when `gpu=false`, the
same `randn(rng, Float32, ...)` draws stay on the host.

`n_in`, `n_out`, and `N` must be positive.

# Keyword Arguments
- `n_in::Int=16`: input dimension (`REF_IN_DEFAULT`)
- `n_out::Int=16`: readout dimension (`REF_OUT_DEFAULT`)
- `N::Int=2048`: reservoir size (`REF_N`)
- `rng::AbstractRNG`: host RNG for weight draws
- `gpu::Bool`: allocate `CuArray` weights when `true`
"""
function ReferenceLSM(; n_in::Int=REF_IN_DEFAULT, n_out::Int=REF_OUT_DEFAULT,
                      N::Int=REF_N, rng::AbstractRNG=Random.default_rng(),
                      gpu::Bool=_cuda_available[])
    n_in > 0 || throw(ArgumentError("n_in must be positive, got $n_in"))
    n_out > 0 || throw(ArgumentError("n_out must be positive, got $n_out"))
    N > 0 || throw(ArgumentError("N must be positive, got $N"))
    if gpu
        _require_cuda("Reference LSM")
        W = cpu_randn_cu(rng, N, N) .* 0.02f0
        Win = cpu_randn_cu(rng, N, n_in) .* 0.5f0
        Wout = cpu_randn_cu(rng, n_out, N) .* 0.1f0
        x = CUDA.zeros(Float32, N)
    else
        W = randn(rng, Float32, N, N) .* 0.02f0
        Win = randn(rng, Float32, N, n_in) .* 0.5f0
        Wout = randn(rng, Float32, n_out, N) .* 0.1f0
        x = zeros(Float32, N)
    end
    return ReferenceLSM(W, Win, Wout, x)
end

# ── Process-wide default (compat for run_lsm_step) ─────────────────────────
# Lazily allocated on first `run_lsm_step` / `_init_ref_lsm!`. New code
# should construct `ReferenceLSM` instead of sharing this instance.

const _default_ref_lsm = Ref{Union{Nothing, ReferenceLSM}}(nothing)

_ref_is_initialized() = _default_ref_lsm[] !== nothing

# Deprecated accessors for the default instance. `_ref_*[]` still works.
struct _DeprecatedRefField{T}
    name::Symbol
    fallback::T
end

function _depwarn_ref_field(name::Symbol)
    Base.depwarn(
        "`LiquidCortex._ref_$name` is deprecated; hold state on a `ReferenceLSM`.",
        Symbol("_ref_$name"))
    return nothing
end

function Base.getindex(r::_DeprecatedRefField)
    _depwarn_ref_field(r.name)
    lsm = _default_ref_lsm[]
    r.name === :initialized && return lsm !== nothing
    lsm === nothing && return r.fallback
    return getfield(lsm, r.name)
end

function Base.setindex!(r::_DeprecatedRefField, val)
    _depwarn_ref_field(r.name)
    if r.name === :initialized
        val || (_default_ref_lsm[] = nothing)
        return val
    end
    lsm = _default_ref_lsm[]
    if lsm === nothing
        val === nothing && return nothing
        throw(ErrorException("default ReferenceLSM is not initialized"))
    end
    if val === nothing && r.name in (:W, :Win, :Wout, :x)
        _default_ref_lsm[] = nothing
        return nothing
    end
    setfield!(lsm, r.name, val)
    return val
end

const _ref_W = _DeprecatedRefField{Any}(:W, nothing)
const _ref_Win = _DeprecatedRefField{Any}(:Win, nothing)
const _ref_Wout = _DeprecatedRefField{Any}(:Wout, nothing)
const _ref_x = _DeprecatedRefField{Any}(:x, nothing)
const _ref_n_in = _DeprecatedRefField{Int}(:n_in, REF_IN_DEFAULT)
const _ref_n_out = _DeprecatedRefField{Int}(:n_out, REF_OUT_DEFAULT)
const _ref_initialized = _DeprecatedRefField{Bool}(:initialized, false)

"""
    _init_ref_lsm!(; n_in=16, n_out=16, rng=default_rng()) -> Nothing

Allocate the process-wide default 2,048-neuron reference LSM on GPU.

Call this **manually** when you need to pin `n_in` / `n_out` before the first
step, or to pre-allocate so the first `run_lsm_step` is not the
allocation. Otherwise the first `run_lsm_step` / `run_lsm_step_str`
initializes lazily from `length(inputs_vec)` and the `n_out` keyword.

Prefer [`ReferenceLSM`](@ref) when you need an isolated reservoir.
Re-calling after the default instance exists replaces it. `n_in` and `n_out`
must be positive.

# Keyword Arguments
- `n_in::Int=16`: input dimension (`REF_IN_DEFAULT`)
- `n_out::Int=16`: readout dimension (`REF_OUT_DEFAULT`)
- `rng::AbstractRNG`: host RNG forwarded to `ReferenceLSM`

# Returns
- `Nothing`: `_default_ref_lsm` is replaced with a new GPU `ReferenceLSM`

# Examples
```julia
using LiquidCortex
LiquidCortex._init_ref_lsm!(; n_in=8, n_out=4)   # pre-allocate custom dims
y = LiquidCortex.run_lsm_step(zeros(Float32, 8), 0.0f0)
length(y) == 4
```
"""
function _init_ref_lsm!(; n_in::Int=REF_IN_DEFAULT, n_out::Int=REF_OUT_DEFAULT,
                        rng::AbstractRNG=Random.default_rng())
    n_in > 0 || throw(ArgumentError("n_in must be positive, got $n_in"))
    n_out > 0 || throw(ArgumentError("n_out must be positive, got $n_out"))
    _default_ref_lsm[] = ReferenceLSM(; n_in=n_in, n_out=n_out, rng=rng, gpu=true)
    return nothing
end

function _as_step_input(Win::AbstractMatrix{Float32},
                        inputs_vec::AbstractVector{<:Real})
    u = inputs_vec isa Vector{Float32} ? inputs_vec : Float32.(inputs_vec)
    return Win isa CuArray ? cu(u) : u
end

"""
    run_lsm_step!(lsm, inputs_vec, inhibit_val) -> Vector{Float32}

Advance `lsm` by one tick. Recurrent gain is `1 - 0.4 * inhibit_val`.
`length(inputs_vec)` must equal `lsm.n_in`.
"""
function run_lsm_step!(lsm::ReferenceLSM, inputs_vec::AbstractVector{<:Real},
                       inhibit_val::Real)
    length(inputs_vec) == lsm.n_in || throw(DimensionMismatch(
        "input has length $(length(inputs_vec)), expected $(lsm.n_in)"))

    u = _as_step_input(lsm.Win, inputs_vec)
    gain = 1.0f0 - (Float32(inhibit_val) * 0.4f0)
    lsm.x .= tanh.(gain .* (lsm.W * lsm.x) .+ (lsm.Win * u))
    y = lsm.Wout * lsm.x
    return Array(y)
end

"""
    run_lsm_step(inputs_vec, inhibit_val; n_out=16) -> Vector{Float32}

Advance the process-wide default 2,048-neuron dense reference reservoir
by one tick.

On the first call (when the default instance is unset) this lazily allocates
GPU state: `n_in = length(inputs_vec)` and `n_out` from the keyword.
Later calls require `length(inputs_vec) == lsm.n_in` or they throw
`DimensionMismatch`. Requires a CUDA GPU.

For isolated reservoirs, construct [`ReferenceLSM`](@ref) and call
[`run_lsm_step!`](@ref).

# Arguments
- `inputs_vec::Vector{Float32}`: host input; length sets `n_in` on lazy init
- `inhibit_val::Float32`: inhibition in `[0, 1]`; recurrent gain is
  `1 - 0.4 * inhibit_val`

# Keyword Arguments
- `n_out::Int=16`: readout size used **only** on first-call lazy init.
  Ignored once the default reservoir exists — call
  `LiquidCortex._init_ref_lsm!` first to set both dims explicitly.

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
    if !_ref_is_initialized()
        _init_ref_lsm!(; n_in=length(inputs_vec), n_out=n_out)
    end
    return run_lsm_step!(_default_ref_lsm[], inputs_vec, inhibit_val)
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

function run_lsm_step_str(lsm::ReferenceLSM, inputs_vec::Vector{Float32},
                          inhibit_val::Float32)
    return join(run_lsm_step!(lsm, inputs_vec, inhibit_val), ",")
end
