# SPDX-License-Identifier: MIT OR Apache-2.0
#
# sparse_brain.jl — Ensemble 65,536-Neuron Sparse CUDA Liquid State Machine
#
# LiquidCortex V2 "Brain" — 4-Lobe Ensemble Architecture
#
# Architecture:
#   4 parallel lobes × 65,536 LIF neurons each
#   Varying time constants: τ_m ∈ {10ms, 25ms, 50ms, 100ms}
#   Xavier/Glorot W_out initialization (breaks zero-readout deadlock)
#   Sparse connectivity (1% connection probability, Float16 weights)
#   STDP covariance learning rule
#   Rolling 1,000-tick spike history for deep temporal covariance
#   Generic inhibition interface (caller provides stress signal)
#
# ═══════════════════════════════════════════════════════════════════════════════

using CUDA
using SparseArrays
using LinearAlgebra
using Statistics
using Random
using Printf

# ─── Constants ────────────────────────────────────────────────────────────────

const N = 65_536        # Reservoir neuron count per lobe
const CONN_PROB = 0.01          # 1% sparse connectivity → ~42M non-zero synapses
const DT = 1.0f0         # Simulation timestep (normalized to tick interval)
const HIST_DEPTH = 1000          # Rolling history depth (ticks) for deep temporal covariance
const COV_SUBSAMPLE = 8192          # Subsampled neurons for tractable covariance (avoids 17 GB N×N)

# ── Lobe time constants (membrane τ_m in ms) ─────────────────────────────────
const LOBE_TAUS = Float32[10.0, 25.0, 50.0, 100.0]
const LOBE_NAMES = ["Fast", "Medium", "Slow", "Integrator"]
const N_LOBES = 4

# ── Ensemble Aggregation Weights ─────────────────────────────────────────────
# Fast lobe reacts quickest → highest weight for immediate signals
const LOBE_WEIGHTS = Float32[0.4, 0.3, 0.2, 0.1]

# ── Ornstein-Uhlenbeck SDE Parameters ────────────────────────────────────────
# dV_j = ((V_rest - V_j) / τ_m + Σᵢ Wᵢⱼ · Sᵢ(t)) dt + σ dWₜ

const V_REST = -65.0f0       # Resting potential (mV)
const V_THRESH = -50.0f0       # Spike threshold (mV)
const V_RESET = -70.0f0       # Post-spike reset (mV)
const SIGMA = 2.0f0         # OU noise amplitude
const REFRAC_T = 5             # Refractory period (timesteps)

# ── STDP Parameters ──────────────────────────────────────────────────────────
# Pair-based trace rule applied to the sparse recurrent edges
# (see `_pair_stdp_kernel!`):
#   Δwᵢⱼ = η · (trace_pre[i] · s[j] − s[i] · trace_post[j])
# Both traces decay with TAU_TRACE.

const ETA = 0.001f0       # Learning rate
const TAU_TRACE = 20.0f0        # Eligibility trace decay
const W_MAX = 1.0f0         # Weight saturation (Float16 range)

# Precompute Float32 scalars on the CPU so CUDA broadcasts stay monomorphic.
const OU_NOISE_SCALE = SIGMA * sqrt(DT)

# ── Inhibition parameters ───────────────────────────────────────────────────
const INHIBITION_GAIN = 15.0f0    # mV increase per unit of inhibition
const MAX_INHIBITION = 3.0f0       # Maximum inhibition clamp

"""
    cpu_randn_cu(dims...) -> CuArray{Float32}

Work around CUDA.jl RNG compilation failures on this stack by generating
Float32 Gaussian samples on the host and uploading them to the device.
"""
function cpu_randn_cu(dims::Vararg{Int,N}) where {N}
    return cu(randn(Float32, dims...))
end

# ── Host-side kernels (plain arrays; unit-tested on CPU) ─────────────────────
# GPU-locked only by SparseBrain field types. Hosted CI asserts the same
# arithmetic the CUDA path uses (#55). `_pair_stdp_kernel!` calls `_pair_stdp_dw`.

"""Dynamic spike threshold after clamping `inhibition` to `[0, MAX_INHIBITION]`."""
function _inhibited_threshold(inhibition::Real)
    inhib = clamp(Float32(inhibition), 0.0f0, MAX_INHIBITION)
    return V_THRESH + inhib * INHIBITION_GAIN
end

"""Fast-lobe readout rate: 5× when `|reflex_signal| > 0.1`."""
function _reflex_fast_eta(reflex_eta::Float32, reflex_signal::Float32)
    return abs(reflex_signal) > 0.1f0 ? reflex_eta * 5.0f0 : reflex_eta
end

@inline function _lobe_reflex_eta(lobe_index::Int, reflex_eta::Float32, reflex_fast::Float32)
    return lobe_index == 1 ? reflex_fast : reflex_eta
end

@inline function _trace_decay_factor(dt::Float32, tau_trace::Float32)
    return 1.0f0 - dt / tau_trace
end

@inline function _spike_rate(n_spikes, n::Integer)
    return Float32(n_spikes) / Float32(n)
end

"""Advance circular history index. Returns `(next_idx, wrapped)`."""
function _advance_history_index(hist_idx::Integer, hist_depth::Integer)
    next = Int(hist_idx) + 1
    if next > hist_depth
        return 1, true
    end
    return next, false
end

"""Pair-STDP Δw on one edge. LTP when pre-trace co-occurs with a post spike."""
@inline function _pair_stdp_dw(trace_pre_i::Float32, s_j::Float32, s_i::Float32,
                                trace_post_j::Float32, eta::Float32)
    return eta * (trace_pre_i * s_j - s_i * trace_post_j)
end

"""Apply pair STDP to existing sparse edges (host arrays or GPU vectors)."""
function _pair_stdp_apply!(nzVal, pre_idx, post_idx, trace_pre, trace_post, S,
                           eta::Float32, w_max::Float32)
    @inbounds for i in eachindex(nzVal)
        pre = Int(pre_idx[i])
        post = Int(post_idx[i])
        dw = _pair_stdp_dw(Float32(trace_pre[pre]), Float32(S[post]),
                            Float32(S[pre]), Float32(trace_post[post]), eta)
        if dw != 0.0f0
            nzVal[i] = Float16(clamp(Float32(nzVal[i]) + dw, -w_max, w_max))
        end
    end
    return nzVal
end

"""Hebbian readout update: `ΔW_out = η · 1(y>0) · trace_preᵀ`, then clamp."""
function _readout_hebbian_update!(W_out, output, trace_pre, reflex_eta::Float32, w_max::Float32)
    S_out = output .> 0.0f0
    W_out .+= reflex_eta .* (Float32.(S_out) * trace_pre')
    clamp!(W_out, -w_max, w_max)
    return nothing
end

function _validate_csc(colPtr::AbstractVector{<:Integer}, edge_nnz::Int)
    isempty(colPtr) &&
        throw(ArgumentError("Malformed CSC: empty colPtr"))
    colPtr[1] == 1 ||
        throw(ArgumentError("Malformed CSC: colPtr[1]=$(colPtr[1]), expected 1"))
    colPtr[end] == edge_nnz + 1 ||
        throw(ArgumentError("Malformed CSC: colPtr[end]=$(colPtr[end]) vs nnz+1=$(edge_nnz + 1)"))
    n_cols = length(colPtr) - 1
    @inbounds for col in 1:n_cols
        colPtr[col + 1] >= colPtr[col] ||
            throw(ArgumentError(
                "Malformed CSC: non-monotonic colPtr at col=$col ($(colPtr[col]) > $(colPtr[col + 1]))"))
    end
    return nothing
end

function _csc_edge_lists(colPtr::AbstractVector{<:Integer}, rowVal::AbstractVector{<:Integer})
    edge_nnz = length(rowVal)
    _validate_csc(colPtr, edge_nnz)
    n_cols = length(colPtr) - 1
    pre = Vector{Int32}(undef, edge_nnz)
    post = Vector{Int32}(undef, edge_nnz)
    k = 1
    @inbounds for col in 1:n_cols
        p_lo = Int(colPtr[col])
        p_hi = Int(colPtr[col + 1]) - 1
        p_hi > edge_nnz && throw(ArgumentError(
            "Malformed CSC: colPtr[$(col + 1)]=$(colPtr[col + 1]) exceeds nnz=$edge_nnz"))
        for p in p_lo:p_hi
            post[k] = Int32(rowVal[p])
            pre[k] = Int32(col)
            k += 1
        end
    end
    return pre, post
end

function _weighted_sum!(agg, weights, outputs)
    n = length(outputs)
    length(weights) == n || throw(BoundsError(weights, n))
    fill!(agg, zero(eltype(agg)))
    for (w, y) in zip(weights, outputs)
        agg .+= w .* y
    end
    return agg
end

function _cov_subsample_count(n::Int, cov_subsample::Int)
    return min(cov_subsample, n)
end

function _cov_subsample_indices(n::Int, cov_subsample::Int,
                               rng::AbstractRNG=Random.default_rng())
    k = _cov_subsample_count(n, cov_subsample)
    return sort(randperm(rng, n)[1:k])
end

function _spike_history_covariance(X, hist_depth::Integer)
    μ = mean(X, dims=1)
    X_centered = X .- μ
    return (X_centered' * X_centered) ./ Float32(hist_depth - 1)
end

function _format_diagnostics(tick_count, total_spikes, last_spike_rate,
                             v_thresh_dynamic, w_out_norm)
    return string(
        "[brain] tick=", tick_count,
        " spikes=", total_spikes,
        " rate=", round(last_spike_rate * 100, digits=2), "%",
        " V_thresh=", round(v_thresh_dynamic, digits=1),
        " W_out_norm=", round(Float64(w_out_norm), digits=4)
    )
end

function _format_lobe_diagnostics(name, tau_m, tick_count, rate_pct)
    return @sprintf("[%s:τ=%d] tick=%d rate=%.2f%% W=n/a",
        name, Int(tau_m), tick_count, rate_pct)
end

function _format_lobe_diagnostics(name, tau_m, tick_count, rate_pct, w_norm)
    return @sprintf("[%s:τ=%d] tick=%d rate=%.2f%% W=%.4f",
        name, Int(tau_m), tick_count, rate_pct, w_norm)
end

# ── Pair STDP on existing sparse edges (experimental plasticity=:recurrent_stdp) ─
function _pair_stdp_kernel!(nzVal, pre_idx, post_idx, trace_pre, trace_post, S,
                            eta::Float32, w_max::Float32, nnz::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    @inbounds if i <= nnz
        pre = pre_idx[i]
        post = post_idx[i]
        # Pair rule: LTP when pre-trace co-occurs with post spike; LTD reverse
        dw = _pair_stdp_dw(trace_pre[pre], S[post], S[pre], trace_post[post], eta)
        # Skip clamp/write when dw==0 so eta=0 (or silent edges) never truncates
        # constructor weights that may exceed |W_MAX| before any learning.
        if dw != 0.0f0
            w = clamp(Float32(nzVal[i]) + dw, -w_max, w_max)
            nzVal[i] = Float16(w)
        end
    end
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════════
# SparseBrain: The 65,536-Neuron CUDA Reservoir
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SparseBrain

A 65,536-neuron sparse CUDA reservoir lobe with OU-SDE membrane dynamics,
STDP-capable recurrent weights, and a dense readout.

Requires a CUDA GPU. Construct with `SparseBrain(tau_m; n_in, n_out, name)`.
Reuse a lobe across trials with [`reset!`](@ref); release device memory with
[`free!`](@ref). CuArray finalizers run if the caller never calls `free!`.

# Fields
- `W::CuSparseMatrixCSC{Float16,Int32}`: sparse recurrent weights (1% connectivity)
- `pre_idx::CuVector{Int32}`: pre-synaptic edge indices (lazy; empty until STDP)
- `post_idx::CuVector{Int32}`: post-synaptic edge indices (lazy; empty until STDP)
- `nnz::Int`: number of recurrent nonzeros
- `W_in::CuMatrix{Float32}`: dense input weights (`N × n_in`)
- `W_out::CuMatrix{Float32}`: dense readout weights (`n_out × N`)
- `V::CuVector{Float32}`: membrane potential
- `S::CuVector{Float32}`: spike state (0 or 1)
- `refrac::CuVector{Int32}`: refractory counters
- `S_f16::CuVector{Float16}`: Float16 spike buffer for cuSPARSE SpMV
- `I_rec::CuVector{Float32}`: recurrent current
- `I_ext::CuVector{Float32}`: external (input) current
- `noise::CuVector{Float32}`: OU process noise
- `trace_pre::CuVector{Float32}`: pre-synaptic eligibility traces
- `trace_post::CuVector{Float32}`: post-synaptic eligibility traces
- `output::CuVector{Float32}`: GPU readout (`n_out`)
- `n_in::Int`: input dimension
- `n_out::Int`: output dimension
- `tau_m::Float32`: membrane time constant (ms)
- `history::CuMatrix{Float32}`: rolling spike history (`HIST_DEPTH × N`)
- `hist_idx::Int64`: next history write index (circular)
- `hist_full::Bool`: `true` once the history buffer has wrapped
- `v_thresh_dynamic::Float32`: adaptive spike threshold (mV)
- `tick_count::Int64`: completed timesteps
- `total_spikes::Int64`: cumulative spike count (committed with the tick when `sync=true`)
- `last_spike_rate::Float32`: last-tick spike fraction (committed with the tick when `sync=true`)

# Examples
```julia
using LiquidCortex, CUDA
brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="demo")
u = CUDA.zeros(Float32, 8)
step!(brain, u; inhibition=0.3f0)
y = get_output(brain)
```
"""
mutable struct SparseBrain
    # ── Synaptic weights (sparse, Float16 on GPU) ────────────────────────────
    W::CUDA.CUSPARSE.CuSparseMatrixCSC{Float16,Int32}

    # Edge lists aligned with W.nzVal (1-based): pre = column, post = row for W*S
    pre_idx::CuVector{Int32}
    post_idx::CuVector{Int32}
    nnz::Int

    # ── Input / Output weight matrices (dense, Float32) ──────────────────────
    W_in::CuMatrix{Float32}
    W_out::CuMatrix{Float32}

    # ── Neuron state vectors (Float32 on GPU) ────────────────────────────────
    V::CuVector{Float32}          # Membrane potential
    S::CuVector{Float32}          # Spike state (0 or 1)
    refrac::CuVector{Int32}       # Refractory counter

    # ── Work buffers (reused every tick) ─────────────────────────────────────
    S_f16::CuVector{Float16}
    I_rec::CuVector{Float32}
    I_ext::CuVector{Float32}
    noise::CuVector{Float32}

    # ── STDP eligibility traces ──────────────────────────────────────────────
    trace_pre::CuVector{Float32}  # Pre-synaptic trace
    trace_post::CuVector{Float32} # Post-synaptic trace

    # ── Readout state ────────────────────────────────────────────────────────
    output::CuVector{Float32}     # n_out-element readout

    # ── Dimensions ───────────────────────────────────────────────────────────
    n_in::Int                     # Input dimension
    n_out::Int                    # Output dimension

    # ── Per-lobe membrane time constant ──────────────────────────────────────
    tau_m::Float32                # τ_m in ms

    # ── Rolling spike history (HIST_DEPTH × N) for deep temporal covariance ──
    history::CuMatrix{Float32}    # 1000 × 65536 on GPU
    hist_idx::Int64               # Current write index (circular)
    hist_full::Bool               # True once buffer wraps at least once

    # ── Adaptive threshold (global inhibition) ───────────────────────────────
    v_thresh_dynamic::Float32

    # ── Diagnostics ──────────────────────────────────────────────────────────
    tick_count::Int64
    total_spikes::Int64
    last_spike_rate::Float32
end

function _validate_lobe_dims(n_in::Int, n_out::Int)
    n_in > 0 || throw(ArgumentError("n_in must be positive, got $n_in"))
    n_out > 0 || throw(ArgumentError("n_out must be positive, got $n_out"))
    return nothing
end

function _validate_tau_m(tau_m::Float32)
    (isfinite(tau_m) && tau_m > 0) || throw(ArgumentError(
        "tau_m must be positive and finite, got $tau_m"))
    return nothing
end

"""
    SparseBrain(tau_m; n_in=14, n_out=16, name="default") -> SparseBrain

Initialize a 65,536-neuron sparse CUDA reservoir lobe.

Weight initialization:
  - W_recurrent: Sparse CSC, 1% connectivity, Float16
    Spectral radius controlled via scaling: ||W|| ≈ 0.9 (echo state property)
  - W_in: Dense Float32, Xavier initialization √(2/n_in)
  - W_out: Dense Float32, Xavier/Glorot initialization √(2/N)
    (Breaks zero-readout deadlock — reservoir produces signals from tick 1)

# Arguments
- `tau_m::Float32`: membrane time constant in milliseconds (must be
  positive and finite). All of `0`, negatives, `NaN`, and `Inf` are
  rejected. `V` starts at `V_REST`, so the first membrane term is
  `0 / tau_m`; only `0` and `NaN` make that term `NaN`.

# Keyword Arguments
- `n_in::Int=14`: input dimension (must be positive)
- `n_out::Int=16`: readout dimension (must be positive)
- `name::String="default"`: label used in constructor progress logs

Prefer [`free!`](@ref) when peak VRAM matters; CuArray finalizers are the
GC fallback and must not be used as a substitute for `free!` after
`step!(; sync=false)`.

# Returns
- `SparseBrain`: GPU-resident lobe ready for [`step!`](@ref)

# Examples
```julia
using LiquidCortex, CUDA
brain = SparseBrain(20.0f0)                      # defaults: n_in=14, n_out=16
brain = SparseBrain(25.0f0; n_in=8, n_out=4, name="custom")
u = CUDA.zeros(Float32, brain.n_in)
step!(brain, u; inhibition=0.5f0)
```
"""
function SparseBrain(tau_m::Float32; n_in::Int=14, n_out::Int=16, name::String="default")
    _validate_lobe_dims(n_in, n_out)
    _validate_tau_m(tau_m)
    @debug "[brain:$name] Initializing 65,536-neuron lobe (τ_m=$(tau_m)ms, in=$(n_in), out=$(n_out))..."

    xavier_std_in = sqrt(2.0f0 / Float32(n_in))
    xavier_std_out = sqrt(2.0f0 / Float32(N))

    # ── 1. Sparse recurrent weight matrix (Float16, 1% connectivity) ─────────
    nnz_expected = round(Int, N * N * CONN_PROB)
    @debug "[brain:$name] Generating sparse connectivity (~$(round(nnz_expected / 1e6, digits=1))M synapses)..."

    # Build sparse matrix in COO format for efficiency
    rows = rand(1:N, nnz_expected)
    cols = rand(1:N, nnz_expected)
    vals = Float16.(randn(Float32, nnz_expected) .* 0.02f0)

    W_cpu = sparse(rows, cols, vals, N, N)

    # Remove self-connections (Dale's law approximation)
    for i in 1:min(N, size(W_cpu, 1))
        W_cpu[i, i] = Float16(0)
    end

    # Scale for spectral radius ≈ 0.9 (echo state property)
    frob = norm(W_cpu)
    actual_nnz = nnz(W_cpu)
    spectral_approx = frob / sqrt(actual_nnz)
    target_rho = 0.9f0
    scale_factor = target_rho / max(spectral_approx, 1e-6)
    W_cpu .*= Float16(scale_factor)

    @debug "[brain:$name] W_sparse: $(actual_nnz) nnz, ρ≈$(round(target_rho, digits=2))"

    # Transfer to GPU as CuSparseMatrixCSC
    # Edge lists for pair STDP are built lazily in `_ensure_edge_indices!`
    # (saves ~2×Int32×nnz ≈ 340 MB/lobe when STDP is unused).
    W_gpu = CUDA.CUSPARSE.CuSparseMatrixCSC(W_cpu)
    edge_nnz = nnz(W_cpu)
    pre_idx = CUDA.zeros(Int32, 0)
    post_idx = CUDA.zeros(Int32, 0)

    # ── 2. Input weight matrix (Dense, Xavier init) ──────────────────────────
    W_in = cpu_randn_cu(N, n_in)
    W_in .*= xavier_std_in

    # ── 3. Output weight matrix — Xavier/Glorot (breaks zero-readout deadlock)
    # W_out ~ N(0, √(2/N)) — ensures non-trivial readout from tick 1
    W_out = cpu_randn_cu(n_out, N)
    W_out .*= xavier_std_out
    @debug "[brain:$name] W_out: Xavier/Glorot init σ=$(round(Float64(xavier_std_out), sigdigits=4))"

    # ── 4. Neuron state vectors ──────────────────────────────────────────────
    V = CUDA.fill(Float32(V_REST), N)
    S = CUDA.zeros(Float32, N)
    refrac = CUDA.zeros(Int32, N)
    S_f16 = CUDA.zeros(Float16, N)
    I_rec = CUDA.zeros(Float32, N)
    I_ext = CUDA.zeros(Float32, N)
    noise = CUDA.zeros(Float32, N)

    # ── 5. STDP traces ──────────────────────────────────────────────────────
    trace_pre = CUDA.zeros(Float32, N)
    trace_post = CUDA.zeros(Float32, N)

    # ── 6. Output ────────────────────────────────────────────────────────────
    output = CUDA.zeros(Float32, n_out)

    # ── 7. Rolling spike history for deep temporal covariance (on GPU) ───────
    history = CUDA.zeros(Float32, HIST_DEPTH, N)
    @debug "[brain:$name] History buffer: $(HIST_DEPTH)×$(N) = $(round(HIST_DEPTH * N * 4 / 1e6, digits=1)) MB"

    CUDA.synchronize()
    @debug "[brain:$name] ✓ Lobe initialized (τ_m=$(tau_m)ms)"

    SparseBrain(
        W_gpu, pre_idx, post_idx, edge_nnz,
        W_in, W_out,
        V, S, refrac,
        S_f16, I_rec, I_ext, noise,
        trace_pre, trace_post,
        output,
        n_in, n_out,
        tau_m,
        history, 1, false,
        Float32(V_THRESH),
        0, 0, 0.0f0
    )
end

const PLASTICITY_MODES = (:readout_only, :recurrent_stdp, :none)

"""Materialize CSC edge lists for pair STDP (lazy — avoids ~300MB/lobe when unused)."""
function _ensure_edge_indices!(brain::SparseBrain)
    # Both buffers must be complete; a partial upload (pre ok, post failed) must rebuild.
    length(brain.pre_idx) == brain.nnz && length(brain.post_idx) == brain.nnz &&
        brain.nnz > 0 && return nothing
    colPtr = Array(brain.W.colPtr)
    rowVal = Array(brain.W.rowVal)
    # CSC invariants (1-based Julia SparseArrays). Failures here are internal
    # faults and remain Sentry-captured (not LiquidCortexValidationError).
    pre, post = _csc_edge_lists(colPtr, rowVal)
    brain.pre_idx = CuArray(pre)
    brain.post_idx = CuArray(post)
    brain.nnz = length(rowVal)
    return nothing
end

function _apply_pair_stdp!(brain; eta::Float32)
    # eta==0: no learning and no clamp/rewrite of existing weights
    eta == 0.0f0 && return nothing
    _ensure_edge_indices!(brain)
    nnz = Int32(brain.nnz)
    nnz == 0 && return nothing
    threads = 256
    blocks = cld(Int(nnz), threads)
    @cuda threads=threads blocks=blocks _pair_stdp_kernel!(
        brain.W.nzVal, brain.pre_idx, brain.post_idx,
        brain.trace_pre, brain.trace_post, brain.S,
        eta, W_MAX, nnz)
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════════
# Simulation Step: OU-SDE Dynamics + STDP Learning
# ═══════════════════════════════════════════════════════════════════════════════

"""CPU-safe plasticity/recurrent_eta checks (no GPU types)."""
function _validate_plasticity_kwargs(; plasticity::Symbol, recurrent_eta::Real)
    plasticity in PLASTICITY_MODES || throw(LiquidCortexValidationError(
        "plasticity must be one of $PLASTICITY_MODES, got :$plasticity"))
    if plasticity === :recurrent_stdp
        isfinite(Float32(recurrent_eta)) || throw(LiquidCortexValidationError(
            "recurrent_eta must be finite, got $recurrent_eta"))
    end
    return nothing
end

"""Validate public step kwargs. Throws `LiquidCortexValidationError` on misuse."""
function _validate_step_kwargs!(brain::SparseBrain, u::AbstractVector;
    plasticity::Symbol, recurrent_eta::Real)
    length(u) == brain.n_in || throw(LiquidCortexValidationError(
        "input has length $(length(u)), expected $(brain.n_in)"))
    _validate_plasticity_kwargs(; plasticity=plasticity, recurrent_eta=recurrent_eta)
    return nothing
end

"""Queue a history row. Host `hist_idx` / `tick_count` stay unchanged until `_advance_lobe_clock!`."""
function _queue_history_row!(brain::SparseBrain, record_history::Bool)
    if record_history
        brain.history[brain.hist_idx, :] .= brain.S
    end
    return nothing
end

"""Advance `hist_idx` / `tick_count` after GPU work (including any queued history write) has succeeded."""
function _advance_lobe_clock!(brain::SparseBrain, upcoming_tick::Int64, record_history::Bool)
    if record_history
        brain.hist_idx, wrapped = _advance_history_index(brain.hist_idx, HIST_DEPTH)
        wrapped && (brain.hist_full = true)
    end
    brain.tick_count = upcoming_tick
    return nothing
end

# Internal implementation; public entry point is `step!` (with Sentry capture).
function _step_impl!(brain::SparseBrain, u::CuVector{Float32};
    inhibition::Real=0.0f0,
    reflex_eta::Real=ETA,
    plasticity::Symbol=:readout_only,
    recurrent_eta::Real=1.0f-4,
    sync::Bool=true,
    record_history::Bool=true,
    use_device_noise::Bool=false,
    commit_clock::Bool=true)
    _validate_step_kwargs!(brain, u; plasticity=plasticity, recurrent_eta=recurrent_eta)
    upcoming_tick = brain.tick_count + 1
    inhibition = Float32(inhibition)
    reflex_eta = Float32(reflex_eta)
    recurrent_eta = Float32(recurrent_eta)

    # ── 1. Global Inhibition ─────────────────────────────────────────────────
    brain.v_thresh_dynamic = _inhibited_threshold(inhibition)

    # ── 2. OU-SDE Membrane Dynamics (per-lobe τ_m) ──────────────────────────
    # F16×F16 SpMV via * (generic mul! on F16 CSC can hit scalar indexing)
    brain.S_f16 .= Float16.(brain.S)
    y_sp = brain.W * brain.S_f16
    brain.I_rec .= Float32.(y_sp)

    mul!(brain.I_ext, brain.W_in, u)

    # Host noise is the portable default; device RNG may fail on some stacks.
    if use_device_noise
        try
            Random.randn!(brain.noise)
        catch e
            # Preserve cancel / GPU faults; only fall back for RNG path failures.
            e isa InterruptException && rethrow()
            e isa CUDA.OutOfGPUMemoryError && rethrow()
            e isa CUDA.CuError && rethrow()
            copyto!(brain.noise, randn(Float32, N))
        end
    else
        copyto!(brain.noise, randn(Float32, N))
    end
    brain.noise .*= OU_NOISE_SCALE

    dV = ((V_REST .- brain.V) ./ brain.tau_m .+ brain.I_rec .+ brain.I_ext) .* DT .+ brain.noise

    active_mask = brain.refrac .<= 0
    brain.V .+= dV .* Float32.(active_mask)

    # ── 3. Spike Detection ───────────────────────────────────────────────────
    spiked = brain.V .> brain.v_thresh_dynamic
    brain.S .= Float32.(spiked)

    brain.V .= ifelse.(spiked, Float32(V_RESET), brain.V)
    brain.refrac .= ifelse.(spiked, Int32(REFRAC_T), max.(brain.refrac .- Int32(1), Int32(0)))

    # Host reductions force a stream wait. Skip when sync=false so ensemble
    # mid-lobe loops do not reintroduce implicit barriers (bench / chain mode).
    # Apply the host fields only after the later barrier / clock commit so a
    # failed readout does not inflate `total_spikes` on a tick that did not
    # complete.
    n_spikes = 0.0f0
    if sync
        n_spikes = sum(brain.S)
    end

    # ── 4. Traces + learning ─────────────────────────────────────────────────
    decay = _trace_decay_factor(DT, TAU_TRACE)
    brain.trace_pre .= brain.trace_pre .* decay .+ brain.S
    brain.trace_post .= brain.trace_post .* decay .+ brain.S

    if plasticity === :recurrent_stdp
        _apply_pair_stdp!(brain; eta=recurrent_eta)
    end

    if plasticity !== :none && upcoming_tick % 10 == 0
        _readout_hebbian_update!(brain.W_out, brain.output, brain.trace_pre, reflex_eta, W_MAX)
    end

    # ── 5. Readout ───────────────────────────────────────────────────────────
    mul!(brain.output, brain.W_out, brain.S)

    # Queue history before the optional barrier so a failed synchronize()
    # leaves `hist_idx` unchanged (the next successful tick overwrites the
    # same slot). Host clocks and spike diagnostics commit only afterward.
    if commit_clock
        _queue_history_row!(brain, record_history)
    end
    sync && CUDA.synchronize()
    if commit_clock
        if sync
            brain.total_spikes += round(Int64, n_spikes)
            brain.last_spike_rate = _spike_rate(n_spikes, N)
        end
        _advance_lobe_clock!(brain, upcoming_tick, record_history)
    end
    return nothing
end

"""
    step!(brain::SparseBrain, u; inhibition=0.0, reflex_eta=ETA,
          plasticity=:readout_only, recurrent_eta=1f-4, sync=true,
          record_history=true, use_device_noise=false) -> Nothing

Execute one OU-SDE simulation timestep on a single lobe.

# Arguments
- `brain::SparseBrain`: reservoir lobe (mutated in place)
- `u::CuVector{Float32}`: input current; `length(u)` must equal `brain.n_in`

# Keyword Arguments
- `inhibition`: global inhibition level (default `0.0`, clamped to `[0, MAX_INHIBITION]`).
  Raises the spike threshold by `inhibition * INHIBITION_GAIN` mV.
- `reflex_eta`: Hebbian learning rate for `W_out` (default `ETA`).
- `plasticity`: one of `:readout_only` (default; frozen `W` plus Hebbian `W_out`
  every 10 ticks), `:recurrent_stdp` (pair STDP every tick on sparse `W`
  nonzeros plus readout Hebbian), or `:none` (no weight updates).
- `recurrent_eta`: learning rate for pair STDP (default `1f-4`; must be finite when
  `plasticity=:recurrent_stdp`). Independent of `reflex_eta` / reflex gating.
- `sync`: call `CUDA.synchronize()` at end (default `true`).
- `record_history`: write spike history row (default `true`).
- `use_device_noise`: device `randn!` vs host upload (default `false`).

Runtime exceptions are captured to Sentry (when configured) before rethrow.
API misuse raises `LiquidCortexValidationError` and is not reported to Sentry.

# Returns
- `Nothing`: the lobe is updated in place; read the readout with [`get_output`](@ref)

# Examples
```julia
using LiquidCortex, CUDA
brain = SparseBrain(20.0f0; n_in=8, n_out=4)
u = CUDA.zeros(Float32, 8)
step!(brain, u; inhibition=0.5f0)          # raise threshold (quieter lobe)
step!(brain, u; plasticity=:none)          # freeze all weights
y = get_output(brain)                      # Vector{Float32} of length 4
```
"""
function step!(brain::SparseBrain, u::CuVector{Float32};
    inhibition::Real=0.0f0,
    reflex_eta::Real=ETA,
    plasticity::Symbol=:readout_only,
    recurrent_eta::Real=1.0f-4,
    sync::Bool=true,
    record_history::Bool=true,
    use_device_noise::Bool=false)
    try
        _step_impl!(brain, u;
            inhibition=inhibition,
            reflex_eta=reflex_eta,
            plasticity=plasticity,
            recurrent_eta=recurrent_eta,
            sync=sync,
            record_history=record_history,
            use_device_noise=use_device_noise)
    catch exc
        _capture_runtime_exception(exc, catch_backtrace())
        rethrow()
    end
end

"""
    get_output(brain::SparseBrain) -> Vector{Float32}

Copy the lobe readout from GPU to CPU.

# Arguments
- `brain::SparseBrain`: lobe whose `output` buffer is copied

# Returns
- `Vector{Float32}`: host copy of the `n_out`-element readout

# Examples
```julia
using LiquidCortex, CUDA
brain = SparseBrain(20.0f0; n_in=8, n_out=4)
step!(brain, CUDA.zeros(Float32, 8); inhibition=0.2f0)
y = get_output(brain)
length(y) == 4
```
"""
function get_output(brain::SparseBrain)
    return Array(brain.output)
end

"""
    diagnostics(brain::SparseBrain) -> String

Return a one-line diagnostic string with current lobe state.

# Arguments
- `brain::SparseBrain`: lobe to summarize

# Returns
- `String`: tick count, cumulative spikes, last spike rate, dynamic
  threshold, and `W_out` Frobenius norm. Spike totals/rates are only
  refreshed when the last [`step!`](@ref) used `sync=true`.

# Examples
```julia
using LiquidCortex, CUDA
brain = SparseBrain(20.0f0; n_in=8, n_out=4)
step!(brain, CUDA.zeros(Float32, 8))
println(diagnostics(brain))
# [brain] tick=1 spikes=… rate=…% V_thresh=-50.0 W_out_norm=…
```
"""
function diagnostics(brain::SparseBrain)
    return _format_diagnostics(
        brain.tick_count, brain.total_spikes, brain.last_spike_rate,
        brain.v_thresh_dynamic, norm(brain.W_out))
end

# ═══════════════════════════════════════════════════════════════════════════════
# High-Frequency Covariance Computation (GPU-intensive)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    compute_reservoir_covariance!(brain::SparseBrain) -> Union{Tuple{CuMatrix,Vector{Int}}, Nothing}

Compute a subsampled covariance matrix of reservoir spike history.
Subsamples `COV_SUBSAMPLE` (8192) neurons to avoid the full N×N matrix
which would be 65536² × 4 = 17 GB — doesn't fit in 16GB VRAM.

8192² × 4 bytes = 268 MB — fits comfortably while still driving GPU hard.

Uses the brain's internal rolling history buffer (`HIST_DEPTH × N`).
Returns `nothing` until the circular history has wrapped at least once
(`brain.hist_full`).

# Arguments
- `brain::SparseBrain`: lobe with a filled history buffer

# Returns
- `nothing` if `brain.hist_full == false`
- `(C, indices)` otherwise, where `C::CuMatrix{Float32}` is
  `8192 × 8192` and `indices::Vector{Int}` are the subsampled neuron ids

# Examples
```julia
using LiquidCortex, CUDA
brain = SparseBrain(20.0f0; n_in=8, n_out=4)
u = CUDA.zeros(Float32, 8)
for _ in 1:1000
    step!(brain, u; inhibition=0.1f0)
end
result = compute_reservoir_covariance!(brain)
C, indices = result                      # after hist_full
size(C) == (8192, 8192)
```
"""
function compute_reservoir_covariance!(brain::SparseBrain)
    if !brain.hist_full
        return nothing
    end

    # Subsample (clamped so N < COV_SUBSAMPLE does not BoundsError).
    indices = _cov_subsample_indices(N, COV_SUBSAMPLE)
    X = brain.history[:, indices]  # HIST_DEPTH × k on GPU

    # Covariance via CUBLAS SYRK: C = (1/(T-1)) * Xᵀ * X
    C = _spike_history_covariance(X, HIST_DEPTH)

    CUDA.synchronize()
    return (C, indices)
end

# ═══════════════════════════════════════════════════════════════════════════════
# EnsembleBrain: 4-Lobe Parallel Architecture
# ═══════════════════════════════════════════════════════════════════════════════

"""
    EnsembleBrain

Four parallel [`SparseBrain`](@ref) lobes with distinct membrane time constants,
combined by a fixed weighted sum of their readouts.

| Lobe        | `τ_m` | Default weight | Role                    |
|-------------|-------|----------------|-------------------------|
| Fast        | 10 ms | 0.4            | micro-structure         |
| Medium      | 25 ms | 0.3            | short-term patterns     |
| Slow        | 50 ms | 0.2            | multi-period swings     |
| Integrator  | 100 ms| 0.1            | trend following         |

Aggregation: `agg_output = Σᵢ weights[i] * lobes[i].output`.
Weights are `LOBE_WEIGHTS = Float32[0.4, 0.3, 0.2, 0.1]` (copied into `weights`).

# Fields
- `lobes::Vector{SparseBrain}`: the four lobes in Fast → Integrator order
- `lobe_names::Vector{String}`: `["Fast", "Medium", "Slow", "Integrator"]`
- `agg_output::CuVector{Float32}`: weighted-sum readout (`n_out`)
- `weights::Vector{Float32}`: per-lobe aggregation weights (sum to 1.0)
- `desynchronized::Bool`: `true` after a failed [`ensemble_step!`](@ref);
  further steps and reads raise [`EnsembleDesynchronizedError`](@ref)

Reuse an ensemble across trials with [`reset!`](@ref); release device memory
with [`free!`](@ref). Do not rely on GC after `ensemble_step!(; sync=false)`.

# Examples
```julia
using LiquidCortex, CUDA
ensemble = EnsembleBrain(; n_in=8, n_out=4)
u = CUDA.zeros(Float32, 8)
ensemble_step!(ensemble, u; inhibition=0.3f0)
y = get_ensemble_output(ensemble)
```
"""
mutable struct EnsembleBrain
    lobes::Vector{SparseBrain}
    lobe_names::Vector{String}
    agg_output::CuVector{Float32}   # Aggregated readout
    weights::Vector{Float32}        # Per-lobe aggregation weights
    desynchronized::Bool            # Failed step: do not mix mixed-time readouts
end

"""
    EnsembleDesynchronizedError <: Exception

Lobe `tick_count`s disagree, or a prior [`ensemble_step!`](@ref) failed after
partially mutating the ensemble, so aggregated output would mix simulated times.

Raised by [`ensemble_step!`](@ref) and [`get_ensemble_output`](@ref).
GPU neuron state is not rolled back; the ensemble is unusable until discarded.
"""
struct EnsembleDesynchronizedError <: Exception
    msg::String
end
Base.showerror(io::IO, e::EnsembleDesynchronizedError) =
    print(io, "EnsembleDesynchronizedError: ", e.msg)

function _ensemble_tick_mismatch(ticks::AbstractVector{<:Integer})
    isempty(ticks) && return nothing
    t0 = Int64(first(ticks))
    for i in eachindex(ticks)
        ti = Int64(ticks[i])
        ti == t0 && continue
        return (t0, Int(i), ti)
    end
    return nothing
end

function _assert_ticks_synchronized(ticks::AbstractVector{<:Integer})
    mismatch = _ensemble_tick_mismatch(ticks)
    mismatch === nothing && return nothing
    t0, i, ti = mismatch
    throw(EnsembleDesynchronizedError(
        "ensemble lobes desynchronized: lobe 1 tick=$(t0), lobe $i tick=$(ti)"))
end

function _assert_ensemble_clocks(ticks::AbstractVector{<:Integer}; desynchronized::Bool=false)
    if desynchronized
        throw(EnsembleDesynchronizedError(
            "ensemble is unusable after a failed step; lobe state may mix simulated times"))
    end
    _assert_ticks_synchronized(ticks)
    return nothing
end

function _assert_ensemble_synchronized!(eb::EnsembleBrain)
    ticks = Vector{Int64}(undef, length(eb.lobes))
    @inbounds for i in eachindex(eb.lobes)
        ticks[i] = eb.lobes[i].tick_count
    end
    _assert_ensemble_clocks(ticks; desynchronized=eb.desynchronized)
    return nothing
end

function _poison_ensemble!(eb::EnsembleBrain, prev_agg)
    eb.desynchronized = true
    try
        copyto!(eb.agg_output, prev_agg)
    catch
        # Best-effort: keep the poison flag even if restore fails.
    end
    return nothing
end

function _commit_ensemble_aggregate!(eb::EnsembleBrain)
    prev = copy(eb.agg_output)
    try
        _weighted_sum!(eb.agg_output, eb.weights, (lobe.output for lobe in eb.lobes))
    catch
        copyto!(eb.agg_output, prev)
        rethrow()
    end
    return nothing
end

"""
    EnsembleBrain(; n_in=14, n_out=16) -> EnsembleBrain

Initialize 4 parallel lobes × 65,536 neurons = 262,144 total neurons on GPU.

Each lobe is a [`SparseBrain`](@ref) with `τ_m ∈ {10, 25, 50, 100}` ms.
Readouts are aggregated with weights `[0.4, 0.3, 0.2, 0.1]`
(Fast / Medium / Slow / Integrator). Requires ≥14 GB VRAM.

# Keyword Arguments
- `n_in::Int=14`: input dimension shared by every lobe (must be positive)
- `n_out::Int=16`: readout dimension shared by every lobe (must be positive)

# Returns
- `EnsembleBrain`: four initialized lobes plus an aggregated readout buffer

# Examples
```julia
using LiquidCortex, CUDA
ensemble = EnsembleBrain()                    # defaults: n_in=14, n_out=16
ensemble = EnsembleBrain(; n_in=8, n_out=4)
u = CUDA.zeros(Float32, 8)
ensemble_step!(ensemble, u; inhibition=0.3f0, reflex_signal=0.2f0)
y = get_ensemble_output(ensemble)             # Vector{Float32} of length 4
```
"""
function EnsembleBrain(; n_in::Int=14, n_out::Int=16)
    _validate_lobe_dims(n_in, n_out)
    @debug "[ensemble] Initializing $(N_LOBES) lobes × $(N) = $(N_LOBES * N) neurons"

    lobes = SparseBrain[]
    for i in 1:N_LOBES
        @debug "[ensemble] Lobe $i/$(N_LOBES): $(LOBE_NAMES[i]) (τ_m=$(LOBE_TAUS[i])ms)"
        push!(lobes, SparseBrain(LOBE_TAUS[i]; n_in=n_in, n_out=n_out, name=LOBE_NAMES[i]))
    end

    agg_output = CUDA.zeros(Float32, n_out)

    CUDA.synchronize()
    # CUDA.jl 6+: free_memory() replaces available_memory()
    free_mem = CUDA.free_memory() / 1e9
    total_mem = CUDA.total_memory() / 1e9
    used = total_mem - free_mem
    @debug @sprintf("[ensemble] ✓ All %d lobes online — %d total neurons", N_LOBES, N_LOBES * N)
    @debug @sprintf("[ensemble] VRAM: %.2f / %.2f GB (%.0f%% used)", used, total_mem, used / total_mem * 100)

    EnsembleBrain(lobes, copy(LOBE_NAMES), agg_output, copy(LOBE_WEIGHTS), false)
end

# Internal implementation; public entry point is `ensemble_step!` (with Sentry capture).
function _ensemble_step_impl!(eb::EnsembleBrain, u::CuVector{Float32};
    inhibition::Real=0.0f0,
    reflex_eta::Real=ETA,
    reflex_signal::Real=0.0f0,
    plasticity::Symbol=:readout_only,
    recurrent_eta::Real=1.0f-4,
    sync::Bool=true,
    record_history::Bool=true,
    use_device_noise::Bool=false)
    inhibition = Float32(inhibition)
    reflex_eta = Float32(reflex_eta)
    reflex_signal = Float32(reflex_signal)
    # Reflex gating boosts Fast-lobe *readout* Hebbian only (`reflex_eta`).
    # Recurrent pair-STDP always uses the caller's `recurrent_eta` (independent).
    reflex_fast = _reflex_fast_eta(reflex_eta, reflex_signal)

    # Validate once before any STDP edge prewarm (avoids large allocs on bad kwargs).
    isempty(eb.lobes) || _validate_step_kwargs!(eb.lobes[1], u;
        plasticity=plasticity, recurrent_eta=recurrent_eta)
    _assert_ensemble_synchronized!(eb)

    prev_agg = copy(eb.agg_output)
    try
        # Prewarm STDP edge lists before the async lobe loop so the first
        # :recurrent_stdp ensemble step does not host-sync mid-loop per lobe.
        if plasticity === :recurrent_stdp && Float32(recurrent_eta) != 0.0f0
            for lobe in eb.lobes
                _ensure_edge_indices!(lobe)
            end
        end

        # Step all lobes without committing clocks, history, or mid-lobe sync.
        # Clocks and history advance together after aggregate + optional device
        # barrier so a later CUDA.synchronize() failure cannot look like a
        # completed tick or leave a ghost history row.
        for (i, lobe) in enumerate(eb.lobes)
            eta_lobe = _lobe_reflex_eta(i, reflex_eta, reflex_fast)  # Lobe 1 = Fast
            _step_impl!(lobe, u;
                inhibition=inhibition,
                reflex_eta=eta_lobe,
                plasticity=plasticity,
                recurrent_eta=recurrent_eta,
                sync=false,
                record_history=record_history,
                use_device_noise=use_device_noise,
                commit_clock=false)
        end

        # Aggregate readouts: weighted sum across lobes. Restore the previous
        # complete vector if this loop throws, so get_ensemble_output is never
        # a partial sum.
        _commit_ensemble_aggregate!(eb)

        # Queue history, then optional barrier, then host clocks / spike
        # fields. A failed synchronize() leaves hist_idx unchanged.
        for lobe in eb.lobes
            _queue_history_row!(lobe, record_history)
        end
        if sync
            n_spikes = [sum(lobe.S) for lobe in eb.lobes]
            CUDA.synchronize()
            for (i, lobe) in enumerate(eb.lobes)
                lobe.total_spikes += round(Int64, n_spikes[i])
                lobe.last_spike_rate = _spike_rate(n_spikes[i], N)
            end
        end

        for lobe in eb.lobes
            _advance_lobe_clock!(lobe, lobe.tick_count + 1, record_history)
        end
    catch
        _poison_ensemble!(eb, prev_agg)
        rethrow()
    end
    return nothing
end

"""
    ensemble_step!(eb, u; inhibition=0.0, reflex_eta=ETA, reflex_signal=0.0,
                   plasticity=:readout_only, recurrent_eta=1f-4, sync=true,
                   record_history=true, use_device_noise=false) -> Nothing

Step all 4 lobes independently on the same input, then aggregate readouts
with weights `[0.4, 0.3, 0.2, 0.1]`.

# Arguments
- `eb::EnsembleBrain`: ensemble (mutated in place)
- `u::CuVector{Float32}`: input shared by every lobe; `length(u)` must
  equal each lobe's `n_in`

# Keyword Arguments
- `inhibition`, `reflex_eta`, `plasticity`, `recurrent_eta`, `sync`,
  `record_history`, `use_device_noise`: forwarded to each lobe's `step!`.
- `reflex_signal`: when `|reflex_signal| > 0.1`, Fast lobe (index 1, τ_m=10ms)
  gets a 5× **readout** learning-rate boost (`reflex_eta` only). Does not scale
  `recurrent_eta` / pair STDP.

Mid-lobe `CUDA.synchronize()` is suppressed; one sync runs after aggregation
when `sync=true`. Spike-rate host reductions also run only when `sync=true`.

Runtime exceptions are captured to Sentry (when configured) before rethrow.
If lobe `tick_count`s disagree, or a prior ensemble step failed partway,
raises [`EnsembleDesynchronizedError`](@ref) before any lobe is stepped.
A throw after that check poisons the ensemble (`desynchronized=true`) so
a later read cannot mix simulated times or return a stale aggregate.

# Returns
- `Nothing`: read the aggregated readout with [`get_ensemble_output`](@ref)

# Examples
```julia
using LiquidCortex, CUDA
ensemble = EnsembleBrain(; n_in=8, n_out=4)
u = CUDA.zeros(Float32, 8)
ensemble_step!(ensemble, u; inhibition=0.3f0)
ensemble_step!(ensemble, u; inhibition=0.1f0, reflex_signal=0.5f0)  # Fast-lobe boost
y = get_ensemble_output(ensemble)
```
"""
function ensemble_step!(eb::EnsembleBrain, u::CuVector{Float32};
    inhibition::Real=0.0f0,
    reflex_eta::Real=ETA,
    reflex_signal::Real=0.0f0,
    plasticity::Symbol=:readout_only,
    recurrent_eta::Real=1.0f-4,
    sync::Bool=true,
    record_history::Bool=true,
    use_device_noise::Bool=false)
    try
        _ensemble_step_impl!(eb, u;
            inhibition=inhibition,
            reflex_eta=reflex_eta,
            reflex_signal=reflex_signal,
            plasticity=plasticity,
            recurrent_eta=recurrent_eta,
            sync=sync,
            record_history=record_history,
            use_device_noise=use_device_noise)
    catch exc
        _capture_runtime_exception(exc, catch_backtrace())
        rethrow()
    end
end

"""
    get_ensemble_output(eb::EnsembleBrain) -> Vector{Float32}

Copy the weighted-sum ensemble readout from GPU to CPU.

Raises [`EnsembleDesynchronizedError`](@ref) if lobe clocks disagree or a
prior ensemble step failed, so a caller cannot silently read a mix of
simulated times.

# Arguments
- `eb::EnsembleBrain`: ensemble whose `agg_output` is copied

# Returns
- `Vector{Float32}`: host copy of the `n_out`-element aggregated readout

# Examples
```julia
using LiquidCortex, CUDA
ensemble = EnsembleBrain(; n_in=8, n_out=4)
ensemble_step!(ensemble, CUDA.zeros(Float32, 8); inhibition=0.3f0)
y = get_ensemble_output(ensemble)
length(y) == 4
```
"""
function get_ensemble_output(eb::EnsembleBrain)
    _assert_ensemble_synchronized!(eb)
    return Array(eb.agg_output)
end

# Host-only desync predicate. Lives above `ensemble_diagnostics` so the
# public docstring attaches to the exported function, not this helper.
function _ensemble_diag_desync(desynchronized::Bool, ticks::AbstractVector{<:Integer})
    return desynchronized || _ensemble_tick_mismatch(ticks) !== nothing
end

"""
    ensemble_diagnostics(eb::EnsembleBrain) -> String

One-line-per-lobe diagnostic summary, joined with ` | `.

# Arguments
- `eb::EnsembleBrain`: ensemble to summarize

# Returns
- `String`: for each lobe, name, `τ_m`, tick, spike rate, and `W_out` norm.
  Prefixed with `[DESYNC] ` when lobe clocks disagree or the ensemble was
  poisoned by a failed step; that path is host-only (`W=n/a`) so a sticky
  device error cannot hide the prefix. Example shape:
  `[Fast:τ=10] tick=1 rate=1.23% W=0.4567 | [Medium:τ=25] …`

# Examples
```julia
using LiquidCortex, CUDA
ensemble = EnsembleBrain(; n_in=8, n_out=4)
ensemble_step!(ensemble, CUDA.zeros(Float32, 8))
println(ensemble_diagnostics(ensemble))
# [Fast:τ=10] tick=1 rate=…% W=… | [Medium:τ=25] tick=1 rate=…% W=… | …
```
"""
function ensemble_diagnostics(eb::EnsembleBrain)
    ticks = Vector{Int64}(undef, length(eb.lobes))
    @inbounds for i in eachindex(eb.lobes)
        ticks[i] = eb.lobes[i].tick_count
    end
    # Poison / clock mismatch is host-only. Skip `norm(W_out)` so a sticky
    # CUDA fault that poisoned the ensemble cannot hide the `[DESYNC]` line.
    desync = _ensemble_diag_desync(eb.desynchronized, ticks)
    lines = String[]
    for (i, lobe) in enumerate(eb.lobes)
        rate_pct = round(lobe.last_spike_rate * 100, digits=2)
        if desync
            push!(lines, _format_lobe_diagnostics(
                eb.lobe_names[i], lobe.tau_m, lobe.tick_count, rate_pct))
        else
            w_norm = round(Float64(norm(lobe.W_out)), digits=4)
            push!(lines, _format_lobe_diagnostics(
                eb.lobe_names[i], lobe.tau_m, lobe.tick_count, rate_pct, w_norm))
        end
    end
    prefix = desync ? "[DESYNC] " : ""
    return prefix * join(lines, " | ")
end

# ── step! for EnsembleBrain ──

"""
    step!(eb::EnsembleBrain, u; inhibition=0.0, reflex_eta=ETA, reflex_signal=0.0,
          plasticity=:readout_only, recurrent_eta=1f-4, sync=true,
          record_history=true, use_device_noise=false) -> Nothing

Forwards to [`ensemble_step!`](@ref). Same arguments, keywords, and
aggregation weights as that method.

# Arguments
- `eb::EnsembleBrain`: ensemble (mutated in place)
- `u::CuVector{Float32}`: input shared by every lobe

# Keyword Arguments
- `reflex_signal` (default `0`): Fast-lobe readout-learning boost when
  `|reflex_signal| > 0.1`
- `inhibition`, `reflex_eta`, `plasticity`, `recurrent_eta`, `sync`,
  `record_history`, `use_device_noise`: forwarded unchanged

# Returns
- `Nothing`: read the aggregated readout with [`get_ensemble_output`](@ref)

# Examples
```julia
using LiquidCortex, CUDA
ensemble = EnsembleBrain(; n_in=8, n_out=4)
step!(ensemble, CUDA.zeros(Float32, 8); inhibition=0.3f0, reflex_signal=0.0)
```
"""
function step!(eb::EnsembleBrain, u::CuVector{Float32};
    inhibition::Real=0.0f0,
    reflex_eta::Real=ETA,
    reflex_signal::Real=0.0f0,
    plasticity::Symbol=:readout_only,
    recurrent_eta::Real=1.0f-4,
    sync::Bool=true,
    record_history::Bool=true,
    use_device_noise::Bool=false)
    ensemble_step!(eb, u;
        inhibition=inhibition,
        reflex_eta=reflex_eta,
        reflex_signal=reflex_signal,
        plasticity=plasticity,
        recurrent_eta=recurrent_eta,
        sync=sync,
        record_history=record_history,
        use_device_noise=use_device_noise)
    return nothing
end

