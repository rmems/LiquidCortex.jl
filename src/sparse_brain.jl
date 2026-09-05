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

# ── STDP Covariance Learning Parameters ──────────────────────────────────────
# ΔWᵢⱼ = η · Cᵢⱼ · exp(-|Δt| / τ_spike)

const ETA = 0.001f0       # Learning rate
const TAU_SPIKE = 20.0f0        # STDP time constant
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

# ── Pair STDP on existing sparse edges (experimental plasticity=:recurrent_stdp) ─
function _pair_stdp_kernel!(nzVal, pre_idx, post_idx, trace_pre, trace_post, S,
                            eta::Float32, w_max::Float32, nnz::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    @inbounds if i <= nnz
        pre = pre_idx[i]
        post = post_idx[i]
        # Pair rule: LTP when pre-trace co-occurs with post spike; LTD reverse
        dw = eta * (trace_pre[pre] * S[post] - S[pre] * trace_post[post])
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
- `total_spikes::Int64`: cumulative spike count (updated when `sync=true`)
- `last_spike_rate::Float32`: last-tick spike fraction (updated when `sync=true`)

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
- `tau_m::Float32`: membrane time constant in milliseconds

# Keyword Arguments
- `n_in::Int=14`: input dimension (must be positive)
- `n_out::Int=16`: readout dimension (must be positive)
- `name::String="default"`: label used in constructor progress logs

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
    n_in > 0 || throw(ArgumentError("n_in must be positive, got $n_in"))
    n_out > 0 || throw(ArgumentError("n_out must be positive, got $n_out"))
    println("[brain:$name] Initializing 65,536-neuron lobe (τ_m=$(tau_m)ms, in=$(n_in), out=$(n_out))...")

    xavier_std_in = sqrt(2.0f0 / Float32(n_in))
    xavier_std_out = sqrt(2.0f0 / Float32(N))

    # ── 1. Sparse recurrent weight matrix (Float16, 1% connectivity) ─────────
    nnz_expected = round(Int, N * N * CONN_PROB)
    println("[brain:$name] Generating sparse connectivity (~$(round(nnz_expected / 1e6, digits=1))M synapses)...")

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

    println("[brain:$name] W_sparse: $(actual_nnz) nnz, ρ≈$(round(target_rho, digits=2))")

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
    println("[brain:$name] W_out: Xavier/Glorot init σ=$(round(Float64(xavier_std_out), sigdigits=4))")

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
    println("[brain:$name] History buffer: $(HIST_DEPTH)×$(N) = $(round(HIST_DEPTH * N * 4 / 1e6, digits=1)) MB")

    CUDA.synchronize()
    println("[brain:$name] ✓ Lobe initialized (τ_m=$(tau_m)ms)")

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
    n_cols = length(colPtr) - 1
    edge_nnz = length(rowVal)
    # CSC invariants (1-based Julia SparseArrays). Failures here are internal
    # faults and remain Sentry-captured (not LiquidCortexValidationError).
    colPtr[1] == 1 ||
        throw(ArgumentError("Malformed CSC: colPtr[1]=$(colPtr[1]), expected 1"))
    colPtr[end] == edge_nnz + 1 ||
        throw(ArgumentError("Malformed CSC: colPtr[end]=$(colPtr[end]) vs nnz+1=$(edge_nnz + 1)"))
    @inbounds for col in 1:n_cols
        colPtr[col + 1] >= colPtr[col] ||
            throw(ArgumentError(
                "Malformed CSC: non-monotonic colPtr at col=$col ($(colPtr[col]) > $(colPtr[col + 1]))"))
    end
    pre = Vector{Int32}(undef, edge_nnz)
    post = Vector{Int32}(undef, edge_nnz)
    k = 1
    @inbounds for col in 1:n_cols
        p_lo = colPtr[col]
        p_hi = colPtr[col + 1] - 1
        p_hi > edge_nnz && throw(ArgumentError(
            "Malformed CSC: colPtr[$(col + 1)]=$(colPtr[col + 1]) exceeds nnz=$edge_nnz"))
        for p in p_lo:p_hi
            post[k] = Int32(rowVal[p])
            pre[k] = Int32(col)
            k += 1
        end
    end
    brain.pre_idx = CuArray(pre)
    brain.post_idx = CuArray(post)
    brain.nnz = edge_nnz
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

# Internal implementation; public entry point is `step!` (with Sentry capture).
function _step_impl!(brain::SparseBrain, u::CuVector{Float32};
    inhibition::Real=0.0f0,
    reflex_eta::Real=ETA,
    plasticity::Symbol=:readout_only,
    recurrent_eta::Real=1.0f-4,
    sync::Bool=true,
    record_history::Bool=true,
    use_device_noise::Bool=false)
    _validate_step_kwargs!(brain, u; plasticity=plasticity, recurrent_eta=recurrent_eta)
    brain.tick_count += 1
    inhibition = Float32(inhibition)
    reflex_eta = Float32(reflex_eta)
    recurrent_eta = Float32(recurrent_eta)

    # ── 1. Global Inhibition ─────────────────────────────────────────────────
    inhib = clamp(inhibition, 0.0f0, MAX_INHIBITION)
    brain.v_thresh_dynamic = V_THRESH + inhib * INHIBITION_GAIN

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
    if sync
        n_spikes = sum(brain.S)
        brain.total_spikes += round(Int64, n_spikes)
        brain.last_spike_rate = n_spikes / N
    end

    # ── 4. Optional history ──────────────────────────────────────────────────
    if record_history
        brain.history[brain.hist_idx, :] .= brain.S
        brain.hist_idx += 1
        if brain.hist_idx > HIST_DEPTH
            brain.hist_idx = 1
            brain.hist_full = true
        end
    end

    # ── 5. Traces + learning ─────────────────────────────────────────────────
    brain.trace_pre .= brain.trace_pre .* (1.0f0 - DT / TAU_TRACE) .+ brain.S
    brain.trace_post .= brain.trace_post .* (1.0f0 - DT / TAU_TRACE) .+ brain.S

    if plasticity === :recurrent_stdp
        _apply_pair_stdp!(brain; eta=recurrent_eta)
    end

    if plasticity !== :none && brain.tick_count % 10 == 0
        S_out = brain.output .> 0.0f0
        dW_out = reflex_eta .* (Float32.(S_out) * brain.trace_pre')
        brain.W_out .+= dW_out
        clamp!(brain.W_out, -W_MAX, W_MAX)
    end

    # ── 6. Readout ───────────────────────────────────────────────────────────
    mul!(brain.output, brain.W_out, brain.S)

    sync && CUDA.synchronize()
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
- `plasticity`: `:readout_only` (default, frozen W + Hebbian W_out every 10 ticks),
  `:recurrent_stdp` (pair STDP every tick on sparse W nonzeros + readout Hebbian),
  `:none` (no weight updates).
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
    return string(
        "[brain] tick=", brain.tick_count,
        " spikes=", brain.total_spikes,
        " rate=", round(brain.last_spike_rate * 100, digits=2), "%",
        " V_thresh=", round(brain.v_thresh_dynamic, digits=1),
        " W_out_norm=", round(Float64(norm(Array(brain.W_out))), digits=4)
    )
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

    # Subsample COV_SUBSAMPLE random neurons for tractable covariance
    indices = sort(randperm(N)[1:COV_SUBSAMPLE])
    X = brain.history[:, indices]  # HIST_DEPTH × COV_SUBSAMPLE on GPU

    # Mean-center the activity matrix
    μ = mean(X, dims=1)           # 1 × COV_SUBSAMPLE
    X_centered = X .- μ           # HIST_DEPTH × COV_SUBSAMPLE

    # Covariance via CUBLAS SYRK: C = (1/(T-1)) * Xᵀ * X
    # 8192 × 8192 dense matmul — drives tensor core utilization
    C = (X_centered' * X_centered) ./ Float32(HIST_DEPTH - 1)

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
    println()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║  Ensemble Brain — 4 Lobes × 65,536 = 262,144 Neurons      ║")
    println("║  Fast(10ms) │ Medium(25ms) │ Slow(50ms) │ Integrator(100ms) ║")
    println("╚══════════════════════════════════════════════════════════════╝")
    println()

    lobes = SparseBrain[]
    for i in 1:N_LOBES
        println("─── Lobe $i/$(N_LOBES): $(LOBE_NAMES[i]) (τ_m=$(LOBE_TAUS[i])ms) ───")
        push!(lobes, SparseBrain(LOBE_TAUS[i]; n_in=n_in, n_out=n_out, name=LOBE_NAMES[i]))
        println()
    end

    agg_output = CUDA.zeros(Float32, n_out)

    CUDA.synchronize()
    # CUDA.jl 6+: free_memory() replaces available_memory()
    free_mem = CUDA.free_memory() / 1e9
    total_mem = CUDA.total_memory() / 1e9
    used = total_mem - free_mem
    println("═══════════════════════════════════════════════════════════════")
    @printf("[ensemble] ✓ All %d lobes online — %d total neurons\n", N_LOBES, N_LOBES * N)
    @printf("[ensemble] VRAM: %.2f / %.2f GB (%.0f%% used)\n", used, total_mem, used / total_mem * 100)
    println("═══════════════════════════════════════════════════════════════")

    EnsembleBrain(lobes, copy(LOBE_NAMES), agg_output, copy(LOBE_WEIGHTS))
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
    reflex_fast = if abs(reflex_signal) > 0.1f0
        reflex_eta * 5.0f0   # 5× flash-learning rate
    else
        reflex_eta            # Normal learning rate
    end

    # Validate once before any STDP edge prewarm (avoids large allocs on bad kwargs).
    isempty(eb.lobes) || _validate_step_kwargs!(eb.lobes[1], u;
        plasticity=plasticity, recurrent_eta=recurrent_eta)

    # Prewarm STDP edge lists before the async lobe loop so the first
    # :recurrent_stdp ensemble step does not host-sync mid-loop per lobe.
    if plasticity === :recurrent_stdp && Float32(recurrent_eta) != 0.0f0
        for lobe in eb.lobes
            _ensure_edge_indices!(lobe)
        end
    end

    # Step all lobes; suppress mid-lobe sync (single sync after aggregate)
    for (i, lobe) in enumerate(eb.lobes)
        eta_lobe = (i == 1) ? reflex_fast : reflex_eta  # Lobe 1 = Fast
        _step_impl!(lobe, u;
            inhibition=inhibition,
            reflex_eta=eta_lobe,
            plasticity=plasticity,
            recurrent_eta=recurrent_eta,
            sync=false,
            record_history=record_history,
            use_device_noise=use_device_noise)
    end

    # Aggregate readouts: weighted sum across lobes
    eb.agg_output .= 0.0f0
    for (i, lobe) in enumerate(eb.lobes)
        eb.agg_output .+= eb.weights[i] .* lobe.output
    end

    # Diagnostics deferred to end of ensemble (lobes used sync=false).
    if sync
        for lobe in eb.lobes
            n_spikes = sum(lobe.S)
            lobe.total_spikes += round(Int64, n_spikes)
            lobe.last_spike_rate = n_spikes / N
        end
        CUDA.synchronize()
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
    return Array(eb.agg_output)
end

"""
    ensemble_diagnostics(eb::EnsembleBrain) -> String

One-line-per-lobe diagnostic summary, joined with ` | `.

# Arguments
- `eb::EnsembleBrain`: ensemble to summarize

# Returns
- `String`: for each lobe, name, `τ_m`, tick, spike rate, and `W_out` norm.
  Example shape:
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
    lines = String[]
    for (i, lobe) in enumerate(eb.lobes)
        rate_pct = round(lobe.last_spike_rate * 100, digits=2)
        w_norm = round(Float64(norm(Array(lobe.W_out))), digits=4)
        push!(lines, @sprintf("[%s:τ=%d] tick=%d rate=%.2f%% W=%.4f",
            eb.lobe_names[i], Int(lobe.tau_m), lobe.tick_count, rate_pct, w_norm))
    end
    return join(lines, " | ")
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

println("[brain] sparse_brain.jl loaded — EnsembleBrain (4-lobe, 262,144 neurons) ready")
