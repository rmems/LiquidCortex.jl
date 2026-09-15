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
const SPECTRAL_RADIUS = 0.9f0       # Target spectral radius (echo-state scaling)

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
# Used as the default noise scale; per-brain steps use `cfg.sigma * sqrt(cfg.dt)`.
const OU_NOISE_SCALE = SIGMA * sqrt(DT)

# ── Inhibition parameters ───────────────────────────────────────────────────
const INHIBITION_GAIN = 15.0f0    # mV increase per unit of inhibition
const MAX_INHIBITION = 3.0f0       # Maximum inhibition clamp

"""
    BrainConfig

Runtime hyperparameters for a [`SparseBrain`](@ref) lobe. Module-level
constants (`N`, `CONN_PROB`, …) remain the defaults; pass a customized
`BrainConfig` to sweep reservoir size, sparsity, spectral radius, LIF
biophysics, or the RNG stream.

`cov_subsample` is clamped to `min(cov_subsample, N)` so small reservoirs
do not `BoundsError` in [`compute_reservoir_covariance!`](@ref).

# Fields
- `N`: reservoir neuron count
- `conn_prob`: recurrent connection probability
- `spectral_radius`: target ρ used to scale sparse `W`
- `dt`: Euler–Maruyama step
- `hist_depth`: rolling spike-history rows
- `cov_subsample`: neurons kept for covariance (clamped to `N`)
- `v_rest`, `v_thresh`, `v_reset`: LIF voltages (mV)
- `sigma`: OU noise amplitude
- `refrac_t`: refractory period (ticks)
- `tau_trace`: STDP eligibility decay
- `w_max`: weight clamp for STDP / readout Hebbian
- `inhibition_gain`, `max_inhibition`: public `inhibition` kwarg scaling
- `rng`: host `AbstractRNG` used for topology, weight init, host noise, and
  covariance subsampling. Default is `Random.default_rng()` (same
  `Random.seed!` path as before). Pass `Random.Xoshiro(seed)` to isolate
  the reservoir from unrelated `rand` calls. A CUDA device generator is
  not valid here — use `use_device_noise=true` on [`step!`](@ref), which
  is seeded with `CUDA.seed!`, not `Random.seed!`.

# Examples
```julia
using LiquidCortex, Random
cfg = BrainConfig(N=256, spectral_radius=0.8f0, rng=Random.Xoshiro(42))
```
"""
struct BrainConfig
    N::Int
    conn_prob::Float64
    spectral_radius::Float32
    dt::Float32
    hist_depth::Int
    cov_subsample::Int
    v_rest::Float32
    v_thresh::Float32
    v_reset::Float32
    sigma::Float32
    refrac_t::Int
    tau_trace::Float32
    w_max::Float32
    inhibition_gain::Float32
    max_inhibition::Float32
    rng::AbstractRNG
end

function _validate_brain_config(
    n::Int, conn_prob::Float64, spectral_radius::Float32, dt::Float32,
    hist_depth::Int, cov_subsample::Int, v_rest::Float32, v_thresh::Float32,
    v_reset::Float32, sigma::Float32, refrac_t::Int, tau_trace::Float32,
    w_max::Float32, inhibition_gain::Float32, max_inhibition::Float32,
)
    n > 0 || throw(ArgumentError("N must be positive, got $n"))
    (isfinite(conn_prob) && 0 <= conn_prob <= 1) || throw(ArgumentError(
        "conn_prob must be in [0, 1], got $conn_prob"))
    (isfinite(spectral_radius) && 0 <= spectral_radius <= floatmax(Float16)) || throw(ArgumentError(
        "spectral_radius must be finite, ≥ 0, and ≤ floatmax(Float16), got $spectral_radius"))
    (isfinite(dt) && dt > 0) || throw(ArgumentError(
        "dt must be positive and finite, got $dt"))
    hist_depth > 0 || throw(ArgumentError(
        "hist_depth must be positive, got $hist_depth"))
    cov_subsample > 0 || throw(ArgumentError(
        "cov_subsample must be positive, got $cov_subsample"))
    isfinite(v_rest) || throw(ArgumentError("v_rest must be finite, got $v_rest"))
    isfinite(v_thresh) || throw(ArgumentError("v_thresh must be finite, got $v_thresh"))
    isfinite(v_reset) || throw(ArgumentError("v_reset must be finite, got $v_reset"))
    (isfinite(sigma) && sigma >= 0) || throw(ArgumentError(
        "sigma must be finite and ≥ 0, got $sigma"))
    refrac_t >= 0 || throw(ArgumentError("refrac_t must be ≥ 0, got $refrac_t"))
    (isfinite(tau_trace) && tau_trace > 0) || throw(ArgumentError(
        "tau_trace must be positive and finite, got $tau_trace"))
    dt < tau_trace || throw(ArgumentError(
        "dt must be < tau_trace so eligibility traces decay (got dt=$dt, tau_trace=$tau_trace)"))
    (isfinite(w_max) && 0 < w_max <= floatmax(Float16)) || throw(ArgumentError(
        "w_max must be finite, > 0, and ≤ floatmax(Float16), got $w_max"))
    isfinite(inhibition_gain) || throw(ArgumentError(
        "inhibition_gain must be finite, got $inhibition_gain"))
    (isfinite(max_inhibition) && max_inhibition >= 0) || throw(ArgumentError(
        "max_inhibition must be finite and ≥ 0, got $max_inhibition"))
    return nothing
end

"""
    BrainConfig(; N=N, conn_prob=CONN_PROB, spectral_radius=SPECTRAL_RADIUS, ...) -> BrainConfig

Keyword constructor. Numeric arguments are converted to the stored types.
`cov_subsample` is stored as `min(cov_subsample, N)`.
"""
function BrainConfig(;
    N::Integer=N,
    conn_prob::Real=CONN_PROB,
    spectral_radius::Real=SPECTRAL_RADIUS,
    dt::Real=DT,
    hist_depth::Integer=HIST_DEPTH,
    cov_subsample::Integer=COV_SUBSAMPLE,
    v_rest::Real=V_REST,
    v_thresh::Real=V_THRESH,
    v_reset::Real=V_RESET,
    sigma::Real=SIGMA,
    refrac_t::Integer=REFRAC_T,
    tau_trace::Real=TAU_TRACE,
    w_max::Real=W_MAX,
    inhibition_gain::Real=INHIBITION_GAIN,
    max_inhibition::Real=MAX_INHIBITION,
    rng::AbstractRNG=Random.default_rng(),
)
    n = Int(N)
    cp = Float64(conn_prob)
    rho = Float32(spectral_radius)
    dt32 = Float32(dt)
    hd = Int(hist_depth)
    cov = Int(cov_subsample)
    vr = Float32(v_rest)
    vt = Float32(v_thresh)
    vreset = Float32(v_reset)
    sig = Float32(sigma)
    rt = Int(refrac_t)
    tt = Float32(tau_trace)
    wm = Float32(w_max)
    ig = Float32(inhibition_gain)
    mi = Float32(max_inhibition)
    _validate_brain_config(n, cp, rho, dt32, hd, cov, vr, vt, vreset, sig, rt, tt, wm, ig, mi)
    _uses_device_rng(rng) && throw(ArgumentError(
        "BrainConfig.rng must be a host RNG; pass use_device_noise=true to step! for CUDA device noise"))
    return BrainConfig(n, cp, rho, dt32, hd, min(cov, n), vr, vt, vreset, sig, rt, tt, wm, ig, mi, rng)
end

function Base.show(io::IO, cfg::BrainConfig)
    print(io, "BrainConfig(N=", cfg.N,
        ", conn_prob=", cfg.conn_prob,
        ", spectral_radius=", cfg.spectral_radius,
        ", dt=", cfg.dt,
        ", hist_depth=", cfg.hist_depth,
        ", cov_subsample=", cfg.cov_subsample, ")")
end

@inline _ou_noise_scale(cfg::BrainConfig) = cfg.sigma * sqrt(cfg.dt)

# CUDA.jl 6: the device generator is `CUDA.RNG` (`CUDA.default_rng()` / `CUDA.seed!`).
function _uses_device_rng(rng::AbstractRNG)
    T = typeof(rng)
    return nameof(T) === :RNG && parentmodule(T) === CUDA
end

"""
    cpu_randn_cu(rng, dims...) -> CuArray{Float32}

Work around CUDA.jl RNG compilation failures on this stack by generating
Float32 Gaussian samples on the host (from `rng`) and uploading them.
"""
function cpu_randn_cu(rng::AbstractRNG, dims::Integer...)
    return cu(randn(rng, Float32, Int.(dims)...))
end

"""
    _generate_recurrent_cpu(cfg) -> SparseMatrixCSC{Float16,Int}

CPU-side sparse recurrent matrix: independent Bernoulli edges at
`cfg.conn_prob` (`sprand`), zero diagonal, then Frobenius / √nnz scaling
toward `cfg.spectral_radius`. Uses `cfg.rng`.
"""
function _generate_recurrent_cpu(cfg::BrainConfig)
    n = cfg.N
    p = cfg.conn_prob
    if p <= 0 || n <= 1
        return spzeros(Float16, n, n)
    end
    rng = cfg.rng
    W_cpu = sprand(rng, n, n, p, (r, len) -> Float16.(randn(r, Float32, len) .* 0.02f0))
    @inbounds for i in 1:n
        W_cpu[i, i] = Float16(0)
    end
    dropzeros!(W_cpu)
    actual_nnz = nnz(W_cpu)
    actual_nnz == 0 && return W_cpu
    frob = norm(W_cpu)
    (frob > 0 && isfinite(frob)) || return W_cpu
    spectral_approx = frob / sqrt(actual_nnz)
    scale_factor = cfg.spectral_radius / max(spectral_approx, 1.0f-6)
    scale_f16 = Float16(scale_factor)
    isfinite(scale_f16) || throw(ArgumentError(
        "recurrent Float16 scale overflowed (scale=$scale_factor, ρ=$(cfg.spectral_radius))"))
    W_cpu .*= scale_f16
    return W_cpu
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

A sparse CUDA reservoir lobe with OU-SDE membrane dynamics,
STDP-capable recurrent weights, and a dense readout.

Requires a CUDA GPU. Construct with
`SparseBrain(tau_m; cfg=BrainConfig(), n_in, n_out, name)`.
Default `cfg.N` is 65,536.

# Fields
- `W::CuSparseMatrixCSC{Float16,Int32}`: sparse recurrent weights
- `pre_idx::CuVector{Int32}`: pre-synaptic edge indices (lazy; empty until STDP)
- `post_idx::CuVector{Int32}`: post-synaptic edge indices (lazy; empty until STDP)
- `nnz::Int`: number of recurrent nonzeros
- `W_in::CuMatrix{Float32}`: dense input weights (`cfg.N × n_in`)
- `W_out::CuMatrix{Float32}`: dense readout weights (`n_out × cfg.N`)
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
- `cfg::BrainConfig`: reservoir hyperparameters and RNG
- `history::CuMatrix{Float32}`: rolling spike history (`cfg.hist_depth × cfg.N`)
- `hist_idx::Int64`: next history write index (circular)
- `hist_full::Bool`: `true` once the history buffer has wrapped
- `v_thresh_dynamic::Float32`: adaptive spike threshold (mV)
- `tick_count::Int64`: completed timesteps
- `total_spikes::Int64`: cumulative spike count (updated when `sync=true`)
- `last_spike_rate::Float32`: last-tick spike fraction (updated when `sync=true`)

# Examples
```julia
using LiquidCortex, CUDA, Random
brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="demo")
cfg = BrainConfig(N=256, rng=Random.Xoshiro(1))
small = SparseBrain(20.0f0; cfg=cfg, n_in=8, n_out=4)
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

    # ── Runtime hyperparameters + RNG ───────────────────────────────────────
    cfg::BrainConfig

    # ── Rolling spike history (hist_depth × N) for deep temporal covariance ──
    history::CuMatrix{Float32}
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
    SparseBrain(tau_m; cfg=BrainConfig(), n_in=14, n_out=16, name="default") -> SparseBrain

Initialize a sparse CUDA reservoir lobe.

Weight initialization:
  - W_recurrent: Sparse CSC, `cfg.conn_prob` connectivity, Float16
    Spectral radius controlled via scaling toward `cfg.spectral_radius`
  - W_in: Dense Float32, Xavier initialization √(2/n_in)
  - W_out: Dense Float32, Xavier/Glorot initialization √(2/N)
    (Breaks zero-readout deadlock — reservoir produces signals from tick 1)

# Arguments
- `tau_m::Float32`: membrane time constant in milliseconds (must be
  positive and finite). All of `0`, negatives, `NaN`, and `Inf` are
  rejected. `V` starts at `cfg.v_rest`, so the first membrane term is
  `0 / tau_m`; only `0` and `NaN` make that term `NaN`.

# Keyword Arguments
- `cfg::BrainConfig=BrainConfig()`: reservoir size, sparsity, LIF
  parameters, and RNG. Defaults match the module constants (`N=65_536`, …).
- `n_in::Int=14`: input dimension (must be positive)
- `n_out::Int=16`: readout dimension (must be positive)
- `name::String="default"`: label used in constructor progress logs

# Returns
- `SparseBrain`: GPU-resident lobe ready for [`step!`](@ref)

# Examples
```julia
using LiquidCortex, CUDA, Random
brain = SparseBrain(20.0f0)                      # defaults: n_in=14, n_out=16
brain = SparseBrain(25.0f0; n_in=8, n_out=4, name="custom")
small = SparseBrain(20.0f0; cfg=BrainConfig(N=256, rng=Random.Xoshiro(1)))
u = CUDA.zeros(Float32, brain.n_in)
step!(brain, u; inhibition=0.5f0)
```
"""
function SparseBrain(tau_m::Float32; cfg::BrainConfig=BrainConfig(),
    n_in::Int=14, n_out::Int=16, name::String="default")
    _validate_lobe_dims(n_in, n_out)
    _validate_tau_m(tau_m)
    n = cfg.N
    @debug "[brain:$name] Initializing $(n)-neuron lobe (τ_m=$(tau_m)ms, in=$(n_in), out=$(n_out))..."

    xavier_std_in = sqrt(2.0f0 / Float32(n_in))
    xavier_std_out = sqrt(2.0f0 / Float32(n))
    rng = cfg.rng

    # ── 1. Sparse recurrent weight matrix (Float16, cfg.conn_prob) ──────────
    nnz_expected = round(Int, n * n * cfg.conn_prob)
    @debug "[brain:$name] Generating sparse connectivity (~$(round(nnz_expected / 1e6, digits=1))M synapses)..."

    W_cpu = _generate_recurrent_cpu(cfg)

    @debug "[brain:$name] W_sparse: $(nnz(W_cpu)) nnz, ρ≈$(round(cfg.spectral_radius, digits=2))"

    # Transfer to GPU as CuSparseMatrixCSC
    # Edge lists for pair STDP are built lazily in `_ensure_edge_indices!`
    # (saves ~2×Int32×nnz ≈ 340 MB/lobe when STDP is unused).
    W_gpu = CUDA.CUSPARSE.CuSparseMatrixCSC(W_cpu)
    edge_nnz = nnz(W_cpu)
    pre_idx = CUDA.zeros(Int32, 0)
    post_idx = CUDA.zeros(Int32, 0)

    # ── 2. Input weight matrix (Dense, Xavier init) ──────────────────────────
    W_in = cpu_randn_cu(rng, n, n_in)
    W_in .*= xavier_std_in

    # ── 3. Output weight matrix — Xavier/Glorot (breaks zero-readout deadlock)
    # W_out ~ N(0, √(2/N)) — ensures non-trivial readout from tick 1
    W_out = cpu_randn_cu(rng, n_out, n)
    W_out .*= xavier_std_out
    @debug "[brain:$name] W_out: Xavier/Glorot init σ=$(round(Float64(xavier_std_out), sigdigits=4))"

    # ── 4. Neuron state vectors ──────────────────────────────────────────────
    V = CUDA.fill(Float32(cfg.v_rest), n)
    S = CUDA.zeros(Float32, n)
    refrac = CUDA.zeros(Int32, n)
    S_f16 = CUDA.zeros(Float16, n)
    I_rec = CUDA.zeros(Float32, n)
    I_ext = CUDA.zeros(Float32, n)
    noise = CUDA.zeros(Float32, n)

    # ── 5. STDP traces ──────────────────────────────────────────────────────
    trace_pre = CUDA.zeros(Float32, n)
    trace_post = CUDA.zeros(Float32, n)

    # ── 6. Output ────────────────────────────────────────────────────────────
    output = CUDA.zeros(Float32, n_out)

    # ── 7. Rolling spike history for deep temporal covariance (on GPU) ───────
    history = CUDA.zeros(Float32, cfg.hist_depth, n)
    @debug "[brain:$name] History buffer: $(cfg.hist_depth)×$(n) = $(round(cfg.hist_depth * n * 4 / 1e6, digits=1)) MB"

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
        cfg,
        history, 1, false,
        Float32(cfg.v_thresh),
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
        eta, brain.cfg.w_max, nnz)
    return nothing
end

"""Fill `brain.noise` from `cfg.rng`, or CUDA's device RNG when requested."""
function _fill_noise!(brain::SparseBrain; use_device_noise::Bool)
    cfg = brain.cfg
    rng = cfg.rng
    n = cfg.N
    want_device = use_device_noise || _uses_device_rng(rng)
    if want_device
        try
            if _uses_device_rng(rng)
                Random.randn!(rng, brain.noise)
            else
                Random.randn!(brain.noise)
            end
        catch e
            # Preserve cancel / GPU faults; only fall back for RNG path failures.
            e isa InterruptException && rethrow()
            e isa CUDA.OutOfGPUMemoryError && rethrow()
            e isa CUDA.CuError && rethrow()
            @warn "LiquidCortex: device RNG failed; falling back to host rng on BrainConfig. This desynchronizes the noise stream from a pure-device run." exception = e maxlog = 1
            copyto!(brain.noise, randn(rng, Float32, n))
        end
    else
        copyto!(brain.noise, randn(rng, Float32, n))
    end
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

    cfg = brain.cfg

    # ── 1. Global Inhibition ─────────────────────────────────────────────────
    inhib = clamp(inhibition, 0.0f0, cfg.max_inhibition)
    brain.v_thresh_dynamic = cfg.v_thresh + inhib * cfg.inhibition_gain

    # ── 2. OU-SDE Membrane Dynamics (per-lobe τ_m) ──────────────────────────
    # F16×F16 SpMV via * (generic mul! on F16 CSC can hit scalar indexing)
    brain.S_f16 .= Float16.(brain.S)
    y_sp = brain.W * brain.S_f16
    brain.I_rec .= Float32.(y_sp)

    mul!(brain.I_ext, brain.W_in, u)

    # Host noise is the portable default; `use_device_noise=true` uses
    # CUDA.default_rng() (seeded with CUDA.seed!).
    _fill_noise!(brain; use_device_noise=use_device_noise)
    brain.noise .*= _ou_noise_scale(cfg)

    dV = ((cfg.v_rest .- brain.V) ./ brain.tau_m .+ brain.I_rec .+ brain.I_ext) .* cfg.dt .+ brain.noise

    active_mask = brain.refrac .<= 0
    brain.V .+= dV .* Float32.(active_mask)

    # ── 3. Spike Detection ───────────────────────────────────────────────────
    spiked = brain.V .> brain.v_thresh_dynamic
    brain.S .= Float32.(spiked)

    brain.V .= ifelse.(spiked, Float32(cfg.v_reset), brain.V)
    brain.refrac .= ifelse.(spiked, Int32(cfg.refrac_t), max.(brain.refrac .- Int32(1), Int32(0)))

    # Host reductions force a stream wait. Skip when sync=false so ensemble
    # mid-lobe loops do not reintroduce implicit barriers (bench / chain mode).
    if sync
        n_spikes = sum(brain.S)
        brain.total_spikes += round(Int64, n_spikes)
        brain.last_spike_rate = n_spikes / cfg.N
    end

    # ── 4. Optional history ──────────────────────────────────────────────────
    if record_history
        brain.history[brain.hist_idx, :] .= brain.S
        brain.hist_idx += 1
        if brain.hist_idx > cfg.hist_depth
            brain.hist_idx = 1
            brain.hist_full = true
        end
    end

    # ── 5. Traces + learning ─────────────────────────────────────────────────
    brain.trace_pre .= brain.trace_pre .* (1.0f0 - cfg.dt / cfg.tau_trace) .+ brain.S
    brain.trace_post .= brain.trace_post .* (1.0f0 - cfg.dt / cfg.tau_trace) .+ brain.S

    if plasticity === :recurrent_stdp
        _apply_pair_stdp!(brain; eta=recurrent_eta)
    end

    if plasticity !== :none && brain.tick_count % 10 == 0
        S_out = brain.output .> 0.0f0
        dW_out = reflex_eta .* (Float32.(S_out) * brain.trace_pre')
        brain.W_out .+= dW_out
        clamp!(brain.W_out, -cfg.w_max, cfg.w_max)
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
- `inhibition`: global inhibition level (default `0.0`, clamped to
  `[0, brain.cfg.max_inhibition]`). Raises the spike threshold by
  `inhibition * brain.cfg.inhibition_gain` mV.
- `reflex_eta`: Hebbian learning rate for `W_out` (default `ETA`).
- `plasticity`: one of `:readout_only` (default; frozen `W` plus Hebbian `W_out`
  every 10 ticks), `:recurrent_stdp` (pair STDP every tick on sparse `W`
  nonzeros plus readout Hebbian), or `:none` (no weight updates).
- `recurrent_eta`: learning rate for pair STDP (default `1f-4`; must be finite when
  `plasticity=:recurrent_stdp`). Independent of `reflex_eta` / reflex gating.
- `sync`: call `CUDA.synchronize()` at end (default `true`).
- `record_history`: write spike history row (default `true`).
- `use_device_noise`: force CUDA's device RNG (default `false`, host samples
  from `brain.cfg.rng`). `true` uses `CUDA.default_rng()`, which is seeded
  by `CUDA.seed!` — not `Random.seed!`. `BrainConfig.rng` must remain a
  host generator so topology init can run on CPU.

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
        " W_out_norm=", round(Float64(norm(brain.W_out)), digits=4)
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# High-Frequency Covariance Computation (GPU-intensive)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    compute_reservoir_covariance!(brain::SparseBrain) -> Union{Tuple{CuMatrix,Vector{Int}}, Nothing}

Compute a subsampled covariance matrix of reservoir spike history.
Subsamples `brain.cfg.cov_subsample` neurons (clamped to `cfg.N` at
config construction) so the full N×N matrix is never materialized.

Uses the brain's internal rolling history buffer
(`cfg.hist_depth × cfg.N`). Neuron indices are drawn from `cfg.rng`.
Returns `nothing` until the circular history has wrapped at least once
(`brain.hist_full`).

# Arguments
- `brain::SparseBrain`: lobe with a filled history buffer

# Returns
- `nothing` if `brain.hist_full == false`
- `(C, indices)` otherwise, where `C::CuMatrix{Float32}` is
  `k × k` (`k = cfg.cov_subsample`) and `indices::Vector{Int}` are the
  subsampled neuron ids

# Examples
```julia
using LiquidCortex, CUDA, Random
cfg = BrainConfig(N=256, hist_depth=16, rng=Random.Xoshiro(1))
brain = SparseBrain(20.0f0; cfg=cfg, n_in=8, n_out=4)
u = CUDA.zeros(Float32, 8)
for _ in 1:16
    step!(brain, u; inhibition=0.1f0)
end
result = compute_reservoir_covariance!(brain)
C, indices = result                      # after hist_full
size(C) == (256, 256)
```
"""
function compute_reservoir_covariance!(brain::SparseBrain)
    if !brain.hist_full
        return nothing
    end

    cfg = brain.cfg
    k = cfg.cov_subsample
    # Subsample k neurons for tractable covariance (k already ≤ N)
    indices = sort(randperm(cfg.rng, cfg.N)[1:k])
    X = brain.history[:, indices]  # hist_depth × k on GPU

    # Mean-center the activity matrix
    μ = mean(X, dims=1)           # 1 × k
    X_centered = X .- μ           # hist_depth × k

    # Covariance via CUBLAS SYRK: C = (1/(T-1)) * Xᵀ * X
    denom = Float32(max(cfg.hist_depth - 1, 1))
    C = (X_centered' * X_centered) ./ denom

    CUDA.synchronize()
    return (C, indices)
end

# ═══════════════════════════════════════════════════════════════════════════════
# EnsembleBrain: 4-Lobe Parallel Architecture
# ═══════════════════════════════════════════════════════════════════════════════

"""
    EnsembleBrain

Parallel [`SparseBrain`](@ref) lobes with distinct membrane time constants,
combined by a weighted sum of their readouts.

Default construction uses four lobes:

| Lobe        | `τ_m` | Default weight | Role                    |
|-------------|-------|----------------|-------------------------|
| Fast        | 10 ms | 0.4            | micro-structure         |
| Medium      | 25 ms | 0.3            | short-term patterns     |
| Slow        | 50 ms | 0.2            | multi-period swings     |
| Integrator  | 100 ms| 0.1            | trend following         |

Pass `taus` / `weights` / `names` to change the lobe count. Aggregation:
`agg_output = Σᵢ weights[i] * lobes[i].output`. Default weights are
`LOBE_WEIGHTS = Float32[0.4, 0.3, 0.2, 0.1]` (copied into `weights`).

# Fields
- `lobes::Vector{SparseBrain}`: lobes in constructor order
- `lobe_names::Vector{String}`: display names (default Fast → Integrator)
- `agg_output::CuVector{Float32}`: weighted-sum readout (`n_out`)
- `weights::Vector{Float32}`: per-lobe aggregation weights

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
    EnsembleBrain(; n_in=14, n_out=16, cfg=BrainConfig(),
                  taus=LOBE_TAUS, weights=LOBE_WEIGHTS, names=nothing) -> EnsembleBrain

Initialize parallel lobes sharing `n_in` / `n_out` / `cfg`.

Default: 4 lobes × 65,536 neurons = 262,144 total neurons on GPU,
`τ_m ∈ {10, 25, 50, 100}` ms, weights `[0.4, 0.3, 0.2, 0.1]`.
Requires ≥14 GB VRAM at those defaults. Smaller `cfg.N` is the
supported way to prototype.

# Keyword Arguments
- `n_in::Int=14`: input dimension shared by every lobe (must be positive)
- `n_out::Int=16`: readout dimension shared by every lobe (must be positive)
- `cfg::BrainConfig=BrainConfig()`: shared reservoir hyperparameters / RNG
  stream (lobes draw topology sequentially from `cfg.rng`)
- `taus`: membrane time constants (ms); length sets the lobe count
- `weights`: per-lobe aggregation weights (same length as `taus`)
- `names`: optional display names (same length as `taus`); defaults to
  `LOBE_NAMES` when the count matches, otherwise `"Lobe1"`, `"Lobe2"`, …

# Returns
- `EnsembleBrain`: initialized lobes plus an aggregated readout buffer

# Examples
```julia
using LiquidCortex, CUDA, Random
ensemble = EnsembleBrain()                    # defaults: n_in=14, n_out=16
ensemble = EnsembleBrain(; n_in=8, n_out=4)
small = EnsembleBrain(; n_in=4, n_out=2,
    cfg=BrainConfig(N=128, hist_depth=8, rng=Random.Xoshiro(1)),
    taus=Float32[10, 50], weights=Float32[0.6, 0.4])
u = CUDA.zeros(Float32, 8)
ensemble_step!(ensemble, u; inhibition=0.3f0, reflex_signal=0.2f0)
y = get_ensemble_output(ensemble)             # Vector{Float32} of length 4
```
"""
function _validate_ensemble_spec(taus, weights, names)
    n_lobes = length(taus)
    n_lobes > 0 || throw(ArgumentError("taus must be non-empty"))
    length(weights) == n_lobes || throw(ArgumentError(
        "weights has length $(length(weights)), expected $n_lobes to match taus"))
    tau32 = Vector{Float32}(undef, n_lobes)
    for i in 1:n_lobes
        t = Float32(taus[i])
        (isfinite(t) && t > 0) || throw(ArgumentError(
            "taus[$i] must be positive and finite, got $(taus[i])"))
        tau32[i] = t
    end
    for (i, wt) in enumerate(weights)
        isfinite(Float32(wt)) || throw(ArgumentError(
            "weights[$i] must be finite, got $wt"))
    end
    if names !== nothing
        length(names) == n_lobes || throw(ArgumentError(
            "names has length $(length(names)), expected $n_lobes to match taus"))
    end
    return n_lobes, tau32
end

function _ensemble_lobe_names(n_lobes::Int, names)
    if names === nothing
        return n_lobes == length(LOBE_NAMES) ? copy(LOBE_NAMES) :
            ["Lobe$i" for i in 1:n_lobes]
    end
    return String[string(n) for n in names]
end

function EnsembleBrain(; n_in::Int=14, n_out::Int=16,
    cfg::BrainConfig=BrainConfig(),
    taus::AbstractVector{<:Real}=LOBE_TAUS,
    weights::AbstractVector{<:Real}=LOBE_WEIGHTS,
    names::Union{Nothing,AbstractVector}=nothing)
    _validate_lobe_dims(n_in, n_out)
    n_lobes, tau32 = _validate_ensemble_spec(taus, weights, names)
    lobe_names = _ensemble_lobe_names(n_lobes, names)
    w = Float32[Float32(x) for x in weights]
    @debug "[ensemble] Initializing $(n_lobes) lobes × $(cfg.N) = $(n_lobes * cfg.N) neurons"

    lobes = SparseBrain[]
    for i in 1:n_lobes
        @debug "[ensemble] Lobe $i/$(n_lobes): $(lobe_names[i]) (τ_m=$(tau32[i])ms)"
        push!(lobes, SparseBrain(tau32[i]; cfg=cfg, n_in=n_in, n_out=n_out, name=lobe_names[i]))
    end

    agg_output = CUDA.zeros(Float32, n_out)

    CUDA.synchronize()
    # CUDA.jl 6+: free_memory() replaces available_memory()
    free_mem = CUDA.free_memory() / 1e9
    total_mem = CUDA.total_memory() / 1e9
    used = total_mem - free_mem
    @debug @sprintf("[ensemble] ✓ All %d lobes online — %d total neurons", n_lobes, n_lobes * cfg.N)
    @debug @sprintf("[ensemble] VRAM: %.2f / %.2f GB (%.0f%% used)", used, total_mem, used / total_mem * 100)

    EnsembleBrain(lobes, lobe_names, agg_output, w)
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
            lobe.last_spike_rate = n_spikes / lobe.cfg.N
        end
        CUDA.synchronize()
    end
    return nothing
end

"""
    ensemble_step!(eb, u; inhibition=0.0, reflex_eta=ETA, reflex_signal=0.0,
                   plasticity=:readout_only, recurrent_eta=1f-4, sync=true,
                   record_history=true, use_device_noise=false) -> Nothing

Step all lobes independently on the same input, then aggregate readouts
with `eb.weights`.

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
        w_norm = round(Float64(norm(lobe.W_out)), digits=4)
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

