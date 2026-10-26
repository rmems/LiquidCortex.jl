# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Phase 0 — timebase contract (see docs/machine-interoception.md).
#
# One model step (`BrainConfig.dt` = 1 tick) == one *replay* interval. The replay
# interval is chosen explicitly and never silently inherited from the source
# sample cadence. Resampling is causal (a replay step only sees source samples at
# or before its own timestamp).

struct TimebaseError <: Exception
    msg::String
end
Base.showerror(io::IO, e::TimebaseError) = print(io, "TimebaseError: ", e.msg)

"""
    Timebase

- `source_dt_ms`: inferred source sample interval.
- `replay_dt_ms`: wall-clock span of one model step.
- `mode`: `:identity`, `:mean_downsample` (mean; `max` for binary flags) or
  `:hold_upsample` (zero-order hold, opt-in only).
- `factor`: source samples per replay step (`:mean_downsample`) or replay steps
  per source sample (`:hold_upsample`).
- `tau_mode`: `:relative_ticks` (τ_m are model ticks) or `:literal_ms` (τ_m are
  milliseconds, converted with `replay_dt_ms`).
- `model_dt`: `BrainConfig.dt` in ticks (always 1.0 in this benchmark).
"""
struct Timebase
    source_dt_ms::Float64
    replay_dt_ms::Float64
    mode::Symbol
    factor::Int
    tau_mode::Symbol
    model_dt::Float32
end

function _as_int_ratio(x::Float64; tol::Float64=1e-6)
    r = round(x)
    return (abs(x - r) <= tol * max(1.0, abs(x)) && r >= 1) ? Int(r) : nothing
end

"""
    infer_source_dt_ms(t_ms; jitter_tol=0.25) -> Float64

Median spacing of strictly increasing timestamps. Throws `TimebaseError` when any
gap deviates from the median by more than `jitter_tol` (relative), so dropped
samples or gaps are never resampled over silently.
"""
function infer_source_dt_ms(t_ms::AbstractVector{<:Real}; jitter_tol::Real=0.25)
    length(t_ms) >= 2 || throw(TimebaseError("need at least 2 timestamps"))
    d = diff(Float64.(t_ms))
    all(>(0), d) || throw(TimebaseError("timestamps must be strictly increasing"))
    m = sort(d)[cld(length(d), 2)]
    bad = findfirst(x -> abs(x - m) > jitter_tol * m, d)
    bad === nothing || throw(TimebaseError(
        "irregular sampling: gap of $(d[bad]) ms at row $(bad + 1) vs median $(m) ms " *
        "(jitter_tol=$(jitter_tol)); split or repair the session before replay"))
    return m
end

"""
    plan_timebase(source_dt_ms, replay_dt_ms; upsample=:error, tau_mode=:relative_ticks)

Choose the resampling policy:
- equal cadence → `:identity`;
- slower replay (integer multiple) → `:mean_downsample`;
- faster replay → `TimebaseError` unless `upsample=:hold` (zero-order hold, flagged
  in the result as synthetic fill);
- non-integer ratios → `TimebaseError` (no interpolation).
"""
function plan_timebase(source_dt_ms::Real, replay_dt_ms::Real;
    upsample::Symbol=:error, tau_mode::Symbol=:relative_ticks, model_dt::Real=1.0)
    s = Float64(source_dt_ms); r = Float64(replay_dt_ms)
    (isfinite(s) && s > 0 && isfinite(r) && r > 0) ||
        throw(TimebaseError("source_dt_ms and replay_dt_ms must be positive and finite"))
    tau_mode in (:relative_ticks, :literal_ms) ||
        throw(TimebaseError("tau_mode must be :relative_ticks or :literal_ms, got :$tau_mode"))
    upsample in (:error, :hold) ||
        throw(TimebaseError("upsample must be :error or :hold, got :$upsample"))
    if abs(s - r) <= 1e-9 * max(s, r)
        return Timebase(s, r, :identity, 1, tau_mode, Float32(model_dt))
    elseif r > s
        f = _as_int_ratio(r / s)
        f === nothing && throw(TimebaseError(
            "replay_dt_ms=$r is not an integer multiple of source_dt_ms=$s; " *
            "refusing to interpolate"))
        return Timebase(s, r, :mean_downsample, f, tau_mode, Float32(model_dt))
    else
        upsample === :hold || throw(TimebaseError(
            "replay_dt_ms=$r is faster than source_dt_ms=$s; the source has no data " *
            "between samples. Use a replay interval >= $s or set upsample = \"hold\" " *
            "to opt into zero-order hold"))
        f = _as_int_ratio(s / r)
        f === nothing && throw(TimebaseError(
            "source_dt_ms=$s is not an integer multiple of replay_dt_ms=$r"))
        return Timebase(s, r, :hold_upsample, f, tau_mode, Float32(model_dt))
    end
end

"""
    resample(tb, X, max_cols) -> Matrix{Float64}

Causal resampling of a `T×F` source matrix to the replay cadence. `max_cols[j]`
selects `max` instead of `mean` when binning column `j` (binary flags). Trailing
partial bins are dropped.
"""
function resample(tb::Timebase, X::AbstractMatrix{<:Real}, max_cols::AbstractVector{Bool})
    T, F = size(X)
    length(max_cols) == F || throw(TimebaseError("max_cols length $(length(max_cols)) != $F"))
    if tb.mode === :identity
        return Matrix{Float64}(X)
    elseif tb.mode === :mean_downsample
        nb = fld(T, tb.factor)
        nb >= 1 || throw(TimebaseError("session shorter than one replay step"))
        out = Matrix{Float64}(undef, nb, F)
        for k in 1:nb, j in 1:F
            seg = @view X[(k-1)*tb.factor+1:k*tb.factor, j]
            out[k, j] = max_cols[j] ? maximum(seg) : sum(seg) / tb.factor
        end
        return out
    else
        return Matrix{Float64}(X[repeat(1:T; inner=tb.factor), :])
    end
end

"""
    lobe_taus_ticks(tb, taus) -> Vector{Float32}

τ_m in model ticks. `:relative_ticks` uses `taus` as-is; `:literal_ms` divides by
`replay_dt_ms`. Throws when the forward-Euler leak would not decay
(`model_dt >= 2τ`), which is exactly the failure of running ms-calibrated
dynamics on a much slower cadence.
"""
function lobe_taus_ticks(tb::Timebase, taus::AbstractVector{<:Real})
    out = Float32[]
    for t in taus
        τ = tb.tau_mode === :literal_ms ? Float64(t) / tb.replay_dt_ms : Float64(t)
        (isfinite(τ) && tb.model_dt < 2 * τ) || throw(TimebaseError(
            "τ_m=$(t) ($(tb.tau_mode)) is $(τ) ticks at replay_dt_ms=$(tb.replay_dt_ms); " *
            "needs dt=$(tb.model_dt) < 2τ. Dynamics calibrated for faster samples cannot " *
            "be replayed at this cadence — use tau_mode=\"relative_ticks\" or a faster replay"))
        push!(out, Float32(τ))
    end
    return out
end

"""
    horizon_steps(tb, horizons_ms) -> Vector{NamedTuple}

Map physical horizons to integer model steps. Horizons that are not a positive
integer multiple of `replay_dt_ms` are reported with `steps = 0` and a reason
instead of being rounded.
"""
function horizon_steps(tb::Timebase, horizons_ms::AbstractVector{<:Real})
    map(horizons_ms) do h
        k = _as_int_ratio(Float64(h) / tb.replay_dt_ms)
        k === nothing ?
            (ms=Float64(h), steps=0, reason="not a positive integer multiple of replay_dt_ms=$(tb.replay_dt_ms)") :
            (ms=Float64(h), steps=k, reason="")
    end
end
