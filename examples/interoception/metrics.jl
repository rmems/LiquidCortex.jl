# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Forecast + supervisor metrics. Pure functions over plain arrays (CPU-safe).

mae(y, ŷ) = sum(abs.(y .- ŷ)) / length(y)
rmse(y, ŷ) = sqrt(sum((y .- ŷ) .^ 2) / length(y))

"""
    reliability(p, y; bins=10) -> (ece, brier, table)

Calibration of probabilities `p ∈ [0,1]` against binary outcomes `y`.
"""
function reliability(p::AbstractVector, y::AbstractVector; bins::Int=10)
    n = length(p)
    brier = sum((p .- y) .^ 2) / n
    ece = 0.0
    table = Dict{String,Any}[]
    for b in 1:bins
        lo, hi = (b - 1) / bins, b / bins
        sel = findall(x -> x >= lo && (b == bins ? x <= hi : x < hi), p)
        isempty(sel) && continue
        conf = sum(p[sel]) / length(sel); acc = sum(y[sel]) / length(sel)
        ece += length(sel) / n * abs(conf - acc)
        push!(table, Dict("bin" => b, "count" => length(sel), "mean_pred" => conf, "freq_true" => acc))
    end
    return ece, brier, table
end

function confusion(truth::AbstractVector{<:Integer}, pred::AbstractVector{<:Integer}, K::Int=4)
    cm = zeros(Int, K, K)                  # rows = truth, cols = prediction
    for (t, p) in zip(truth, pred)
        cm[t, p] += 1
    end
    return cm
end

"""
    macro_f1(cm) -> Float64

Mean F1 over classes with truth support (`sum(cm[k,:]) > 0`); a supported class that
is never predicted scores 0. Classes absent from the truth are excluded.
"""
function macro_f1(cm::AbstractMatrix{<:Integer})
    f1s = Float64[]
    for k in 1:size(cm, 1)
        support = sum(cm[k, :])
        support == 0 && continue
        tp = cm[k, k]; fp = sum(cm[:, k]) - tp; fn = support - tp
        d = 2tp + fp + fn
        push!(f1s, d == 0 ? 0.0 : 2tp / d)
    end
    return isempty(f1s) ? NaN : sum(f1s) / length(f1s)
end

"""
    false_safe_rate(cm; allow=1, pause=4) -> Union{Float64,Nothing}

`P(pred = ALLOW | truth = PAUSE)`; `nothing` when the truth has no PAUSE rows.
"""
function false_safe_rate(cm::AbstractMatrix{<:Integer}; allow::Int=1, pause::Int=4)
    n = sum(cm[pause, :])
    return n == 0 ? nothing : cm[pause, allow] / n
end

"""
    lead_time_stats(alert, exceed; h_steps, lookback_steps, dt_ms)

`alert[i]`/`exceed[i]` are per-row booleans of one contiguous session range. Row `i`
is issued at step `t_i` and refers to target time `t_i + h`; `exceed[i]` means the
true target at `t_i + h` is beyond the threshold. An *event* is an onset
(`exceed[j] && !exceed[j-1]`). The earliest alert at a row `i ∈ [j-lookback, j]`
covers it with lead `(j - i + h)·dt_ms` (time from issuing the alert to the
crossing). A *false alert* is an alerting row with no exceedance in rows
`[i, i+lookback]`. Alerts within `lookback_steps` of the range end lack a full
lookahead window and are excluded from `n_alerts` / `n_false_alerts`. Each onset only
considers alerts after the end of the previous exceedance episode, so one alert is
never credited to two events.
"""
function lead_time_stats(alert::AbstractVector{Bool}, exceed::AbstractVector{Bool};
    h_steps::Int, lookback_steps::Int, dt_ms::Real)
    n = length(alert)
    leads = Float64[]
    n_events = 0
    last_exceed = 0
    for j in 1:n
        if !exceed[j]
            continue
        end
        if j == 1 || exceed[j-1]
            last_exceed = j
            continue
        end
        n_events += 1
        lo = max(1, j - lookback_steps, last_exceed + 1)
        last_exceed = j
        k = findfirst(view(alert, lo:j))
        k === nothing && continue
        push!(leads, (j - (lo + k - 1) + h_steps) * dt_ms)
    end
    n_false = 0
    n_scored = 0
    for i in findall(alert)
        i + lookback_steps <= n || continue
        n_scored += 1
        any(view(exceed, i:i+lookback_steps)) || (n_false += 1)
    end
    return (n_events=n_events, leads=leads, n_alerts=n_scored, n_false_alerts=n_false)
end

function combine_lead_stats(parts)
    ne = sum(p.n_events for p in parts)
    leads = reduce(vcat, (p.leads for p in parts); init=Float64[])
    na = sum(p.n_alerts for p in parts); nf = sum(p.n_false_alerts for p in parts)
    return Dict{String,Any}("n_events" => ne, "n_detected" => length(leads),
        "detection_rate" => ne == 0 ? nothing : length(leads) / ne,
        "mean_lead_ms" => isempty(leads) ? nothing : sum(leads) / length(leads),
        "n_alert_rows" => na, "false_alert_fraction" => na == 0 ? nothing : nf / na)
end
