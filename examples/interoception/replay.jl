# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Machine Interoception Benchmark v1 — deterministic replay of held-out telemetry
# sessions through memoryless baselines and LiquidCortex reservoirs.
#
#   julia --project=. examples/interoception/replay.jl \
#       --config examples/interoception/config.example.toml --out result.json
#
# Evaluation only: nothing here controls a host. See docs/machine-interoception.md.

module InteroceptionBenchmark

using Random
using LinearAlgebra
using TOML
using Dates

include("timebase.jl")
include("dataset.jl")
include("metrics.jl")
include("baselines.jl")
include("json.jl")
include("reservoir.jl")

export run_benchmark, write_result, summaries_match, deterministic_view, TimebaseError, DatasetError

const SCHEMA_VERSION = 1

_get(d, path::Tuple, default) = begin
    cur = d
    for k in path
        (cur isa AbstractDict && haskey(cur, k)) || return default
        cur = cur[k]
    end
    cur
end

function _git_info(dir)
    try
        commit = readchomp(pipeline(`git -C $dir rev-parse HEAD`; stderr=devnull))
        dirty = !isempty(readchomp(pipeline(`git -C $dir status --porcelain`; stderr=devnull)))
        return Dict("commit" => commit, "dirty" => dirty)
    catch
        return Dict("commit" => "unknown", "dirty" => nothing)
    end
end

_onehot(lab, K=4) = [lab[i] == k ? 1.0 : 0.0 for i in eachindex(lab), k in 1:K]

"Mean-reduction of (temperature, VRAM) into a [0,1] stress score; recorded in the result."
function make_stress_fn(features, rcfg)
    it = findfirst(==("gpu_temp_c"), features); iv = findfirst(==("vram_used_pct"), features)
    tlo = Float64(get(rcfg, "temp_lo_c", 70.0)); thi = Float64(get(rcfg, "temp_hi_c", 90.0))
    vlo = Float64(get(rcfg, "vram_lo_pct", 80.0)); vhi = Float64(get(rcfg, "vram_hi_pct", 98.0))
    return row -> clamp(max((row[it] - tlo) / (thi - tlo), (row[iv] - vlo) / (vhi - vlo)), 0.0, 1.0)
end

struct EvalCtx
    targets::Vector{String}
    hs::Vector{NamedTuple{(:ms, :steps, :reason),Tuple{Float64,Int,String}}}
    class_steps::Int
    policy::Policy
    dt_ms::Float64
    lookback_steps::Int
end

function evaluate(ctx::EvalCtx, split::Split, Ŷ::Matrix{Float64}, pred_lab::Vector{Int})
    nT = length(ctx.targets)
    out = Dict{String,Any}()
    fc = Dict{String,Any}()
    for (hi, h) in enumerate(ctx.hs), (k, tg) in enumerate(ctx.targets)
        col = (hi - 1) * nT + k
        y = split.Yraw[:, col]; p = Ŷ[:, col]
        e = Dict{String,Any}("mae" => mae(y, p), "rmse" => rmse(y, p))
        if tg == "throttle_active"
            ece, brier, table = reliability(clamp.(p, 0, 1), y)
            e["ece"] = ece; e["brier"] = brier; e["reliability"] = table
        end
        thr = tg == "gpu_temp_c" ? ctx.policy.throttle_temp_c :
              tg == "vram_used_pct" ? ctx.policy.throttle_vram_pct : nothing
        if thr !== nothing
            parts = [lead_time_stats(p[rg] .>= thr, y[rg] .>= thr; h_steps=h.steps,
                lookback_steps=ctx.lookback_steps, dt_ms=ctx.dt_ms) for rg in split.ranges]
            e["threshold"] = thr
            e["crossing_lead"] = combine_lead_stats(parts)
        end
        get!(fc, tg, Dict{String,Any}())["h$(round(Int, h.ms))ms"] = e
    end
    out["forecast"] = fc
    cm = confusion(split.lab_fut, pred_lab)
    parts = [lead_time_stats(pred_lab[rg] .>= 2, split.lab_fut[rg] .>= 3; h_steps=ctx.class_steps,
        lookback_steps=ctx.lookback_steps, dt_ms=ctx.dt_ms) for rg in split.ranges]
    out["classification"] = Dict{String,Any}(
        "labels" => LABELS, "confusion_matrix_rows_truth" => cm,
        "accuracy" => sum(diag(cm)) / sum(cm), "macro_f1" => macro_f1(cm),
        "classes_with_support" => count(>(0), vec(sum(cm; dims=2))),
        "false_safe_rate" => false_safe_rate(cm),
        "warning_lead" => combine_lead_stats(parts))
    return out
end

function size_info(; trainable, frozen=0, flops)
    return Dict{String,Any}("trainable_params" => trainable, "frozen_params" => frozen,
        "total_params" => trainable + frozen, "flops_per_step_est" => flops)
end

function summaries_match(a, b; rtol::Real=1e-4, atol::Real=1e-6)
    if a isa AbstractDict && b isa AbstractDict
        keys(a) == keys(b) || return false
        return all(summaries_match(a[k], b[k]; rtol, atol) for k in keys(a))
    elseif a isa AbstractVector && b isa AbstractVector
        return length(a) == length(b) && all(summaries_match(x, y; rtol, atol) for (x, y) in zip(a, b))
    elseif a isa AbstractMatrix && b isa AbstractMatrix
        return size(a) == size(b) && all(summaries_match(x, y; rtol, atol) for (x, y) in zip(a, b))
    elseif a isa Real && b isa Real && !(a isa Bool) && !(b isa Bool)
        return (isnan(a) && isnan(b)) || isapprox(a, b; rtol=rtol, atol=atol)
    else
        return a == b
    end
end

"""
    deterministic_view(result) -> Dict

The parts of a result that must repeat for an identical fixture, seed and device:
everything except `meta`, `systems`, per-model `systems` timings and git info.
Compare two of them with [`summaries_match`](@ref) (documented tolerance:
`rtol = 1e-4`, `atol = 1e-6`).
"""
function deterministic_view(result::AbstractDict)
    out = Dict{String,Any}(k => v for (k, v) in result if !(k in ("meta", "systems", "git", "models")))
    out["models"] = Dict{String,Any}(n => Dict{String,Any}(k => v for (k, v) in m if k != "systems")
                                     for (n, m) in result["models"])
    return out
end

"""
    run_benchmark(cfg; base_dir, use_gpu=true) -> Dict

Run every enabled experiment and return the machine-readable result. `use_gpu=false`
(or no CUDA device) skips the reservoir sections and says so in the result.
"""
function run_benchmark(cfg::AbstractDict; base_dir::AbstractString=pwd(), use_gpu::Bool=true)
    t_wall = time()
    seed = Int(get(cfg, "seed", 1234))
    ds, tbc, tk = get(cfg, "dataset", Dict()), get(cfg, "timebase", Dict()), get(cfg, "task", Dict())
    targets = String.(get(tk, "targets", ["gpu_temp_c", "vram_used_pct", "gpu_power_w", "throttle_active"]))
    dir = get(ds, "dir", "fixtures"); dir = isabspath(dir) ? dir : joinpath(base_dir, dir)

    # ── Dataset ──────────────────────────────────────────────────────────────
    required = String.(get(ds, "required_features", unique(vcat(targets, ["gpu_temp_c", "vram_used_pct"]))))
    sessions, features, missing_features, src_dt = load_sessions(dir;
        time_column=get(ds, "time_column", "t_ms"), required=required,
        jitter_tol=Float64(get(ds, "jitter_tol", 0.25)),
        column_map=Dict{String,String}(string(k) => string(v) for (k, v) in get(ds, "column_map", Dict())))

    # ── Timebase ─────────────────────────────────────────────────────────────
    replay_dt = Float64(get(tbc, "replay_dt_ms", src_dt))
    tb = plan_timebase(src_dt, replay_dt; upsample=Symbol(get(tbc, "upsample", "error")),
        tau_mode=Symbol(get(tbc, "tau_mode", "relative_ticks")))
    max_cols = [f in MAX_AGG_FEATURES for f in features]
    replay = Dict(s.id => resample(tb, s.X, max_cols) for s in sessions)

    # ── Split + normalization (train sessions only) ─────────────────────────
    sp = get(cfg, "split", Dict())
    split = split_sessions([s.id for s in sessions]; train=get(sp, "train", String[]),
        val=get(sp, "val", String[]), test=get(sp, "test", String[]))
    μ, σ = fit_normalization([replay[i] for i in split.train])
    ti = [findfirst(==(t), features) for t in targets]
    any(isnothing, ti) && throw(DatasetError("targets not in features: $(targets[findall(isnothing, ti)])"))

    # ── Horizons ─────────────────────────────────────────────────────────────
    hs_all = horizon_steps(tb, Float64.(get(tk, "horizons_ms", [100, 500, 1000, 5000])))
    hs = [h for h in hs_all if h.steps > 0]
    isempty(hs) && throw(TimebaseError("no forecast horizon is a multiple of replay_dt_ms=$(tb.replay_dt_ms)"))
    class_ms = Float64(get(tk, "class_horizon_ms", hs[1].ms))
    ci = findfirst(h -> h.ms == class_ms, hs)
    ci === nothing && throw(TimebaseError("class_horizon_ms=$class_ms is not one of the usable horizons"))
    hsteps = [h.steps for h in hs]; hmax = maximum(hsteps)
    warmup = Int(get(tk, "warmup_steps", 20))
    policy = Policy(get(cfg, "policy", Dict()))
    stress_fn = make_stress_fn(features, get(cfg, "reflex", Dict()))
    lookback = max(1, round(Int, Float64(get(tk, "lead_lookback_ms", 10_000)) / tb.replay_dt_ms))
    mk(ids) = build_split(ids, replay, features, targets, hsteps, hs[ci].steps, policy, warmup, stress_fn)
    S = (train=mk(split.train), val=mk(split.val), test=mk(split.test))
    ctx = EvalCtx(targets, hs, hs[ci].steps, policy, tb.replay_dt_ms, lookback)
    nT, nH = length(targets), length(hs)
    HT = nT * nH
    ymu = repeat(μ[ti], nH); ysd = repeat(σ[ti], nH)
    tgt_z(s) = hcat((s.Yraw .- ymu') ./ ysd', _onehot(s.lab_fut))
    Ytr, Yval = tgt_z(S.train), tgt_z(S.val)

    function decode(P)
        Ŷ = P[:, 1:HT] .* ysd' .+ ymu'
        for (k, tg) in enumerate(targets), hi in 1:nH
            tg == "throttle_active" && (Ŷ[:, (hi-1)*nT+k] .= clamp.(Ŷ[:, (hi-1)*nT+k], 0, 1))
        end
        return Ŷ, [argmax(@view P[r, HT+1:end]) for r in 1:size(P, 1)]
    end

    models = Dict{String,Any}()
    function record!(name, family, split_eval, P_or_pair, sz; notes="", extra=Dict{String,Any}())
        Ŷ, lab = P_or_pair isa Tuple ? P_or_pair : decode(P_or_pair)
        models[name] = merge(Dict{String,Any}("family" => family, "notes" => notes,
            "size" => sz, "metrics" => evaluate(ctx, split_eval, Ŷ, lab)), extra)
    end

    # ── Baselines ────────────────────────────────────────────────────────────
    F = length(features); O = HT + 4
    record!("persistence", "baseline", S.test,
        (repeat(S.test.cur, 1, nH), copy(S.test.lab_now)), size_info(trainable=0, flops=0);
        notes="next = current; label = policy applied to the current state")
    Xz(s) = zscore(s.X, μ, σ)
    ridge_lams = Float64.(get(cfg, "baselines", Dict()) |> d -> get(d, "ridge_lambdas", [1e-4, 1e-3, 1e-2, 1e-1, 1.0]))
    lin, lin_sel = ridge_select(Xz(S.train), Ytr, Xz(S.val), Yval, ridge_lams)
    record!("linear_current_features", "baseline", S.test, predict(lin, Xz(S.test)),
        size_info(trainable=n_params(lin), flops=2F * O); extra=Dict{String,Any}("selection" => lin_sel))
    bc = get(cfg, "baselines", Dict())
    mlp = fit_mlp(Xz(S.train), Ytr; hidden=Int(get(bc, "mlp_hidden", 32)),
        epochs=Int(get(bc, "mlp_epochs", 300)), lr=Float64(get(bc, "mlp_lr", 0.01)),
        rng=Random.Xoshiro(seed + 1), Xval=Xz(S.val), Yval=Yval)
    H = size(mlp.W1, 2)
    record!("mlp_current_features", "baseline", S.test, predict(mlp, Xz(S.test)),
        size_info(trainable=n_params(mlp), flops=2 * (F * H + H * O)); notes="memoryless, tanh, Adam, early-stopped on val")

    # ── LiquidCortex ─────────────────────────────────────────────────────────
    systems = Dict{String,Any}("julia" => string(VERSION))
    experiments = Dict{String,Any}()
    rc = get(cfg, "reservoir", Dict())
    gpu_ok = use_gpu && LiquidCortex._cuda_available[]
    if !gpu_ok
        experiments["skipped"] = use_gpu ? "no CUDA device available" : "use_gpu=false"
    else
        systems["cuda_device"] = CUDA.name(CUDA.device())
        systems["vram_total_gb"] = CUDA.total_memory() / 1e9
        systems["liquidcortex_version"] = string(pkgversion(LiquidCortex))
        order = vcat(split.train, split.val, split.test)
        zs = Dict(id => Float32.(zscore(replay[id], μ, σ)) for id in order)
        stress_series = Dict(id => [stress_fn(@view replay[id][t, :]) for t in 1:size(replay[id], 1)] for id in order)
        feat_syms = Symbol.(get(rc, "features", ["membrane", "trace"]))
        taus_cfg = Float64.(get(rc, "taus", [10.0, 25.0, 50.0, 100.0]))
        wts = Float64.(get(rc, "weights", [0.4, 0.3, 0.2, 0.1]))
        taus_ticks = lobe_taus_ticks(tb, taus_cfg)
        single_tau = lobe_taus_ticks(tb, [Float64(get(rc, "single_tau", 25.0))])
        mkspec(kind, taus) = ReservoirSpec(kind, taus, Float32.(wts), Int(get(rc, "N", 256)),
            Float64(get(rc, "conn_prob", 0.05)), Float32(get(rc, "spectral_radius", 0.9)),
            Float32(get(rc, "sigma", 2.0)), Int(get(rc, "hist_depth", 4)),
            Float32(get(rc, "input_gain", 4.0)), Int(get(rc, "readout_neurons", 128)),
            feat_syms, Int(get(rc, "n_out", 4)), seed)
        rlams = Float64.(get(rc, "readout_lambdas", [1e-3, 1e-2, 1e-1, 1.0, 10.0]))
        exps = get(cfg, "experiments", Dict())
        experiments["timebase"] = Dict{String,Any}("tau_mode" => String(tb.tau_mode),
            "lobe_taus_input" => taus_cfg, "lobe_taus_ticks" => taus_ticks,
            "lobe_taus_ms_equivalent" => taus_ticks .* tb.replay_dt_ms)

        rows(run, s) = reduce(vcat, [run.feats[id][s.tidx[rg], :] for (id, rg) in zip(s.ids, s.ranges)];
            init=zeros(Float32, 0, size(first(values(run.feats)), 2)))
        function readout(run, cols)
            Ftr = Float64.(rows(run, S.train)[:, cols])
            fm = vec(sum(Ftr; dims=1)) ./ size(Ftr, 1)
            fs = [max(sqrt(sum((Ftr[:, j] .- fm[j]) .^ 2) / size(Ftr, 1)), 1e-6) for j in eachindex(fm)]
            feats_of(sp) = zscore(Float64.(rows(run, sp)[:, cols]), fm, fs)
            fit, selection = ridge_select(feats_of(S.train), Ytr, feats_of(S.val), Yval, rlams)
            return (model=fit, sel=selection, feats=feats_of)
        end
        function sysinfo(run)
            Dict{String,Any}("ms_per_replay_step" => run.ms_per_step, "replay_steps" => run.steps,
                "peak_vram_used_gb_sampled" => run.vram_used_gb_peak,
                "vram_before_run_gb" => run.vram_baseline_gb,
                "total_neurons" => run.neurons, "total_nonzeros" => run.nnz)
        end
        function res_size(run, model, D)
            nl = length(run.lobe_names)
            size_info(trainable=n_params(model), frozen=run.nnz + nl * Int(get(rc, "N", 256)) * F,
                flops=2 * run.nnz + 2 * nl * Int(get(rc, "N", 256)) * F + 2D * O)
        end
        evalname(n) = "liquid_$n"

        # Experiment A — frozen reservoir + trained readout (plasticity = :readout_only)
        runs = Dict{Symbol,ReservoirRun}()
        for (kind, taus) in ((:single, single_tau), (:ensemble, taus_ticks))
            run = run_reservoir(mkspec(kind, taus), zs, order; plasticity=:readout_only)
            runs[kind] = run
            D = run.cols_per_lobe * length(run.lobe_names)
            rd = readout(run, 1:D)
            nm = evalname("$(kind)_A_readout_only")
            record!(nm, "liquidcortex", S.test, predict(rd.model, rd.feats(S.test)), res_size(run, rd.model, D);
                notes="Experiment A: frozen recurrent W, host ridge readout on reservoir state",
                extra=Dict{String,Any}("selection" => rd.sel, "lobes" => run.lobe_names,
                    "recurrent_weights_unchanged" => run.w_hash_before == run.w_hash_after,
                    "recurrent_weight_sha256_before" => run.w_hash_before,
                    "recurrent_weight_sha256_after" => run.w_hash_after, "systems" => sysinfo(run)))
            if kind === :ensemble
                lob = Dict{String,Any}()
                for (li, lname) in enumerate(run.lobe_names)
                    cols = (li-1)*run.cols_per_lobe+1:li*run.cols_per_lobe
                    ab = readout(run, cols)
                    ab_hat, ab_lab = decode(predict(ab.model, ab.feats(S.test)))
                    lob[lname] = Dict("tau_ticks" => taus[li], "metrics" => evaluate(ctx, S.test, ab_hat, ab_lab), "selection" => ab.sel)
                end
                experiments["lobe_ablation_A"] = lob

                # Experiment B — reflex-gated online readout adaptation
                if get(exps, "B", true)
                    gain = Float64(get(_get(cfg, ("reflex",), Dict()), "gain", 4.0))
                    lr = Float64(get(_get(cfg, ("reflex",), Dict()), "online_lr", 0.05))
                    Fte = rd.feats(S.test)
                    for (tag, g) in (("frozen", nothing), ("online_ungated", 0.0), ("online_reflex_gated", gain))
                        P = g === nothing ? predict(rd.model, Fte) :
                            prequential_nlms(rd.model.W, rd.model.b, Fte, tgt_z(S.test), S.test.ranges, hmax;
                                lr=lr, gain=g, stress=S.test.stress)
                        record!(evalname("ensemble_B_$tag"), "liquidcortex", S.test, P, res_size(run, rd.model, D);
                            notes="Experiment B ($tag): prequential NLMS from the Experiment A readout; delay = max horizon",
                            extra=Dict{String,Any}("online_lr" => g === nothing ? nothing : lr,
                                "reflex_gain" => g, "systems" => sysinfo(run)))
                    end
                    experiments["reflex"] = Dict{String,Any}(
                        "signal" => "stress = clamp(max((gpu_temp_c-temp_lo)/(temp_hi-temp_lo), (vram_used_pct-vram_lo)/(vram_hi-vram_lo)), 0, 1)",
                        "source" => "current replay-cadence row of the same session (causal, raw units)",
                        "params" => _get(cfg, ("reflex",), Dict()), "gain_scale" => "step = lr*(1 + gain*stress)",
                        "note" => "the brain's built-in Hebbian W_out does not feed back into reservoir state, so reflex gating changes only the host online readout")
                end
            end
        end

        # Experiment C — recurrent pair-STDP ablation
        if get(exps, "C", false)
            run = run_reservoir(mkspec(:ensemble, taus_ticks), zs, order; plasticity=:recurrent_stdp)
            D = run.cols_per_lobe * length(run.lobe_names)
            rc_rd = readout(run, 1:D)
            record!(evalname("ensemble_C_recurrent_stdp"), "liquidcortex", S.test, predict(rc_rd.model, rc_rd.feats(S.test)),
                res_size(run, rc_rd.model, D);
                notes="Experiment C (ablation): recurrent pair-STDP stays on through train, val and test replay",
                extra=Dict{String,Any}("selection" => rc_rd.sel,
                    "recurrent_weights_unchanged" => run.w_hash_before == run.w_hash_after,
                    "recurrent_weight_sha256_before" => run.w_hash_before,
                    "recurrent_weight_sha256_after" => run.w_hash_after, "systems" => sysinfo(run)))
        end
        systems["reservoir"] = Dict(String(k) => sysinfo(v) for (k, v) in runs)
    end

    # Skill vs persistence (RMSE), filled after every model has been scored.
    pers = models["persistence"]["metrics"]["forecast"]
    for (_, m) in models, (tg, byh) in m["metrics"]["forecast"], (h, e) in byh
        p = pers[tg][h]["rmse"]
        e["rmse_skill_vs_persistence"] = p > 0 ? 1 - e["rmse"] / p : nothing
    end

    return Dict{String,Any}(
        "schema_version" => SCHEMA_VERSION,
        "benchmark" => "machine-interoception-v1",
        "seed" => seed,
        "git" => _git_info(@__DIR__),
        "meta" => Dict("created_utc" => string(Dates.now(Dates.UTC)), "wall_seconds" => time() - t_wall),
        "dataset" => Dict{String,Any}("dir" => get(ds, "dir", "fixtures"), "sha256" => dataset_hash(sessions),
            "sessions" => [Dict("id" => s.id, "sha256" => s.sha256, "source_rows" => length(s.t_ms),
                "replay_rows" => size(replay[s.id], 1)) for s in sessions],
            "partitions" => Dict("train" => split.train, "val" => split.val, "test" => split.test),
            "rows" => Dict(k => size(getfield(S, k).X, 1) for k in keys(S)),
            "missing_features" => missing_features),
        "timebase" => Dict{String,Any}("source_dt_ms" => tb.source_dt_ms, "replay_dt_ms" => tb.replay_dt_ms,
            "resample_mode" => String(tb.mode), "factor" => tb.factor, "tau_mode" => String(tb.tau_mode),
            "model_dt_ticks" => tb.model_dt,
            "synthetic_fill" => tb.mode === :hold_upsample),
        "features" => Dict("names" => features, "normalization" => Dict("fit_on" => "train sessions only",
            "mean" => Dict(zip(features, μ)), "std" => Dict(zip(features, σ)))),
        "task" => Dict{String,Any}("targets" => targets, "horizons" => [Dict("ms" => h.ms, "steps" => h.steps) for h in hs],
            "skipped_horizons" => [Dict("ms" => h.ms, "reason" => h.reason) for h in hs_all if h.steps == 0],
            "classification_horizon_ms" => class_ms, "labels" => LABELS, "policy" => Dict(string(k) => getfield(policy, k) for k in fieldnames(Policy)),
            "warmup_steps" => warmup, "lead_lookback_steps" => lookback),
        "config" => cfg,
        "models" => models,
        "experiments" => experiments,
        "systems" => systems,
    )
end

function write_result(path::AbstractString, result)
    mkpath(dirname(abspath(path)))
    open(io -> write_json(io, result), path, "w")
    return path
end

function main(args=ARGS)
    config = nothing; out = "interoception_result.json"; use_gpu = true; seed = nothing
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--config"; config = args[i+1]; i += 2
        elseif a == "--out"; out = args[i+1]; i += 2
        elseif a == "--seed"; seed = parse(Int, args[i+1]); i += 2
        elseif a == "--no-gpu"; use_gpu = false; i += 1
        else error("unknown argument $a (use --config FILE --out FILE [--seed N] [--no-gpu])")
        end
    end
    config === nothing && (config = joinpath(@__DIR__, "config.example.toml"))
    cfg = TOML.parsefile(config)
    seed === nothing || (cfg["seed"] = seed)
    result = run_benchmark(cfg; base_dir=dirname(abspath(config)), use_gpu=use_gpu)
    write_result(out, result)
    println("wrote ", out)
    return result
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    InteroceptionBenchmark.main()
end
