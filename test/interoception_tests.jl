# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Machine Interoception Benchmark v1 (examples/interoception). CPU-safe cases
# always run; the replay case is gated by LiquidCortex._cuda_available[].

using Test
using TOML
using Random
using LiquidCortex

include(joinpath(@__DIR__, "..", "examples", "interoception", "replay.jl"))
const IB = InteroceptionBenchmark
const IB_DIR = joinpath(@__DIR__, "..", "examples", "interoception")
const IB_CFG = joinpath(IB_DIR, "config.example.toml")

@testset "Interoception benchmark" begin

    @testset "CPU: timebase policy is explicit" begin
        tb = IB.plan_timebase(100, 100)
        @test tb.mode === :identity
        tb = IB.plan_timebase(100, 500)
        @test (tb.mode, tb.factor) == (:mean_downsample, 5)
        # faster than the source: refuse unless zero-order hold is requested
        @test_throws IB.TimebaseError IB.plan_timebase(5000, 5)
        up = IB.plan_timebase(500, 100; upsample=:hold)
        @test (up.mode, up.factor) == (:hold_upsample, 5)
        @test_throws IB.TimebaseError IB.plan_timebase(100, 250)       # no interpolation
        @test_throws IB.TimebaseError IB.plan_timebase(100, 100; tau_mode=:bogus)

        X = Float64[1 0; 2 1; 3 0; 4 0; 5 1; 6 0; 7 9]
        out = IB.resample(IB.plan_timebase(100, 300), X, [false, true])
        @test out == [2.0 1.0; 5.0 1.0]                 # mean / max, trailing partial bin dropped
        held = IB.resample(IB.plan_timebase(300, 100; upsample=:hold), X[1:2, :], [false, true])
        @test size(held) == (6, 2) && held[1:3, 1] == [1.0, 1.0, 1.0]

        # irregular sampling is an error, not something to resample over
        @test IB.infer_source_dt_ms([0, 100, 200, 300]) == 100
        @test_throws IB.TimebaseError IB.infer_source_dt_ms([0, 100, 200, 900])
        @test_throws IB.TimebaseError IB.infer_source_dt_ms([0, 100, 100])

        hs = IB.horizon_steps(IB.plan_timebase(100, 100), [100, 250, 1000])
        @test [h.steps for h in hs] == [1, 0, 10]
        @test !isempty(hs[2].reason)
    end

    @testset "CPU: τ_m is never silently reinterpreted" begin
        slow = IB.plan_timebase(5000, 5000; tau_mode=:literal_ms)
        # a 10 ms lobe on a 5 s step would be 0.002 ticks: must fail loudly
        @test_throws IB.TimebaseError IB.lobe_taus_ticks(slow, [10.0, 25.0])
        @test IB.lobe_taus_ticks(slow, [20_000.0]) == Float32[4.0]
        rel = IB.plan_timebase(5000, 5000; tau_mode=:relative_ticks)
        @test IB.lobe_taus_ticks(rel, [10.0, 25.0]) == Float32[10.0, 25.0]
        fast = IB.plan_timebase(5, 5; tau_mode=:literal_ms)
        @test IB.lobe_taus_ticks(fast, [10.0, 100.0]) == Float32[2.0, 20.0]
    end

    @testset "CPU: dataset contract" begin
        sessions, feats, missing_feats, dt = IB.load_sessions(joinpath(IB_DIR, "fixtures");
            required=["gpu_temp_c", "vram_used_pct"])
        @test feats == IB.CANONICAL_FEATURES && isempty(missing_feats) && dt == 100
        @test length(sessions) == 6
        @test length(IB.dataset_hash(sessions)) == 64

        mktempdir() do d
            for f in readdir(joinpath(IB_DIR, "fixtures"); join=true)
                lines = readlines(f)
                keep = findall(!=("cpu_temp_c"), split(lines[1], ','))
                open(joinpath(d, basename(f)), "w") do io
                    for l in lines
                        println(io, join(split(l, ',')[keep], ','))
                    end
                end
            end
            _, f2, miss, _ = IB.load_sessions(d; required=["gpu_temp_c"])
            @test miss == ["cpu_temp_c"] && !("cpu_temp_c" in f2)     # reported, not faked
            @test_throws IB.DatasetError IB.load_sessions(d; required=["cpu_temp_c"])
        end
        mktempdir() do d
            write(joinpath(d, "a.csv"), "t_ms,gpu_temp_c,cpu_tctl_c\n0,50,40\n100,51,41\n")
            _, f3, _, _ = IB.load_sessions(d; column_map=Dict("cpu_tctl_c" => "cpu_temp_c"))
            @test f3 == ["gpu_temp_c", "cpu_temp_c"]                   # explicit rename only
            _, f4, _, _ = IB.load_sessions(d)
            @test f4 == ["gpu_temp_c"]                                 # unmapped columns are ignored
        end
        mktempdir() do d
            write(joinpath(d, "a.csv"), "t_ms,gpu_temp_c\n0,50\n100,\n200,52\n")
            @test_throws IB.DatasetError IB.load_sessions(d)           # missing cell: no implicit fill
        end
        mktempdir() do d
            write(joinpath(d, "a.csv"), "t_ms,gpu_temp_c\n0,50\n100,51\n")
            write(joinpath(d, "b.csv"), "t_ms,gpu_temp_c\n0,50\n500,51\n")
            @test_throws IB.DatasetError IB.load_sessions(d)           # mixed cadences
        end
    end

    @testset "CPU: split is by session and never overlaps" begin
        ids = ["s$i" for i in 1:8]
        sp = IB.split_sessions(ids)
        @test isempty(intersect(sp.train, sp.val)) && isempty(intersect(sp.train, sp.test)) &&
              isempty(intersect(sp.val, sp.test))
        @test sort(vcat(sp.train, sp.val, sp.test)) == ids
        @test IB.split_sessions(reverse(ids)) == sp                      # independent of file order
        @test_throws IB.DatasetError IB.split_sessions(ids; train=["s1", "s2"], val=["s2"], test=["s3"])
        @test_throws IB.DatasetError IB.split_sessions(ids; train=["s1"], test=["nope"])
        @test_throws IB.DatasetError IB.split_sessions(ids; train=["s1"], test=String[], val=["s2"])
        @test_throws IB.DatasetError IB.split_sessions(["a", "b"])       # too few for auto split
    end

    @testset "CPU: normalization uses train sessions only" begin
        tr = [randn(Random.Xoshiro(1), 50, 3) .* [1 5 0] .+ [0 10 2]]
        μ, σ = IB.fit_normalization(tr)
        @test σ[3] == 1.0                                                 # constant column -> 1, no NaN
        z = IB.zscore(tr[1], μ, σ)
        @test all(abs.(sum(z; dims=1) ./ 50) .< 1e-9)
        @test all(isfinite, IB.zscore(fill(1e6, 20, 3), μ, σ))
        # statistics are a function of the train matrices alone
        @test IB.fit_normalization(tr) == (μ, σ)
        @test IB.fit_normalization([tr[1], tr[1] .+ 100]) != (μ, σ)
    end

    @testset "CPU: windows do not leak future samples" begin
        sessions, feats, _, _ = IB.load_sessions(joinpath(IB_DIR, "fixtures"))
        S = Dict(s.id => copy(s.X) for s in sessions)
        policy = IB.Policy(Dict())
        targets = ["gpu_temp_c", "vram_used_pct", "gpu_power_w", "throttle_active"]
        stress = IB.make_stress_fn(feats, Dict())
        mk(d) = IB.build_split(["s01"], d, feats, targets, [1, 5], 5, policy, 10, stress)
        a = mk(S)
        t0 = 120
        S2 = Dict(k => copy(v) for (k, v) in S)
        S2["s01"][t0+1:end, :] .= -999.0                                  # rewrite the future
        b = mk(S2)
        early = findall(<=(t0), a.tidx)
        @test !isempty(early)
        @test a.X[early, :] == b.X[early, :]                              # features at t see only <= t
        @test a.cur[early, :] == b.cur[early, :]
        @test a.lab_now[early] == b.lab_now[early]
        @test a.stress[early] == b.stress[early]
        @test a.tidx[1] == 11                                             # warmup excluded
        @test a.tidx[end] == size(S["s01"], 1) - 5                        # max horizon fits in-session
        @test size(a.Yraw, 2) == 2 * length(targets)
    end

    @testset "CPU: supervisor policy + metrics" begin
        p = IB.Policy(Dict())
        @test IB.supervisor_label(p, 60, 30, 0) == 1
        @test IB.supervisor_label(p, 81, 30, 0) == 2
        @test IB.supervisor_label(p, 60, 96, 0) == 3
        @test IB.supervisor_label(p, 60, 30, 1) == 3                      # hardware throttle forces THROTTLE
        @test IB.supervisor_label(p, 91, 30, 1) == 4

        cm = IB.confusion([1, 1, 2, 4, 4, 4], [1, 2, 2, 1, 4, 4])
        @test cm[4, 1] == 1 && IB.false_safe_rate(cm) ≈ 1 / 3
        @test IB.false_safe_rate(IB.confusion([1, 2], [1, 2])) === nothing
        @test IB.macro_f1(IB.confusion([1, 2, 3, 4], [1, 2, 3, 4])) == 1.0
        @test IB.macro_f1(IB.confusion([1, 1, 2], [1, 1, 1])) ≈ (0.8 + 0.0) / 2

        ece, brier, _ = IB.reliability([0.0, 0.0, 1.0, 1.0], [0, 0, 1, 1])
        @test ece == 0 && brier == 0

        exceed = [false, false, false, false, true, true, false, false]
        alert = [false, false, true, false, false, false, false, false]
        st = IB.lead_time_stats(alert, exceed; h_steps=2, lookback_steps=4, dt_ms=100)
        @test st.n_events == 1 && st.leads == [(5 - 3 + 2) * 100.0]
        @test st.n_false_alerts == 0
        miss = IB.lead_time_stats(falses(8), exceed; h_steps=2, lookback_steps=4, dt_ms=100)
        @test miss.n_events == 1 && isempty(miss.leads)
        ex12 = [false, false, false, false, true, true, false, false, false, false, false, false]
        late = IB.lead_time_stats(push!(falses(11), false) .| (1:12 .== 8), ex12;
            h_steps=2, lookback_steps=4, dt_ms=100)
        @test isempty(late.leads) && late.n_false_alerts == 1 && late.n_alerts == 1
        # alerts without a full lookahead window are not scored as false
        edge = IB.lead_time_stats([false, false, false, false, false, false, true, false], exceed;
            h_steps=2, lookback_steps=4, dt_ms=100)
        @test edge.n_alerts == 0 && edge.n_false_alerts == 0
        # one alert is never credited to two separate onsets
        ex2 = [false, false, true, false, true, false, false, false, false, false]
        one = IB.lead_time_stats(1:10 .== 1, ex2; h_steps=1, lookback_steps=4, dt_ms=100)
        @test one.n_events == 2 && length(one.leads) == 1
    end

    @testset "CPU: baselines" begin
        rng = Random.Xoshiro(3)
        X = randn(rng, 200, 4); Wt = randn(rng, 4, 2)
        Y = X * Wt .+ [1.0 -2.0]
        m = IB.ridge_fit(X, Y, 1e-9)
        @test maximum(abs.(IB.predict(m, X) .- Y)) < 1e-4
        _, tab = IB.ridge_select(X[1:150, :], Y[1:150, :], X[151:end, :], Y[151:end, :], [1e-6, 1.0, 1e3])
        @test tab["selected"] == 1e-6
        _, tab0 = IB.ridge_select(X, Y, zeros(0, 4), zeros(0, 2), [1e-3, 1e-2, 1e-1])
        @test tab0["criterion"] == "none (no validation sessions)"

        Yn = tanh.(X * Wt)
        mlp = IB.fit_mlp(X, Yn; hidden=16, epochs=200, lr=0.02, rng=Random.Xoshiro(1))
        @test IB.mse(Yn, IB.predict(mlp, X)) < IB.mse(Yn, zeros(size(Yn)))
        a = IB.fit_mlp(X, Yn; hidden=8, epochs=20, rng=Random.Xoshiro(5))
        b = IB.fit_mlp(X, Yn; hidden=8, epochs=20, rng=Random.Xoshiro(5))
        @test a.W1 == b.W1                                                # seeded
    end

    @testset "CPU: online readout only learns from revealed targets" begin
        rng = Random.Xoshiro(2)
        F = randn(rng, 60, 3)
        Y = reshape(F * [1.0, 0.5, -1.0], :, 1)
        W0 = zeros(3, 1); b0 = zeros(1)
        P0 = IB.prequential_nlms(W0, b0, F, Y, [1:60], 5; lr=0.0, stress=zeros(60))
        @test all(P0 .== 0)
        P = IB.prequential_nlms(W0, b0, F, Y, [1:60], 5; lr=0.5, stress=zeros(60))
        @test all(P[1:6] .== 0)                                           # nothing revealed yet
        @test sum(abs.(P[40:end] .- Y[40:end])) < sum(abs.(Y[40:end]))
        # corrupting a target that is not revealed yet cannot change earlier predictions
        Y2 = copy(Y); Y2[30:end] .= 1e6
        P2 = IB.prequential_nlms(W0, b0, F, Y2, [1:60], 5; lr=0.5, stress=zeros(60))
        @test P2[1:35] == P[1:35]
    end

    @testset "CPU: JSON writer + CPU-only benchmark result" begin
        s = IB.json_string(Dict("a" => [1, 2.5, NaN], "b" => nothing, "c" => "x\"y"))
        @test occursin("null", s) && occursin("\\\"", s)

        cfg = TOML.parsefile(IB_CFG)
        r = IB.run_benchmark(cfg; base_dir=IB_DIR, use_gpu=false)
        part = r["dataset"]["partitions"]
        @test isempty(intersect(part["train"], part["test"])) && isempty(intersect(part["val"], part["test"])) &&
              isempty(intersect(part["train"], part["val"]))
        @test part["test"] == ["s05", "s06"]
        @test r["dataset"]["sha256"] isa String && length(r["dataset"]["sha256"]) == 64
        @test r["features"]["normalization"]["fit_on"] == "train sessions only"
        @test haskey(r["models"], "persistence") && haskey(r["models"], "linear_current_features") &&
              haskey(r["models"], "mlp_current_features")
        @test r["timebase"]["replay_dt_ms"] == 100 && r["timebase"]["synthetic_fill"] == false
        @test [h["ms"] for h in r["task"]["horizons"]] == [100, 500, 1000, 5000]
        pers = r["models"]["persistence"]["metrics"]
        @test pers["classification"]["labels"] == ["ALLOW", "WARN", "THROTTLE", "PAUSE"]
        @test r["experiments"]["skipped"] == "use_gpu=false"
        @test r["seed"] == 1234 && haskey(r["git"], "commit")

        # same fixture, same seed -> same summary
        r2 = IB.run_benchmark(cfg; base_dir=IB_DIR, use_gpu=false)
        @test IB.summaries_match(IB.deterministic_view(r), IB.deterministic_view(r2))

        # a replay cadence the source cannot support is rejected before any model work
        bad = deepcopy(cfg); bad["timebase"]["replay_dt_ms"] = 10
        @test_throws IB.TimebaseError IB.run_benchmark(bad; base_dir=IB_DIR, use_gpu=false)
        # horizons that are not a multiple of the replay cadence are listed, not rounded
        odd = deepcopy(cfg); odd["task"]["horizons_ms"] = [100, 250, 500]
        ro = IB.run_benchmark(odd; base_dir=IB_DIR, use_gpu=false)
        @test [h["ms"] for h in ro["task"]["skipped_horizons"]] == [250]
        # overlapping partitions are rejected
        ov = deepcopy(cfg); ov["split"]["val"] = ["s05"]
        @test_throws IB.DatasetError IB.run_benchmark(ov; base_dir=IB_DIR, use_gpu=false)
    end

    if LiquidCortex._cuda_available[]
        @testset "GPU: tiny replay fixture end to end" begin
            cfg = TOML.parsefile(IB_CFG)
            cfg["reservoir"]["readout_neurons"] = 64
            cfg["experiments"]["C"] = true
            r1 = IB.run_benchmark(cfg; base_dir=IB_DIR)
            r2 = IB.run_benchmark(cfg; base_dir=IB_DIR)

            # reproducible within the documented tolerance
            @test IB.summaries_match(IB.deterministic_view(r1), IB.deterministic_view(r2); rtol=1e-4, atol=1e-6)

            m = r1["models"]
            for n in ("liquid_single_A_readout_only", "liquid_ensemble_A_readout_only",
                      "liquid_ensemble_B_frozen", "liquid_ensemble_B_online_ungated",
                      "liquid_ensemble_B_online_reflex_gated", "liquid_ensemble_C_recurrent_stdp")
                @test haskey(m, n)
            end
            # :readout_only never touches recurrent W; recurrent_stdp does
            a = m["liquid_ensemble_A_readout_only"]
            @test a["recurrent_weights_unchanged"] == true
            @test a["recurrent_weight_sha256_before"] == a["recurrent_weight_sha256_after"]
            @test m["liquid_single_A_readout_only"]["recurrent_weights_unchanged"] == true
            @test m["liquid_ensemble_C_recurrent_stdp"]["recurrent_weights_unchanged"] == false
            # the frozen readout of Experiment B is exactly Experiment A
            @test IB.summaries_match(m["liquid_ensemble_B_frozen"]["metrics"], a["metrics"])

            @test a["lobes"] == ["Fast", "Medium", "Slow", "Integrator"]
            @test Set(keys(r1["experiments"]["lobe_ablation_A"])) == Set(a["lobes"])
            sysm = a["systems"]
            @test sysm["total_neurons"] == 4 * 256 && sysm["total_nonzeros"] > 0
            @test sysm["ms_per_replay_step"] > 0 && sysm["peak_vram_used_gb_sampled"] > 0
            @test haskey(r1["systems"], "cuda_device") && haskey(r1["experiments"], "reflex")
            @test a["size"]["total_params"] > a["size"]["trainable_params"]

            # causal: rewriting the future of a session leaves earlier reservoir features untouched
            sessions, feats, _, _ = IB.load_sessions(joinpath(IB_DIR, "fixtures"))
            Z = Float32.(sessions[1].X ./ 50)
            Z2 = copy(Z); Z2[101:end, :] .= 3.0f0
            tb = IB.plan_timebase(100, 100)
            spec = IB.ReservoirSpec(:ensemble, IB.lobe_taus_ticks(tb, [10.0, 25.0, 50.0, 100.0]),
                Float32[0.4, 0.3, 0.2, 0.1], 128, 0.05, 0.9f0, 2.0f0, 4, 4.0f0, 32,
                [:membrane, :trace], 4, 7)
            ra = IB.run_reservoir(spec, Dict("x" => Z), ["x"])
            rb = IB.run_reservoir(spec, Dict("x" => Z2), ["x"])
            @test ra.feats["x"][1:100, :] == rb.feats["x"][1:100, :]
            @test ra.feats["x"][101:end, :] != rb.feats["x"][101:end, :]
        end
    else
        @info "Skipping interoception GPU replay tests — no CUDA device available"
    end
end
