# SPDX-License-Identifier: MIT OR Apache-2.0
#
using Test
using LiquidCortex
using CUDA
using Sentry
using LinearAlgebra: norm
using Random
using SparseArrays

# Free VRAM between heavy GPU cases (65k lobes / ensembles leave large pools).
function reclaim_gpu!()
    CUDA.synchronize()
    GC.gc(true)
    CUDA.reclaim()
    GC.gc(true)
    CUDA.reclaim()
    return nothing
end

# CUDA.jl 6 made `device_reset!` a documented no-op (depwarn only). GPU cases
# call `free!` so the ~0.5–2 GB per reservoir returns to the pool instead of
# waiting on CuArray finalizers. This helper then syncs and reclaims; it cannot
# unreserve VRAM that is still held by a live 65k ensemble.
function reclaim_gpu_hard!()
    reclaim_gpu!()
    return nothing
end

# Kernel tests use a tiny reservoir. Default N=65_536 × hist_depth=1000
# (and 4-lobe ensembles of those) exhaust the 16GB CI GPU.
gpu_test_cfg(; N::Int=128, hist_depth::Int=8,
             rng::AbstractRNG=Random.Xoshiro(1), kwargs...) =
    BrainConfig(; N=N, hist_depth=hist_depth, rng=rng, kwargs...)

function snapshot_reference_lsm_state()
    # Copy reservoir state: run_lsm_step mutates _ref_x[] in-place.
    x_snap = LiquidCortex._ref_x[]
    return (
        W=LiquidCortex._ref_W[],
        Win=LiquidCortex._ref_Win[],
        Wout=LiquidCortex._ref_Wout[],
        x=isnothing(x_snap) ? nothing : copy(x_snap),
        n_in=LiquidCortex._ref_n_in[],
        n_out=LiquidCortex._ref_n_out[],
        initialized=LiquidCortex._ref_initialized[],
    )
end

function clear_reference_lsm_state!()
    LiquidCortex._ref_W[] = nothing
    LiquidCortex._ref_Win[] = nothing
    LiquidCortex._ref_Wout[] = nothing
    LiquidCortex._ref_x[] = nothing
    LiquidCortex._ref_n_in[] = LiquidCortex.REF_IN_DEFAULT
    LiquidCortex._ref_n_out[] = LiquidCortex.REF_OUT_DEFAULT
    LiquidCortex._ref_initialized[] = false
    return nothing
end

function restore_reference_lsm_state!(state)
    LiquidCortex._ref_W[] = state.W
    LiquidCortex._ref_Win[] = state.Win
    LiquidCortex._ref_Wout[] = state.Wout
    LiquidCortex._ref_x[] = state.x
    LiquidCortex._ref_n_in[] = state.n_in
    LiquidCortex._ref_n_out[] = state.n_out
    LiquidCortex._ref_initialized[] = state.initialized
    return nothing
end

function seed_test_rng!(seed::Integer)
    Random.seed!(seed)
    if LiquidCortex._cuda_available[]
        CUDA.seed!(seed)
    end
    return nothing
end

@testset "LiquidCortex" begin

    @testset "Package loads" begin
        @test @isdefined(LiquidCortex)
        @test LiquidCortex isa Module
    end

    @testset "Public API is exported" begin
        # Verify each name is both defined AND exported (not just defined)
        exports = names(LiquidCortex; all=false)
        for sym in [:SparseBrain, :EnsembleBrain, :BrainConfig, :EnsembleDesynchronizedError,
                    :step!, :ensemble_step!, :get_output, :get_ensemble_output,
                    :compute_reservoir_covariance!, :diagnostics, :ensemble_diagnostics,
                    :reset!, :free!, :enable_telemetry!]
            @test sym in exports
        end
        @test hasmethod(reset!, Tuple{SparseBrain})
        @test hasmethod(reset!, Tuple{EnsembleBrain})
        @test hasmethod(free!, Tuple{SparseBrain})
        @test hasmethod(free!, Tuple{EnsembleBrain})
    end

    @testset "Removed domain symbols are NOT exported" begin
        exports = names(LiquidCortex; all=false)
        for sym in [:MarketPulse, :decode_market_pulse, :pulse_to_input]
            @test !(sym in exports)
        end
    end

    @testset "CPU: plasticity mode validation" begin
        modes = LiquidCortex.PLASTICITY_MODES
        @test :readout_only in modes
        @test :recurrent_stdp in modes
        @test :none in modes
        @test !(:typo in modes)
        # Real CPU-safe validator (no CuArray)
        @test_throws LiquidCortex.LiquidCortexValidationError (
            LiquidCortex._validate_plasticity_kwargs(; plasticity=:typo, recurrent_eta=1f-4)
        )
        @test_throws LiquidCortex.LiquidCortexValidationError (
            LiquidCortex._validate_plasticity_kwargs(; plasticity=:recurrent_stdp, recurrent_eta=NaN32)
        )
        LiquidCortex._validate_plasticity_kwargs(; plasticity=:none, recurrent_eta=NaN32)
        @test LiquidCortex._should_capture_runtime_exception(
            LiquidCortex.LiquidCortexValidationError("x")) == false
        @test LiquidCortex._should_capture_runtime_exception(ErrorException("x")) == true
        @test LiquidCortex._should_capture_runtime_exception(ArgumentError("internal")) == true
    end

    @testset "CPU: reset! kwargs" begin
        LiquidCortex._validate_reset_kwargs(; keep_weights=true)
        @test_throws LiquidCortex.LiquidCortexValidationError (
            LiquidCortex._validate_reset_kwargs(; keep_weights=false)
        )
    end

    @testset "CPU: SparseBrain constructor validation" begin
        # Guards run before any host COO draw or CUDA allocation, so these
        # are CPU-safe. Exception type is pinned to ArgumentError to match
        # n_in/n_out and the reference LSM (not LiquidCortexValidationError).
        @test_throws ArgumentError SparseBrain(0.0f0)
        @test_throws ArgumentError SparseBrain(-1.0f0)
        @test_throws ArgumentError SparseBrain(NaN32)
        @test_throws ArgumentError SparseBrain(Inf32)
        @test_throws ArgumentError SparseBrain(20.0f0; n_in=0)
        @test_throws ArgumentError SparseBrain(20.0f0; n_out=-3)
        @test_throws ArgumentError EnsembleBrain(; n_in=0)
        @test_throws ArgumentError EnsembleBrain(; n_out=0)
        @test_throws ArgumentError EnsembleBrain(;
            taus=Float32[10.0], weights=Float32[0.5, 0.5])
        @test_throws ArgumentError EnsembleBrain(;
            taus=Float32[], weights=Float32[])
        @test_throws ArgumentError EnsembleBrain(;
            taus=Float32[10.0, 20.0], weights=Float32[0.5, 0.5],
            names=["only-one"])
        @test_throws ArgumentError EnsembleBrain(;
            taus=Float32[10.0], weights=Float32[NaN32])
        @test_throws ArgumentError EnsembleBrain(;
            taus=Float32[10.0], weights=Float32[Inf32])
        @test_throws ArgumentError EnsembleBrain(;
            taus=Float32[10.0, NaN32], weights=Float32[0.5, 0.5])
        @test_throws ArgumentError EnsembleBrain(;
            taus=Float32[10.0, 0.0], weights=Float32[0.5, 0.5])

        err = try
            SparseBrain(0.0f0)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("tau_m", err.msg)
        @test LiquidCortex._should_capture_runtime_exception(err) == true
    end

    @testset "CPU: BrainConfig defaults, validation, RNG isolation" begin
        cfg = BrainConfig()
        @test cfg.N == LiquidCortex.N
        @test cfg.conn_prob == LiquidCortex.CONN_PROB
        @test cfg.spectral_radius == LiquidCortex.SPECTRAL_RADIUS
        @test cfg.dt == LiquidCortex.DT
        @test cfg.hist_depth == LiquidCortex.HIST_DEPTH
        @test cfg.cov_subsample == LiquidCortex.COV_SUBSAMPLE
        @test cfg.v_rest == LiquidCortex.V_REST
        @test cfg.v_thresh == LiquidCortex.V_THRESH
        @test cfg.v_reset == LiquidCortex.V_RESET
        @test cfg.sigma == LiquidCortex.SIGMA
        @test cfg.refrac_t == LiquidCortex.REFRAC_T
        @test cfg.tau_trace == LiquidCortex.TAU_TRACE
        @test cfg.w_max == LiquidCortex.W_MAX
        @test cfg.inhibition_gain == LiquidCortex.INHIBITION_GAIN
        @test cfg.max_inhibition == LiquidCortex.MAX_INHIBITION
        @test cfg.rng === Random.default_rng()

        small = BrainConfig(N=100, cov_subsample=8192)
        @test small.N == 100
        @test small.cov_subsample == 100  # clamped to N

        @test_throws ArgumentError BrainConfig(N=0)
        @test_throws ArgumentError BrainConfig(N=-8)
        @test_throws ArgumentError BrainConfig(conn_prob=-0.1)
        @test_throws ArgumentError BrainConfig(conn_prob=1.1)
        @test_throws ArgumentError BrainConfig(spectral_radius=-1)
        @test_throws ArgumentError BrainConfig(spectral_radius=70_000)
        @test_throws ArgumentError BrainConfig(dt=0)
        @test_throws ArgumentError BrainConfig(dt=20, tau_trace=20)
        @test_throws ArgumentError BrainConfig(dt=25, tau_trace=20)
        @test_throws ArgumentError BrainConfig(hist_depth=0)
        @test_throws ArgumentError BrainConfig(hist_depth=1)
        @test_throws ArgumentError BrainConfig(cov_subsample=0)
        @test_throws ArgumentError BrainConfig(sigma=-1)
        @test_throws ArgumentError BrainConfig(refrac_t=-1)
        @test_throws ArgumentError BrainConfig(refrac_t=Int64(typemax(Int32)) + 1)
        @test_throws ArgumentError BrainConfig(tau_trace=0)
        @test_throws ArgumentError BrainConfig(w_max=0)
        @test_throws ArgumentError BrainConfig(w_max=70_000)
        @test_throws ArgumentError BrainConfig(inhibition_gain=-15)
        @test_throws ArgumentError BrainConfig(max_inhibition=-1)
        @test_throws ArgumentError BrainConfig(v_rest=NaN32)
        @test_throws ArgumentError BrainConfig(dt=Inf32)

        errN = try
            BrainConfig(N=0)
        catch e
            e
        end
        @test errN isa ArgumentError
        @test occursin("N must be positive", errN.msg)

        @test_throws ArgumentError BrainConfig(
            0, cfg.conn_prob, cfg.spectral_radius, cfg.dt, cfg.hist_depth,
            cfg.cov_subsample, cfg.v_rest, cfg.v_thresh, cfg.v_reset, cfg.sigma,
            cfg.refrac_t, cfg.tau_trace, cfg.w_max, cfg.inhibition_gain,
            cfg.max_inhibition, cfg.rng)

        W1 = LiquidCortex._generate_recurrent_cpu(BrainConfig(
            N=64, conn_prob=0.05, rng=Random.Xoshiro(42)))
        rand()  # pollute the global / task-local RNG
        W2 = LiquidCortex._generate_recurrent_cpu(BrainConfig(
            N=64, conn_prob=0.05, rng=Random.Xoshiro(42)))
        @test W1 == W2
        @test size(W1) == (64, 64)
        @test W1 isa SparseMatrixCSC
        @test all(i -> W1[i, i] == Float16(0), 1:64)

        W3 = LiquidCortex._generate_recurrent_cpu(BrainConfig(
            N=64, conn_prob=0.05, rng=Random.Xoshiro(43)))
        @test W3 != W1

        emptyW = LiquidCortex._generate_recurrent_cpu(BrainConfig(
            N=16, conn_prob=0.0, rng=Random.Xoshiro(1)))
        @test nnz(emptyW) == 0
        @test size(emptyW) == (16, 16)
        oneW = LiquidCortex._generate_recurrent_cpu(BrainConfig(
            N=1, conn_prob=1.0, rng=Random.Xoshiro(1)))
        @test nnz(oneW) == 0
        @test !any(isnan, nonzeros(oneW))
        fullW = LiquidCortex._generate_recurrent_cpu(BrainConfig(
            N=20, conn_prob=1.0, rng=Random.Xoshiro(1)))
        @test nnz(fullW) == 20 * 19
        @test all(isfinite, nonzeros(fullW))
        @test LiquidCortex._uses_device_rng(Random.Xoshiro(1)) == false
        @test LiquidCortex._uses_device_rng(Random.default_rng()) == false

        # Fractional τ is legal after BrainConfig; diagnostics must not Int() it.
        frac = LiquidCortex._ensemble_diag_line("A", 12.5f0, 3, 0.0123, 1.23456)
        @test occursin("[A:τ=12.5]", frac)
        @test occursin("tick=3", frac)
        @test occursin("rate=1.23%", frac)
        @test occursin("W=1.2346", frac)
        @test LiquidCortex._ensemble_diag_line("Fast", 10.0f0, 1, 0.0, 0.0) ==
            "[Fast:τ=10] tick=1 rate=0.00% W=0.0000"
        @test LiquidCortex._fast_lobe_index(Float32[100, 10, 50]) == 2
        @test LiquidCortex._lobe_reflex_eta(2, 0.001f0, 0.005f0, 2) == 0.005f0
        @test LiquidCortex._lobe_reflex_eta(1, 0.001f0, 0.005f0, 2) == 0.001f0
    end

    @testset "CPU: CUDA availability matches CUDA.functional" begin
        @test LiquidCortex._cuda_available[] == CUDA.functional()
    end

    @testset "CPU: host-side reservoir kernels" begin
        # Inhibition: clamp then raise threshold by INHIBITION_GAIN.
        @test LiquidCortex._inhibited_threshold(0.0f0) == LiquidCortex.V_THRESH
        @test LiquidCortex._inhibited_threshold(1.0f0) ==
              LiquidCortex.V_THRESH + LiquidCortex.INHIBITION_GAIN
        @test LiquidCortex._inhibited_threshold(100.0f0) ==
              LiquidCortex.V_THRESH + LiquidCortex.MAX_INHIBITION * LiquidCortex.INHIBITION_GAIN
        @test LiquidCortex._inhibited_threshold(-1.0f0) == LiquidCortex.V_THRESH

        # Reflex gating: 5× only when |signal| > 0.1.
        @test LiquidCortex._reflex_fast_eta(0.001f0, 0.0f0) == 0.001f0
        @test LiquidCortex._reflex_fast_eta(0.001f0, 0.1f0) == 0.001f0
        @test LiquidCortex._reflex_fast_eta(0.001f0, 0.2f0) == 0.001f0 * 5.0f0
        @test LiquidCortex._reflex_fast_eta(0.001f0, -0.2f0) == 0.001f0 * 5.0f0
        @test LiquidCortex._lobe_reflex_eta(1, 0.001f0, 0.005f0) == 0.005f0
        @test LiquidCortex._lobe_reflex_eta(2, 0.001f0, 0.005f0) == 0.001f0

        @test LiquidCortex._trace_decay_factor(LiquidCortex.DT, LiquidCortex.TAU_TRACE) ≈
              1.0f0 - 1.0f0 / 20.0f0
        @test LiquidCortex._spike_rate(128, 256) == 0.5f0

        idx, wrapped = LiquidCortex._advance_history_index(5, 1000)
        @test idx == 6 && wrapped == false
        idx, wrapped = LiquidCortex._advance_history_index(1000, 1000)
        @test idx == 1 && wrapped == true
        idx, wrapped = LiquidCortex._advance_history_index(1, 1)
        @test idx == 1 && wrapped == true

        # Pair STDP direction on a 2-neuron, 1-edge matrix (issue #55 item 4).
        eta = 0.1f0
        w_max = 1.0f0
        @test LiquidCortex._pair_stdp_dw(1.0f0, 1.0f0, 0.0f0, 0.0f0, eta) > 0   # LTP
        @test LiquidCortex._pair_stdp_dw(0.0f0, 0.0f0, 1.0f0, 1.0f0, eta) < 0   # LTD
        @test LiquidCortex._pair_stdp_dw(1.0f0, 1.0f0, 0.0f0, 0.0f0, 0.0f0) == 0

        nzVal = Float16[0.5]
        pre_idx = Int32[1]
        post_idx = Int32[2]
        LiquidCortex._pair_stdp_apply!(
            nzVal, pre_idx, post_idx,
            Float32[1.0, 0.0], Float32[0.0, 0.0], Float32[0.0, 1.0], eta, w_max)
        @test nzVal[1] > Float16(0.5)

        nzVal = Float16[0.5]
        LiquidCortex._pair_stdp_apply!(
            nzVal, pre_idx, post_idx,
            Float32[0.0, 0.0], Float32[0.0, 1.0], Float32[1.0, 0.0], eta, w_max)
        @test nzVal[1] < Float16(0.5)

        nzVal = Float16[0.5]
        LiquidCortex._pair_stdp_apply!(
            nzVal, pre_idx, post_idx,
            Float32[1.0, 0.0], Float32[0.0, 1.0], Float32[1.0, 1.0], 0.0f0, w_max)
        @test nzVal[1] == Float16(0.5)

        # Readout Hebbian: only positive outputs potentiate, in the trace direction.
        W = zeros(Float32, 2, 3)
        LiquidCortex._readout_hebbian_update!(
            W, Float32[1.0, -0.5], Float32[1.0, 0.0, 2.0], 0.1f0, 1.0f0)
        @test W[1, 1] ≈ 0.1f0
        @test W[1, 2] ≈ 0.0f0
        @test W[1, 3] ≈ 0.2f0
        @test all(W[2, :] .== 0.0f0)

        W_clamp = fill(0.95f0, 1, 1)
        LiquidCortex._readout_hebbian_update!(
            W_clamp, Float32[1.0], Float32[1.0], 1.0f0, 1.0f0)
        @test W_clamp[1, 1] == 1.0f0

        # CSC validation + edge-list construction (host arrays).
        colPtr = [1, 2, 2, 4]
        rowVal = [1, 1, 2]
        pre, post = LiquidCortex._csc_edge_lists(colPtr, rowVal)
        @test pre == Int32[1, 3, 3]
        @test post == Int32[1, 1, 2]
        @test_throws ArgumentError LiquidCortex._validate_csc(Int[], 0)
        @test_throws ArgumentError LiquidCortex._validate_csc([2, 2], 0)
        @test_throws ArgumentError LiquidCortex._validate_csc([1, 5], 3)
        @test_throws ArgumentError LiquidCortex._validate_csc([1, 3, 2, 4], 3)

        # Ensemble aggregation weights.
        agg = zeros(Float32, 4)
        outs = (ones(Float32, 4), fill(2.0f0, 4), zeros(Float32, 4), fill(0.5f0, 4))
        LiquidCortex._weighted_sum!(agg, Float32[0.4, 0.3, 0.2, 0.1], outs)
        @test all(agg .≈ 0.4f0 * 1 + 0.3f0 * 2 + 0.2f0 * 0 + 0.1f0 * 0.5f0)
        @test_throws BoundsError LiquidCortex._weighted_sum!(
            agg, Float32[0.4, 0.3], outs)

        # Covariance subsample clamps to N (would BoundsError at COV_SUBSAMPLE=8192).
        @test LiquidCortex._cov_subsample_count(256, 8192) == 256
        @test LiquidCortex._cov_subsample_count(65_536, 8192) == 8192
        rng = MersenneTwister(20260915)
        idx = LiquidCortex._cov_subsample_indices(10, 4, rng)
        @test length(idx) == 4
        @test issorted(idx)
        @test allunique(idx)
        rng = MersenneTwister(20260915)
        @test LiquidCortex._cov_subsample_indices(10, 4, rng) == idx

        X = Float32[1 2; 3 4; 5 6]
        C = LiquidCortex._spike_history_covariance(X, 3)
        @test C ≈ Float32[4 4; 4 4]

        diag = LiquidCortex._format_diagnostics(1, 10, 0.0123f0, -50.0f0, 1.2345)
        @test occursin("tick=1", diag)
        @test occursin("spikes=10", diag)
        @test occursin("rate=1.23%", diag)
        @test occursin("V_thresh=-50.0", diag)
        @test occursin("W_out_norm=1.2345", diag)

        lobe = LiquidCortex._format_lobe_diagnostics("Fast", 10.0f0, 1, 1.23, 0.4567)
        @test occursin("[Fast:τ=10]", lobe)
        @test occursin("tick=1", lobe)
        @test occursin("rate=1.23%", lobe)
        desync_lobe = LiquidCortex._format_lobe_diagnostics("Fast", 10.0f0, 1, 1.23)
        @test occursin("W=n/a", desync_lobe)
        @test occursin("[A:τ=12.5]",
            LiquidCortex._format_lobe_diagnostics("A", 12.5f0, 3, 1.23, 1.2346))
    end

    @testset "CPU: ensemble clock desync detection" begin
        # Production change that would fail this: dropping the mismatch helper
        # or treating a distinct lobe tick as synchronized.
        @test LiquidCortex._ensemble_tick_mismatch(Int64[]) === nothing
        @test LiquidCortex._ensemble_tick_mismatch(Int64[4, 4, 4, 4]) === nothing
        @test LiquidCortex._ensemble_tick_mismatch(Int64[4, 4, 5, 4]) == (Int64(4), 3, Int64(5))
        LiquidCortex._assert_ticks_synchronized(Int64[0, 0, 0, 0])
        LiquidCortex._assert_ensemble_clocks(Int64[1, 1, 1, 1]; desynchronized=false)
        @test_throws LiquidCortex.EnsembleDesynchronizedError (
            LiquidCortex._assert_ticks_synchronized(Int64[1, 1, 0, 1])
        )
        @test_throws LiquidCortex.EnsembleDesynchronizedError (
            LiquidCortex._assert_ensemble_clocks(Int64[1, 1, 1, 1]; desynchronized=true)
        )
        err = try
            LiquidCortex._assert_ticks_synchronized(Int64[1, 2])
        catch e
            e
        end
        @test err isa LiquidCortex.EnsembleDesynchronizedError
        @test occursin("desynchronized", err.msg)
        @test occursin("lobe 2", err.msg)
        @test LiquidCortex._should_capture_runtime_exception(err) == true
        poisoned = try
            LiquidCortex._assert_ensemble_clocks(Int64[4, 4, 4, 4]; desynchronized=true)
        catch e
            e
        end
        @test poisoned isa LiquidCortex.EnsembleDesynchronizedError
        @test occursin("unusable", poisoned.msg)
        @test LiquidCortex._should_capture_runtime_exception(poisoned) == true
        @test LiquidCortex._ensemble_diag_desync(false, Int64[1, 1, 1, 1]) == false
        @test LiquidCortex._ensemble_diag_desync(true, Int64[1, 1, 1, 1]) == true
        @test LiquidCortex._ensemble_diag_desync(false, Int64[1, 1, 2, 1]) == true
    end

    @testset "CPU: Reference LSM dimension validation" begin
        # Guards run before GPU allocation, so these are CPU-safe.
        @test_throws ArgumentError LiquidCortex._init_ref_lsm!(; n_in=0, n_out=4)
        @test_throws ArgumentError LiquidCortex._init_ref_lsm!(; n_in=4, n_out=0)
    end

    @testset "CPU: Sentry opt-in" begin
        @test !LiquidCortex._sentry_dsn_usable("")
        @test !LiquidCortex._sentry_dsn_usable("http://abc@host/1")
        @test !LiquidCortex._sentry_dsn_usable("not-a-dsn")
        @test LiquidCortex._sentry_dsn_usable("https://abcdef1234567890@a12345.us.sentry.io/1234567890123456789")
        @test_throws LiquidCortex.LiquidCortexValidationError enable_telemetry!("http://abc@host/1")
        @test_throws LiquidCortex.LiquidCortexValidationError enable_telemetry!("not-a-dsn")
        if isempty(get(ENV, "LIQUIDCORTEX_SENTRY_DSN", ""))
            @test LiquidCortex._sentry_enabled[] == false
        end
        tags = LiquidCortex._runtime_exception_tags(ErrorException("x"))
        @test tags["package"] == "LiquidCortex.jl"
        @test tags["gpu_failure"] == "false"
        @test tags["error_class"] == "runtime"
        @test !haskey(Sentry.global_tags, "error_class")
    end

    # Reference LSM (2,048-neuron dense reservoir). GPU-only; skip cleanly on CPU.
    @testset "Reference LSM" begin
        if LiquidCortex._cuda_available[]
            @testset "Lazy initialization" begin
                original_state = snapshot_reference_lsm_state()
                try
                    clear_reference_lsm_state!()
                    @test !LiquidCortex._ref_is_initialized()

                    # First call triggers lazy GPU init from input length / default n_out.
                    input = zeros(Float32, LiquidCortex.REF_IN_DEFAULT)
                    output = LiquidCortex.run_lsm_step(input, 0.5f0)
                    @test LiquidCortex._ref_is_initialized()
                    @test length(output) == LiquidCortex.REF_OUT_DEFAULT
                    @test size(LiquidCortex._ref_Win[]) == (
                        LiquidCortex.REF_N,
                        LiquidCortex.REF_IN_DEFAULT,
                    )
                    @test size(LiquidCortex._ref_Wout[]) == (
                        LiquidCortex.REF_OUT_DEFAULT,
                        LiquidCortex.REF_N,
                    )
                finally
                    restore_reference_lsm_state!(original_state)
                end
            end

            @testset "Custom dimensions" begin
                original_state = snapshot_reference_lsm_state()
                try
                    clear_reference_lsm_state!()
                    input = zeros(Float32, 8)
                    output = LiquidCortex.run_lsm_step(input, 0.0f0; n_out=4)
                    @test length(output) == 4
                    @test LiquidCortex._ref_n_in[] == 8
                    @test LiquidCortex._ref_n_out[] == 4
                    @test size(LiquidCortex._ref_Win[]) == (LiquidCortex.REF_N, 8)
                    @test size(LiquidCortex._ref_Wout[]) == (4, LiquidCortex.REF_N)
                finally
                    restore_reference_lsm_state!(original_state)
                end
            end

            @testset "Dimension mismatch after init" begin
                original_state = snapshot_reference_lsm_state()
                try
                    clear_reference_lsm_state!()
                    # Init with 16 inputs, then reject a mismatched length.
                    LiquidCortex.run_lsm_step(zeros(Float32, 16), 0.0f0)
                    @test_throws DimensionMismatch LiquidCortex.run_lsm_step(
                        zeros(Float32, 8),
                        0.0f0,
                    )
                finally
                    restore_reference_lsm_state!(original_state)
                end
            end

            @testset "run_lsm_step_str" begin
                original_state = snapshot_reference_lsm_state()
                try
                    clear_reference_lsm_state!()
                    output = LiquidCortex.run_lsm_step_str(zeros(Float32, 16), 0.0f0)
                    parts = split(output, ",")

                    @test output isa String
                    @test occursin(",", output)
                    @test length(parts) == LiquidCortex.REF_OUT_DEFAULT
                    @test all(part -> !isempty(part), parts)
                finally
                    restore_reference_lsm_state!(original_state)
                end
            end
        else
            @info "Skipping reference LSM GPU tests — no CUDA device"
        end
    end

    # ── GPU tests (only run when CUDA is available) ──────────────────────────

    if LiquidCortex._cuda_available[]
        seed_test_rng!(20260915)

        @testset "GPU: SparseBrain default dims" begin
            reclaim_gpu_hard!()
            brain = SparseBrain(20.0f0; name="test")
            @test brain isa SparseBrain
            @test brain.tau_m == 20.0f0
            @test brain.tick_count == 0
            @test brain.n_in == 14
            @test brain.n_out == 16
            free!(brain); reclaim_gpu!()
        end

        @testset "GPU: SparseBrain custom dims" begin
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="custom",
                cfg=gpu_test_cfg())
            @test brain isa SparseBrain
            @test brain.n_in == 8
            @test brain.n_out == 4
            @test length(brain.output) == 4
            @test size(brain.W_in, 2) == 8
            @test hasproperty(brain, :cfg)
            @test brain.cfg.N == 128
            free!(brain); reclaim_gpu!()
        end

        @testset "GPU: SparseBrain BrainConfig small N + covariance" begin
            cfg = BrainConfig(N=128, hist_depth=8, cov_subsample=32,
                conn_prob=0.05, rng=Random.Xoshiro(7))
            brain = SparseBrain(20.0f0; cfg=cfg, n_in=4, n_out=2, name="cfg-small")
            @test length(brain.V) == 128
            @test size(brain.history) == (8, 128)
            @test brain.cfg.cov_subsample == 32
            u = CUDA.zeros(Float32, 4)
            step!(brain, u; inhibition=0.2f0)
            @test brain.tick_count == 1
            @test all(isfinite, Array(get_output(brain)))
            @test brain.v_thresh_dynamic > brain.cfg.v_thresh
            for _ in 1:8
                step!(brain, u; record_history=true)
            end
            @test brain.hist_full
            result = compute_reservoir_covariance!(brain)
            @test result !== nothing
            C, indices = result
            @test size(C) == (32, 32)
            @test length(indices) == 32
            brain = nothing; reclaim_gpu!()
        end

        @testset "GPU: seeded SparseBrain is reproducible" begin
            make() = SparseBrain(15.0f0; n_in=3, n_out=2, name="seed",
                cfg=BrainConfig(N=64, hist_depth=4, conn_prob=0.08,
                    rng=Random.Xoshiro(1234)))
            a = make()
            b = make()
            @test Array(a.W.nzVal) == Array(b.W.nzVal)
            @test Array(a.W_in) == Array(b.W_in)
            a = nothing; b = nothing; reclaim_gpu!()
        end

        @testset "GPU: EnsembleBrain default dims" begin
            reclaim_gpu_hard!()
            ensemble = EnsembleBrain(; cfg=gpu_test_cfg())
            @test ensemble isa EnsembleBrain
            @test length(ensemble.lobes) == 4
            @test ensemble.lobes[1].n_in == 14
            @test ensemble.lobes[1].n_out == 16
            @test ensemble.lobes[1].cfg.N == 128
            free!(ensemble); reclaim_gpu_hard!()
        end

        @testset "GPU: EnsembleBrain custom dims" begin
            reclaim_gpu_hard!()
            ensemble = EnsembleBrain(; n_in=8, n_out=4, cfg=gpu_test_cfg())
            @test ensemble isa EnsembleBrain
            @test length(ensemble.lobes) == 4
            @test ensemble.lobes[1].n_in == 8
            @test ensemble.lobes[1].n_out == 4
            free!(ensemble); reclaim_gpu_hard!()
        end

        @testset "GPU: EnsembleBrain custom lobe count + small N" begin
            reclaim_gpu_hard!()
            eb = nothing
            try
                cfg = gpu_test_cfg(N=64, hist_depth=4, rng=Random.Xoshiro(9))
                eb = EnsembleBrain(; n_in=4, n_out=2, cfg=cfg,
                    taus=Float32[12.5, 37.5], weights=Float32[0.7, 0.3],
                    names=["A", "B"])
                @test length(eb.lobes) == 2
                @test eb.lobe_names == ["A", "B"]
                @test eb.lobes[1].cfg.N == 64
                @test eb.lobes[1].tau_m == 12.5f0
                @test eb.lobes[2].tau_m == 37.5f0
                u = CUDA.zeros(Float32, 4)
                ensemble_step!(eb, u; inhibition=0.1f0)
                @test length(get_ensemble_output(eb)) == 2
                diag = ensemble_diagnostics(eb)
                @test occursin("[A:τ=12.5]", diag)
                @test occursin("[B:τ=37.5]", diag)
            finally
                eb = nothing
                reclaim_gpu_hard!()
            end
        end

        @testset "GPU: step! with generic inhibition" begin
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="step-test",
                cfg=gpu_test_cfg())
            u = CUDA.zeros(Float32, 8)
            W0 = copy(Array(brain.W_out))
            step!(brain, u; inhibition=0.5f0)
            @test brain.tick_count == 1
            @test brain.v_thresh_dynamic > LiquidCortex.V_THRESH
            reset!(brain)
            @test brain.tick_count == 0
            @test brain.total_spikes == 0
            @test brain.last_spike_rate == 0.0f0
            @test brain.hist_idx == 1
            @test brain.hist_full == false
            @test brain.v_thresh_dynamic == LiquidCortex.V_THRESH
            @test CUDA.minimum(brain.V) == LiquidCortex.V_REST
            @test CUDA.maximum(brain.V) == LiquidCortex.V_REST
            @test iszero(CUDA.maximum(abs, brain.S))
            @test iszero(CUDA.maximum(abs, brain.output))
            @test iszero(CUDA.maximum(abs, brain.trace_pre))
            @test iszero(CUDA.maximum(abs, brain.history))
            @test Array(brain.W_out) == W0
            @test_throws LiquidCortex.LiquidCortexValidationError reset!(brain; keep_weights=false)
            step!(brain, u; inhibition=0.1f0)
            @test brain.tick_count == 1
            free!(brain)
            free!(brain)  # idempotent
            reclaim_gpu!()
        end

        @testset "GPU: ensemble_step! inhibition + plasticity=:none freeze" begin
            # Single 4-lobe construction covers freeze, reset!, desync, and free!
            # (a second EnsembleBrain late in the suite OOMs on 16GB after pool growth).
            seed_test_rng!(20260915)
            reclaim_gpu_hard!()
            ensemble = EnsembleBrain(; n_in=8, n_out=4, cfg=gpu_test_cfg())
            try
                u = CUDA.zeros(Float32, 8)
                ensemble_step!(ensemble, u; inhibition=0.3f0, reflex_signal=0.2f0)
                output = get_ensemble_output(ensemble)
                @test length(output) == 4
                W0 = [copy(Array(l.W_out)) for l in ensemble.lobes]
                u_act = cu(randn(Float32, 8) .* 0.2f0)
                n_steps = 5
                for _ in 1:n_steps
                    ensemble_step!(ensemble, u_act; plasticity=:none, inhibition=0.1f0)
                end
                @test all(l.tick_count == 1 + n_steps for l in ensemble.lobes)
                @test all(Array(ensemble.lobes[i].W_out) == W0[i] for i in eachindex(ensemble.lobes))

                reset!(ensemble)
                @test all(l.tick_count == 0 for l in ensemble.lobes)
                @test all(l.hist_idx == 1 && l.hist_full == false for l in ensemble.lobes)
                @test iszero(CUDA.maximum(abs, ensemble.agg_output))
                @test all(Array(ensemble.lobes[i].W_out) == W0[i] for i in eachindex(ensemble.lobes))
                @test ensemble.desynchronized == false

                ensemble_step!(ensemble, u; plasticity=:none, inhibition=0.1f0)
                saved_agg = copy(Array(ensemble.agg_output))
                saved_weights = copy(ensemble.weights)
                @test ensemble.desynchronized == false
                ensemble.weights = saved_weights[1:2]
                @test_throws BoundsError LiquidCortex._commit_ensemble_aggregate!(ensemble)
                @test Array(ensemble.agg_output) == saved_agg
                @test ensemble.desynchronized == false
                ensemble.weights = saved_weights

                ensemble.lobes[3].tick_count += 1
                @test_throws LiquidCortex.EnsembleDesynchronizedError (
                    ensemble_step!(ensemble, u_act; plasticity=:none)
                )
                @test ensemble.desynchronized == false
                @test_throws LiquidCortex.EnsembleDesynchronizedError get_ensemble_output(ensemble)
                @test Array(ensemble.agg_output) == saved_agg
                @test ensemble.lobes[3].tick_count == ensemble.lobes[1].tick_count + 1
                diag_mismatch = ensemble_diagnostics(ensemble)
                @test startswith(diag_mismatch, "[DESYNC]")
                @test occursin("W=n/a", diag_mismatch)

                ensemble.lobes[3].tick_count -= 1
                ensemble.weights = saved_weights[1:2]
                ticks_before_poison = [l.tick_count for l in ensemble.lobes]
                @test_throws BoundsError ensemble_step!(ensemble, u_act; plasticity=:none)
                @test ensemble.desynchronized
                @test all(l.tick_count == ticks_before_poison[i] for (i, l) in enumerate(ensemble.lobes))
                @test Array(ensemble.agg_output) == saved_agg
                ensemble.weights = saved_weights
                @test_throws LiquidCortex.EnsembleDesynchronizedError (
                    ensemble_step!(ensemble, u_act; plasticity=:none)
                )
                @test_throws LiquidCortex.EnsembleDesynchronizedError get_ensemble_output(ensemble)
                diag_poison = ensemble_diagnostics(ensemble)
                @test startswith(diag_poison, "[DESYNC]")
                @test occursin("W=n/a", diag_poison)

                # Same 4-lobe construction: a second EnsembleBrain late in the
                # suite OOMs on 16GB after pool growth (#70).
                reset!(ensemble)
                @test all(l.tick_count == 0 for l in ensemble.lobes)
                @test all(l.hist_idx == 1 && l.hist_full == false for l in ensemble.lobes)
                @test iszero(CUDA.maximum(abs, ensemble.agg_output))
                @test all(Array(ensemble.lobes[i].W_out) == W0[i] for i in eachindex(ensemble.lobes))
            finally
                free!(ensemble)
                free!(ensemble)  # idempotent
                reclaim_gpu_hard!()
            end
        end

        @testset "GPU: failed step! does not commit tick or history" begin
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="tdd-clock-last",
                cfg=gpu_test_cfg())
            u = CUDA.zeros(Float32, 8)
            tick0 = brain.tick_count
            hist0 = brain.hist_idx
            spikes0 = brain.total_spikes
            rate0 = brain.last_spike_rate
            CUDA.unsafe_free!(brain.W_out)
            threw = false
            try
                step!(brain, u)
            catch
                threw = true
            end
            @test threw
            @test brain.tick_count == tick0
            @test brain.hist_idx == hist0
            @test brain.total_spikes == spikes0
            @test brain.last_spike_rate == rate0
            free!(brain); reclaim_gpu!()
        end

        @testset "GPU: default step! advances tick and keeps finite output" begin
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="tdd-default",
                cfg=gpu_test_cfg())
            u = CUDA.zeros(Float32, 8)
            step!(brain, u; inhibition=0.1f0)
            @test brain.tick_count == 1
            @test all(isfinite, Array(get_output(brain)))
            free!(brain); reclaim_gpu!()
        end

        @testset "GPU: plasticity=:none freezes W_out" begin
            seed_test_rng!(20260915)
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="tdd-none",
                cfg=gpu_test_cfg())
            u = cu(randn(Float32, 8) .* 0.2f0)
            W0 = copy(Array(brain.W_out))
            for _ in 1:40
                step!(brain, u; plasticity=:none, inhibition=0.1f0)
            end
            @test Array(brain.W_out) == W0
            @test brain.tick_count == 40
            @test_throws LiquidCortex.LiquidCortexValidationError step!(brain, u; plasticity=:typo)
            @test_throws LiquidCortex.LiquidCortexValidationError step!(
                brain, u; plasticity=:recurrent_stdp, recurrent_eta=NaN32)
            free!(brain); reclaim_gpu!()
        end

        @testset "GPU: plasticity=:readout_only updates W_out in Hebbian direction" begin
            seed_test_rng!(20260915)
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="tdd-ro",
                cfg=gpu_test_cfg())
            u = cu(randn(Float32, 8) .* 0.3f0)
            W0 = copy(Array(brain.W_out))
            for _ in 1:50
                step!(brain, u; plasticity=:readout_only, inhibition=0.05f0,
                      reflex_eta=1f-2)
            end
            @test brain.tick_count == 50
            @test all(isfinite, Array(get_output(brain)))
            W1 = Array(brain.W_out)
            @test !all(W1 .== W0)
            # Host kernel on the last traces/output must match the same sign rule.
            W_dir = copy(W0)
            LiquidCortex._readout_hebbian_update!(
                W_dir, Float32[1.0, 1.0, 1.0, 1.0], ones(Float32, size(W0, 2)),
                0.01f0, LiquidCortex.W_MAX)
            @test all(W_dir .>= W0)
            free!(brain); reclaim_gpu!()
        end

        @testset "GPU: record_history=false steps without filling history" begin
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="tdd-hist",
                cfg=gpu_test_cfg())
            u = CUDA.zeros(Float32, 8)
            step!(brain, u; record_history=false)
            @test brain.tick_count == 1
            @test brain.hist_full == false
            @test brain.hist_idx == 1
            free!(brain); reclaim_gpu!()
        end

        @testset "GPU: sync=false advances tick (caller may sync)" begin
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="tdd-sync",
                cfg=gpu_test_cfg())
            u = CUDA.zeros(Float32, 8)
            step!(brain, u; sync=false)
            CUDA.synchronize()
            @test brain.tick_count == 1
            @test all(isfinite, Array(get_output(brain)))
            free!(brain); reclaim_gpu!()
        end

        @testset "GPU: use_device_noise=true stays finite" begin
            seed_test_rng!(20260915)
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="tdd-noise",
                cfg=gpu_test_cfg())
            u = CUDA.zeros(Float32, 8)
            for _ in 1:20
                step!(brain, u; use_device_noise=true, record_history=false)
            end
            @test brain.tick_count == 20
            @test all(isfinite, Array(get_output(brain)))
            free!(brain); reclaim_gpu!()
        end

        @testset "GPU: recurrent_stdp mutates sparse W.nzVal" begin
            seed_test_rng!(20260915)
            reclaim_gpu_hard!()
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="tdd-stdp",
                cfg=gpu_test_cfg(conn_prob=0.08))
            u = cu(randn(Float32, 8) .* 0.35f0)
            # eta=0 must not clamp/rewrite constructor weights before learning
            w_init = copy(Array(brain.W.nzVal))
            for _ in 1:5
                step!(brain, u; plasticity=:recurrent_stdp, recurrent_eta=0.0f0,
                      record_history=false)
            end
            @test Array(brain.W.nzVal) == w_init
            w0 = copy(Array(brain.W.nzVal))
            for _ in 1:30
                step!(brain, u; plasticity=:recurrent_stdp, recurrent_eta=1f-3,
                      record_history=false)
            end
            @test Array(brain.W.nzVal) != w0
            @test all(isfinite, Array(get_output(brain)))
            # Drop lazy STDP edge buffers before reclaim
            brain.pre_idx = CUDA.zeros(Int32, 0)
            brain.post_idx = CUDA.zeros(Int32, 0)
            free!(brain); reclaim_gpu_hard!()
        end

        @testset "GPU: pair STDP LTP/LTD on a known edge" begin
            seed_test_rng!(20260915)
            reclaim_gpu_hard!()
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="tdd-stdp-dir")
            LiquidCortex._ensure_edge_indices!(brain)
            @test brain.nnz > 0
            pre = Array(brain.pre_idx)
            post = Array(brain.post_idx)
            nz = Array(brain.W.nzVal)
            edge = findfirst(i -> abs(Float32(nz[i])) < 0.5f0, eachindex(pre))
            edge === nothing && (edge = 1)
            pre_n = Int(pre[edge])
            post_n = Int(post[edge])

            # LTP: pre-trace co-occurs with post spike.
            S = zeros(Float32, LiquidCortex.N)
            S[post_n] = 1.0f0
            trace_pre = zeros(Float32, LiquidCortex.N)
            trace_pre[pre_n] = 1.0f0
            copyto!(brain.S, S)
            copyto!(brain.trace_pre, trace_pre)
            fill!(brain.trace_post, 0.0f0)
            w_before = Array(brain.W.nzVal)[edge]
            LiquidCortex._apply_pair_stdp!(brain; eta=0.1f0)
            w_ltp = Array(brain.W.nzVal)[edge]
            @test w_ltp > w_before

            # LTD: pre spike co-occurs with post-trace.
            S .= 0
            S[pre_n] = 1.0f0
            trace_post = zeros(Float32, LiquidCortex.N)
            trace_post[post_n] = 1.0f0
            copyto!(brain.S, S)
            fill!(brain.trace_pre, 0.0f0)
            copyto!(brain.trace_post, trace_post)
            LiquidCortex._apply_pair_stdp!(brain; eta=0.1f0)
            w_ltd = Array(brain.W.nzVal)[edge]
            @test w_ltd < w_ltp

            brain.pre_idx = CUDA.zeros(Int32, 0)
            brain.post_idx = CUDA.zeros(Int32, 0)
            free!(brain); reclaim_gpu_hard!()
        end
    else
        @info "Skipping GPU tests — no CUDA device available"
    end

end
