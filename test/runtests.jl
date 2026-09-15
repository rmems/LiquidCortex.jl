# SPDX-License-Identifier: MIT OR Apache-2.0
#
using Test
using LiquidCortex
using CUDA
using Sentry
using LinearAlgebra: norm

# Free VRAM between heavy GPU cases (65k lobes / ensembles leave large pools).
function reclaim_gpu!()
    CUDA.synchronize()
    GC.gc(true)
    CUDA.reclaim()
    GC.gc(true)
    CUDA.reclaim()
    return nothing
end

# Full device reset — required between EnsembleBrain cases on 16GB cards.
# Soft reclaim leaves the memory pool reserved (~2GB leak per ensemble in CI).
function reclaim_gpu_hard!()
    reclaim_gpu!()
    try
        CUDA.device_reset!()
    catch
    end
    return nothing
end

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

@testset "LiquidCortex" begin

    @testset "Package loads" begin
        @test @isdefined(LiquidCortex)
        @test LiquidCortex isa Module
    end

    @testset "Public API is exported" begin
        # Verify each name is both defined AND exported (not just defined)
        exports = names(LiquidCortex; all=false)
        for sym in [:SparseBrain, :EnsembleBrain, :EnsembleDesynchronizedError,
                    :step!, :ensemble_step!, :get_output, :get_ensemble_output,
                    :compute_reservoir_covariance!, :diagnostics, :ensemble_diagnostics,
                    :enable_telemetry!]
            @test sym in exports
        end
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

        err = try
            SparseBrain(0.0f0)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("tau_m", err.msg)
        @test LiquidCortex._should_capture_runtime_exception(err) == true
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

    @testset "CPU: ensemble clock desync detection" begin
        # Production change that would fail this: dropping the mismatch helper
        # or treating a distinct lobe tick as synchronized.
        @test LiquidCortex._ensemble_tick_mismatch(Int64[]) === nothing
        @test LiquidCortex._ensemble_tick_mismatch(Int64[4, 4, 4, 4]) === nothing
        @test LiquidCortex._ensemble_tick_mismatch(Int64[4, 4, 5, 4]) == (Int64(4), 3, Int64(5))
        LiquidCortex._assert_ticks_synchronized(Int64[0, 0, 0, 0])
        @test_throws LiquidCortex.EnsembleDesynchronizedError (
            LiquidCortex._assert_ticks_synchronized(Int64[1, 1, 0, 1])
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

            @testset "Dimension validation" begin
                original_state = snapshot_reference_lsm_state()
                try
                    clear_reference_lsm_state!()
                    @test_throws ArgumentError LiquidCortex._init_ref_lsm!(; n_in=0, n_out=4)
                    @test_throws ArgumentError LiquidCortex._init_ref_lsm!(; n_in=4, n_out=0)

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
            @info "Skipping reference LSM tests — no CUDA device"
            @test_skip "Reference LSM GPU tests skipped"
        end
    end

    # ── GPU tests (only run when CUDA is available) ──────────────────────────

    if LiquidCortex._cuda_available[]
        @testset "GPU: SparseBrain default dims" begin
            reclaim_gpu_hard!()
            brain = SparseBrain(20.0f0; name="test")
            @test brain isa SparseBrain
            @test brain.tau_m == 20.0f0
            @test brain.tick_count == 0
            @test brain.n_in == 14
            @test brain.n_out == 16
            brain = nothing; reclaim_gpu!()
        end

        @testset "GPU: SparseBrain custom dims" begin
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="custom")
            @test brain isa SparseBrain
            @test brain.n_in == 8
            @test brain.n_out == 4
            @test length(brain.output) == 4
            @test size(brain.W_in, 2) == 8
            brain = nothing; reclaim_gpu!()
        end

        @testset "GPU: EnsembleBrain default dims" begin
            reclaim_gpu_hard!()
            ensemble = EnsembleBrain()
            @test ensemble isa EnsembleBrain
            @test length(ensemble.lobes) == 4
            @test ensemble.lobes[1].n_in == 14
            @test ensemble.lobes[1].n_out == 16
            ensemble = nothing; reclaim_gpu_hard!()
        end

        @testset "GPU: EnsembleBrain custom dims" begin
            reclaim_gpu_hard!()
            ensemble = EnsembleBrain(n_in=8, n_out=4)
            @test ensemble isa EnsembleBrain
            @test length(ensemble.lobes) == 4
            @test ensemble.lobes[1].n_in == 8
            @test ensemble.lobes[1].n_out == 4
            ensemble = nothing; reclaim_gpu_hard!()
        end

        @testset "GPU: step! with generic inhibition" begin
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="step-test")
            u = CUDA.zeros(Float32, 8)
            step!(brain, u; inhibition=0.5f0)
            @test brain.tick_count == 1
            @test brain.v_thresh_dynamic > LiquidCortex.V_THRESH
            brain = nothing; reclaim_gpu!()
        end

        @testset "GPU: ensemble_step! inhibition + plasticity=:none freeze" begin
            # Single 4-lobe construction covers both inhibition path and :none freeze
            # (a second EnsembleBrain late in the suite OOMs on 16GB after pool growth).
            reclaim_gpu_hard!()
            ensemble = EnsembleBrain(n_in=8, n_out=4)
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

                saved_agg = copy(Array(ensemble.agg_output))
                saved_weights = copy(ensemble.weights)
                ensemble.weights = saved_weights[1:2]
                @test_throws BoundsError LiquidCortex._commit_ensemble_aggregate!(ensemble)
                @test Array(ensemble.agg_output) == saved_agg
                ensemble.weights = saved_weights

                ensemble.lobes[3].tick_count += 1
                @test_throws LiquidCortex.EnsembleDesynchronizedError (
                    ensemble_step!(ensemble, u_act; plasticity=:none)
                )
                @test_throws LiquidCortex.EnsembleDesynchronizedError get_ensemble_output(ensemble)
                @test Array(ensemble.agg_output) == saved_agg
                @test ensemble.lobes[3].tick_count == ensemble.lobes[1].tick_count + 1
            finally
                ensemble = nothing
                reclaim_gpu_hard!()
            end
        end

        @testset "GPU: failed step! does not commit tick or history" begin
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="tdd-clock-last")
            u = CUDA.zeros(Float32, 8)
            tick0 = brain.tick_count
            hist0 = brain.hist_idx
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
            brain = nothing; reclaim_gpu!()
        end

        @testset "GPU: default step! advances tick and keeps finite output" begin
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="tdd-default")
            u = CUDA.zeros(Float32, 8)
            step!(brain, u; inhibition=0.1f0)
            @test brain.tick_count == 1
            @test all(isfinite, Array(get_output(brain)))
            brain = nothing; reclaim_gpu!()
        end

        @testset "GPU: plasticity=:none freezes W_out" begin
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="tdd-none")
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
            brain = nothing; reclaim_gpu!()
        end

        @testset "GPU: plasticity=:readout_only can update W_out" begin
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="tdd-ro")
            u = cu(randn(Float32, 8) .* 0.3f0)
            W0 = copy(Array(brain.W_out))
            for _ in 1:50
                step!(brain, u; plasticity=:readout_only, inhibition=0.05f0,
                      reflex_eta=1f-2)
            end
            @test brain.tick_count == 50
            @test all(isfinite, Array(get_output(brain)))
            @test !all(Array(brain.W_out) .== W0)
            brain = nothing; reclaim_gpu!()
        end

        @testset "GPU: record_history=false steps without filling history" begin
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="tdd-hist")
            u = CUDA.zeros(Float32, 8)
            step!(brain, u; record_history=false)
            @test brain.tick_count == 1
            @test brain.hist_full == false
            @test brain.hist_idx == 1
            brain = nothing; reclaim_gpu!()
        end

        @testset "GPU: sync=false advances tick (caller may sync)" begin
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="tdd-sync")
            u = CUDA.zeros(Float32, 8)
            step!(brain, u; sync=false)
            CUDA.synchronize()
            @test brain.tick_count == 1
            @test all(isfinite, Array(get_output(brain)))
            brain = nothing; reclaim_gpu!()
        end

        @testset "GPU: use_device_noise=true stays finite" begin
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="tdd-noise")
            u = CUDA.zeros(Float32, 8)
            for _ in 1:20
                step!(brain, u; use_device_noise=true, record_history=false)
            end
            @test brain.tick_count == 20
            @test all(isfinite, Array(get_output(brain)))
            brain = nothing; reclaim_gpu!()
        end

        @testset "GPU: recurrent_stdp mutates sparse W.nzVal" begin
            reclaim_gpu_hard!()
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="tdd-stdp")
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
            brain = nothing; reclaim_gpu_hard!()
        end
    else
        @info "Skipping GPU tests — no CUDA device available"
        @test_skip "GPU tests skipped (no CUDA)"
    end

end
