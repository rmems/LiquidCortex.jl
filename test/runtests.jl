# SPDX-License-Identifier: MIT OR Apache-2.0
#
using Test
using LiquidCortex
using CUDA
using LinearAlgebra: norm
using Random

# Free VRAM between heavy GPU cases (65k lobes / ensembles leave large pools).
function reclaim_gpu!()
    CUDA.synchronize()
    GC.gc(true)
    CUDA.reclaim()
    GC.gc(true)
    CUDA.reclaim()
    return nothing
end

# CUDA.jl 6 made `device_reset!` a documented no-op (depwarn only). GPU
# cases call `free!` so the ~0.5–2 GB per reservoir returns to the pool
# instead of waiting on CuArray finalizers. This helper then syncs and
# reclaims; the old hard reset is gone.
function reclaim_gpu_hard!()
    reclaim_gpu!()
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
        for sym in [:SparseBrain, :EnsembleBrain, :EnsembleDesynchronizedError,
                    :step!, :ensemble_step!, :get_output, :get_ensemble_output,
                    :compute_reservoir_covariance, :compute_reservoir_covariance!,
                    :spikes, :membrane, :traces,
                    :diagnostics, :ensemble_diagnostics,
                    :reset!, :free!,
                    :LiquidCortexValidationError, :ETA, :MAX_INHIBITION,
                    :run_lsm_step, :run_lsm_step_str,
                    :REF_N, :REF_IN_DEFAULT, :REF_OUT_DEFAULT]
            @test sym in exports
        end
        # Documented names resolve unqualified after `using LiquidCortex`.
        @test LiquidCortexValidationError isa DataType
        @test ETA == 0.001f0
        @test MAX_INHIBITION == 3.0f0
        @test REF_N == 2048
        @test run_lsm_step isa Function
        @test compute_reservoir_covariance! === compute_reservoir_covariance
        @test parentmodule(step!) === LiquidCortex.CommonSolve
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
        @test_throws LiquidCortexValidationError (
            LiquidCortex._validate_plasticity_kwargs(; plasticity=:typo, recurrent_eta=1f-4)
        )
        @test_throws LiquidCortexValidationError (
            LiquidCortex._validate_plasticity_kwargs(; plasticity=:recurrent_stdp, recurrent_eta=NaN32)
        )
        LiquidCortex._validate_plasticity_kwargs(; plasticity=:none, recurrent_eta=NaN32)
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
        @test_throws ArgumentError SparseBrain(0.0)   # Float64
        @test_throws ArgumentError SparseBrain(0)     # Int
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

        # No leaked all-fields positional constructor (would dominate MethodError).
        ctor_nargs = map(methods(SparseBrain)) do m
            sig = m.sig
            while sig isa UnionAll
                sig = sig.body
            end
            length(sig.parameters)
        end
        @test nfields(SparseBrain) + 1 ∉ ctor_nargs
        err_partial = try
            SparseBrain(Val(:new))
        catch e
            e
        end
        @test err_partial isa ArgumentError
        @test occursin("fields", err_partial.msg)
        err_short = try
            SparseBrain(Val(:new), 1, 2, 3)
        catch e
            e
        end
        @test err_short isa ArgumentError
        @test occursin("fields", err_short.msg)
    end

    @testset "CPU: _require_cuda messaging" begin
        if LiquidCortex._cuda_available[]
            # GPU runner: device is present; assert the under-VRAM error path.
            err = try
                LiquidCortex._require_cuda("SparseBrain"; min_vram_gb=1e12)
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("SparseBrain", err.msg)
            @test occursin("GB", err.msg)
        else
            err = try
                LiquidCortex._require_cuda("SparseBrain"; min_vram_gb=14)
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("SparseBrain", err.msg)
            @test occursin("14", err.msg)
        end
    end

    @testset "CPU: CUDA guard fails fast (no COO draw)" begin
        if LiquidCortex._cuda_available[]
            @test_skip "CUDA present — constructor GPU-guard timing covered on CPU CI"
        else
            elapsed = @elapsed begin
                err = try
                    SparseBrain(20.0)
                catch e
                    e
                end
                @test err isa ErrorException
                @test occursin("CUDA", err.msg)
                @test occursin("14 GB", err.msg)
                @test occursin("SparseBrain", err.msg)
            end
            @test elapsed < 2.0

            err_int = try
                SparseBrain(20)
            catch e
                e
            end
            @test err_int isa ErrorException
            @test occursin("CUDA", err_int.msg)

            err_ens = try
                EnsembleBrain()
            catch e
                e
            end
            @test err_ens isa ErrorException
            @test occursin("CUDA", err_ens.msg)

            err_ref = try
                run_lsm_step(zeros(Float32, 16), 0.0f0)
            catch e
                e
            end
            @test err_ref isa ErrorException
            @test occursin("CUDA", err_ref.msg)
            @test occursin("Reference LSM", err_ref.msg)
        end
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
        diag_named = LiquidCortex._format_diagnostics(
            1, 10, 0.0123f0, -50.0f0, 1.2345; name="lobe-a")
        @test occursin("[brain:lobe-a]", diag_named)
        @test LiquidCortex._device_memory_gb(15_000_000_000) ≈ 15.0 rtol=0.01
        @test LiquidCortex._vram_meets_floor(15_000_000_000, 14)
        @test !LiquidCortex._vram_meets_floor(10_000_000_000, 14)
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

        if CUDA.functional()
            dest = CUDA.zeros(Float32, 4)
            @test_throws LiquidCortex.LiquidCortexValidationError (
                LiquidCortex._upload_input!(dest, Float32[1, 2, 3])
            )
        end
    end

    @testset "CPU: lifecycle frees host step! input buffer" begin
        lifecycle_src = read(
            joinpath(@__DIR__, "..", "src", "brain_lifecycle.jl"), String)
        @test occursin("CUDA.unsafe_free!(brain.u_buf)", lifecycle_src)
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
        poisoned = try
            LiquidCortex._assert_ensemble_clocks(Int64[4, 4, 4, 4]; desynchronized=true)
        catch e
            e
        end
        @test poisoned isa LiquidCortex.EnsembleDesynchronizedError
        @test occursin("unusable", poisoned.msg)
        @test LiquidCortex._ensemble_diag_desync(false, Int64[1, 1, 1, 1]) == false
        @test LiquidCortex._ensemble_diag_desync(true, Int64[1, 1, 1, 1]) == true
        @test LiquidCortex._ensemble_diag_desync(false, Int64[1, 1, 2, 1]) == true
    end

    @testset "CPU: Reference LSM dimension validation" begin
        # Guards run before GPU allocation, so these are CPU-safe.
        @test_throws ArgumentError LiquidCortex._init_ref_lsm!(; n_in=0, n_out=4)
        @test_throws ArgumentError LiquidCortex._init_ref_lsm!(; n_in=4, n_out=0)
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
                    output = run_lsm_step(input, 0.5f0)
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
                    output = run_lsm_step(input, 0.0f0; n_out=4)
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
                    @test_throws DimensionMismatch run_lsm_step(
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
                    output = run_lsm_step_str(zeros(Float32, 16), 0.0f0)
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

        @testset "GPU: _require_cuda VRAM floor" begin
            @test LiquidCortex._require_cuda("Probe"; min_vram_gb=1) === nothing
            err_vram = try
                LiquidCortex._require_cuda("Probe"; min_vram_gb=1e12)
            catch e
                e
            end
            @test err_vram isa ErrorException
            @test occursin("GB", err_vram.msg)
        end

        @testset "GPU: SparseBrain default dims" begin
            reclaim_gpu_hard!()
            brain = SparseBrain(20.0f0; name="test")
            @test brain isa SparseBrain
            @test brain.tau_m == 20.0f0
            @test brain.tick_count == 0
            @test brain.n_in == 14
            @test brain.n_out == 16
            @test brain.name == "test"
            shown = sprint(show, brain)
            @test occursin("SparseBrain", shown)
            @test occursin("test", shown)
            @test length(shown) < 400
            free!(brain); reclaim_gpu!()
        end

        @testset "GPU: SparseBrain Real tau_m + host-vector step!" begin
            brain = SparseBrain(20.0; n_in=8, n_out=4, name="host-u")
            @test brain.tau_m == 20.0f0
            @test brain.name == "host-u"
            step!(brain, zeros(Float32, 8); inhibition=0.5f0)
            @test brain.tick_count == 1
            @test brain.v_thresh_dynamic > LiquidCortex.V_THRESH
            step!(brain, zeros(Float64, 8))
            @test brain.tick_count == 2
            @test_throws LiquidCortexValidationError step!(brain, zeros(Float32, 3))
            @test length(spikes(brain)) == LiquidCortex.N
            @test length(membrane(brain)) == LiquidCortex.N
            pre, post = traces(brain)
            @test length(pre) == LiquidCortex.N
            @test length(post) == LiquidCortex.N
            @test occursin("host-u", diagnostics(brain))
            @test_throws LiquidCortexValidationError compute_reservoir_covariance(brain)
            brain = nothing; reclaim_gpu!()
            brain2 = SparseBrain(20; n_in=8, n_out=4, name="int-tau")
            @test brain2.tau_m == 20.0f0
            brain2 = nothing; reclaim_gpu!()
        end

        @testset "GPU: SparseBrain custom dims" begin
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="custom")
            @test brain isa SparseBrain
            @test brain.n_in == 8
            @test brain.n_out == 4
            @test length(brain.output) == 4
            @test size(brain.W_in, 2) == 8
            free!(brain); reclaim_gpu!()
        end

        @testset "GPU: EnsembleBrain default dims" begin
            reclaim_gpu_hard!()
            ensemble = EnsembleBrain()
            @test ensemble isa EnsembleBrain
            @test length(ensemble.lobes) == 4
            @test ensemble.lobes[1].n_in == 14
            @test ensemble.lobes[1].n_out == 16
            free!(ensemble); reclaim_gpu_hard!()
        end

        @testset "GPU: EnsembleBrain custom dims" begin
            reclaim_gpu_hard!()
            ensemble = EnsembleBrain(n_in=8, n_out=4)
            @test ensemble isa EnsembleBrain
            @test length(ensemble.lobes) == 4
            @test ensemble.lobes[1].n_in == 8
            @test ensemble.lobes[1].n_out == 4
            shown = sprint(show, ensemble)
            @test occursin("EnsembleBrain", shown)
            @test length(shown) < 400
            ensemble_step!(ensemble, zeros(Float32, 8); inhibition=0.1f0)
            @test ensemble.lobes[1].tick_count == 1
            free!(ensemble); reclaim_gpu_hard!()
        end

        @testset "GPU: step! with generic inhibition" begin
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="step-test")
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
                reset!(ensemble)
                @test all(l.tick_count == 0 for l in ensemble.lobes)
                @test all(l.hist_idx == 1 && l.hist_full == false for l in ensemble.lobes)
                @test iszero(CUDA.maximum(abs, ensemble.agg_output))
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
                ensemble = nothing
                reclaim_gpu_hard!()
            end
        end

        @testset "GPU: failed step! does not commit tick or history" begin
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="tdd-clock-last")
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
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="tdd-default")
            u = CUDA.zeros(Float32, 8)
            step!(brain, u; inhibition=0.1f0)
            @test brain.tick_count == 1
            @test all(isfinite, Array(get_output(brain)))
            free!(brain); reclaim_gpu!()
        end

        @testset "GPU: plasticity=:none freezes W_out" begin
            seed_test_rng!(20260915)
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="tdd-none")
            u = cu(randn(Float32, 8) .* 0.2f0)
            W0 = copy(Array(brain.W_out))
            for _ in 1:40
                step!(brain, u; plasticity=:none, inhibition=0.1f0)
            end
            @test Array(brain.W_out) == W0
            @test brain.tick_count == 40
            @test_throws LiquidCortexValidationError step!(brain, u; plasticity=:typo)
            @test_throws LiquidCortexValidationError step!(
                brain, u; plasticity=:recurrent_stdp, recurrent_eta=NaN32)
            free!(brain); reclaim_gpu!()
        end

        @testset "GPU: plasticity=:readout_only updates W_out in Hebbian direction" begin
            seed_test_rng!(20260915)
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="tdd-ro")
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
                W_dir, Float32[1.0, 1.0, 1.0, 1.0], ones(Float32, LiquidCortex.N),
                0.01f0, LiquidCortex.W_MAX)
            @test all(W_dir .>= W0)
            free!(brain); reclaim_gpu!()
        end

        @testset "GPU: record_history=false steps without filling history" begin
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="tdd-hist")
            u = CUDA.zeros(Float32, 8)
            step!(brain, u; record_history=false)
            @test brain.tick_count == 1
            @test brain.hist_full == false
            @test brain.hist_idx == 1
            free!(brain); reclaim_gpu!()
        end

        @testset "GPU: sync=false advances tick (caller may sync)" begin
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="tdd-sync")
            u = CUDA.zeros(Float32, 8)
            step!(brain, u; sync=false)
            CUDA.synchronize()
            @test brain.tick_count == 1
            @test all(isfinite, Array(get_output(brain)))
            free!(brain); reclaim_gpu!()
        end

        @testset "GPU: use_device_noise=true stays finite" begin
            seed_test_rng!(20260915)
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="tdd-noise")
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
