# SPDX-License-Identifier: MIT OR Apache-2.0

"""
    LiquidCortex

GPU-accelerated sparse Liquid State Machine for neuromorphic computing.

Provides two LSM implementations:
- **EnsembleBrain** (`sparse_brain.jl`) — 4-lobe, 65,536-neuron/lobe sparse CUDA LSM
  with OU-SDE dynamics, STDP covariance learning, and rolling 1,000-tick spike history.
  Configurable input/output dimensions. Requires RTX-class GPU with ≥14 GB VRAM.

- **Reference LSM** (`reference_lsm.jl`) — 2,048-neuron dense CUDA reservoir for
  rapid prototyping. Configurable input/output dimensions.

On CPU-only systems, the module loads cleanly — types and API functions are defined
but GPU allocations are deferred until a CUDA device is available at runtime.
Check `LiquidCortex._cuda_available[]` to test CUDA availability.
"""
module LiquidCortex

using CUDA
using PrecompileTools
using Sentry

# ── CUDA availability flag ────────────────────────────────────────────────
# Checked at __init__ time. All GPU allocations are deferred until this is true.
const _cuda_available = Ref{Bool}(false)
const _sentry_enabled = Ref{Bool}(false)

function __init__()
    _cuda_available[] = false
    _sentry_enabled[] = false

    if CUDA.functional()
        _cuda_available[] = true
        @info "LiquidCortex: CUDA functional — GPU kernels available on $(CUDA.name(CUDA.device()))."
    else
        @warn "LiquidCortex: No CUDA-capable GPU found. " *
              "Core types will load, but step! and GPU operations require a CUDA device."
    end

    # Initialize Sentry error capture if DSN is configured
    dsn = get(ENV, "SENTRY_DSN", "")
    if !isempty(dsn)
        try
            pv = Base.pkgversion(@__MODULE__)
            version = pv === nothing ? "unknown" : string(pv)
            Sentry.init(dsn; release="LiquidCortex.jl@$version")
            Sentry.set_tag("package", "LiquidCortex.jl")
            Sentry.set_tag("version", version)
            Sentry.set_tag("julia_version", string(VERSION))
            _sentry_enabled[] = true
            @info "LiquidCortex: Sentry error capture enabled."
        catch e
            @warn "LiquidCortex: Failed to initialize Sentry" exception=(e, catch_backtrace())
        end
    end
end

# Caller-facing validation (wrong kwargs / input size). Filtered from Sentry so
# unit tests and API misuse do not create issues. Internal faults (malformed
# CSC, CUDA OOM, unexpected ArgumentError from libs) remain captured.
struct LiquidCortexValidationError <: Exception
    msg::String
end
Base.showerror(io::IO, e::LiquidCortexValidationError) =
    print(io, "LiquidCortexValidationError: ", e.msg)

# Accept any thrown value (Julia allows non-Exception throws) so capture never
# raises MethodError and masks the original failure path.
#
# Never block the rethrow path: Sentry.jl enqueues via a bounded Channel(100),
# so a full backlog (network outage / burst of failures) would hang step! /
# ensemble_step! before rethrow if capture were synchronous. Schedule capture
# asynchronously and only wait briefly; drop waiting (and leave the task
# running best-effort) if the queue is blocked.
@noinline function _should_capture_runtime_exception(@nospecialize(exc))
    return !(exc isa LiquidCortexValidationError)
end

# Sentry.jl tags are process-global. Serialize tag+capture so concurrent async
# captures cannot cross-label each other's events.
const _sentry_capture_lock = ReentrantLock()

@noinline function _tag_runtime_exception!(@nospecialize(exc))
    if exc isa CUDA.OutOfGPUMemoryError
        Sentry.set_tag("gpu_failure", "true")
        Sentry.set_tag("error_class", "gpu_oom")
    elseif exc isa CUDA.CuError
        Sentry.set_tag("gpu_failure", "true")
        Sentry.set_tag("error_class", "cuda_error")
    else
        Sentry.set_tag("gpu_failure", "false")
        Sentry.set_tag("error_class", "runtime")
    end
    return nothing
end

@noinline function _capture_runtime_exception(@nospecialize(exc), bt)
    _sentry_enabled[] || return nothing
    _should_capture_runtime_exception(exc) || return nothing
    try
        t = @async begin
            try
                lock(_sentry_capture_lock) do
                    _tag_runtime_exception!(exc)
                    Sentry.capture_exception([(exc, bt)])
                end
            catch sentry_error
                @warn "LiquidCortex: Failed to capture exception in Sentry" exception=(sentry_error, catch_backtrace())
            end
        end
        # Best-effort window for format+enqueue; never hang rethrow on a full queue.
        timedwait(() -> istaskdone(t), 0.05; pollint=0.005)
    catch
        # Drop capture entirely if scheduling/wait itself fails.
    end
    return nothing
end

# ── GPU source files (structs defined at load; GPU allocations deferred to
#    constructors/runtime, guarded by _cuda_available[]) ─────────────────────
include("sparse_brain.jl")
include("reference_lsm.jl")

# ── Public API ───────────────────────────────────────────────────────────────

export SparseBrain, EnsembleBrain
export step!, ensemble_step!, get_output, get_ensemble_output
export compute_reservoir_covariance!, diagnostics, ensemble_diagnostics

# Precompile the SparseBrain hot path at install time. `__init__` has not
# run yet, so `_cuda_available[]` is still false — probe the device directly.
# CUDA kernels cannot be recorded without a visible GPU; CPU-only caches
# stay empty by design. Skip EnsembleBrain (four 65k-neuron lobes) and the
# 2,048-neuron reference LSM so precompile cannot OOM smaller cards.
@compile_workload begin
    if CUDA.functional()
        try
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="precompile")
            u = CUDA.zeros(Float32, 8)
            step!(brain, u; inhibition=0.5f0)
            get_output(brain)
        catch
            # Best-effort: constructor/step failures must not abort install.
        finally
            # Drop the warmup lobe and CUDA pool so Pkg.test() / later
            # EnsembleBrain construction is not starved on 16 GB cards.
            try
                GC.gc(true)
                CUDA.reclaim()
                CUDA.device_reset!()
            catch
            end
        end
    end
end

end # module LiquidCortex
