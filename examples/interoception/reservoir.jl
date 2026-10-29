# SPDX-License-Identifier: MIT OR Apache-2.0
#
# LiquidCortex adapter: replay z-scored telemetry through a SparseBrain or
# EnsembleBrain and collect readout features. All GPU work lives here; nothing in
# the core package knows about telemetry.
#
# Readout features per lobe: membrane potential `V` (carries the lobe's own τ_m
# memory) and/or eligibility trace `trace_post` (a fixed ~20-tick low-pass of
# spikes), gathered at a fixed seeded subset of neurons. The brain's built-in
# Hebbian `W_out`/`output` is NOT used as a feature: it is unsupervised and drifts
# during replay.

using CUDA
using LiquidCortex
using SHA

struct ReservoirSpec
    kind::Symbol                    # :single or :ensemble
    taus_ticks::Vector{Float32}
    weights::Vector{Float32}
    N::Int
    conn_prob::Float64
    spectral_radius::Float32
    sigma::Float32
    hist_depth::Int
    input_gain::Float32
    readout_neurons::Int
    features::Vector{Symbol}        # subset of [:membrane, :trace]
    n_out::Int
    seed::Int
end

struct ReservoirRun
    feats::Dict{String,Matrix{Float32}}     # session id => T×D
    lobe_names::Vector{String}
    cols_per_lobe::Int
    neurons::Int
    nnz::Int
    w_hash_before::Vector{String}
    w_hash_after::Vector{String}
    ms_per_step::Float64
    steps::Int
    vram_used_gb_peak::Float64
    vram_baseline_gb::Float64
end

_vram_used_gb() = (CUDA.total_memory() - CUDA.free_memory()) / 1e9
_lobes(b::SparseBrain) = [b]
_lobes(b::EnsembleBrain) = b.lobes
w_hash(l::SparseBrain) = bytes2hex(sha256(reinterpret(UInt8, Array(l.W.nzVal))))

function build_brain(spec::ReservoirSpec, n_in::Int)
    cfg = BrainConfig(N=spec.N, conn_prob=spec.conn_prob, spectral_radius=spec.spectral_radius,
        sigma=spec.sigma, hist_depth=spec.hist_depth, rng=Random.Xoshiro(spec.seed))
    if spec.kind === :single
        return SparseBrain(spec.taus_ticks[1]; cfg=cfg, n_in=n_in, n_out=spec.n_out, name="single")
    end
    return EnsembleBrain(; n_in=n_in, n_out=spec.n_out, cfg=cfg, taus=spec.taus_ticks,
        weights=spec.weights)
end

"""
    run_reservoir(spec, sessions_z, order; plasticity, stress, reflex_gain)

Replay each session in `order` (`id => T×F Float32`, z-scored) from a reset state.
The state at step `t` has seen inputs `1..t` only. `reflex_gain > 0` forwards the
stress signal to the brain's reflex-gated readout adaptation (Experiment B); with
`0` the brain is driven with defaults.
"""
function run_reservoir(spec::ReservoirSpec, sessions_z::AbstractDict, order::Vector{String};
    plasticity::Symbol=:readout_only, stress::Union{Nothing,AbstractDict}=nothing,
    reflex_gain::Real=0.0)
    bad = setdiff(spec.features, [:membrane, :trace])
    isempty(bad) || throw(ArgumentError("unsupported reservoir feature(s) $(bad); use :membrane and/or :trace"))
    n_in = size(sessions_z[order[1]], 2)
    base_gb = _vram_used_gb()
    brain = build_brain(spec, n_in)
    lobes = _lobes(brain)
    try
        pick = sort(randperm(Random.Xoshiro(spec.seed + 17), spec.N)[1:min(spec.readout_neurons, spec.N)])
        idx = CuArray(Int32.(pick))
        cpl = length(pick) * length(spec.features)
        names = brain isa EnsembleBrain ? copy(brain.lobe_names) : ["single"]
        nnz_total = sum(l.nnz for l in lobes)
        before = [w_hash(l) for l in lobes]
        feats = Dict{String,Matrix{Float32}}()
        peak = _vram_used_gb(); steps = 0
        t_start = time_ns()
        for id in order
            Z = sessions_z[id]; T = size(Z, 1)
            reset!(brain)
            F = Matrix{Float32}(undef, T, cpl * length(lobes))
            for t in 1:T
                u = Z[t, :] .* spec.input_gain
                s = stress === nothing ? 0.0f0 : Float32(stress[id][t])
                reflex_eta = LiquidCortex.ETA * (1 + Float32(reflex_gain) * s)
                if brain isa EnsembleBrain
                    ensemble_step!(brain, u; plasticity=plasticity, reflex_eta=reflex_eta,
                        reflex_signal=reflex_gain > 0 ? s : 0.0f0, record_history=false)
                else
                    step!(brain, u; plasticity=plasticity, reflex_eta=reflex_eta, record_history=false)
                end
                for (li, l) in enumerate(lobes)
                    c = (li - 1) * cpl
                    for f in spec.features
                        src = f === :membrane ? l.V : l.trace_post
                        F[t, c+1:c+length(pick)] = Array(src[idx])
                        c += length(pick)
                    end
                end
                steps += 1
                steps % 64 == 0 && (peak = max(peak, _vram_used_gb()))
            end
            feats[id] = F
        end
        CUDA.synchronize()
        elapsed = (time_ns() - t_start) / 1e9
        peak = max(peak, _vram_used_gb())
        after = [w_hash(l) for l in lobes]
        return ReservoirRun(feats, names, cpl, spec.N * length(lobes), nnz_total, before, after,
            1e3 * elapsed / max(steps, 1), steps, peak, base_gb)
    finally
        free!(brain)
        CUDA.reclaim()
    end
end
