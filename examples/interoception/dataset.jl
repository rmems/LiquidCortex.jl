# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Dataset contract: one CSV per session (id = file stem) with a `t_ms` column and
# canonical feature columns. Splits are by session, normalization is fit on train
# sessions only, and nothing past time `t` enters the features at `t`.

using SHA

struct DatasetError <: Exception
    msg::String
end
Base.showerror(io::IO, e::DatasetError) = print(io, "DatasetError: ", e.msg)

const CANONICAL_FEATURES = ["gpu_temp_c", "gpu_power_w", "gpu_util_pct", "vram_used_pct",
    "cpu_temp_c", "cpu_power_w", "pcie_rx_rate", "pcie_tx_rate", "throttle_active"]
const MAX_AGG_FEATURES = Set(["throttle_active"])   # binary flags: bin with max, not mean
const LABELS = ["ALLOW", "WARN", "THROTTLE", "PAUSE"]

struct SourceSession
    id::String
    sha256::String
    t_ms::Vector{Float64}
    X::Matrix{Float64}          # T×F in `features` order, source cadence
end

function _parse_cell(s::AbstractString)
    t = lowercase(strip(s))
    (isempty(t) || t in ("nan", "na", "null", "none")) && return NaN
    t == "true" && return 1.0
    t == "false" && return 0.0
    v = tryparse(Float64, t)
    v === nothing && throw(DatasetError("non-numeric cell \"$s\""))
    return v
end

"""
    read_table(path) -> (header, columns::Dict{String,Vector{Float64}})

Minimal unquoted CSV reader. Empty/`NaN` cells become `NaN` (rejected later for
any column the benchmark uses).
"""
function read_table(path::AbstractString)
    lines = filter(l -> !isempty(strip(l)), readlines(path))
    isempty(lines) && throw(DatasetError("$path is empty"))
    header = String.(strip.(split(lines[1], ',')))
    length(unique(header)) == length(header) || throw(DatasetError("$path: duplicate column names"))
    cols = Dict(h => Float64[] for h in header)
    for (i, line) in enumerate(@view lines[2:end])
        f = split(line, ',')
        length(f) == length(header) || throw(DatasetError(
            "$path row $(i + 1): expected $(length(header)) cells, got $(length(f))"))
        for (h, c) in zip(header, f)
            v = try
                _parse_cell(c)
            catch e
                e isa DatasetError || rethrow()
                throw(DatasetError("$path row $(i + 1) column $h: $(e.msg)"))
            end
            push!(cols[h], v)
        end
    end
    return header, cols
end

"""
    load_sessions(dir; time_column="t_ms", required, jitter_tol=0.25)

Load every `*.csv` in `dir` (sorted by id). The feature list is the canonical
columns present in **all** sessions. A missing `required` column throws; a missing
optional canonical column is dropped and listed in `missing_features` — it is never
filled with a constant. `column_map` (`source => canonical`) renames export-specific
columns explicitly; unmapped columns are ignored. Returns `(sessions, features, missing_features, source_dt_ms)`.
"""
function load_sessions(dir::AbstractString; time_column::AbstractString="t_ms",
    required::AbstractVector{<:AbstractString}=String[], jitter_tol::Real=0.25,
    column_map::AbstractDict=Dict{String,String}())
    isdir(dir) || throw(DatasetError("dataset dir not found: $dir"))
    files = sort(filter(f -> endswith(f, ".csv"), readdir(dir)))
    isempty(files) && throw(DatasetError("no *.csv session files in $dir"))
    tables = Dict{String,Tuple{Vector{String},Dict{String,Vector{Float64}},String}}()
    for f in files
        path = joinpath(dir, f)
        header, cols = read_table(path)
        for (src, dst) in column_map                 # explicit rename, e.g. cpu_tctl_c => cpu_temp_c
            if haskey(cols, src)
                haskey(cols, dst) && throw(DatasetError("$f: column_map target \"$dst\" already exists"))
                cols[dst] = cols[src]; push!(header, dst)
            end
        end
        time_column in header || throw(DatasetError("$f: missing time column \"$time_column\""))
        tables[splitext(f)[1]] = (header, cols, bytes2hex(open(sha256, path)))
    end
    ids = sort(collect(keys(tables)))
    features = [c for c in CANONICAL_FEATURES if all(c in tables[i][1] for i in ids)]
    missing_features = [c for c in CANONICAL_FEATURES if !(c in features)]
    for r in required
        r in features || throw(DatasetError(
            "required feature \"$r\" is not present in every session (have: $(join(features, ", ")))"))
    end
    isempty(features) && throw(DatasetError("no canonical feature columns found"))
    sessions = SourceSession[]
    dts = Float64[]
    for id in ids
        _, cols, sha = tables[id]
        t = cols[time_column]
        any(isnan, t) && throw(DatasetError("$id: NaN in time column"))
        push!(dts, infer_source_dt_ms(t; jitter_tol=jitter_tol))
        X = hcat((cols[c] for c in features)...)
        bad = findfirst(isnan, X)
        bad === nothing || throw(DatasetError(
            "$id: missing value in column \"$(features[bad[2]])\" at row $(bad[1]); " *
            "repair or drop the row before replay (no implicit fill)"))
        push!(sessions, SourceSession(id, sha, t, X))
    end
    all(d -> abs(d - dts[1]) <= jitter_tol * dts[1], dts) || throw(DatasetError(
        "sessions have different source cadences ($(unique(round.(dts; digits=3))) ms); " *
        "resample them to one cadence first"))
    return sessions, features, missing_features, dts[1]
end

"""
    dataset_hash(sessions) -> String

SHA-256 over the sorted `(id, file sha256)` pairs.
"""
dataset_hash(sessions) =
    bytes2hex(sha256(join(("$(s.id):$(s.sha256)" for s in sort(sessions; by=s -> s.id)), "\n")))

"""
    split_sessions(ids; train, val, test) -> NamedTuple

Session-level split. Explicit lists win; with none given, ids are ordered by the
SHA-1 of the id (deterministic, independent of file order) and cut 60/20/20.
Throws on overlap, unknown ids, or an empty train/test partition.
"""
function split_sessions(ids::AbstractVector{<:AbstractString};
    train=String[], val=String[], test=String[])
    ids = sort(String.(ids))
    train, val, test = String.(train), String.(val), String.(test)
    if isempty(train) && isempty(val) && isempty(test)
        order = sort(ids; by=i -> bytes2hex(sha1(i)))
        n = length(order)
        n >= 3 || throw(DatasetError("need at least 3 sessions for an automatic split, got $n"))
        ntest = max(1, round(Int, 0.2n)); nval = max(1, round(Int, 0.2n))
        test = order[1:ntest]; val = order[ntest+1:ntest+nval]; train = order[ntest+nval+1:end]
    end
    all_used = vcat(train, val, test)
    unknown = setdiff(all_used, ids)
    isempty(unknown) || throw(DatasetError("split names unknown session ids: $(join(unknown, ", "))"))
    if length(unique(all_used)) != length(all_used)
        dup = unique(filter(x -> count(==(x), all_used) > 1, all_used))
        throw(DatasetError("train/val/test overlap: $(join(dup, ", "))"))
    end
    isempty(train) && throw(DatasetError("train partition is empty"))
    isempty(test) && throw(DatasetError("test partition is empty"))
    return (train=sort(train), val=sort(val), test=sort(test))
end

"""
    fit_normalization(mats) -> (mean, std)

Per-feature statistics from the **train** matrices only. Constant columns get
`std = 1` (so they normalize to 0, not NaN).
"""
function fit_normalization(mats::AbstractVector{<:AbstractMatrix})
    A = reduce(vcat, mats)
    μ = vec(sum(A; dims=1)) ./ size(A, 1)
    σ = vec(sqrt.(sum((A .- μ') .^ 2; dims=1) ./ size(A, 1)))
    σ = [s < 1e-9 ? 1.0 : s for s in σ]
    return μ, σ
end

zscore(X::AbstractMatrix, μ, σ) = (X .- μ') ./ σ'

# ── Offline deterministic supervisor policy ───────────────────────────────────
# Evaluation/training labels only. Nothing here controls a host.
struct Policy
    warn_temp_c::Float64
    throttle_temp_c::Float64
    pause_temp_c::Float64
    warn_vram_pct::Float64
    throttle_vram_pct::Float64
    pause_vram_pct::Float64
end

Policy(d::AbstractDict) = Policy(
    Float64(get(d, "warn_temp_c", 80.0)), Float64(get(d, "throttle_temp_c", 85.0)),
    Float64(get(d, "pause_temp_c", 90.0)), Float64(get(d, "warn_vram_pct", 90.0)),
    Float64(get(d, "throttle_vram_pct", 95.0)), Float64(get(d, "pause_vram_pct", 98.0)))

"""
    supervisor_label(p, temp, vram, throttle) -> Int

`1=ALLOW 2=WARN 3=THROTTLE 4=PAUSE`. PAUSE/THROTTLE/WARN on temperature or VRAM
thresholds; an active hardware throttle flag forces at least THROTTLE.
"""
function supervisor_label(p::Policy, temp::Real, vram::Real, throttle::Real)
    (temp >= p.pause_temp_c || vram >= p.pause_vram_pct) && return 4
    (temp >= p.throttle_temp_c || vram >= p.throttle_vram_pct || throttle >= 0.5) && return 3
    (temp >= p.warn_temp_c || vram >= p.warn_vram_pct) && return 2
    return 1
end

# ── Windowing ─────────────────────────────────────────────────────────────────
"""
    Split

Concatenated supervised rows for one partition. Row `i` is issued at replay step
`tidx[i]` of its session; features use rows `<= tidx[i]` only and targets are read at
`tidx[i] + h`.
"""
struct Split
    ids::Vector{String}
    ranges::Vector{UnitRange{Int}}      # per-session row ranges (contiguous in time)
    tidx::Vector{Int}
    X::Matrix{Float64}                  # n×F current raw features
    Yraw::Matrix{Float64}               # n×(H*T) future targets, h-major
    cur::Matrix{Float64}                # n×T current target values (persistence)
    lab_now::Vector{Int}
    lab_fut::Vector{Int}
    stress::Vector{Float64}
end

"""
    build_split(ids, sessions, features, targets, hsteps, class_steps, policy, warmup, stress_fn)

`sessions[id]` is the replay-cadence `T×F` matrix. Rows are `t ∈ warmup+1 : T-maxh`
so every horizon is defined for every row (all models are scored on identical rows).
"""
function build_split(ids, sessions::AbstractDict, features, targets, hsteps::Vector{Int},
    class_steps::Int, policy::Policy, warmup::Int, stress_fn)
    ti = Union{Int,Nothing}[findfirst(==(t), features) for t in targets]
    any(isnothing, ti) && throw(DatasetError("target(s) missing from features: " *
        join(targets[findall(isnothing, ti)], ", ")))
    ti = Int.(ti)
    isempty(ids) && return Split(String[], UnitRange{Int}[], Int[], zeros(0, length(features)),
        zeros(0, length(hsteps) * length(ti)), zeros(0, length(ti)), Int[], Int[], Float64[])
    it = findfirst(==("gpu_temp_c"), features)
    iv = findfirst(==("vram_used_pct"), features)
    ith = findfirst(==("throttle_active"), features)
    (it === nothing || iv === nothing) && throw(DatasetError(
        "supervisor labels need gpu_temp_c and vram_used_pct"))
    maxh = maximum(hsteps)
    Xs, Ys, Cs = Matrix{Float64}[], Matrix{Float64}[], Matrix{Float64}[]
    ln, lf, st, tidx = Int[], Int[], Float64[], Int[]
    ranges = UnitRange{Int}[]
    off = 0
    for id in ids
        S = sessions[id]; T = size(S, 1)
        ts = (warmup + 1):(T - maxh)
        isempty(ts) && throw(DatasetError(
            "session $id has $T replay steps: too short for warmup=$warmup + max horizon=$maxh"))
        thr(t) = ith === nothing ? 0.0 : S[t, ith]
        lab(t) = supervisor_label(policy, S[t, it], S[t, iv], thr(t))
        push!(Xs, S[ts, :])
        push!(Ys, hcat((S[ts .+ h, ti] for h in hsteps)...))
        push!(Cs, S[ts, ti])
        append!(ln, (lab(t) for t in ts))
        append!(lf, (lab(t + class_steps) for t in ts))
        append!(st, (stress_fn(@view S[t, :]) for t in ts))
        append!(tidx, ts)
        push!(ranges, off+1:off+length(ts)); off += length(ts)
    end
    return Split(collect(String.(ids)), ranges, tidx, reduce(vcat, Xs), reduce(vcat, Ys),
        reduce(vcat, Cs), ln, lf, st)
end
