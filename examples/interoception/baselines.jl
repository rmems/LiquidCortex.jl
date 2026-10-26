# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Memoryless baselines and shared readouts. Host-only, CPU-safe, stdlib-only.
# Every model maps a feature matrix to the same target layout:
#   [forecast targets (h-major, z-scored) | one-hot supervisor label at +class_h]

using LinearAlgebra
using Random

# ── Ridge (closed form, with intercept) ───────────────────────────────────────
struct Ridge
    W::Matrix{Float64}      # D×O
    b::Vector{Float64}      # O
    λ::Float64
end

predict(m::Ridge, X::AbstractMatrix) = X * m.W .+ m.b'

function ridge_fit(X::AbstractMatrix, Y::AbstractMatrix, λ::Real)
    n, D = size(X)
    xm = vec(sum(X; dims=1)) ./ n; ym = vec(sum(Y; dims=1)) ./ n
    Xc = X .- xm'; Yc = Y .- ym'
    A = Symmetric(Xc' * Xc + (λ * n) * I)
    fac = cholesky(A; check=false)
    W = issuccess(fac) ? fac \ (Xc' * Yc) : (A + 1e-8I) \ (Xc' * Yc)
    return Ridge(W, ym .- W' * xm, Float64(λ))
end

mse(Y, P) = sum((Y .- P) .^ 2) / length(Y)

"""
    ridge_select(Xtr, Ytr, Xval, Yval, grid) -> (model, λ_table)

Pick λ by validation MSE; with no validation rows the median grid value is used and
the table says so.
"""
function ridge_select(Xtr, Ytr, Xval, Yval, grid)
    if size(Xval, 1) == 0
        λ = sort(collect(grid))[cld(length(grid), 2)]
        return ridge_fit(Xtr, Ytr, λ), Dict{String,Any}("selected" => λ, "criterion" => "none (no validation sessions)")
    end
    scores = [(λ, mse(Yval, predict(ridge_fit(Xtr, Ytr, λ), Xval))) for λ in grid]
    best = argmin(last.(scores))
    return ridge_fit(Xtr, Ytr, scores[best][1]),
    Dict{String,Any}("selected" => scores[best][1], "criterion" => "val_mse",
        "val_mse" => Dict(string(λ) => s for (λ, s) in scores))
end

# ── Small memoryless MLP (tanh, full-batch Adam, MSE) ─────────────────────────
struct MLP
    W1::Matrix{Float64}; b1::Vector{Float64}
    W2::Matrix{Float64}; b2::Vector{Float64}
end

predict(m::MLP, X::AbstractMatrix) = tanh.(X * m.W1 .+ m.b1') * m.W2 .+ m.b2'
n_params(m::MLP) = length(m.W1) + length(m.b1) + length(m.W2) + length(m.b2)
n_params(m::Ridge) = length(m.W) + length(m.b)

function fit_mlp(X, Y; hidden::Int=32, epochs::Int=300, lr::Real=0.01, wd::Real=1e-4,
    rng::AbstractRNG=Random.Xoshiro(0), Xval=nothing, Yval=nothing)
    n, F = size(X); O = size(Y, 2)
    W1 = randn(rng, F, hidden) .* sqrt(2 / (F + hidden)); b1 = zeros(hidden)
    W2 = randn(rng, hidden, O) .* sqrt(2 / (hidden + O)); b2 = zeros(O)
    θ = (W1, b1, W2, b2)
    m1 = map(zero, θ); m2 = map(zero, θ)
    best = nothing; best_score = Inf
    β1, β2, ε = 0.9, 0.999, 1e-8
    for ep in 1:epochs
        A = tanh.(X * W1 .+ b1')
        P = A * W2 .+ b2'
        dP = (2 / (n * O)) .* (P .- Y)
        gW2 = A' * dP .+ wd .* W2
        gb2 = vec(sum(dP; dims=1))
        dZ = (dP * W2') .* (1 .- A .^ 2)
        gW1 = X' * dZ .+ wd .* W1
        gb1 = vec(sum(dZ; dims=1))
        for (k, g) in enumerate((gW1, gb1, gW2, gb2))
            m1[k] .= β1 .* m1[k] .+ (1 - β1) .* g
            m2[k] .= β2 .* m2[k] .+ (1 - β2) .* g .^ 2
            θ[k] .-= lr .* (m1[k] ./ (1 - β1^ep)) ./ (sqrt.(m2[k] ./ (1 - β2^ep)) .+ ε)
        end
        if Xval !== nothing && size(Xval, 1) > 0 && ep % 10 == 0
            s = mse(Yval, predict(MLP(W1, b1, W2, b2), Xval))
            if s < best_score
                best_score = s; best = MLP(copy(W1), copy(b1), copy(W2), copy(b2))
            end
        end
    end
    return best === nothing ? MLP(W1, b1, W2, b2) : best
end

# ── Online (NLMS) readout adaptation ──────────────────────────────────────────
"""
    prequential_nlms(W0, b0, F, Y, ranges, hmax; lr, gain=0, stress)

Per-session prequential evaluation of an adapting linear readout. At row `r` the
model predicts first; the update for row `r - hmax` is applied afterwards, because
that row's targets (up to `hmax` steps ahead) have only just been revealed.
The step size is `lr·(1 + gain·stress[r])` (`gain = 0` is the ungated control).
Each session restarts from the frozen `(W0, b0)`. Returns the prediction matrix.
"""
function prequential_nlms(W0::AbstractMatrix, b0::AbstractVector, F::AbstractMatrix,
    Y::AbstractMatrix, ranges, hmax::Int; lr::Real, gain::Real=0.0, stress::AbstractVector)
    P = similar(Y, size(F, 1), size(Y, 2))
    for rg in ranges
        W = copy(W0); b = copy(b0)
        for (k, r) in enumerate(rg)
            φ = @view F[r, :]
            P[r, :] = W' * φ .+ b
            k > hmax || continue
            s = rg[k - hmax]
            φs = @view F[s, :]
            err = Y[s, :] .- (W' * φs .+ b)
            step = lr * (1 + gain * stress[r]) / (1e-6 + dot(φs, φs) + 1.0)
            W .+= step .* (φs * err')
            b .+= step .* err
        end
    end
    return P
end
