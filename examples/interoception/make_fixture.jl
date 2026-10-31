# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Deterministic synthetic telemetry (NOT real hardware data) for tests and the
# one-command demo:  julia examples/interoception/make_fixture.jl [outdir]
# 100 ms cadence, thermally lagged GPU temperature, workload-driven power/VRAM.

using Random

function synth_session(seed::Int; T::Int=240, dt_s::Float64=0.1)
    rng = Random.Xoshiro(seed)
    load = zeros(T); t = 1
    levels = [0.15, 0.5, 0.95, 1.0]
    while t <= T
        len = rand(rng, 25:60)
        lv = t > T ÷ 4 && rand(rng) < 0.6 ? rand(rng, levels[3:4]) : rand(rng, levels[1:3])
        load[t:min(T, t + len - 1)] .= lv
        t += len
    end
    pw = zeros(T); temp = zeros(T); cpu_t = zeros(T); vram = zeros(T); thr = zeros(T)
    p, tmp, ct, alloc, th = 0.1, 55.0 + 8rand(rng), 45.0, 0.25, 0.0
    for i in 1:T
        p += (load[i] - p) * dt_s / 0.4
        tmp += (45 + 55p - tmp) * dt_s / 4.0
        ct += (42 + 25load[i] - ct) * dt_s / 6.0
        alloc += ((0.25 + 0.7load[i]) - alloc) * dt_s / 2.0
        th = tmp >= 88 ? 1.0 : (tmp < 84 ? 0.0 : th)
        pw[i] = 40 + 260p + 4randn(rng); temp[i] = tmp + 0.3randn(rng)
        cpu_t[i] = ct + 0.4randn(rng); vram[i] = clamp(100alloc + 0.5randn(rng), 0, 100); thr[i] = th
    end
    rx = [max(0.0, 200load[i] * (1 + 0.5sin(i / 7)) + 20randn(rng)) for i in 1:T]
    tx = [max(0.0, 90load[i] * (1 + 0.5cos(i / 5)) + 10randn(rng)) for i in 1:T]
    util = clamp.(100 .* load .+ 3 .* randn(rng, T), 0, 100)
    cpuw = 15 .+ 70 .* load .+ 2 .* randn(rng, T)
    return (t_ms=[round(Int, (i - 1) * dt_s * 1000) for i in 1:T], gpu_temp_c=temp, gpu_power_w=pw,
        gpu_util_pct=util, vram_used_pct=vram, cpu_temp_c=cpu_t, cpu_power_w=cpuw,
        pcie_rx_rate=rx, pcie_tx_rate=tx, throttle_active=thr)
end

function write_fixture(dir::AbstractString; n_sessions::Int=6)
    mkpath(dir)
    cols = [:t_ms, :gpu_temp_c, :gpu_power_w, :gpu_util_pct, :vram_used_pct, :cpu_temp_c,
        :cpu_power_w, :pcie_rx_rate, :pcie_tx_rate, :throttle_active]
    for k in 1:n_sessions
        s = synth_session(1000 + k)
        open(joinpath(dir, "s$(lpad(k, 2, '0')).csv"), "w") do io
            println(io, join(cols, ","))
            for i in eachindex(s.t_ms)
                println(io, join((c === :t_ms ? string(s.t_ms[i]) :
                    (c === :throttle_active ? string(Int(s[c][i])) : string(round(s[c][i]; digits=2)))
                    for c in cols), ","))
            end
        end
    end
    return dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("wrote ", write_fixture(isempty(ARGS) ? joinpath(@__DIR__, "fixtures") : ARGS[1]))
end
