# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Standalone brain test — requires CUDA GPU
# Run: julia --project examples/brain_standalone.jl

using LiquidCortex
using CUDA

println("Starting LiquidCortex brain test...")
println("CUDA device: ", CUDA.name(CUDA.device()))

# Create a single lobe
brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="standalone")
println(diagnostics(brain))

# Step with a generic input vector
u = CUDA.zeros(Float32, 8)
for t in 1:100
    step!(brain, u; inhibition=0.2f0)
end
println("After 100 steps: ", diagnostics(brain))
println("Output: ", get_output(brain))
reset!(brain)
println("After reset!: ", diagnostics(brain))
free!(brain)

# Create the full ensemble
ensemble = EnsembleBrain(n_in=8, n_out=4)
println(ensemble_diagnostics(ensemble))

# Step the ensemble
for t in 1:50
    ensemble_step!(ensemble, u; inhibition=0.1f0, reflex_signal=0.15f0)
end
println("After 50 ensemble steps: ", ensemble_diagnostics(ensemble))
println("Ensemble output: ", get_ensemble_output(ensemble))
free!(ensemble)

println("Done.")
