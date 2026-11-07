# Public API

Docstrings for every exported name. Examples allocate CUDA reservoirs and
are not executed as doctests on CPU CI.

```@docs
BrainConfig
SparseBrain
EnsembleBrain
EnsembleDesynchronizedError
LiquidCortexValidationError
step!
ensemble_step!
get_output
get_ensemble_output
reset!
free!
compute_reservoir_covariance
compute_reservoir_covariance!
spikes
membrane
traces
estimate_spectral_radius
diagnostics
ensemble_diagnostics
ReferenceLSM
run_lsm_step
run_lsm_step!
run_lsm_step_str
```
