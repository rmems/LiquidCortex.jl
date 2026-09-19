# Public API

Docstrings for every exported name. Examples allocate CUDA reservoirs and
are not executed as doctests on CPU CI.

```@docs
SparseBrain
EnsembleBrain
EnsembleDesynchronizedError
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
diagnostics
ensemble_diagnostics
enable_telemetry!
run_lsm_step
run_lsm_step_str
```
