# Machine Interoception Benchmark v1

Design note for [`examples/interoception/`](../examples/interoception). The benchmark asks
whether a multi-timescale liquid reservoir predicts **future machine state** from hardware
telemetry better than memoryless baselines. It is an evaluation harness only: it never
controls a host, and the core package stays application-neutral (all telemetry code lives
under `examples/`).

## Phase 0 — timebase contract

One model step (`BrainConfig.dt = 1` tick) is exactly **one replay interval**. The replay
interval is a required, explicit choice (`[timebase] replay_dt_ms`); it is never silently
inherited from the source cadence.

| Quantity | Definition |
|---|---|
| source sample interval | median spacing of `t_ms`; any gap off the median by more than `jitter_tol` (25 % default) is an error, never resampled over |
| replay interval | `replay_dt_ms` — wall-clock span of one model step |
| model `dt` | always `1.0` tick; physical time per step is `replay_dt_ms` |
| forecast horizon | `steps = horizon_ms / replay_dt_ms`; must be a positive integer, otherwise the horizon is skipped and listed under `task.skipped_horizons` (never rounded) |

### Source vs. replay cadence

| Case | Policy |
|---|---|
| `replay == source` | identity |
| `replay` an integer multiple of `source` | causal bin: each replay step is the mean of the source samples in its bin (`max` for binary flags such as `throttle_active`), stamped at the end of the bin; a trailing partial bin is dropped |
| `replay` faster than `source` | **error** — the source has no data between samples. `upsample = "hold"` opts into zero-order hold; the result then carries `timebase.synthetic_fill = true` |
| non-integer ratio | **error** — no interpolation |

Resampling never looks forward: a replay step only sees source samples at or before its own
timestamp.

### What `τ_m` means

`tau_mode` selects the interpretation of lobe time constants (default `10, 25, 50, 100`):

- `"relative_ticks"` (default): `τ_m` are model ticks, independent of cadence. At a 100 ms
  replay the Fast lobe has a 1 s memory, at a 5 s replay it has 50 s. The result records the
  physical equivalent (`experiments.timebase.lobe_taus_ms_equivalent`).
- `"literal_ms"`: `τ_m` are milliseconds, converted with `τ_ticks = τ_ms / replay_dt_ms`.
  The forward-Euler leak in `SparseBrain` needs `dt < 2τ`, so 10 ms on a 5 s step
  (0.002 ticks) is rejected with a `TimebaseError` instead of running ms-calibrated dynamics
  on 5 s samples.

Either way the unit is stated in the result; there is no implicit conversion.

## Dataset contract

One CSV per session (file stem = session id) with a `t_ms` column and any of the canonical
columns:

```text
gpu_temp_c gpu_power_w gpu_util_pct vram_used_pct cpu_temp_c cpu_power_w
pcie_rx_rate pcie_tx_rate throttle_active
```

- `required_features` must exist in every session or loading fails. Other canonical columns
  that are absent are dropped and listed in `dataset.missing_features`; they are **never**
  replaced by a constant.
- Empty/`NaN` cells in a used column are an error (no implicit fill, no null-as-zero).
- Export-specific names are mapped explicitly with `[dataset.column_map]`; unmapped columns
  are ignored.
- Splits are **by session** (`[split]` lists, or a deterministic SHA-1 hash split). Overlap,
  unknown ids or an empty train/test partition are errors; the session ids per partition,
  per-file SHA-256 and a dataset hash are written to the result.
- Normalization (z-score) is fit on **train sessions only** after resampling.
- Rows are issued at step `t ∈ warmup+1 … T−max_horizon`: features use steps `≤ t`, targets
  are read at `t+h` inside the same session. Every model is scored on exactly the same rows.
  The first `warmup_steps` are a washout for the reservoir and are excluded for all models.

### Real data

Pointing the harness at real sessions needs a canonical table with the columns above.
`rmems/gaming-telemetry`'s canonical CSV currently exports only
`timestamp_ms, gpu_temp_c, gpu_power_w, cpu_tctl_c, cpu_package_power_w, session_label`:
map `cpu_tctl_c → cpu_temp_c`, `cpu_package_power_w → cpu_power_w`, set
`time_column = "timestamp_ms"`, and split the multi-title `session_label` into one file per
session. VRAM and throttle state are in the raw Parquet (`memory_used_mb`) but not in that
export, so the forecasting targets `vram_used_pct`/`throttle_active` and the supervisor labels
(which need `vram_used_pct`) require an export extension first. The bundled fixture is
**synthetic** and exists to exercise the pipeline, not to support scientific claims.

## Tasks

**A. Forecasting** — `gpu_temp_c`, `vram_used_pct`, `gpu_power_w`, `throttle_active` at
+100 ms, +500 ms, +1 s, +5 s (whichever are integer multiples of the replay interval).
Metrics: MAE and RMSE per target and horizon (raw units), RMSE skill versus persistence,
threshold-crossing lead time (temperature ≥ `throttle_temp_c`, VRAM ≥ `throttle_vram_pct`),
and for `throttle_active` the Brier score, ECE and a reliability table. The throttle
"probability" is a clipped linear-readout output, so read ECE as a diagnostic of that
estimator, not as a calibrated classifier.

**B. Supervisor classification** — `ALLOW / WARN / THROTTLE / PAUSE` from an offline
deterministic policy over temperature, VRAM and the hardware throttle flag (`[policy]`
thresholds). The label of the state at `+class_horizon_ms` is predicted from the history up
to `t`. Metrics: macro F1 (over classes present in the truth), confusion matrix, false-safe
rate `P(pred = ALLOW | truth = PAUSE)` (`null` when the test truth has no PAUSE), accuracy and
warning lead time (alert = predicted `≥ WARN`, event = onset of true `≥ THROTTLE`). Labels are
for evaluation and training only.

All readouts predict one stacked target (z-scored forecasts plus a one-hot label) and the
label is the arg-max; this keeps every model on identical footing.

## Models

| Name in result | What it is |
|---|---|
| `persistence` | `next = current`; label = policy applied to the current state |
| `linear_current_features` | ridge on the current z-scored features (λ chosen on validation sessions) |
| `mlp_current_features` | memoryless tanh MLP (hidden 32, Adam, early-stopped on validation) |
| `liquid_single_A_readout_only` | one `SparseBrain` lobe, frozen recurrent `W`, host ridge readout |
| `liquid_ensemble_A_readout_only` | four-lobe `EnsembleBrain`, same readout |
| `liquid_ensemble_B_*` | Experiment B (below) |
| `liquid_ensemble_C_recurrent_stdp` | Experiment C (below), off by default |

Reservoir inputs are the z-scored features times `input_gain`. The readout features are the
membrane potential `V` (which carries each lobe's own `τ_m` memory) and/or `trace_post` at a
fixed seeded subset of neurons per lobe (`readout_neurons`). The brain's built-in
Hebbian `W_out`/`output` is **not** used: it is unsupervised and drifts during replay, so the
supervised readout is fit on the host. `experiments.lobe_ablation_A` refits the readout on each
lobe's columns alone so lobe contributions can be read per horizon.

### Experiments

- **A — frozen reservoir + trained readout (required first result).** `plasticity =
  :readout_only`; the SHA-256 of every lobe's recurrent `W.nzVal` is recorded before and after
  replay and `recurrent_weights_unchanged` must be `true`.
- **B — reflex-gated readout.** Starting from the Experiment A readout, a normalised-LMS
  update adapts the host readout online on the test sessions (prequential: predict, then update
  with the row whose targets were revealed `max_horizon` steps ago; every session restarts from
  the frozen readout). The step size is `lr·(1 + gain·stress)`. The stress signal is
  `clamp(max((gpu_temp_c − temp_lo)/(temp_hi − temp_lo), (vram_used_pct − vram_lo)/(vram_hi −
  vram_lo)), 0, 1)` from the current causal row, with the scale parameters from `[reflex]`
  logged in `experiments.reflex`. `liquid_ensemble_B_frozen` must equal Experiment A;
  `…_online_ungated` (`gain = 0`) is the control for `…_online_reflex_gated`. Because the
  built-in Hebbian readout does not feed back into reservoir state, reflex gating changes only
  this host-side readout.
- **C — recurrent pair-STDP (ablation).** `plasticity = :recurrent_stdp`, enabled with
  `[experiments] C = true`. STDP stays on through train, validation and test replay, so the
  reservoir the readout is fit on is not the one it is evaluated on; this is deliberately an
  ablation, not a headline number.

## Reporting size, not just score

Each model carries `size.trainable_params`, `size.frozen_params` and
`size.flops_per_step_est` (an estimate: `2·nnz + 2·N·n_in` per lobe plus the readout).
`systems` records neurons, non-zeros, wall time per replay step (including feature gathering),
sampled peak VRAM and the device. Compare the 1 k-neuron ensemble against a 4 k-parameter
baseline on skill *per* parameter and per FLOP, not on score alone. Energy, if any, comes from
the external harness.

## Reproducibility

`seed` drives reservoir construction and noise (`BrainConfig(rng = Xoshiro(seed))`, host
noise), the readout neuron subset and the MLP initialisation. For an identical fixture, seed,
code and device, `InteroceptionBenchmark.deterministic_view(result)` repeats exactly in our
runs; the documented tolerance for comparing runs (`summaries_match`) is `rtol = 1e-4`,
`atol = 1e-6`, to allow for float16 sparse accumulation differences across GPUs/driver
versions. The result also records the git commit (and a dirty flag), the dataset hash and
Julia/CUDA/LiquidCortex versions.

## Out of scope

Host control, FPGA deployment, language-model training and any biological-equivalence claim.
Distillation of a successful teacher into a small `Spikenaut-SNN` student is follow-up work
(`rmems/SynapticDistill.jl`).
