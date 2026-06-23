"""Heterogeneous TP/PP serving planner (v2).

Active entry points (v2, calibrated from measured sweeps):
  - planner.perf_planner   : closed-form TPS prediction + plan() + CLI + validate
  - planner.fit_planner    : fit the free parameters to calibration_data.csv
  - planner.build_calibration : (re)build calibration_data.csv from results/
  - planner.cluster_env    : cluster paths / IB ifaces (CFG)
  - planner.hetero_sweep   : generalized measurement sweep (any model × GPU layout)

Artifacts: hw_params.json, fitted_params.json, calibration_data.csv,
PLANNER_SPEC.md, HANDOFF.md, mistral_prediction.json, mistral_validation.json.

The original cost-model planner (v1: cost_model / planner / scorer / workload /
gpu_library / network_library / model_spec) is archived under planner/legacy_v1/.
This file is intentionally a thin package marker so `import planner.perf_planner`
etc. work without pulling in v1.
"""
