# Figure Index (final planner: APC-corrected workloads, twin-identified engine costs)

All figures regenerated 2026-07-02 with the fit-free measured calibration
(measured_params.json; PLANNER_MATH.md is the model spec). `all_figures.pdf`
bundles everything below in reading order.

## Headline
| file | shows |
|---|---|
| `fig_mixed_traffic_combined.png` | Mixed-traffic serving, 4+4 & 2+2 × {8B, OPT-30B, 70B}: uniform-TP baseline vs planner pick vs measured oracle. Planner +40~+173% over TP8; 70B/OPT30B ≈ oracle |
| `fig_mixed_traffic_4x4.png` / `_2x2.png` | Same, per layout (`fig_mixed_traffic.png` = 4+4 canonical copy) |

## Planner accuracy & validation
| file | shows |
|---|---|
| `fig_planner_validation.png` | Predicted-vs-measured TPS scatter (all layouts) + mean regret per layout (calibrated on 4+4, zero-refit 2+2/1+1) |
| `fig_selfval_vs_baseline.png` | Self-validation: planner pick vs uniform baseline across measured cells |
| `fig_mistral123b_prereg.png` | Pre-registered prediction test on an UNSEEN model (Mistral-123B): champion match 3/3, ρ=0.84 |
| `fig_mistral123b_workload_rows.png` | Mistral-123B measured vs predicted per workload row |

## Gains
| file | shows |
|---|---|
| `planner_uplift/planner_vs_baseline_uplift_{4x4,2x2,1x1}.png` | Planner uplift over uniform TP per model × workload × n_req |
| `fig_crossover_concurrency.png` | Optimal topology flips with concurrency (TP-heavy ↔ PP-heavy crossover) |
| `fig_layout_gain.png` | Gain by cluster layout |

## Per-model configuration rankings
| file | shows |
|---|---|
| `per_model_configs/per_model_configs_{4x4,2x2,1x1}_<workload>.png` | Measured throughput of every serving config vs planner prediction, ranked, per model |

## Notes
- Sweep "prefill_heavy" cells were measured with identical prompts + vLLM prefix
  caching ON → effectively decode-heavy (see PLANNER_MATH.md §12). Evaluations
  model this via `prefill_unique_frac=1/n`; future sweeps need unique prompts.
- Headline metrics: 17/17 invariants; mean uplift over uniform TP +89.9%
  (96/103 cells ≥ baseline); mean regret vs oracle 8.6% (2+2 2.8% / 4+4 6.9%).
