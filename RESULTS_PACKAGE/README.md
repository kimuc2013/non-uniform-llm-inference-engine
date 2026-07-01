# Non-Uniform TP/PP Serving Planner — Results Package

Start with **docs/SERVING_PLANNER_EXPLAINED.md** (architecture + data + results +
honest findings). Deeper math/code walkthrough: docs/planner_describe.md.
Raw measured data: data/measured_results_all_layouts.csv. Figures: figures/.

## Update 2026-06-30
- **Baselines in the figures are pure uniform tensor-parallel (stock vLLM)** — TP=world
  per layout (TP8 at 4+4, TP4 at 2+2). PP / hybrid / non-uniform are the planner's own
  contributions, so the comparison is planner pick vs. stock-vLLM uniform TP only.
- **Mixed-traffic serving** (varied (input,output) shapes in one concurrent stream):
  figures/fig_mixed_traffic_{combined,4x4,2x2}.png, data/mixed_traffic_results.csv.
  4+4: opt30b +72/+163%, 70b +90/+170% over uniform TP. 2+2: opt30b +55/+61%, 70b
  +40/+56% (planner pick = measured oracle).
- **Cross-node AR cost model is now constant-free**: the hand-typed AR_EFFECTIVE_FACTOR
  and the per-radix isolated-bench surface were removed; the inter-node AllReduce term
  is radix-independent (per-node IB-NIC bottleneck) and the effective in-serving AR
  bandwidth is fit from serving throughput. This fixed the 2+2 large-model TP-vs-hybrid
  crossover (2+2 mean regret 17.6% -> 6.2%; 4+4 picks byte-identical).

Generated 2026-06-28, updated 2026-06-30. Source: github.com/kimuc2013/non-uniform-llm-inference-engine
