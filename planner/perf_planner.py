"""Universal hetero TP/PP serving planner.

Implements PLANNER_SPEC.md (incl. the 2026-06-23 corrections: hierarchical AR,
per-node NVLink/PCIe intra-AR, embed_on_stage, n_mb=min(pp,n_req), affine layer-
split water-fill, max+(1-ρ)min wall blend): closed-form throughput prediction + optimal
split derivation + config search for arbitrary (model, hardware, workload)
on heterogeneous clusters.

All hardware rates are EFFECTIVE values calibrated from measured sweeps
(planner/hw_params.json) — see spec §7 for the ≤6-cell probe procedure on a
new cluster.

CLI:
  python planner/perf_planner.py --model meta-llama/Llama-3.3-70B-Instruct \
      --in-len 512 --out-len 256 --n-req 128
  python planner/perf_planner.py --validate          # against calibration_data.csv
  python planner/perf_planner.py --predict-mistral   # pre-registered prediction
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
HW_PARAMS = HERE / "hw_params.json"
CALIB_CSV = HERE / "calibration_data.csv"


# ----------------------------------------------------------------------------
# Specs
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    name: str
    n_layers: int
    hidden: int
    ffn_dim: int
    n_q: int
    n_kv: int
    head_dim: int
    vocab: int
    params_b: float                 # total params (B)
    ffn_kind: str = "swiglu"        # swiglu (3 mats) | gelu (2 mats)
    tied_embed: bool = False

    @property
    def p_attn(self) -> float:      # per-layer attention params
        return (self.hidden * self.n_q * self.head_dim
                + 2 * self.hidden * self.n_kv * self.head_dim
                + self.n_q * self.head_dim * self.hidden)

    @property
    def p_ffn(self) -> float:
        mats = 3 if self.ffn_kind == "swiglu" else 2
        return mats * self.hidden * self.ffn_dim

    @property
    def p_layer(self) -> float:
        return self.p_attn + self.p_ffn

    @property
    def p_embed(self) -> float:
        return self.vocab * self.hidden * (1 if self.tied_embed else 2)

    @property
    def gqa_group(self) -> int:
        return self.n_q // self.n_kv

    @property
    def ffn_mats(self) -> int:
        return 3 if self.ffn_kind == "swiglu" else 2


MODELS = {
    "8b": ModelSpec("meta-llama/Llama-3.1-8B-Instruct", 32, 4096, 14336, 32, 8, 128, 128256, 8.0),
    "70b": ModelSpec("meta-llama/Llama-3.3-70B-Instruct", 80, 8192, 28672, 64, 8, 128, 128256, 70.0),
    "opt30b": ModelSpec("facebook/opt-30b", 48, 7168, 28672, 56, 56, 128, 50272, 30.0, ffn_kind="gelu", tied_embed=True),
    "qwen32b": ModelSpec("Qwen/Qwen3-32B", 64, 5120, 25600, 64, 8, 128, 151936, 32.8),
    "mistral123b": ModelSpec("mistralai/Mistral-Large-Instruct-2411", 88, 12288, 28672, 96, 8, 128, 32768, 123.0),
}


@dataclass(frozen=True)
class GpuType:
    name: str
    tflops_prefill: float           # effective bf16 TFLOPS (prefill regime)
    membw_gbs: float                # effective HBM BW streaming weights
    mem_gb: float


@dataclass(frozen=True)
class HardwareSpec:
    nodes: tuple                    # tuple[(GpuType, count), ...] in rank order
    ar_latency_us: float            # cross-node per-hop AR latency
    ar_bw_gbs: float                # cross-node AR algorithm bandwidth
    intra_ar_latency_us: float      # intra-node per-hop latency (PCIe / no-NVLink)
    intra_ar_bw_gbs: float
    p2p_latency_us: float
    p2p_bw_gbs: float
    # NVLink-equipped GPUs (the Blackwell head) get a much faster intra-node AR
    # than PCIe GPUs (the Ada worker has no NVLink). Modeling both with one PCIe-
    # like intra param over-charges the Blackwell stage's AR and under-skews PP
    # layer allocation. Physics-anchored constants (NOT fitted).
    nvlink_ar_latency_us: float = 4.0
    nvlink_ar_bw_gbs: float = 800.0
    overlap_eta: float = 0.65       # fork PP overlap efficiency
    step_floor_ms: float = 30.0     # per-decode-step CPU/dispatch floor (all topologies)
    c_mb_ms: float = 1.5            # per-microbatch CPU dispatch
    c_chunk_ms: float = 10.0        # per-prefill-chunk overhead
    mem_util: float = 0.85
    kv_bw_scale: float = 1.0        # KV read BW relative to weight BW
    prefill_overlap: float = 0.0    # fraction of prefill compute hidden under
                                    # decode HBM-streaming (chunked continuous
                                    # batching: tensor-core vs HBM overlap).
                                    # 0 → phases additive, 1 → fully hidden.
    prefill_tf_mult: float = 1.0    # corrects the ABSOLUTE prefill-TFLOPS level
                                    # (the 289/183 in hw_params were back-derived
                                    # under the old additive model and read low);
                                    # preserves the Blackwell:Ada ratio.
    prefill_ar_overlap: float = 0.0 # cross-node TP AllReduce hidden under the
                                    # GEMMs in prefill (Megatron async-TP /
                                    # sequence-parallel). 0 → AR fully exposed
                                    # (old), 1 → AR fully overlapped.

    @property
    def world(self) -> int:
        return sum(c for _, c in self.nodes)

    def gpu_of_rank(self, rank: int) -> GpuType:
        acc = 0
        for g, c in self.nodes:
            if rank < acc + c:
                return g
            acc += c
        raise IndexError(rank)

    def node_of_rank(self, rank: int) -> int:
        acc = 0
        for i, (_, c) in enumerate(self.nodes):
            if rank < acc + c:
                return i
            acc += c
        raise IndexError(rank)

    def intra_params(self, gpu_name: str):
        """(latency_us, bw_gbs) for an intra-node AR on this GPU type: NVLink for
        Blackwell, PCIe for everything else (the Ada worker)."""
        if "blackwell" in gpu_name.lower():
            return self.nvlink_ar_latency_us, self.nvlink_ar_bw_gbs
        return self.intra_ar_latency_us, self.intra_ar_bw_gbs


def load_hardware(path: Path = HW_PARAMS) -> HardwareSpec:
    d = json.loads(path.read_text())
    bw = GpuType("blackwell", d["blackwell"]["eff_tflops_prefill"],
                 d["blackwell"]["eff_membw_decode_gbs"], d["blackwell"]["mem_gb"])
    ada = GpuType("ada", d["ada"]["eff_tflops_prefill"],
                  d["ada"]["eff_membw_decode_gbs"], d["ada"]["mem_gb"])
    ic = d["interconnect"]
    kw = dict(
        nodes=((bw, 4), (ada, 4)),
        ar_latency_us=ic["ar_latency_us"],
        ar_bw_gbs=ic["ar_bw_gbs"],
        intra_ar_latency_us=20.0,
        intra_ar_bw_gbs=60.0,        # PCIe-class intra-node (no NVLink on these boxes)
        p2p_latency_us=ic["p2p_latency_us"],
        p2p_bw_gbs=ic["p2p_bw_gbs"],
        prefill_ar_overlap=d.get("prefill_ar_overlap", 0.0),
    )
    # Overlay calibration-fitted engine params if present so the planner
    # actually predicts with the fit (fit_planner.py output). Without this the
    # CLI / --validate path silently used dataclass defaults, disconnected
    # from the fit.
    fp = HERE / "fitted_params.json"
    if fp.exists():
        fit = json.loads(fp.read_text()).get("fitted", {})
        for k in ("ar_latency_us", "ar_bw_gbs", "intra_ar_latency_us",
                  "overlap_eta", "step_floor_ms", "c_mb_ms", "c_chunk_ms",
                  "prefill_overlap"):
            if k in fit:
                kw[k] = fit[k]
    return HardwareSpec(**kw)


@dataclass(frozen=True)
class Workload:
    in_len: int
    out_len: int
    n_req: int

    @property
    def kv_avg(self) -> float:
        return self.in_len + self.out_len / 2


@dataclass
class Config:
    tp: int
    pp: int
    layer_split: list               # len pp
    ffn_splits: list                # len tp (per-rank cols, same per stage)
    head_splits: list               # len tp (q heads)
    kv_splits: list                 # len tp
    label: str = ""

    def short(self) -> str:
        if self.pp == 1:
            return f"TP{self.tp} ffn({self.ffn_splits[0]}:{self.ffn_splits[-1]}) head({self.head_splits[0]}:{self.head_splits[-1]})"
        return f"TP{self.tp}PP{self.pp} ({':'.join(str(l) for l in self.layer_split)})"


# ----------------------------------------------------------------------------
# Core cost model (spec §2–§5)
# ----------------------------------------------------------------------------

B_W = 2          # bytes per weight (bf16)
B_KV = 2
B_A = 2
T_CHUNK = 8192   # max_num_batched_tokens-ish prefill chunk
C_ACT = 6        # peak activation multiplier in mem_feasible (PLANNER_SPEC §6, c_act≈4–8)


def embed_on_stage(m: ModelSpec, pp: int, stage: int) -> float:
    """Embedding params physically resident on a PP stage (pre-TP-division).

    pp==1: the single stage holds BOTH the input table and lm_head → p_embed
    (= V·h for tied, 2·V·h for untied). pp>1: the input embedding table (V·h)
    lives on stage 0 and the lm_head (V·h) on the last stage — two *separate*
    V·h tensors regardless of tying (PP can't share a weight across stages, so
    a tied model replicates). Charging the full p_embed on each end stage (the
    old `with_embed` path) double-counted untied embeddings → an anti-PP bias.
    """
    if pp == 1:
        return m.p_embed
    if stage == 0 or stage == pp - 1:
        return float(m.vocab * m.hidden)        # one V·h table per end stage
    return 0.0


def params_on_rank(m: ModelSpec, layers: int, head_r: int, ffn_r: int,
                   tp: int, embed_params: float) -> float:
    p = layers * (m.p_attn * head_r / m.n_q + m.p_ffn * ffn_r / m.ffn_dim)
    p += embed_params / tp
    return p


def t_allreduce_ms(msg_bytes: float, ranks: list, hw: HardwareSpec) -> float:
    """Hierarchical (2-tier) all-reduce cost over the actual rank placement.

    A cross-node TP group does NOT pay 2(N-1) serial IB-latency hops: NCCL runs
    intra-node reduce-scatter + all-gather over NVLink and only an inter-node
    ring over IB. So latency = 2(n_nodes-1)·α_ib + 2(n_local-1)·α_nv, i.e. for a
    4+4 TP8 group just ONE IB hop-pair, not seven. Single-node groups reduce to
    the plain intra-node ring (unchanged behaviour). This is the term that was
    over-charging high-radix cross-node decode AR ~6× and blocking the
    low-concurrency TP8 champion crossover.
    """
    n = len(ranks)
    if n <= 1:
        return 0.0
    counts = {}
    for r in ranks:
        nd = hw.node_of_rank(r)
        counts[nd] = counts.get(nd, 0) + 1
    n_nodes = len(counts)
    n_local = max(counts.values())            # ranks per node (balanced groups)
    if n_nodes == 1:
        # intra-node AR uses this node's interconnect (Blackwell=NVLink fast,
        # Ada=PCIe slow) — not a single shared param.
        lat, bw = hw.intra_params(hw.gpu_of_rank(ranks[0]).name)
        vol = 2 * (n - 1) / n * msg_bytes
        return vol / (bw * 1e9) * 1e3 + 2 * (n - 1) * lat / 1e3
    # 2-tier: NVLink reduce-scatter+all-gather, then an inter-node ring over IB.
    # The per-node IB NIC is shared by the n_local local GPUs, so the aggregate
    # inter-node volume on the bottleneck NIC is 2(n_nodes-1)/n_nodes·msg (the
    # /n_local of per-GPU traffic cancels against n_local GPUs sharing the NIC).
    t_intra = (2 * (n_local - 1) / n_local * msg_bytes) / (hw.intra_ar_bw_gbs * 1e9) * 1e3
    t_inter = (2 * (n_nodes - 1) / n_nodes * msg_bytes) / (hw.ar_bw_gbs * 1e9) * 1e3
    lat = 2 * (n_nodes - 1) * hw.ar_latency_us + 2 * (n_local - 1) * hw.intra_ar_latency_us
    return t_intra + t_inter + lat / 1e3


def stage_ranks(cfg: Config, hw: HardwareSpec, stage: int) -> list:
    """Global rank ids of a PP stage. Convention: stage s occupies ranks
    [s*tp, (s+1)*tp) — head node ranks first (matches sweep placement)."""
    return list(range(stage * cfg.tp, (stage + 1) * cfg.tp))


def stage_is_cross_node(cfg: Config, hw: HardwareSpec, stage: int) -> bool:
    rs = stage_ranks(cfg, hw, stage)
    return len({hw.node_of_rank(r) for r in rs}) > 1


def stage_time_decode_ms(m: ModelSpec, hw: HardwareSpec, w: Workload,
                         cfg: Config, stage: int, batch: int) -> float:
    layers = cfg.layer_split[stage]
    rs = stage_ranks(cfg, hw, stage)
    embed = embed_on_stage(m, cfg.pp, stage)
    t_max = 0.0
    for i, r in enumerate(rs):
        g = hw.gpu_of_rank(r)
        pr = params_on_rank(m, layers, cfg.head_splits[i], cfg.ffn_splits[i],
                            cfg.tp, embed)
        w_bytes = pr * B_W
        kv_read = batch * w.kv_avg * layers * 2 * cfg.kv_splits[i] * m.head_dim * B_KV
        t_mem = (w_bytes / (g.membw_gbs * 1e9)
                 + kv_read / (g.membw_gbs * hw.kv_bw_scale * 1e9)) * 1e3
        t_flop = (2 * pr * batch) / (g.tflops_prefill * 1e12) * 1e3
        t_max = max(t_max, max(t_mem, t_flop))
    msg = batch * m.hidden * B_A
    t_ar = 2 * layers * t_allreduce_ms(msg, rs, hw)
    return t_max + t_ar


def stage_time_prefill_ms(m: ModelSpec, hw: HardwareSpec, w: Workload,
                          cfg: Config, stage: int, chunk_tokens: int) -> float:
    layers = cfg.layer_split[stage]
    rs = stage_ranks(cfg, hw, stage)
    embed = embed_on_stage(m, cfg.pp, stage)
    t_max = 0.0
    for i, r in enumerate(rs):
        g = hw.gpu_of_rank(r)
        pr = params_on_rank(m, layers, cfg.head_splits[i], cfg.ffn_splits[i],
                            cfg.tp, embed)
        t_flop = (2 * pr * chunk_tokens) / (g.tflops_prefill * hw.prefill_tf_mult * 1e12) * 1e3
        t_mem = (pr * B_W) / (g.membw_gbs * 1e9) * 1e3
        t_max = max(t_max, max(t_flop, t_mem))
    msg = chunk_tokens * m.hidden * B_A
    # In prefill the TP AllReduce of large (chunk-sized) messages overlaps the
    # GEMMs under async-TP / sequence-parallel; charge only the exposed fraction.
    t_ar = 2 * layers * t_allreduce_ms(msg, rs, hw)
    return t_max + (1 - hw.prefill_ar_overlap) * t_ar


def predict(m: ModelSpec, hw: HardwareSpec, w: Workload, cfg: Config,
            overlap: bool = True) -> dict:
    """Returns dict(tps, ttft_ms, t_prefill_s, t_decode_s, feasible, mem_note).

    Decode steady-state model:
      pp=1:  cycle = t_stage(B=n_req) + floor
      pp>1:  each token-cycle every stage processes n_mb microbatches
             (each re-streams the stage weights — mem-bound work serializes
             on the same GPUs even under perfect overlap):
               stage_busy_s = n_mb·(t_s(mb) + c_mb)
               cycle = max_s busy + (1−η)·(Σbusy − max) + exposed_p2p + floor
    """
    feas, note = mem_feasible(m, hw, w, cfg)
    if not feas:
        return {"tps": 0.0, "feasible": False, "mem_note": note}

    pp = cfg.pp
    eta = hw.overlap_eta if (overlap and pp > 1) else (0.15 if pp > 1 else 0.0)

    # ---- decode phase ----
    if pp == 1:
        t_step = stage_time_decode_ms(m, hw, w, cfg, 0, w.n_req)
        t_cycle = t_step + hw.step_floor_ms
    else:
        # bq = pp microbatches, but never more than there are requests (else we
        # fabricate phantom microbatches). mb is the *exact* average so the
        # served count n_mb·mb == n_req (no dropped remainder): stage_time is
        # affine in batch, so n_mb·t_s(n_req/n_mb) charges n_mb weight-streams
        # and exactly n_req of linear KV/flop work.
        n_mb = min(pp, w.n_req)
        mb = w.n_req / n_mb
        busy = [n_mb * (stage_time_decode_ms(m, hw, w, cfg, s, mb) + hw.c_mb_ms)
                for s in range(pp)]
        b_max = max(busy)
        b_rest = sum(busy) - b_max
        # exactly pp-1 inter-stage P2P boundaries (last stage has no successor)
        t_send = (mb * m.hidden * B_A) / (hw.p2p_bw_gbs * 1e9) * 1e3 \
                 + hw.p2p_latency_us / 1e3
        t_cycle = b_max + (1 - eta) * b_rest + (1 - eta) * t_send * (pp - 1) \
                  + hw.step_floor_ms
    t_decode_s = w.out_len * t_cycle / 1e3

    # ---- prefill phase ----
    # Chunk the input; the LAST chunk takes the remainder (not a full T_CHUNK —
    # charging full T_CHUNK over-counts tokens and makes T_prefill jump at every
    # chunk boundary). c_chunk is a per-chunk CPU floor → charged once per chunk,
    # not per PP stage. Pipeline = fill (first chunk through all stages) +
    # steady (each later chunk costs the bottleneck stage + exposed bubble).
    total_in = w.n_req * w.in_len
    n_chunks = max(1, math.ceil(total_in / T_CHUNK))
    t_prefill_ms = 0.0
    fill = 0.0
    for j in range(n_chunks):
        ck = T_CHUNK if j < n_chunks - 1 else total_in - (n_chunks - 1) * T_CHUNK
        ts = [stage_time_prefill_ms(m, hw, w, cfg, s, ck) for s in range(pp)]
        t_max_c = max(ts)
        if j == 0:
            fill = sum(ts) + hw.c_chunk_ms                            # first chunk → all stages
            t_prefill_ms += fill
        else:
            t_prefill_ms += t_max_c + (1 - eta) * (sum(ts) - t_max_c) + hw.c_chunk_ms
    t_prefill_s = t_prefill_ms / 1e3

    # Prefill (compute/tensor-core bound) and decode (HBM-bound) run on
    # disjoint hardware resources and partially overlap under chunked
    # continuous batching: the bottleneck rank hides ρ of the smaller phase
    # behind the larger. ρ=0 → serial (additive); ρ=1 → fully hidden (max).
    rho = hw.prefill_overlap
    t_total = max(t_prefill_s, t_decode_s) + (1 - rho) * min(t_prefill_s, t_decode_s)
    tps = w.n_req * w.out_len / t_total if t_total > 0 else 0.0
    return {
        "tps": tps,
        "ttft_ms": fill + t_cycle,
        "t_prefill_s": t_prefill_s,
        "t_decode_s": t_decode_s,
        "t_cycle_ms": t_cycle,
        "feasible": True,
        "mem_note": note,
    }


# ----------------------------------------------------------------------------
# Memory feasibility (spec §6)
# ----------------------------------------------------------------------------

def mem_feasible(m: ModelSpec, hw: HardwareSpec, w: Workload, cfg: Config):
    worst = ""
    for s in range(cfg.pp):
        layers = cfg.layer_split[s]
        rs = stage_ranks(cfg, hw, s)
        embed = embed_on_stage(m, cfg.pp, s)
        for i, r in enumerate(rs):
            g = hw.gpu_of_rank(r)
            weights = params_on_rank(m, layers, cfg.head_splits[i],
                                     cfg.ffn_splits[i], cfg.tp, embed) * B_W
            kv = w.n_req * (w.in_len + w.out_len) * layers * 2 \
                 * cfg.kv_splits[i] * m.head_dim * B_KV
            act = C_ACT * T_CHUNK * max(m.hidden, 2 * cfg.ffn_splits[i]) * B_A
            overhead = (1.5 + 2.0) * 1e9      # CUDA ctx/NCCL + graph pool
            need = weights + kv + act + overhead
            cap = g.mem_gb * 1e9 * hw.mem_util
            if need > cap:
                return False, (f"rank{r}({g.name}) needs {need/1e9:.1f}GB > "
                               f"{cap/1e9:.1f}GB cap")
            worst = f"max used {need/1e9:.1f}GB on rank{r}"
    return True, worst


# ----------------------------------------------------------------------------
# Optimal split derivation (spec §3.3, §4)
# ----------------------------------------------------------------------------

def round_quantized(fracs, total, quantum):
    """Largest-remainder quantized allocation summing to total."""
    raw = [f * total / quantum for f in fracs]
    base = [max(1, int(x)) for x in raw]
    rem = total // quantum - sum(base)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - int(raw[i]), reverse=True)
    for k in range(abs(rem)):
        i = order[k % len(order)]
        base[i] += 1 if rem > 0 else -1
    return [b * quantum for b in base]


def optimal_tp_splits(m: ModelSpec, hw: HardwareSpec, w: Workload, tp: int,
                      decode_weight: float):
    """Closed-form non-uniform TP split for a cross-node TP group of size tp.
    decode_weight in [0,1]: how decode-dominated the workload is.
    Returns (ffn_splits, head_splits, kv_splits)."""
    ranks = list(range(tp))
    speeds = []
    for r in ranks:
        g = hw.gpu_of_rank(r)          # tp==world ⇒ rank id = global rank
        s_dec = g.membw_gbs
        s_pre = g.tflops_prefill
        speeds.append(decode_weight * s_dec / hw.gpu_of_rank(0).membw_gbs
                      + (1 - decode_weight) * s_pre / hw.gpu_of_rank(0).tflops_prefill)
    tot = sum(speeds)
    fracs = [s / tot for s in speeds]

    kv_per_rank = m.n_kv // tp
    biasable_attn = (m.gqa_group == 1 and m.n_kv % tp == 0) or kv_per_rank >= 2

    if biasable_attn:
        head_q = m.gqa_group if m.gqa_group > 1 else 1
        heads = round_quantized(fracs, m.n_q, head_q)
        kv = [h // m.gqa_group for h in heads] if m.gqa_group > 1 else heads[:]
        ffn = round_quantized(fracs, m.ffn_dim, 128)
        return ffn, heads, kv

    # Case B: attention floor — heads uniform, water-fill FFN columns
    heads = [m.n_q // tp] * tp
    kv = [max(1, m.n_kv // tp)] * tp
    a, c = [], []
    for i, r in enumerate(ranks):
        g = hw.gpu_of_rank(r)
        bw = g.membw_gbs * 1e9
        tf = g.tflops_prefill * 1e12
        # attention floor a_r = attn weight stream + KV read (per layer, per
        # rank). The KV term (spec §3.3 Case B) is independent of the FFN width
        # being solved for, so it belongs in the fixed cost a_r; omitting it
        # under-weights the attention floor on KV-heavy (no-GQA) models.
        kv_read = (w.n_req * w.kv_avg * 2 * max(1, m.n_kv // tp)
                   * m.head_dim * B_KV) / (bw * hw.kv_bw_scale)
        a_dec = (m.p_attn / tp * B_W) / bw + kv_read
        c_dec = (m.ffn_mats * m.hidden * B_W) / bw
        a_pre = (2 * m.p_attn / tp) / tf
        c_pre = (2 * m.ffn_mats * m.hidden) / tf
        a.append(decode_weight * a_dec + (1 - decode_weight) * a_pre)
        c.append(decode_weight * c_dec + (1 - decode_weight) * c_pre)
    lam = (m.ffn_dim + sum(a[i] / c[i] for i in ranks)) / sum(1 / c[i] for i in ranks)
    f = [(lam - a[i]) / c[i] for i in ranks]
    f = [max(128, x) for x in f]
    scale = m.ffn_dim / sum(f)
    f = [x * scale for x in f]
    fracs_f = [x / m.ffn_dim for x in f]
    ffn = round_quantized(fracs_f, m.ffn_dim, 128)
    return ffn, heads, kv


def optimal_layer_split(m: ModelSpec, hw: HardwareSpec, w: Workload, tp: int,
                        pp: int, decode_weight: float):
    """Closed-form layer split by water-filling on the *marginal per-layer stage
    cost*, computed by finite difference of the cost model itself so it stays
    consistent with predict().

    Balancing stage busy times needs L_s ∝ 1/c_s where c_s is the per-layer
    cost on stage s. c_s is NOT just 1/membw_s: each layer also pays an
    AllReduce that is (largely) GPU-independent, so c_s = a/speed_s + b with
    b>0. Using the raw membw ratio (b ignored) over-skews toward the fast node
    — the bug invariant #7 caught. The finite difference d(stage_time)/dL
    captures a/speed_s + b exactly (weight stream + KV + per-layer AR)."""
    ffn_u = [m.ffn_dim // tp] * tp
    head_u = [m.n_q // tp] * tp
    kv_u = [max(1, m.n_kv // tp)] * tp
    B = max(1, w.n_req // pp)
    chunk = min(max(1, w.n_req) * w.in_len, T_CHUNK)
    # Affine model of each stage: stage_time_s(L) = c_s·L + f_s, fit by two
    # cost-model evaluations (L=1,2). c_s = per-layer slope (weight stream + KV +
    # per-layer AR); f_s = L-independent intercept (input/output embedding,
    # lm_head GEMM, AR base). The last stage's lm_head makes f_s large there, so
    # it should hold FEWER layers — ignoring f_s under-skews (invariant #7).
    c, f = [], []
    for s in range(pp):
        c2 = Config(tp, pp, [2] * pp, ffn_u, head_u, kv_u)
        c1 = Config(tp, pp, [1] * pp, ffn_u, head_u, kv_u)
        t1 = (decode_weight * stage_time_decode_ms(m, hw, w, c1, s, B)
              + (1 - decode_weight) * stage_time_prefill_ms(m, hw, w, c1, s, chunk))
        t2 = (decode_weight * stage_time_decode_ms(m, hw, w, c2, s, B)
              + (1 - decode_weight) * stage_time_prefill_ms(m, hw, w, c2, s, chunk))
        c_s = max(t2 - t1, 1e-9)
        c.append(c_s)
        f.append(t1 - c_s)                      # intercept = stage_time(L=0)
    # Constrained water-fill: equalize c_s·L_s + f_s = T* s.t. ΣL_s = n_layers
    # and L_s ≥ 1. When a stage would fall below 1 it is pinned to 1 and the
    # remaining layers are RE-SOLVED over the active set (a plain max(1,·) +
    # multiplicative rescale unbalances the survivors and the fix-sum decrement
    # could drive a stage to 0 — the bug invariant caught). pp ≤ n_layers is
    # guaranteed by the caller, so a feasible ≥1 allocation always exists.
    inv = [1.0 / x for x in c]
    L = [0.0] * pp
    active = list(range(pp))
    remaining = float(m.n_layers)
    while active:
        denom = sum(inv[s] for s in active)
        T_star = (remaining + sum(f[s] * inv[s] for s in active)) / denom
        vals = {s: (T_star - f[s]) * inv[s] for s in active}
        below = [s for s in active if vals[s] < 1.0]
        if not below:
            for s in active:
                L[s] = vals[s]
            break
        for s in below:
            L[s] = 1.0
            remaining -= 1.0
            active.remove(s)
    # integer rounding that preserves the sum and never drops a stage below 1
    ls = [max(1, int(round(x))) for x in L]
    while sum(ls) > m.n_layers:
        cand = [j for j in range(pp) if ls[j] > 1]
        i = max(cand, key=lambda j: ls[j] - L[j])
        ls[i] -= 1
    while sum(ls) < m.n_layers:
        i = min(range(pp), key=lambda j: ls[j] - L[j])
        ls[i] += 1
    return ls


def decode_weight_of(w: Workload) -> float:
    """Fraction of wall time expected in decode (heuristic from token counts)."""
    decode_tokens = w.out_len
    prefill_tokens_equiv = w.in_len / 12   # prefill ~12× faster per token
    return decode_tokens / (decode_tokens + prefill_tokens_equiv)


# ----------------------------------------------------------------------------
# Search (spec §7)
# ----------------------------------------------------------------------------

def plan(m: ModelSpec, hw: HardwareSpec, w: Workload, top_k: int = 10,
         overlap: bool = True):
    world = hw.world
    dw = decode_weight_of(w)
    cands = []

    tp = world
    # ---- TP-only (pp=1) ----
    if m.n_q % tp == 0:
        # uniform
        cfg_u = Config(tp, 1, [m.n_layers],
                       [m.ffn_dim // tp] * tp, [m.n_q // tp] * tp,
                       [max(1, m.n_kv // tp)] * tp, label=f"TP{tp} uniform")
        cands.append(cfg_u)
        # optimal non-uniform
        ffn, heads, kv = optimal_tp_splits(m, hw, w, tp, dw)
        cfg_b = Config(tp, 1, [m.n_layers], ffn, heads, kv,
                       label=f"TP{tp} optimal-bias")
        cands.append(cfg_b)

    # ---- TP×PP factorizations ----
    # Enumerate EVERY divisor of world (not just powers of two) so odd layouts
    # in the 1+1..4+4 target are covered — e.g. world=6 (3+3) needs pp∈{2,3,6}.
    for pp in [d for d in range(2, world + 1) if world % d == 0]:
        tp_s = world // pp
        if m.n_q % tp_s != 0:
            continue
        if m.n_layers < pp:
            continue
        # uniform layers
        base = m.n_layers // pp
        rem = m.n_layers - base * pp
        ls_uniform = [base + (1 if s < rem else 0) for s in range(pp)]
        ffn_u = [m.ffn_dim // tp_s] * tp_s
        heads_u = [m.n_q // tp_s] * tp_s
        kv_u = [max(1, m.n_kv // tp_s)] * tp_s
        cands.append(Config(tp_s, pp, ls_uniform, ffn_u, heads_u, kv_u,
                            label=f"TP{tp_s}PP{pp} uniform"))
        # optimal layer split (+ ±1 neighborhood)
        ls_opt = optimal_layer_split(m, hw, w, tp_s, pp, dw)
        seen = {tuple(ls_uniform)}
        for delta in range(-3, 4):      # ±3 around the closed-form optimum
            ls = ls_opt[:]
            ls[0] += delta
            ls[-1] -= delta
            if min(ls) < 1 or sum(ls) != m.n_layers:
                continue
            t = tuple(ls)
            if t in seen:
                continue
            seen.add(t)
            cands.append(Config(tp_s, pp, ls, ffn_u, heads_u, kv_u,
                                label=f"TP{tp_s}PP{pp} opt{'+' if delta>=0 else ''}{delta}"))

    scored = []
    for cfg in cands:
        r = predict(m, hw, w, cfg, overlap=overlap)
        if r["feasible"]:
            scored.append((r["tps"], cfg, r))
    scored.sort(key=lambda x: -x[0])
    return scored[:top_k]


def uniform_tp_baseline(m: ModelSpec, hw: HardwareSpec) -> Config:
    """The naive default a user would pick without the planner: uniform tensor
    parallelism across ALL GPUs (homogeneous treatment). None if n_q isn't
    divisible by world (e.g. world=6)."""
    world = hw.world
    if m.n_q % world != 0:
        return None
    return Config(world, 1, [m.n_layers], [m.ffn_dim // world] * world,
                  [m.n_q // world] * world, [max(1, m.n_kv // world)] * world,
                  label=f"TP{world} uniform (baseline)")


SAFE_MARGIN = 0.30   # empirical never-slower threshold: at this margin plan_safe has
                     # 0 baseline-losses across all 55 measured cells (incl. a held-out
                     # workload + zero-refit 1+1/2+2 layouts). ~2× the MAPE band, to
                     # cover confident crossover mispredictions. Ties traded for safety.


def plan_safe(m: ModelSpec, hw: HardwareSpec, w: Workload,
              margin: float = SAFE_MARGIN, overlap: bool = True):
    """Risk-aware recommendation with a NEVER-SLOWER-THAN-BASELINE guard.

    Returns (cfg, predicted, deviated): the planner only deviates from the naive
    uniform-TP=world baseline when its best pick is predicted to beat the baseline
    by more than `margin` (~the MAPE band). Otherwise it returns the baseline.
    Rationale: prediction error at the TP↔PP crossover can be confident and wrong;
    acting only on a signal larger than the noise band means small mispredictions
    cost a tie (recommend baseline), never a loss. Validated: 0 baseline-losses
    across all 55 measured cells (incl. a held-out workload + zero-refit layouts);
    for an ABSOLUTE guarantee on unseen configs, measure top-2 (pick + baseline)
    and run the faster.
    """
    base = uniform_tp_baseline(m, hw)
    scored = plan(m, hw, w, top_k=10, overlap=overlap)
    if base is None:                      # no uniform-TP=world baseline (odd world)
        return (scored[0][1], scored[0][2], True) if scored else (None, None, False)
    base_r = predict(m, hw, w, base, overlap=False)
    base_tps = base_r["tps"] if base_r.get("feasible") else 0.0
    if scored:
        top_tps, top_cfg, top_r = scored[0]
        if base_tps > 0 and top_tps > base_tps * (1 + margin):
            return top_cfg, top_r, True   # confident enough to deviate
    return base, base_r, False            # fall back to the safe default


# ----------------------------------------------------------------------------
# Validation against calibration data
# ----------------------------------------------------------------------------

def parse_calib_config(row) -> Config:
    layer_split = [int(x) for x in row["layer_split"].split("-")]
    ffn = [int(x) for x in row["ffn_splits"].split(":")]
    heads = [int(x) for x in row["head_splits"].split(":")]
    kv = [int(x) for x in row["kv_splits"].split(":")]
    return Config(int(row["tp"]), int(row["pp"]), layer_split, ffn, heads, kv,
                  label=row["label"])


def validate(csv_path: Path = CALIB_CSV):
    hw = load_hardware()
    rows = list(csv.DictReader(open(csv_path)))
    by_model = {}
    print(f"{'model':10s} {'label':42s} {'wl':14s} {'meas':>8s} {'pred':>8s} {'err%':>7s}")
    for row in rows:
        mkey = row["model"]
        if mkey not in MODELS:
            continue
        m = MODELS[mkey]
        regime = row.get("regime", "")
        # skip stock PP rows (planner models the overlap path); keep tp_only + overlap
        if regime == "stock":
            continue
        w = Workload(int(row["in_len"]), int(row["out_len"]), int(row["n_req"]))
        try:
            cfg = parse_calib_config(row)
        except Exception:
            continue
        meas = float(row["tps"])
        if meas <= 0:
            continue
        # Hard operating rule: n_req ≤ 100 (above it the Ada small-partition rank
        # OOMs into KV preemption/recompute thrashing — an unsupported regime the
        # cost model does not represent; old n=128 sweeps predate the rule).
        if int(row["n_req"]) > 100:
            continue
        pred = predict(m, hw, w, cfg, overlap=True)
        if not pred["feasible"]:
            err = float("nan")
        else:
            err = (pred["tps"] - meas) / meas * 100
        by_model.setdefault(mkey, []).append((row["label"], row["workload"], meas,
                                              pred.get("tps", 0), err,
                                              int(row["n_req"])))

    print("\n==== MAPE per model ====")
    for mkey, lst in by_model.items():
        errs = [abs(e) for _, _, _, _, e, _ in lst if not math.isnan(e)]
        print(f"{mkey:10s} n={len(errs):3d}  MAPE={sum(errs)/len(errs):6.1f}%")

    # champion match + regret per (model, workload, n_req). n_req is part of the
    # key because the champion crosses over with load (TP8 at low n → TP4PP2 at
    # high n); lumping n_req would compare configs optimal at different points.
    print("\n==== champion match + regret per (model × workload × n_req) ====")
    match = total = 0
    regrets = []
    reg_by_n = {}
    for mkey, lst in by_model.items():
        keys = sorted({(wl, nr) for _, wl, _, _, _, nr in lst})
        for wl, nr in keys:
            sub = [(lab, meas, pred) for lab, w2, meas, pred, _, n2 in lst
                   if w2 == wl and n2 == nr]
            if not sub:
                continue
            meas_champ, mc_tps, _ = max(sub, key=lambda x: x[1])
            pred_champ = max(sub, key=lambda x: x[2])[0]
            pc_meas = next(meas for lab, meas, _ in sub if lab == pred_champ)
            regret = (mc_tps - pc_meas) / mc_tps * 100
            ok = meas_champ == pred_champ
            match += ok; total += 1
            regrets.append(regret)
            reg_by_n.setdefault(nr, []).append((ok, regret))
            if not ok:
                print(f"  ✗ {mkey:10s} {wl:13s} n={nr:>3d} "
                      f"meas={meas_champ[:26]:26s} pred={pred_champ[:26]:26s} "
                      f"regret={regret:5.1f}%")
    import numpy as _np
    print(f"\n  champion {match}/{total}; mean regret {_np.mean(regrets):.1f}% "
          f"median {_np.median(regrets):.1f}% max {_np.max(regrets):.1f}%")
    print("  by n_req:  " + "  ".join(
        f"n={nr}:{sum(o for o,_ in v)}/{len(v)},reg{_np.mean([r for _,r in v]):.0f}%"
        for nr, v in sorted(reg_by_n.items())))


# ----------------------------------------------------------------------------
# Mistral pre-registration
# ----------------------------------------------------------------------------

MISTRAL_CONFIGS = [
    ("TP8PP1_uniform", 8, 1, [88], [3584]*8, [12]*8, [1]*8),
    ("TP8PP1_ffn_bias+25", 8, 1, [88], [4480]*4 + [2688]*4, [12]*8, [1]*8),
    ("TP8PP1_ffn_bias+50", 8, 1, [88], [5376]*4 + [1792]*4, [12]*8, [1]*8),
    ("TP8PP1_ffn_bias+75", 8, 1, [88], [6272]*4 + [896]*4, [12]*8, [1]*8),
    ("TP4PP2_layer_uniform_44-44", 4, 2, [44, 44], [7168]*4, [24]*4, [2]*4),
    ("TP4PP2_layer_skew+4_48-40", 4, 2, [48, 40], [7168]*4, [24]*4, [2]*4),
    ("TP4PP2_layer_skew+8_52-36", 4, 2, [52, 36], [7168]*4, [24]*4, [2]*4),
    ("TP4PP2_layer_skew+12_56-32", 4, 2, [56, 32], [7168]*4, [24]*4, [2]*4),
    ("TP4PP2_layer_skew+16_60-28", 4, 2, [60, 28], [7168]*4, [24]*4, [2]*4),
    ("TP2PP4_layer_uniform_22-22-22-22", 2, 4, [22]*4, [14336]*2, [48]*2, [4]*2),
    ("TP2PP4_layer_blackbias_24-24-20-20", 2, 4, [24, 24, 20, 20], [14336]*2, [48]*2, [4]*2),
    ("TP2PP4_layer_blackbias_26-26-18-18", 2, 4, [26, 26, 18, 18], [14336]*2, [48]*2, [4]*2),
    ("TP1PP8_layer_uniform_11x8", 1, 8, [11]*8, [28672], [96], [8]),
    ("TP1PP8_layer_blackbias_13-13-13-13-9-9-9-9", 1, 8, [13]*4 + [9]*4, [28672], [96], [8]),
    ("TP1PP8_layer_blackbias_15-15-15-15-7-7-7-7", 1, 8, [15]*4 + [7]*4, [28672], [96], [8]),
]

WORKLOADS = {
    "balanced": Workload(512, 256, 96),
    "decode_heavy": Workload(128, 512, 96),
    "prefill_heavy": Workload(1024, 128, 96),
}


def predict_mistral(out_path: Path = HERE / "mistral_prediction.json"):
    m = MODELS["mistral123b"]
    hw = load_hardware()
    out = {"model": m.name, "note": "PRE-REGISTERED prediction generated before sweep data existed",
           "predictions": {}, "champions": {}}
    for wl_name, w in WORKLOADS.items():
        preds = {}
        for label, tp, pp, ls, ffn, heads, kv in MISTRAL_CONFIGS:
            cfg = Config(tp, pp, list(ls), list(ffn), list(heads), list(kv), label)
            r = predict(m, hw, w, cfg, overlap=True)
            preds[label] = {
                "tps": round(r.get("tps", 0), 1),
                "feasible": r["feasible"],
                "ttft_ms": round(r.get("ttft_ms", 0), 1) if r["feasible"] else None,
                "note": r.get("mem_note", ""),
            }
        out["predictions"][wl_name] = preds
        feas = {k: v["tps"] for k, v in preds.items() if v["feasible"]}
        if feas:
            champ = max(feas, key=feas.get)
            out["champions"][wl_name] = {"config": champ, "tps": feas[champ]}
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["champions"], indent=2))
    print(f"\nwrote {out_path}")
    # Also brief table
    for wl_name in WORKLOADS:
        print(f"\n--- {wl_name} ---")
        preds = out["predictions"][wl_name]
        for label, p in sorted(preds.items(), key=lambda x: -x[1]["tps"]):
            f = "" if p["feasible"] else " (INFEASIBLE)"
            print(f"  {label:48s} {p['tps']:8.1f}{f}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="model key or HF name")
    ap.add_argument("--in-len", type=int, default=512)
    ap.add_argument("--out-len", type=int, default=256)
    ap.add_argument("--n-req", type=int, default=128)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--predict-mistral", action="store_true")
    args = ap.parse_args()

    if args.validate:
        validate()
        return
    if args.predict_mistral:
        predict_mistral()
        return
    if not args.model:
        ap.error("--model required (or --validate / --predict-mistral)")

    key = args.model
    if key not in MODELS:
        matches = [k for k, v in MODELS.items() if v.name == key]
        if matches:
            key = matches[0]
        else:
            ap.error(f"unknown model {key}; known: {list(MODELS)}")
    m = MODELS[key]
    hw = load_hardware()
    w = Workload(args.in_len, args.out_len, args.n_req)
    print(f"PLAN for {m.name}  in={w.in_len} out={w.out_len} n_req={w.n_req}\n")
    ranked = plan(m, hw, w)
    print(f"{'rank':4s} {'config':52s} {'pred TPS':>9s} {'TTFT ms':>9s}")
    for i, (tps, cfg, r) in enumerate(ranked):
        print(f"{i+1:4d} {cfg.label + ' ' + cfg.short():52s} {tps:9.0f} {r['ttft_ms']:9.0f}")
    # SAFE recommendation (never-slower-than-baseline guard)
    scfg, sr, dev = plan_safe(m, hw, w)
    base = uniform_tp_baseline(m, hw)
    note = (f"deviates from baseline (predicted >{int(SAFE_MARGIN*100)}% over it)"
            if dev else "= uniform-TP baseline (not confident enough to deviate)")
    print(f"\nSAFE recommendation (margin {SAFE_MARGIN:.0%}): {scfg.label} {scfg.short()}"
          f"  pred {sr.get('tps',0):.0f} tok/s  [{note}]")
    if base is not None:
        print(f"  baseline (uniform TP{hw.world}): measure both & serve faster for an absolute never-slower guarantee.")


if __name__ == "__main__":
    main()
