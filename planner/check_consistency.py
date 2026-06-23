"""Planner self-consistency / invariant suite (정합성 검증).

These are properties the cost model MUST satisfy for its logic to be sound —
independent of any measured data. A failure here is a logic bug, not a
calibration gap. Run: python planner/check_consistency.py

Invariants
----------
 1 finite        predict() returns finite, positive tps/times for feasible cfgs
 2 tps_mono      TPS non-decreasing in n_req (until memory-infeasible)
 3 cycle_mono    decode t_cycle non-decreasing in n_req
 4 tp8_saturate  pp=1 throughput is sub-linear in n_req (a ceiling exists)
 5 homog_collapse on a HOMOGENEOUS cluster the optimal non-uniform TP split == uniform,
                 and uniform is not beaten by any bias (no spurious hetero gain)
 6 tp_split_opt  closed-form TP FFN-bias is within ε of a brute-force search
 7 layer_split_opt closed-form PP layer split is within ε of brute-force
 8 ar_sane       AR(cross) ≥ AR(intra); AR ↑ in msg and in Nt; intra branch continuous
 9 feas_mono     infeasible at n_req ⇒ infeasible at every larger n_req
10 bias_favors_fast hetero optimal FFN split gives Blackwell ranks ≥ Ada ranks
11 crossover_mono argmax topology over n_req flips at most once (TP-heavy → PP-heavy)
12 layout_sane   plan() at 1+1/2+2/4+4 returns a feasible, positive-tps champion
13 embed_charge  embedding mass charged per stage == physical (no double-count)
14 no_zero_layer optimal_layer_split gives every stage ≥1 layer, sums to n_layers
15 prefill_no_sawtooth  T_prefill has no chunk-boundary jump (partial last chunk)
16 factorization_complete  plan() enumerates non-power-of-2 PP (3+3 pp∈{3,6})
17 tps_mono_odd  TPS monotone in n_req at non-divisible points (no dropped remainder)
"""
from __future__ import annotations
import dataclasses, itertools, math, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
import planner.perf_planner as P

HW = P.load_hardware()
MODELS = ["8b", "70b", "mistral123b", "opt30b", "qwen32b"]
NREQS = [8, 16, 32, 64, 96]
WLS = {"balanced": P.Workload(512, 256, 0),
       "decode_heavy": P.Workload(128, 512, 0),
       "prefill_heavy": P.Workload(1024, 128, 0)}


def wl(name, n):
    b = WLS[name]; return P.Workload(b.in_len, b.out_len, n)


def homogeneous(hw, which="blackwell"):
    """Same total GPU count, but every node is the SAME gpu type."""
    g = hw.nodes[0][0] if which == "blackwell" else hw.nodes[-1][0]
    return dataclasses.replace(hw, nodes=tuple((g, c) for _, c in hw.nodes))


def relayout(hw, n):
    """n+n layout: first node n Blackwell, second node n Ada."""
    bw = hw.nodes[0][0]; ada = hw.nodes[-1][0]
    return dataclasses.replace(hw, nodes=((bw, n), (ada, n)))


def uniform_cfg(m, tp, pp):
    base = m.n_layers // pp; rem = m.n_layers - base * pp
    ls = [base + (1 if s < rem else 0) for s in range(pp)]
    return P.Config(tp, pp, ls, [m.ffn_dim // tp] * tp, [m.n_q // tp] * tp,
                    [max(1, m.n_kv // tp)] * tp, label=f"TP{tp}PP{pp}u")


# ---------------------------------------------------------------- checks
def c1_finite():
    v = []
    for mk in MODELS:
        m = P.MODELS[mk]
        for tp, pp in [(8, 1), (4, 2), (2, 4)]:
            if m.n_q % tp or m.n_layers < pp: continue
            cfg = uniform_cfg(m, tp, pp)
            for wn in WLS:
                for n in NREQS:
                    r = P.predict(m, HW, wl(wn, n), cfg, overlap=(pp > 1))
                    if not r["feasible"]: continue
                    for k in ("tps", "t_decode_s", "t_prefill_s", "t_cycle_ms"):
                        x = r.get(k, 0)
                        if not math.isfinite(x) or x < 0:
                            v.append(f"{mk} {cfg.label} {wn} n{n}: {k}={x}")
    return v


def c2_tps_mono():
    v = []
    for mk in MODELS:
        m = P.MODELS[mk]
        for tp, pp in [(8, 1), (4, 2), (2, 4)]:
            if m.n_q % tp or m.n_layers < pp: continue
            cfg = uniform_cfg(m, tp, pp)
            for wn in WLS:
                prev = None; prev_n = None
                for n in NREQS:
                    r = P.predict(m, HW, wl(wn, n), cfg, overlap=(pp > 1))
                    if not r["feasible"]: continue
                    t = r["tps"]
                    if prev is not None and t < prev * 0.999:
                        v.append(f"{mk} {cfg.label} {wn}: tps n{prev_n}={prev:.0f} > n{n}={t:.0f}")
                    prev, prev_n = t, n
    return v


def c3_cycle_mono():
    v = []
    for mk in MODELS:
        m = P.MODELS[mk]
        for tp, pp in [(8, 1), (4, 2)]:
            if m.n_q % tp or m.n_layers < pp: continue
            cfg = uniform_cfg(m, tp, pp)
            prev = None; prev_n = None
            for n in NREQS:
                r = P.predict(m, HW, wl("balanced", n), cfg, overlap=(pp > 1))
                if not r["feasible"]: continue
                t = r["t_cycle_ms"]
                if prev is not None and t < prev * 0.999:
                    v.append(f"{mk} {cfg.label}: cycle n{prev_n}={prev:.2f} > n{n}={t:.2f}")
                prev, prev_n = t, n
    return v


def c4_tp8_saturate():
    v = []
    for mk in MODELS:
        m = P.MODELS[mk]
        if m.n_q % 8: continue
        cfg = uniform_cfg(m, 8, 1)
        r8 = P.predict(m, HW, wl("balanced", 8), cfg)
        r96 = P.predict(m, HW, wl("balanced", 96), cfg)
        if not (r8["feasible"] and r96["feasible"]): continue
        # sub-linear: 12× the load must NOT give ≥12× the throughput
        ratio = r96["tps"] / r8["tps"]
        if ratio > 12.0 * 0.999:
            v.append(f"{mk} TP8: tps grew {ratio:.1f}× for 12× load (not saturating)")
    return v


def c5_homog_collapse():
    v = []
    for which in ("blackwell", "ada"):
        hwh = homogeneous(HW, which)
        for mk in MODELS:
            m = P.MODELS[mk]
            if m.n_q % 8: continue
            dw = P.decode_weight_of(wl("balanced", 64))
            ffn, heads, kv = P.optimal_tp_splits(m, hwh, wl("balanced", 64), 8, dw)
            if len(set(ffn)) != 1:
                v.append(f"{which} {mk}: optimal FFN not uniform on homog HW: {ffn}")
            if len(set(heads)) != 1:
                v.append(f"{which} {mk}: optimal heads not uniform on homog HW: {heads}")
            # uniform must not be beaten by the optimal-bias on homogeneous HW
            uni = uniform_cfg(m, 8, 1)
            bias = P.Config(8, 1, [m.n_layers], ffn, heads, kv)
            ru = P.predict(m, hwh, wl("balanced", 64), uni)
            rb = P.predict(m, hwh, wl("balanced", 64), bias)
            if ru["feasible"] and rb["feasible"] and rb["tps"] > ru["tps"] * 1.002:
                v.append(f"{which} {mk}: bias beats uniform on homog HW "
                         f"{rb['tps']:.0f}>{ru['tps']:.0f}")
    return v


def c6_tp_split_opt():
    """Brute-force the Blackwell-bias axis (all B ranks=x, all A ranks share rest)
    and confirm the closed-form is within 2% of the best on that axis."""
    v = []
    for mk in MODELS:
        m = P.MODELS[mk]
        if m.n_q % 8: continue
        w = wl("balanced", 64); dw = P.decode_weight_of(w)
        ffn, heads, kv = P.optimal_tp_splits(m, HW, w, 8, dw)
        cf_opt = P.Config(8, 1, [m.n_layers], ffn, heads, kv)
        r_opt = P.predict(m, HW, w, cf_opt)
        if not r_opt["feasible"]: continue
        best = r_opt["tps"]; best_lab = "closed-form"
        # sweep FFN bias (keep heads uniform to isolate the FFN axis)
        hu = [m.n_q // 8] * 8; ku = [max(1, m.n_kv // 8)] * 8
        for xb in range(m.ffn_dim // 8, m.ffn_dim // 8 * 2 + 1, 256):
            rest = m.ffn_dim - 4 * xb
            if rest < 4 * 128: break
            xa = rest // 4
            ff = [xb] * 4 + [xa] * 4
            if sum(ff) != m.ffn_dim:
                ff[0] += m.ffn_dim - sum(ff)
            r = P.predict(m, HW, w, P.Config(8, 1, [m.n_layers], ff, hu, ku))
            if r["feasible"] and r["tps"] > best:
                best = r["tps"]; best_lab = f"ffn B={xb}"
        if best > r_opt["tps"] * 1.02:
            v.append(f"{mk}: brute {best:.0f} ({best_lab}) > closed-form "
                     f"{r_opt['tps']:.0f} by {(best/r_opt['tps']-1)*100:.1f}%")
    return v


def c7_layer_split_opt():
    """The planner's recommended TP4PP2 layer split (closed-form + the ±2
    neighborhood plan() searches) must be within 2% of a brute-force 1-D scan."""
    v = []
    for mk in MODELS:
        m = P.MODELS[mk]
        if m.n_q % 4: continue
        w = wl("balanced", 64)
        ffn_u = [m.ffn_dim // 4] * 4; hu = [m.n_q // 4] * 4; ku = [max(1, m.n_kv // 4)] * 4
        # what the planner actually recommends among pp=2 configs
        scored = P.plan(m, HW, w, top_k=50)
        pp2 = [(tps, cfg) for tps, cfg, _ in scored if cfg.pp == 2]
        if not pp2: continue
        plan_tps, plan_cfg = max(pp2, key=lambda x: x[0])
        # brute-force the full 1-D layer split
        best = 0.0; best_split = None
        for l0 in range(1, m.n_layers):
            ls = [l0, m.n_layers - l0]
            r = P.predict(m, HW, w, P.Config(4, 2, ls, ffn_u, hu, ku), overlap=True)
            if r["feasible"] and r["tps"] > best:
                best = r["tps"]; best_split = tuple(ls)
        if best > plan_tps * 1.02:
            v.append(f"{mk}: brute {best_split} {best:.0f} > planner "
                     f"{tuple(plan_cfg.layer_split)} {plan_tps:.0f} by "
                     f"{(best/plan_tps-1)*100:.1f}%")
    return v


def c8_ar_sane():
    v = []
    bw, ada = HW.nodes[0][0], HW.nodes[-1][0]
    msg = 1e6
    intra = P.t_allreduce_ms(msg, [0, 1, 2, 3], HW)          # all head node
    cross = P.t_allreduce_ms(msg, [0, 1, 2, 3, 4, 5, 6, 7], HW)  # 4+4
    if cross < intra:
        v.append(f"AR cross {cross:.3f} < intra {intra:.3f} (same msg)")
    # monotone in msg
    if P.t_allreduce_ms(2e6, list(range(8)), HW) < P.t_allreduce_ms(1e6, list(range(8)), HW):
        v.append("AR not monotone in msg")
    # monotone in group size (intra)
    if P.t_allreduce_ms(msg, [0, 1, 2, 3], HW) < P.t_allreduce_ms(msg, [0, 1], HW):
        v.append("AR not monotone in Nt (intra)")
    # zero for singleton
    if P.t_allreduce_ms(msg, [0], HW) != 0.0:
        v.append("AR(singleton) != 0")
    return v


def c9_feas_mono():
    v = []
    for mk in MODELS:
        m = P.MODELS[mk]
        for tp, pp in [(8, 1), (4, 2), (2, 4)]:
            if m.n_q % tp or m.n_layers < pp: continue
            cfg = uniform_cfg(m, tp, pp)
            infeas_at = None
            for n in [8, 16, 32, 64, 96, 100]:
                feas, _ = P.mem_feasible(m, HW, wl("balanced", n), cfg)
                if not feas and infeas_at is None:
                    infeas_at = n
                if feas and infeas_at is not None:
                    v.append(f"{mk} {cfg.label}: feasible at n{n} but infeasible at n{infeas_at}")
    return v


def c10_bias_favors_fast():
    v = []
    for mk in MODELS:
        m = P.MODELS[mk]
        if m.n_q % 8: continue
        w = wl("balanced", 64); dw = P.decode_weight_of(w)
        ffn, heads, kv = P.optimal_tp_splits(m, HW, w, 8, dw)
        # ranks 0-3 = Blackwell (fast), 4-7 = Ada (slow)
        if ffn[0] < ffn[4]:
            v.append(f"{mk}: FFN gives slow Ada more than fast Blackwell: {ffn}")
        if heads[0] < heads[4]:
            v.append(f"{mk}: heads give slow Ada more than Blackwell: {heads}")
    return v


def c11_crossover_mono():
    v = []
    for mk in MODELS:
        m = P.MODELS[mk]
        if m.n_q % 8: continue
        picks = []
        for n in NREQS:
            w = wl("balanced", n)
            scored = P.plan(m, HW, w, top_k=1)
            if scored:
                picks.append(scored[0][1].pp)   # pp of champion
        # pp should be non-decreasing in n (TP-heavy pp=1 → PP-heavy pp>1)
        for i in range(1, len(picks)):
            if picks[i] < picks[i-1]:
                v.append(f"{mk}: champion pp non-monotone over n: {picks}")
                break
    return v


def c12_layout_sane():
    v = []
    for n in (1, 2, 4):
        hw = relayout(HW, n)
        for mk in MODELS:
            m = P.MODELS[mk]
            scored = P.plan(m, hw, wl("balanced", 32), top_k=1)
            if not scored:
                # may be genuinely infeasible (e.g. 123b on 1+1) — only flag if it
                # SHOULD fit: total weight ≤ total cap
                tot_cap = sum(c * g.mem_gb for g, c in hw.nodes) * HW.mem_util * 1e9
                if m.params_b * 1e9 * 2 < tot_cap * 0.5:
                    v.append(f"{n}+{n} {mk}: no feasible config but model should fit")
                continue
            tps, cfg, r = scored[0]
            if not (math.isfinite(tps) and tps > 0):
                v.append(f"{n}+{n} {mk}: champion tps={tps}")
    return v


# --- regression guards for the logic bugs the adversarial audit found ---
def c13_embed_charge():
    """Embedding mass charged across PP stages must equal the physical amount:
    pp=1 → p_embed (input+lm_head on one stage); pp>1 → V·h on stage 0 (input)
    + V·h on the last stage (lm_head), nothing on middle stages. (Audit: untied
    models were double-charged 2·V·h on each end stage.)"""
    v = []
    for mk in MODELS:
        m = P.MODELS[mk]; vh = m.vocab * m.hidden
        for pp in (1, 2, 4, 8):
            if m.n_layers < pp: continue
            total = sum(P.embed_on_stage(m, pp, s) for s in range(pp))
            expect = m.p_embed if pp == 1 else 2 * vh
            if abs(total - expect) > 1:
                v.append(f"{mk} pp{pp}: embed total {total:.3e} != physical {expect:.3e}")
            if pp > 2 and any(P.embed_on_stage(m, pp, s) != 0 for s in range(1, pp - 1)):
                v.append(f"{mk} pp{pp}: a middle stage carries embedding")
    return v


def c14_no_zero_layer():
    """optimal_layer_split must give every stage ≥1 layer and sum to n_layers,
    across all layouts 1+1..4+4 (audit: a stage could get 0 layers)."""
    v = []
    for n in (1, 2, 3, 4):
        hw = relayout(HW, n); world = 2 * n
        for mk in MODELS:
            m = P.MODELS[mk]
            for pp in [d for d in range(2, world + 1) if world % d == 0]:
                tp = world // pp
                if m.n_q % tp or m.n_layers < pp: continue
                ls = P.optimal_layer_split(m, hw, wl("balanced", 64), tp, pp,
                                           P.decode_weight_of(wl("balanced", 64)))
                if min(ls) < 1 or sum(ls) != m.n_layers or len(ls) != pp:
                    v.append(f"{n}+{n} {mk} tp{tp}pp{pp}: bad split {ls} (sum {sum(ls)})")
    return v


def c15_prefill_no_sawtooth():
    """t_prefill must not jump at a chunk boundary: crossing total_in past a
    multiple of T_CHUNK should change t_prefill ~proportionally, not ~2×.
    (Audit: the partial last chunk was charged a full T_CHUNK.)"""
    v = []
    for mk in ("8b", "70b"):
        m = P.MODELS[mk]; cfg = uniform_cfg(m, 8, 1)
        for n_lo in (16, 32, 48, 64):   # in=512 ⇒ boundaries at multiples of 16
            r_lo = P.predict(m, HW, P.Workload(512, 256, n_lo), cfg)
            r_hi = P.predict(m, HW, P.Workload(512, 256, n_lo + 1), cfg)
            if not (r_lo["feasible"] and r_hi["feasible"]): continue
            jump = r_hi["t_prefill_s"] / max(r_lo["t_prefill_s"], 1e-12)
            if jump > 1.25:             # (n+1)/n ≈ 1.0x expected; 2x ⇒ sawtooth
                v.append(f"{mk} n{n_lo}->{n_lo+1}: t_prefill jumped {jump:.2f}x")
    return v


def c16_factorization_complete():
    """plan() must enumerate non-power-of-2 PP factorizations: world=6 (3+3)
    must yield at least one pp∈{3,6} config for a divisible model (mistral
    n_q=96). (Audit: 'for pp in [2,4,8]' skipped pp=3,6.)"""
    v = []
    m = P.MODELS["mistral123b"]
    hw6 = relayout(HW, 3)               # 3+3, world=6
    got = {(c.tp, c.pp) for _, c, _ in P.plan(m, hw6, wl("balanced", 16), top_k=50)}
    if not any(pp in (3, 6) for _, pp in got):
        v.append(f"3+3 mistral: no pp in {{3,6}} enumerated; got {sorted(got)}")
    return v


def c17_tps_mono_odd():
    """TPS monotone in n_req even at non-divisible points (n%pp≠0): the decode
    model must not drop the remainder. (Audit: mb=n_req//pp dropped n%pp.)"""
    v = []
    for mk in ("8b", "70b", "mistral123b"):
        m = P.MODELS[mk]
        for tp, pp in [(4, 2), (2, 4)]:
            if m.n_q % tp or m.n_layers < pp: continue
            cfg = uniform_cfg(m, tp, pp)
            prev = None; prev_n = None
            for n in [30, 31, 32, 33, 62, 63, 64, 65]:
                r = P.predict(m, HW, wl("balanced", n), cfg, overlap=True)
                if not r["feasible"]: continue
                if prev is not None and r["tps"] < prev * 0.999:
                    v.append(f"{mk} {cfg.label}: tps n{prev_n}={prev:.0f} > n{n}={r['tps']:.0f}")
                prev, prev_n = r["tps"], n
    return v


CHECKS = [
    ("1  finite", c1_finite), ("2  tps_mono", c2_tps_mono),
    ("3  cycle_mono", c3_cycle_mono), ("4  tp8_saturate", c4_tp8_saturate),
    ("5  homog_collapse", c5_homog_collapse), ("6  tp_split_opt", c6_tp_split_opt),
    ("7  layer_split_opt", c7_layer_split_opt), ("8  ar_sane", c8_ar_sane),
    ("9  feas_mono", c9_feas_mono), ("10 bias_favors_fast", c10_bias_favors_fast),
    ("11 crossover_mono", c11_crossover_mono), ("12 layout_sane", c12_layout_sane),
    ("13 embed_charge", c13_embed_charge), ("14 no_zero_layer", c14_no_zero_layer),
    ("15 prefill_no_sawtooth", c15_prefill_no_sawtooth),
    ("16 factorization_complete", c16_factorization_complete),
    ("17 tps_mono_odd", c17_tps_mono_odd),
]


def main():
    npass = 0
    print("planner self-consistency suite\n" + "=" * 60)
    for name, fn in CHECKS:
        try:
            viol = fn()
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            continue
        if not viol:
            print(f"  PASS  {name}"); npass += 1
        else:
            print(f"  FAIL  {name}  ({len(viol)} violations)")
            for x in viol[:6]:
                print(f"          - {x}")
            if len(viol) > 6:
                print(f"          ... +{len(viol)-6} more")
    print("=" * 60)
    print(f"{npass}/{len(CHECKS)} invariants hold")
    return 0 if npass == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
