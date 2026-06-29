"""Mixed-traffic benchmark: send n_req requests whose (input_len, output_len) are
each sampled from a set of diverse context shapes — i.e. heterogeneous request
shapes MIXED in one concurrent stream (realistic serving), not a fixed shape per
run. Deterministic per --seed so baseline and planner-pick configs see the SAME
request mix. Measures aggregate throughput (total output tokens / makespan).

Usage: python perf/mixed_bench.py --base-url http://127.0.0.1:PORT/v1 --model M --n-req 64 --seed 0
"""
import argparse, concurrent.futures as cf, json, os, random, statistics, time, urllib.request

WORD = ("the analysis of large language model inference systems on heterogeneous "
        "GPU clusters with tensor and pipeline parallelism ").split()
# diverse (input_len, output_len) shapes; total <= 4096 to fit max-model-len.
# MIXED_MAX_TOTAL caps (in+out) so short-context models (e.g. OPT-30B = 2048) fit.
_ALL_SHAPES = [(128, 128), (128, 1024), (256, 512), (512, 256), (1024, 128),
               (2048, 256), (512, 1024), (2048, 512), (3072, 512),
               (1024, 512), (1536, 256), (256, 1024)]


def build_prompt(in_len):
    w = []
    while len(w) < int(in_len / 1.3):
        w.extend(WORD)
    return " ".join(w[:int(in_len / 1.3)])


def one(base, model, in_len, out_len):
    body = json.dumps({"model": model, "prompt": build_prompt(in_len),
                       "max_tokens": out_len, "ignore_eos": True, "temperature": 0.0}).encode()
    req = urllib.request.Request(base + "/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        d = json.loads(r.read())
    ct = d.get("usage", {}).get("completion_tokens", out_len)
    return ct, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-req", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-total", type=int,
                    default=int(os.environ.get("MIXED_MAX_TOTAL", "4096")),
                    help="cap (in+out) so short-context models fit (e.g. OPT-30B=2048)")
    ap.add_argument("--summary-csv", default="")
    a = ap.parse_args()
    shapes = [s for s in _ALL_SHAPES if s[0] + s[1] <= a.max_total]
    random.seed(a.seed)
    reqs = [random.choice(shapes) for _ in range(a.n_req)]
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=a.n_req) as ex:
        res = [f.result() for f in [ex.submit(one, a.base_url, a.model, i, o) for i, o in reqs]]
    wall = time.time() - t0
    tot_out = sum(c for c, _ in res)
    lat = [l for _, l in res]
    tput = tot_out / wall if wall > 0 else 0
    mean_in = sum(i for i, _ in reqs) / len(reqs)
    mean_out = sum(o for _, o in reqs) / len(reqs)
    print(f"MIXED n_req={a.n_req} seed={a.seed}  shapes={len(shapes)} (max_total={a.max_total})")
    print(f"  total_output_tokens={tot_out}  wall_s={wall:.1f}  throughput_tok_s={tput:.1f}")
    print(f"  mean_in={mean_in:.0f}  mean_out={mean_out:.0f}")
    print(f"  req_latency_s mean={statistics.mean(lat):.1f} p50={statistics.median(lat):.1f} max={max(lat):.1f}")
    if a.summary_csv:
        with open(a.summary_csv, "w") as f:
            f.write("metric,value\n")
            f.write(f"total_wall_throughput_tok_s,{tput:.1f}\n")
            f.write(f"itl_ms_mean,{1000 * statistics.mean(lat) / max(1, mean_out):.1f}\n")
            f.write(f"mean_in,{mean_in:.0f}\nmean_out,{mean_out:.0f}\n")
    print(f"THROUGHPUT_TOK_S={tput:.1f}")


if __name__ == "__main__":
    main()
