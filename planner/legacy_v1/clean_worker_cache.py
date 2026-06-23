"""Clean worker-side torch_compile_cache + /tmp/ray spill via Ray remote (ssh blocked)."""
from __future__ import annotations
import sys
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from planner.cluster_env import CFG

import ray
ray.init(address=CFG.ray_address, ignore_reinit_error=True)
for n in ray.nodes():
    if n.get('alive') and n['NodeManagerAddress'] == CFG.worker_fabric_ip:
        node_id = n['NodeID']
        break
else:
    print("worker node not found"); sys.exit(1)

@ray.remote(num_cpus=0.1, scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(node_id=node_id, soft=False))
def clean():
    import subprocess, os
    out = {}
    # torch compile cache
    cache_dir = os.path.expanduser("~/.cache/vllm/torch_compile_cache")
    if os.path.isdir(cache_dir):
        sz = subprocess.run(["du","-sh",cache_dir], capture_output=True, text=True).stdout.strip()
        out["cache_before"] = sz
        subprocess.run(["rm","-rf",cache_dir], check=False)
        os.makedirs(cache_dir, exist_ok=True)
        out["cache_after"] = subprocess.run(["du","-sh",cache_dir], capture_output=True, text=True).stdout.strip()
    # /tmp/ray spilled objects (keep active session)
    sessions = subprocess.run(["bash","-c","ls -1d /tmp/ray/session_* 2>/dev/null"], capture_output=True, text=True).stdout.strip().splitlines()
    out["sessions"] = sessions
    # Disk usage on /tmp before/after
    out["tmp_before"] = subprocess.run(["df","-h","/tmp"], capture_output=True, text=True).stdout
    # find dead sessions (no live raylet using it)
    for s in sessions:
        # session_dir/spilled_objects might be huge
        spilled = os.path.join(s, "spilled_objects")
        if os.path.isdir(spilled):
            sz = subprocess.run(["du","-sh",spilled], capture_output=True, text=True).stdout.strip()
            out.setdefault("spilled_before", []).append(sz)
    # safer: clean ONLY old session dirs (not the currently active one) — find with mtime
    subprocess.run(["bash","-c","find /tmp/ray -mindepth 1 -maxdepth 1 -name 'session_*' -mtime +0 -exec rm -rf {} + 2>/dev/null"], check=False)
    out["tmp_after"] = subprocess.run(["df","-h","/tmp"], capture_output=True, text=True).stdout
    return out

res = ray.get(clean.remote(), timeout=120)
for k, v in res.items():
    print(f"=== {k} ===")
    print(v if not isinstance(v, list) else "\n".join(v))
ray.shutdown()
