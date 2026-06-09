"""NetworkProfile catalog from interconnect specs.

NCCL AllReduce on a ring topology achieves bus bandwidth ≈ (n-1)/n × link_bw.
For paper purposes we use the link's *achievable* AR busbw (typically 50-80%
of the raw link bandwidth due to NCCL overhead + protocol framing).

Latencies are NCCL kernel launch + first-byte-out; calibrated against our
real cluster: intra-node 4-rank AR floor ≈ 20 μs, cross-node 8-rank ≈ 1 ms.
"""

from __future__ import annotations

from .cost_model import NetworkProfile


# ---- Intra-node (TP group lives on one node) ----

# Hopper NVLink 4 (H100, H200) — 900 GB/s aggregate per GPU, achievable AR busbw ~250-300 GB/s for ring on 8 ranks
NVLINK4 = NetworkProfile(
    intra_allreduce_busbw_GBs=300.0,
    cross_allreduce_busbw_GBs=40.0,    # filled by inter-node link if used (overridden in cluster spec)
    intra_allreduce_latency_ms=0.005,  # 5 μs
    cross_allreduce_latency_ms=0.002,  # IB-tier latency
    inter_node_p2p_GBs=40.0,           # default to IB-400
    intra_node_p2p_GBs=300.0,
)

# Ampere NVLink 3 (A100) — 600 GB/s aggregate, ~150 GB/s AR busbw for 8 ranks ring
NVLINK3 = NetworkProfile(
    intra_allreduce_busbw_GBs=150.0,
    cross_allreduce_busbw_GBs=40.0,
    intra_allreduce_latency_ms=0.008,
    cross_allreduce_latency_ms=0.002,
    inter_node_p2p_GBs=40.0,
    intra_node_p2p_GBs=150.0,
)

# Volta NVLink 2 (V100) — 300 GB/s aggregate, ~75 GB/s AR busbw
NVLINK2 = NetworkProfile(
    intra_allreduce_busbw_GBs=75.0,
    cross_allreduce_busbw_GBs=20.0,
    intra_allreduce_latency_ms=0.010,
    cross_allreduce_latency_ms=0.005,
    inter_node_p2p_GBs=20.0,
    intra_node_p2p_GBs=75.0,
)

# PCIe Gen5 ×16 — 128 GB/s raw, ~50-60 GB/s AR busbw (no peer-to-peer atomics, slower than NVLink)
PCIE_GEN5 = NetworkProfile(
    intra_allreduce_busbw_GBs=50.0,
    cross_allreduce_busbw_GBs=20.0,
    intra_allreduce_latency_ms=0.020,
    cross_allreduce_latency_ms=0.005,
    inter_node_p2p_GBs=20.0,
    intra_node_p2p_GBs=50.0,
)

# PCIe Gen4 ×16 — 64 GB/s raw, ~30 GB/s AR busbw
PCIE_GEN4 = NetworkProfile(
    intra_allreduce_busbw_GBs=30.0,
    cross_allreduce_busbw_GBs=20.0,
    intra_allreduce_latency_ms=0.025,
    cross_allreduce_latency_ms=0.005,
    inter_node_p2p_GBs=20.0,
    intra_node_p2p_GBs=30.0,
)


# ---- Cross-node interconnects (PP send / cross-node TP AR) ----
# These are typically used to OVERRIDE the `cross_*` + `inter_node_*` fields of
# the intra-node profile when building a cluster spec.

def override_cross(intra: NetworkProfile, *, cross_busbw: float, cross_latency_ms: float,
                   p2p_GBs: float) -> NetworkProfile:
    """Return a new NetworkProfile keeping intra-node fields but overriding cross."""
    return NetworkProfile(
        intra_allreduce_busbw_GBs=intra.intra_allreduce_busbw_GBs,
        intra_allreduce_latency_ms=intra.intra_allreduce_latency_ms,
        intra_node_p2p_GBs=intra.intra_node_p2p_GBs,
        cross_allreduce_busbw_GBs=cross_busbw,
        cross_allreduce_latency_ms=cross_latency_ms,
        inter_node_p2p_GBs=p2p_GBs,
    )


# Cross-node link specs (single GPU pair; NCCL multi-GPU ring scales differently)
CROSS_IB_NDR_400 = {"cross_busbw": 40.0, "cross_latency_ms": 0.002, "p2p_GBs": 40.0}   # InfiniBand NDR 400 Gb/s
CROSS_IB_HDR_200 = {"cross_busbw": 20.0, "cross_latency_ms": 0.002, "p2p_GBs": 20.0}   # InfiniBand HDR 200 Gb/s
CROSS_ETH_100G   = {"cross_busbw": 10.0, "cross_latency_ms": 0.020, "p2p_GBs": 10.0}
CROSS_ETH_25G    = {"cross_busbw": 2.5,  "cross_latency_ms": 0.050, "p2p_GBs": 2.5}
CROSS_ETH_10G    = {"cross_busbw": 1.1,  "cross_latency_ms": 1.000, "p2p_GBs": 1.1}    # our real cluster


NETWORK_CATALOG = {
    "NVLINK4":   NVLINK4,
    "NVLINK3":   NVLINK3,
    "NVLINK2":   NVLINK2,
    "PCIE_GEN5": PCIE_GEN5,
    "PCIE_GEN4": PCIE_GEN4,
}


CROSS_NODE_CATALOG = {
    "IB-NDR-400": CROSS_IB_NDR_400,
    "IB-HDR-200": CROSS_IB_HDR_200,
    "ETH-100G":   CROSS_ETH_100G,
    "ETH-25G":    CROSS_ETH_25G,
    "ETH-10G":    CROSS_ETH_10G,
}


def build_network(intra_name: str, cross_name: str | None = None) -> NetworkProfile:
    """Convenience: combine an intra-node profile with a cross-node override."""
    intra = NETWORK_CATALOG[intra_name]
    if cross_name is None:
        return intra
    cross = CROSS_NODE_CATALOG[cross_name]
    return override_cross(intra, **cross)
