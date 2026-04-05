"""
Compare baseline (single-threaded C++ path) vs optimized (batched scheduler + workers).

Usage (from repo root, API running on localhost:8000):
  python benchmarks/run_benchmark.py --url http://127.0.0.1:8000

Or standalone Python engine (no server):
  python benchmarks/run_benchmark.py --standalone
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "src" / "python"))

from orchestration.engine_client import EngineClient  # noqa: E402
from utils.config import load_config  # noqa: E402


def _tensor() -> tuple[list[float], int, int, int]:
    """Smaller spatial size so the benchmark highlights batching/scheduling, not raw matmul time."""
    h = int(os.environ.get("EDGE_BENCH_H", "96"))
    w = int(os.environ.get("EDGE_BENCH_W", "96"))
    c = 3
    return [0.03] * (c * h * w), c, h, w


async def _burst_http(base_url: str, concurrent: int, requests: int) -> dict[str, Any]:
    tensor, c, h, w = _tensor()
    sem = asyncio.Semaphore(concurrent)

    async with httpx.AsyncClient(base_url=base_url, timeout=300.0) as client:
        latencies: list[float] = []

        async def one() -> None:
            async with sem:
                t0 = time.perf_counter()
                r = await client.post("/infer", json={"tensor": tensor, "channels": c, "height": h, "width": w})
                r.raise_for_status()
                latencies.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        await asyncio.gather(*[one() for _ in range(requests)])
        elapsed = time.perf_counter() - t0
        rps = requests / max(elapsed, 1e-9)
        lat_sorted = sorted(latencies)
        p95 = lat_sorted[int(0.95 * (len(lat_sorted) - 1))] if lat_sorted else 0.0
        return {
            "mode": "http",
            "requests": requests,
            "concurrency": concurrent,
            "elapsed_sec": elapsed,
            "throughput_rps": rps,
            "latency_p50_ms": statistics.median(latencies),
            "latency_p95_ms": p95,
        }


async def _standalone_pair() -> dict[str, Any]:
    os.environ["EDGE_FORCE_PYTHON_ENGINE"] = "1"
    cfg = load_config()

    async def run_mode(baseline: bool) -> float:
        cfg.raw.setdefault("inference", {})
        cfg.raw["inference"]["baseline_mode"] = baseline
        n = int(cfg.benchmarks.get("concurrent_requests", 120))
        # Baseline models a single worker edge device; optimized allows a wide client pool to stress batching.
        exec_workers = 1 if baseline else min(32, max(8, n // 4))
        c = EngineClient(cfg=cfg, executor=ThreadPoolExecutor(max_workers=exec_workers))
        tensor, ch, h, w = _tensor()

        async def job() -> None:
            await c.infer_async(tensor, ch, h, w, False)

        t0 = time.perf_counter()
        await asyncio.gather(*[job() for _ in range(n)])
        elapsed = time.perf_counter() - t0
        c.shutdown()
        return n / max(elapsed, 1e-9)

    base_rps = await run_mode(True)
    opt_rps = await run_mode(False)
    return {
        "mode": "standalone_python_sim",
        "baseline_rps": base_rps,
        "optimized_rps": opt_rps,
        "speedup": opt_rps / max(base_rps, 1e-9),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=os.environ.get("EDGE_BENCHMARK_URL", "http://127.0.0.1:8000"))
    p.add_argument("--concurrent", type=int, default=200)
    p.add_argument("--requests", type=int, default=400)
    p.add_argument("--standalone", action="store_true")
    args = p.parse_args()

    async def _amain() -> dict[str, Any]:
        if args.standalone:
            return await _standalone_pair()
        return await _burst_http(args.url.rstrip("/"), args.concurrent, args.requests)

    result = asyncio.run(_amain())
    out_dir = ROOT / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "last_run.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
