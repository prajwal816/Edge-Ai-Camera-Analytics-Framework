from __future__ import annotations

import asyncio
import os
import statistics
import time

import httpx
import pytest

pytestmark = pytest.mark.asyncio


async def _run_burst(base_url: str, concurrent: int, tensor: list[float]) -> tuple[float, list[float]]:
    async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as client:

        async def one() -> float:
            t0 = time.perf_counter()
            r = await client.post("/infer", json={"tensor": tensor, "channels": 3, "height": 224, "width": 224})
            r.raise_for_status()
            return (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        latencies = await asyncio.gather(*[one() for _ in range(concurrent)])
        elapsed = time.perf_counter() - t0
        return elapsed, latencies


@pytest.mark.skipif(not os.environ.get("EDGE_LOAD_TEST_URL"), reason="set EDGE_LOAD_TEST_URL to run against live server")
async def test_concurrent_burst_against_live_server() -> None:
    url = os.environ["EDGE_LOAD_TEST_URL"].rstrip("/")
    tensor = [0.01] * (3 * 224 * 224)
    concurrent = int(os.environ.get("EDGE_LOAD_CONCURRENCY", "50"))
    elapsed, latencies = await _run_burst(url, concurrent, tensor)
    rps = concurrent / max(elapsed, 1e-9)
    assert rps > 0.5
    assert statistics.median(latencies) < 60_000.0
