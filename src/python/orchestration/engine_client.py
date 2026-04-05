from __future__ import annotations

import asyncio
import os
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import httpx
import numpy as np

from utils.config import AppConfig, repo_root
from utils.logging_config import get_logger
from utils.native_loader import load_edge_infer_native

log = get_logger("orchestration.engine_client")


@dataclass
class EngineMetrics:
    requests: int = 0
    total_latency_ms: float = 0.0
    last_gpu_log: str = ""

    def record(self, latency_ms: float, gpu_log: str) -> None:
        self.requests += 1
        self.total_latency_ms += latency_ms
        self.last_gpu_log = gpu_log

    @property
    def avg_latency_ms(self) -> float:
        if self.requests == 0:
            return 0.0
        return self.total_latency_ms / self.requests


class InferenceBackend(Protocol):
    def infer(self, tensor: list[float], channels: int, height: int, width: int, force_cpu: bool) -> list[float]: ...

    def batch_infer(
        self, tensors: Sequence[list[float]], channels: int, height: int, width: int, force_cpu: bool
    ) -> list[list[float]]: ...

    def snapshot(self) -> dict[str, Any]: ...

    def shutdown(self) -> None: ...


class CppBackend:
    def __init__(self, cfg: AppConfig) -> None:
        mod = load_edge_infer_native()
        if mod is None:
            raise RuntimeError("edge_infer_native not built; run CMake or set EDGE_NATIVE_MODULE")
        inf = cfg.inference
        manifest = repo_root() / str(inf.get("engine_path", "models/engine_manifest.json"))
        self._svc = mod.EdgeInferenceService()
        self._svc.configure(
            str(manifest),
            bool(inf.get("baseline_mode", False)),
            bool(inf.get("prefer_gpu", True)),
            True,
            int(inf.get("max_batch_size", 8)),
            float(inf.get("batch_timeout_ms", 4.0)),
            int(inf.get("worker_threads", 4)),
            float(inf.get("cpu_fallback_util_threshold", 0.92)),
        )

    def infer(self, tensor: list[float], channels: int, height: int, width: int, force_cpu: bool) -> list[float]:
        return list(self._svc.infer(tensor, channels, height, width, force_cpu))

    def batch_infer(
        self, tensors: Sequence[list[float]], channels: int, height: int, width: int, force_cpu: bool
    ) -> list[list[float]]:
        frames = [list(t) for t in tensors]
        out = self._svc.batch_infer(frames, channels, height, width, force_cpu)
        return [list(x) for x in out]

    def snapshot(self) -> dict[str, Any]:
        return dict(self._svc.snapshot())

    def shutdown(self) -> None:
        self._svc.shutdown()


class PythonSimBackend:
    """CPU-only stand-in when the native module is unavailable (dev / CI)."""

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._baseline = bool(cfg.inference.get("baseline_mode", False))
        self._c = int(os.environ.get("EDGE_SIM_CHANNELS", "3"))
        self._h = int(os.environ.get("EDGE_SIM_HEIGHT", "224"))
        self._w = int(os.environ.get("EDGE_SIM_WIDTH", "224"))
        self._od = 64
        self._batches = 0
        self._frames = 0
        self._submitted = 0
        self._batch_events = 0
        self._q: queue.Queue[tuple[list[float], int, int, int, bool, Future[list[float]]] | None] = queue.Queue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        if not self._baseline:
            n = max(1, int(cfg.inference.get("worker_threads", 4)))
            self._max_batch = max(1, int(cfg.inference.get("max_batch_size", 8)))
            self._max_wait_s = max(0.0005, float(cfg.inference.get("batch_timeout_ms", 4.0)) / 1000.0)
            for i in range(n):
                t = threading.Thread(target=self._worker_loop, name=f"py-sim-sched-{i}", daemon=True)
                t.start()
                self._threads.append(t)

    def _sleep_batch(self, batch: int, gpu: bool) -> None:
        if self._baseline:
            # Per-request penalty without cross-request batching (mirrors C++ baseline path).
            per = 0.00105 if gpu else 0.0024
            time.sleep(per * max(1, batch))
            return
        if gpu:
            time.sleep(0.00055 + 0.00004 * max(0, batch - 1))
        else:
            time.sleep(0.00135 + 0.00012 * max(0, batch - 1))

    def _run(self, nchw: list[float], batch: int, force_cpu: bool) -> list[float]:
        prefer_gpu = bool(self._cfg.inference.get("prefer_gpu", True))
        on_gpu = prefer_gpu and not force_cpu
        self._sleep_batch(batch, on_gpu)
        self._batches += 1
        self._frames += batch
        arr = np.asarray(nchw, dtype=np.float32).reshape(batch, -1)
        span = min(arr.shape[1], 4096)
        acc = arr[:, :span].sum(axis=1, keepdims=True)
        tile = np.repeat(acc, self._od, axis=1)
        out = np.tanh(tile * (1.0 / (span + 1.0))).astype(np.float32).reshape(-1)
        return out.tolist()

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                first = self._q.get(timeout=0.05)
            except queue.Empty:
                continue
            if first is None:
                break
            jobs: list[tuple[list[float], int, int, int, bool, Future[list[float]]]] = [first]
            deadline = time.perf_counter() + self._max_wait_s
            while len(jobs) < self._max_batch:
                try:
                    jobs.append(self._q.get_nowait())
                except queue.Empty:
                    if time.perf_counter() >= deadline:
                        break
                    time.sleep(0.00002)

            self._batch_events += 1
            batch = len(jobs)
            any_cpu = any(j[4] for j in jobs)
            concat: list[float] = []
            ch = jobs[0][1]
            h = jobs[0][2]
            w = jobs[0][3]
            for tensor, _, _, _, _, _ in jobs:
                concat.extend(tensor)
            flat = self._run(concat, batch, any_cpu)
            od = self._od
            for i, (_, _, _, _, _, fut) in enumerate(jobs):
                if fut.cancelled():
                    continue
                off = i * od
                fut.set_result(flat[off : off + od])

    def infer(self, tensor: list[float], channels: int, height: int, width: int, force_cpu: bool) -> list[float]:
        ch = channels or self._c
        h = height or self._h
        w = width or self._w
        if self._baseline:
            return self._infer_direct(list(tensor), ch, h, w, force_cpu)
        self._submitted += 1
        fut: Future[list[float]] = Future()
        self._q.put((list(tensor), ch, h, w, force_cpu, fut))
        return fut.result(timeout=120.0)

    def _infer_direct(self, tensor: list[float], ch: int, h: int, w: int, force_cpu: bool) -> list[float]:
        return self._run(tensor, 1, force_cpu)

    def batch_infer(
        self, tensors: Sequence[list[float]], channels: int, height: int, width: int, force_cpu: bool
    ) -> list[list[float]]:
        ch = channels or self._c
        h = height or self._h
        w = width or self._w
        if self._baseline:
            return [self._infer_direct(list(t), ch, h, w, force_cpu) for t in tensors]
        if not tensors:
            return []
        concat: list[float] = []
        for t in tensors:
            concat.extend(t)
        flat = self._run(concat, len(tensors), force_cpu)
        od = self._od
        return [flat[i * od : (i + 1) * od] for i in range(len(tensors))]

    def snapshot(self) -> dict[str, Any]:
        return {
            "gpu_log_line": "gpu_utilization_percent=42.0 gpu_mem_used_mb=900.0 (python sim)",
            "scheduler_submitted": self._submitted,
            "scheduler_batches": self._batch_events,
            "engine_batches": self._batches,
            "engine_frames": self._frames,
            "engine_gpu_path": self._batches,
            "engine_cpu_path": 0,
            "baseline_mode": self._baseline,
        }

    def shutdown(self) -> None:
        self._stop.set()
        for _ in self._threads:
            self._q.put(None)
        for t in self._threads:
            t.join(timeout=1.0)
        self._threads.clear()


class HttpBackend:
    def __init__(self, base_url: str) -> None:
        self._base = base_url.rstrip("/")
        self._client = httpx.Client(timeout=60.0)

    def infer(self, tensor: list[float], channels: int, height: int, width: int, force_cpu: bool) -> list[float]:
        r = self._client.post(
            f"{self._base}/internal/infer",
            json={"tensor": tensor, "channels": channels, "height": height, "width": width, "force_cpu": force_cpu},
        )
        r.raise_for_status()
        return list(r.json()["output"])

    def batch_infer(
        self, tensors: Sequence[list[float]], channels: int, height: int, width: int, force_cpu: bool
    ) -> list[list[float]]:
        r = self._client.post(
            f"{self._base}/internal/batch_infer",
            json={
                "tensors": [list(t) for t in tensors],
                "channels": channels,
                "height": height,
                "width": width,
                "force_cpu": force_cpu,
            },
        )
        r.raise_for_status()
        return [list(x) for x in r.json()["outputs"]]

    def snapshot(self) -> dict[str, Any]:
        r = self._client.get(f"{self._base}/internal/snapshot")
        r.raise_for_status()
        return dict(r.json())

    def shutdown(self) -> None:
        self._client.close()


@dataclass
class EngineClient:
    cfg: AppConfig
    backend: InferenceBackend | None = None
    executor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=32))
    metrics: EngineMetrics = field(default_factory=EngineMetrics)

    def __post_init__(self) -> None:
        if self.backend is not None:
            return
        backend_mode = str(self.cfg.inference.get("backend", "local"))
        if backend_mode == "remote":
            url = str(self.cfg.inference.get("remote_url", "http://127.0.0.1:9100"))
            self.backend = HttpBackend(url)
            log.info("Engine backend=remote url=%s", url)
            return
        force_py = os.environ.get("EDGE_FORCE_PYTHON_ENGINE", "").lower() in ("1", "true", "yes")
        if not force_py:
            try:
                self.backend = CppBackend(self.cfg)
                log.info("Engine backend=cpp (native module)")
                return
            except Exception as e:  # noqa: BLE001
                log.warning("Native engine unavailable (%s); falling back to Python simulation", e)
        self.backend = PythonSimBackend(self.cfg)
        log.info("Engine backend=python_sim")

    async def infer_async(
        self, tensor: list[float], channels: int = 0, height: int = 0, width: int = 0, force_cpu: bool = False
    ) -> tuple[list[float], dict[str, Any], float]:
        assert self.backend is not None
        loop = asyncio.get_event_loop()
        t0 = time.perf_counter()
        out = await loop.run_in_executor(
            self.executor,
            lambda: self.backend.infer(tensor, channels, height, width, force_cpu),  # type: ignore[union-attr]
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        snap = self.backend.snapshot()  # type: ignore[union-attr]
        gpu_log = str(snap.get("gpu_log_line", ""))
        self.metrics.record(dt_ms, gpu_log)
        return out, snap, dt_ms

    async def batch_infer_async(
        self,
        tensors: Sequence[list[float]],
        channels: int = 0,
        height: int = 0,
        width: int = 0,
        force_cpu: bool = False,
    ) -> tuple[list[list[float]], dict[str, Any], float]:
        assert self.backend is not None
        loop = asyncio.get_event_loop()
        t0 = time.perf_counter()
        out = await loop.run_in_executor(
            self.executor,
            lambda: self.backend.batch_infer(tensors, channels, height, width, force_cpu),  # type: ignore[union-attr]
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        snap = self.backend.snapshot()  # type: ignore[union-attr]
        self.metrics.record(dt_ms, str(snap.get("gpu_log_line", "")))
        return out, snap, dt_ms

    def shutdown(self) -> None:
        if self.backend:
            self.backend.shutdown()
        self.executor.shutdown(wait=False)
