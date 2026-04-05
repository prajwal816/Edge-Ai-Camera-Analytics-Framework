from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure src/python is on path when running as `python src/python/api/main.py`
_PY_ROOT = Path(__file__).resolve().parents[1]
if str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))

from orchestration.camera_manager import CameraKind, CameraManager, CameraStreamConfig  # noqa: E402
from orchestration.engine_client import EngineClient  # noqa: E402
from orchestration.pipeline_coordinator import PipelineCoordinator  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.logging_config import get_logger, setup_logging  # noqa: E402

log = get_logger("api.main")

_engine: EngineClient | None = None
_cameras: CameraManager | None = None
_pipeline: PipelineCoordinator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _cameras, _pipeline
    cfg = load_config()
    setup_logging(str(cfg.service.get("log_level", "INFO")))
    _engine = EngineClient(cfg=cfg)
    _cameras = CameraManager(cfg)
    streams = [
        CameraStreamConfig(stream_id="csi-0", kind=CameraKind.CSI_SIM),
        CameraStreamConfig(stream_id="usb-0", kind=CameraKind.USB, device_index=int(cfg.camera.get("usb_default_device", 0))),
    ]
    _cameras.start_streams(streams)
    _pipeline = PipelineCoordinator(cameras=_cameras, engine=_engine)
    log.info("API ready (cameras=%s)", [s.stream_id for s in streams])
    yield
    if _cameras:
        _cameras.stop()
    if _engine:
        _engine.shutdown()
    _engine = None
    _cameras = None
    _pipeline = None


app = FastAPI(title="Edge AI Camera Analytics API", version="0.1.0", lifespan=lifespan)


class InferRequest(BaseModel):
    tensor: list[float] = Field(..., description="Flattened NCHW float32 values")
    channels: int = 0
    height: int = 0
    width: int = 0
    force_cpu: bool = False


class InferResponse(BaseModel):
    output: list[float]
    latency_ms: float
    engine_snapshot: dict[str, Any]


class BatchInferRequest(BaseModel):
    tensors: list[list[float]]
    channels: int = 0
    height: int = 0
    width: int = 0
    force_cpu: bool = False


class BatchInferResponse(BaseModel):
    outputs: list[list[float]]
    latency_ms: float
    engine_snapshot: dict[str, Any]


@app.get("/health")
async def health() -> dict[str, Any]:
    if _engine is None or _engine.backend is None:
        raise HTTPException(status_code=503, detail="engine not initialized")
    snap = _engine.backend.snapshot()
    fps = _cameras.fps_snapshot() if _cameras else {}
    return {
        "status": "ok",
        "backend": type(_engine.backend).__name__,
        "engine": snap,
        "camera_fps": fps,
        "avg_client_latency_ms": _engine.metrics.avg_latency_ms,
        "concurrency_model": "asyncio + thread_pool_blocking_cpp",
    }


@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest) -> InferResponse:
    if _engine is None:
        raise HTTPException(status_code=503, detail="engine not initialized")
    out, snap, dt = await _engine.infer_async(req.tensor, req.channels, req.height, req.width, req.force_cpu)
    return InferResponse(output=out, latency_ms=dt, engine_snapshot=snap)


@app.post("/batch_infer", response_model=BatchInferResponse)
async def batch_infer(req: BatchInferRequest) -> BatchInferResponse:
    if _engine is None:
        raise HTTPException(status_code=503, detail="engine not initialized")
    if not req.tensors:
        raise HTTPException(status_code=400, detail="empty batch")
    out, snap, dt = await _engine.batch_infer_async(req.tensors, req.channels, req.height, req.width, req.force_cpu)
    return BatchInferResponse(outputs=out, latency_ms=dt, engine_snapshot=snap)


@app.post("/infer_latest/{stream_id}", response_model=InferResponse)
async def infer_latest(stream_id: str, force_cpu: bool = False) -> InferResponse:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline not initialized")
    out, snap, dt = await _pipeline.infer_latest(stream_id, force_cpu=force_cpu)
    return InferResponse(output=out, latency_ms=dt, engine_snapshot=snap)

