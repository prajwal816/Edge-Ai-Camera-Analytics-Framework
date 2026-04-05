"""
Dedicated inference sidecar (FastAPI) for docker-compose multi-service layout.
Gateway API uses inference.backend=remote to forward work here.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

_PY_ROOT = Path(__file__).resolve().parents[1]
if str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))

from orchestration.engine_client import EngineClient  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.logging_config import get_logger, setup_logging  # noqa: E402

log = get_logger("orchestration.inference_service")
_engine: EngineClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    cfg = load_config()
    setup_logging(str(cfg.service.get("log_level", "INFO")))
    # Sidecar always uses local native / python sim (ignore remote in nested calls)
    cfg.raw.setdefault("inference", {})
    cfg.raw["inference"]["backend"] = "local"
    _engine = EngineClient(cfg=cfg)
    log.info("Inference sidecar online")
    yield
    if _engine:
        _engine.shutdown()
    _engine = None


app = FastAPI(title="Edge Inference Engine", version="0.1.0", lifespan=lifespan)


class InternalInferRequest(BaseModel):
    tensor: list[float]
    channels: int = 0
    height: int = 0
    width: int = 0
    force_cpu: bool = False


class InternalBatchRequest(BaseModel):
    tensors: list[list[float]]
    channels: int = 0
    height: int = 0
    width: int = 0
    force_cpu: bool = False


@app.get("/health")
async def health() -> dict[str, Any]:
    if _engine is None or _engine.backend is None:
        raise HTTPException(status_code=503, detail="engine not initialized")
    return {"status": "ok", "engine": _engine.backend.snapshot()}


@app.post("/internal/infer")
async def internal_infer(req: InternalInferRequest) -> dict[str, Any]:
    if _engine is None:
        raise HTTPException(status_code=503, detail="engine not initialized")
    out, snap, _dt = await _engine.infer_async(req.tensor, req.channels, req.height, req.width, req.force_cpu)
    return {"output": out, "engine_snapshot": snap}


@app.post("/internal/batch_infer")
async def internal_batch(req: InternalBatchRequest) -> dict[str, Any]:
    if _engine is None:
        raise HTTPException(status_code=503, detail="engine not initialized")
    outs, snap, _dt = await _engine.batch_infer_async(req.tensors, req.channels, req.height, req.width, req.force_cpu)
    return {"outputs": outs, "engine_snapshot": snap}


@app.get("/internal/snapshot")
async def internal_snapshot() -> dict[str, Any]:
    if _engine is None or _engine.backend is None:
        raise HTTPException(status_code=503, detail="engine not initialized")
    return dict(_engine.backend.snapshot())
