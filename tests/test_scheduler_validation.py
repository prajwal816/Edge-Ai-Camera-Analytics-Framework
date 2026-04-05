from __future__ import annotations

import os

import pytest

os.environ.setdefault("EDGE_FORCE_PYTHON_ENGINE", "1")

from utils.config import load_config  # noqa: E402
from orchestration.engine_client import EngineClient  # noqa: E402


@pytest.mark.asyncio
async def test_engine_client_batches_sequentially_python_sim() -> None:
    cfg = load_config()
    cfg.raw.setdefault("inference", {})
    cfg.raw["inference"]["baseline_mode"] = False
    client = EngineClient(cfg=cfg)
    t = [0.02] * (3 * 224 * 224)
    outs, snap, dt = await client.batch_infer_async([t, t], 3, 224, 224, False)
    assert len(outs) == 2
    assert len(outs[0]) == 64
    assert "engine_batches" in snap or "engine_frames" in snap
    client.shutdown()
