from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orchestration.camera_manager import CameraManager
from orchestration.engine_client import EngineClient
from utils.logging_config import get_logger

log = get_logger("orchestration.pipeline_coordinator")


@dataclass
class PipelineCoordinator:
    """Camera → ring buffer → flatten → engine (documents the production flow)."""

    cameras: CameraManager
    engine: EngineClient

    async def infer_latest(self, stream_id: str, force_cpu: bool = False):
        frame = self.cameras.latest_frame_nchw(stream_id)
        if frame is None:
            raise ValueError(f"no frames for stream_id={stream_id}")
        flat = frame.reshape(-1).tolist()
        return await self.engine.infer_async(flat, frame.shape[0], frame.shape[1], frame.shape[2], force_cpu)
