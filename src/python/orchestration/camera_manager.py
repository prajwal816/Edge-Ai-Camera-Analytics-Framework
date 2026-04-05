from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Optional

import numpy as np

from utils.config import AppConfig
from utils.logging_config import get_logger

log = get_logger("orchestration.camera_manager")


class CameraKind(str, Enum):
    CSI_SIM = "csi_sim"
    USB = "usb"


@dataclass
class CameraStreamConfig:
    stream_id: str
    kind: CameraKind
    device_index: int = 0


@dataclass
class CameraManager:
    cfg: AppConfig
    _threads: list[threading.Thread] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _buffers: dict[str, Deque[np.ndarray]] = field(default_factory=dict)
    _fps: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def start_streams(self, streams: list[CameraStreamConfig]) -> None:
        self._stop.clear()
        buf_size = int(self.cfg.camera.get("ring_buffer_size", 32))
        h = int(self.cfg.camera.get("frame_height", 224))
        w = int(self.cfg.camera.get("frame_width", 224))
        c = 3
        for s in streams:
            q: Deque[np.ndarray] = deque(maxlen=buf_size)
            with self._lock:
                self._buffers[s.stream_id] = q
                self._fps[s.stream_id] = 0.0
            t = threading.Thread(
                target=self._ingest_loop,
                args=(s, h, w, c),
                name=f"cam-{s.stream_id}",
                daemon=True,
            )
            self._threads.append(t)
            t.start()
            log.info("Started camera thread stream_id=%s kind=%s", s.stream_id, s.kind)

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()

    def latest_frame_nchw(self, stream_id: str) -> np.ndarray | None:
        with self._lock:
            q = self._buffers.get(stream_id)
            if not q:
                return None
            return q[-1].copy() if len(q) else None

    def fps_snapshot(self) -> dict[str, float]:
        with self._lock:
            return dict(self._fps)

    def _try_open_cv_capture(self, device_index: int):
        try:
            import cv2  # type: ignore
        except Exception:
            return None
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            return None
        return cap

    def _ingest_loop(self, stream: CameraStreamConfig, h: int, w: int, c: int) -> None:
        cap = None
        simulate = os.environ.get("EDGE_SIMULATE_CAMERAS", "1").lower() in ("1", "true", "yes")
        if stream.kind == CameraKind.USB and not simulate:
            cap = self._try_open_cv_capture(stream.device_index)
        target_fps = float(self.cfg.camera.get("csi_sim_fps", 30))
        frame_interval = 1.0 / max(1.0, target_fps)
        frames = 0
        t0 = time.perf_counter()
        while not self._stop.is_set():
            loop_start = time.perf_counter()
            if cap is not None:
                ok, bgr = cap.read()
                if ok and bgr is not None:
                    try:
                        import cv2  # type: ignore

                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        rgb = cv2.resize(rgb, (w, h))
                    except Exception:
                        rgb = self._synthetic_frame(h, w, stream.stream_id, frames)
                else:
                    rgb = self._synthetic_frame(h, w, stream.stream_id, frames)
            else:
                rgb = self._synthetic_frame(h, w, stream.stream_id, frames)
            nchw = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
            with self._lock:
                self._buffers[stream.stream_id].append(nchw)
            frames += 1
            if frames % 30 == 0:
                elapsed = time.perf_counter() - t0
                with self._lock:
                    self._fps[stream.stream_id] = frames / max(elapsed, 1e-6)
            sleep = frame_interval - (time.perf_counter() - loop_start)
            if sleep > 0:
                time.sleep(sleep)

    def _synthetic_frame(self, h: int, w: int, label: str, frame_idx: int) -> np.ndarray:
        x = np.linspace(0, 1, w, dtype=np.float32)
        y = np.linspace(0, 1, h, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)
        phase = (hash(label) % 997) * 0.01 + frame_idx * 0.02
        r = 0.5 + 0.5 * np.sin(2 * np.pi * (xv + phase))
        g = 0.5 + 0.5 * np.sin(2 * np.pi * (yv + phase))
        b = 0.5 + 0.5 * np.sin(2 * np.pi * ((xv + yv) * 0.5 + phase))
        rgb = np.stack([r, g, b], axis=-1)
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
