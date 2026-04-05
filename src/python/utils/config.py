from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(dict(out[k]), v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


@dataclass
class AppConfig:
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def service(self) -> dict[str, Any]:
        return self.raw.get("service", {})

    @property
    def inference(self) -> dict[str, Any]:
        return self.raw.get("inference", {})

    @property
    def camera(self) -> dict[str, Any]:
        return self.raw.get("camera", {})

    @property
    def metrics(self) -> dict[str, Any]:
        return self.raw.get("metrics", {})

    @property
    def benchmarks(self) -> dict[str, Any]:
        return self.raw.get("benchmarks", {})


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_config(path: str | None = None) -> AppConfig:
    root = repo_root()
    p = Path(path or os.environ.get("EDGE_CONFIG_PATH", str(root / "configs" / "default.yaml")))
    data: dict[str, Any] = {}
    if p.is_file():
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    local = root / "configs" / "local.yaml"
    if local.is_file():
        with local.open("r", encoding="utf-8") as f:
            override = yaml.safe_load(f) or {}
        data = _deep_merge(data, override)
    cfg = AppConfig(raw=data)
    _apply_env_overrides(cfg)
    return cfg


def _apply_env_overrides(cfg: AppConfig) -> None:
    b = os.environ.get("EDGE_INFERENCE_BACKEND")
    if b:
        cfg.raw.setdefault("inference", {})["backend"] = b
    r = os.environ.get("EDGE_INFERENCE_REMOTE_URL")
    if r:
        cfg.raw.setdefault("inference", {})["remote_url"] = r
    if os.environ.get("EDGE_BASELINE_MODE", "").lower() in ("1", "true", "yes"):
        cfg.raw.setdefault("inference", {})["baseline_mode"] = True
    if os.environ.get("EDGE_OPTIMIZED_MODE", "").lower() in ("1", "true", "yes"):
        cfg.raw.setdefault("inference", {})["baseline_mode"] = False
