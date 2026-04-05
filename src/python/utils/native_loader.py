from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

from .config import repo_root


def load_edge_infer_native() -> ModuleType | None:
    """Load pybind11 module from EDGE_NATIVE_MODULE path or common build output locations."""
    env_path = os.environ.get("EDGE_NATIVE_MODULE")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    root = repo_root()
    for pattern in (
        "engine/edge_infer_native*.so",
        "engine/edge_infer_native*.pyd",
        "build/python_module/edge_infer_native*.so",
        "build/python_module/edge_infer_native*.pyd",
        "build/python_module/Debug/edge_infer_native*.pyd",
        "build/python_module/Release/edge_infer_native*.pyd",
        "build/python_module/*/edge_infer_native*.so",
        "cmake-build-release/python_module/edge_infer_native*.so",
        "cmake-build-debug/python_module/edge_infer_native*.pyd",
    ):
        for p in sorted(root.glob(pattern)):
            candidates.append(p)
    for p in candidates:
        if p.is_file():
            spec = importlib.util.spec_from_file_location("edge_infer_native", str(p))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules["edge_infer_native"] = mod
                spec.loader.exec_module(mod)
                return mod
    try:
        import edge_infer_native as mod  # type: ignore

        return mod
    except Exception:
        return None
