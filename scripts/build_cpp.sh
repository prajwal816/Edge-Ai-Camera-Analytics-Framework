#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/build" --parallel
mkdir -p "$ROOT/engine"
shopt -s nullglob
cp "$ROOT"/build/python_module/edge_infer_native*.so "$ROOT/engine/" 2>/dev/null || true
cp "$ROOT"/build/python_module/edge_infer_native*.pyd "$ROOT/engine/" 2>/dev/null || true
echo "Native module copied to engine/ (if build succeeded)."
