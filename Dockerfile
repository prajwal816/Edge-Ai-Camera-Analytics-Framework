# Multi-stage: build C++17 pybind11 module, ship slim API runtime.
FROM python:3.11-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY CMakeLists.txt /app/
COPY src/cpp /app/src/cpp
RUN cmake -S /app -B /app/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /app/build --parallel \
    && mkdir -p /app/engine \
    && bash -c 'shopt -s nullglob; cp /app/build/python_module/edge_infer_native*.so /app/engine/ || true'

FROM python:3.11-slim-bookworm AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app/src/python
ENV PYTHONUNBUFFERED=1
ENV EDGE_SIMULATE_CAMERAS=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY configs /app/configs
COPY models /app/models
COPY camera /app/camera
COPY scheduler /app/scheduler
COPY src/python /app/src/python
COPY benchmarks /app/benchmarks
COPY tests /app/tests
COPY pytest.ini /app/pytest.ini

COPY --from=builder /app/engine /app/engine

EXPOSE 8000 9100

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
