#include "inference/inference_engine.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <thread>

#include "common/logging.hpp"

#if defined(EDGE_USE_TENSORRT)
// TensorRT integration: link nvinfer and implement deserialize + enqueue in production builds.
#include <NvInfer.h>
#endif

namespace edge {

namespace {

bool read_manifest_simple(const std::string& path, EngineManifest& out) {
  std::ifstream f(path);
  if (!f) {
    return false;
  }
  std::string raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  auto grab_int = [&](const char* key, int& dst) {
    const std::string k = std::string("\"") + key + "\":";
    const auto p = raw.find(k);
    if (p == std::string::npos) {
      return;
    }
    std::size_t i = p + k.size();
    while (i < raw.size() && (raw[i] == ' ' || raw[i] == '\t')) {
      ++i;
    }
    dst = std::atoi(raw.c_str() + i);
  };
  grab_int("input_channels", out.input_c);
  grab_int("input_height", out.input_h);
  grab_int("input_width", out.input_w);
  grab_int("output_dim", out.output_dim);
  const std::string nk = "\"name\":";
  const auto np = raw.find(nk);
  if (np != std::string::npos) {
    std::size_t a = raw.find('"', np + nk.size());
    std::size_t b = (a == std::string::npos) ? std::string::npos : raw.find('"', a + 1);
    if (a != std::string::npos && b != std::string::npos) {
      out.name = raw.substr(a + 1, b - a - 1);
    }
  }
  return true;
}

}  // namespace

InferenceEngine::InferenceEngine() = default;

void InferenceEngine::load_manifest(const std::string& json_path) {
  EngineManifest m;
  if (!read_manifest_simple(json_path, m)) {
    log_warn("tensorrt", "manifest missing or unreadable; using defaults: " + json_path);
    m.input_c = 3;
    m.input_h = 224;
    m.input_w = 224;
    m.output_dim = 64;
    m.name = "default_sim";
  }
  {
    std::lock_guard<std::mutex> lk(mu_);
    manifest_ = m;
  }
#if defined(EDGE_USE_TENSORRT)
  log_info("tensorrt", "EDGE_USE_TENSORRT enabled: deserialize .trt from manifest in production images.");
#else
  log_info("tensorrt", "TensorRT path not compiled in; using deterministic simulation kernel.");
#endif
}

void InferenceEngine::simulate_compute(int batch, bool on_gpu) const {
  if (baseline_mode_) {
    // Single-threaded feel: longer per-frame delay, no batch amortization.
    const int us = on_gpu ? 900 : 3500;
    std::this_thread::sleep_for(std::chrono::microseconds(us * batch));
    return;
  }
  if (on_gpu) {
    const int us = 220 + 40 * (batch - 1);  // batch amortization
    std::this_thread::sleep_for(std::chrono::microseconds(us));
  } else {
    const int us = 1800 + 200 * (batch - 1);
    std::this_thread::sleep_for(std::chrono::microseconds(us));
  }
}

std::vector<float> InferenceEngine::execute(const std::vector<float>& nchw, int batch, bool force_cpu) {
  EngineManifest man;
  {
    std::lock_guard<std::mutex> lk(mu_);
    man = manifest_;
  }
  const int c = man.input_c;
  const int h = man.input_h;
  const int w = man.input_w;
  const int od = man.output_dim;
  const std::size_t expected = static_cast<std::size_t>(batch) * static_cast<std::size_t>(c) * static_cast<std::size_t>(h) *
                               static_cast<std::size_t>(w);
  if (nchw.size() != expected) {
    log_error("tensorrt", "input tensor size mismatch for batch execution");
    return {};
  }

  const bool on_gpu = !force_cpu && prefer_gpu_;
  simulate_compute(batch, on_gpu);
  if (on_gpu) {
    gpu_path_.fetch_add(1, std::memory_order_relaxed);
  } else {
    cpu_path_.fetch_add(1, std::memory_order_relaxed);
  }
  batches_.fetch_add(1, std::memory_order_relaxed);
  frames_.fetch_add(static_cast<uint64_t>(batch), std::memory_order_relaxed);

  std::vector<float> out(static_cast<std::size_t>(batch) * static_cast<std::size_t>(od));
  // Lightweight deterministic "inference": partial sums of input patches -> output vector
  const int patch = std::max(1, (c * h * w) / od);
  for (int b = 0; b < batch; ++b) {
    const std::size_t base_in = static_cast<std::size_t>(b) * static_cast<std::size_t>(c * h * w);
    for (int o = 0; o < od; ++o) {
      double acc = 0.0;
      const std::size_t start = base_in + static_cast<std::size_t>(o * patch);
      const std::size_t end = std::min(base_in + static_cast<std::size_t>(c * h * w), start + static_cast<std::size_t>(patch));
      for (std::size_t i = start; i < end; ++i) {
        acc += static_cast<double>(nchw[i]);
      }
      const std::size_t out_idx = static_cast<std::size_t>(b) * static_cast<std::size_t>(od) + static_cast<std::size_t>(o);
      out[out_idx] = static_cast<float>(std::tanh(acc / static_cast<double>(patch + 1)));
    }
  }
  return out;
}

}  // namespace edge
