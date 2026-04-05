#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace edge {

struct EngineManifest {
  int input_c{3};
  int input_h{224};
  int input_w{224};
  int output_dim{64};
  std::string name;
};

struct InferenceCounters {
  uint64_t batches{0};
  uint64_t frames{0};
  uint64_t gpu_path{0};
  uint64_t cpu_path{0};
};

class InferenceEngine {
 public:
  InferenceEngine();

  void load_manifest(const std::string& json_path);
  void set_baseline_mode(bool v) { baseline_mode_ = v; }
  void set_prefer_gpu(bool v) { prefer_gpu_ = v; }

  // Runs NCHW FP32 batch; returns flat [N * output_dim]
  std::vector<float> execute(const std::vector<float>& nchw, int batch, bool force_cpu);

  EngineManifest manifest() const {
    std::lock_guard<std::mutex> lk(mu_);
    return manifest_;
  }

  InferenceCounters counters() const {
    InferenceCounters c;
    c.batches = batches_.load();
    c.frames = frames_.load();
    c.gpu_path = gpu_path_.load();
    c.cpu_path = cpu_path_.load();
    return c;
  }

 private:
  void simulate_compute(int batch, bool on_gpu) const;

  mutable std::mutex mu_;
  EngineManifest manifest_;
  std::atomic<bool> baseline_mode_{false};
  std::atomic<bool> prefer_gpu_{true};

  std::atomic<uint64_t> batches_{0};
  std::atomic<uint64_t> frames_{0};
  std::atomic<uint64_t> gpu_path_{0};
  std::atomic<uint64_t> cpu_path_{0};
};

}  // namespace edge
