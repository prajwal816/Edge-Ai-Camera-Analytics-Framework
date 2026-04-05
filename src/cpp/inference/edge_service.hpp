#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "cuda_utils/gpu_metrics.hpp"
#include "inference/inference_engine.hpp"
#include "preprocessing/preprocessor.hpp"
#include "scheduler/dynamic_scheduler.hpp"

namespace edge {

struct ServiceSnapshot {
  std::string gpu_log_line;
  uint64_t scheduler_submitted{0};
  uint64_t scheduler_batches{0};
  uint64_t engine_batches{0};
  uint64_t engine_frames{0};
  uint64_t engine_gpu_path{0};
  uint64_t engine_cpu_path{0};
  bool baseline_mode{false};
};

class EdgeInferenceService {
 public:
  EdgeInferenceService();

  void configure(const std::string& manifest_path, bool baseline_mode, bool prefer_gpu, bool use_gpu_preprocess,
                 std::size_t max_batch, double max_wait_ms, int worker_threads, double cpu_fallback_util_threshold);

  void shutdown();

  std::vector<float> infer(const std::vector<float>& frame_nchw, int channels, int height, int width,
                           bool force_cpu);

  std::vector<std::vector<float>> batch_infer(const std::vector<std::vector<float>>& frames, int channels,
                                              int height, int width, bool force_cpu);

  ServiceSnapshot snapshot() const;

 private:
  mutable std::mutex mu_;
  InferenceEngine engine_;
  Preprocessor pre_;
  GpuMetricsSimulator gpu_;
  std::unique_ptr<DynamicScheduler> sched_;
  bool started_{false};
};

}  // namespace edge
