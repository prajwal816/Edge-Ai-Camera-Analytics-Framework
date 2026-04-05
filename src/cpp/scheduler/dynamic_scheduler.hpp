#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace edge {

class InferenceEngine;
class Preprocessor;
class GpuMetricsSimulator;

struct SchedulerConfig {
  std::size_t max_batch{8};
  double max_wait_ms{4.0};
  double cpu_fallback_util_threshold{0.92};
  int worker_threads{2};
};

// Queue-based dynamic batching with optional CPU fallback under simulated GPU pressure.
class DynamicScheduler {
 public:
  DynamicScheduler(InferenceEngine* engine, Preprocessor* pre, GpuMetricsSimulator* gpu,
                   SchedulerConfig cfg);
  ~DynamicScheduler();

  DynamicScheduler(const DynamicScheduler&) = delete;
  DynamicScheduler& operator=(const DynamicScheduler&) = delete;

  void start();
  void stop();

  std::vector<float> submit(const std::vector<float>& frame_nchw, int channels, int height, int width,
                            bool force_cpu_hint);

  SchedulerConfig config() const { return cfg_; }

  std::uint64_t submitted_total() const { return submitted_.load(std::memory_order_relaxed); }
  std::uint64_t batch_events() const { return batched_.load(std::memory_order_relaxed); }

 private:
  struct Job {
    std::vector<float> input;
    int c{0};
    int h{0};
    int w{0};
    bool force_cpu_hint{false};
    std::shared_ptr<std::promise<std::vector<float>>> prom;
  };

  void worker_loop();

  InferenceEngine* engine_{nullptr};
  Preprocessor* pre_{nullptr};
  GpuMetricsSimulator* gpu_{nullptr};
  SchedulerConfig cfg_;

  std::vector<std::thread> workers_;
  std::atomic<bool> running_{false};

  std::mutex q_mu_;
  std::condition_variable q_cv_;
  std::queue<std::shared_ptr<Job>> q_;

  std::atomic<std::uint64_t> submitted_{0};
  std::atomic<std::uint64_t> batched_{0};
};

}  // namespace edge
