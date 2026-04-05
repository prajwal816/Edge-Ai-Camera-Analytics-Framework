#include "scheduler/dynamic_scheduler.hpp"

#include <algorithm>
#include <cmath>

#include "common/logging.hpp"
#include "cuda_utils/gpu_metrics.hpp"
#include "inference/inference_engine.hpp"
#include "preprocessing/preprocessor.hpp"

namespace edge {

DynamicScheduler::DynamicScheduler(InferenceEngine* engine, Preprocessor* pre, GpuMetricsSimulator* gpu,
                                   SchedulerConfig cfg)
    : engine_(engine), pre_(pre), gpu_(gpu), cfg_(std::move(cfg)) {}

DynamicScheduler::~DynamicScheduler() { stop(); }

void DynamicScheduler::start() {
  if (running_.exchange(true)) {
    return;
  }
  const int n = std::max(1, cfg_.worker_threads);
  workers_.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    workers_.emplace_back([this] { worker_loop(); });
  }
  log_info("scheduler", "started workers=" + std::to_string(n));
}

void DynamicScheduler::stop() {
  if (!running_.exchange(false)) {
    return;
  }
  q_cv_.notify_all();
  for (auto& t : workers_) {
    if (t.joinable()) {
      t.join();
    }
  }
  workers_.clear();
  log_info("scheduler", "stopped all workers");
}

std::vector<float> DynamicScheduler::submit(const std::vector<float>& frame_nchw, int channels, int height,
                                            int width, bool force_cpu_hint) {
  submitted_.fetch_add(1, std::memory_order_relaxed);
  auto prom = std::make_shared<std::promise<std::vector<float>>>();
  auto fut = prom->get_future();
  auto job = std::make_shared<Job>();
  job->input = frame_nchw;
  job->c = channels;
  job->h = height;
  job->w = width;
  job->force_cpu_hint = force_cpu_hint;
  job->prom = std::move(prom);
  {
    std::lock_guard<std::mutex> lk(q_mu_);
    q_.push(job);
  }
  q_cv_.notify_one();
  return fut.get();
}

void DynamicScheduler::worker_loop() {
  while (running_.load()) {
    std::vector<std::shared_ptr<Job>> batch;
    batch.reserve(cfg_.max_batch);
    const auto deadline =
        std::chrono::steady_clock::now() +
        std::chrono::duration<double, std::milli>(std::max(0.1, cfg_.max_wait_ms));

    {
      std::unique_lock<std::mutex> lk(q_mu_);
      q_cv_.wait_until(lk, deadline, [this] { return !q_.empty() || !running_.load(); });
      if (!running_.load() && q_.empty()) {
        return;
      }
      while (!q_.empty() && batch.size() < cfg_.max_batch) {
        batch.push_back(q_.front());
        q_.pop();
      }
    }

    if (batch.empty()) {
      continue;
    }

    batched_.fetch_add(1, std::memory_order_relaxed);

    const int c = batch[0]->c;
    const int h = batch[0]->h;
    const int w = batch[0]->w;
    const int b = static_cast<int>(batch.size());

    std::vector<float> concat;
    concat.reserve(static_cast<std::size_t>(b * c * h * w));
    bool any_force_cpu = false;
    for (const auto& j : batch) {
      any_force_cpu = any_force_cpu || j->force_cpu_hint;
      concat.insert(concat.end(), j->input.begin(), j->input.end());
    }

    if (pre_) {
      pre_->preprocess_batch(concat, b, c, h, w);
    }

    const double util = gpu_ ? (gpu_->utilization_percent() / 100.0) : 0.0;
    const bool force_cpu = any_force_cpu || (util > cfg_.cpu_fallback_util_threshold);

    auto out = engine_->execute(concat, b, force_cpu);
    const int od = static_cast<int>(out.size() / static_cast<std::size_t>(b));
    if (od <= 0 || static_cast<std::size_t>(b) * static_cast<std::size_t>(od) != out.size()) {
      log_error("scheduler", "engine output size unexpected");
      for (const auto& j : batch) {
        j->prom->set_value({});
      }
      continue;
    }

    if (gpu_) {
      const double hint = std::min(1.0, 0.25 + 0.12 * static_cast<double>(b));
      gpu_->tick(hint);
    }

    for (int i = 0; i < b; ++i) {
      std::vector<float> slice(static_cast<std::size_t>(od));
      const std::size_t off = static_cast<std::size_t>(i) * static_cast<std::size_t>(od);
      for (int k = 0; k < od; ++k) {
        slice[static_cast<std::size_t>(k)] = out[off + static_cast<std::size_t>(k)];
      }
      batch[static_cast<std::size_t>(i)]->prom->set_value(std::move(slice));
    }
  }
}

}  // namespace edge
