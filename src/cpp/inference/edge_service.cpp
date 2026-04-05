#include "inference/edge_service.hpp"

#include <algorithm>

#include "common/logging.hpp"

namespace edge {

EdgeInferenceService::EdgeInferenceService() : pre_(false) {}

void EdgeInferenceService::configure(const std::string& manifest_path, bool baseline_mode, bool prefer_gpu,
                                     bool use_gpu_preprocess, std::size_t max_batch, double max_wait_ms,
                                     int worker_threads, double cpu_fallback_util_threshold) {
  std::lock_guard<std::mutex> lk(mu_);
  if (started_) {
    shutdown();
  }
  engine_.load_manifest(manifest_path);
  engine_.set_baseline_mode(baseline_mode);
  engine_.set_prefer_gpu(prefer_gpu);
  const auto man = engine_.manifest();
  pre_ = Preprocessor(use_gpu_preprocess);
  pre_.configure(man.input_c, man.input_h, man.input_w);

  sched_.reset();
  std::size_t effective_max_batch = 1;
  if (!baseline_mode) {
    SchedulerConfig sc;
    sc.max_batch = std::max<std::size_t>(1, max_batch);
    effective_max_batch = sc.max_batch;
    sc.max_wait_ms = max_wait_ms;
    sc.cpu_fallback_util_threshold = cpu_fallback_util_threshold;
    sc.worker_threads = std::max(1, worker_threads);
    sched_ = std::make_unique<DynamicScheduler>(&engine_, &pre_, &gpu_, sc);
    sched_->start();
  }
  started_ = true;
  log_info("edge_service", "configured baseline=" + std::string(baseline_mode ? "true" : "false") +
                               " max_batch=" + std::to_string(effective_max_batch));
}

void EdgeInferenceService::shutdown() {
  if (sched_) {
    sched_->stop();
    sched_.reset();
  }
  started_ = false;
}

std::vector<float> EdgeInferenceService::infer(const std::vector<float>& frame_nchw, int channels, int height,
                                               int width, bool force_cpu) {
  DynamicScheduler* sched = nullptr;
  EngineManifest man;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (!started_) {
      log_error("edge_service", "infer called before configure");
      return {};
    }
    man = engine_.manifest();
    sched = sched_.get();
  }
  if (channels <= 0) {
    channels = man.input_c;
  }
  if (height <= 0) {
    height = man.input_h;
  }
  if (width <= 0) {
    width = man.input_w;
  }

  if (sched) {
    return sched->submit(frame_nchw, channels, height, width, force_cpu);
  }

  std::vector<float> copy = frame_nchw;
  {
    std::lock_guard<std::mutex> lk(mu_);
    pre_.preprocess_batch(copy, 1, channels, height, width);
    return engine_.execute(copy, 1, force_cpu);
  }
}

std::vector<std::vector<float>> EdgeInferenceService::batch_infer(const std::vector<std::vector<float>>& frames,
                                                                  int channels, int height, int width,
                                                                  bool force_cpu) {
  std::vector<std::vector<float>> out;
  out.reserve(frames.size());
  for (const auto& f : frames) {
    out.push_back(infer(f, channels, height, width, force_cpu));
  }
  return out;
}

ServiceSnapshot EdgeInferenceService::snapshot() const {
  ServiceSnapshot s;
  s.gpu_log_line = gpu_.last_log_line();
  std::lock_guard<std::mutex> lk(mu_);
  if (sched_) {
    s.scheduler_submitted = sched_->submitted_total();
    s.scheduler_batches = sched_->batch_events();
  }
  const auto c = engine_.counters();
  s.engine_batches = c.batches;
  s.engine_frames = c.frames;
  s.engine_gpu_path = c.gpu_path;
  s.engine_cpu_path = c.cpu_path;
  s.baseline_mode = !static_cast<bool>(sched_);
  return s;
}

}  // namespace edge
