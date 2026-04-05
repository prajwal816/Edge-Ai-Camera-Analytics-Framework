#pragma once

#include <atomic>
#include <mutex>
#include <random>
#include <string>
#include <vector>

namespace edge {

// Simulated GPU utilization and memory telemetry for edge dashboards (replace with NVML on production hardware).
class GpuMetricsSimulator {
 public:
  GpuMetricsSimulator();

  void tick(double batch_utilization_hint);
  double utilization_percent() const { return util_.load(); }
  double mem_used_mb() const { return mem_used_.load(); }
  std::string last_log_line() const;

 private:
  std::atomic<double> util_{0.0};
  std::atomic<double> mem_used_{512.0};
  mutable std::mutex rng_mu_;
  std::mt19937 rng_;
  std::normal_distribution<double> noise_;
};

}  // namespace edge
