#include "cuda_utils/gpu_metrics.hpp"

#include <cmath>
#include <sstream>

#include "common/logging.hpp"

namespace edge {

GpuMetricsSimulator::GpuMetricsSimulator() : rng_(std::random_device{}()), noise_(0.0, 2.5) {}

void GpuMetricsSimulator::tick(double batch_utilization_hint) {
  std::lock_guard<std::mutex> lk(rng_mu_);
  const double n = noise_(rng_);
  double u = batch_utilization_hint * 100.0 + n;
  if (u < 5.0) {
    u = 5.0;
  }
  if (u > 98.0) {
    u = 98.0;
  }
  util_.store(u);

  // Pretend memory scales mildly with "load"
  const double base = 420.0 + batch_utilization_hint * 900.0;
  mem_used_.store(base + std::fabs(n) * 10.0);
}

std::string GpuMetricsSimulator::last_log_line() const {
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision(1);
  oss << "gpu_utilization_percent=" << util_.load() << " gpu_mem_used_mb=" << mem_used_.load()
      << " (simulated telemetry)";
  return oss.str();
}

}  // namespace edge
