#include "preprocessing/preprocessor.hpp"

#include <cmath>

#include "common/logging.hpp"

namespace edge {

Preprocessor::Preprocessor(bool use_gpu_path) : use_gpu_path_(use_gpu_path) {}

void Preprocessor::configure(int channels, int height, int width) {
  c_ = channels;
  h_ = height;
  w_ = width;
}

void Preprocessor::preprocess_batch(std::vector<float>& nchw, int batch, int channels, int height, int width) {
  const std::size_t expected = static_cast<std::size_t>(batch) * static_cast<std::size_t>(channels) *
                               static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
  if (nchw.size() != expected) {
    log_error("preprocess", "nchw size mismatch");
    return;
  }
  // Simulated fused kernel: slight contrast + clamp (CPU stand-in for CUDA kernel).
  const float scale = use_gpu_path_ ? 1.02f : 1.0f;
  for (float& v : nchw) {
    v = std::tanh(v * scale * 0.5f);
  }
}

}  // namespace edge
