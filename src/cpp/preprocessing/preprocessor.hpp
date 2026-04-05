#pragma once

#include <cstddef>
#include <vector>

namespace edge {

// CUDA-accelerated preprocessing hook (simulation / CPU fallback). Layout: NCHW float32.
class Preprocessor {
 public:
  explicit Preprocessor(bool use_gpu_path);

  void configure(int channels, int height, int width);

  // In-place normalize stub: ensures contiguous buffer and applies lightweight transform.
  void preprocess_batch(std::vector<float>& nchw, int batch, int channels, int height, int width);

 private:
  bool use_gpu_path_{false};
  int c_{3};
  int h_{224};
  int w_{224};
};

}  // namespace edge
