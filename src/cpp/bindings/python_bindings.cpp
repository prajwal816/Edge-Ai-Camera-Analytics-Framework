#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>

#include "inference/edge_service.hpp"

namespace py = pybind11;

PYBIND11_MODULE(edge_infer_native, m) {
  m.doc() = "C++17 edge inference core: batching scheduler, preprocessing hook, TensorRT-ready engine (sim default).";

  py::class_<edge::EdgeInferenceService, std::shared_ptr<edge::EdgeInferenceService>>(m, "EdgeInferenceService")
      .def(py::init<>())
      .def(
          "configure",
          [](edge::EdgeInferenceService& self, const std::string& manifest_path, bool baseline_mode, bool prefer_gpu,
             bool use_gpu_preprocess, std::size_t max_batch, double max_wait_ms, int worker_threads,
             double cpu_fallback_util_threshold) {
            self.configure(manifest_path, baseline_mode, prefer_gpu, use_gpu_preprocess, max_batch, max_wait_ms,
                           worker_threads, cpu_fallback_util_threshold);
          },
          py::arg("manifest_path"), py::arg("baseline_mode") = false, py::arg("prefer_gpu") = true,
          py::arg("use_gpu_preprocess") = true, py::arg("max_batch") = 8, py::arg("max_wait_ms") = 4.0,
          py::arg("worker_threads") = 4, py::arg("cpu_fallback_util_threshold") = 0.92)
      .def("shutdown", &edge::EdgeInferenceService::shutdown)
      .def("infer", &edge::EdgeInferenceService::infer, py::arg("frame_nchw"), py::arg("channels") = 0,
           py::arg("height") = 0, py::arg("width") = 0, py::arg("force_cpu") = false)
      .def("batch_infer", &edge::EdgeInferenceService::batch_infer, py::arg("frames"), py::arg("channels") = 0,
           py::arg("height") = 0, py::arg("width") = 0, py::arg("force_cpu") = false)
      .def(
          "snapshot",
          [](const edge::EdgeInferenceService& self) {
            const auto s = self.snapshot();
            py::dict d;
            d["gpu_log_line"] = s.gpu_log_line;
            d["scheduler_submitted"] = s.scheduler_submitted;
            d["scheduler_batches"] = s.scheduler_batches;
            d["engine_batches"] = s.engine_batches;
            d["engine_frames"] = s.engine_frames;
            d["engine_gpu_path"] = s.engine_gpu_path;
            d["engine_cpu_path"] = s.engine_cpu_path;
            d["baseline_mode"] = s.baseline_mode;
            return d;
          });
}
