#include <torch/extension.h>
#include <cmath>
#include <limits>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime_api.h>

void shortcut_launch(const float*, const int64_t*, const bool*, float*, int64_t*,
                     int, int, int, int, int, float, float, hipStream_t);

void shortcut(torch::Tensor path, torch::Tensor lengths, torch::Tensor blocked,
              torch::Tensor simplified, torch::Tensor simple_lengths,
              int64_t sample_count, double resolution) {
  TORCH_CHECK(path.is_cuda() && path.scalar_type() == at::kFloat && path.dim() == 3 &&
              path.size(2) == 2 && path.is_contiguous(), "path must be contiguous CUDA float32 [B,P,2]");
  TORCH_CHECK(lengths.is_cuda() && lengths.device() == path.device() &&
              lengths.scalar_type() == at::kLong && lengths.is_contiguous() &&
              lengths.sizes() == torch::IntArrayRef({path.size(0)}), "invalid lengths");
  TORCH_CHECK(blocked.is_cuda() && blocked.device() == path.device() &&
              blocked.scalar_type() == at::kBool && blocked.dim() == 3 &&
              blocked.is_contiguous() && blocked.size(0) == path.size(0), "invalid blocked grid");
  TORCH_CHECK(simplified.device() == path.device() && simplified.sizes() == path.sizes() &&
              simplified.scalar_type() == at::kFloat && simplified.is_contiguous(), "invalid output path");
  TORCH_CHECK(simple_lengths.device() == path.device() && simple_lengths.sizes() == lengths.sizes() &&
              simple_lengths.scalar_type() == at::kLong && simple_lengths.is_contiguous(), "invalid output lengths");
  TORCH_CHECK(path.size(0) > 0 && path.size(1) >= 2 && blocked.size(1) > 0 && blocked.size(2) > 0 &&
              path.size(0) <= std::numeric_limits<int>::max() - 63 &&
              path.size(1) <= std::numeric_limits<int>::max() / 2 &&
              blocked.size(1) <= 16777216 &&
              blocked.size(2) <= 16777216 &&
              sample_count >= 2 && sample_count <= 16777216 &&
              std::isfinite(resolution) && resolution > 0 &&
              std::isfinite(static_cast<float>(1.0 / resolution)) &&
              std::isfinite(static_cast<float>(1.0 / (resolution * 0.5))),
              "invalid shortcut parameters");
  c10::cuda::CUDAGuard guard(path.device());
  shortcut_launch(path.data_ptr<float>(), lengths.data_ptr<int64_t>(), blocked.data_ptr<bool>(),
                  simplified.data_ptr<float>(), simple_lengths.data_ptr<int64_t>(),
                  path.size(0), path.size(1), blocked.size(1), blocked.size(2),
                  sample_count, static_cast<float>(1.0 / resolution),
                  static_cast<float>(1.0 / (resolution * 0.5)),
                  c10::cuda::getCurrentCUDAStream(path.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("shortcut", &shortcut); }
