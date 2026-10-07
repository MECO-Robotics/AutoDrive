#include <torch/extension.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime_api.h>

void candidate_rank_launch(const float*, const float*, const bool*, int64_t*,
    bool*, float*, int64_t, int64_t, hipStream_t);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> candidates(
    torch::Tensor points, torch::Tensor robot_xy, torch::Tensor free) {
  TORCH_CHECK(points.is_cuda() && points.scalar_type() == at::kFloat &&
      points.dim() == 3 && points.size(2) == 2 && points.is_contiguous(),
      "points must be contiguous HIP float [N,P,2]");
  const auto n = points.size(0), p = points.size(1);
  TORCH_CHECK(robot_xy.is_cuda() && robot_xy.scalar_type() == at::kFloat &&
      robot_xy.sizes() == torch::IntArrayRef({n, 2}) && robot_xy.is_contiguous() &&
      robot_xy.device() == points.device(), "robot_xy must be contiguous HIP float [N,2]");
  TORCH_CHECK(free.is_cuda() && free.scalar_type() == at::kBool &&
      free.sizes() == torch::IntArrayRef({n, p}) && free.is_contiguous() &&
      free.device() == points.device(), "free must be contiguous HIP bool [N,P]");
  TORCH_CHECK(p >= 4, "at least four candidate positions are required");
  c10::cuda::CUDAGuard guard(points.device());
  auto indices = torch::empty({n, 4}, points.options().dtype(at::kLong));
  auto valid = torch::empty({n, 4}, points.options().dtype(at::kBool));
  auto nearest = torch::empty({n, 4}, points.options());
  candidate_rank_launch(points.data_ptr<float>(), robot_xy.data_ptr<float>(),
      free.data_ptr<bool>(), indices.data_ptr<int64_t>(), valid.data_ptr<bool>(),
      nearest.data_ptr<float>(), n, p,
      c10::cuda::getCurrentCUDAStream(points.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {indices, valid, nearest};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("candidates", &candidates, "Fused fuel-candidate distance and top-four ranking");
}
