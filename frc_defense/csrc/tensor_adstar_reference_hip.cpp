#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime_api.h>

void path_reference_launch(const float*, const int64_t*, const float*,
                           const float*, const float*, int64_t,
                           const float*, int64_t, const float*, int64_t,
                           float*, float*, int, hipStream_t);

std::tuple<torch::Tensor, torch::Tensor> path_reference(
    torch::Tensor path, torch::Tensor lengths, torch::Tensor goal,
    torch::Tensor speed_profile, torch::Tensor position, torch::Tensor velocity,
    torch::Tensor speed_limit) {
  TORCH_CHECK(path.is_cuda() && path.scalar_type() == at::kFloat && path.dim() == 3 &&
              path.size(1) == 72 && path.size(2) == 2,
              "AD* path reference expects CUDA float32 [N,72,2] path");
  const auto n = path.size(0);
  TORCH_CHECK(lengths.sizes() == torch::IntArrayRef({n}) &&
              lengths.scalar_type() == at::kLong &&
              goal.sizes() == torch::IntArrayRef({n, 2}) &&
              speed_profile.sizes() == torch::IntArrayRef({n, 72}) &&
              position.sizes() == torch::IntArrayRef({n, 2}) &&
              velocity.sizes() == torch::IntArrayRef({n, 2}) &&
              speed_limit.sizes() == torch::IntArrayRef({n}),
              "AD* path reference input shape mismatch");
  for (const auto& t : {path, goal, speed_profile, position, velocity, speed_limit})
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kFloat &&
                t.device() == path.device(),
                "AD* path reference float inputs must use one device and float32");
  TORCH_CHECK(path.is_contiguous() && goal.is_contiguous() && speed_profile.is_contiguous() &&
              position.stride(1) == 1 && velocity.stride(1) == 1,
              "AD* path reference path, goal, profile and inner state strides are invalid");
  TORCH_CHECK(lengths.is_cuda() && lengths.is_contiguous() &&
              lengths.device() == path.device(),
              "AD* path reference lengths must be contiguous on the same device");
  c10::cuda::CUDAGuard guard(path.device());
  auto command = torch::empty({n, 2}, path.options());
  auto tangent = torch::empty_like(command);
  path_reference_launch(path.data_ptr<float>(), lengths.data_ptr<int64_t>(),
      goal.data_ptr<float>(), speed_profile.data_ptr<float>(), position.data_ptr<float>(),
      position.stride(0), velocity.data_ptr<float>(), velocity.stride(0),
      speed_limit.data_ptr<float>(), speed_limit.stride(0), command.data_ptr<float>(),
      tangent.data_ptr<float>(), static_cast<int>(n),
      c10::cuda::getCurrentCUDAStream(path.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {command, tangent};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("path_reference", &path_reference,
        "Fused exact AD* path projection and velocity reference (HIP, strict FP)");
}
