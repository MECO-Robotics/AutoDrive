#include <torch/extension.h>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime_api.h>
void plan_a_sha256_chunks_launch(const uint8_t*, int64_t, int64_t, uint8_t*, hipStream_t);
torch::Tensor sha256_chunks(torch::Tensor input, int64_t chunk_size) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kByte && input.dim() == 1 && input.is_contiguous(), "expected contiguous HIP byte vector");
  TORCH_CHECK(input.numel() > 0 && chunk_size > 0, "input and chunk size must be positive");
  c10::cuda::CUDAGuard guard(input.device());
  auto count = (input.numel() + chunk_size - 1) / chunk_size;
  auto output = torch::empty({count, 32}, input.options());
  plan_a_sha256_chunks_launch(input.data_ptr<uint8_t>(), input.numel(), chunk_size,
      output.data_ptr<uint8_t>(), c10::cuda::getCurrentCUDAStream(input.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("sha256_chunks", &sha256_chunks);
}
