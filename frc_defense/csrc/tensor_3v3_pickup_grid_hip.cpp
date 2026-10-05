#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime.h>

void pickup_grid_launch(const bool*, const float*, const float*, const float*,
    const float*, const bool*, int64_t*, bool*, const int16_t*, float*,
    const float*, const bool*, const bool*, const bool*, int64_t*, bool*, int,
    int, int, int, int, float, int, bool, hipStream_t);

void pickup_grid(torch::Tensor active, torch::Tensor pose, torch::Tensor length,
    torch::Tensor width, torch::Tensor piece_pos, torch::Tensor piece_active,
    torch::Tensor piece_owner, torch::Tensor free, torch::Tensor possible_cells,
    torch::Tensor next_intake, torch::Tensor elapsed, torch::Tensor controlled,
    torch::Tensor deterministic, torch::Tensor defense_role,
    torch::Tensor acquired_event, torch::Tensor track_mask,
    int64_t robot, int64_t nx, int64_t ny,
    double cell_size, int64_t capacity, bool allow_sweep) {
  pickup_grid_launch(active.data_ptr<bool>(), pose.data_ptr<float>(),
      length.data_ptr<float>(), width.data_ptr<float>(), piece_pos.data_ptr<float>(),
      piece_active.data_ptr<bool>(), piece_owner.data_ptr<int64_t>(),
      free.data_ptr<bool>(), possible_cells.data_ptr<int16_t>(), next_intake.data_ptr<float>(),
      elapsed.data_ptr<float>(), controlled.data_ptr<bool>(),
      deterministic.data_ptr<bool>(), defense_role.data_ptr<bool>(),
      acquired_event.data_ptr<int64_t>(), track_mask.data_ptr<bool>(),
      active.size(0), piece_active.size(1),
      static_cast<int>(robot), static_cast<int>(nx), static_cast<int>(ny),
      static_cast<float>(cell_size), static_cast<int>(capacity), allow_sweep,
      c10::cuda::getCurrentCUDAStream(active.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pickup", &pickup_grid, "Fused spatially-filtered 3v3 FUEL pickup (HIP)");
}
