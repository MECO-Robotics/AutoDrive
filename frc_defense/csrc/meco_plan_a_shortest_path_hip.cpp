#include <torch/extension.h>
#include <cmath>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime_api.h>

void plan_a_dijkstra_launch(const int64_t*, const double*, const double*,
    const bool*, double*, double*, int64_t*, double*, int64_t*, int64_t,
    int64_t, int64_t, int64_t, double, double, double*, int64_t*, hipStream_t);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor> dijkstra(torch::Tensor edge_dst,
    torch::Tensor edge_cost, torch::Tensor edge_length, torch::Tensor edge_allowed,
    int64_t root, double root_start_cost, double root_connector_m) {
  TORCH_CHECK(edge_dst.is_cuda() && edge_dst.scalar_type() == at::kLong &&
      edge_dst.dim() == 2 && edge_dst.is_contiguous(),
      "edge_dst must be contiguous HIP int64 [N,S]");
  TORCH_CHECK(edge_cost.is_cuda() && edge_cost.scalar_type() == at::kDouble &&
      edge_cost.sizes() == edge_dst.sizes() && edge_cost.is_contiguous() &&
      edge_cost.device() == edge_dst.device(), "edge_cost must be HIP float64 [N,S]");
  TORCH_CHECK(edge_length.is_cuda() && edge_length.scalar_type() == at::kDouble &&
      edge_length.sizes() == edge_dst.sizes() && edge_length.is_contiguous() &&
      edge_length.device() == edge_dst.device(), "edge_length must be HIP float64 [N,S]");
  TORCH_CHECK(edge_allowed.is_cuda() && edge_allowed.scalar_type() == at::kBool &&
      edge_allowed.sizes() == edge_dst.sizes() && edge_allowed.is_contiguous() &&
      edge_allowed.device() == edge_dst.device(), "edge_allowed must be HIP bool [N,S]");
  const auto n = edge_dst.size(0), slots = edge_dst.size(1);
  TORCH_CHECK(n > 0 && slots > 0 && root >= 0 && root < n,
      "invalid graph shape or root");
  TORCH_CHECK(std::isfinite(root_start_cost) && std::isfinite(root_connector_m) &&
      root_connector_m >= 0., "root costs must be finite and connector nonnegative");
  c10::cuda::CUDAGuard guard(edge_dst.device());
  auto options = edge_cost.options();
  auto route = torch::full({n}, INFINITY, options);
  auto physical = torch::full({n}, INFINITY, options);
  auto parent = torch::full({n}, -1, edge_dst.options());
  auto parent_length = torch::zeros({n}, options);
  auto traversal = torch::full({n}, -1, edge_dst.options());
  auto heap_capacity = n * slots + 1;
  auto heap_node = torch::empty({heap_capacity}, edge_dst.options());
  auto heap_distance = torch::empty({heap_capacity}, options);
  plan_a_dijkstra_launch(edge_dst.data_ptr<int64_t>(), edge_cost.data_ptr<double>(),
      edge_length.data_ptr<double>(), edge_allowed.data_ptr<bool>(),
      route.data_ptr<double>(), physical.data_ptr<double>(), parent.data_ptr<int64_t>(),
      parent_length.data_ptr<double>(), traversal.data_ptr<int64_t>(), n, slots,
      heap_capacity, root, root_start_cost, root_connector_m,
      heap_distance.data_ptr<double>(), heap_node.data_ptr<int64_t>(),
      c10::cuda::getCurrentCUDAStream(edge_dst.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {route, physical, parent, parent_length, traversal};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("dijkstra", &dijkstra, "Plan A deterministic float64 heap Dijkstra prototype");
}
