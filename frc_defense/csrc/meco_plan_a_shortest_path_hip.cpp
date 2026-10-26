#include <torch/extension.h>
#include <cmath>
#include <limits>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime_api.h>

void plan_a_dijkstra_launch(const int64_t*, const double*, const double*,
    const bool*, double*, double*, int64_t*, double*, int64_t*, int64_t,
    int64_t, int64_t, int64_t, const int64_t*, const double*, const double*,
    double*, int64_t*, int64_t*, hipStream_t);
void plan_a_dijkstra_graph_launch(const double*, const double*, const int64_t*,
    const bool*, double*, double*, int64_t*, double*, int64_t*, int64_t,
    int64_t, int64_t, int64_t, const int64_t*, const double*, const double*,
    double*, int64_t*, int64_t*, hipStream_t);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor> dijkstra(torch::Tensor edge_dst,
    torch::Tensor edge_cost, torch::Tensor edge_length, torch::Tensor edge_allowed,
    torch::Tensor root_index, torch::Tensor root_start_cost,
    torch::Tensor root_connector_m) {
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
  TORCH_CHECK(root_index.is_cuda() && root_index.scalar_type() == at::kLong &&
      root_index.dim() == 0 && root_index.is_contiguous() &&
      root_index.device() == edge_dst.device(),
      "root_index must be same-device HIP int64 scalar");
  TORCH_CHECK(root_start_cost.is_cuda() &&
      root_start_cost.scalar_type() == at::kDouble &&
      root_start_cost.dim() == 0 && root_start_cost.is_contiguous() &&
      root_start_cost.device() == edge_dst.device(),
      "root_start_cost must be same-device HIP float64 scalar");
  TORCH_CHECK(root_connector_m.is_cuda() &&
      root_connector_m.scalar_type() == at::kDouble &&
      root_connector_m.dim() == 0 && root_connector_m.is_contiguous() &&
      root_connector_m.device() == edge_dst.device(),
      "root_connector_m must be same-device HIP float64 scalar");
  const auto n = edge_dst.size(0), slots = edge_dst.size(1);
  TORCH_CHECK(n > 0 && slots > 0, "invalid graph shape");
  TORCH_CHECK(n <= (std::numeric_limits<int64_t>::max() - 1) / slots,
      "graph edge count overflows the HIP heap capacity");
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
  auto status = torch::zeros({}, edge_dst.options());
  plan_a_dijkstra_launch(edge_dst.data_ptr<int64_t>(), edge_cost.data_ptr<double>(),
      edge_length.data_ptr<double>(), edge_allowed.data_ptr<bool>(),
      route.data_ptr<double>(), physical.data_ptr<double>(), parent.data_ptr<int64_t>(),
      parent_length.data_ptr<double>(), traversal.data_ptr<int64_t>(), n, slots,
      heap_capacity, root_index.data_ptr<int64_t>(),
      root_start_cost.data_ptr<double>(), root_connector_m.data_ptr<double>(),
      heap_distance.data_ptr<double>(), heap_node.data_ptr<int64_t>(),
      status.data_ptr<int64_t>(),
      c10::cuda::getCurrentCUDAStream(edge_dst.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {route, physical, parent, parent_length, traversal, status};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor> dijkstra_graph(torch::Tensor xy,
    torch::Tensor heading, torch::Tensor edge_dst, torch::Tensor edge_allowed,
    torch::Tensor root_index, torch::Tensor root_start_cost,
    torch::Tensor root_connector_m) {
  TORCH_CHECK(xy.is_cuda() && xy.dim() == 2 && xy.size(1) == 2 &&
      (xy.scalar_type() == at::kFloat || xy.scalar_type() == at::kDouble),
      "xy must be HIP float32/float64 [N,2]");
  TORCH_CHECK(heading.is_cuda() && heading.dim() == 1 &&
      heading.size(0) == xy.size(0) &&
      (heading.scalar_type() == at::kFloat || heading.scalar_type() == at::kDouble) &&
      heading.device() == xy.device(), "heading must match xy on HIP [N]");
  TORCH_CHECK(edge_dst.is_cuda() && edge_dst.scalar_type() == at::kLong &&
      edge_dst.dim() == 2 && edge_dst.size(0) == xy.size(0) &&
      edge_dst.is_contiguous() && edge_dst.device() == xy.device(),
      "edge_dst must be contiguous HIP int64 [N,S]");
  TORCH_CHECK(edge_allowed.is_cuda() && edge_allowed.scalar_type() == at::kBool &&
      edge_allowed.sizes() == edge_dst.sizes() && edge_allowed.is_contiguous() &&
      edge_allowed.device() == xy.device(), "edge_allowed must be HIP bool [N,S]");
  TORCH_CHECK(root_index.is_cuda() && root_index.scalar_type() == at::kLong &&
      root_index.dim() == 0 && root_index.is_contiguous() &&
      root_index.device() == xy.device(),
      "root_index must be same-device HIP int64 scalar");
  TORCH_CHECK(root_start_cost.is_cuda() &&
      root_start_cost.scalar_type() == at::kDouble &&
      root_start_cost.dim() == 0 && root_start_cost.is_contiguous() &&
      root_start_cost.device() == xy.device(),
      "root_start_cost must be same-device HIP float64 scalar");
  TORCH_CHECK(root_connector_m.is_cuda() &&
      root_connector_m.scalar_type() == at::kDouble &&
      root_connector_m.dim() == 0 && root_connector_m.is_contiguous() &&
      root_connector_m.device() == xy.device(),
      "root_connector_m must be same-device HIP float64 scalar");
  const auto n = edge_dst.size(0), slots = edge_dst.size(1);
  TORCH_CHECK(n > 0 && slots > 0, "invalid graph shape");
  TORCH_CHECK(n <= (std::numeric_limits<int64_t>::max() - 1) / slots,
      "graph edge count overflows the HIP heap capacity");
  c10::cuda::CUDAGuard guard(xy.device());
  // Conversion stays on the selected device. The kernel promotes each stored
  // coordinate to double before applying the Python math geometry contract.
  auto xy64 = xy.to(at::kDouble).contiguous();
  auto heading64 = heading.to(at::kDouble).contiguous();
  auto options = xy64.options();
  auto route = torch::full({n}, INFINITY, options);
  auto physical = torch::full({n}, INFINITY, options);
  auto parent = torch::full({n}, -1, edge_dst.options());
  auto parent_length = torch::zeros({n}, options);
  auto traversal = torch::full({n}, -1, edge_dst.options());
  auto heap_capacity = n * slots + 1;
  auto heap_node = torch::empty({heap_capacity}, edge_dst.options());
  auto heap_distance = torch::empty({heap_capacity}, options);
  auto status = torch::zeros({}, edge_dst.options());
  plan_a_dijkstra_graph_launch(xy64.data_ptr<double>(), heading64.data_ptr<double>(),
      edge_dst.data_ptr<int64_t>(), edge_allowed.data_ptr<bool>(),
      route.data_ptr<double>(), physical.data_ptr<double>(), parent.data_ptr<int64_t>(),
      parent_length.data_ptr<double>(), traversal.data_ptr<int64_t>(), n, slots,
      heap_capacity, root_index.data_ptr<int64_t>(),
      root_start_cost.data_ptr<double>(), root_connector_m.data_ptr<double>(),
      heap_distance.data_ptr<double>(), heap_node.data_ptr<int64_t>(),
      status.data_ptr<int64_t>(), c10::cuda::getCurrentCUDAStream(xy.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {route, physical, parent, parent_length, traversal, status};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("dijkstra", &dijkstra, "Plan A deterministic float64 heap Dijkstra prototype");
  module.def("dijkstra_graph", &dijkstra_graph,
      "Plan A device-resident graph geometry and deterministic Dijkstra prototype");
}
