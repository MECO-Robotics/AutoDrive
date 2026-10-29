#include <torch/extension.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime_api.h>

void plan_a_dijkstra_launch(const int64_t*, const double*, const double*,
    const bool*, double*, double*, int64_t*, double*, int64_t*, int64_t,
    int64_t, int64_t, int64_t, double, double, double*, int64_t*, int64_t*,
    hipStream_t);
void plan_a_dijkstra_graph_launch(const double*, const double*, const int64_t*,
    const bool*, const bool*, const double*, const double*, int64_t,
    const int64_t*, const double*, const double*, double*, double*,
    int64_t*, double*, int64_t*, int64_t, int64_t, int64_t, double*, int64_t*,
    int64_t*, double*, double*, uint8_t*, hipStream_t);
void plan_a_dijkstra_frontier_launch(const double*, const double*, const int64_t*,
    const bool*, const bool*, const double*, const double*, int64_t, const int64_t*,
    const double*, const double*, double*, double*, int64_t*, double*, int64_t*,
    int64_t, int64_t, int64_t, double*, int64_t*, int64_t*, double*, double*,
    uint8_t*, int64_t*, int64_t*, int64_t*, int64_t*, int64_t*, bool*, bool*, int64_t*, int64_t*, int, hipStream_t);
void plan_a_dijkstra_batch_launch(const int64_t*, const double*, const double*,
    const bool*, const int64_t*, const double*, const double*, double*, double*,
    int64_t*, double*, int64_t*, int64_t, int64_t, int64_t, int64_t, double*,
    int64_t*, int64_t*, hipStream_t);
void plan_a_pose_regions_launch(const int64_t*, const bool*, const bool*, int64_t*, int64_t*,
    bool*, int64_t*, int64_t, int64_t, int64_t, hipStream_t);
void plan_a_sha256_bytes_launch(const uint8_t*, int64_t, uint8_t*, hipStream_t);

// Experimental preallocated shared-graph batch. Status is per search and must
// be checked by the caller before using any result.
void dijkstra_batch_into(torch::Tensor dst, torch::Tensor cost, torch::Tensor length,
    torch::Tensor allowed, torch::Tensor roots, torch::Tensor starts,
    torch::Tensor connectors, std::vector<torch::Tensor> out) {
  TORCH_CHECK(dst.is_cuda() && dst.scalar_type() == at::kLong && dst.dim() == 2 &&
      dst.is_contiguous(), "dst must be contiguous HIP int64 [N,S]");
  const auto n = dst.size(0), slots = dst.size(1);
  TORCH_CHECK(n > 0 && slots > 0 && n <= (std::numeric_limits<int64_t>::max()-1)/slots,
      "invalid graph shape");
  auto check = [&](const torch::Tensor& t, at::ScalarType type) {
    TORCH_CHECK(t.is_cuda() && t.device() == dst.device() && t.scalar_type() == type &&
        t.is_contiguous(), "batch tensor has invalid device, dtype, or layout");
  };
  check(cost, at::kDouble); check(length, at::kDouble); check(allowed, at::kBool);
  TORCH_CHECK(cost.sizes() == dst.sizes() && length.sizes() == dst.sizes() &&
      allowed.sizes() == dst.sizes(), "graph tensors must match");
  check(roots, at::kLong); check(starts, at::kDouble); check(connectors, at::kDouble);
  TORCH_CHECK(roots.dim() == 1 && roots.numel() > 0 && roots.numel() <= 65535 &&
      starts.sizes() == roots.sizes() && connectors.sizes() == roots.sizes(),
      "batch roots and costs must be equal nonempty vectors");
  const auto b = roots.numel(), cap = n*slots+1;
  TORCH_CHECK(out.size() == 8, "batch requires five outputs, two heaps, and status");
  for (int i = 0; i < 8; ++i) {
    check(out[i], (i == 2 || i == 4 || i == 6 || i == 7) ? at::kLong : at::kDouble);
    if (i < 5) {
      TORCH_CHECK(out[i].dim() == 2 && out[i].size(0) == b && out[i].size(1) == n,
          "invalid batch output shape");
    } else if (i < 7) {
      TORCH_CHECK(out[i].dim() == 2 && out[i].size(0) == b && out[i].size(1) == cap,
          "invalid batch heap shape");
    } else {
      TORCH_CHECK(out[i].dim() == 1 && out[i].size(0) == b, "invalid batch status shape");
    }
  }
  c10::cuda::CUDAGuard guard(dst.device());
  plan_a_dijkstra_batch_launch(dst.data_ptr<int64_t>(), cost.data_ptr<double>(),
      length.data_ptr<double>(), allowed.data_ptr<bool>(), roots.data_ptr<int64_t>(),
      starts.data_ptr<double>(), connectors.data_ptr<double>(), out[0].data_ptr<double>(),
      out[1].data_ptr<double>(), out[2].data_ptr<int64_t>(), out[3].data_ptr<double>(),
      out[4].data_ptr<int64_t>(), n, slots, cap, b, out[5].data_ptr<double>(),
      out[6].data_ptr<int64_t>(), out[7].data_ptr<int64_t>(),
      c10::cuda::getCurrentCUDAStream(dst.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
  TORCH_CHECK(n <= (std::numeric_limits<int64_t>::max() - 1) / slots,
      "graph edge count overflows the HIP heap capacity");
  TORCH_CHECK(std::isfinite(root_start_cost) && std::isfinite(root_connector_m) &&
      root_start_cost >= 0. && root_connector_m >= 0.,
      "root costs must be finite and nonnegative");
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
      heap_capacity, root, root_start_cost, root_connector_m,
      heap_distance.data_ptr<double>(), heap_node.data_ptr<int64_t>(),
      status.data_ptr<int64_t>(),
      c10::cuda::getCurrentCUDAStream(edge_dst.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const auto device_status = status.item<int64_t>();
  TORCH_CHECK(device_status == 0,
      device_status == 1 ? "invalid active edge in Plan A HIP graph" :
      "Plan A HIP heap capacity exceeded");
  return {route, physical, parent, parent_length, traversal};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor> dijkstra_graph(torch::Tensor xy,
    torch::Tensor heading, torch::Tensor edge_dst, torch::Tensor edge_allowed,
    torch::Tensor node_valid, torch::Tensor root, torch::Tensor root_start_cost,
    torch::Tensor root_connector_m, torch::Tensor heading_values, torch::Tensor turns) {
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
  TORCH_CHECK(node_valid.is_cuda() && node_valid.scalar_type() == at::kBool &&
      node_valid.dim() == 1 && node_valid.size(0) == xy.size(0) &&
      node_valid.is_contiguous() && node_valid.device() == xy.device(),
      "node_valid must be contiguous HIP bool [N]");
  TORCH_CHECK(heading_values.is_cuda() && heading_values.device() == xy.device() &&
      heading_values.scalar_type() == at::kDouble && heading_values.dim() == 1 &&
      heading_values.numel() > 0 && heading_values.is_contiguous(),
      "heading_values must be a nonempty contiguous HIP float64 vector");
  TORCH_CHECK(turns.is_cuda() && turns.device() == xy.device() &&
      turns.scalar_type() == at::kDouble && turns.dim() == 2 &&
      turns.size(0) == heading_values.numel() && turns.size(1) == heading_values.numel() &&
      turns.is_contiguous(), "turns must be a matching contiguous HIP float64 matrix");
  const auto n = edge_dst.size(0), slots = edge_dst.size(1);
  TORCH_CHECK(n > 0 && slots > 0, "invalid graph shape");
  TORCH_CHECK(n <= (std::numeric_limits<int64_t>::max() - 1) / slots,
      "graph edge count overflows the HIP heap capacity");
  TORCH_CHECK(root.is_cuda() && root.dim() == 0 &&
      root.scalar_type() == at::kLong && root.device() == xy.device(),
      "root must be a scalar HIP int64 tensor");
  TORCH_CHECK(root_start_cost.is_cuda() && root_start_cost.dim() == 0 &&
      root_start_cost.scalar_type() == at::kDouble &&
      root_start_cost.device() == xy.device(),
      "root_start_cost must be a scalar HIP float64 tensor");
  TORCH_CHECK(root_connector_m.is_cuda() && root_connector_m.dim() == 0 &&
      root_connector_m.scalar_type() == at::kDouble &&
      root_connector_m.device() == xy.device(),
      "root_connector_m must be a scalar HIP float64 tensor");
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
  auto edge_cost = torch::empty({n, slots}, options);
  auto edge_length = torch::empty({n, slots}, options);
  auto edge_error = torch::empty({n, slots}, edge_dst.options().dtype(at::kByte));
  plan_a_dijkstra_graph_launch(xy64.data_ptr<double>(), heading64.data_ptr<double>(),
      edge_dst.data_ptr<int64_t>(), edge_allowed.data_ptr<bool>(),
      node_valid.data_ptr<bool>(), heading_values.data_ptr<double>(),
      turns.data_ptr<double>(), heading_values.numel(),
      root.data_ptr<int64_t>(), root_start_cost.data_ptr<double>(),
      root_connector_m.data_ptr<double>(),
      route.data_ptr<double>(), physical.data_ptr<double>(), parent.data_ptr<int64_t>(),
      parent_length.data_ptr<double>(), traversal.data_ptr<int64_t>(), n, slots,
      heap_capacity,
      heap_distance.data_ptr<double>(), heap_node.data_ptr<int64_t>(),
      status.data_ptr<int64_t>(), edge_cost.data_ptr<double>(), edge_length.data_ptr<double>(),
      edge_error.data_ptr<uint8_t>(), c10::cuda::getCurrentCUDAStream(xy.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // Stable sorting preserves ascending node IDs for equal route costs.
  // Leave unreachable rows and all rows from a failed search as -1.
  auto sorted = at::sort(route, /*stable=*/true, /*dim=*/0, /*descending=*/false);
  traversal = torch::where(torch::isfinite(std::get<0>(sorted)) & (status == 0),
      std::get<1>(sorted), -1);
  return {route, physical, parent, parent_length, traversal, status};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor> dijkstra_graph_frontier(torch::Tensor xy,
    torch::Tensor heading, torch::Tensor edge_dst, torch::Tensor edge_allowed,
    torch::Tensor node_valid, torch::Tensor root, torch::Tensor root_start_cost,
    torch::Tensor root_connector_m, torch::Tensor heading_values, torch::Tensor turns) {
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
  TORCH_CHECK(node_valid.is_cuda() && node_valid.scalar_type() == at::kBool &&
      node_valid.dim() == 1 && node_valid.size(0) == xy.size(0) &&
      node_valid.is_contiguous() && node_valid.device() == xy.device(),
      "node_valid must be contiguous HIP bool [N]");
  TORCH_CHECK(heading_values.is_cuda() && heading_values.device() == xy.device() &&
      heading_values.scalar_type() == at::kDouble && heading_values.dim() == 1 &&
      heading_values.numel() > 0 && heading_values.is_contiguous(),
      "heading_values must be a nonempty contiguous HIP float64 vector");
  TORCH_CHECK(turns.is_cuda() && turns.device() == xy.device() &&
      turns.scalar_type() == at::kDouble && turns.dim() == 2 &&
      turns.size(0) == heading_values.numel() && turns.size(1) == heading_values.numel() &&
      turns.is_contiguous(), "turns must be a matching contiguous HIP float64 matrix");
  const auto n = edge_dst.size(0), slots = edge_dst.size(1);
  TORCH_CHECK(n > 0 && slots > 0, "invalid graph shape");
  TORCH_CHECK(n <= (std::numeric_limits<int64_t>::max() - 1) / slots,
      "graph edge count overflows the HIP heap capacity");
  TORCH_CHECK(root.is_cuda() && root.dim() == 0 &&
      root.scalar_type() == at::kLong && root.device() == xy.device(),
      "root must be a scalar HIP int64 tensor");
  TORCH_CHECK(root_start_cost.is_cuda() && root_start_cost.dim() == 0 &&
      root_start_cost.scalar_type() == at::kDouble &&
      root_start_cost.device() == xy.device(),
      "root_start_cost must be a scalar HIP float64 tensor");
  TORCH_CHECK(root_connector_m.is_cuda() && root_connector_m.dim() == 0 &&
      root_connector_m.scalar_type() == at::kDouble &&
      root_connector_m.device() == xy.device(),
      "root_connector_m must be a scalar HIP float64 tensor");
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
  auto edge_cost = torch::empty({n, slots}, options);
  auto edge_length = torch::empty({n, slots}, options);
  auto edge_error = torch::empty({n, slots}, edge_dst.options().dtype(at::kByte));
  TORCH_CHECK(n <= std::numeric_limits<int64_t>::max() / 64, "reverse adjacency overflow");
  auto counts = torch::zeros({n}, edge_dst.options());
  auto incoming = torch::empty({n, 64}, edge_dst.options());
  auto overflow_edges = torch::empty({4096}, edge_dst.options());
  auto overflow_count = torch::zeros({}, edge_dst.options());
  auto fallback = torch::zeros({}, edge_dst.options());
  auto visited = torch::empty({n}, edge_dst.options().dtype(at::kBool));
  auto frontier = torch::empty({n}, edge_dst.options().dtype(at::kBool));
  auto active = torch::zeros({n}, edge_dst.options());
  auto active_nodes = torch::empty({n}, edge_dst.options());
  hipDeviceProp_t properties;
  TORCH_CHECK(hipGetDeviceProperties(&properties, xy.get_device()) == hipSuccess,
      "failed to query HIP block limit");
  TORCH_CHECK(properties.maxThreadsPerBlock > 0, "invalid HIP block limit");
  int block_size = 1;
  while (block_size < 1024 && block_size * 2 <= properties.maxThreadsPerBlock)
    block_size *= 2;
  plan_a_dijkstra_frontier_launch(xy64.data_ptr<double>(), heading64.data_ptr<double>(),
      edge_dst.data_ptr<int64_t>(), edge_allowed.data_ptr<bool>(),
      node_valid.data_ptr<bool>(), heading_values.data_ptr<double>(),
      turns.data_ptr<double>(), heading_values.numel(),
      root.data_ptr<int64_t>(), root_start_cost.data_ptr<double>(),
      root_connector_m.data_ptr<double>(),
      route.data_ptr<double>(), physical.data_ptr<double>(), parent.data_ptr<int64_t>(),
      parent_length.data_ptr<double>(), traversal.data_ptr<int64_t>(), n, slots,
      heap_capacity,
      heap_distance.data_ptr<double>(), heap_node.data_ptr<int64_t>(),
      status.data_ptr<int64_t>(), edge_cost.data_ptr<double>(), edge_length.data_ptr<double>(),
      edge_error.data_ptr<uint8_t>(), counts.data_ptr<int64_t>(), incoming.data_ptr<int64_t>(),
      fallback.data_ptr<int64_t>(), overflow_edges.data_ptr<int64_t>(),
      overflow_count.data_ptr<int64_t>(), visited.data_ptr<bool>(), frontier.data_ptr<bool>(),
      active.data_ptr<int64_t>(), active_nodes.data_ptr<int64_t>(),
      block_size, c10::cuda::getCurrentCUDAStream(xy.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // Stable sorting preserves ascending node IDs for equal route costs.
  // Leave unreachable rows and all rows from a failed search as -1.
  auto sorted = at::sort(route, /*stable=*/true, /*dim=*/0, /*descending=*/false);
  traversal = torch::where(torch::isfinite(std::get<0>(sorted)) & (status == 0),
      std::get<1>(sorted), -1);
  return {route, physical, parent, parent_length, traversal, status};
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
pose_regions(torch::Tensor edge_dst, torch::Tensor edge_allowed,
             torch::Tensor node_valid, int64_t region_count) {
  TORCH_CHECK(edge_dst.is_cuda() && edge_dst.scalar_type() == at::kLong &&
      edge_dst.dim() == 2 && edge_dst.is_contiguous(),
      "edge_dst must be contiguous HIP int64 [N,D]");
  TORCH_CHECK(edge_allowed.is_cuda() && edge_allowed.scalar_type() == at::kBool &&
      edge_allowed.sizes() == edge_dst.sizes() && edge_allowed.is_contiguous() &&
      edge_allowed.device() == edge_dst.device(),
      "edge_allowed must be matching contiguous HIP bool [N,D]");
  TORCH_CHECK(node_valid.is_cuda() && node_valid.scalar_type() == at::kBool &&
      node_valid.dim() == 1 && node_valid.size(0) == edge_dst.size(0) &&
      node_valid.is_contiguous() && node_valid.device() == edge_dst.device(),
      "node_valid must be contiguous HIP bool [N]");
  const auto n = edge_dst.size(0), degree = edge_dst.size(1);
  TORCH_CHECK(n > 0 && degree > 0 && region_count > 0,
      "pose region graph dimensions must be positive");
  c10::cuda::CUDAGuard guard(edge_dst.device());
  auto region = torch::full({n}, -1, edge_dst.options());
  auto coarse_edges = torch::zeros({region_count, region_count}, edge_dst.options());
  auto coarse_mask = torch::zeros({region_count, region_count}, edge_allowed.options());
  auto status = torch::zeros({}, edge_dst.options());
  plan_a_pose_regions_launch(edge_dst.data_ptr<int64_t>(),
      edge_allowed.data_ptr<bool>(), node_valid.data_ptr<bool>(),
      region.data_ptr<int64_t>(),
      coarse_edges.data_ptr<int64_t>(), coarse_mask.data_ptr<bool>(),
      status.data_ptr<int64_t>(), n, degree, region_count,
      c10::cuda::getCurrentCUDAStream(edge_dst.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {region, coarse_edges, coarse_mask, status};
}

torch::Tensor sha256_bytes(torch::Tensor bytes) {
  TORCH_CHECK(bytes.is_cuda() && bytes.scalar_type() == at::kByte &&
      bytes.dim() == 1 && bytes.is_contiguous(),
      "SHA-256 input must be contiguous HIP uint8 [M]");
  c10::cuda::CUDAGuard guard(bytes.device());
  auto output = torch::empty({32}, bytes.options());
  plan_a_sha256_bytes_launch(bytes.data_ptr<uint8_t>(), bytes.numel(),
      output.data_ptr<uint8_t>(), c10::cuda::getCurrentCUDAStream(bytes.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("dijkstra_batch_into", &dijkstra_batch_into, "Experimental preallocated shared-graph batch");
  module.def("dijkstra", &dijkstra, "Plan A deterministic float64 heap Dijkstra prototype");
  module.def("dijkstra_graph_serial", &dijkstra_graph,
      "Serial Plan A graph route reference");
  module.def("dijkstra_graph", &dijkstra_graph_frontier,
      "Plan A ordered-frontier graph route");
  module.def("pose_regions", &pose_regions,
      "Exact ordered Plan A connected-region partition on HIP");
  module.def("sha256_bytes", &sha256_bytes,
      "SHA-256 over fixed-shape device bytes for post-decision artifacts");
}
