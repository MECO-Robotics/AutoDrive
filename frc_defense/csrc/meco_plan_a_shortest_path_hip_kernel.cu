#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

__device__ bool less_pair(double da, int64_t na, double db, int64_t nb) {
  return da < db || (da == db && na < nb);
}

__device__ void heap_push(double* distances, int64_t* nodes, int64_t* size,
                          double distance, int64_t node) {
  int64_t index = (*size)++;
  while (index > 0) {
    const int64_t parent = (index - 1) / 2;
    if (!less_pair(distance, node, distances[parent], nodes[parent])) break;
    distances[index] = distances[parent];
    nodes[index] = nodes[parent];
    index = parent;
  }
  distances[index] = distance;
  nodes[index] = node;
}

__device__ void heap_pop(double* distances, int64_t* nodes, int64_t* size,
                         double* distance, int64_t* node) {
  *distance = distances[0];
  *node = nodes[0];
  const int64_t last = --(*size);
  if (last == 0) return;
  const double tail_distance = distances[last];
  const int64_t tail_node = nodes[last];
  int64_t index = 0;
  while (true) {
    const int64_t left = index * 2 + 1;
    if (left >= last) break;
    const int64_t right = left + 1;
    int64_t child = left;
    if (right < last && less_pair(distances[right], nodes[right],
                                  distances[left], nodes[left])) child = right;
    if (!less_pair(distances[child], nodes[child], tail_distance, tail_node)) break;
    distances[index] = distances[child];
    nodes[index] = nodes[child];
    index = child;
  }
  distances[index] = tail_distance;
  nodes[index] = tail_node;
}

__global__ void plan_a_dijkstra_kernel(const int64_t* edge_dst,
    const double* edge_cost, const double* edge_length, const bool* edge_allowed,
    double* route, double* physical, int64_t* parent, double* parent_length,
    int64_t* traversal, int64_t n, int64_t slots, int64_t heap_capacity,
    const int64_t* root_index, const double* root_start_cost,
    const double* root_connector_m,
    double* heap_distance, int64_t* heap_node, int64_t* status) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  *status = 0;
  for (int64_t node = 0; node < n; ++node) {
    route[node] = INFINITY;
    physical[node] = INFINITY;
    parent[node] = -1;
    parent_length[node] = 0.0;
    traversal[node] = -1;
  }
  const int64_t root = *root_index;
  const double start_cost = *root_start_cost;
  const double connector_m = *root_connector_m;
  if (root < 0 || root >= n) {
    *status = 3;
    return;
  }
  if (!isfinite(start_cost) || start_cost < 0.0 ||
      !isfinite(connector_m) || connector_m < 0.0) {
    *status = 4;
    return;
  }
  route[root] = start_cost;
  physical[root] = connector_m;
  int64_t heap_size = 0;
  int64_t settled_count = 0;
  heap_push(heap_distance, heap_node, &heap_size, start_cost, root);
  while (heap_size > 0) {
    double distance;
    int64_t src;
    heap_pop(heap_distance, heap_node, &heap_size, &distance, &src);
    if (distance != route[src]) continue;
    traversal[settled_count++] = src;
    for (int64_t slot = 0; slot < slots; ++slot) {
      const int64_t offset = src * slots + slot;
      if (!edge_allowed[offset]) continue;
      const int64_t dst = edge_dst[offset];
      const double cost = edge_cost[offset];
      const double length = edge_length[offset];
      if (dst < 0 || dst >= n || !isfinite(cost) || cost < 0.0 ||
          !isfinite(length) || length < 0.0) {
        *status = 1;
        return;
      }
      const double proposal = distance + cost;
      if (!isfinite(proposal)) {
        *status = 1;
        return;
      }
      if (proposal < route[dst] - 1.0e-9) {
        route[dst] = proposal;
        physical[dst] = physical[src] + length;
        if (!isfinite(physical[dst])) {
          *status = 1;
          return;
        }
        parent[dst] = src;
        parent_length[dst] = length;
        if (heap_size >= heap_capacity) {
          *status = 2;
          return;
        }
        heap_push(heap_distance, heap_node, &heap_size, proposal, dst);
      }
    }
  }
}

__global__ void plan_a_dijkstra_graph_kernel(const double* xy,
    const double* heading, const int64_t* edge_dst, const bool* edge_allowed,
    double* route, double* physical, int64_t* parent, double* parent_length,
    int64_t* traversal, int64_t n, int64_t slots, int64_t heap_capacity,
    const int64_t* root_index, const double* root_start_cost,
    const double* root_connector_m,
    double* heap_distance, int64_t* heap_node, int64_t* status) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  *status = 0;
  for (int64_t node = 0; node < n; ++node) {
    route[node] = INFINITY;
    physical[node] = INFINITY;
    parent[node] = -1;
    parent_length[node] = 0.0;
    traversal[node] = -1;
  }
  const int64_t root = *root_index;
  const double start_cost = *root_start_cost;
  const double connector_m = *root_connector_m;
  if (root < 0 || root >= n) {
    *status = 3;
    return;
  }
  if (!isfinite(start_cost) || start_cost < 0.0 ||
      !isfinite(connector_m) || connector_m < 0.0) {
    *status = 4;
    return;
  }
  route[root] = start_cost;
  physical[root] = connector_m;
  int64_t heap_size = 0;
  int64_t settled_count = 0;
  heap_push(heap_distance, heap_node, &heap_size, start_cost, root);
  while (heap_size > 0) {
    double distance;
    int64_t src;
    heap_pop(heap_distance, heap_node, &heap_size, &distance, &src);
    if (distance != route[src]) continue;
    traversal[settled_count++] = src;
    for (int64_t slot = 0; slot < slots; ++slot) {
      const int64_t offset = src * slots + slot;
      if (!edge_allowed[offset]) continue;
      const int64_t dst = edge_dst[offset];
      if (dst < 0 || dst >= n) {
        *status = 1;
        return;
      }
      const double sx = xy[src * 2], sy = xy[src * 2 + 1];
      const double dx = xy[dst * 2], dy = xy[dst * 2 + 1];
      const double hs = heading[src], hd = heading[dst];
      if (!isfinite(sx) || !isfinite(sy) || !isfinite(dx) || !isfinite(dy) ||
          !isfinite(hs) || !isfinite(hd)) {
        *status = 1;
        return;
      }
      const double length = hypot(sx - dx, sy - dy);
      const double delta = hd - hs;
      const double turn = fabs(atan2(sin(delta), cos(delta)));
      const double cost = length + 0.25 * turn;
      if (!isfinite(length) || !isfinite(cost) || length < 0.0 || cost < 0.0) {
        *status = 1;
        return;
      }
      const double proposal = distance + cost;
      if (!isfinite(proposal)) {
        *status = 1;
        return;
      }
      if (proposal < route[dst] - 1.0e-9) {
        route[dst] = proposal;
        physical[dst] = physical[src] + length;
        if (!isfinite(physical[dst])) {
          *status = 1;
          return;
        }
        parent[dst] = src;
        parent_length[dst] = length;
        if (heap_size >= heap_capacity) {
          *status = 2;
          return;
        }
        heap_push(heap_distance, heap_node, &heap_size, proposal, dst);
      }
    }
  }
}

void plan_a_dijkstra_launch(const int64_t* edge_dst, const double* edge_cost,
    const double* edge_length, const bool* edge_allowed, double* route,
    double* physical, int64_t* parent, double* parent_length,
    int64_t* traversal, int64_t n, int64_t slots, int64_t heap_capacity,
    const int64_t* root_index, const double* root_start_cost,
    const double* root_connector_m,
    double* heap_distance, int64_t* heap_node, int64_t* status,
    hipStream_t stream) {
  hipLaunchKernelGGL(plan_a_dijkstra_kernel, dim3(1), dim3(1), 0, stream,
      edge_dst, edge_cost, edge_length, edge_allowed, route, physical, parent,
      parent_length, traversal, n, slots, heap_capacity, root_index,
      root_start_cost, root_connector_m, heap_distance, heap_node, status);
}

void plan_a_dijkstra_graph_launch(const double* xy, const double* heading,
    const int64_t* edge_dst, const bool* edge_allowed, double* route,
    double* physical, int64_t* parent, double* parent_length,
    int64_t* traversal, int64_t n, int64_t slots, int64_t heap_capacity,
    const int64_t* root_index, const double* root_start_cost,
    const double* root_connector_m,
    double* heap_distance, int64_t* heap_node, int64_t* status,
    hipStream_t stream) {
  hipLaunchKernelGGL(plan_a_dijkstra_graph_kernel, dim3(1), dim3(1), 0, stream,
      xy, heading, edge_dst, edge_allowed, route, physical, parent,
      parent_length, traversal, n, slots, heap_capacity, root_index,
      root_start_cost, root_connector_m, heap_distance, heap_node, status);
}
