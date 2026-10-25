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
    int64_t root, double root_start_cost, double root_connector_m,
    double* heap_distance, int64_t* heap_node, int64_t* status) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  for (int64_t node = 0; node < n; ++node) {
    route[node] = INFINITY;
    physical[node] = INFINITY;
    parent[node] = -1;
    parent_length[node] = 0.0;
    traversal[node] = -1;
  }
  route[root] = root_start_cost;
  physical[root] = root_connector_m;
  int64_t heap_size = 0;
  int64_t settled_count = 0;
  heap_push(heap_distance, heap_node, &heap_size, root_start_cost, root);
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
    const int64_t* root_index, const double* root_start_cost_ptr,
    const double* root_connector_m_ptr,
    double* route, double* physical, int64_t* parent, double* parent_length,
    int64_t* traversal, int64_t n, int64_t slots, int64_t heap_capacity,
    double* heap_distance, int64_t* heap_node, int64_t* status) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  for (int64_t node = 0; node < n; ++node) {
    route[node] = INFINITY;
    physical[node] = INFINITY;
    parent[node] = -1;
    parent_length[node] = 0.0;
    traversal[node] = -1;
  }
  const int64_t root = root_index[0];
  const double root_start_cost = root_start_cost_ptr[0];
  const double root_connector_m = root_connector_m_ptr[0];
  if (root < 0 || root >= n || !isfinite(root_start_cost) ||
      !isfinite(root_connector_m) || root_start_cost < 0.0 ||
      root_connector_m < 0.0) {
    *status = 3;
    return;
  }
  route[root] = root_start_cost;
  physical[root] = root_connector_m;
  int64_t heap_size = 0;
  int64_t settled_count = 0;
  heap_push(heap_distance, heap_node, &heap_size, root_start_cost, root);
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

// Exact ordered connectivity partition used by the Python oracle. A single
// thread preserves root/edge FIFO order and the historical low-ID tie break.
__global__ void plan_a_pose_regions_kernel(const int64_t* edge_dst,
    const bool* edge_allowed, int64_t* region, int64_t* coarse_edges,
    bool* coarse_mask, int64_t* status, int64_t n, int64_t degree,
    int64_t region_count) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int64_t* component = new int64_t[n];
  int64_t* queue = new int64_t[n];
  int64_t* component_sizes = new int64_t[n];
  int64_t* allocations = new int64_t[n];
  int64_t* members = new int64_t[n];
  bool* adjacency = new bool[region_count * region_count];
  for (int64_t i = 0; i < n; ++i) {
    component[i] = -1;
    region[i] = -1;
  }
  int64_t component_count = 0;
  int64_t member_count = 0;
  for (int64_t start = 0; start < n; ++start) {
    if (component[start] >= 0) continue;
    const int64_t label = component_count++;
    int64_t head = 0, tail = 0;
    component[start] = label;
    queue[tail++] = start;
    component_sizes[label] = 0;
    while (head < tail) {
      const int64_t current = queue[head++];
      members[member_count++] = current;
      component_sizes[label] += 1;
      for (int64_t slot = 0; slot < degree; ++slot) {
        const int64_t offset = current * degree + slot;
        if (!edge_allowed[offset]) continue;
        const int64_t next = edge_dst[offset];
        if (next < 0 || next >= n) {
          *status = 1;
          delete[] component; delete[] queue; delete[] component_sizes;
          delete[] allocations; delete[] members; delete[] adjacency;
          return;
        }
        if (component[next] < 0) {
          component[next] = label;
          queue[tail++] = next;
        }
      }
    }
    // Separator records component BFS member order in the flat `members`
    // array; component sizes recover each component's exact slice.
  }
  if (component_count > region_count) {
    *status = 2;
    delete[] component; delete[] queue; delete[] component_sizes;
    delete[] allocations; delete[] members; delete[] adjacency;
    return;
  }
  for (int64_t c = 0; c < component_count; ++c) allocations[c] = 1;
  for (int64_t spare = component_count; spare < region_count; ++spare) {
    int64_t best = 0;
    for (int64_t c = 1; c < component_count; ++c) {
      // Compare size/allocation ratios exactly; equal ratios retain lower ID.
      if (component_sizes[c] * allocations[best] >
          component_sizes[best] * allocations[c]) best = c;
    }
    allocations[best] += 1;
  }
  int64_t region_id = 0, member_offset = 0;
  for (int64_t c = 0; c < component_count; ++c) {
    const int64_t size = component_sizes[c];
    const int64_t allocation = allocations[c];
    int64_t* seeds = new int64_t[allocation];
    int64_t seed_count = 0;
    for (int64_t i = 0; i < allocation; ++i) {
      const int64_t raw_index = (2 * i + 1) * size / (2 * allocation);
      const int64_t index = raw_index < size ? raw_index : size - 1;
      const int64_t seed = members[member_offset + index];
      if (region[seed] < 0) {
        region[seed] = region_id++;
        seeds[seed_count++] = seed;
      }
    }
    int64_t head = 0, tail = seed_count;
    for (int64_t i = 0; i < seed_count; ++i) queue[i] = seeds[i];
    delete[] seeds;
    while (head < tail) {
      const int64_t current = queue[head++];
      for (int64_t slot = 0; slot < degree; ++slot) {
        const int64_t offset = current * degree + slot;
        if (!edge_allowed[offset]) continue;
        const int64_t next = edge_dst[offset];
        if (region[next] < 0) {
          region[next] = region[current];
          queue[tail++] = next;
        }
      }
    }
    member_offset += size;
  }
  for (int64_t row = 0; row < region_count * region_count; ++row) {
    coarse_edges[row] = 0;
    coarse_mask[row] = false;
    adjacency[row] = false;
  }
  // A fixed-width bitmap makes coarse adjacency deterministic and avoids a
  // host read to discover the maximum degree. Write sorted neighbors by scan.
  for (int64_t src = 0; src < n; ++src) {
    const int64_t src_region = region[src];
    for (int64_t slot = 0; slot < degree; ++slot) {
      const int64_t offset = src * degree + slot;
      if (!edge_allowed[offset]) continue;
      const int64_t dst_region = region[edge_dst[offset]];
      if (src_region != dst_region)
        adjacency[src_region * region_count + dst_region] = true;
    }
  }
  for (int64_t r = 0; r < region_count; ++r) {
    int64_t column = 0;
    for (int64_t neighbor = 0; neighbor < region_count; ++neighbor) {
      if (!adjacency[r * region_count + neighbor]) continue;
      const int64_t output = r * region_count + column++;
      coarse_edges[output] = neighbor;
      coarse_mask[output] = true;
    }
  }
  delete[] component;
  delete[] queue;
  delete[] component_sizes;
  delete[] allocations;
  delete[] members;
  delete[] adjacency;
}

void plan_a_dijkstra_launch(const int64_t* edge_dst, const double* edge_cost,
    const double* edge_length, const bool* edge_allowed, double* route,
    double* physical, int64_t* parent, double* parent_length,
    int64_t* traversal, int64_t n, int64_t slots, int64_t heap_capacity,
    int64_t root, double root_start_cost, double root_connector_m,
    double* heap_distance, int64_t* heap_node, int64_t* status,
    hipStream_t stream) {
  hipLaunchKernelGGL(plan_a_dijkstra_kernel, dim3(1), dim3(1), 0, stream,
      edge_dst, edge_cost, edge_length, edge_allowed, route, physical, parent,
      parent_length, traversal, n, slots, heap_capacity, root, root_start_cost,
      root_connector_m, heap_distance, heap_node, status);
}

void plan_a_dijkstra_graph_launch(const double* xy, const double* heading,
    const int64_t* edge_dst, const bool* edge_allowed,
    const int64_t* root_index, const double* root_start_cost,
    const double* root_connector_m, double* route, double* physical,
    int64_t* parent, double* parent_length, int64_t* traversal, int64_t n,
    int64_t slots, int64_t heap_capacity, double* heap_distance,
    int64_t* heap_node, int64_t* status, hipStream_t stream) {
  hipLaunchKernelGGL(plan_a_dijkstra_graph_kernel, dim3(1), dim3(1), 0, stream,
      xy, heading, edge_dst, edge_allowed, root_index, root_start_cost,
      root_connector_m, route, physical, parent, parent_length, traversal, n,
      slots, heap_capacity, heap_distance, heap_node, status);
}

void plan_a_pose_regions_launch(const int64_t* edge_dst,
    const bool* edge_allowed, int64_t* region, int64_t* coarse_edges,
    bool* coarse_mask, int64_t* status, int64_t n, int64_t degree,
    int64_t region_count, hipStream_t stream) {
  hipLaunchKernelGGL(plan_a_pose_regions_kernel, dim3(1), dim3(1), 0, stream,
      edge_dst, edge_allowed, region, coarse_edges, coarse_mask, status, n,
      degree, region_count);
}
