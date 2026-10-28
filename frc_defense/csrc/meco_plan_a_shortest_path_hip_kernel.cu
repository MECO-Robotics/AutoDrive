#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t sha_rotr(uint32_t value, uint32_t shift) {
  return (value >> shift) | (value << (32 - shift));
}

__global__ void plan_a_sha256_bytes_kernel(const uint8_t* input, int64_t length,
                                           uint8_t* output) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const uint32_t k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
  uint32_t state[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                       0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  const int64_t padded = ((length + 9 + 63) / 64) * 64;
  const uint64_t bit_length = static_cast<uint64_t>(length) * 8;
  for (int64_t base = 0; base < padded; base += 64) {
    uint32_t words[64];
    for (int i = 0; i < 16; ++i) {
      uint32_t value = 0;
      for (int j = 0; j < 4; ++j) {
        const int64_t position = base + i * 4 + j;
        uint8_t byte = 0;
        if (position < length) byte = input[position];
        else if (position == length) byte = 0x80;
        else if (position >= padded - 8) {
          const int shift = static_cast<int>((padded - 1 - position) * 8);
          byte = static_cast<uint8_t>((bit_length >> shift) & 0xff);
        }
        value = (value << 8) | byte;
      }
      words[i] = value;
    }
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = sha_rotr(words[i - 15], 7) ^ sha_rotr(words[i - 15], 18) ^ (words[i - 15] >> 3);
      const uint32_t s1 = sha_rotr(words[i - 2], 17) ^ sha_rotr(words[i - 2], 19) ^ (words[i - 2] >> 10);
      words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }
    uint32_t a=state[0], b=state[1], c=state[2], d=state[3];
    uint32_t e=state[4], f=state[5], g=state[6], h=state[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t s1 = sha_rotr(e, 6) ^ sha_rotr(e, 11) ^ sha_rotr(e, 25);
      const uint32_t choice = (e & f) ^ ((~e) & g);
      const uint32_t temp1 = h + s1 + choice + k[i] + words[i];
      const uint32_t s0 = sha_rotr(a, 2) ^ sha_rotr(a, 13) ^ sha_rotr(a, 22);
      const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t temp2 = s0 + majority;
      h=g; g=f; f=e; e=d+temp1; d=c; c=b; b=a; a=temp1+temp2;
    }
    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
  }
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 4; ++j)
      output[i * 4 + j] = static_cast<uint8_t>((state[i] >> (24 - j * 8)) & 0xff);
}

void plan_a_sha256_bytes_launch(const uint8_t* input, int64_t length,
    uint8_t* output, hipStream_t stream) {
  hipLaunchKernelGGL(plan_a_sha256_bytes_kernel, dim3(1), dim3(1), 0, stream,
      input, length, output);
}

// Adaptation of CPython 3.12 vector_norm (PSF License 2.0):
// https://github.com/python/cpython/blob/v3.12.14/Modules/mathmodule.c
// Specialized for two operands and HIP, with nonrecursive subnormal scaling.
__device__ double plan_a_python_hypot(double a,double b) {
  a=fabs(a); b=fabs(b);
  double max=fmax(a,b);
  if (max==0.0 || isinf(max)) return max;
  int exponent; frexp(max,&exponent);
  if (exponent < -1023) {
    // Graph coordinates are normally far from this range; use explicit
    // rescaling to avoid recursive device calls for subnormals.
    a /= 0x1p-1022; b /= 0x1p-1022;
    max=fmax(a,b); frexp(max,&exponent);
    double scale=ldexp(1.0,-exponent), csum=1.0, f1=0.0, f2=0.0;
    double values[2]={a*scale,b*scale};
    for(int i=0;i<2;++i) { double x=values[i], hi=x*x, lo=fma(x,x,-hi), sum=csum+hi; f1+=lo; f2+=(csum-sum)+hi; csum=sum; }
    double h=sqrt(csum-1.0+(f1+f2)), hi=-h*h, lo=fma(-h,h,-hi), sum=csum+hi;
    f1+=lo; f2+=(csum-sum)+hi; csum=sum; h+=(csum-1.0+(f1+f2))/(2.0*h);
    return 0x1p-1022*(h/scale);
  }
  double scale=ldexp(1.0,-exponent), csum=1.0, f1=0.0, f2=0.0;
  double values[2]={a*scale,b*scale};
  for(int i=0;i<2;++i) { double x=values[i], hi=x*x, lo=fma(x,x,-hi), sum=csum+hi; f1+=lo; f2+=(csum-sum)+hi; csum=sum; }
  double h=sqrt(csum-1.0+(f1+f2)), hi=-h*h, lo=fma(-h,h,-hi), sum=csum+hi;
  f1+=lo; f2+=(csum-sum)+hi; csum=sum; h+=(csum-1.0+(f1+f2))/(2.0*h);
  return h/scale;
}

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

__device__ void sorted_traversal(const double* route, int64_t* traversal,
    int64_t n, double* heap_distance, int64_t* heap_node) {
  int64_t size = 0;
  for (int64_t node = 0; node < n; ++node)
    if (isfinite(route[node]))
      heap_push(heap_distance, heap_node, &size, route[node], node);
  int64_t count = 0;
  while (size > 0) {
    double distance;
    int64_t node;
    heap_pop(heap_distance, heap_node, &size, &distance, &node);
    traversal[count++] = node;
  }
}

__device__ void plan_a_dijkstra_search(const int64_t* edge_dst,
    const double* edge_cost, const double* edge_length, const bool* edge_allowed,
    double* route, double* physical, int64_t* parent, double* parent_length,
    int64_t* traversal, int64_t n, int64_t slots, int64_t heap_capacity,
    int64_t root, double root_start_cost, double root_connector_m,
    double* heap_distance, int64_t* heap_node, int64_t* status) {
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
  heap_push(heap_distance, heap_node, &heap_size, root_start_cost, root);
  while (heap_size > 0) {
    double distance;
    int64_t src;
    heap_pop(heap_distance, heap_node, &heap_size, &distance, &src);
    if (distance != route[src]) continue;
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
  sorted_traversal(route, traversal, n, heap_distance, heap_node);
}

__global__ void plan_a_dijkstra_kernel(const int64_t* edge_dst,
    const double* edge_cost, const double* edge_length, const bool* edge_allowed,
    double* route, double* physical, int64_t* parent, double* parent_length,
    int64_t* traversal, int64_t n, int64_t slots, int64_t heap_capacity,
    int64_t root, double root_start_cost, double root_connector_m,
    double* heap_distance, int64_t* heap_node, int64_t* status) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  plan_a_dijkstra_search(edge_dst, edge_cost, edge_length, edge_allowed,
      route, physical, parent, parent_length, traversal, n, slots, heap_capacity,
      root, root_start_cost, root_connector_m, heap_distance, heap_node, status);
}

// Experimental shared-graph batch: each block owns all mutable search state.
__global__ void plan_a_dijkstra_batch_kernel(const int64_t* edge_dst,
    const double* edge_cost, const double* edge_length, const bool* edge_allowed,
    const int64_t* roots, const double* start_costs, const double* connectors,
    double* route, double* physical, int64_t* parent, double* parent_length,
    int64_t* traversal, int64_t n, int64_t slots, int64_t heap_capacity,
    double* heap_distance, int64_t* heap_node, int64_t* status) {
  if (threadIdx.x != 0) return;
  const int64_t b = blockIdx.x;
  status[b] = 0;
  if (roots[b] < 0 || roots[b] >= n || !isfinite(start_costs[b]) ||
      !isfinite(connectors[b]) || start_costs[b] < 0 || connectors[b] < 0) {
    status[b] = 3;
    return;
  }
  plan_a_dijkstra_search(edge_dst, edge_cost, edge_length, edge_allowed,
      route + b*n, physical + b*n, parent + b*n, parent_length + b*n,
      traversal + b*n, n, slots, heap_capacity, roots[b], start_costs[b],
      connectors[b], heap_distance + b*heap_capacity, heap_node + b*heap_capacity,
      status + b);
}

__global__ void plan_a_dijkstra_graph_kernel(const double* xy,
    const double* heading, const int64_t* edge_dst, const bool* edge_allowed,
    const bool* node_valid, const double* heading_values, const double* turns,
    int64_t heading_count,
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
  if (root < 0 || root >= n || !node_valid[root] || !isfinite(root_start_cost) ||
      !isfinite(root_connector_m) || root_start_cost < 0.0 ||
      root_connector_m < 0.0) {
    *status = 3;
    return;
  }
  route[root] = root_start_cost;
  physical[root] = root_connector_m;
  int64_t heap_size = 0;
  heap_push(heap_distance, heap_node, &heap_size, root_start_cost, root);
  while (heap_size > 0) {
    double distance;
    int64_t src;
    heap_pop(heap_distance, heap_node, &heap_size, &distance, &src);
    if (distance != route[src]) continue;
    for (int64_t slot = 0; slot < slots; ++slot) {
      const int64_t offset = src * slots + slot;
      if (!edge_allowed[offset]) continue;
      const int64_t dst = edge_dst[offset];
      if (dst < 0 || dst >= n || !node_valid[dst]) {
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
      const double length = plan_a_python_hypot(sx - dx, sy - dy);
      int64_t hi = -1, hj = -1;
      for (int64_t i = 0; i < heading_count; ++i) {
        if (heading_values[i] == hs) hi = i;
        if (heading_values[i] == hd) hj = i;
      }
      if (hi < 0 || hj < 0) {
        *status = 4;
        return;
      }
      const double turn = turns[hi * heading_count + hj];
      const double cost = length + 0.25 * turn;
      if (!isfinite(length) || !isfinite(turn) || !isfinite(cost) ||
          length < 0.0 || turn < 0.0 || cost < 0.0) {
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
    const bool* edge_allowed, const bool* node_valid, int64_t* region, int64_t* coarse_edges,
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
    region[i] = node_valid[i] ? -1 : 0;
  }
  int64_t component_count = 0;
  int64_t member_count = 0;
  for (int64_t start = 0; start < n; ++start) {
    if (!node_valid[start]) continue;
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
        if (next < 0 || next >= n || !node_valid[next]) {
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
  if (component_count == 0) {
    *status = 3;
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
        if (node_valid[next] && region[next] < 0) {
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
    if (!node_valid[src]) continue;
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

void plan_a_dijkstra_batch_launch(const int64_t* edge_dst, const double* edge_cost,
    const double* edge_length, const bool* edge_allowed, const int64_t* roots,
    const double* starts, const double* connectors, double* route, double* physical,
    int64_t* parent, double* parent_length, int64_t* traversal, int64_t n,
    int64_t slots, int64_t heap_capacity, int64_t batch, double* heap_distance,
    int64_t* heap_node, int64_t* status, hipStream_t stream) {
  hipLaunchKernelGGL(plan_a_dijkstra_batch_kernel, dim3(batch), dim3(1), 0, stream,
      edge_dst, edge_cost, edge_length, edge_allowed, roots, starts, connectors,
      route, physical, parent, parent_length, traversal, n, slots, heap_capacity,
      heap_distance, heap_node, status);
}

void plan_a_dijkstra_graph_launch(const double* xy, const double* heading,
    const int64_t* edge_dst, const bool* edge_allowed, const bool* node_valid,
    const double* heading_values, const double* turns, int64_t heading_count,
    const int64_t* root_index, const double* root_start_cost,
    const double* root_connector_m, double* route, double* physical,
    int64_t* parent, double* parent_length, int64_t* traversal, int64_t n,
    int64_t slots, int64_t heap_capacity, double* heap_distance,
    int64_t* heap_node, int64_t* status, hipStream_t stream) {
  hipLaunchKernelGGL(plan_a_dijkstra_graph_kernel, dim3(1), dim3(1), 0, stream,
      xy, heading, edge_dst, edge_allowed, node_valid, heading_values, turns,
      heading_count, root_index, root_start_cost,
      root_connector_m, route, physical, parent, parent_length, traversal, n,
      slots, heap_capacity, heap_distance, heap_node, status);
}

void plan_a_pose_regions_launch(const int64_t* edge_dst,
    const bool* edge_allowed, const bool* node_valid, int64_t* region, int64_t* coarse_edges,
    bool* coarse_mask, int64_t* status, int64_t n, int64_t degree,
    int64_t region_count, hipStream_t stream) {
  hipLaunchKernelGGL(plan_a_pose_regions_kernel, dim3(1), dim3(1), 0, stream,
      edge_dst, edge_allowed, node_valid, region, coarse_edges, coarse_mask, status, n,
      degree, region_count);
}
