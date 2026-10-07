#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

__global__ void candidate_rank_kernel(
    const float* points, const float* robot, const bool* free,
    int64_t* indices, bool* valid, float* nearest,
    int64_t n, int64_t p) {
  const int64_t row = blockIdx.x;
  const int lane = threadIdx.x;
  __shared__ float shared_distance[256];
  __shared__ int64_t shared_index[256];
  __shared__ int64_t selected_index;
  float local_distance[4] = {INFINITY, INFINITY, INFINITY, INFINITY};
  int64_t local_index[4] = {p, p, p, p};
  int cursor = 0;
  for (int64_t piece = lane; piece < p; piece += blockDim.x) {
    float distance = INFINITY;
    if (free[row * p + piece]) {
      const float dx = points[2 * (row * p + piece)] - robot[2 * row];
      const float dy = points[2 * (row * p + piece) + 1] - robot[2 * row + 1];
      distance = sqrtf(dx * dx + dy * dy);
    }
    for (int slot = 0; slot < 4; ++slot) {
      if (distance < local_distance[slot] ||
          (distance == local_distance[slot] && piece < local_index[slot])) {
        for (int move = 3; move > slot; --move) {
          local_distance[move] = local_distance[move - 1];
          local_index[move] = local_index[move - 1];
        }
        local_distance[slot] = distance;
        local_index[slot] = piece;
        break;
      }
    }
  }

  for (int rank = 0; rank < 4; ++rank) {
    shared_distance[lane] = local_distance[cursor];
    shared_index[lane] = local_index[cursor];
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (lane < stride) {
        const float other_distance = shared_distance[lane + stride];
        const int64_t other_index = shared_index[lane + stride];
        if (other_distance < shared_distance[lane] ||
            (other_distance == shared_distance[lane] &&
             other_index < shared_index[lane])) {
          shared_distance[lane] = other_distance;
          shared_index[lane] = other_index;
        }
      }
      __syncthreads();
    }
    if (lane == 0) {
      selected_index = shared_index[0];
      indices[row * 4 + rank] = selected_index;
      nearest[row * 4 + rank] = shared_distance[0];
      valid[row * 4 + rank] = isfinite(shared_distance[0]);
    }
    __syncthreads();
    if (local_index[cursor] == selected_index && cursor < 3) ++cursor;
  }
}

void candidate_rank_launch(const float* points, const float* robot,
    const bool* free, int64_t* indices, bool* valid, float* nearest,
    int64_t n, int64_t p, hipStream_t stream) {
  candidate_rank_kernel<<<static_cast<unsigned>(n), 256, 0, stream>>>(
      points, robot, free, indices, valid, nearest, n, p);
}
