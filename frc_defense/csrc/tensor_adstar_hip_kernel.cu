#include <hip/hip_runtime.h>

namespace {
constexpr float kInfinity = 1.0e6f;
constexpr float kSqrt2 = 1.41421356237f;

__device__ __forceinline__ bool cell_blocked(const bool* blocked, int b,
                                              int x, int y, int nx, int ny) {
  return x < 0 || x >= nx || y < 0 || y >= ny ||
         blocked[(static_cast<int64_t>(b) * nx + x) * ny + y];
}

__global__ void bellman_sweeps_kernel(const float* value, const bool* blocked,
                                      const float* bump, float* output,
                                      int nx, int ny, int cells, int sweeps,
                                      bool stop_when_converged) {
  const int b = blockIdx.x;
  const int lane = threadIdx.x;
  extern __shared__ float shared_values[];
  float* current = shared_values;
  float* next = shared_values + cells;
  __shared__ int changed_threads;
  const int64_t base = static_cast<int64_t>(b) * cells;
  for (int cell = lane; cell < cells; cell += blockDim.x)
    current[cell] = value[base + cell];
  __syncthreads();

  for (int sweep = 0; sweep < sweeps; ++sweep) {
    int thread_changed = 0;
    for (int cell = lane; cell < cells; cell += blockDim.x) {
      const int x = cell / ny;
      const int y = cell - x * ny;
      const int64_t linear = base + cell;
      if (blocked[linear]) {
        next[cell] = kInfinity;
        thread_changed |= current[cell] != kInfinity;
        continue;
      }
      float best = current[cell];
      const float edge = bump[linear];
      #pragma unroll
      for (int dx = -1; dx <= 1; ++dx) {
        #pragma unroll
        for (int dy = -1; dy <= 1; ++dy) {
          const int xx = x + dx;
          const int yy = y + dy;
          float neighbor = kInfinity;
          if (xx >= 0 && xx < nx && yy >= 0 && yy < ny)
            neighbor = current[xx * ny + yy];
          if (dx != 0 && dy != 0 &&
              (cell_blocked(blocked, b, x + dx, y, nx, ny) ||
               cell_blocked(blocked, b, x, y + dy, nx, ny)))
            continue;
          const float distance = (dx != 0 && dy != 0) ? kSqrt2 : 1.0f;
          const float candidate = __fadd_rn(neighbor, __fmul_rn(distance, edge));
          best = fminf(best, candidate);
        }
      }
      next[cell] = best;
      thread_changed |= best < current[cell];
    }
    if (stop_when_converged) {
      const int changed_count = __syncthreads_count(thread_changed != 0);
      if (changed_count == 0) break;
    } else {
      __syncthreads();
    }
    float* swap = current;
    current = next;
    next = swap;
  }
  for (int cell = lane; cell < cells; cell += blockDim.x)
    output[base + cell] = current[cell];
}
}  // namespace

void adstar_launch(const float* value, const bool* blocked, const float* bump,
                   float* output, int nx, int ny, int64_t total, int sweeps,
                   bool stop_when_converged, hipStream_t stream) {
  const int cells = nx * ny;
  const int batch = static_cast<int>(total / cells);
  constexpr int threads = 256;
  const size_t shared_bytes = static_cast<size_t>(2) * cells * sizeof(float);
  hipLaunchKernelGGL(bellman_sweeps_kernel, dim3(batch), dim3(threads), shared_bytes,
                     stream, value, blocked, bump, output, nx, ny, cells, sweeps,
                     stop_when_converged);
}
