#include <hip/hip_runtime.h>
#include <math.h>

__global__ void candidate_distance_kernel(const float* points, const float* robot,
    const bool* free, float* distances, int64_t n, int64_t p) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = n * p;
  if (i >= total) return;
  const int64_t row = i / p;
  const int64_t piece = i - row * p;
  if (!free[i]) {
    distances[i] = INFINITY;
    return;
  }
  const float dx = points[2 * i] - robot[2 * row];
  const float dy = points[2 * i + 1] - robot[2 * row + 1];
  distances[i] = sqrtf(dx * dx + dy * dy);
}

void candidate_distance_launch(const float* points, const float* robot,
    const bool* free, float* distances, int64_t n, int64_t p, hipStream_t stream) {
  constexpr int threads = 256;
  const int64_t total = n * p;
  candidate_distance_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(
      points, robot, free, distances, n, p);
}
