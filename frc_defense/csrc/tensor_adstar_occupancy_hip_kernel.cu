// Isolated HIP kernel matching TensorADStar._blocked's occupancy tests.
#include <hip/hip_runtime.h>

#pragma clang fp contract(off)

namespace {
__global__ void blocked_grid_kernel(const float* heading, const float* length,
    const float* width, const float* x, const float* y, const float* boxes,
    const float* bumps, const float* circles, const float* dynamic,
    int n, int nx, int ny, int box_count, int bump_count, int circle_count,
    float field_length, float field_width, float grid_clearance,
    bool avoid_bumps, bool has_dynamic, bool* out) {
  const int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(n) * nx * ny;
  if (flat >= total) return;
  const int cell = static_cast<int>(flat % (static_cast<int64_t>(nx) * ny));
  const int world = static_cast<int>(flat / (static_cast<int64_t>(nx) * ny));
  const int ix = cell / ny;
  const int iy = cell % ny;
  const float gx = x[ix];
  const float gy = y[iy];
  const float h = heading[world];
  const float c = cosf(h), s = sinf(h);
  const float ac = fabsf(c), as = fabsf(s);
  const float l = length[world], w = width[world];
  const float ex = (l * ac + w * as) * .5f + grid_clearance;
  const float ey = (l * as + w * ac) * .5f + grid_clearance;
  bool blocked = gx < ex || gx > field_length - ex ||
                 gy < ey || gy > field_width - ey;

  for (int i = 0; i < box_count && !blocked; ++i) {
    const float bx = boxes[i * 4], by = boxes[i * 4 + 1];
    const float bhx = boxes[i * 4 + 2], bhy = boxes[i * 4 + 3];
    const float dx = gx - bx, dy = gy - by;
    const float hx = bhx + ex, hy = bhy + ey;
    const float forward = dx * c + dy * s;
    const float lateral = -dx * s + dy * c;
    const float sat_f = l * .5f + bhx * ac + bhy * as;
    const float sat_l = w * .5f + bhx * as + bhy * ac;
    blocked = fabsf(dx) <= hx && fabsf(dy) <= hy &&
              fabsf(forward) <= sat_f && fabsf(lateral) <= sat_l;
  }

  // Bumps are traversable terrain. The planner adds their traversal cost in
  // the route solver, so occupancy must not turn them into hard obstacles.
  for (int i = 0; i < circle_count && !blocked; ++i) {
    const float dx = gx - circles[i * 3];
    const float dy = gy - circles[i * 3 + 1];
    const float r = circles[i * 3 + 2] + fmaxf(ex, ey);
    blocked = dx * dx + dy * dy <= r * r;
  }

  if (has_dynamic && !blocked) {
    const float dx = gx - dynamic[world * 4];
    const float dy = gy - dynamic[world * 4 + 1];
    const float dhx = dynamic[world * 4 + 2];
    const float dhy = dynamic[world * 4 + 3];
    const float hx = dhx + ex, hy = dhy + ey;
    const float forward = dx * c + dy * s;
    const float lateral = -dx * s + dy * c;
    const float sat_f = l * .5f + dhx * ac + dhy * as;
    const float sat_l = w * .5f + dhx * as + dhy * ac;
    blocked = fabsf(dx) <= hx && fabsf(dy) <= hy &&
              fabsf(forward) <= sat_f && fabsf(lateral) <= sat_l;
  }
  out[flat] = blocked;
}
}  // namespace

void blocked_grid_launch(const float* heading, const float* length,
    const float* width, const float* x, const float* y, const float* boxes,
    const float* bumps, const float* circles, const float* dynamic,
    int n, int nx, int ny, int box_count, int bump_count, int circle_count,
    float field_length, float field_width, float grid_clearance,
    bool avoid_bumps, bool has_dynamic, bool* out, hipStream_t stream) {
  const int64_t total = static_cast<int64_t>(n) * nx * ny;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(blocked_grid_kernel, dim3(blocks), dim3(threads), 0, stream,
      heading, length, width, x, y, boxes, bumps, circles, dynamic, n, nx, ny,
      box_count, bump_count, circle_count, field_length, field_width,
      grid_clearance, avoid_bumps, has_dynamic, out);
}
