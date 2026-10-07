#include <hip/hip_runtime.h>

#pragma clang fp contract(off)

namespace {
__global__ void path_reference_kernel(const float* path, const int64_t* lengths,
    const float* goal, const float* profile, const float* position,
    int64_t position_stride, const float* velocity, int64_t velocity_stride,
    const float* speed_limit, int64_t speed_stride, float* progress_state,
    float* command, float* tangent_out) {
  const int world = blockIdx.x;
  const int lane = threadIdx.x;
  __shared__ float best_distance[128];
  __shared__ int best_index[128];
  const int64_t path_base = static_cast<int64_t>(world) * 144;
  const float px = position[static_cast<int64_t>(world) * position_stride];
  const float py = position[static_cast<int64_t>(world) * position_stride + 1];
  float local_distance = INFINITY;
  int local_index = 72;
  for (int i = lane; i < 72; i += blockDim.x) {
    const float dx = path[path_base + i * 2] - px;
    const float dy = path[path_base + i * 2 + 1] - py;
    const float d2 = __fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy));
    if (d2 < local_distance || (d2 == local_distance && i < local_index)) {
      local_distance = d2;
      local_index = i;
    }
  }
  best_distance[lane] = local_distance;
  best_index[lane] = local_index;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (lane < offset) {
      const float other_distance = best_distance[lane + offset];
      const int other_index = best_index[lane + offset];
      if (other_distance < best_distance[lane] ||
          (other_distance == best_distance[lane] && other_index < best_index[lane])) {
        best_distance[lane] = other_distance;
        best_index[lane] = other_index;
      }
    }
    __syncthreads();
  }
  if (lane == 0) {
    if (lengths[world] <= 1) {
      command[world * 2] = 0.f;
      command[world * 2 + 1] = 0.f;
      tangent_out[world * 2] = 0.f;
      tangent_out[world * 2 + 1] = 0.f;
      return;
    }
    const float hint = progress_state[world];
    float cumulative = 0.f, best_distance = INFINITY, progress = 0.f;
    float projection_x = path[path_base], projection_y = path[path_base + 1];
    int index = 0;
    for (int i = 0; i < lengths[world] - 1; ++i) {
      const float ax = path[path_base + i * 2];
      const float ay = path[path_base + i * 2 + 1];
      const float dx = path[path_base + (i + 1) * 2] - ax;
      const float dy = path[path_base + (i + 1) * 2 + 1] - ay;
      const float segment2 = __fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy));
      const float segment = sqrtf(segment2);
      if (cumulative + segment >= fmaxf(0.f, hint - .08f) && segment2 > 1.0e-12f) {
        const float fraction = fminf(fmaxf(__fdiv_rn(
            __fadd_rn(__fmul_rn(px - ax, dx), __fmul_rn(py - ay, dy)), segment2), 0.f), 1.f);
        const float qx = __fadd_rn(ax, __fmul_rn(dx, fraction));
        const float qy = __fadd_rn(ay, __fmul_rn(dy, fraction));
        const float ex = qx - px, ey = qy - py;
        const float d2 = __fadd_rn(__fmul_rn(ex, ex), __fmul_rn(ey, ey));
        if (d2 < best_distance) {
          best_distance = d2; index = i;
          progress = cumulative + fraction * segment;
          projection_x = qx; projection_y = qy;
        }
      }
      cumulative += segment;
    }
    progress_state[world] = fmaxf(hint, fminf(cumulative, progress));
    progress = progress_state[world];
    float target_x = path[path_base + (lengths[world] - 1) * 2];
    float target_y = path[path_base + (lengths[world] - 1) * 2 + 1];
    const float lookahead = 0.7f + 0.12f * fminf(speed_limit[static_cast<int64_t>(world) * speed_stride], 4.5f);
    const float target_station = fminf(cumulative, progress + fmaxf(.12f, lookahead));
    float traversed = 0.f, target_fraction = 0.f;
    int target_index = lengths[world] - 2;
    for (int i = 0; i < lengths[world] - 1; ++i) {
      const float ax = path[path_base + i * 2];
      const float ay = path[path_base + i * 2 + 1];
      const float bx = path[path_base + (i + 1) * 2];
      const float by = path[path_base + (i + 1) * 2 + 1];
      const float dx = bx - ax, dy = by - ay;
      const float segment = sqrtf(__fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy)));
      if (traversed + segment >= target_station) {
        const float fraction = fminf(fmaxf((target_station - traversed) /
                                           fmaxf(segment, 1.0e-6f), 0.f), 1.f);
        target_x = ax + dx * fraction;
        target_y = ay + dy * fraction;
        target_index = i;
        target_fraction = fraction;
        break;
      }
      traversed += segment;
    }
    float tx = path[path_base + (target_index + 1) * 2] - path[path_base + target_index * 2];
    float ty = path[path_base + (target_index + 1) * 2 + 1] - path[path_base + target_index * 2 + 1];
    const float tangent_norm = sqrtf(__fadd_rn(__fmul_rn(tx, tx), __fmul_rn(ty, ty)));
    tx = __fdiv_rn(tx, fmaxf(tangent_norm, 1.0e-6f));
    ty = __fdiv_rn(ty, fmaxf(tangent_norm, 1.0e-6f));
    const float nx = -ty;
    const float ny = tx;
    const float cross = __fadd_rn(__fmul_rn(px - projection_x, nx),
                                  __fmul_rn(py - projection_y, ny));
    const float cross_velocity = __fadd_rn(
        __fmul_rn(velocity[static_cast<int64_t>(world) * velocity_stride], nx),
        __fmul_rn(velocity[static_cast<int64_t>(world) * velocity_stride + 1], ny));
    const float lateral = fminf(fmaxf(__fadd_rn(__fmul_rn(-3.f, cross),
                                               __fmul_rn(-4.f, cross_velocity)), -.75f), .75f);
    const float v0 = profile[static_cast<int64_t>(world) * 72 + target_index];
    const float v1 = profile[static_cast<int64_t>(world) * 72 + target_index + 1];
    const float planned = fmaxf(fminf(v0 + (v1 - v0) * target_fraction,
                                      speed_limit[static_cast<int64_t>(world) * speed_stride]), 0.f);
    const float vx = velocity[static_cast<int64_t>(world) * velocity_stride];
    const float vy = velocity[static_cast<int64_t>(world) * velocity_stride + 1];
    const float current = sqrtf(__fadd_rn(__fmul_rn(vx, vx), __fmul_rn(vy, vy)));
    const float excess = fmaxf(current - planned, 0.f);
    const float target = fmaxf(__fadd_rn(planned, __fmul_rn(-.55f, excess)), 0.f);
    command[world * 2] = __fadd_rn(__fmul_rn(tx, target), __fmul_rn(nx, lateral));
    command[world * 2 + 1] = __fadd_rn(__fmul_rn(ty, target), __fmul_rn(ny, lateral));
    tangent_out[world * 2] = tx;
    tangent_out[world * 2 + 1] = ty;
  }
}
}  // namespace

void path_reference_launch(const float* path, const int64_t* lengths,
    const float* goal, const float* profile, const float* position,
    int64_t position_stride, const float* velocity, int64_t velocity_stride,
    const float* speed_limit, int64_t speed_stride, float* progress_state, float* command,
    float* tangent, int worlds, hipStream_t stream) {
  hipLaunchKernelGGL(path_reference_kernel, dim3(worlds), dim3(128), 0, stream,
      path, lengths, goal, profile, position, position_stride, velocity,
      velocity_stride, speed_limit, speed_stride, progress_state, command, tangent);
}
