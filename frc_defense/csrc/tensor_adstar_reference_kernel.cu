#include <hip/hip_runtime.h>

#pragma clang fp contract(off)

namespace {
__global__ void path_reference_kernel(const float* path, const int64_t* lengths,
    const float* goal, const float* profile, const float* position,
    int64_t position_stride, const float* velocity, int64_t velocity_stride,
    const float* speed_limit, int64_t speed_stride, float* command, float* tangent_out) {
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
    const int index = best_index[0];
    const float here_x = path[path_base + index * 2];
    const float here_y = path[path_base + index * 2 + 1];
    float target_x = path[path_base + (lengths[world] - 1) * 2];
    float target_y = path[path_base + (lengths[world] - 1) * 2 + 1];
    const float lookahead = 0.7f + 0.12f * fminf(speed_limit[static_cast<int64_t>(world) * speed_stride], 4.5f);
    float traversed = 0.f;
    for (int i = index; i < lengths[world] - 1; ++i) {
      const float ax = path[path_base + i * 2];
      const float ay = path[path_base + i * 2 + 1];
      const float bx = path[path_base + (i + 1) * 2];
      const float by = path[path_base + (i + 1) * 2 + 1];
      const float dx = bx - ax, dy = by - ay;
      const float segment = sqrtf(__fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy)));
      if (traversed + segment >= lookahead) {
        const float fraction = fminf(fmaxf((lookahead - traversed) /
                                           fmaxf(segment, 1.0e-6f), 0.f), 1.f);
        target_x = ax + dx * fraction;
        target_y = ay + dy * fraction;
        break;
      }
      traversed += segment;
    }
    float tx = target_x - px;
    float ty = target_y - py;
    const float tangent_norm = sqrtf(__fadd_rn(__fmul_rn(tx, tx), __fmul_rn(ty, ty)));
    const float norm_denominator = fmaxf(tangent_norm, 1.0e-6f);
    tx = __fdiv_rn(tx, norm_denominator);
    ty = __fdiv_rn(ty, norm_denominator);
    const float nx = -ty;
    const float ny = tx;
    const float cross = __fadd_rn(__fmul_rn(px - here_x, nx),
                                  __fmul_rn(py - here_y, ny));
    const float cross_velocity = __fadd_rn(
        __fmul_rn(velocity[static_cast<int64_t>(world) * velocity_stride], nx),
        __fmul_rn(velocity[static_cast<int64_t>(world) * velocity_stride + 1], ny));
    const float lateral = fminf(fmaxf(__fadd_rn(__fmul_rn(-3.f, cross),
                                               __fmul_rn(-4.f, cross_velocity)), -.75f), .75f);
    const float planned = fmaxf(fminf(profile[static_cast<int64_t>(world) * 72 + index],
                                      speed_limit[static_cast<int64_t>(world) * speed_stride]), .2f);
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
    const float* speed_limit, int64_t speed_stride, float* command,
    float* tangent, int worlds, hipStream_t stream) {
  hipLaunchKernelGGL(path_reference_kernel, dim3(worlds), dim3(128), 0, stream,
      path, lengths, goal, profile, position, position_stride, velocity,
      velocity_stride, speed_limit, speed_stride, command, tangent);
}
