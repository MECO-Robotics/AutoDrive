#include <hip/hip_runtime.h>

namespace {
constexpr int kFeatures = 137;

__global__ void pack_kernel(const float* base, const float* route,
    const float* local0, const float* local1, const float* possession,
    const float* match, const float* candidate, const float* action_mask,
    const float* speed, const float* omega, float* output, int64_t rows,
    float inv_length, float inv_width, float inv_max_dim, float inv_pi,
    float inv_16_54, float inv_8_07, bool normalize) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t count = rows * kFeatures;
  if (index >= count) return;
  const int64_t row = index / kFeatures;
  const int col = static_cast<int>(index - row * kFeatures);
  float value;
  if (col < 35) value = base[row * 35 + col];
  else if (col < 38) value = route[row * 3 + (col - 35)];
  else if (col < 58) value = local0[row * 20 + (col - 38)];
  else if (col < 78) value = local1[row * 20 + (col - 58)];
  else if (col < 80) value = possession[row * 2 + (col - 78)];
  else if (col < 89) value = match[row * 9 + (col - 80)];
  else if (col < 129) value = candidate[row * 40 + (col - 89)];
  else value = action_mask[row * 8 + (col - 129)];

  if (normalize) {
    switch (col) {
      case 0: case 6: case 12: value *= inv_length; break;
      case 1: case 7: case 13: value *= inv_width; break;
      case 2: case 8: value *= inv_pi; break;
      case 3: case 4: value /= fmaxf(speed[row * 2], .1f); break;
      case 5: value /= fmaxf(omega[row * 2], .1f); break;
      case 9: case 10: value /= fmaxf(speed[row * 2 + 1], .1f); break;
      case 11: value /= fmaxf(omega[row * 2 + 1], .1f); break;
      case 14: value *= inv_max_dim; break;
      case 15: value *= inv_16_54; break;
      case 16: value *= inv_8_07; break;
      case 23: case 26: case 29: case 32: value *= inv_length; break;
      case 24: case 27: case 30: case 33: value *= inv_width; break;
      default: break;
    }
  }
  output[index] = value;
}

__global__ void pack_fused_kernel(const float* base, const float* route,
    const float* local1, const float* possession, const float* match,
    const float* action_mask, const float* speed, const float* omega,
    const float* track_pos, const float* track_vel, const int64_t* indices,
    const bool* valid, const float* nearest, const float* own_pose,
    const float* other_pose, const float* own_count, const float* robot_length,
    const float* robot_width, const float* hub, float* output, int64_t rows,
    int64_t piece_count, int64_t track_pos_row_stride,
    int64_t track_vel_row_stride, float field_length, float field_width,
    int fuel_capacity,
    float inv_length, float inv_width, float inv_max_dim, float inv_pi,
    float inv_16_54, float inv_8_07, bool normalize) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t count = rows * kFeatures;
  if (index >= count) return;
  const int64_t row = index / kFeatures;
  const int col = static_cast<int>(index - row * kFeatures);
  float value;
  if (col < 35) {
    value = base[row * 35 + col];
  } else if (col < 38) {
    value = route[row * 3 + (col - 35)];
  } else if (col < 58) {
    const int slot = (col - 38) / 5;
    const int component = (col - 38) % 5;
    if (!valid[row * 4 + slot]) {
      value = 0.f;
    } else {
      const int64_t piece = indices[row * 4 + slot];
      const int64_t point_offset = row * track_pos_row_stride + piece * 2;
      const int64_t velocity_offset = row * track_vel_row_stride + piece * 2;
      if (component == 0)
        value = (track_pos[point_offset] - own_pose[row * 3]) * inv_length;
      else if (component == 1)
        value = (track_pos[point_offset + 1] - own_pose[row * 3 + 1]) * inv_width;
      else if (component == 2)
        value = track_vel[velocity_offset] / fmaxf(speed[row * 2], .1f);
      else if (component == 3)
        value = track_vel[velocity_offset + 1] / fmaxf(speed[row * 2], .1f);
      else
        value = 1.f;
    }
  } else if (col < 78) {
    value = local1[row * 20 + (col - 58)];
  } else if (col < 80) {
    value = possession[row * 2 + (col - 78)];
  } else if (col < 89) {
    value = match[row * 9 + (col - 80)];
  } else if (col < 129) {
    const int slot = (col - 89) / 10;
    const int component = (col - 89) % 10;
    if (!valid[row * 4 + slot]) {
      value = 0.f;
    } else {
      const int64_t piece = indices[row * 4 + slot];
      const int64_t point_offset = row * track_pos_row_stride + piece * 2;
      const float px = track_pos[point_offset];
      const float py = track_pos[point_offset + 1];
      const float own_speed = fmaxf(speed[row * 2], .1f);
      const float opponent_speed = fmaxf(speed[row * 2 + 1], .1f);
      const float own_eta = nearest[row * 4 + slot] / own_speed;
      const float odx = px - other_pose[row * 2];
      const float ody = py - other_pose[row * 2 + 1];
      const float opponent_distance = sqrtf(odx * odx + ody * ody);
      const float opponent_eta = opponent_distance / opponent_speed;
      const float radius = .5f * sqrtf(robot_length[row] * robot_length[row] +
                                       robot_width[row] * robot_width[row]);
      const float hdx = px - hub[0];
      const float hdy = py - hub[1];
      const float score_distance = sqrtf(hdx * hdx + hdy * hdy) - (.595f + radius);
      const float to_score = fmaxf(score_distance, 0.f) / own_speed;
      const float risk_input = (own_eta - opponent_eta) * 2.f;
      const float risk = 1.f / (1.f + expf(-risk_input));
      const float zone = fminf(fmaxf(floorf(px * inv_length * 6.f), 0.f), 5.f) * (1.f / 6.f);
      const float capacity_inv = 1.f / static_cast<float>(fuel_capacity > 0 ? fuel_capacity : 1);
      const float own_pos = own_count[row] * capacity_inv;
      const float capacity_left = fmaxf(static_cast<float>(fuel_capacity) - own_count[row], 0.f) *
                                  capacity_inv;
      switch (component) {
        case 0: value = (px - own_pose[row * 3]) * inv_length; break;
        case 1: value = (py - own_pose[row * 3 + 1]) * inv_width; break;
        case 2: value = own_eta * .05f; break;
        case 3: value = opponent_eta * .05f; break;
        case 4: value = to_score * .05f; break;
        case 5: value = risk; break;
        case 6: value = zone; break;
        case 7: value = own_pos; break;
        case 8: value = capacity_left; break;
        default: value = 1.f; break;
      }
    }
  } else {
    value = action_mask[row * 8 + (col - 129)];
  }

  if (normalize) {
    switch (col) {
      case 0: case 6: case 12: value *= inv_length; break;
      case 1: case 7: case 13: value *= inv_width; break;
      case 2: case 8: value *= inv_pi; break;
      case 3: case 4: value /= fmaxf(speed[row * 2], .1f); break;
      case 5: value /= fmaxf(omega[row * 2], .1f); break;
      case 9: case 10: value /= fmaxf(speed[row * 2 + 1], .1f); break;
      case 11: value /= fmaxf(omega[row * 2 + 1], .1f); break;
      case 14: value *= inv_max_dim; break;
      case 15: value *= inv_16_54; break;
      case 16: value *= inv_8_07; break;
      case 23: case 26: case 29: case 32: value *= inv_length; break;
      case 24: case 27: case 30: case 33: value *= inv_width; break;
      default: break;
    }
  }
  output[index] = value;
}
}  // namespace

void pack_launch(const float* base, const float* route, const float* local0,
    const float* local1, const float* possession, const float* match,
    const float* candidate, const float* action_mask, const float* speed,
    const float* omega, float* output, int64_t rows, float inv_length,
    float inv_width, float inv_max_dim, float inv_pi, float inv_16_54,
    float inv_8_07, bool normalize, hipStream_t stream) {
  const int threads = 256;
  const int blocks = static_cast<int>((rows * kFeatures + threads - 1) / threads);
  if (blocks == 0) return;
  hipLaunchKernelGGL(pack_kernel, dim3(blocks), dim3(threads), 0, stream,
      base, route, local0, local1, possession, match, candidate, action_mask,
      speed, omega, output, rows, inv_length, inv_width, inv_max_dim, inv_pi,
      inv_16_54, inv_8_07, normalize);
}

void pack_fused_launch(const float* base, const float* route, const float* local1,
    const float* possession, const float* match, const float* action_mask,
    const float* speed, const float* omega, const float* track_pos,
    const float* track_vel, const int64_t* indices, const bool* valid,
    const float* nearest, const float* own_pose, const float* other_pose,
    const float* own_count, const float* robot_length, const float* robot_width,
    const float* hub, float* output, int64_t rows, int64_t piece_count,
    int64_t track_pos_row_stride, int64_t track_vel_row_stride,
    float field_length, float field_width, int fuel_capacity, float inv_length,
    float inv_width, float inv_max_dim, float inv_pi, float inv_16_54,
    float inv_8_07, bool normalize, hipStream_t stream) {
  constexpr int threads = 256;
  const int blocks = static_cast<int>((rows * kFeatures + threads - 1) / threads);
  if (blocks == 0) return;
  hipLaunchKernelGGL(pack_fused_kernel, dim3(blocks), dim3(threads), 0, stream,
      base, route, local1, possession, match, action_mask, speed, omega,
      track_pos, track_vel, indices, valid, nearest, own_pose, other_pose,
      own_count, robot_length, robot_width, hub, output, rows, piece_count,
      track_pos_row_stride, track_vel_row_stride, field_length, field_width,
      fuel_capacity, inv_length, inv_width,
      inv_max_dim, inv_pi, inv_16_54, inv_8_07, normalize);
}
