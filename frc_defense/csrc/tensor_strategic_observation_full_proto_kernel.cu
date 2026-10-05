#include <hip/hip_runtime.h>

namespace {
constexpr int kFeatures = 137;

__global__ void assemble_kernel(const float* own_pose, const float* own_vel,
    const float* other_pose, const float* other_vel, const float* opponent,
    const float* obstacles, const float* length, const float* width,
    const float* accel, const float* speed, const float* omega,
    const float* path, const int64_t* path_lengths, const float* track_pos,
    const float* track_vel, const int64_t* indices, const bool* valid,
    const float* nearest, const float* own_count, const bool* piece_active,
    const int64_t* piece_owner, const float* elapsed, const bool* hub_active,
    const int64_t* scores, const float* hubs, const float* hub,
    const bool* capture_mask, float* output, int64_t rows,
    int64_t pieces, int64_t path_capacity, int64_t track_pos_stride,
    int64_t track_vel_stride, float field_length, float field_width,
    float inv_l, float inv_w, float inv_diag, float inv_max, float inv_pi,
    float inv_16_54, float inv_8_07,
    int fuel_capacity, int fuel_count, bool offense, bool normalize) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= rows * kFeatures) return;
  const int64_t row = index / kFeatures;
  const int col = static_cast<int>(index - row * kFeatures);
  if (!capture_mask[row]) {
    output[index] = 0.f;
    return;
  }
  float value = 0.f;
  if (col < 3) value = own_pose[row * 3 + col];
  else if (col < 6) value = own_vel[row * 3 + col - 3];
  else if (col < 9) value = other_pose[row * 3 + col - 6];
  else if (col < 12) value = other_vel[row * 3 + col - 9];
  else if (col < 14) value = opponent[row * 2 + col - 12];
  else if (col == 14) value = .595f;
  else if (col == 15) value = field_length;
  else if (col == 16) value = field_width;
  else if (col < 19) value = length[row * 2 + col - 17];
  else if (col < 21) value = width[row * 2 + col - 19];
  else if (col < 23) value = accel[row * 2 + col - 21] * .1f;
  else if (col < 35) value = obstacles[row * 12 + col - 23];
  else if (col < 38) {
    const int64_t count = path_lengths[row];
    if (count > 1) {
      if (col == 35) value = (path[row * path_capacity * 2 + 2] - own_pose[row * 3]) * inv_l;
      else if (col == 36) value = (path[row * path_capacity * 2 + 3] - own_pose[row * 3 + 1]) * inv_w;
      else {
        float cost = 0.f;
        const int64_t end = count - 1 < path_capacity - 1 ? count - 1 : path_capacity - 1;
        for (int64_t j = 0; j < end; ++j) {
          const int64_t a = row * path_capacity * 2 + j * 2;
          const int64_t b = a + 2;
          const float dx = path[b] - path[a];
          const float dy = path[b + 1] - path[a + 1];
          cost += sqrtf(dx * dx + dy * dy);
        }
        value = cost * inv_diag;
      }
    }
  } else if (col < 58) {
    const int slot = (col - 38) / 5;
    const int component = (col - 38) % 5;
    if (valid[row * 4 + slot]) {
      const int64_t piece = indices[row * 4 + slot];
      const int64_t po = row * track_pos_stride + piece * 2;
      const int64_t vo = row * track_vel_stride + piece * 2;
      if (component == 0) value = (track_pos[po] - own_pose[row * 3]) * inv_l;
      else if (component == 1) value = (track_pos[po + 1] - own_pose[row * 3 + 1]) * inv_w;
      else if (component == 2) value = track_vel[vo] / fmaxf(speed[row * 2], .1f);
      else if (component == 3) value = track_vel[vo + 1] / fmaxf(speed[row * 2], .1f);
      else value = 1.f;
    }
  } else if (col < 78) {
    value = 0.f;
  } else if (col == 78) {
    value = own_count[row] / static_cast<float>(fuel_capacity > 0 ? fuel_capacity : 1);
  } else if (col == 79) {
    value = 0.f;
  } else if (col == 80) {
    value = elapsed[row] * .00625f;
  } else if (col < 83) {
    value = hub_active[row * 2 + col - 81] ? 1.f : 0.f;
  } else if (col < 85) {
    value = static_cast<float>(scores[row * 2 + col - 83]) /
        static_cast<float>(fuel_count > 0 ? fuel_count : 1);
  } else if (col < 89) {
    const int k = col - 85;
    const float center = hubs[k];
    const float own = own_pose[row * 3 + (k % 2)];
    value = __fdiv_rn(center - own, k % 2 == 0 ? field_length : field_width);
  } else if (col < 129) {
    const int slot = (col - 89) / 10;
    const int component = (col - 89) % 10;
    if (valid[row * 4 + slot]) {
      const int64_t piece = indices[row * 4 + slot];
      const int64_t po = row * track_pos_stride + piece * 2;
      const float px = track_pos[po], py = track_pos[po + 1];
      const float own_speed = fmaxf(speed[row * 2], .1f);
      const float opponent_speed = fmaxf(speed[row * 2 + 1], .1f);
      const float own_eta = nearest[row * 4 + slot] / own_speed;
      const float odx = px - other_pose[row * 3];
      const float ody = py - other_pose[row * 3 + 1];
      const float opponent_eta = sqrtf(odx * odx + ody * ody) / opponent_speed;
      const float radius = .5f * sqrtf(length[row * 2] * length[row * 2] +
                                       width[row * 2] * width[row * 2]);
      const float hdx = px - hub[0], hdy = py - hub[1];
      const float to_score = fmaxf(sqrtf(hdx * hdx + hdy * hdy) - (.595f + radius), 0.f) / own_speed;
      const float risk = 1.f / (1.f + expf(-(own_eta - opponent_eta) * 2.f));
      const float zone = fminf(fmaxf(floorf(px * inv_l * 6.f), 0.f), 5.f) * (1.f / 6.f);
      const float cap_inv = 1.f / static_cast<float>(fuel_capacity > 0 ? fuel_capacity : 1);
      const float own_pos = own_count[row] * cap_inv;
      const float capacity_left = fmaxf(static_cast<float>(fuel_capacity) - own_count[row], 0.f) * cap_inv;
      switch (component) {
        case 0: value = (px - own_pose[row * 3]) * inv_l; break;
        case 1: value = (py - own_pose[row * 3 + 1]) * inv_w; break;
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
    const int action = col - 129;
    if (offense) {
      if (action < 4) value = valid[row * 4 + action] ? 1.f : 0.f;
      else if (action == 4) value = own_count[row] > 0.f ? 1.f : 0.f;
      else if (action == 6 || action == 7) value = 1.f;
    } else {
      if (action == 0 || action == 5 || action == 6) value = 1.f;
      else if (action >= 1 && action <= 4) value = valid[row * 4 + action - 1] ? 1.f : 0.f;
    }
  }

  if (normalize) {
    switch (col) {
      case 0: case 6: case 12: value *= inv_l; break;
      case 1: case 7: case 13: value *= inv_w; break;
      case 2: case 8: value *= inv_pi; break;
      case 3: case 4: value /= fmaxf(speed[row * 2], .1f); break;
      case 5: value /= fmaxf(omega[row * 2], .1f); break;
      case 9: case 10: value /= fmaxf(speed[row * 2 + 1], .1f); break;
      case 11: value /= fmaxf(omega[row * 2 + 1], .1f); break;
      case 14: value *= inv_max; break;
      case 15: value *= inv_16_54; break;
      case 16: value *= inv_8_07; break;
      case 23: case 26: case 29: case 32: value *= inv_l; break;
      case 24: case 27: case 30: case 33: value *= inv_w; break;
      default: break;
    }
  }
  output[index] = value;
}
}  // namespace

void assemble_launch(const float* own_pose, const float* own_vel,
    const float* other_pose, const float* other_vel, const float* opponent,
    const float* obstacles, const float* length, const float* width,
    const float* accel, const float* speed, const float* omega,
    const float* path, const int64_t* path_lengths, const float* track_pos,
    const float* track_vel, const int64_t* indices, const bool* valid,
    const float* nearest, const float* own_count, const bool* piece_active,
    const int64_t* piece_owner, const float* elapsed, const bool* hub_active,
    const int64_t* scores, const float* hubs, const float* hub,
    const bool* capture_mask, float* output, int64_t rows,
    int64_t pieces, int64_t path_capacity, int64_t track_pos_stride,
    int64_t track_vel_stride, float field_length, float field_width,
    float inv_l, float inv_w, float inv_diag, float inv_max, float inv_pi,
    float inv_16_54, float inv_8_07,
    int fuel_capacity, int fuel_count, bool offense, bool normalize,
    hipStream_t stream) {
  (void)piece_active; (void)piece_owner;
  const int threads = 256;
  const int blocks = static_cast<int>((rows * kFeatures + threads - 1) / threads);
  if (!blocks) return;
  hipLaunchKernelGGL(assemble_kernel, dim3(blocks), dim3(threads), 0, stream,
      own_pose, own_vel, other_pose, other_vel, opponent, obstacles, length,
      width, accel, speed, omega, path, path_lengths, track_pos, track_vel,
      indices, valid, nearest, own_count, piece_active, piece_owner, elapsed,
      hub_active, scores, hubs, hub, capture_mask, output, rows, pieces, path_capacity,
      track_pos_stride, track_vel_stride, field_length, field_width,
      inv_l, inv_w, inv_diag, inv_max, inv_pi, inv_16_54, inv_8_07,
      fuel_capacity, fuel_count, offense, normalize);
}
