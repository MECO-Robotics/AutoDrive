#include <hip/hip_runtime.h>

namespace {

__global__ void piece_occlusion_kernel(const float* pose_xy,
                                       const float* segment,
                                       const float* obstacles,
                                       const bool* eligible,
                                       bool* output,
                                       int pieces,
                                       int obstacle_count,
                                       int64_t total) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  if (!eligible[index]) {
    output[index] = false;
    return;
  }

  const int world = static_cast<int>(index / pieces);
  const float* pose = pose_xy + static_cast<int64_t>(world) * 2;
  const float* delta = segment + index * 2;
  // Match Torch's square-then-sum denominator order. Explicit round-to-nearest
  // operations plus --ffp-contract=off avoid contraction across the predicate.
  const float dx = delta[0];
  const float dy = delta[1];
  float denominator = __fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy));
  if (denominator < 1.0e-8f) denominator = 1.0e-8f;

  bool blocked = false;
  for (int obstacle = 0; obstacle < obstacle_count; ++obstacle) {
    const float* circle = obstacles + static_cast<int64_t>(obstacle) * 3;
    const float radius = __fadd_rn(circle[2], 0.03f);
    if (!(radius > 0.0f)) continue;

    const float rel_x = __fadd_rn(circle[0], -pose[0]);
    const float rel_y = __fadd_rn(circle[1], -pose[1]);
    const float dot = __fadd_rn(__fmul_rn(rel_x, dx), __fmul_rn(rel_y, dy));
    const float t = __fdiv_rn(dot, denominator);
    if (!(t > 0.02f && t < 0.98f)) continue;
    const float clamped_t = fminf(1.0f, fmaxf(0.0f, t));
    const float closest_x = __fadd_rn(pose[0], __fmul_rn(clamped_t, dx));
    const float closest_y = __fadd_rn(pose[1], __fmul_rn(clamped_t, dy));
    const float distance_x = __fadd_rn(closest_x, -circle[0]);
    const float distance_y = __fadd_rn(closest_y, -circle[1]);
    const float distance_sq = __fadd_rn(__fmul_rn(distance_x, distance_x),
                                        __fmul_rn(distance_y, distance_y));
    if (sqrtf(distance_sq) <= radius) {
      blocked = true;
      break;
    }
  }
  output[index] = blocked;
}

__global__ void visibility_kernel(const float* pose, const float* pieces_xy,
                                  const bool* piece_active,
                                  const int64_t* piece_owner, const bool* active,
                                  const float* obstacles, const float* other_xy,
                                  const float* other_radius, bool* output,
                                  int pieces, int obstacle_count, float range_m,
                                  float half_fov, bool full_fov, int64_t total) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int world = static_cast<int>(index / pieces);
  if (!active[world] || !piece_active[index] || piece_owner[index] >= 0) {
    output[index] = false;
    return;
  }
  const float* robot = pose + static_cast<int64_t>(world) * 3;
  const float* point = pieces_xy + index * 2;
  const float dx = __fadd_rn(point[0], -robot[0]);
  const float dy = __fadd_rn(point[1], -robot[1]);
  const float distance_sq = __fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy));
  if (!(sqrtf(distance_sq) <= range_m)) {
    output[index] = false;
    return;
  }
  if (!full_fov) {
    const float bearing = atan2f(dy, dx);
    const float difference = __fadd_rn(bearing, -robot[2]);
    const float angle = fabsf(atan2f(sinf(difference), cosf(difference)));
    if (!(angle <= half_fov)) {
      output[index] = false;
      return;
    }
  }

  const float denominator_raw = __fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy));
  const float denominator = fmaxf(denominator_raw, 1.0e-8f);
  for (int obstacle = 0; obstacle < obstacle_count; ++obstacle) {
    const float* circle = obstacles + static_cast<int64_t>(obstacle) * 3;
    const float radius = __fadd_rn(circle[2], 0.03f);
    if (!(radius > 0.0f)) continue;
    const float rel_x = __fadd_rn(circle[0], -robot[0]);
    const float rel_y = __fadd_rn(circle[1], -robot[1]);
    const float dot = __fadd_rn(__fmul_rn(rel_x, dx), __fmul_rn(rel_y, dy));
    const float t = __fdiv_rn(dot, denominator);
    if (!(t > 0.02f && t < 0.98f)) continue;
    const float clamped_t = fminf(1.0f, fmaxf(0.0f, t));
    const float closest_x = __fadd_rn(robot[0], __fmul_rn(clamped_t, dx));
    const float closest_y = __fadd_rn(robot[1], __fmul_rn(clamped_t, dy));
    const float distance_x = __fadd_rn(closest_x, -circle[0]);
    const float distance_y = __fadd_rn(closest_y, -circle[1]);
    const float closest_sq = __fadd_rn(__fmul_rn(distance_x, distance_x),
                                       __fmul_rn(distance_y, distance_y));
    if (sqrtf(closest_sq) <= radius) {
      output[index] = false;
      return;
    }
  }

  const float* opponent = other_xy + static_cast<int64_t>(world) * 2;
  const float rel_x = __fadd_rn(opponent[0], -robot[0]);
  const float rel_y = __fadd_rn(opponent[1], -robot[1]);
  const float dot = __fadd_rn(__fmul_rn(rel_x, dx), __fmul_rn(rel_y, dy));
  const float t = __fdiv_rn(dot, denominator);
  if (t > 0.02f && t < 0.98f) {
    const float clamped_t = fminf(1.0f, fmaxf(0.0f, t));
    const float closest_x = __fadd_rn(robot[0], __fmul_rn(clamped_t, dx));
    const float closest_y = __fadd_rn(robot[1], __fmul_rn(clamped_t, dy));
    const float distance_x = __fadd_rn(closest_x, -opponent[0]);
    const float distance_y = __fadd_rn(closest_y, -opponent[1]);
    const float closest_sq = __fadd_rn(__fmul_rn(distance_x, distance_x),
                                       __fmul_rn(distance_y, distance_y));
    if (sqrtf(closest_sq) <= other_radius[world]) {
      output[index] = false;
      return;
    }
  }
  output[index] = true;
}

__global__ void visibility_3v3_kernel(const float* pose, const float* pieces_xy,
                                      const bool* piece_active,
                                      const int64_t* piece_owner, const bool* active,
                                      const float* obstacles,
                                      const float* robot_radius, bool* output,
                                      int robots, int pieces, int obstacle_count,
                                      float range_m, float half_fov, bool full_fov,
                                      int64_t total) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int64_t per_world = static_cast<int64_t>(robots) * pieces;
  const int world = static_cast<int>(index / per_world);
  const int within_world = static_cast<int>(index - static_cast<int64_t>(world) * per_world);
  const int robot_id = within_world / pieces;
  const int piece_id = within_world - robot_id * pieces;
  const int64_t piece_index = static_cast<int64_t>(world) * pieces + piece_id;
  if (!active[world] || !piece_active[piece_index] || piece_owner[piece_index] >= 0) {
    output[index] = false;
    return;
  }

  const float* robot = pose + static_cast<int64_t>(world) * robots * 3 + robot_id * 3;
  const float* point = pieces_xy + piece_index * 2;
  const float dx = __fadd_rn(point[0], -robot[0]);
  const float dy = __fadd_rn(point[1], -robot[1]);
  const float distance_sq = __fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy));
  if (!(sqrtf(distance_sq) <= range_m)) {
    output[index] = false;
    return;
  }
  if (!full_fov) {
    const float bearing = atan2f(dy, dx);
    const float difference = __fadd_rn(bearing, -robot[2]);
    const float angle = fabsf(atan2f(sinf(difference), cosf(difference)));
    if (!(angle <= half_fov)) {
      output[index] = false;
      return;
    }
  }

  const float denominator_raw = __fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy));
  const float denominator = fmaxf(denominator_raw, 1.0e-8f);
  for (int obstacle = 0; obstacle < obstacle_count; ++obstacle) {
    const float* circle = obstacles + static_cast<int64_t>(obstacle) * 3;
    const float radius = __fadd_rn(circle[2], 0.03f);
    if (!(radius > 0.0f)) continue;
    const float rel_x = __fadd_rn(circle[0], -robot[0]);
    const float rel_y = __fadd_rn(circle[1], -robot[1]);
    const float dot = __fadd_rn(__fmul_rn(rel_x, dx), __fmul_rn(rel_y, dy));
    const float t = __fdiv_rn(dot, denominator);
    if (!(t > 0.02f && t < 0.98f)) continue;
    const float clamped_t = fminf(1.0f, fmaxf(0.0f, t));
    const float closest_x = __fadd_rn(robot[0], __fmul_rn(clamped_t, dx));
    const float closest_y = __fadd_rn(robot[1], __fmul_rn(clamped_t, dy));
    const float distance_x = __fadd_rn(closest_x, -circle[0]);
    const float distance_y = __fadd_rn(closest_y, -circle[1]);
    const float closest_sq = __fadd_rn(__fmul_rn(distance_x, distance_x),
                                       __fmul_rn(distance_y, distance_y));
    if (sqrtf(closest_sq) <= radius) {
      output[index] = false;
      return;
    }
  }

  for (int peer = 0; peer < robots; ++peer) {
    if (peer == robot_id) continue;
    const float* other = pose + static_cast<int64_t>(world) * robots * 3 + peer * 3;
    const float rel_x = __fadd_rn(other[0], -robot[0]);
    const float rel_y = __fadd_rn(other[1], -robot[1]);
    const float dot = __fadd_rn(__fmul_rn(rel_x, dx), __fmul_rn(rel_y, dy));
    const float t = __fdiv_rn(dot, denominator);
    if (!(t > 0.02f && t < 0.98f)) continue;
    const float clamped_t = fminf(1.0f, fmaxf(0.0f, t));
    const float closest_x = __fadd_rn(robot[0], __fmul_rn(clamped_t, dx));
    const float closest_y = __fadd_rn(robot[1], __fmul_rn(clamped_t, dy));
    const float distance_x = __fadd_rn(closest_x, -other[0]);
    const float distance_y = __fadd_rn(closest_y, -other[1]);
    const float closest_sq = __fadd_rn(__fmul_rn(distance_x, distance_x),
                                       __fmul_rn(distance_y, distance_y));
    const float radius = robot_radius[static_cast<int64_t>(world) * robots + peer];
    if (sqrtf(closest_sq) <= radius) {
      output[index] = false;
      return;
    }
  }
  output[index] = true;
}

__device__ __forceinline__ uint32_t perception_hash(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  value ^= value >> 16;
  return value;
}

__device__ __forceinline__ float perception_uniform(uint32_t seed,
                                                     uint32_t tick,
                                                     uint32_t world,
                                                     uint32_t robot,
                                                     uint32_t piece,
                                                     uint32_t lane) {
  uint32_t key = seed + tick * 0x9e3779b9u + world * 0x85ebca6bu +
                 robot * 0xc2b2ae35u + piece * 0x27d4eb2fu +
                 lane * 0x165667b1u;
  const uint32_t bits = perception_hash(key);
  return (static_cast<float>(bits) + 0.5f) * 2.3283064365386963e-10f;
}

__global__ void perception_commit_3v3_kernel(
    const bool* visible, const bool* active, const int64_t* ticks,
    const float* piece_pos, const float* piece_vel, float* track_pos,
    float* track_vel, float* track_age, bool* track_mask,
    uint32_t seed, int robots, int pieces, float dt, float dropout,
    float position_noise, float velocity_noise, int64_t total) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int64_t per_world = static_cast<int64_t>(robots) * pieces;
  const int world = static_cast<int>(index / per_world);
  if (!active[world]) return;
  const int within_world = static_cast<int>(index - static_cast<int64_t>(world) * per_world);
  const int robot = within_world / pieces;
  const int piece = within_world - robot * pieces;
  const int64_t piece_index = static_cast<int64_t>(world) * pieces + piece;
  const int64_t pair_index = index * 2;

  float age = track_age[index];
  if (track_mask[index]) age = __fadd_rn(age, dt);
  bool observed = visible[index];
  const uint32_t tick = static_cast<uint32_t>(ticks[world]);
  if (observed && dropout > 0.0f) {
    observed = perception_uniform(seed, tick, world, robot, piece, 0u) >= dropout;
  }
  if (observed) {
    const float p0 = perception_uniform(seed, tick, world, robot, piece, 1u);
    const float p1 = perception_uniform(seed, tick, world, robot, piece, 2u);
    const float v0 = perception_uniform(seed, tick, world, robot, piece, 3u);
    const float v1 = perception_uniform(seed, tick, world, robot, piece, 4u);
    const float p_radius = sqrtf(-2.0f * logf(p0));
    const float p_angle = 6.283185307179586f * p1;
    const float v_radius = sqrtf(-2.0f * logf(v0));
    const float v_angle = 6.283185307179586f * v1;
    track_pos[pair_index] = __fadd_rn(piece_pos[piece_index * 2],
        __fmul_rn(p_radius * cosf(p_angle), position_noise));
    track_pos[pair_index + 1] = __fadd_rn(piece_pos[piece_index * 2 + 1],
        __fmul_rn(p_radius * sinf(p_angle), position_noise));
    track_vel[pair_index] = __fadd_rn(piece_vel[piece_index * 2],
        __fmul_rn(v_radius * cosf(v_angle), velocity_noise));
    track_vel[pair_index + 1] = __fadd_rn(piece_vel[piece_index * 2 + 1],
        __fmul_rn(v_radius * sinf(v_angle), velocity_noise));
    track_age[index] = 0.0f;
    track_mask[index] = true;
  } else {
    track_age[index] = age;
    if (age > 1.0f) track_mask[index] = false;
  }
}

__device__ __forceinline__ float perception_normal(uint32_t seed,
                                                   uint32_t tick,
                                                   uint32_t world,
                                                   uint32_t robot,
                                                   uint32_t lane_a,
                                                   uint32_t lane_b) {
  const float u0 = perception_uniform(seed, tick, world, robot, 0u, lane_a);
  const float u1 = perception_uniform(seed, tick, world, robot, 0u, lane_b);
  return sqrtf(-2.0f * logf(u0)) * cosf(6.283185307179586f * u1);
}

__device__ __forceinline__ bool segment_blocked_by_circles(
    const float* origin, float dx, float dy, const float* obstacles,
    int obstacle_count, float denominator) {
  for (int obstacle = 0; obstacle < obstacle_count; ++obstacle) {
    const float* circle = obstacles + static_cast<int64_t>(obstacle) * 3;
    const float radius = __fadd_rn(circle[2], 0.03f);
    if (!(radius > 0.0f)) continue;
    const float rel_x = __fadd_rn(circle[0], -origin[0]);
    const float rel_y = __fadd_rn(circle[1], -origin[1]);
    const float dot = __fadd_rn(__fmul_rn(rel_x, dx), __fmul_rn(rel_y, dy));
    const float t = __fdiv_rn(dot, denominator);
    if (!(t > 0.02f && t < 0.98f)) continue;
    const float clamped_t = fminf(1.0f, fmaxf(0.0f, t));
    const float closest_x = __fadd_rn(origin[0], __fmul_rn(clamped_t, dx));
    const float closest_y = __fadd_rn(origin[1], __fmul_rn(clamped_t, dy));
    const float distance_x = __fadd_rn(closest_x, -circle[0]);
    const float distance_y = __fadd_rn(closest_y, -circle[1]);
    const float distance_sq = __fadd_rn(__fmul_rn(distance_x, distance_x),
                                        __fmul_rn(distance_y, distance_y));
    if (sqrtf(distance_sq) <= radius) return true;
  }
  return false;
}

__global__ void opponent_tracks_3v3_kernel(
    const float* pose, const float* length, const float* width,
    const float* acceleration, const float* robot_radius,
    const float* obstacles, const bool* active, const bool* controlled,
    const bool* defense_role, const int64_t* ticks, float* opponent_pose,
    float* opponent_velocity, float* opponent_size, float* opponent_age,
    bool* opponent_valid, uint32_t seed, int worlds, int obstacle_count,
    float range_m, float half_fov, float dropout, float position_noise,
    float velocity_noise, float dt, int64_t total) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int world = static_cast<int>(index / 6);
  const int robot_id = static_cast<int>(index - static_cast<int64_t>(world) * 6);
  if (!active[world]) return;
  const int64_t pose_base = static_cast<int64_t>(world) * 18;
  const int64_t track_index = static_cast<int64_t>(world) * 6 + robot_id;
  const float* own = pose + pose_base + robot_id * 3;
  float age = opponent_age[track_index];
  if (opponent_valid[track_index]) age = __fadd_rn(age, dt);

  int selected = -1;
  float selected_distance = HUGE_VALF;
  for (int enemy_slot = 0; enemy_slot < 3; ++enemy_slot) {
    const int enemy_id = robot_id < 3 ? enemy_slot + 3 : enemy_slot;
    if (defense_role[robot_id] && !controlled[enemy_id]) continue;
    const float* enemy = pose + pose_base + enemy_id * 3;
    const float dx = __fadd_rn(enemy[0], -own[0]);
    const float dy = __fadd_rn(enemy[1], -own[1]);
    const float distance_sq = __fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy));
    const float distance = sqrtf(distance_sq);
    if (!(distance <= range_m)) continue;
    const float bearing = atan2f(dy, dx);
    const float difference = __fadd_rn(bearing, -own[2]);
    const float angle = fabsf(atan2f(sinf(difference), cosf(difference)));
    if (!(angle <= half_fov)) continue;
    const float denominator = fmaxf(distance_sq, 1.0e-8f);
    if (segment_blocked_by_circles(own, dx, dy, obstacles,
                                   obstacle_count, denominator)) continue;

    bool occluded = false;
    for (int peer = 0; peer < 6; ++peer) {
      if (peer == robot_id || peer == enemy_id) continue;
      const float* other = pose + pose_base + peer * 3;
      const float rel_x = __fadd_rn(other[0], -own[0]);
      const float rel_y = __fadd_rn(other[1], -own[1]);
      const float dot = __fadd_rn(__fmul_rn(rel_x, dx), __fmul_rn(rel_y, dy));
      const float t = __fdiv_rn(dot, denominator);
      if (!(t > 0.02f && t < 0.98f)) continue;
      const float clamped_t = fminf(1.0f, fmaxf(0.0f, t));
      const float closest_x = __fadd_rn(own[0], __fmul_rn(clamped_t, dx));
      const float closest_y = __fadd_rn(own[1], __fmul_rn(clamped_t, dy));
      const float distance_x = __fadd_rn(closest_x, -other[0]);
      const float distance_y = __fadd_rn(closest_y, -other[1]);
      const float closest_sq = __fadd_rn(__fmul_rn(distance_x, distance_x),
                                         __fmul_rn(distance_y, distance_y));
      const float radius = robot_radius[static_cast<int64_t>(world) * 6 + peer];
      if (sqrtf(closest_sq) <= radius) {
        occluded = true;
        break;
      }
    }
    if (!occluded && distance < selected_distance) {
      selected = enemy_id;
      selected_distance = distance;
    }
  }

  bool observed = selected >= 0;
  const uint32_t tick = static_cast<uint32_t>(ticks[world]);
  if (observed && dropout > 0.0f) {
    observed = perception_uniform(seed, tick, world, robot_id, 0u, 16u) >= dropout;
  }
  if (observed) {
    const float* enemy = pose + pose_base + selected * 3;
    const float px = perception_normal(seed, tick, world, robot_id, 17u, 18u) * position_noise;
    const float py = perception_normal(seed, tick, world, robot_id, 19u, 20u) * position_noise;
    const float heading = perception_normal(seed, tick, world, robot_id, 21u, 22u) *
                          fminf(0.05f, position_noise);
    const float vx = perception_normal(seed, tick, world, robot_id, 23u, 24u) * velocity_noise;
    const float vy = perception_normal(seed, tick, world, robot_id, 25u, 26u) * velocity_noise;
    const float omega = perception_normal(seed, tick, world, robot_id, 27u, 28u) * velocity_noise;
    const int64_t base = track_index * 3;
    opponent_pose[base] = __fadd_rn(enemy[0], px);
    opponent_pose[base + 1] = __fadd_rn(enemy[1], py);
    opponent_pose[base + 2] = __fadd_rn(enemy[2], heading);
    opponent_velocity[base] = __fadd_rn(pose[pose_base + selected * 3], vx);
    opponent_velocity[base + 1] = __fadd_rn(pose[pose_base + selected * 3 + 1], vy);
    opponent_velocity[base + 2] = __fadd_rn(pose[pose_base + selected * 3 + 2], omega);
    opponent_size[base] = length[track_index - robot_id + selected];
    opponent_size[base + 1] = width[track_index - robot_id + selected];
    opponent_size[base + 2] = acceleration[track_index - robot_id + selected] / 10.0f;
    opponent_age[track_index] = 0.0f;
    opponent_valid[track_index] = true;
  } else {
    opponent_age[track_index] = age;
    if (age > 1.0f) opponent_valid[track_index] = false;
  }
}

}  // namespace

void piece_occlusion_launch(const float* pose_xy, const float* segment,
                            const float* obstacles, const bool* eligible,
                            bool* output,
                            int worlds, int pieces, int obstacle_count,
                            hipStream_t stream) {
  const int64_t total = static_cast<int64_t>(worlds) * pieces;
  if (total == 0) return;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(piece_occlusion_kernel, dim3(blocks), dim3(threads), 0,
                     stream, pose_xy, segment, obstacles, eligible, output, pieces,
                     obstacle_count, total);
}

void visibility_launch(const float* pose, const float* pieces_xy,
                       const bool* piece_active, const int64_t* piece_owner,
                       const bool* active, const float* obstacles,
                       const float* other_xy, const float* other_radius,
                       bool* output, int worlds, int pieces, int obstacle_count,
                       float range_m, float half_fov, bool full_fov,
                       hipStream_t stream) {
  const int64_t total = static_cast<int64_t>(worlds) * pieces;
  if (total == 0) return;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(visibility_kernel, dim3(blocks), dim3(threads), 0,
                     stream, pose, pieces_xy, piece_active, piece_owner, active,
                     obstacles, other_xy, other_radius, output, pieces,
                     obstacle_count, range_m, half_fov, full_fov, total);
}

void visibility_3v3_launch(const float* pose, const float* pieces_xy,
                           const bool* piece_active, const int64_t* piece_owner,
                           const bool* active, const float* obstacles,
                           const float* robot_radius, bool* output, int worlds,
                           int robots, int pieces, int obstacle_count, float range_m,
                           float half_fov, bool full_fov, hipStream_t stream) {
  const int64_t total = static_cast<int64_t>(worlds) * robots * pieces;
  if (total == 0) return;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(visibility_3v3_kernel, dim3(blocks), dim3(threads), 0,
                     stream, pose, pieces_xy, piece_active, piece_owner, active,
                     obstacles, robot_radius, output, robots, pieces,
                     obstacle_count, range_m, half_fov, full_fov, total);
}

void perception_commit_3v3_launch(const bool* visible, const bool* active,
                                  const int64_t* ticks, const float* piece_pos,
                                  const float* piece_vel, float* track_pos,
                                  float* track_vel, float* track_age,
                                  bool* track_mask, uint32_t seed, int worlds,
                                  int robots, int pieces, float dt, float dropout,
                                  float position_noise, float velocity_noise,
                                  hipStream_t stream) {
  const int64_t total = static_cast<int64_t>(worlds) * robots * pieces;
  if (total == 0) return;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(perception_commit_3v3_kernel, dim3(blocks), dim3(threads), 0,
      stream, visible, active, ticks, piece_pos, piece_vel, track_pos,
      track_vel, track_age, track_mask, seed, robots, pieces, dt, dropout,
      position_noise, velocity_noise, total);
}

void opponent_tracks_3v3_launch(const float* pose, const float* length,
                                const float* width, const float* acceleration,
                                const float* robot_radius, const float* obstacles,
                                const bool* active, const bool* controlled,
                                const bool* defense_role, const int64_t* ticks,
                                float* opponent_pose, float* opponent_velocity,
                                float* opponent_size, float* opponent_age,
                                bool* opponent_valid, uint32_t seed, int worlds,
                                int obstacle_count, float range_m, float half_fov,
                                float dropout, float position_noise,
                                float velocity_noise, float dt, hipStream_t stream) {
  const int64_t total = static_cast<int64_t>(worlds) * 6;
  if (total == 0) return;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(opponent_tracks_3v3_kernel, dim3(blocks), dim3(threads), 0,
      stream, pose, length, width, acceleration, robot_radius, obstacles, active,
      controlled, defense_role, ticks, opponent_pose, opponent_velocity,
      opponent_size, opponent_age, opponent_valid, seed, worlds, obstacle_count,
      range_m, half_fov, dropout, position_noise, velocity_noise, dt, total);
}
