#include <hip/hip_runtime.h>

namespace {

__global__ void perception_commit_kernel(
    const bool* active, const bool* visible, const float* measured,
    const float* measured_velocity, const bool* opponent_detected,
    const float* opponent_pose, const float* opponent_velocity,
    float* track_pos, float* track_vel, bool* track_mask, float* track_age,
    float* opponent_track_pose, float* opponent_track_velocity,
    bool* opponent_track_valid, float* opponent_track_age,
    int worlds, int pieces, float dt, float timeout, int64_t fuel_total) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= fuel_total + static_cast<int64_t>(worlds) * 2) return;

  if (index < fuel_total) {
    const int world = static_cast<int>(index / (2 * pieces));
    if (!active[world]) return;
    const int robot = static_cast<int>((index / pieces) % 2);
    const int piece = static_cast<int>(index % pieces);
    const int64_t track_index = index;
    float age = track_age[track_index];
    bool seen = visible[track_index];
    if (track_mask[track_index]) age = __fadd_rn(age, dt);
    if (seen) {
      const int64_t xy = index * 2;
      track_pos[xy] = measured[xy];
      track_pos[xy + 1] = measured[xy + 1];
      track_vel[xy] = measured_velocity[xy];
      track_vel[xy + 1] = measured_velocity[xy + 1];
      age = 0.0f;
    }
    bool tracked = track_mask[track_index] || seen;
    if (age > timeout) tracked = false;
    track_age[track_index] = age;
    track_mask[track_index] = tracked;
    return;
  }

  const int64_t opponent_index = index - fuel_total;
  const int world = static_cast<int>(opponent_index / 2);
  if (!active[world]) return;
  const int robot = static_cast<int>(opponent_index % 2);
  float age = opponent_track_age[opponent_index];
  const bool was_valid = opponent_track_valid[opponent_index];
  const bool detected = opponent_detected[opponent_index];
  if (was_valid) age = __fadd_rn(age, dt);
  if (detected) {
    const int64_t state_index = opponent_index * 3;
    opponent_track_pose[state_index] = opponent_pose[state_index];
    opponent_track_pose[state_index + 1] = opponent_pose[state_index + 1];
    opponent_track_pose[state_index + 2] = opponent_pose[state_index + 2];
    opponent_track_velocity[state_index] = opponent_velocity[state_index];
    opponent_track_velocity[state_index + 1] = opponent_velocity[state_index + 1];
    opponent_track_velocity[state_index + 2] = opponent_velocity[state_index + 2];
    age = 0.0f;
  }
  bool valid = was_valid || detected;
  if (age > timeout) valid = false;
  opponent_track_age[opponent_index] = age;
  opponent_track_valid[opponent_index] = valid;
}

}  // namespace

void perception_commit_launch(
    const bool* active, const bool* visible, const float* measured,
    const float* measured_velocity, const bool* opponent_detected,
    const float* opponent_pose, const float* opponent_velocity,
    float* track_pos, float* track_vel, bool* track_mask, float* track_age,
    float* opponent_track_pose, float* opponent_track_velocity,
    bool* opponent_track_valid, float* opponent_track_age,
    int worlds, int pieces, float dt, float timeout, hipStream_t stream) {
  const int64_t fuel_total = static_cast<int64_t>(worlds) * 2 * pieces;
  const int64_t total = fuel_total + static_cast<int64_t>(worlds) * 2;
  if (total == 0) return;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(perception_commit_kernel, dim3(blocks), dim3(threads), 0,
      stream, active, visible, measured, measured_velocity, opponent_detected,
      opponent_pose, opponent_velocity, track_pos, track_vel, track_mask,
      track_age, opponent_track_pose, opponent_track_velocity,
      opponent_track_valid, opponent_track_age, worlds, pieces, dt, timeout,
      fuel_total);
}
