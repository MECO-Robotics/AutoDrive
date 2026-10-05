// Fused implementation of TensorVectorizedSimulator._swerve_eager followed by
// pose integration. A lane owns one robot and indexes world-level state by its
// robot batch stride.
#include <hip/hip_runtime.h>
#include <cmath>

#pragma clang fp contract(off)

namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 6.28318530717958647692f;
constexpr float kGravity = 9.81f;
constexpr float kBumpPanel = 0.0127f;
constexpr float kBumpRise = 0.1527302f; // Tensor field constant, meters.
constexpr float kBumpCg = 0.30f;
constexpr float kBumpRolling = 0.025f;

__device__ __forceinline__ float clampf(float x, float lo, float hi) {
  return fminf(fmaxf(x, lo), hi);
}
__device__ __forceinline__ float signf_torch(float x) {
  return static_cast<float>((x > 0.f) - (x < 0.f));
}
__device__ __forceinline__ float sum4_torch(float a, float b, float c, float d) {
  return (a + b) + (c + d);
}
__device__ __forceinline__ float remainder_positive(float x, float y) {
  float r = fmodf(x, y);
  return r < 0.f ? r + y : r;
}

__global__ void swerve_pose_kernel(
    float* pose, float* velocity, const float* command, const bool* active,
    const float* length, const float* width, const float* mass,
    const float* speed_limit, const float* accel_limit,
    const float* omega_limit, const float* alpha_limit,
    const float* ground_mu, const float* lateral_mu,
    const float* yaw_inertia_multiplier, const float* drive_ratio,
    const float* drive_current_limit, const float* drive_supply_limit,
    const float* robot_supply_limit, const float* battery_resistance,
    float* module_angle, float* module_steer_rate,
    float* module_drive_speed, float* module_current,
    float* module_supply_current, float* robot_current,
    const float* bumps, float* debug, int worlds, int robots, int bump_count,
    float dt, float module_x_offset, float module_y_offset,
    float inverse_wheel_radius, float steer_current_limit, float steer_rate_limit,
    float steer_acceleration, float inverse_steer_current_limit,
    float inverse_resistance, float inverse_stall, float inverse_kv, float kt,
    float free_current_a, float motor_efficiency, float battery_voltage,
    float inverse_four_dt) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= worlds * robots) return;
  const int world = robots == 2 ? (row >> 1) : (row / robots);
  if (!active[world]) return;

  const int p3 = row * 3;
  const int m4 = row * 4;
  const float mx[4] = {-module_x_offset, module_x_offset,
                       -module_x_offset, module_x_offset};
  const float my[4] = {-module_y_offset, -module_y_offset,
                        module_y_offset, module_y_offset};
  const float theta = pose[p3 + 2];
  const float c = cosf(theta);
  const float s = sinf(theta);
  float wheel_height[4] = {0.f, 0.f, 0.f, 0.f};
  float wheel_grade[4] = {0.f, 0.f, 0.f, 0.f};
  bool on_bump[4] = {false, false, false, false};

  // Match Torch argmax's first-index tie behavior by replacing a selected
  // bump only for a strictly greater height.
  for (int k = 0; k < 4; ++k) {
    const float wx = pose[p3] + c * mx[k] - s * my[k];
    const float wy = pose[p3 + 1] + s * mx[k] + c * my[k];
    float best_height = 0.f;
    float selected_grade = 0.f;
    for (int b = 0; b < bump_count; ++b) {
      const float dx = wx - bumps[b * 4];
      const float dy = wy - bumps[b * 4 + 1];
      const float half_x = fmaxf(bumps[b * 4 + 2], 1.0e-6f);
      const bool within = fabsf(dx) <= half_x && fabsf(dy) <= bumps[b * 4 + 3];
      const float h = within ? kBumpPanel + kBumpRise * (1.f - fabsf(dx) / half_x) : 0.f;
      const float grade = within ? -signf_torch(dx) * kBumpRise / half_x : 0.f;
      on_bump[k] = on_bump[k] || within;
      if (h > best_height) {
        best_height = h;
        selected_grade = grade;
      }
    }
    wheel_height[k] = best_height;
    wheel_grade[k] = selected_grade;
  }
  const float front_height = (wheel_height[1] + wheel_height[3]) * 0.5f;
  const float rear_height = (wheel_height[0] + wheel_height[2]) * 0.5f;
  const float chassis_pitch = atan2f(front_height - rear_height, length[row]);
  const float pitch_cos = cosf(chassis_pitch);
  const float pitch_sin = sinf(chassis_pitch);
  float normal[4];
  for (int k = 0; k < 4; ++k) {
    normal[k] = fmaxf(mass[row] * kGravity * pitch_cos / 4.f -
        mass[row] * kGravity * kBumpCg * pitch_sin * signf_torch(mx[k]) /
            (2.f * fmaxf(length[row], 1.0e-6f)), 0.f);
  }

  float vx_cmd = c * command[p3] + s * command[p3 + 1];
  float vy_cmd = -s * command[p3] + c * command[p3 + 1];
  const float command_norm = fmaxf(sqrtf(vx_cmd * vx_cmd + vy_cmd * vy_cmd), 1.0e-8f);
  const float command_scale = fminf(speed_limit[row] / command_norm, 1.f);
  vx_cmd *= command_scale;
  vy_cmd *= command_scale;
  const float omega_cmd = clampf(command[p3 + 2], -omega_limit[row], omega_limit[row]);

  float target[4], steer_current[4], new_steer_rate[4], new_angle[4];
  float delta_x[4], delta_y[4];
  float max_target = 0.f;
  for (int k = 0; k < 4; ++k) {
    delta_x[k] = vx_cmd - omega_cmd * my[k];
    delta_y[k] = vy_cmd + omega_cmd * mx[k];
    target[k] = sqrtf(delta_x[k] * delta_x[k] + delta_y[k] * delta_y[k]);
    max_target = fmaxf(max_target, target[k]);
  }
  const float module_scale = fminf(speed_limit[row] / fmaxf(max_target, 1.0e-8f), 1.f);
  for (int k = 0; k < 4; ++k) {
    target[k] *= module_scale;
    float angle = atan2f(delta_y[k], delta_x[k]);
    float delta = remainder_positive(angle - module_angle[m4 + k] + kPi, kTwoPi) - kPi;
    const bool reverse = fabsf(delta) > kPi * 0.5f;
    if (reverse) {
      angle += kPi;
      target[k] = -target[k];
    }
    angle = remainder_positive(angle + kPi, kTwoPi) - kPi;
    delta = remainder_positive(angle - module_angle[m4 + k] + kPi, kTwoPi) - kPi;
    const float steer_target = clampf(8.f * delta, -steer_rate_limit, steer_rate_limit);
    const float se = steer_target - module_steer_rate[m4 + k];
    steer_current[k] = fminf(2.f * fabsf(se), steer_current_limit);
    new_steer_rate[k] = clampf(module_steer_rate[m4 + k] +
        signf_torch(se) * steer_acceleration * steer_current[k] *
            inverse_steer_current_limit * dt,
        -steer_rate_limit, steer_rate_limit);
    new_angle[k] = remainder_positive(module_angle[m4 + k] +
        new_steer_rate[k] * dt + kPi, kTwoPi) - kPi;
  }

  const float bus = fmaxf(battery_voltage - robot_current[row] * battery_resistance[row], 0.f);
  float amps[4], supply[4];
  float supply_sum, steer_sum;
  for (int k = 0; k < 4; ++k) {
    const float motor_target = target[k] * inverse_wheel_radius * drive_ratio[row];
    const float actual = module_drive_speed[m4 + k] * inverse_wheel_radius * drive_ratio[row];
    const float duty_term1 = motor_target * inverse_stall;
    const float duty_term2_base = 0.8f * (motor_target - actual);
    const float duty_term2 = duty_term2_base * inverse_stall;
    const float duty = clampf(duty_term1 + duty_term2, -1.f, 1.f);
    const float numerator = duty * bus - actual * inverse_kv;
    const float amps_unlimited = numerator * inverse_resistance;
    amps[k] = clampf(amps_unlimited, -drive_current_limit[row], drive_current_limit[row]);
    supply[k] = fabsf(duty * amps[k]);
    const float supply_pre_controller = supply[k];
    const float controller_scale = fminf(__fdiv_rn(drive_supply_limit[row],
        fmaxf(supply[k], 1.0e-6f)), 1.f);
    amps[k] *= controller_scale;
    supply[k] = fabsf(duty * amps[k]);
    if (debug) {
    const int d = (m4 + k) * 26;
    debug[d + 0] = target[k];
    debug[d + 1] = actual;
    debug[d + 2] = motor_target;
    debug[d + 3] = duty_term1;
    debug[d + 4] = duty_term2_base;
    debug[d + 5] = duty_term2;
    debug[d + 6] = duty;
    debug[d + 7] = bus;
    debug[d + 8] = numerator;
    debug[d + 9] = amps_unlimited;
    debug[d + 10] = clampf(amps_unlimited, -drive_current_limit[row], drive_current_limit[row]);
    debug[d + 11] = supply_pre_controller;
    debug[d + 12] = controller_scale;
    debug[d + 13] = amps[k];
    debug[d + 14] = supply[k];
    debug[d + 15] = steer_current[k];
    }
  }
  supply_sum = sum4_torch(supply[0], supply[1], supply[2], supply[3]);
  steer_sum = sum4_torch(steer_current[0], steer_current[1],
                         steer_current[2], steer_current[3]);
  const float drive_budget = fmaxf(robot_supply_limit[row] - steer_sum, 0.f);
  const float cap = fminf(__fdiv_rn(drive_budget,
      fmaxf(supply_sum, 1.0e-6f)), 1.f);
  for (int k = 0; k < 4; ++k) {
    amps[k] *= cap;
    supply[k] *= cap;
    if (debug) {
    const int d = (m4 + k) * 26;
    debug[d + 16] = supply_sum;
    debug[d + 17] = steer_sum;
    debug[d + 18] = drive_budget;
    debug[d + 19] = cap;
    debug[d + 20] = amps[k];
    debug[d + 21] = supply[k];
    }
    module_steer_rate[m4 + k] = new_steer_rate[k];
    module_angle[m4 + k] = new_angle[k];
    module_current[m4 + k] = amps[k];
    module_supply_current[m4 + k] = supply[k];
  }
  const float capped_supply_sum = sum4_torch(
      supply[0], supply[1], supply[2], supply[3]);
  robot_current[row] = capped_supply_sum + steer_sum;

  float body_vx = c * velocity[p3] + s * velocity[p3 + 1];
  float body_vy = -s * velocity[p3] + c * velocity[p3 + 1];
  float fx_drive[4], fy_drive[4], fx_lateral[4], fy_lateral[4];
  float lateral_force[4], force[4];
  const float lateral_limit_scale = lateral_mu[row];
  for (int k = 0; k < 4; ++k) {
    const float actual = module_drive_speed[m4 + k] * inverse_wheel_radius * drive_ratio[row];
    const float torque = kt * (amps[k] - free_current_a * signf_torch(actual));
    force[k] = torque * drive_ratio[row] * motor_efficiency * inverse_wheel_radius;
    const float longitudinal_limit = ground_mu[world] * normal[k];
    force[k] = fmaxf(fminf(force[k], longitudinal_limit), -longitudinal_limit);
    const float module_vx = body_vx - velocity[p3 + 2] * my[k];
    const float module_vy = body_vy + velocity[p3 + 2] * mx[k];
    const float ca = cosf(new_angle[k]);
    const float sa = sinf(new_angle[k]);
    const float lateral_velocity = -module_vx * sa + module_vy * ca;
    const float lateral_limit = lateral_limit_scale * normal[k];
    lateral_force[k] = clampf(-lateral_velocity * mass[row] * inverse_four_dt,
                              -lateral_limit, lateral_limit);
    const float long_norm = force[k] / fmaxf(longitudinal_limit, 1.0e-8f);
    const float lat_norm = lateral_force[k] / fmaxf(lateral_limit, 1.0e-8f);
    const float friction_scale = fmaxf(sqrtf(long_norm * long_norm + lat_norm * lat_norm), 1.f);
    force[k] /= friction_scale;
    lateral_force[k] /= friction_scale;
    fx_drive[k] = force[k] * ca;
    fy_drive[k] = force[k] * sa;
    fx_lateral[k] = -lateral_force[k] * sa;
    fy_lateral[k] = lateral_force[k] * ca;
  }
  float torque_terms[4];
  for (int k = 0; k < 4; ++k) {
    const float fx = fx_drive[k] + fx_lateral[k];
    const float fy = fy_drive[k] + fy_lateral[k];
    torque_terms[k] = mx[k] * fy - my[k] * fx;
  }
  const float sum_fx_drive = sum4_torch(
      fx_drive[0], fx_drive[1], fx_drive[2], fx_drive[3]);
  const float sum_fy_drive = sum4_torch(
      fy_drive[0], fy_drive[1], fy_drive[2], fy_drive[3]);
  const float sum_fx_lateral = sum4_torch(
      fx_lateral[0], fx_lateral[1], fx_lateral[2], fx_lateral[3]);
  const float sum_fy_lateral = sum4_torch(
      fy_lateral[0], fy_lateral[1], fy_lateral[2], fy_lateral[3]);
  const float torque_z = sum4_torch(
      torque_terms[0], torque_terms[1], torque_terms[2], torque_terms[3]);
  const float drive_ax = sum_fx_drive / mass[row];
  const float drive_ay = sum_fy_drive / mass[row];
  const float drive_scale = fminf(accel_limit[row] /
      fmaxf(sqrtf(drive_ax * drive_ax + drive_ay * drive_ay), 1.0e-8f), 1.f);
  const float ax = drive_ax * drive_scale + sum_fx_lateral / mass[row];
  const float ay = drive_ay * drive_scale + sum_fy_lateral / mass[row];
  const float inertia = mass[row] * (length[row] * length[row] +
      width[row] * width[row]) * (1.f / 12.f) * yaw_inertia_multiplier[row];
  const float az = torque_z / inertia;
  float vx = velocity[p3] + (c * ax - s * ay) * dt;
  const float vy = velocity[p3 + 1] + (s * ax + c * ay) * dt;
  const float mean_grade = sum4_torch(wheel_grade[0], wheel_grade[1],
                                      wheel_grade[2], wheel_grade[3]) * 0.25f;
  const float rolling_load = sum4_torch(
      kBumpRolling * normal[0] * static_cast<float>(on_bump[0]),
      kBumpRolling * normal[1] * static_cast<float>(on_bump[1]),
      kBumpRolling * normal[2] * static_cast<float>(on_bump[2]),
      kBumpRolling * normal[3] * static_cast<float>(on_bump[3]));
  const float rolling_accel = __fdiv_rn(rolling_load, mass[row]);
  for (int k = 0; k < 4; ++k) {
    if (debug) {
    const int d = (m4 + k) * 26;
    debug[d + 22] = wheel_grade[k];
    debug[d + 23] = normal[k];
    debug[d + 24] = mean_grade;
    debug[d + 25] = rolling_accel;
    }
  }
  vx += (-kGravity * mean_grade - rolling_accel * signf_torch(vx)) * dt;
  float vy_final = vy;
  const float vnorm = fmaxf(sqrtf(vx * vx + vy_final * vy_final), 1.0e-8f);
  const float velocity_scale = fminf(speed_limit[row] / vnorm, 1.f);
  vx *= velocity_scale;
  vy_final *= velocity_scale;
  const float omega = clampf(velocity[p3 + 2] + clampf(az, -alpha_limit[row],
      alpha_limit[row]) * dt, -omega_limit[row], omega_limit[row]);
  velocity[p3] = vx;
  velocity[p3 + 1] = vy_final;
  velocity[p3 + 2] = omega;
  body_vx = c * vx + s * vy_final;
  body_vy = -s * vx + c * vy_final;
  for (int k = 0; k < 4; ++k) {
    module_drive_speed[m4 + k] =
        (body_vx - omega * my[k]) * cosf(new_angle[k]) +
        (body_vy + omega * mx[k]) * sinf(new_angle[k]);
  }

  pose[p3] += vx * dt;
  pose[p3 + 1] += vy_final * dt;
  pose[p3 + 2] = remainder_positive(pose[p3 + 2] + omega * dt + kPi, kTwoPi) - kPi;
}
} // namespace

void swerve_pose_launch(
    float* pose, float* velocity, const float* command, const bool* active,
    const float* length, const float* width, const float* mass,
    const float* speed_limit, const float* accel_limit,
    const float* omega_limit, const float* alpha_limit,
    const float* ground_mu, const float* lateral_mu,
    const float* yaw_inertia_multiplier, const float* drive_ratio,
    const float* drive_current_limit, const float* drive_supply_limit,
    const float* robot_supply_limit, const float* battery_resistance,
    float* module_angle, float* module_steer_rate,
    float* module_drive_speed, float* module_current,
    float* module_supply_current, float* robot_current,
    const float* bumps, float* debug, int worlds, int robots, int bump_count,
    float dt, float module_x_offset, float module_y_offset,
    float inverse_wheel_radius, float steer_current_limit, float steer_rate_limit,
    float steer_acceleration, float inverse_steer_current_limit,
    float inverse_resistance, float inverse_stall, float inverse_kv, float kt,
    float free_current_a, float motor_efficiency, float battery_voltage,
    float inverse_four_dt,
    hipStream_t stream) {
  constexpr int threads = 128;
  const int rows = worlds * robots;
  const int blocks = (rows + threads - 1) / threads;
  hipLaunchKernelGGL(swerve_pose_kernel, dim3(blocks), dim3(threads), 0, stream,
      pose, velocity, command, active, length, width, mass, speed_limit,
      accel_limit, omega_limit, alpha_limit, ground_mu, lateral_mu,
      yaw_inertia_multiplier, drive_ratio, drive_current_limit,
      drive_supply_limit, robot_supply_limit, battery_resistance, module_angle,
      module_steer_rate, module_drive_speed, module_current, module_supply_current,
      robot_current, bumps, debug, worlds, robots, bump_count, dt, module_x_offset,
      module_y_offset, inverse_wheel_radius, steer_current_limit, steer_rate_limit,
      steer_acceleration, inverse_steer_current_limit, inverse_resistance,
      inverse_stall, inverse_kv, kt, free_current_a, motor_efficiency,
      battery_voltage, inverse_four_dt);
}
