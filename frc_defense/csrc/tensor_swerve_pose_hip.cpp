// Fused swerve + pose integration for two- and six-robot batches.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime_api.h>

void swerve_pose_launch(
    float* pose, float* velocity, const float* command, const bool* active,
    const float* length, const float* width, const float* mass,
    const float* speed_limit, const float* accel_limit,
    const float* omega_limit, const float* alpha_limit,
    const float* ground_mu, const float* lateral_mu,
    const float* yaw_inertia_multiplier, const float* drive_ratio,
    const float* drive_current_limit, const float* drive_supply_limit,
    const float* robot_supply_limit, const float* battery_resistance,
    float* module_angle, float* module_steer_rate, float* module_drive_speed,
    float* module_current, float* module_supply_current, float* robot_current,
    bool* robot_contact, bool* robot_contact_by_robot, bool* opponent_contact,
    bool* field_contact, bool* wall_contact, bool clear_contact_flags,
    const float* bumps, float* debug,
    int worlds, int robots, int bump_count, float dt, float module_x_offset,
    float module_y_offset, float inverse_wheel_radius, float steer_current_limit,
    float steer_rate_limit, float steer_acceleration,
    float inverse_steer_current_limit, float inverse_resistance,
    float inverse_stall, float inverse_kv, float kt, float free_current_a,
    float motor_efficiency, float battery_voltage, float inverse_four_dt,
    hipStream_t stream);

void step(torch::Tensor pose, torch::Tensor velocity, torch::Tensor command,
    torch::Tensor active, torch::Tensor length, torch::Tensor width,
    torch::Tensor mass, torch::Tensor speed, torch::Tensor acceleration,
    torch::Tensor omega_limit, torch::Tensor alpha_limit,
    torch::Tensor ground_mu, torch::Tensor lateral_mu,
    torch::Tensor yaw_inertia_multiplier, torch::Tensor drive_ratio,
    torch::Tensor drive_current_limit, torch::Tensor drive_supply_limit,
    torch::Tensor robot_supply_limit, torch::Tensor battery_resistance,
    torch::Tensor module_angle, torch::Tensor module_steer_rate,
    torch::Tensor module_drive_speed, torch::Tensor module_current,
    torch::Tensor module_supply_current, torch::Tensor robot_current,
    torch::Tensor robot_contact, torch::Tensor robot_contact_by_robot,
    torch::Tensor opponent_contact, torch::Tensor field_contact,
    torch::Tensor wall_contact, bool clear_contact_flags,
    torch::Tensor bump_regions, torch::Tensor debug, double dt, double module_x_offset,
    double module_y_offset, double inverse_wheel_radius, double steer_current_limit,
    double steer_rate_limit, double steer_acceleration,
    double inverse_steer_current_limit, double inverse_resistance,
    double inverse_stall, double inverse_kv, double kt, double free_current_a,
    double motor_efficiency, double battery_voltage, double inverse_four_dt) {
  TORCH_CHECK(pose.is_cuda() && pose.scalar_type() == at::kFloat &&
              pose.dim() == 3 && (pose.size(1) == 2 || pose.size(1) == 6) &&
              pose.size(2) == 3,
              "pose must be CUDA float [world,2|6,3]");
  const auto n = pose.size(0);
  const auto robots = pose.size(1);
  auto require_shape = [n, robots](const torch::Tensor& t, at::IntArrayRef shape,
                           at::ScalarType dtype, const char* name) {
    TORCH_CHECK(t.sizes() == shape && t.scalar_type() == dtype,
                name, " has invalid shape or dtype");
  };
  require_shape(velocity, {n,robots,3}, at::kFloat, "velocity");
  require_shape(command, {n,robots,3}, at::kFloat, "command");
  require_shape(active, {n}, at::kBool, "active");
  for (const auto& t : {length,width,mass,speed,acceleration,omega_limit,alpha_limit,
       lateral_mu,yaw_inertia_multiplier,drive_ratio,drive_current_limit,
       drive_supply_limit,robot_supply_limit,battery_resistance})
    require_shape(t, {n,robots}, at::kFloat, "row parameter");
  require_shape(ground_mu, {n}, at::kFloat, "ground_mu");
  for (const auto& t : {module_angle,module_steer_rate,module_drive_speed,
       module_current,module_supply_current})
    require_shape(t, {n,robots,4}, at::kFloat, "module state");
  require_shape(robot_current, {n,robots}, at::kFloat, "robot_current");
  require_shape(robot_contact, {n}, at::kBool, "robot_contact");
  require_shape(robot_contact_by_robot, {n,robots}, at::kBool,
                "robot_contact_by_robot");
  require_shape(opponent_contact, {n}, at::kBool, "opponent_contact");
  require_shape(field_contact, {n,robots}, at::kBool, "field_contact");
  require_shape(wall_contact, {n,robots,2}, at::kBool, "wall_contact");
  TORCH_CHECK(bump_regions.dim() == 2 && bump_regions.size(1) == 4 &&
              bump_regions.scalar_type() == at::kFloat,
              "bump_regions must be float [B,4]");
  TORCH_CHECK(debug.scalar_type() == at::kFloat,
              "debug must be float");
  if (debug.numel() != 0)
    require_shape(debug, {n,robots,4,26}, at::kFloat, "debug");
  for (const auto& t : {pose,velocity,command,active,length,width,mass,speed,
       acceleration,omega_limit,alpha_limit,ground_mu,lateral_mu,
       yaw_inertia_multiplier,drive_ratio,drive_current_limit,drive_supply_limit,
       robot_supply_limit,battery_resistance,module_angle,module_steer_rate,
       module_drive_speed,module_current,module_supply_current,robot_current,
       robot_contact,robot_contact_by_robot,opponent_contact,field_contact,
       wall_contact,bump_regions,debug})
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.device() == pose.device(),
                "all swerve tensors must be contiguous on one accelerator");
  c10::cuda::CUDAGuard guard(pose.device());
  swerve_pose_launch(pose.data_ptr<float>(), velocity.data_ptr<float>(),
      command.data_ptr<float>(), active.data_ptr<bool>(), length.data_ptr<float>(),
      width.data_ptr<float>(), mass.data_ptr<float>(), speed.data_ptr<float>(),
      acceleration.data_ptr<float>(), omega_limit.data_ptr<float>(),
      alpha_limit.data_ptr<float>(), ground_mu.data_ptr<float>(),
      lateral_mu.data_ptr<float>(), yaw_inertia_multiplier.data_ptr<float>(),
      drive_ratio.data_ptr<float>(), drive_current_limit.data_ptr<float>(),
      drive_supply_limit.data_ptr<float>(), robot_supply_limit.data_ptr<float>(),
      battery_resistance.data_ptr<float>(), module_angle.data_ptr<float>(),
      module_steer_rate.data_ptr<float>(), module_drive_speed.data_ptr<float>(),
      module_current.data_ptr<float>(), module_supply_current.data_ptr<float>(),
      robot_current.data_ptr<float>(), robot_contact.data_ptr<bool>(),
      robot_contact_by_robot.data_ptr<bool>(), opponent_contact.data_ptr<bool>(),
      field_contact.data_ptr<bool>(), wall_contact.data_ptr<bool>(),
      clear_contact_flags,
      bump_regions.data_ptr<float>(),
      debug.numel() == 0 ? nullptr : debug.data_ptr<float>(),
      static_cast<int>(n), static_cast<int>(robots),
      static_cast<int>(bump_regions.size(0)),
      static_cast<float>(dt),
      static_cast<float>(module_x_offset), static_cast<float>(module_y_offset),
      static_cast<float>(inverse_wheel_radius), static_cast<float>(steer_current_limit),
      static_cast<float>(steer_rate_limit), static_cast<float>(steer_acceleration),
      static_cast<float>(inverse_steer_current_limit),
      static_cast<float>(inverse_resistance), static_cast<float>(inverse_stall),
      static_cast<float>(inverse_kv), static_cast<float>(kt),
      static_cast<float>(free_current_a), static_cast<float>(motor_efficiency),
      static_cast<float>(battery_voltage), static_cast<float>(inverse_four_dt),
      c10::cuda::getCurrentCUDAStream(pose.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("step", &step, "Experimental fused swerve and pose integration (HIP)");
}
