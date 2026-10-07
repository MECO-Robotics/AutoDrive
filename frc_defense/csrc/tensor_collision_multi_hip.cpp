#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

void robot_contacts_launch(float*, float*, const float*, const float*,
    const float*, const float*, const float*, const bool*, const int64_t*,
    bool*, bool*, int, hipStream_t);
void field_contacts_launch(float*, float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const bool*,
    bool*, int, int, hipStream_t);
void field_sweep_contacts_launch(float*, float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const bool*,
    bool*, int, int, int, float, hipStream_t);
void safe_score_targets_launch(const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const bool*,
    float*, int, int, int, float, hipStream_t);
void avoidance_launch(const float*, const float*, const float*, const float*,
    const float*, const float*, const bool*, const int64_t*, const bool*,
    int64_t*, float*, int, bool, hipStream_t);

void robot_contacts(torch::Tensor pose, torch::Tensor velocity,
    torch::Tensor length, torch::Tensor width, torch::Tensor mass,
    torch::Tensor yaw_inertia_multiplier, torch::Tensor mu,
    torch::Tensor active, torch::Tensor team_ids,
    torch::Tensor robot_contact, torch::Tensor opponent_contact) {
  const auto n = pose.size(0);
  TORCH_CHECK(pose.is_cuda() && pose.scalar_type() == at::kFloat &&
              pose.sizes() == torch::IntArrayRef({n, 6, 3}),
              "pose must be HIP float [world,6,3]");
  TORCH_CHECK(velocity.sizes() == pose.sizes() && velocity.scalar_type() == at::kFloat &&
              length.sizes() == torch::IntArrayRef({n, 6}) &&
              width.sizes() == length.sizes() && mass.sizes() == length.sizes() &&
              yaw_inertia_multiplier.sizes() == length.sizes() &&
              mu.sizes() == torch::IntArrayRef({n}) &&
              active.sizes() == mu.sizes() && active.scalar_type() == at::kBool &&
              team_ids.sizes() == torch::IntArrayRef({6}) &&
              team_ids.scalar_type() == at::kLong &&
              robot_contact.sizes() == mu.sizes() && robot_contact.scalar_type() == at::kBool &&
              opponent_contact.sizes() == mu.sizes() && opponent_contact.scalar_type() == at::kBool,
              "six-robot collision shape or dtype mismatch");
  for (const auto& t : {pose, velocity, length, width, mass, yaw_inertia_multiplier,
                        mu, active, team_ids, robot_contact, opponent_contact})
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.device() == pose.device(),
                "collision tensors must be contiguous on the same HIP device");
  robot_contacts_launch(pose.data_ptr<float>(), velocity.data_ptr<float>(),
      length.data_ptr<float>(), width.data_ptr<float>(), mass.data_ptr<float>(),
      yaw_inertia_multiplier.data_ptr<float>(), mu.data_ptr<float>(),
      active.data_ptr<bool>(), team_ids.data_ptr<int64_t>(),
      robot_contact.data_ptr<bool>(), opponent_contact.data_ptr<bool>(),
      static_cast<int>(n), c10::cuda::getCurrentCUDAStream(pose.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void field_contacts(torch::Tensor pose, torch::Tensor velocity,
    torch::Tensor length, torch::Tensor width, torch::Tensor mass,
    torch::Tensor yaw_inertia_multiplier, torch::Tensor wall_mu,
    torch::Tensor boxes, torch::Tensor active, torch::Tensor field_contact) {
  const auto n = pose.size(0);
  TORCH_CHECK(pose.is_cuda() && pose.scalar_type() == at::kFloat &&
              pose.sizes() == torch::IntArrayRef({n, 6, 3}),
              "pose must be HIP float [world,6,3]");
  TORCH_CHECK(velocity.sizes() == pose.sizes() && velocity.scalar_type() == at::kFloat &&
              length.sizes() == torch::IntArrayRef({n, 6}) &&
              width.sizes() == length.sizes() && mass.sizes() == length.sizes() &&
              yaw_inertia_multiplier.sizes() == length.sizes() &&
              wall_mu.sizes() == torch::IntArrayRef({n}) &&
              boxes.dim() == 2 && boxes.size(1) == 4 && boxes.scalar_type() == at::kFloat &&
              active.sizes() == wall_mu.sizes() && active.scalar_type() == at::kBool &&
              field_contact.sizes() == length.sizes() && field_contact.scalar_type() == at::kBool,
              "six-robot field collision shape or dtype mismatch");
  for (const auto& t : {pose, velocity, length, width, mass, yaw_inertia_multiplier,
                        wall_mu, boxes, active, field_contact})
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.device() == pose.device(),
                "field collision tensors must be contiguous on the same HIP device");
  field_contacts_launch(pose.data_ptr<float>(), velocity.data_ptr<float>(),
      length.data_ptr<float>(), width.data_ptr<float>(), mass.data_ptr<float>(),
      yaw_inertia_multiplier.data_ptr<float>(), wall_mu.data_ptr<float>(),
      boxes.data_ptr<float>(), active.data_ptr<bool>(), field_contact.data_ptr<bool>(),
      static_cast<int>(n), static_cast<int>(boxes.size(0)),
      c10::cuda::getCurrentCUDAStream(pose.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void field_sweep_contacts(torch::Tensor pose, torch::Tensor velocity,
    torch::Tensor length, torch::Tensor width, torch::Tensor mass,
    torch::Tensor yaw_inertia_multiplier, torch::Tensor wall_mu,
    torch::Tensor boxes, torch::Tensor active, torch::Tensor field_contact,
    int sweep_steps, double substep_dt) {
  const auto n = pose.size(0);
  TORCH_CHECK(sweep_steps > 0 && substep_dt > 0.0,
              "field sweep steps and substep dt must be positive");
  TORCH_CHECK(pose.is_cuda() && pose.scalar_type() == at::kFloat &&
              pose.sizes() == torch::IntArrayRef({n, 6, 3}),
              "pose must be HIP float [world,6,3]");
  TORCH_CHECK(velocity.sizes() == pose.sizes() && velocity.scalar_type() == at::kFloat &&
              length.sizes() == torch::IntArrayRef({n, 6}) &&
              width.sizes() == length.sizes() && mass.sizes() == length.sizes() &&
              yaw_inertia_multiplier.sizes() == length.sizes() &&
              wall_mu.sizes() == torch::IntArrayRef({n}) &&
              boxes.dim() == 2 && boxes.size(1) == 4 && boxes.scalar_type() == at::kFloat &&
              active.sizes() == wall_mu.sizes() && active.scalar_type() == at::kBool &&
              field_contact.sizes() == length.sizes() && field_contact.scalar_type() == at::kBool,
              "six-robot field sweep shape or dtype mismatch");
  for (const auto& t : {pose, velocity, length, width, mass, yaw_inertia_multiplier,
                        wall_mu, boxes, active, field_contact})
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.device() == pose.device(),
                "field sweep tensors must be contiguous on the same HIP device");
  field_sweep_contacts_launch(pose.data_ptr<float>(), velocity.data_ptr<float>(),
      length.data_ptr<float>(), width.data_ptr<float>(), mass.data_ptr<float>(),
      yaw_inertia_multiplier.data_ptr<float>(), wall_mu.data_ptr<float>(),
      boxes.data_ptr<float>(), active.data_ptr<bool>(), field_contact.data_ptr<bool>(),
      static_cast<int>(n), static_cast<int>(boxes.size(0)), sweep_steps,
      static_cast<float>(substep_dt),
      c10::cuda::getCurrentCUDAStream(pose.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor safe_score_targets(torch::Tensor pose,
    torch::Tensor score_target, torch::Tensor robot_radius,
    torch::Tensor x_min, torch::Tensor x_max, torch::Tensor boxes,
    torch::Tensor grid_points, torch::Tensor grid_clear, double field_width) {
  const auto n = pose.size(0);
  const auto grid_count = grid_points.size(2);
  TORCH_CHECK(pose.is_cuda() && pose.scalar_type() == at::kFloat &&
              pose.sizes() == torch::IntArrayRef({n, 6, 3}) &&
              score_target.sizes() == torch::IntArrayRef({n, 6, 2}) &&
              robot_radius.sizes() == torch::IntArrayRef({n, 6}) &&
              x_min.sizes() == robot_radius.sizes() &&
              x_max.sizes() == robot_radius.sizes() &&
              boxes.dim() == 2 && boxes.size(1) == 4 &&
              grid_points.sizes() == torch::IntArrayRef({n, 6, grid_count, 2}) &&
              grid_clear.sizes() == torch::IntArrayRef({n, 6, grid_count}) &&
              score_target.scalar_type() == at::kFloat &&
              robot_radius.scalar_type() == at::kFloat &&
              x_min.scalar_type() == at::kFloat && x_max.scalar_type() == at::kFloat &&
              boxes.scalar_type() == at::kFloat && grid_points.scalar_type() == at::kFloat &&
              grid_clear.scalar_type() == at::kBool,
              "safe score target tensor shape or dtype mismatch");
  for (const auto& t : {pose, score_target, robot_radius, x_min, x_max,
                        boxes, grid_points, grid_clear})
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.device() == pose.device(),
                "safe score target tensors must be contiguous on one HIP device");
  auto output = torch::empty_like(score_target);
  safe_score_targets_launch(pose.data_ptr<float>(), score_target.data_ptr<float>(),
      robot_radius.data_ptr<float>(), x_min.data_ptr<float>(), x_max.data_ptr<float>(),
      boxes.data_ptr<float>(), grid_points.data_ptr<float>(), grid_clear.data_ptr<bool>(),
      output.data_ptr<float>(), static_cast<int>(n), static_cast<int>(boxes.size(0)),
      static_cast<int>(grid_count), static_cast<float>(field_width),
      c10::cuda::getCurrentCUDAStream(pose.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor avoid_robot_contention(torch::Tensor pose, torch::Tensor command,
    torch::Tensor targets, torch::Tensor length, torch::Tensor width,
    torch::Tensor speed, torch::Tensor controlled, torch::Tensor team_ids,
    torch::Tensor active, torch::Tensor winner, bool teammate_intent) {
  const auto n = pose.size(0);
  TORCH_CHECK(pose.sizes() == torch::IntArrayRef({n, 6, 3}) &&
              command.sizes() == torch::IntArrayRef({n, 6, 2}) &&
              targets.sizes() == command.sizes() &&
              length.sizes() == torch::IntArrayRef({n, 6}) &&
              width.sizes() == length.sizes() && speed.sizes() == length.sizes() &&
              controlled.sizes() == torch::IntArrayRef({6}) &&
              team_ids.sizes() == torch::IntArrayRef({6}) &&
              active.sizes() == torch::IntArrayRef({n}) &&
              winner.sizes() == torch::IntArrayRef({n, 15}),
              "six-robot avoidance tensor shape mismatch");
  for (const auto& t : {pose, command, targets, length, width, speed,
                        controlled, team_ids, active, winner})
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.device() == pose.device(),
                "avoidance tensors must be contiguous on the same HIP device");
  auto adjusted = torch::empty_like(command);
  avoidance_launch(pose.data_ptr<float>(), command.data_ptr<float>(),
      targets.data_ptr<float>(), length.data_ptr<float>(), width.data_ptr<float>(),
      speed.data_ptr<float>(), controlled.data_ptr<bool>(),
      team_ids.data_ptr<int64_t>(), active.data_ptr<bool>(),
      winner.data_ptr<int64_t>(), adjusted.data_ptr<float>(),
      static_cast<int>(n), teammate_intent,
      c10::cuda::getCurrentCUDAStream(pose.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return adjusted;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("robot_contacts", &robot_contacts, "Fused six-robot SAT contacts (HIP)");
  m.def("field_contacts", &field_contacts, "Fused six-robot field SAT contacts (HIP)");
  m.def("field_sweep_contacts", &field_sweep_contacts,
        "Fused six-robot field sweep and SAT contacts (HIP)");
  m.def("safe_score_targets", &safe_score_targets,
        "Fused six-robot safe score target search (HIP)");
  m.def("avoid_robot_contention", &avoid_robot_contention,
        "Fused six-robot motion contention adjustment (HIP)");
}
