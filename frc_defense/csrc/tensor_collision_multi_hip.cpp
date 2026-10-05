#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

void robot_contacts_launch(float*, float*, const float*, const float*,
    const float*, const float*, const float*, const bool*, const int64_t*,
    bool*, bool*, int, hipStream_t);

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("robot_contacts", &robot_contacts, "Fused six-robot SAT contacts (HIP)");
}
