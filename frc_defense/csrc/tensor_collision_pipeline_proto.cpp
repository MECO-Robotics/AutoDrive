// Experimental whole-world fused contact-iteration prototype. Kept separate
// from the production wall-only path pending full parity and throughput proof.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime_api.h>

void contact_pipeline_launch(float*, float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const bool*, bool*, bool*, bool*, bool*,
    const float*, int, const float*, int, int, float, float, int, int, hipStream_t);

void contact_pipeline(torch::Tensor pose, torch::Tensor velocity,
    torch::Tensor length, torch::Tensor width, torch::Tensor mass,
    torch::Tensor yaw_inertia_multiplier, torch::Tensor mu, torch::Tensor wall_mu,
    torch::Tensor active, torch::Tensor robot_contact,
    torch::Tensor opponent_contact, torch::Tensor field_contact,
    torch::Tensor wall_contact, torch::Tensor obstacles,
    torch::Tensor field_colliders, int64_t iterations,
    double field_length, double field_width, int64_t stage) {
  TORCH_CHECK(pose.is_cuda() && pose.scalar_type() == at::kFloat && pose.dim() == 3 &&
              pose.size(1) == 2 && pose.size(2) == 3, "pose must be CUDA float [world,2,3]");
  const auto n = pose.size(0);
  TORCH_CHECK(velocity.sizes() == pose.sizes() && velocity.scalar_type() == at::kFloat,
              "velocity must match pose");
  TORCH_CHECK(length.sizes() == torch::IntArrayRef({n, 2}) &&
              width.sizes() == length.sizes() && mass.sizes() == length.sizes() &&
              yaw_inertia_multiplier.sizes() == length.sizes() &&
              mu.sizes() == torch::IntArrayRef({n}) && wall_mu.sizes() == mu.sizes(),
              "robot property shape mismatch");
  TORCH_CHECK(active.sizes() == mu.sizes() && active.scalar_type() == at::kBool &&
              robot_contact.sizes() == mu.sizes() && robot_contact.scalar_type() == at::kBool &&
              opponent_contact.sizes() == mu.sizes() && opponent_contact.scalar_type() == at::kBool &&
              field_contact.sizes() == torch::IntArrayRef({n, 2}) && field_contact.scalar_type() == at::kBool &&
              wall_contact.sizes() == torch::IntArrayRef({n, 2, 2}) && wall_contact.scalar_type() == at::kBool,
              "contact mask shape or dtype mismatch");
  TORCH_CHECK(obstacles.dim() == 2 && obstacles.size(1) == 3 && obstacles.scalar_type() == at::kFloat &&
              field_colliders.dim() == 2 && field_colliders.size(1) == 4 &&
              field_colliders.scalar_type() == at::kFloat, "collider shape or dtype mismatch");
  TORCH_CHECK(iterations >= 1, "iterations must be positive");
  TORCH_CHECK(stage >= -1 && stage <= 3, "stage must be -1 (all) or 0..3");
  for (const auto& t : {pose, velocity, length, width, mass, yaw_inertia_multiplier,
                        mu, wall_mu, active, robot_contact, opponent_contact,
                        field_contact, wall_contact, obstacles, field_colliders}) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.device() == pose.device(),
                "contact tensors must be contiguous on the same accelerator");
  }
  c10::cuda::CUDAGuard guard(pose.device());
  contact_pipeline_launch(pose.data_ptr<float>(), velocity.data_ptr<float>(),
      length.data_ptr<float>(), width.data_ptr<float>(), mass.data_ptr<float>(),
      yaw_inertia_multiplier.data_ptr<float>(), mu.data_ptr<float>(), wall_mu.data_ptr<float>(),
      active.data_ptr<bool>(), robot_contact.data_ptr<bool>(), opponent_contact.data_ptr<bool>(),
      field_contact.data_ptr<bool>(), wall_contact.data_ptr<bool>(), obstacles.data_ptr<float>(),
      static_cast<int>(obstacles.size(0)), field_colliders.data_ptr<float>(),
      static_cast<int>(field_colliders.size(0)), static_cast<int>(n),
      static_cast<float>(field_length), static_cast<float>(field_width),
      static_cast<int>(iterations), static_cast<int>(stage),
      c10::cuda::getCurrentCUDAStream(pose.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("contact_pipeline", &contact_pipeline,
        "Experimental fused ordered robot/wall/obstacle/field contacts (HIP)");
}
