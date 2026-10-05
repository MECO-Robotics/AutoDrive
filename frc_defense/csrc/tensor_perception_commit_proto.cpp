#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime_api.h>

void perception_commit_launch(
    const bool* active, const bool* visible, const float* measured,
    const float* measured_velocity, const bool* opponent_detected,
    const float* opponent_pose, const float* opponent_velocity,
    float* track_pos, float* track_vel, bool* track_mask, float* track_age,
    float* opponent_track_pose, float* opponent_track_velocity,
    bool* opponent_track_valid, float* opponent_track_age,
    int worlds, int pieces, float dt, float timeout, hipStream_t stream);
void commit(torch::Tensor active, torch::Tensor visible, torch::Tensor measured,
            torch::Tensor measured_velocity, torch::Tensor opponent_detected,
            torch::Tensor opponent_pose, torch::Tensor opponent_velocity,
            torch::Tensor track_pos, torch::Tensor track_vel,
            torch::Tensor track_mask, torch::Tensor track_age,
            torch::Tensor opponent_track_pose,
            torch::Tensor opponent_track_velocity,
            torch::Tensor opponent_track_valid,
            torch::Tensor opponent_track_age, double dt, double timeout) {
  TORCH_CHECK(active.is_cuda() && visible.is_cuda() && measured.is_cuda() &&
              measured_velocity.is_cuda() && opponent_detected.is_cuda() &&
              opponent_pose.is_cuda() && opponent_velocity.is_cuda() &&
              track_pos.is_cuda() && track_vel.is_cuda() && track_mask.is_cuda() &&
              track_age.is_cuda() && opponent_track_pose.is_cuda() &&
              opponent_track_velocity.is_cuda() && opponent_track_valid.is_cuda() &&
              opponent_track_age.is_cuda(), "all inputs must be device tensors");
  TORCH_CHECK(active.scalar_type() == at::kBool && visible.scalar_type() == at::kBool &&
              opponent_detected.scalar_type() == at::kBool &&
              track_mask.scalar_type() == at::kBool &&
              opponent_track_valid.scalar_type() == at::kBool,
              "mask tensors must be bool");
  TORCH_CHECK(measured.scalar_type() == at::kFloat &&
              measured_velocity.scalar_type() == at::kFloat &&
              opponent_pose.scalar_type() == at::kFloat &&
              opponent_velocity.scalar_type() == at::kFloat &&
              track_pos.scalar_type() == at::kFloat && track_vel.scalar_type() == at::kFloat &&
              track_age.scalar_type() == at::kFloat &&
              opponent_track_pose.scalar_type() == at::kFloat &&
              opponent_track_velocity.scalar_type() == at::kFloat &&
              opponent_track_age.scalar_type() == at::kFloat,
              "state tensors must be float32");
  TORCH_CHECK(active.dim() == 1 && visible.dim() == 3 &&
              visible.size(0) == active.size(0) && visible.size(1) == 2,
              "active/visible shape mismatch");
  const int worlds = static_cast<int>(active.size(0));
  const int pieces = static_cast<int>(visible.size(2));
  TORCH_CHECK(measured.sizes() == torch::IntArrayRef({worlds, 2, pieces, 2}) &&
              measured_velocity.sizes() == measured.sizes() &&
              track_pos.sizes() == measured.sizes() && track_vel.sizes() == measured.sizes() &&
              track_mask.sizes() == visible.sizes() && track_age.sizes() == visible.sizes(),
              "fuel track shape mismatch");
  TORCH_CHECK(opponent_detected.sizes() == torch::IntArrayRef({worlds, 2}) &&
              opponent_pose.sizes() == torch::IntArrayRef({worlds, 2, 3}) &&
              opponent_velocity.sizes() == opponent_pose.sizes() &&
              opponent_track_pose.sizes() == opponent_pose.sizes() &&
              opponent_track_velocity.sizes() == opponent_pose.sizes() &&
              opponent_track_valid.sizes() == opponent_detected.sizes() &&
              opponent_track_age.sizes() == opponent_detected.sizes(),
              "opponent track shape mismatch");
  TORCH_CHECK(active.device() == visible.device() && active.device() == measured.device() &&
              active.device() == track_pos.device() && active.device() == opponent_pose.device(),
              "all inputs must share a device");
  TORCH_CHECK(active.is_contiguous() && visible.is_contiguous() && measured.is_contiguous() &&
              measured_velocity.is_contiguous() && opponent_detected.is_contiguous() &&
              opponent_pose.is_contiguous() && opponent_velocity.is_contiguous() &&
              track_pos.is_contiguous() && track_vel.is_contiguous() &&
              track_mask.is_contiguous() && track_age.is_contiguous() &&
              opponent_track_pose.is_contiguous() && opponent_track_velocity.is_contiguous() &&
              opponent_track_valid.is_contiguous() && opponent_track_age.is_contiguous(),
              "all inputs must be contiguous");
  c10::cuda::CUDAGuard guard(active.device());
  perception_commit_launch(active.data_ptr<bool>(), visible.data_ptr<bool>(),
      measured.data_ptr<float>(), measured_velocity.data_ptr<float>(),
      opponent_detected.data_ptr<bool>(), opponent_pose.data_ptr<float>(),
      opponent_velocity.data_ptr<float>(), track_pos.data_ptr<float>(),
      track_vel.data_ptr<float>(), track_mask.data_ptr<bool>(), track_age.data_ptr<float>(),
      opponent_track_pose.data_ptr<float>(), opponent_track_velocity.data_ptr<float>(),
      opponent_track_valid.data_ptr<bool>(), opponent_track_age.data_ptr<float>(),
      worlds, pieces, static_cast<float>(dt), static_cast<float>(timeout),
      c10::cuda::getCurrentCUDAStream(active.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("commit", &commit, "Test-only fused perception track-state commit (HIP)");
}
