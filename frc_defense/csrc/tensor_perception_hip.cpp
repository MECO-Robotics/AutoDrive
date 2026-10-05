#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime_api.h>
#include <cmath>
#include <cstdint>

void piece_occlusion_launch(const float* pose_xy, const float* segment,
                            const float* obstacles, const bool* eligible,
                            bool* output,
                            int worlds, int pieces, int obstacle_count,
                            hipStream_t stream);
void visibility_launch(const float* pose, const float* pieces_xy,
                       const bool* piece_active, const int64_t* piece_owner,
                       const bool* active, const float* obstacles,
                       const float* other_xy, const float* other_radius,
                       bool* output, int worlds, int pieces, int obstacle_count,
                       float range_m, float half_fov, bool full_fov,
                       hipStream_t stream);
void visibility_3v3_launch(const float* pose, const float* pieces_xy,
                           const bool* piece_active, const int64_t* piece_owner,
                           const bool* active, const float* obstacles,
                           const float* robot_radius, bool* output, int worlds,
                           int robots, int pieces, int obstacle_count, float range_m,
                           float half_fov, bool full_fov, hipStream_t stream);
void perception_commit_3v3_launch(const bool* visible, const bool* active,
                                 const int64_t* ticks, const float* piece_pos,
                                 const float* piece_vel, float* track_pos,
                                 float* track_vel, float* track_age,
                                 bool* track_mask, uint32_t seed, int worlds,
                                 int robots, int pieces, float dt, float dropout,
                                 float position_noise, float velocity_noise,
                                 hipStream_t stream);
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
                                float velocity_noise, float dt, hipStream_t stream);

torch::Tensor piece_occlusion(torch::Tensor pose_xy, torch::Tensor segment,
                              torch::Tensor obstacles, torch::Tensor eligible) {
  TORCH_CHECK(pose_xy.is_cuda() && segment.is_cuda() && obstacles.is_cuda() &&
              eligible.is_cuda(),
              "piece occlusion requires device tensors");
  TORCH_CHECK(pose_xy.scalar_type() == at::kFloat &&
              segment.scalar_type() == at::kFloat &&
              obstacles.scalar_type() == at::kFloat,
              "piece occlusion expects float32 inputs");
  TORCH_CHECK(pose_xy.dim() == 2 && pose_xy.size(1) == 2,
              "pose_xy must have shape [world,2]");
  TORCH_CHECK(segment.dim() == 3 && segment.size(0) == pose_xy.size(0) &&
              segment.size(2) == 2,
              "segment must have shape [world,piece,2]");
  TORCH_CHECK(obstacles.dim() == 2 && obstacles.size(1) == 3,
              "obstacles must have shape [circle,3]");
  TORCH_CHECK(eligible.scalar_type() == at::kBool && eligible.dim() == 2 &&
              eligible.size(0) == segment.size(0) &&
              eligible.size(1) == segment.size(1),
              "eligible must be bool with shape [world,piece]");
  TORCH_CHECK(pose_xy.is_contiguous() && segment.is_contiguous() &&
              obstacles.is_contiguous() && eligible.is_contiguous(),
              "piece occlusion inputs must be contiguous");
  c10::cuda::CUDAGuard guard(pose_xy.device());
  auto output = torch::empty({segment.size(0), segment.size(1)},
                             pose_xy.options().dtype(torch::kBool));
  piece_occlusion_launch(pose_xy.data_ptr<float>(), segment.data_ptr<float>(),
                         obstacles.data_ptr<float>(), eligible.data_ptr<bool>(),
                         output.data_ptr<bool>(),
                         static_cast<int>(segment.size(0)),
                         static_cast<int>(segment.size(1)),
                         static_cast<int>(obstacles.size(0)),
                         c10::cuda::getCurrentCUDAStream(pose_xy.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor visibility_mask(torch::Tensor pose, torch::Tensor pieces_xy,
                              torch::Tensor piece_active,
                              torch::Tensor piece_owner, torch::Tensor active,
                              double perception_range, double fov_degrees,
                              torch::Tensor obstacles, torch::Tensor other_xy,
                              torch::Tensor other_radius) {
  TORCH_CHECK(pose.is_cuda() && pieces_xy.is_cuda() && piece_active.is_cuda() &&
              piece_owner.is_cuda() && active.is_cuda() && obstacles.is_cuda() &&
              other_xy.is_cuda() && other_radius.is_cuda(),
              "visibility inputs must be device tensors");
  TORCH_CHECK(pose.scalar_type() == at::kFloat && pieces_xy.scalar_type() == at::kFloat &&
              piece_active.scalar_type() == at::kBool && piece_owner.scalar_type() == at::kLong &&
              active.scalar_type() == at::kBool && obstacles.scalar_type() == at::kFloat &&
              other_xy.scalar_type() == at::kFloat && other_radius.scalar_type() == at::kFloat,
              "visibility input dtype mismatch");
  TORCH_CHECK(pose.dim() == 2 && pose.size(1) == 3, "pose must have shape [world,3]");
  TORCH_CHECK(pieces_xy.dim() == 3 && pieces_xy.size(0) == pose.size(0) &&
              pieces_xy.size(2) == 2, "pieces_xy must have shape [world,piece,2]");
  TORCH_CHECK(piece_active.sizes() == piece_owner.sizes() && piece_active.dim() == 2 &&
              piece_active.size(0) == pose.size(0) && piece_active.size(1) == pieces_xy.size(1),
              "piece state must have shape [world,piece]");
  TORCH_CHECK(active.dim() == 1 && active.size(0) == pose.size(0) &&
              obstacles.dim() == 2 && obstacles.size(1) == 3 &&
              other_xy.dim() == 2 && other_xy.size(0) == pose.size(0) && other_xy.size(1) == 2 &&
              other_radius.dim() == 1 && other_radius.size(0) == pose.size(0),
              "visibility input shape mismatch");
  TORCH_CHECK(pose.device() == pieces_xy.device() && pose.device() == piece_active.device() &&
              pose.device() == piece_owner.device() && pose.device() == active.device() &&
              pose.device() == obstacles.device() && pose.device() == other_xy.device() &&
              pose.device() == other_radius.device(), "visibility inputs must share a device");
  TORCH_CHECK(pose.is_contiguous() && pieces_xy.is_contiguous() && piece_active.is_contiguous() &&
              piece_owner.is_contiguous() && active.is_contiguous() && obstacles.is_contiguous() &&
              other_xy.is_contiguous() && other_radius.is_contiguous(),
              "visibility inputs must be contiguous");
  c10::cuda::CUDAGuard guard(pose.device());
  auto output = torch::empty({pose.size(0), pieces_xy.size(1)},
                             pose.options().dtype(torch::kBool));
  visibility_launch(pose.data_ptr<float>(), pieces_xy.data_ptr<float>(),
                    piece_active.data_ptr<bool>(), piece_owner.data_ptr<int64_t>(),
                    active.data_ptr<bool>(), obstacles.data_ptr<float>(),
                    other_xy.data_ptr<float>(), other_radius.data_ptr<float>(),
                    output.data_ptr<bool>(), static_cast<int>(pose.size(0)),
                    static_cast<int>(pieces_xy.size(1)),
                    static_cast<int>(obstacles.size(0)),
                    static_cast<float>(perception_range),
                    static_cast<float>((fov_degrees * M_PI / 180.0) * 0.5),
                    fov_degrees >= 360.0,
                    c10::cuda::getCurrentCUDAStream(pose.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor visibility_mask_3v3(torch::Tensor pose, torch::Tensor pieces_xy,
                                  torch::Tensor piece_active,
                                  torch::Tensor piece_owner, torch::Tensor active,
                                  double perception_range, double fov_degrees,
                                  torch::Tensor obstacles,
                                  torch::Tensor robot_radius) {
  TORCH_CHECK(pose.is_cuda() && pieces_xy.is_cuda() && piece_active.is_cuda() &&
              piece_owner.is_cuda() && active.is_cuda() && obstacles.is_cuda() &&
              robot_radius.is_cuda(), "3v3 visibility inputs must be device tensors");
  TORCH_CHECK(pose.scalar_type() == at::kFloat && pieces_xy.scalar_type() == at::kFloat &&
              piece_active.scalar_type() == at::kBool && piece_owner.scalar_type() == at::kLong &&
              active.scalar_type() == at::kBool && obstacles.scalar_type() == at::kFloat &&
              robot_radius.scalar_type() == at::kFloat,
              "3v3 visibility input dtype mismatch");
  TORCH_CHECK(pose.dim() == 3 && pose.size(2) == 3,
              "pose must have shape [world,robot,3]");
  const auto worlds = pose.size(0);
  const auto robots = pose.size(1);
  TORCH_CHECK(pieces_xy.dim() == 3 && pieces_xy.size(0) == worlds &&
              pieces_xy.size(2) == 2, "pieces_xy must have shape [world,piece,2]");
  TORCH_CHECK(piece_active.dim() == 2 && piece_active.size(0) == worlds &&
              piece_active.size(1) == pieces_xy.size(1) &&
              piece_owner.sizes() == piece_active.sizes(),
              "piece state must have shape [world,piece]");
  TORCH_CHECK(active.dim() == 1 && active.size(0) == worlds &&
              obstacles.dim() == 2 && obstacles.size(1) == 3 &&
              robot_radius.sizes() == torch::IntArrayRef({worlds, robots}),
              "3v3 visibility input shape mismatch");
  TORCH_CHECK(pose.device() == pieces_xy.device() && pose.device() == piece_active.device() &&
              pose.device() == piece_owner.device() && pose.device() == active.device() &&
              pose.device() == obstacles.device() && pose.device() == robot_radius.device(),
              "3v3 visibility inputs must share a device");
  TORCH_CHECK(pose.is_contiguous() && pieces_xy.is_contiguous() &&
              piece_active.is_contiguous() && piece_owner.is_contiguous() &&
              active.is_contiguous() && obstacles.is_contiguous() && robot_radius.is_contiguous(),
              "3v3 visibility inputs must be contiguous");
  c10::cuda::CUDAGuard guard(pose.device());
  auto output = torch::empty({worlds, robots, pieces_xy.size(1)},
                             pose.options().dtype(torch::kBool));
  visibility_3v3_launch(pose.data_ptr<float>(), pieces_xy.data_ptr<float>(),
      piece_active.data_ptr<bool>(), piece_owner.data_ptr<int64_t>(),
      active.data_ptr<bool>(), obstacles.data_ptr<float>(), robot_radius.data_ptr<float>(),
      output.data_ptr<bool>(), static_cast<int>(worlds), static_cast<int>(robots),
      static_cast<int>(pieces_xy.size(1)), static_cast<int>(obstacles.size(0)),
      static_cast<float>(perception_range),
      static_cast<float>((fov_degrees * M_PI / 180.0) * 0.5),
      fov_degrees >= 360.0,
      c10::cuda::getCurrentCUDAStream(pose.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

void perception_commit_3v3(torch::Tensor visible, torch::Tensor active,
                           torch::Tensor ticks, torch::Tensor piece_pos,
                           torch::Tensor piece_vel, torch::Tensor track_pos,
                           torch::Tensor track_vel, torch::Tensor track_age,
                           torch::Tensor track_mask, int64_t seed, double dt,
                           double dropout, double position_noise,
                           double velocity_noise) {
  TORCH_CHECK(visible.is_cuda() && active.is_cuda() && ticks.is_cuda() &&
              piece_pos.is_cuda() && piece_vel.is_cuda() && track_pos.is_cuda() &&
              track_vel.is_cuda() && track_age.is_cuda() && track_mask.is_cuda(),
              "perception commit inputs must be device tensors");
  TORCH_CHECK(visible.scalar_type() == at::kBool && active.scalar_type() == at::kBool &&
              ticks.scalar_type() == at::kLong && piece_pos.scalar_type() == at::kFloat &&
              piece_vel.scalar_type() == at::kFloat && track_pos.scalar_type() == at::kFloat &&
              track_vel.scalar_type() == at::kFloat && track_age.scalar_type() == at::kFloat &&
              track_mask.scalar_type() == at::kBool,
              "perception commit input dtype mismatch");
  TORCH_CHECK(visible.dim() == 3 && active.dim() == 1 &&
              visible.size(0) == active.size(0) && ticks.sizes() == active.sizes(),
              "perception commit mask/tick shape mismatch");
  const auto worlds = visible.size(0);
  const auto robots = visible.size(1);
  const auto pieces = visible.size(2);
  TORCH_CHECK(piece_pos.sizes() == torch::IntArrayRef({worlds, pieces, 2}) &&
              piece_vel.sizes() == piece_pos.sizes() &&
              track_pos.sizes() == torch::IntArrayRef({worlds, robots, pieces, 2}) &&
              track_vel.sizes() == track_pos.sizes() &&
              track_age.sizes() == visible.sizes() &&
              track_mask.sizes() == visible.sizes(),
              "perception commit state shape mismatch");
  TORCH_CHECK(visible.device() == active.device() && visible.device() == ticks.device() &&
              visible.device() == piece_pos.device() && visible.device() == piece_vel.device() &&
              visible.device() == track_pos.device() && visible.device() == track_vel.device() &&
              visible.device() == track_age.device() && visible.device() == track_mask.device(),
              "perception commit inputs must share a device");
  TORCH_CHECK(visible.is_contiguous() && active.is_contiguous() && ticks.is_contiguous() &&
              piece_pos.is_contiguous() && piece_vel.is_contiguous() && track_pos.is_contiguous() &&
              track_vel.is_contiguous() && track_age.is_contiguous() && track_mask.is_contiguous(),
              "perception commit inputs must be contiguous");
  c10::cuda::CUDAGuard guard(visible.device());
  perception_commit_3v3_launch(visible.data_ptr<bool>(), active.data_ptr<bool>(),
      ticks.data_ptr<int64_t>(), piece_pos.data_ptr<float>(), piece_vel.data_ptr<float>(),
      track_pos.data_ptr<float>(), track_vel.data_ptr<float>(), track_age.data_ptr<float>(),
      track_mask.data_ptr<bool>(), static_cast<uint32_t>(seed), static_cast<int>(worlds),
      static_cast<int>(robots), static_cast<int>(pieces), static_cast<float>(dt),
      static_cast<float>(dropout), static_cast<float>(position_noise),
      static_cast<float>(velocity_noise),
      c10::cuda::getCurrentCUDAStream(visible.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void opponent_tracks_3v3(torch::Tensor pose, torch::Tensor length,
                        torch::Tensor width, torch::Tensor acceleration,
                        torch::Tensor robot_radius, torch::Tensor obstacles,
                        torch::Tensor active, torch::Tensor controlled,
                        torch::Tensor defense_role, torch::Tensor ticks,
                        torch::Tensor opponent_pose,
                        torch::Tensor opponent_velocity,
                        torch::Tensor opponent_size, torch::Tensor opponent_age,
                        torch::Tensor opponent_valid, int64_t seed,
                        double range_m, double fov_degrees, double dropout,
                        double position_noise, double velocity_noise, double dt) {
  TORCH_CHECK(pose.is_cuda() && length.is_cuda() && width.is_cuda() &&
              acceleration.is_cuda() && robot_radius.is_cuda() && obstacles.is_cuda() &&
              active.is_cuda() && controlled.is_cuda() && defense_role.is_cuda() &&
              ticks.is_cuda() && opponent_pose.is_cuda() && opponent_velocity.is_cuda() &&
              opponent_size.is_cuda() && opponent_age.is_cuda() && opponent_valid.is_cuda(),
              "opponent track inputs must be device tensors");
  TORCH_CHECK(pose.scalar_type() == at::kFloat && length.scalar_type() == at::kFloat &&
              width.scalar_type() == at::kFloat && acceleration.scalar_type() == at::kFloat &&
              robot_radius.scalar_type() == at::kFloat && obstacles.scalar_type() == at::kFloat &&
              active.scalar_type() == at::kBool && controlled.scalar_type() == at::kBool &&
              defense_role.scalar_type() == at::kBool && ticks.scalar_type() == at::kLong &&
              opponent_pose.scalar_type() == at::kFloat &&
              opponent_velocity.scalar_type() == at::kFloat &&
              opponent_size.scalar_type() == at::kFloat && opponent_age.scalar_type() == at::kFloat &&
              opponent_valid.scalar_type() == at::kBool,
              "opponent track input dtype mismatch");
  TORCH_CHECK(pose.dim() == 3 && pose.size(1) == 6 && pose.size(2) == 3,
              "pose must have shape [world,6,3]");
  const auto worlds = pose.size(0);
  TORCH_CHECK(length.sizes() == torch::IntArrayRef({worlds, 6}) &&
              width.sizes() == length.sizes() && acceleration.sizes() == length.sizes() &&
              robot_radius.sizes() == length.sizes() && obstacles.dim() == 2 && obstacles.size(1) == 3 &&
              active.sizes() == torch::IntArrayRef({worlds}) && controlled.sizes() == torch::IntArrayRef({6}) &&
              defense_role.sizes() == controlled.sizes() && ticks.sizes() == active.sizes() &&
              opponent_pose.sizes() == pose.sizes() && opponent_velocity.sizes() == pose.sizes() &&
              opponent_size.sizes() == pose.sizes() && opponent_age.sizes() == torch::IntArrayRef({worlds, 6}) &&
              opponent_valid.sizes() == opponent_age.sizes(),
              "opponent track input shape mismatch");
  TORCH_CHECK(pose.device() == length.device() && pose.device() == width.device() &&
              pose.device() == acceleration.device() && pose.device() == robot_radius.device() &&
              pose.device() == obstacles.device() && pose.device() == active.device() &&
              pose.device() == controlled.device() && pose.device() == defense_role.device() &&
              pose.device() == ticks.device() && pose.device() == opponent_pose.device() &&
              pose.device() == opponent_velocity.device() && pose.device() == opponent_size.device() &&
              pose.device() == opponent_age.device() && pose.device() == opponent_valid.device(),
              "opponent track inputs must share a device");
  TORCH_CHECK(pose.is_contiguous() && length.is_contiguous() && width.is_contiguous() &&
              acceleration.is_contiguous() && robot_radius.is_contiguous() && obstacles.is_contiguous() &&
              active.is_contiguous() && controlled.is_contiguous() && defense_role.is_contiguous() &&
              ticks.is_contiguous() && opponent_pose.is_contiguous() && opponent_velocity.is_contiguous() &&
              opponent_size.is_contiguous() && opponent_age.is_contiguous() && opponent_valid.is_contiguous(),
              "opponent track inputs must be contiguous");
  c10::cuda::CUDAGuard guard(pose.device());
  opponent_tracks_3v3_launch(pose.data_ptr<float>(), length.data_ptr<float>(),
      width.data_ptr<float>(), acceleration.data_ptr<float>(), robot_radius.data_ptr<float>(),
      obstacles.data_ptr<float>(), active.data_ptr<bool>(), controlled.data_ptr<bool>(),
      defense_role.data_ptr<bool>(), ticks.data_ptr<int64_t>(), opponent_pose.data_ptr<float>(),
      opponent_velocity.data_ptr<float>(), opponent_size.data_ptr<float>(), opponent_age.data_ptr<float>(),
      opponent_valid.data_ptr<bool>(), static_cast<uint32_t>(seed), static_cast<int>(worlds),
      static_cast<int>(obstacles.size(0)), static_cast<float>(range_m),
      static_cast<float>((fov_degrees * M_PI / 180.0) * 0.5), static_cast<float>(dropout),
      static_cast<float>(position_noise), static_cast<float>(velocity_noise), static_cast<float>(dt),
      c10::cuda::getCurrentCUDAStream(pose.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("piece_occlusion", &piece_occlusion,
        "Fused eligible piece-vs-circle segment occlusion (HIP)");
  m.def("visibility_mask", &visibility_mask,
        "Fused deterministic FUEL visibility mask (HIP)");
  m.def("visibility_mask_3v3", &visibility_mask_3v3,
        "Fused six-robot FUEL visibility mask with chassis occlusion (HIP)");
  m.def("perception_commit_3v3", &perception_commit_3v3,
        "Fused counter-based sensor noise and six-robot track update (HIP)");
  m.def("opponent_tracks_3v3", &opponent_tracks_3v3,
        "Fused opponent visibility and six-robot sensor track update (HIP)");
}
