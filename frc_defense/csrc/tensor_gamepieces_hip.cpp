#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime_api.h>

void gamepieces_launch(const bool*, const float*, const float*, const float*, const float*,
    float*, float*, bool*, int64_t*, int64_t*, float*, float*, bool*,
    const int64_t*, int64_t*, float*, float*, bool*, const int64_t*, int64_t*,
    int64_t*, int64_t*, int64_t*, int64_t*, int64_t*, int64_t*, int64_t*,
    const float*, const float*, bool*, float*, int, int, float, float, float,
    int, bool, float, float, hipStream_t);

void update(torch::Tensor active, torch::Tensor pose, torch::Tensor velocity,
    torch::Tensor length, torch::Tensor width, torch::Tensor piece_pos,
    torch::Tensor piece_vel, torch::Tensor piece_active, torch::Tensor owner,
    torch::Tensor zone, torch::Tensor elapsed, torch::Tensor remaining,
    torch::Tensor hub_active, torch::Tensor hub_first, torch::Tensor auto_scores,
    torch::Tensor next_intake, torch::Tensor next_score, torch::Tensor last_hub,
    torch::Tensor last_action, torch::Tensor acquisition_count,
    torch::Tensor score_count, torch::Tensor denied_count,
    torch::Tensor abandoned_count, torch::Tensor acquired_event,
    torch::Tensor scored_event, torch::Tensor denied_event,
    torch::Tensor abandoned_event, torch::Tensor hub_centers,
    torch::Tensor respawn_positions, torch::Tensor track_mask,
    torch::Tensor track_age,
    double dt, double intake_interval, double score_interval, int64_t capacity,
    bool strategic, int64_t pieces, double field_length,
    double alliance_zone_depth) {
  TORCH_CHECK(active.is_cuda() && active.scalar_type() == at::kBool && active.dim() == 1,
              "active mask must be CUDA bool [world]");
  const auto n = active.size(0);
  TORCH_CHECK(pose.sizes() == torch::IntArrayRef({n, 2, 3}) &&
              velocity.sizes() == torch::IntArrayRef({n, 2, 3}), "robot state shape mismatch");
  TORCH_CHECK(piece_pos.sizes() == torch::IntArrayRef({n, pieces, 2}) &&
              piece_vel.sizes() == piece_pos.sizes() &&
              piece_active.sizes() == torch::IntArrayRef({n, pieces}) &&
              owner.sizes() == piece_active.sizes() && zone.sizes() == piece_active.sizes(),
              "piece state shape mismatch");
  TORCH_CHECK(respawn_positions.sizes() == torch::IntArrayRef({pieces, 2}) &&
              track_mask.sizes() == torch::IntArrayRef({n, 2, pieces}) &&
              track_age.sizes() == track_mask.sizes(),
              "respawn or perception track shape mismatch");
  TORCH_CHECK(pose.scalar_type() == at::kFloat && owner.scalar_type() == at::kLong &&
              zone.scalar_type() == at::kLong && piece_active.scalar_type() == at::kBool,
              "gamepiece state dtype mismatch");
  for (const auto& t : {pose, velocity, length, width, piece_pos, piece_vel, piece_active,
       owner, zone, elapsed, remaining, hub_active, hub_first, auto_scores, next_intake,
       next_score, last_hub, last_action, acquisition_count, score_count, denied_count,
       abandoned_count, acquired_event, scored_event, denied_event, abandoned_event,
       hub_centers, respawn_positions, track_mask, track_age}) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.device() == pose.device(),
                "gamepiece tensors must be contiguous on one CUDA device");
  }
  c10::cuda::CUDAGuard guard(pose.device());
  gamepieces_launch(active.data_ptr<bool>(), pose.data_ptr<float>(), velocity.data_ptr<float>(),
      length.data_ptr<float>(), width.data_ptr<float>(), piece_pos.data_ptr<float>(),
      piece_vel.data_ptr<float>(), piece_active.data_ptr<bool>(), owner.data_ptr<int64_t>(),
      zone.data_ptr<int64_t>(), elapsed.data_ptr<float>(), remaining.data_ptr<float>(),
      hub_active.data_ptr<bool>(), hub_first.data_ptr<int64_t>(), auto_scores.data_ptr<int64_t>(),
      next_intake.data_ptr<float>(), next_score.data_ptr<float>(), last_hub.data_ptr<bool>(),
      last_action.data_ptr<int64_t>(), acquisition_count.data_ptr<int64_t>(),
      score_count.data_ptr<int64_t>(), denied_count.data_ptr<int64_t>(),
      abandoned_count.data_ptr<int64_t>(), acquired_event.data_ptr<int64_t>(),
      scored_event.data_ptr<int64_t>(), denied_event.data_ptr<int64_t>(),
      abandoned_event.data_ptr<int64_t>(), hub_centers.data_ptr<float>(),
      respawn_positions.data_ptr<float>(), track_mask.data_ptr<bool>(),
      track_age.data_ptr<float>(), static_cast<int>(n),
      static_cast<int>(pieces), static_cast<float>(dt), static_cast<float>(intake_interval),
      static_cast<float>(score_interval), static_cast<int>(capacity), strategic,
      static_cast<float>(field_length), static_cast<float>(alliance_zone_depth),
      c10::cuda::getCurrentCUDAStream(pose.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("update", &update, "Fused FUEL pickup, following, and scoring update (HIP)");
}
