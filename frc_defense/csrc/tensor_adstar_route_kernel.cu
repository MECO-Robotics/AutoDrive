#include <hip/hip_runtime.h>
#include <math.h>

namespace {
constexpr float kInfinity = 1.0e6f;
constexpr float kSqrt2 = 1.41421356237f;
constexpr int kMaxPoints = 72;

__device__ __forceinline__ bool cell_blocked(const bool* blocked, int b,
                                              int x, int y, int nx, int ny,
                                              int start_x, int start_y,
                                              int goal_x, int goal_y) {
  return x < 0 || x >= nx || y < 0 || y >= ny ||
         ((x != start_x || y != start_y) &&
          (x != goal_x || y != goal_y) &&
          blocked[(static_cast<int64_t>(b) * nx + x) * ny + y]);
}

__device__ __forceinline__ bool line_of_sight_clear(
    const bool* blocked, int b, float x0, float y0, float x1, float y1,
    int nx, int ny, float resolution, int start_x, int start_y,
    int goal_x, int goal_y) {
  const float dx=x1-x0, dy=y1-y0;
  int steps=static_cast<int>(ceilf(
      fmaxf(fabsf(dx),fabsf(dy))/(resolution*.5f)));
  if (steps < 1) steps=1;
  for (int step=1; step<steps; ++step) {
    const float t=static_cast<float>(step)/steps;
    const int x=static_cast<int>(floorf((x0+dx*t)/resolution));
    const int y=static_cast<int>(floorf((y0+dy*t)/resolution));
    if (cell_blocked(blocked,b,x,y,nx,ny,start_x,start_y,goal_x,goal_y))
      return false;
  }
  return true;
}

__global__ void fused_bellman_route_kernel(
    const float* value_in, const bool* blocked, const float* bump,
    const float* start_xy, const float* goal_xy,
    const int64_t* start_x, const int64_t* start_y,
    const int64_t* goal_x, const int64_t* goal_y,
    const float* speed, const float* friction, const float* acceleration,
    float* potential, float* path, int64_t* lengths, float* profile,
    float* resolved_goal,
    int32_t* active_checkpoints, float steer_rate_limit,
    int nx, int ny, int cells, int sweeps, bool stop_when_converged,
    bool project_blocked_goal, float resolution) {
  const int b = blockIdx.x;
  const int lane = threadIdx.x;
  extern __shared__ float shared_values[];
  float* current = shared_values;
  float* next = shared_values + cells;
  float* path_x = shared_values + 2 * cells;
  float* path_y = path_x + kMaxPoints;
  float* distance = path_y + kMaxPoints;
  float* station = distance + kMaxPoints;
  float* caps = station + kMaxPoints;
  __shared__ int changed_threads;
  __shared__ int route_length;
  __shared__ int reached_x;
  __shared__ int reached_y;
  __shared__ int checkpoint_mask;
  __shared__ int source_x;
  __shared__ int source_y;
  __shared__ int target_x;
  __shared__ int target_y;
  __shared__ float start_x_metric;
  __shared__ float start_y_metric;
  const int64_t base = static_cast<int64_t>(b) * cells;

  if (lane == 0) {
    source_x = static_cast<int>(start_x[b]);
    source_y = static_cast<int>(start_y[b]);
    target_x = static_cast<int>(goal_x[b]);
    target_y = static_cast<int>(goal_y[b]);
    const int raw_goal = target_x * ny + target_y;
    if (project_blocked_goal && blocked[base + raw_goal]) {
      const float gx = goal_xy[static_cast<int64_t>(b) * 2];
      const float gy = goal_xy[static_cast<int64_t>(b) * 2 + 1];
      float best_distance = INFINITY;
      int best_cell = 0;
      for (int cell = 0; cell < cells; ++cell) {
        if (!blocked[base + cell]) {
          const int x = cell / ny;
          const int y = cell - x * ny;
          const float dx = (static_cast<float>(x) + 0.5f) * resolution - gx;
          const float dy = (static_cast<float>(y) + 0.5f) * resolution - gy;
          const float distance = __fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy));
          if (distance < best_distance) {
            best_distance = distance;
            best_cell = cell;
          }
        }
      }
      target_x = best_cell / ny;
      target_y = best_cell - target_x * ny;
      resolved_goal[static_cast<int64_t>(b) * 2] =
          (static_cast<float>(target_x) + 0.5f) * resolution;
      resolved_goal[static_cast<int64_t>(b) * 2 + 1] =
          (static_cast<float>(target_y) + 0.5f) * resolution;
    } else {
      resolved_goal[static_cast<int64_t>(b) * 2] =
          goal_xy[static_cast<int64_t>(b) * 2];
      resolved_goal[static_cast<int64_t>(b) * 2 + 1] =
          goal_xy[static_cast<int64_t>(b) * 2 + 1];
    }
  }
  __syncthreads();

  const int target_cell = target_x * ny + target_y;
  const int source_cell = source_x * ny + source_y;
  for (int cell = lane; cell < cells; cell += blockDim.x) {
    float initial = value_in[base + cell];
    if (cell == target_cell) initial = 0.f;
    else if (cell == source_cell || blocked[base + cell]) initial = kInfinity;
    current[cell] = initial;
  }
  __syncthreads();

  for (int sweep = 0; sweep < sweeps; ++sweep) {
    int thread_changed = 0;
    for (int cell = lane; cell < cells; cell += blockDim.x) {
      const int x = cell / ny;
      const int y = cell - x * ny;
      const int64_t linear = base + cell;
      if (blocked[linear] && cell != source_cell && cell != target_cell) {
        next[cell] = kInfinity;
        thread_changed |= current[cell] != kInfinity;
        continue;
      }
      float best = current[cell];
      const float edge = bump[cell];
      #pragma unroll
      for (int dx = -1; dx <= 1; ++dx) {
        #pragma unroll
        for (int dy = -1; dy <= 1; ++dy) {
          const int xx = x + dx;
          const int yy = y + dy;
          float neighbor = kInfinity;
          if (xx >= 0 && xx < nx && yy >= 0 && yy < ny)
            neighbor = current[xx * ny + yy];
          if (dx != 0 && dy != 0 &&
              (cell_blocked(blocked, b, x + dx, y, nx, ny,
                            source_x, source_y, target_x, target_y) ||
               cell_blocked(blocked, b, x, y + dy, nx, ny,
                            source_x, source_y, target_x, target_y)))
            continue;
          const float edge_distance = (dx != 0 && dy != 0) ? kSqrt2 : 1.0f;
          const float candidate = __fadd_rn(neighbor, __fmul_rn(edge_distance, edge));
          best = fminf(best, candidate);
        }
      }
      next[cell] = best;
      thread_changed |= best < current[cell];
    }
    if (stop_when_converged) {
      const int changed_count = __syncthreads_count(thread_changed != 0);
      if (changed_count == 0) break;
    } else {
      __syncthreads();
    }
    float* swap = current;
    current = next;
    next = swap;
  }
  for (int cell = lane; cell < cells; cell += blockDim.x)
    potential[base + cell] = current[cell];

  if (lane == 0) {
    start_x_metric = start_xy[static_cast<int64_t>(b) * 2];
    start_y_metric = start_xy[static_cast<int64_t>(b) * 2 + 1];
    int px = source_x;
    int py = source_y;
    const int gx = target_x;
    const int gy = target_y;
    path_x[0] = start_x_metric;
    path_y[0] = start_y_metric;
    int length = 1;
    int local_checkpoint_mask = 0;
    bool active = true;
    #pragma unroll
    for (int k = 1; k < kMaxPoints; ++k) {
      if (active) {
        float scores[8];
        int nx_choice[8];
        int ny_choice[8];
        const int dxs[8] = {1, -1, 0, 0, 1, 1, -1, -1};
        const int dys[8] = {0, 0, 1, -1, 1, -1, 1, -1};
        float best = kInfinity;
        int choice = 0;
        for (int q = 0; q < 8; ++q) {
          const int qx = px + dxs[q];
          const int qy = py + dys[q];
          nx_choice[q] = qx;
          ny_choice[q] = qy;
          bool valid = qx >= 0 && qx < nx && qy >= 0 && qy < ny;
          if (valid && dxs[q] != 0 && dys[q] != 0)
            valid = !cell_blocked(blocked, b, qx, py, nx, ny,
                                  source_x, source_y, gx, gy) &&
                    !cell_blocked(blocked, b, px, qy, nx, ny,
                                  source_x, source_y, gx, gy);
          if (valid && cell_blocked(blocked, b, qx, qy, nx, ny,
                                    source_x, source_y, gx, gy)) valid = false;
          scores[q] = valid ? current[qx * ny + qy] : kInfinity;
          if (scores[q] < best) { best = scores[q]; choice = q; }
        }
        if (best < current[px * ny + py] - 1.0e-5f) {
          px = nx_choice[choice];
          py = ny_choice[choice];
          ++length;
          path_x[k] = (static_cast<float>(px) + 0.5f) * resolution;
          path_y[k] = (static_cast<float>(py) + 0.5f) * resolution;
          const bool reached = px == gx && py == gy;
          active = !reached;
        } else {
          active = false;
          path_x[k] = path_x[k - 1];
          path_y[k] = path_y[k - 1];
        }
      } else {
        path_x[k] = path_x[k - 1];
        path_y[k] = path_y[k - 1];
      }
      if (k % 16 == 0 && active) local_checkpoint_mask |= 1 << (k / 16 - 1);
    }
    // Greedily shortcut the 8-connected route wherever a swept-cell line of
    // sight is clear. Keep the occupancy inflation, so shortcuts retain robot
    // clearance around walls, robots, and other hard obstacles.
    int anchor=0, output_length=1;
    while (anchor < length-1) {
      int chosen=anchor+1;
      for (int candidate=length-1; candidate>anchor+1; --candidate) {
        if (line_of_sight_clear(blocked,b,path_x[anchor],path_y[anchor],
              path_x[candidate],path_y[candidate],nx,ny,resolution,
              source_x,source_y,gx,gy)) {
          chosen=candidate;
          break;
        }
      }
      if (output_length != chosen+1) {
        path_x[output_length]=path_x[chosen];
        path_y[output_length]=path_y[chosen];
      }
      ++output_length;
      anchor=chosen;
    }
    for (int k=output_length;k<kMaxPoints;++k) {
      path_x[k]=path_x[output_length-1];
      path_y[k]=path_y[output_length-1];
    }
    checkpoint_mask = 0;
    if (output_length > 16) checkpoint_mask |= 1;
    if (output_length > 32) checkpoint_mask |= 2;
    if (output_length > 48) checkpoint_mask |= 4;
    if (output_length > 64) checkpoint_mask |= 8;
    route_length = output_length;
    lengths[b] = output_length;
    active_checkpoints[b] = checkpoint_mask;
    active_checkpoints[b] = local_checkpoint_mask;
    for (int k = 0; k < kMaxPoints; ++k) {
      path[(static_cast<int64_t>(b) * kMaxPoints + k) * 2] = path_x[k];
      path[(static_cast<int64_t>(b) * kMaxPoints + k) * 2 + 1] = path_y[k];
    }
  }
  __syncthreads();

  // Match torch.linalg.vector_norm of each path segment and prefix stations.
  for (int i = lane; i < kMaxPoints - 1; i += blockDim.x) {
    const float dx = path_x[i + 1] - path_x[i];
    const float dy = path_y[i + 1] - path_y[i];
    distance[i] = sqrtf(dx * dx + dy * dy);
  }
  if (lane == kMaxPoints - 1) distance[kMaxPoints - 1] = 0.f;
  for (int i = lane; i < kMaxPoints; i += blockDim.x) {
    float sum = 0.f;
    for (int j = 0; j < i; ++j) sum += distance[j];
    station[i] = sum;
  }
  __syncthreads();

  const float mu = fmaxf(friction[b], 0.1f);
  const float accel = fmaxf(acceleration[b], 0.2f);
  const float vcap = fmaxf(speed[b], 0.1f);
  for (int i = lane; i < kMaxPoints; i += blockDim.x) {
    float curve = 0.f;
    if (i > 0 && i < kMaxPoints - 1) {
      const float d0 = distance[i - 1];
      const float d1 = distance[i];
      const float ux0 = (path_x[i] - path_x[i - 1]) / fmaxf(d0, 1.0e-6f);
      const float uy0 = (path_y[i] - path_y[i - 1]) / fmaxf(d0, 1.0e-6f);
      const float ux1 = (path_x[i + 1] - path_x[i]) / fmaxf(d1, 1.0e-6f);
      const float uy1 = (path_y[i + 1] - path_y[i]) / fmaxf(d1, 1.0e-6f);
      const float cross = ux0 * uy1 - uy0 * ux1;
      const float dot = ux0 * ux1 + uy0 * uy1;
      const float turn = fabsf(atan2f(cross, dot));
      const float arc = fmaxf(0.5f * (d0 + d1), 1.0e-4f);
      curve = turn / arc;
    }
    float cap = vcap;
    if (curve > 1.0e-6f) {
      const float lateral = sqrtf((0.65f * mu * 9.81f) / fmaxf(curve, 1.0e-6f));
      const float steer = steer_rate_limit / fmaxf(curve, 1.0e-6f);
      cap = fminf(vcap, fminf(lateral, steer));
    }
    if (i == route_length - 1) cap = 0.f;
    caps[i] = cap;
  }
  __syncthreads();

  const float braking = 0.65f * accel;
  for (int i = lane; i < kMaxPoints; i += blockDim.x) {
    float best = INFINITY;
    if (i < route_length) {
      for (int j = i; j < route_length; ++j) {
        const float ds = fmaxf(station[j] - station[i], 0.f);
        const float reachable = sqrtf(caps[j] * caps[j] + 2.f * braking * ds);
        best = fminf(best, reachable);
      }
    }
    profile[static_cast<int64_t>(b) * kMaxPoints + i] = best;
  }
}

__global__ void cleanup_batch_path_kernel(float* path, const int32_t* active_checkpoints,
                                          int batch) {
  __shared__ int aggregate_mask;
  if (threadIdx.x == 0) aggregate_mask = 0;
  __syncthreads();
  int local = 0;
  for (int b = threadIdx.x; b < batch; b += blockDim.x)
    local |= active_checkpoints[b];
  atomicOr(&aggregate_mask, local);
  __syncthreads();
  int zero_from = kMaxPoints;
  if ((aggregate_mask & 1) == 0) zero_from = 17;
  else if ((aggregate_mask & 2) == 0) zero_from = 33;
  else if ((aggregate_mask & 4) == 0) zero_from = 49;
  else if ((aggregate_mask & 8) == 0) zero_from = 65;
  if (zero_from < kMaxPoints) {
    const int64_t total = static_cast<int64_t>(batch) * kMaxPoints * 2;
    for (int64_t i = static_cast<int64_t>(threadIdx.x); i < total; i += blockDim.x) {
      const int point = static_cast<int>((i / 2) % kMaxPoints);
      if (point >= zero_from) path[i] = 0.f;
    }
  }
}
}  // namespace

void fused_route_launch(const float* value, const bool* blocked, const float* bump,
                        const float* start_xy, const float* goal_xy,
                        const int64_t* start_x,
                        const int64_t* start_y, const int64_t* goal_x,
                        const int64_t* goal_y, const float* speed,
                        const float* friction, const float* acceleration,
                        float* potential, float* path, int64_t* lengths,
                        float* profile, float* resolved_goal,
                        int32_t* active_checkpoints, float steer_rate_limit,
                        int nx, int ny, int64_t total,
                        int sweeps, bool stop_when_converged, bool project_blocked_goal,
                        float resolution, hipStream_t stream) {
  const int cells = nx * ny;
  const int batch = static_cast<int>(total / cells);
  constexpr int threads = 256;
  const size_t shared_bytes = static_cast<size_t>(2 * cells + 5 * kMaxPoints) * sizeof(float);
  hipLaunchKernelGGL(fused_bellman_route_kernel, dim3(batch), dim3(threads), shared_bytes,
                     stream, value, blocked, bump, start_xy, goal_xy, start_x, start_y,
                     goal_x, goal_y, speed, friction, acceleration, potential,
                     path, lengths, profile, resolved_goal, active_checkpoints,
                     steer_rate_limit,
                     nx, ny, cells, sweeps,
                     stop_when_converged, project_blocked_goal, resolution);
  hipLaunchKernelGGL(cleanup_batch_path_kernel, dim3(1), dim3(256), 0, stream,
                     path, active_checkpoints, batch);
}
