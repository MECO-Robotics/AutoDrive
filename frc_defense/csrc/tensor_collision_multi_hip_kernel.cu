#include <hip/hip_runtime.h>
#include <cmath>

__device__ __forceinline__ float sign_ref(float x) {
  return static_cast<float>((x > 0.0f) - (x < 0.0f));
}
__device__ __forceinline__ float cross_ref(float ax, float ay, float bx, float by) {
  return ax * by - ay * bx;
}

__global__ void robot_contacts_kernel(float* pose, float* velocity,
    const float* length, const float* width, const float* mass,
    const float* yaw_mult, const float* friction, const bool* active,
    const int64_t* team_ids, bool* robot_contact, bool* opponent_contact,
    int n) {
  const int world = blockIdx.x * blockDim.x + threadIdx.x;
  if (world >= n || !active[world]) return;

  float px[6], py[6], theta[6], vx[6], vy[6], omega[6];
  float len[6], wid[6], inv_mass[6], inertia[6];
  float ux[6], uy[6], lx[6], ly[6];
  float pos_dx[6] = {0.f}, pos_dy[6] = {0.f};
  float dvx[6] = {0.f}, dvy[6] = {0.f}, dw[6] = {0.f};
  const int base = world * 18;
  const int prop = world * 6;
  #pragma unroll
  for (int r = 0; r < 6; ++r) {
    const int p = base + r * 3;
    px[r] = pose[p]; py[r] = pose[p + 1]; theta[r] = pose[p + 2];
    vx[r] = velocity[p]; vy[r] = velocity[p + 1]; omega[r] = velocity[p + 2];
    len[r] = length[prop + r]; wid[r] = width[prop + r];
    inv_mass[r] = 1.f / mass[prop + r];
    const float inertia_base = (mass[prop + r] *
        (len[r] * len[r] + wid[r] * wid[r]) / 12.f) * yaw_mult[prop + r];
    inertia[r] = fmaxf(inertia_base, 1.e-8f);
    float s, c; __sincosf(theta[r], &s, &c);
    ux[r] = c; uy[r] = s; lx[r] = -s; ly[r] = c;
  }

  bool any_robot = false, any_opponent = false;
  const float mu = sqrtf(fmaxf(friction[world], 0.f));
  // Match torch.triu_indices ordering and calculate all impulses from the
  // pre-contact velocities, then sum them before mutating the state.
  #pragma unroll
  for (int i = 0; i < 6; ++i) {
    #pragma unroll
    for (int j = i + 1; j < 6; ++j) {
      float ax[4] = {ux[i], lx[i], ux[j], lx[j]};
      float ay[4] = {uy[i], ly[i], uy[j], ly[j]};
      const float hi_l = len[i] * .5f, hi_w = wid[i] * .5f;
      const float hj_l = len[j] * .5f, hj_w = wid[j] * .5f;
      const float dx = px[j] - px[i], dy = py[j] - py[i];
      float overlap[4], signed_axis[4];
      #pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float pu = fabsf(ax[k] * ux[i] + ay[k] * uy[i]);
        const float pv = fabsf(ax[k] * lx[i] + ay[k] * ly[i]);
        const float qu = fabsf(ax[k] * ux[j] + ay[k] * uy[j]);
        const float qv = fabsf(ax[k] * lx[j] + ay[k] * ly[j]);
        const float radii = pu * hi_l + pv * hi_w + qu * hj_l + qv * hj_w;
        signed_axis[k] = ax[k] * dx + ay[k] * dy;
        overlap[k] = radii - fabsf(signed_axis[k]);
      }
      int axis = 0; float depth = overlap[0];
      #pragma unroll
      for (int k = 1; k < 4; ++k) {
        if (overlap[k] < depth) { depth = overlap[k]; axis = k; }
      }
      const bool valid = depth > 0.f;
      const float sign = signed_axis[axis] >= 0.f ? 1.f : -1.f;
      const float nx = ax[axis] * sign, ny = ay[axis] * sign;
      any_robot |= valid;
      any_opponent |= valid && team_ids[i] != team_ids[j];
      const float ti = inv_mass[i], tj = inv_mass[j];
      const float total = fmaxf(ti + tj, 1.e-8f);
      const float correction = valid ? fmaxf(depth, 0.f) + 1.e-4f : 0.f;
      const float move_i = correction * ti / total;
      const float move_j = correction * tj / total;
      pos_dx[i] -= nx * move_i; pos_dy[i] -= ny * move_i;
      pos_dx[j] += nx * move_j; pos_dy[j] += ny * move_j;

      const float su_i = sign_ref(nx * ux[i] + ny * uy[i]);
      const float sv_i = sign_ref(nx * lx[i] + ny * ly[i]);
      const float su_j = sign_ref(nx * ux[j] + ny * uy[j]);
      const float sv_j = sign_ref(nx * lx[j] + ny * ly[j]);
      const float six = (px[i] + su_i * ux[i] * hi_l) + sv_i * lx[i] * hi_w;
      const float siy = (py[i] + su_i * uy[i] * hi_l) + sv_i * ly[i] * hi_w;
      const float sjx = (px[j] - su_j * ux[j] * hj_l) - sv_j * lx[j] * hj_w;
      const float sjy = (py[j] - su_j * uy[j] * hj_l) - sv_j * ly[j] * hj_w;
      const float cpx = .5f * (six + sjx), cpy = .5f * (siy + sjy);
      const float rix = cpx - px[i], riy = cpy - py[i];
      const float rjx = cpx - px[j], rjy = cpy - py[j];
      const float cvi_x = vx[i] - omega[i] * riy;
      const float cvi_y = vy[i] + omega[i] * rix;
      const float cvj_x = vx[j] - omega[j] * rjy;
      const float cvj_y = vy[j] + omega[j] * rjx;
      const float vn = (cvj_x - cvi_x) * nx + (cvj_y - cvi_y) * ny;
      const float rn_i = cross_ref(rix, riy, nx, ny);
      const float rn_j = cross_ref(rjx, rjy, nx, ny);
      const float ii = inertia[i], ij = inertia[j];
      const float eff = total + rn_i * rn_i / ii + rn_j * rn_j / ij;
      const float jn = (valid && vn < 0.f) ? (-1.05f * vn) / fmaxf(eff,1.e-8f) : 0.f;

      // The simulator computes tangential impulse using the same original
      // contact velocities as the normal impulse; it applies both together.
      const float tx = -ny, ty = nx;
      const float vt = (cvj_x - cvi_x) * tx + (cvj_y - cvi_y) * ty;
      const float rt_i = cross_ref(rix, riy, tx, ty);
      const float rt_j = cross_ref(rjx, rjy, tx, ty);
      const float eff_t = total + rt_i * rt_i / ii + rt_j * rt_j / ij;
      float jt = -vt / fmaxf(eff_t,1.e-8f);
      const float limit = mu * jn;
      jt = fminf(fmaxf(jt,-limit),limit);
      const float ix = jn * nx + jt * tx;
      const float iy = jn * ny + jt * ty;
      dvx[i] -= ix * ti; dvy[i] -= iy * ti;
      dvx[j] += ix * tj; dvy[j] += iy * tj;
      dw[i] -= (jn * rn_i + jt * rt_i) / ii;
      dw[j] += (jn * rn_j + jt * rt_j) / ij;
    }
  }

  #pragma unroll
  for (int r = 0; r < 6; ++r) {
    const int p = base + r * 3;
    pose[p] = px[r] + pos_dx[r];
    pose[p + 1] = py[r] + pos_dy[r];
    velocity[p] = vx[r] + dvx[r];
    velocity[p + 1] = vy[r] + dvy[r];
    velocity[p + 2] = omega[r] + dw[r];
  }
  robot_contact[world] = robot_contact[world] || any_robot;
  opponent_contact[world] = opponent_contact[world] || any_opponent;
}

void robot_contacts_launch(float* pose, float* velocity, const float* length,
    const float* width, const float* mass, const float* yaw_mult,
    const float* friction, const bool* active, const int64_t* team_ids,
    bool* robot_contact, bool* opponent_contact, int n, hipStream_t stream) {
  constexpr int threads = 128;
  const int blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(robot_contacts_kernel, dim3(blocks), dim3(threads), 0, stream,
      pose, velocity, length, width, mass, yaw_mult, friction, active, team_ids,
      robot_contact, opponent_contact, n);
}
