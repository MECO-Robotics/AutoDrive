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

__global__ void robot_contacts_single_world_parallel_kernel(
    float* pose, float* velocity, const float* length, const float* width,
    const float* mass, const float* yaw_mult, const float* friction,
    const bool* active, const int64_t* team_ids, bool* robot_contact,
    bool* opponent_contact) {
  const int lane = threadIdx.x;
  if (!active[0]) return;
  __shared__ float pair_delta[15][10];
  __shared__ int any_robot, any_opponent;
  const int pair_i[15] = {0,0,0,0,0,1,1,1,1,2,2,2,3,3,4};
  const int pair_j[15] = {1,2,3,4,5,2,3,4,5,3,4,5,4,5,5};
  if (lane == 0) { any_robot = 0; any_opponent = 0; }
  __syncthreads();

  if (lane < 15) {
    const int i = pair_i[lane], j = pair_j[lane];
    const int pi = i * 3, pj = j * 3;
    const float pxi = pose[pi], pyi = pose[pi + 1], thi = pose[pi + 2];
    const float pxj = pose[pj], pyj = pose[pj + 1], thj = pose[pj + 2];
    const float vxi = velocity[pi], vyi = velocity[pi + 1], wi = velocity[pi + 2];
    const float vxj = velocity[pj], vyj = velocity[pj + 1], wj = velocity[pj + 2];
    const float li = length[i], wi_robot = width[i];
    const float lj = length[j], wj_robot = width[j];
    const float hi_l = li * .5f, hi_w = wi_robot * .5f;
    const float hj_l = lj * .5f, hj_w = wj_robot * .5f;
    const float inv_i = 1.f / mass[i], inv_j = 1.f / mass[j];
    const float inertia_i = fmaxf((mass[i] * (li * li + wi_robot * wi_robot) / 12.f) * yaw_mult[i], 1.e-8f);
    const float inertia_j = fmaxf((mass[j] * (lj * lj + wj_robot * wj_robot) / 12.f) * yaw_mult[j], 1.e-8f);
    float si, ci, sj, cj;
    __sincosf(thi, &si, &ci); __sincosf(thj, &sj, &cj);
    const float uix=ci, uiy=si, lix=-si, liy=ci;
    const float ujx=cj, ujy=sj, ljx=-sj, ljy=cj;
    float ax[4]={uix,lix,ujx,ljx}, ay[4]={uiy,liy,ujy,ljy};
    const float dx=pxj-pxi, dy=pyj-pyi;
    float overlap[4], signed_axis[4];
    #pragma unroll
    for (int k=0;k<4;++k) {
      const float pu=fabsf(ax[k]*uix+ay[k]*uiy), pv=fabsf(ax[k]*lix+ay[k]*liy);
      const float qu=fabsf(ax[k]*ujx+ay[k]*ujy), qv=fabsf(ax[k]*ljx+ay[k]*ljy);
      overlap[k]=pu*hi_l+pv*hi_w+qu*hj_l+qv*hj_w-fabsf(ax[k]*dx+ay[k]*dy);
      signed_axis[k]=ax[k]*dx+ay[k]*dy;
    }
    int axis=0; float depth=overlap[0];
    #pragma unroll
    for (int k=1;k<4;++k) if (overlap[k]<depth) { depth=overlap[k]; axis=k; }
    const bool valid=depth>0.f;
    if (valid) {
      atomicExch(&any_robot,1);
      if (team_ids[i]!=team_ids[j]) atomicExch(&any_opponent,1);
    }
    const float sign=signed_axis[axis]>=0.f?1.f:-1.f;
    const float nx=ax[axis]*sign, ny=ay[axis]*sign;
    const float total=fmaxf(inv_i+inv_j,1.e-8f);
    const float correction=valid?fmaxf(depth,0.f)+1.e-4f:0.f;
    float dxi=-nx*(correction*inv_i/total), dyi=-ny*(correction*inv_i/total);
    float dxj= nx*(correction*inv_j/total), dyj= ny*(correction*inv_j/total);
    const float sui=sign_ref(nx*uix+ny*uiy), svi=sign_ref(nx*lix+ny*liy);
    const float suj=sign_ref(nx*ujx+ny*ujy), svj=sign_ref(nx*ljx+ny*ljy);
    const float six=(pxi+sui*uix*hi_l)+svi*lix*hi_w;
    const float siy=(pyi+sui*uiy*hi_l)+svi*liy*hi_w;
    const float sjx=(pxj-suj*ujx*hj_l)-svj*ljx*hj_w;
    const float sjy=(pyj-suj*ujy*hj_l)-svj*ljy*hj_w;
    const float cpx=.5f*(six+sjx), cpy=.5f*(siy+sjy);
    const float rix=cpx-pxi, riy=cpy-pyi, rjx=cpx-pxj, rjy=cpy-pyj;
    const float cvix=vxi-wi*riy, cviy=vyi+wi*rix;
    const float cvjx=vxj-wj*rjy, cvjy=vyj+wj*rjx;
    const float vn=(cvjx-cvix)*nx+(cvjy-cviy)*ny;
    const float rn_i=cross_ref(rix,riy,nx,ny), rn_j=cross_ref(rjx,rjy,nx,ny);
    const float eff=total+rn_i*rn_i/inertia_i+rn_j*rn_j/inertia_j;
    const float jn=(valid&&vn<0.f)?(-1.05f*vn)/fmaxf(eff,1.e-8f):0.f;
    const float tx=-ny, ty=nx;
    const float vt=(cvjx-cvix)*tx+(cvjy-cviy)*ty;
    const float rt_i=cross_ref(rix,riy,tx,ty), rt_j=cross_ref(rjx,rjy,tx,ty);
    const float eff_t=total+rt_i*rt_i/inertia_i+rt_j*rt_j/inertia_j;
    float jt=-vt/fmaxf(eff_t,1.e-8f);
    const float limit=sqrtf(fmaxf(friction[0],0.f))*jn;
    jt=fminf(fmaxf(jt,-limit),limit);
    const float ix=jn*nx+jt*tx, iy=jn*ny+jt*ty;
    pair_delta[lane][0]=dxi; pair_delta[lane][1]=dyi;
    pair_delta[lane][2]=-ix*inv_i; pair_delta[lane][3]=-iy*inv_i;
    pair_delta[lane][4]=-(jn*rn_i+jt*rt_i)/inertia_i;
    pair_delta[lane][5]=dxj; pair_delta[lane][6]=dyj;
    pair_delta[lane][7]= ix*inv_j; pair_delta[lane][8]= iy*inv_j;
    pair_delta[lane][9]=(jn*rn_j+jt*rt_j)/inertia_j;
  }
  __syncthreads();

  if (lane < 6) {
    float delta[5]={0.f,0.f,0.f,0.f,0.f};
    // Sum in the same lexicographic pair order as the reference resolver.
    for (int pair=0;pair<15;++pair) {
      if (pair_i[pair]==lane) {
        #pragma unroll
        for (int k=0;k<5;++k) delta[k]+=pair_delta[pair][k];
      } else if (pair_j[pair]==lane) {
        #pragma unroll
        for (int k=0;k<5;++k) delta[k]+=pair_delta[pair][k+5];
      }
    }
    const int p=lane*3;
    pose[p]+=delta[0]; pose[p+1]+=delta[1];
    velocity[p]+=delta[2]; velocity[p+1]+=delta[3]; velocity[p+2]+=delta[4];
  }
  __syncthreads();
  if (lane==0) {
    robot_contact[0]=robot_contact[0]||any_robot;
    opponent_contact[0]=opponent_contact[0]||any_opponent;
  }
}

void robot_contacts_launch(float* pose, float* velocity, const float* length,
    const float* width, const float* mass, const float* yaw_mult,
    const float* friction, const bool* active, const int64_t* team_ids,
    bool* robot_contact, bool* opponent_contact, int n,
    bool parallel_single_world, hipStream_t stream) {
  if (n == 1 && parallel_single_world) {
    hipLaunchKernelGGL(robot_contacts_single_world_parallel_kernel,
        dim3(1), dim3(32), 0, stream, pose, velocity, length, width, mass,
        yaw_mult, friction, active, team_ids, robot_contact, opponent_contact);
  } else {
    constexpr int threads = 128;
    const int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(robot_contacts_kernel, dim3(blocks), dim3(threads), 0, stream,
        pose, velocity, length, width, mass, yaw_mult, friction, active, team_ids,
        robot_contact, opponent_contact, n);
  }
}

__global__ void field_contacts_kernel(float* pose, float* velocity,
    const float* length, const float* width, const float* mass,
    const float* yaw_mult, const float* friction, const float* boxes,
    const bool* active, bool* field_contact, int n, int box_count,
    int sweep_steps, float substep_dt) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n * 6;
  if (row >= total) return;
  const int world = row / 6;
  const int robot = row - world * 6;
  if (!active[world]) return;

  const int p = row * 3;
  for (int sweep = 0; sweep < sweep_steps; ++sweep) {
    if (substep_dt > 0.f) {
      pose[p] += velocity[p] * substep_dt;
      pose[p + 1] += velocity[p + 1] * substep_dt;
      float theta = pose[p + 2] + velocity[p + 2] * substep_dt;
      constexpr float pi = 3.14159265358979323846f;
      constexpr float two_pi = 6.28318530717958647692f;
      theta = fmodf(theta + pi, two_pi);
      if (theta < 0.f) theta += two_pi;
      pose[p + 2] = theta - pi;
    }
    const float x = pose[p], y = pose[p + 1], theta = pose[p + 2];
    const float len = length[row], wid = width[row];
    const float hl = len * .5f, hw = wid * .5f;
    float s, c; __sincosf(theta, &s, &c);
    const float axes_x[4] = {1.f, 0.f, c, -s};
    const float axes_y[4] = {0.f, 1.f, s, c};
    const float robot_r[4] = {
        fabsf(c) * hl + fabsf(s) * hw,
        fabsf(s) * hl + fabsf(c) * hw,
        hl, hw};

    float best_depth = -1.f;
    int best_box = 0;
    for (int b = 0; b < box_count; ++b) {
      const float cx = boxes[b * 4], cy = boxes[b * 4 + 1];
      const float hx = boxes[b * 4 + 2], hy = boxes[b * 4 + 3];
      const float dx = x - cx, dy = y - cy;
      const float box_r[4] = {
          hx, hy,
          hx * fabsf(c) + hy * fabsf(s),
          hx * fabsf(s) + hy * fabsf(c)};
      float depth = INFINITY;
      #pragma unroll
      for (int a = 0; a < 4; ++a) {
        const float signed_distance = axes_x[a] * dx + axes_y[a] * dy;
        const float penetration = robot_r[a] + box_r[a] - fabsf(signed_distance);
        depth = fminf(depth, penetration);
      }
      if (depth >= 0.f && depth > best_depth) {
        best_depth = depth;
        best_box = b;
      }
    }

    const bool contact = best_depth >= 0.f;
    field_contact[row] = field_contact[row] || contact;
    if (!contact) continue;

    const float cx = boxes[best_box * 4], cy = boxes[best_box * 4 + 1];
    const float hx = boxes[best_box * 4 + 2], hy = boxes[best_box * 4 + 3];
    const float dx = x - cx, dy = y - cy;
    const float box_r[4] = {
        hx, hy,
        hx * fabsf(c) + hy * fabsf(s),
        hx * fabsf(s) + hy * fabsf(c)};
    float penetration[4], signed_distance[4];
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
      signed_distance[a] = axes_x[a] * dx + axes_y[a] * dy;
      penetration[a] = robot_r[a] + box_r[a] - fabsf(signed_distance[a]);
    }
    float normal_x[4], normal_y[4], approach[4];
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
      const float sign = signed_distance[a] >= 0.f ? 1.f : -1.f;
      normal_x[a] = axes_x[a] * sign;
      normal_y[a] = axes_y[a] * sign;
      approach[a] = normal_x[a] * velocity[p] + normal_y[a] * velocity[p + 1];
    }
    float approach_speed = INFINITY;
    int approach_axis = 0;
    int min_penetration_axis = 0;
    float min_penetration = penetration[0];
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
      if (penetration[a] < min_penetration) {
        min_penetration = penetration[a];
        min_penetration_axis = a;
      }
      if (penetration[a] <= best_depth + 1.e-6f && penetration[a] >= 0.f &&
          approach[a] < approach_speed) {
        approach_speed = approach[a];
        approach_axis = a;
      }
    }
    const int axis = approach_speed < -1.e-6f ? approach_axis : min_penetration_axis;
    const float nx = normal_x[axis], ny = normal_y[axis];
    const float depth = fmaxf(best_depth, 0.f);
    const float corrected_x = x + nx * (depth + 1.e-4f);
    const float corrected_y = y + ny * (depth + 1.e-4f);
    pose[p] = corrected_x;
    pose[p + 1] = corrected_y;

    const float ux = c, uy = s, vx = -s, vy = c;
    const float tx = -ny, ty = nx;
    const float robot_normal = fabsf(nx * ux + ny * uy) * hl +
                               fabsf(nx * vx + ny * vy) * hw;
    const float box_normal = fabsf(nx) * hx + fabsf(ny) * hy;
    const float normal_coordinate = .5f * (
        (nx * corrected_x + ny * corrected_y) - robot_normal +
        (nx * cx + ny * cy) + box_normal);
    const float robot_tangent = fabsf(tx * ux + ty * uy) * hl +
                                fabsf(tx * vx + ty * vy) * hw;
    const float box_tangent = fabsf(tx) * hx + fabsf(ty) * hy;
    const float robot_center = tx * corrected_x + ty * corrected_y;
    const float box_center = tx * cx + ty * cy;
    const float overlap_low = fmaxf(robot_center - robot_tangent,
                                    box_center - box_tangent);
    const float overlap_high = fminf(robot_center + robot_tangent,
                                     box_center + box_tangent);
    const float tangent_coordinate = .5f * (overlap_low + overlap_high);
    const float point_x = nx * normal_coordinate + tx * tangent_coordinate;
    const float point_y = ny * normal_coordinate + ty * tangent_coordinate;

    const float lever_x = point_x - corrected_x;
    const float lever_y = point_y - corrected_y;
    const float inertia = fmaxf((mass[row] * (len * len + wid * wid) / 12.f) *
                                yaw_mult[row], 1.e-8f);
    const float inv_mass = 1.f / mass[row];
    const float arm_x = -lever_y, arm_y = lever_x;
    const float omega = velocity[p + 2];
    float contact_vx = velocity[p] + omega * arm_x;
    float contact_vy = velocity[p + 1] + omega * arm_y;
    const float vn = contact_vx * nx + contact_vy * ny;
    const float rn = lever_x * ny - lever_y * nx;
    const float effective = inv_mass + rn * rn / inertia;
    const float jn = vn < 0.f ? (-1.05f * vn) / fmaxf(effective, 1.e-8f) : 0.f;
    velocity[p] += jn * nx * inv_mass;
    velocity[p + 1] += jn * ny * inv_mass;
    velocity[p + 2] += (lever_x * (jn * ny) - lever_y * (jn * nx)) / inertia;

    contact_vx = velocity[p] + velocity[p + 2] * arm_x;
    contact_vy = velocity[p + 1] + velocity[p + 2] * arm_y;
    const float vt = contact_vx * tx + contact_vy * ty;
    const float rt = lever_x * ty - lever_y * tx;
    const float effective_tangent = inv_mass + rt * rt / inertia;
    float jt = -vt / fmaxf(effective_tangent, 1.e-8f);
    const float friction_limit = friction[world] * jn;
    jt = fminf(fmaxf(jt, -friction_limit), friction_limit);
    velocity[p] += jt * tx * inv_mass;
    velocity[p + 1] += jt * ty * inv_mass;
    velocity[p + 2] += (lever_x * (jt * ty) - lever_y * (jt * tx)) / inertia;
  }
}

void field_contacts_launch(float* pose, float* velocity, const float* length,
    const float* width, const float* mass, const float* yaw_mult,
    const float* friction, const float* boxes, const bool* active,
    bool* field_contact, int n, int box_count, hipStream_t stream) {
  constexpr int sweep_steps = 1;
  constexpr float substep_dt = 0.f;
  constexpr int threads = 128;
  const int total = n * 6;
  const int blocks = (total + threads - 1) / threads;
  hipLaunchKernelGGL(field_contacts_kernel, dim3(blocks), dim3(threads), 0, stream,
      pose, velocity, length, width, mass, yaw_mult, friction, boxes, active,
      field_contact, n, box_count, sweep_steps, substep_dt);
}

void field_sweep_contacts_launch(float* pose, float* velocity, const float* length,
    const float* width, const float* mass, const float* yaw_mult,
    const float* friction, const float* boxes, const bool* active,
    bool* field_contact, int n, int box_count, int sweep_steps,
    float substep_dt, hipStream_t stream) {
  constexpr int threads = 128;
  const int total = n * 6;
  const int blocks = (total + threads - 1) / threads;
  hipLaunchKernelGGL(field_contacts_kernel, dim3(blocks), dim3(threads), 0, stream,
      pose, velocity, length, width, mass, yaw_mult, friction, boxes, active,
      field_contact, n, box_count, sweep_steps, substep_dt);
}

__device__ __forceinline__ bool score_point_clear(float x, float y, float radius,
    const float* boxes, int box_count) {
  for (int b = 0; b < box_count; ++b) {
    const float dx = fabsf(x - boxes[b * 4]);
    const float dy = fabsf(y - boxes[b * 4 + 1]);
    if (dx <= boxes[b * 4 + 2] + radius + .15f &&
        dy <= boxes[b * 4 + 3] + radius + .15f)
      return false;
  }
  return true;
}

__global__ void safe_score_targets_kernel(const float* pose,
    const float* score_target, const float* robot_radius, const float* x_min,
    const float* x_max, const float* boxes, const float* grid_points,
    const bool* grid_clear, float* output, int n, int box_count,
    int grid_count, float field_width) {
  const int row = blockIdx.x;
  const int total = n * 6;
  if (row >= total) return;
  const int lane = threadIdx.x;
  const int pose_index = row * 3;
  const int target_index = row * 2;
  const float pose_x = pose[pose_index], pose_y = pose[pose_index + 1];
  const float radius = robot_radius[row];
  const float raw_x = score_target[target_index];
  const float raw_y = score_target[target_index + 1];
  const int side_count = 1 + 4 * box_count;
  const int candidate_count = side_count + grid_count;
  const int grid_offset = row * grid_count;
  __shared__ float shared_distance[128];
  __shared__ int shared_index[128];
  __shared__ float shared_x[128], shared_y[128];
  float lane_distance = INFINITY;
  int lane_index = 0x7fffffff;
  float lane_x = raw_x, lane_y = raw_y;

  // Candidate order is raw pose, left/right/lower/upper field-box points,
  // then the fixed safe grid. Lower-index ties preserve torch.argmin behavior.
  for (int candidate = lane; candidate < candidate_count;
       candidate += blockDim.x) {
    float x = raw_x, y = raw_y;
    bool precleared = false;
    if (candidate > 0 && candidate < side_count) {
      const int side_index = candidate - 1;
      const int group = side_index / box_count;
      const int b = side_index % box_count;
      const float bx = boxes[b * 4], by = boxes[b * 4 + 1];
      const float hx = boxes[b * 4 + 2], hy = boxes[b * 4 + 3];
      if (group == 0)
        x = fmaxf(x_min[row], fminf(bx - hx - radius - .40f, x_max[row]));
      else if (group == 1)
        x = fmaxf(x_min[row], fminf(bx + hx + radius + .40f, x_max[row]));
      else if (group == 2)
        y = fminf(fmaxf(by - hy - radius - .40f, radius), field_width - radius);
      else
        y = fminf(fmaxf(by + hy + radius + .40f, radius), field_width - radius);
    } else if (candidate >= side_count) {
      const int g = candidate - side_count;
      if (!grid_clear[grid_offset + g]) continue;
      const int index = (grid_offset + g) * 2;
      x = grid_points[index];
      y = grid_points[index + 1];
      precleared = true;
    }
    if (!precleared && !score_point_clear(x, y, radius, boxes, box_count))
      continue;
    const float dx = x - pose_x, dy = y - pose_y;
    const float distance = sqrtf(dx * dx + dy * dy);
    if (distance < lane_distance ||
        (distance == lane_distance && candidate < lane_index)) {
      lane_distance = distance;
      lane_index = candidate;
      lane_x = x;
      lane_y = y;
    }
  }
  shared_distance[lane] = lane_distance;
  shared_index[lane] = lane_index;
  shared_x[lane] = lane_x;
  shared_y[lane] = lane_y;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (lane < offset) {
      const float other_distance = shared_distance[lane + offset];
      const int other_index = shared_index[lane + offset];
      if (other_distance < shared_distance[lane] ||
          (other_distance == shared_distance[lane] &&
           other_index < shared_index[lane])) {
        shared_distance[lane] = other_distance;
        shared_index[lane] = other_index;
        shared_x[lane] = shared_x[lane + offset];
        shared_y[lane] = shared_y[lane + offset];
      }
    }
    __syncthreads();
  }
  const bool found = shared_distance[0] < INFINITY;
  output[target_index] = found ? shared_x[0] : raw_x;
  output[target_index + 1] = found ? shared_y[0] : raw_y;
}

void safe_score_targets_launch(const float* pose, const float* score_target,
    const float* robot_radius, const float* x_min, const float* x_max,
    const float* boxes, const float* grid_points, const bool* grid_clear,
    float* output, int n, int box_count, int grid_count, float field_width,
    hipStream_t stream) {
  constexpr int threads = 128;
  const int blocks = n * 6;
  hipLaunchKernelGGL(safe_score_targets_kernel, dim3(blocks), dim3(threads),
      0, stream, pose, score_target, robot_radius, x_min, x_max, boxes,
      grid_points, grid_clear, output, n, box_count, grid_count, field_width);
}

__global__ void avoid_contention_kernel(const float* pose,const float* command,
    const float* targets,const float* length,const float* width,const float* speed,
    const bool* controlled,const int64_t* teams,const bool* active,
    int64_t* winner,float* adjusted,int n,bool teammate_intent) {
  const int world=blockIdx.x, lane=threadIdx.x;
  if(world>=n)return;
  __shared__ int pair_i[15],pair_j[15],pair_winner[15],pair_loser[15],pair_conflict[15];
  __shared__ float avoid_x[15],avoid_y[15];
  if(lane<15){
    int i=0,j=0,k=0;
    for(int a=0;a<6;a++)for(int b=a+1;b<6;b++){
      if(k==lane){i=a;j=b;}++k;
    }
    pair_i[lane]=i;pair_j[lane]=j;
  }
  __syncthreads();
  const int pose_base=world*18,xybase=world*12,prop=world*6;
  if(lane<15){
    const int i=pair_i[lane],j=pair_j[lane];
    const int pi=pose_base+i*3,pj=pose_base+j*3,ci=xybase+i*2,cj=xybase+j*2;
    const float rx=pose[pi]-pose[pj],ry=pose[pi+1]-pose[pj+1];
    const float cix=controlled[i]?command[ci]:0.f;
    const float ciy=controlled[i]?command[ci+1]:0.f;
    const float cjx=controlled[j]?command[cj]:0.f;
    const float cjy=controlled[j]?command[cj+1]:0.f;
    const float rvx=cix-cjx,rvy=ciy-cjy;
    const float closing=rvx*rvx+rvy*rvy;
    const float time=fminf(.7f,fmaxf(0.f,-(rx*rvx+ry*rvy)/fmaxf(closing,1.e-6f)));
    const float closest_x=rx+rvx*time,closest_y=ry+rvy*time;
    const float closest_dist=sqrtf(closest_x*closest_x+closest_y*closest_y);
    const float current_dist=sqrtf(rx*rx+ry*ry);
    const float radius_i=.5f*sqrtf(length[prop+i]*length[prop+i]+
                                   width[prop+i]*width[prop+i]);
    const float radius_j=.5f*sqrtf(length[prop+j]*length[prop+j]+
                                   width[prop+j]*width[prop+j]);
    const float clearance=radius_i+radius_j+.25f;
    const bool same_team=teams[i]==teams[j];
    bool conflict=((closing>.0225f&&closest_dist<clearance)||
                   current_dist<clearance-.08f);
    if(!teammate_intent&&same_team)
      conflict=current_dist<clearance+.05f;
    const float dix=targets[xybase+i*2]-pose[pi];
    const float diy=targets[xybase+i*2+1]-pose[pi+1];
    const float djx=targets[xybase+j*2]-pose[pj];
    const float djy=targets[xybase+j*2+1]-pose[pj+1];
    const bool i_closer=sqrtf(dix*dix+diy*diy)<=sqrtf(djx*djx+djy*djy);
    const int preferred=(same_team&&teammate_intent)?(i_closer?i:j):(i<j?i:j);
    int priority=preferred;
    if(controlled[i]&&!controlled[j])priority=j;
    if(!controlled[i]&&controlled[j])priority=i;
    conflict=conflict&&(controlled[i]||controlled[j]);
    const int old=static_cast<int>(winner[world*15+lane]);
    const bool retained=old>=0&&current_dist<clearance+.35f&&
                        (!same_team||teammate_intent);
    const int selected=retained?old:(conflict?priority:-1);
    pair_winner[lane]=selected;pair_conflict[lane]=conflict?1:0;
    if(active[world])winner[world*15+lane]=selected;
    const bool i_wins=selected==i;
    const int loser=i_wins?j:i;
    pair_loser[lane]=loser;
    const float loser_x=controlled[loser]?command[xybase+loser*2]:0.f;
    const float loser_y=controlled[loser]?command[xybase+loser*2+1]:0.f;
    float sep_x=i_wins?-closest_x:closest_x;
    float sep_y=i_wins?-closest_y:closest_y;
    float sep_norm=sqrtf(sep_x*sep_x+sep_y*sep_y);
    if(sep_norm<.1f){sep_x=i_wins?-rx:rx;sep_y=i_wins?-ry:ry;
      sep_norm=sqrtf(sep_x*sep_x+sep_y*sep_y);}
    const float radial_x=sep_x/fmaxf(sep_norm,1.e-5f);
    const float radial_y=sep_y/fmaxf(sep_norm,1.e-5f);
    float tangent_x=-radial_y,tangent_y=radial_x;
    if(tangent_x*loser_x+tangent_y*loser_y<0.f){
      tangent_x=-tangent_x;tangent_y=-tangent_y;}
    const float desired=fmaxf(.65f,sqrtf(loser_x*loser_x+loser_y*loser_y));
    const float scale=conflict?desired:0.f;
    avoid_x[lane]=scale*(tangent_x+.75f*radial_x);
    avoid_y[lane]=scale*(tangent_y+.75f*radial_y);
  }
  __syncthreads();
  if(lane<6){
    float sum_x=0.f,sum_y=0.f;int count=0;
    for(int p=0;p<15;p++)if(pair_loser[p]==lane&&pair_conflict[p]){
      sum_x+=avoid_x[p];sum_y+=avoid_y[p];++count;
    }
    float out_x=controlled[lane]?command[xybase+lane*2]:0.f;
    float out_y=controlled[lane]?command[xybase+lane*2+1]:0.f;
    if(count>0){
      out_x=sum_x/static_cast<float>(count);
      out_y=sum_y/static_cast<float>(count);
      const float magnitude=sqrtf(out_x*out_x+out_y*out_y);
      const float limit=fminf(1.f,speed[prop+lane]/fmaxf(magnitude,1.e-6f));
      out_x*=limit;out_y*=limit;
    }
    if(!controlled[lane])out_x=out_y=0.f;
    adjusted[xybase+lane*2]=out_x;
    adjusted[xybase+lane*2+1]=out_y;
  }
}

void avoidance_launch(const float* pose,const float* command,const float* targets,
    const float* length,const float* width,const float* speed,
    const bool* controlled,const int64_t* teams,const bool* active,
    int64_t* winner,float* adjusted,int n,bool teammate_intent,hipStream_t stream){
  avoid_contention_kernel<<<dim3(n),dim3(32),0,stream>>>(pose,command,targets,
      length,width,speed,controlled,teams,active,winner,adjusted,n,teammate_intent);
}

__global__ void invalidate_tracks_kernel(const float* pose, const float* length,
    const float* width, const float* track_pos, bool* track_mask,
    float* track_age, const bool* active, int n, int pieces,
    float fuel_radius) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(n) * 6 * pieces;
  if (linear >= total) return;
  const int world = static_cast<int>(linear / (6 * pieces));
  if (!active[world]) return;
  const float x = track_pos[2 * linear];
  const float y = track_pos[2 * linear + 1];
  bool intersects = false;
  for (int robot = 0; robot < 6 && !intersects; ++robot) {
    const int pose_index = world * 18 + robot * 3;
    const int size_index = world * 6 + robot;
    const float dx = x - pose[pose_index];
    const float dy = y - pose[pose_index + 1];
    const float cosine = cosf(pose[pose_index + 2]);
    const float sine = sinf(pose[pose_index + 2]);
    const float longitudinal = dx * cosine + dy * sine;
    const float lateral = fabsf(-dx * sine + dy * cosine);
    intersects =
        longitudinal >= -length[size_index] * .5f - fuel_radius &&
        longitudinal <= length[size_index] * .5f + .35f + fuel_radius &&
        lateral <= width[size_index] * .5f + .075f + fuel_radius;
  }
  if (intersects) {
    track_mask[linear] = false;
    track_age[linear] = INFINITY;
  }
}

void invalidate_tracks_launch(const float* pose, const float* length,
    const float* width, const float* track_pos, bool* track_mask,
    float* track_age, const bool* active, int n, int pieces,
    float fuel_radius, hipStream_t stream) {
  constexpr int threads = 256;
  const int64_t total = static_cast<int64_t>(n) * 6 * pieces;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  invalidate_tracks_kernel<<<dim3(blocks), dim3(threads), 0, stream>>>(pose,
      length, width, track_pos, track_mask, track_age, active, n, pieces,
      fuel_radius);
}

__global__ void footprint_path_clear_kernel(const float* path,
    const float* heading, const float* length, const float* width,
    const float* boxes, bool* clear, int rows, int points, int box_count,
    float field_length, float field_width, float clearance) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const float len = length[row], wid = width[row];
  const float outer = .5f * sqrtf(len * len + wid * wid);
  const float inner = .5f * fminf(len, wid);
  const float half_len = .5f * len, half_wid = .5f * wid;
  bool path_hit = false, wall_hit = false;
  for (int segment = 0; segment < points - 1 && !path_hit && !wall_hit;
       ++segment) {
    const int a = (row * points + segment) * 2;
    const int b = a + 2;
    const float x0 = path[a], y0 = path[a + 1];
    const float dx = path[b] - x0, dy = path[b + 1] - y0;
    const float h0 = heading[row];
    const float dh = 0.f;
    const float steps_f = ceilf(fmaxf(sqrtf(dx * dx + dy * dy),
                                      fabsf(dh) * outer) / .02f);
    const int steps = max(1, min(64, static_cast<int>(steps_f)));
    const float segment_clearance = segment == 0 ? 0.f : clearance;
    for (int sample = 1; sample <= steps && !path_hit && !wall_hit; ++sample) {
      const float fraction = static_cast<float>(sample) / steps;
      const float x = x0 + dx * fraction, y = y0 + dy * fraction;
      const float angle = h0 + dh * fraction;
      const float c = cosf(angle), s = sinf(angle);
      const float ex = .5f * (len * fabsf(c) + wid * fabsf(s));
      const float ey = .5f * (len * fabsf(s) + wid * fabsf(c));
      wall_hit = x < ex || x > field_length - ex ||
                 y < ey || y > field_width - ey;
      for (int box = 0; box < box_count && !path_hit && !wall_hit; ++box) {
        const float* q = boxes + box * 4;
        const float bx = x - q[0], by = y - q[1];
        const float hx = q[2], hy = q[3];
        const float nx = fmaxf(fabsf(bx) - hx, 0.f);
        const float ny = fmaxf(fabsf(by) - hy, 0.f);
        const float separation = sqrtf(nx * nx + ny * ny);
        const float box_outer = sqrtf(hx * hx + hy * hy);
        const bool broad_clear = separation > outer + box_outer + segment_clearance;
        const bool definite_hit = separation <= inner + segment_clearance;
        bool hit = definite_hit;
        if (!broad_clear && !definite_hit) {
          const float signed_axis[4] = {
              bx, by, bx * c + by * s, -bx * s + by * c};
          const float robot_radius[4] = {
              fabsf(c) * half_len + fabsf(s) * half_wid,
              fabsf(s) * half_len + fabsf(c) * half_wid,
              half_len, half_wid};
          const float box_radius[4] = {
              hx, hy, hx * fabsf(c) + hy * fabsf(s),
              hx * fabsf(s) + hy * fabsf(c)};
          float minimum = 1.e30f;
          #pragma unroll
          for (int axis = 0; axis < 4; ++axis)
            minimum = fminf(minimum, robot_radius[axis] + box_radius[axis] -
                                     fabsf(signed_axis[axis]) + segment_clearance);
          hit = minimum >= 0.f;
        }
        path_hit = hit;
      }
    }
  }
  clear[row] = !(path_hit || wall_hit);
}

void footprint_path_clear_launch(const float* path, const float* heading,
    const float* length, const float* width, const float* boxes, bool* clear,
    int rows, int points, int box_count, float field_length,
    float field_width, float clearance, hipStream_t stream) {
  constexpr int threads = 128;
  const int blocks = (rows + threads - 1) / threads;
  footprint_path_clear_kernel<<<dim3(blocks), dim3(threads), 0, stream>>>(
      path, heading, length, width, boxes, clear, rows, points, box_count,
      field_length, field_width, clearance);
}
