// Prototype: one HIP lane owns both robots in a world and runs the complete
// sequential contact solver in Torch order. This file is deliberately separate
// from the production wall-only kernel until full parity is demonstrated.
#include <hip/hip_runtime.h>
#include <cmath>

__device__ __forceinline__ float sign_ref(float x) {
  return static_cast<float>((x > 0.0f) - (x < 0.0f));
}
__device__ __forceinline__ float cross_ref(float ax, float ay, float bx, float by) {
  return ax * by - ay * bx;
}
__device__ __forceinline__ float dot_ref(float ax, float ay, float bx, float by) {
  return ax * bx + ay * by;
}
__device__ __forceinline__ float dot_einsum(float ax, float ay, float bx, float by) {
  // Torch's 2-wide einsum accumulates the second product into the rounded
  // first product on this HIP backend.
  return fmaf(ay, by, ax * bx);
}

__device__ __forceinline__ void static_impulse(
    float* p, float* v, int robot, float point_x, float point_y,
    float nx, float ny, bool contact, float friction,
    const float* length, const float* width, const float* mass,
    const float* yaw_inertia_multiplier) {
  const int row = robot;
  const float rx = point_x - p[row * 3 + 0];
  const float ry = point_y - p[row * 3 + 1];
  const float inertia = (mass[row] * (length[row] * length[row] +
                                      width[row] * width[row]) / 12.0f) *
                         yaw_inertia_multiplier[row];
  const float inv_mass = 1.0f / mass[row];
  const float arm_x = -ry;
  const float arm_y = rx;
  float cv_x = v[row * 3 + 0] + v[row * 3 + 2] * arm_x;
  float cv_y = v[row * 3 + 1] + v[row * 3 + 2] * arm_y;
  const float vn = dot_ref(cv_x, cv_y, nx, ny);
  const float rn = cross_ref(rx, ry, nx, ny);
  const float effective = inv_mass + (rn * rn) / fmaxf(inertia, 1.0e-8f);
  const float jn = (contact && vn < 0.0f) ? (-1.05f * vn) / fmaxf(effective, 1.0e-8f) : 0.0f;
  const float normal_ix = jn * nx;
  const float normal_iy = jn * ny;
  v[row * 3 + 0] += normal_ix * inv_mass;
  v[row * 3 + 1] += normal_iy * inv_mass;
  v[row * 3 + 2] += cross_ref(rx, ry, normal_ix, normal_iy) / fmaxf(inertia, 1.0e-8f);

  cv_x = v[row * 3 + 0] + v[row * 3 + 2] * arm_x;
  cv_y = v[row * 3 + 1] + v[row * 3 + 2] * arm_y;
  const float tx = -ny;
  const float ty = nx;
  const float vt = dot_ref(cv_x, cv_y, tx, ty);
  const float rt = cross_ref(rx, ry, tx, ty);
  const float effective_tangent = inv_mass + (rt * rt) / fmaxf(inertia, 1.0e-8f);
  float jt = -vt / fmaxf(effective_tangent, 1.0e-8f);
  jt = fminf(fmaxf(jt, -friction * jn), friction * jn);
  const float tangent_ix = jt * tx;
  const float tangent_iy = jt * ty;
  v[row * 3 + 0] += tangent_ix * inv_mass;
  v[row * 3 + 1] += tangent_iy * inv_mass;
  v[row * 3 + 2] += cross_ref(rx, ry, tangent_ix, tangent_iy) / fmaxf(inertia, 1.0e-8f);
}

__device__ __forceinline__ void robot_collision(
    float* p, float* v, const float* length, const float* width,
    const float* mass, const float* yaw_inertia_multiplier, float mu,
    bool& robot_contact, bool& opponent_contact) {
  const float theta0 = p[2], theta1 = p[5];
  const float c0 = cosf(theta0), s0 = sinf(theta0);
  const float c1 = cosf(theta1), s1 = sinf(theta1);
  const float ux[2] = {c0, c1}, uy[2] = {s0, s1};
  const float vx[2] = {-s0, -s1}, vy[2] = {c0, c1};
  const float axis_x[4] = {c0, -s0, c1, -s1};
  const float axis_y[4] = {s0, c0, s1, c1};
  float overlap[4];
  const float dx = p[3] - p[0];
  const float dy = p[4] - p[1];
  for (int k = 0; k < 4; ++k) {
    const float signed_k = dot_ref(axis_x[k], axis_y[k], dx, dy);
    const float proj_u0 = fabsf(dot_einsum(axis_x[k], axis_y[k], ux[0], uy[0]));
    const float proj_v0 = fabsf(dot_einsum(axis_x[k], axis_y[k], vx[0], vy[0]));
    const float proj_u1 = fabsf(dot_einsum(axis_x[k], axis_y[k], ux[1], uy[1]));
    const float proj_v1 = fabsf(dot_einsum(axis_x[k], axis_y[k], vx[1], vy[1]));
    const float radius0 = proj_u0 * length[0] / 2.0f + proj_v0 * width[0] / 2.0f;
    const float radius1 = proj_u1 * length[1] / 2.0f + proj_v1 * width[1] / 2.0f;
    const float radii = radius0 + radius1;
    overlap[k] = radii - fabsf(signed_k);
  }
  int axis_index = 0;
  float depth = overlap[0];
  for (int k = 1; k < 4; ++k) {
    if (overlap[k] < depth) { depth = overlap[k]; axis_index = k; }
  }
  const float signed_axis = dot_ref(axis_x[axis_index], axis_y[axis_index], dx, dy);
  const float normal_sign = signed_axis >= 0.0f ? 1.0f : -1.0f;
  const float nx = axis_x[axis_index] * normal_sign;
  const float ny = axis_y[axis_index] * normal_sign;
  const bool valid = depth > 0.0f;
  const float corr = (fmaxf(depth, 0.0f) + 1.0e-4f) * static_cast<float>(valid);
  const float inv0 = 1.0f / mass[0], inv1 = 1.0f / mass[1];
  const float total_inv = inv0 + inv1;
  const float frac0 = inv0 / total_inv, frac1 = inv1 / total_inv;
  p[0] -= (nx * corr) * frac0;
  p[1] -= (ny * corr) * frac0;
  p[3] += (nx * corr) * frac1;
  p[4] += (ny * corr) * frac1;

  // Match Torch's support-point expression order: sign*u*length/2.
  float su = sign_ref(dot_ref(nx, ny, ux[0], uy[0]));
  float sv = sign_ref(dot_ref(nx, ny, vx[0], vy[0]));
  const float s0ux = (su * ux[0]) * length[0] / 2.0f;
  const float s0uy = (su * uy[0]) * length[0] / 2.0f;
  const float s0vx = (sv * vx[0]) * width[0] / 2.0f;
  const float s0vy = (sv * vy[0]) * width[0] / 2.0f;
  const float support0x = (p[0] + s0ux) + s0vx;
  const float support0y = (p[1] + s0uy) + s0vy;
  su = sign_ref(dot_ref(nx, ny, ux[1], uy[1]));
  sv = sign_ref(dot_ref(nx, ny, vx[1], vy[1]));
  const float s1ux = (su * ux[1]) * length[1] / 2.0f;
  const float s1uy = (su * uy[1]) * length[1] / 2.0f;
  const float s1vx = (sv * vx[1]) * width[1] / 2.0f;
  const float s1vy = (sv * vy[1]) * width[1] / 2.0f;
  const float support1x = (p[3] - s1ux) - s1vx;
  const float support1y = (p[4] - s1uy) - s1vy;
  const float cpx = 0.5f * (support0x + support1x);
  const float cpy = 0.5f * (support0y + support1y);
  const float r0x = cpx - p[0], r0y = cpy - p[1];
  const float r1x = cpx - p[3], r1y = cpy - p[4];
  float cv0x = v[0] + v[2] * -r0y;
  float cv0y = v[1] + v[2] * r0x;
  float cv1x = v[3] + v[5] * -r1y;
  float cv1y = v[4] + v[5] * r1x;
  const float relx0 = cv1x - cv0x, rely0 = cv1y - cv0y;
  const float vn = dot_ref(relx0, rely0, nx, ny);
  const float i0 = (mass[0] * (length[0] * length[0] + width[0] * width[0]) / 12.0f) * yaw_inertia_multiplier[0];
  const float i1 = (mass[1] * (length[1] * length[1] + width[1] * width[1]) / 12.0f) * yaw_inertia_multiplier[1];
  const float rn0 = cross_ref(r0x, r0y, nx, ny), rn1 = cross_ref(r1x, r1y, nx, ny);
  const float effective = inv0 + inv1 + (rn0 * rn0) / i0 + (rn1 * rn1) / i1;
  const float jn = (valid && vn < 0.0f) ? (-1.05f * vn) / fmaxf(effective, 1.0e-8f) : 0.0f;
  const float normal_ix = jn * nx, normal_iy = jn * ny;
  v[0] -= normal_ix * inv0; v[1] -= normal_iy * inv0;
  v[3] += normal_ix * inv1; v[4] += normal_iy * inv1;
  v[2] -= jn * rn0 / i0; v[5] += jn * rn1 / i1;
  cv0x = v[0] + v[2] * -r0y; cv0y = v[1] + v[2] * r0x;
  cv1x = v[3] + v[5] * -r1y; cv1y = v[4] + v[5] * r1x;
  const float relx = cv1x - cv0x, rely = cv1y - cv0y;
  const float tx = -ny, ty = nx;
  const float vt = dot_ref(relx, rely, tx, ty);
  const float rt0 = cross_ref(r0x, r0y, tx, ty), rt1 = cross_ref(r1x, r1y, tx, ty);
  const float effective_tangent = inv0 + inv1 + (rt0 * rt0) / i0 + (rt1 * rt1) / i1;
  float jt = -vt / fmaxf(effective_tangent, 1.0e-8f);
  jt = fminf(fmaxf(jt, -mu * jn), mu * jn);
  const float tangent_ix = jt * tx, tangent_iy = jt * ty;
  v[0] -= tangent_ix * inv0; v[1] -= tangent_iy * inv0;
  v[3] += tangent_ix * inv1; v[4] += tangent_iy * inv1;
  v[2] -= jt * rt0 / i0; v[5] += jt * rt1 / i1;
  robot_contact = robot_contact || valid;
  opponent_contact = opponent_contact || valid;
}

__device__ __forceinline__ void wall_contacts(
    float* p, float* v, const float* length, const float* width,
    const float* mass, const float* yaw_inertia_multiplier, float wall_mu,
    bool* wall_contact, float field_length, float field_width) {
  for (int r = 0; r < 2; ++r) {
    const int row = r * 3;
    const float theta = p[row + 2], c = cosf(theta), s = sinf(theta);
    const float hl = length[r] / 2.0f, hw = width[r] / 2.0f;
    const float hx = fabsf(c) * hl + fabsf(s) * hw;
    const float hy = fabsf(s) * hl + fabsf(c) * hw;
    const float penetrations[4] = {hx - p[row], p[row] + hx - field_length,
                                   hy - p[row + 1], p[row + 1] + hy - field_width};
    const int axes[4] = {0, 0, 1, 1};
    const float signs[4] = {1.0f, -1.0f, 1.0f, -1.0f};
    const float ux = c, uy = s, vx = -s, vy = c;
    for (int w = 0; w < 4; ++w) {
      if (!(penetrations[w] > 0.0f)) continue;
      const int axis = axes[w];
      const float sign = signs[w];
      const float nx = axis == 0 ? sign : 0.0f;
      const float ny = axis == 1 ? sign : 0.0f;
      const float correction = fmaxf(penetrations[w], 0.0f) + 1.0e-4f;
      p[row] += nx * correction;
      p[row + 1] += ny * correction;
      const float sw = sign_ref((-nx) * ux + (-ny) * uy);
      const float sv = sign_ref((-nx) * vx + (-ny) * vy);
      const float point_x = (p[row] + (sw * ux) * hl) + (sv * vx) * hw;
      const float point_y = (p[row + 1] + (sw * uy) * hl) + (sv * vy) * hw;
      static_impulse(p, v, r, point_x, point_y, nx, ny, true, wall_mu,
                     length, width, mass, yaw_inertia_multiplier);
      wall_contact[r * 2 + axis] = true;
    }
  }
}

__device__ __forceinline__ void obstacle_collisions(
    float* p, float* v, const float* length, const float* width,
    const float* mass, const float* yaw_inertia_multiplier, float wall_mu,
    const float* obstacles, int obstacle_count, bool& robot_contact,
    bool* field_contact) {
  if (obstacle_count <= 0) return;
  for (int r = 0; r < 2; ++r) {
    const int row = r * 3;
    const float theta = p[row + 2], c = cosf(theta), s = sinf(theta);
    const float ux = c, uy = s, vx = -s, vy = c;
    const float hl = length[r] / 2.0f, hw = width[r] / 2.0f;
    float max_penetration = -1.0f;
    int best = 0;
    float best_nx_local = 0.0f, best_ny_local = 0.0f;
    for (int j = 0; j < obstacle_count; ++j) {
      const float ox = obstacles[j * 3], oy = obstacles[j * 3 + 1];
      const float relx = ox - p[row], rely = oy - p[row + 1];
      const float local_x = relx * ux + rely * uy;
      const float local_y = relx * vx + rely * vy;
      const float closest_x = fmaxf(fminf(local_x, hl), -hl);
      const float closest_y = fmaxf(fminf(local_y, hw), -hw);
      const float delta_x = local_x - closest_x;
      const float delta_y = local_y - closest_y;
      const float distance = sqrtf(delta_x * delta_x + delta_y * delta_y);
      const bool outside = distance > 1.0e-8f;
      const float normal_x_out = -delta_x / fmaxf(distance, 1.0e-8f);
      const float normal_y_out = -delta_y / fmaxf(distance, 1.0e-8f);
      const float gap_x = hl - fabsf(local_x), gap_y = hw - fabsf(local_y);
      const bool use_x = gap_x <= gap_y;
      const float inside_x = local_x >= 0.0f ? -1.0f : 1.0f;
      const float inside_y = local_y >= 0.0f ? -1.0f : 1.0f;
      const float normal_x = outside ? normal_x_out : (use_x ? inside_x : 0.0f);
      const float normal_y = outside ? normal_y_out : (use_x ? 0.0f : inside_y);
      const float radius = obstacles[j * 3 + 2];
      float penetration = outside ? radius - distance : radius + fminf(gap_x, gap_y);
      penetration = fmaxf(penetration, 0.0f);
      if (penetration > max_penetration) {
        max_penetration = penetration;
        best = j;
        best_nx_local = normal_x;
        best_ny_local = normal_y;
      }
    }
    const float depth = max_penetration;
    const bool contact = depth > 0.0f;
    const float nx = best_nx_local * ux + best_ny_local * vx;
    const float ny = best_nx_local * uy + best_ny_local * vy;
    const float circle_x = obstacles[best * 3], circle_y = obstacles[best * 3 + 1];
    const float circle_radius = obstacles[best * 3 + 2];
    const float correction = (depth + 1.0e-4f) * static_cast<float>(contact);
    p[row] += nx * correction;
    p[row + 1] += ny * correction;
    const float sw = sign_ref((-nx) * ux + (-ny) * uy);
    const float sv = sign_ref((-nx) * vx + (-ny) * vy);
    const float robot_point_x = (p[row] + (sw * ux) * hl) + (sv * vx) * hw;
    const float robot_point_y = (p[row + 1] + (sw * uy) * hl) + (sv * vy) * hw;
    const float circle_point_x = circle_x + nx * circle_radius;
    const float circle_point_y = circle_y + ny * circle_radius;
    const float point_x = 0.5f * (robot_point_x + circle_point_x);
    const float point_y = 0.5f * (robot_point_y + circle_point_y);
    static_impulse(p, v, r, point_x, point_y, nx, ny, contact, wall_mu,
                   length, width, mass, yaw_inertia_multiplier);
    robot_contact = robot_contact || contact;
    field_contact[r] = field_contact[r] || contact;
  }
}

__device__ __forceinline__ void field_collisions(
    float* p, float* v, const float* length, const float* width,
    const float* mass, const float* yaw_inertia_multiplier, float wall_mu,
    const float* boxes, int box_count, bool& robot_contact, bool* field_contact) {
  if (box_count <= 0) return;
  for (int r = 0; r < 2; ++r) {
    const int row = r * 3;
    const float theta = p[row + 2], c = cosf(theta), s = sinf(theta);
    const float ac = fabsf(c), as = fabsf(s);
    const float hl = length[r] / 2.0f, hw = width[r] / 2.0f;
    const float axis_x[4] = {1.0f, 0.0f, c, -s};
    const float axis_y[4] = {0.0f, 1.0f, s, c};
    const float robot_r[4] = {ac * hl + as * hw, as * hl + ac * hw, hl, hw};
    float best_depth = -1.0f;
    int best_box = 0;
    float best_pen[4] = {0.f, 0.f, 0.f, 0.f};
    float best_signed[4] = {0.f, 0.f, 0.f, 0.f};
    for (int b = 0; b < box_count; ++b) {
      const float cx = boxes[b * 4], cy = boxes[b * 4 + 1];
      const float hx = boxes[b * 4 + 2], hy = boxes[b * 4 + 3];
      const float dx = p[row] - cx, dy = p[row + 1] - cy;
      float pen[4];
      for (int k = 0; k < 4; ++k) {
        const float signed_k = dot_ref(axis_x[k], axis_y[k], dx, dy);
        float box_r;
        if (k == 0) box_r = hx;
        else if (k == 1) box_r = hy;
        else if (k == 2) box_r = hx * ac + hy * as;
        else box_r = hx * as + hy * ac;
        pen[k] = robot_r[k] + box_r - fabsf(signed_k);
      }
      float d = pen[0];
      for (int k = 1; k < 4; ++k) if (pen[k] < d) d = pen[k];
      if (d >= 0.0f && d > best_depth) {
        best_depth = d;
        best_box = b;
        for (int k = 0; k < 4; ++k) {
          best_pen[k] = pen[k];
          best_signed[k] = dot_ref(axis_x[k], axis_y[k], dx, dy);
        }
      }
    }
    const float depth_raw = best_depth;
    const bool contact = depth_raw >= 0.0f;
    const int box = best_box;
    const float cx = boxes[box * 4], cy = boxes[box * 4 + 1];
    const float hx = boxes[box * 4 + 2], hy = boxes[box * 4 + 3];
    float normal_x[4], normal_y[4], approach[4];
    bool tied[4];
    for (int k = 0; k < 4; ++k) {
      const float sign = best_signed[k] >= 0.0f ? 1.0f : -1.0f;
      normal_x[k] = axis_x[k] * sign;
      normal_y[k] = axis_y[k] * sign;
      approach[k] = dot_ref(normal_x[k], normal_y[k], v[row], v[row + 1]);
      tied[k] = best_pen[k] <= depth_raw + 1.0e-6f && best_pen[k] >= 0.0f;
    }
    float approach_min = INFINITY;
    int approach_axis = 0;
    int penetration_axis = 0;
    for (int k = 0; k < 4; ++k) {
      if (best_pen[k] < best_pen[penetration_axis]) penetration_axis = k;
      const float candidate = tied[k] ? approach[k] : INFINITY;
      if (candidate < approach_min) { approach_min = candidate; approach_axis = k; }
    }
    const int axis_index = approach_min < -1.0e-6f ? approach_axis : penetration_axis;
    const float nx = normal_x[axis_index], ny = normal_y[axis_index];
    const float depth = fmaxf(depth_raw, 0.0f);
    const float correction = (depth + 1.0e-4f) * static_cast<float>(contact);
    p[row] += nx * correction;
    p[row + 1] += ny * correction;
    const float ux = c, uy = s, vx = -s, vy = c;
    const float tx = -ny, ty = nx;
    const float robot_normal = fabsf(dot_ref(nx, ny, ux, uy)) * hl +
                               fabsf(dot_ref(nx, ny, vx, vy)) * hw;
    const float box_normal = fabsf(nx) * hx + fabsf(ny) * hy;
    const float normal_coordinate = 0.5f * (dot_ref(nx, ny, p[row], p[row + 1]) -
        robot_normal + dot_ref(nx, ny, cx, cy) + box_normal);
    const float robot_tangent = fabsf(dot_ref(tx, ty, ux, uy)) * hl +
                                 fabsf(dot_ref(tx, ty, vx, vy)) * hw;
    const float box_tangent = fabsf(tx) * hx + fabsf(ty) * hy;
    const float robot_center = dot_ref(tx, ty, p[row], p[row + 1]);
    const float box_center = dot_ref(tx, ty, cx, cy);
    const float overlap_low = fmaxf(robot_center - robot_tangent, box_center - box_tangent);
    const float overlap_high = fminf(robot_center + robot_tangent, box_center + box_tangent);
    const float tangent_coordinate = 0.5f * (overlap_low + overlap_high);
    const float point_x = nx * normal_coordinate + tx * tangent_coordinate;
    const float point_y = ny * normal_coordinate + ty * tangent_coordinate;
    static_impulse(p, v, r, point_x, point_y, nx, ny, contact, wall_mu,
                   length, width, mass, yaw_inertia_multiplier);
    robot_contact = robot_contact || contact;
    field_contact[r] = field_contact[r] || contact;
  }
}

__global__ void contact_pipeline_kernel(
    float* pose, float* velocity, const float* length, const float* width,
    const float* mass, const float* yaw_inertia_multiplier, const float* mu,
    const float* wall_mu, const bool* active, bool* robot_contact,
    bool* opponent_contact, bool* field_contact, bool* wall_contact,
    const float* obstacles, int obstacle_count, const float* boxes,
    int box_count, int worlds, float field_length, float field_width,
    int iterations, int stage) {
  const int world = blockIdx.x * blockDim.x + threadIdx.x;
  if (world >= worlds || !active[world]) return;
  float p[6], v[6], len[2], wid[2], m[2], inertia_mul[2];
  bool wc[4], fc[2];
  for (int i = 0; i < 6; ++i) { p[i] = pose[world * 6 + i]; v[i] = velocity[world * 6 + i]; }
  for (int r = 0; r < 2; ++r) {
    len[r] = length[world * 2 + r]; wid[r] = width[world * 2 + r];
    m[r] = mass[world * 2 + r]; inertia_mul[r] = yaw_inertia_multiplier[world * 2 + r];
    for (int a = 0; a < 2; ++a) wc[r * 2 + a] = wall_contact[world * 4 + r * 2 + a];
    fc[r] = field_contact[world * 2 + r];
  }
  bool rc = robot_contact[world];
  bool oc = opponent_contact[world];
  for (int it = 0; it < iterations; ++it) {
    if (stage == -1 || stage == 0)
      robot_collision(p, v, len, wid, m, inertia_mul, mu[world], rc, oc);
    if (stage == -1 || stage == 1)
      wall_contacts(p, v, len, wid, m, inertia_mul, wall_mu[world], wc,
                    field_length, field_width);
    if (stage == -1 || stage == 2)
      obstacle_collisions(p, v, len, wid, m, inertia_mul, wall_mu[world],
                          obstacles, obstacle_count, rc, fc);
    if (stage == -1 || stage == 3)
      field_collisions(p, v, len, wid, m, inertia_mul, wall_mu[world],
                       boxes, box_count, rc, fc);
  }
  for (int i = 0; i < 6; ++i) { pose[world * 6 + i] = p[i]; velocity[world * 6 + i] = v[i]; }
  robot_contact[world] = rc;
  opponent_contact[world] = oc;
  for (int r = 0; r < 2; ++r) {
    field_contact[world * 2 + r] = fc[r];
    for (int a = 0; a < 2; ++a) wall_contact[world * 4 + r * 2 + a] = wc[r * 2 + a];
  }
}

void contact_pipeline_launch(float* pose, float* velocity, const float* length,
    const float* width, const float* mass, const float* yaw_inertia_multiplier,
    const float* mu, const float* wall_mu, const bool* active, bool* robot_contact,
    bool* opponent_contact, bool* field_contact, bool* wall_contact,
    const float* obstacles, int obstacle_count, const float* boxes, int box_count,
    int worlds, float field_length, float field_width, int iterations, int stage,
    hipStream_t stream) {
  constexpr int threads = 256;
  const int blocks = (worlds + threads - 1) / threads;
  hipLaunchKernelGGL(contact_pipeline_kernel, dim3(blocks), dim3(threads), 0, stream,
      pose, velocity, length, width, mass, yaw_inertia_multiplier, mu, wall_mu,
      active, robot_contact, opponent_contact, field_contact, wall_contact,
      obstacles, obstacle_count, boxes, box_count, worlds, field_length,
      field_width, iterations, stage);
}
