// Fused wall-contact kernel. One lane owns a single robot in
// a world. It processes the four walls in the same x-min, x-max, y-min,
// y-max order as TensorVectorizedSimulator._walls.
#include <hip/hip_runtime.h>
#include <cmath>

__device__ __forceinline__ float sign_torch(float x) {
  return static_cast<float>((x > 0.0f) - (x < 0.0f));
}

__device__ __forceinline__ float cross2(float ax, float ay, float bx, float by) {
  return ax * by - ay * bx;
}

__device__ __forceinline__ void static_impulse(
    float* vel, float px, float py, float nx, float ny,
    float friction, float mass, float inertia, bool active) {
  const float rx = px;
  const float ry = py;
  const float armx = -ry;
  const float army = rx;
  float cvx = vel[0] + vel[2] * armx;
  float cvy = vel[1] + vel[2] * army;
  const float vn = cvx * nx + cvy * ny;
  const float inv_mass = 1.0f / mass;
  const float rn = cross2(rx, ry, nx, ny);
  const float effective = inv_mass + (rn * rn) / fmaxf(inertia, 1.0e-8f);
  const float jn = (active && vn < 0.0f) ? (-1.05f * vn) / fmaxf(effective, 1.0e-8f) : 0.0f;
  const float inx = jn * nx;
  const float iny = jn * ny;
  vel[0] += inx * inv_mass;
  vel[1] += iny * inv_mass;
  vel[2] += cross2(rx, ry, inx, iny) / fmaxf(inertia, 1.0e-8f);

  cvx = vel[0] + vel[2] * armx;
  cvy = vel[1] + vel[2] * army;
  const float tx = -ny;
  const float ty = nx;
  const float vt = cvx * tx + cvy * ty;
  const float rt = cross2(rx, ry, tx, ty);
  const float effective_tangent = inv_mass + (rt * rt) / fmaxf(inertia, 1.0e-8f);
  float jt = -vt / fmaxf(effective_tangent, 1.0e-8f);
  jt = fminf(fmaxf(jt, -friction * jn), friction * jn);
  const float itx = jt * tx;
  const float ity = jt * ty;
  vel[0] += itx * inv_mass;
  vel[1] += ity * inv_mass;
  vel[2] += cross2(rx, ry, itx, ity) / fmaxf(inertia, 1.0e-8f);
}

__global__ void wall_contacts_kernel(
    float* pose, float* velocity, const float* length, const float* width,
    const float* mass, const float* yaw_inertia_multiplier, const float* wall_mu,
    const bool* active_world, bool* wall_contact, int worlds, int robots,
    float field_length, float field_width) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= worlds * robots) return;
  const int world = row / robots;
  const int robot = row % robots;
  if (!active_world[world]) return;

  float* p = pose + row * 3;
  float* v = velocity + row * 3;
  const float hl = length[row] * 0.5f;
  const float hw = width[row] * 0.5f;
  const float theta = p[2];
  const float c = cosf(theta);
  const float s = sinf(theta);
  const float hx = fabsf(c) * hl + fabsf(s) * hw;
  const float hy = fabsf(s) * hl + fabsf(c) * hw;
  // The reference materializes all four penetrations before resolving any
  // contact; preserve those values while using the progressively corrected
  // pose for support-point and impulse calculations.
  const float penetration[4] = {
      hx - p[0], p[0] + hx - field_length,
      hy - p[1], p[1] + hy - field_width};
  const int axis[4] = {0, 0, 1, 1};
  const float normal_sign[4] = {1.0f, -1.0f, 1.0f, -1.0f};
  const float ux = c, uy = s, vx = -s, vy = c;
  const float inertia = (mass[row] * (length[row] * length[row] +
                                      width[row] * width[row]) / 12.0f) *
                         yaw_inertia_multiplier[row];

  for (int wall = 0; wall < 4; ++wall) {
    const int ax = axis[wall];
    const float sign = normal_sign[wall];
    const bool contact = penetration[wall] > 0.0f;
    if (!contact) continue;

    const float nx = ax == 0 ? sign : 0.0f;
    const float ny = ax == 1 ? sign : 0.0f;
    const float correction = fmaxf(penetration[wall], 0.0f) + 1.0e-4f;
    p[0] += nx * correction;
    p[1] += ny * correction;

    const float twx = -nx, twy = -ny;
    const float support_u = sign_torch(twx * ux + twy * uy);
    const float support_v = sign_torch(twx * vx + twy * vy);
    const float point_x = p[0] + support_u * ux * hl + support_v * vx * hw;
    const float point_y = p[1] + support_u * uy * hl + support_v * vy * hw;
    static_impulse(v, point_x - p[0], point_y - p[1], nx, ny,
                   wall_mu[world], mass[row], inertia, true);
    wall_contact[row * 2 + ax] = true;
  }
}

void wall_contacts_launch(float* pose, float* velocity, const float* length,
    const float* width, const float* mass, const float* yaw_inertia_multiplier,
    const float* wall_mu, const bool* active_world, bool* wall_contact,
    int worlds, int robots, float field_length, float field_width, hipStream_t stream) {
  constexpr int threads = 256;
  const int blocks = (worlds * robots + threads - 1) / threads;
  hipLaunchKernelGGL(wall_contacts_kernel, dim3(blocks), dim3(threads), 0, stream,
      pose, velocity, length, width, mass, yaw_inertia_multiplier, wall_mu,
      active_world, wall_contact, worlds, robots, field_length, field_width);
}
