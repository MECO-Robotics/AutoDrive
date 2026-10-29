#include <hip/hip_runtime.h>
#include <math.h>

__global__ void shortcut_kernel(const float* path, const int64_t* lengths,
                                const bool* blocked, float* output, int64_t* out_length,
                                int batch, int points, int nx, int ny,
                                int samples, float inv_resolution, float inv_half_resolution) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  const float* src = path + static_cast<int64_t>(b) * points * 2;
  float* dst = output + static_cast<int64_t>(b) * points * 2;
  const bool* grid = blocked + static_cast<int64_t>(b) * nx * ny;
  bool valid = lengths[b] >= 0 && lengths[b] <= points;
  for (int i = 0; i < points * 2; ++i) valid = valid && isfinite(src[i]);
  if (!valid) {
    for (int i = 0; i < points * 2; ++i) dst[i] = 0.0f;
    out_length[b] = -1;
    return;
  }
  for (int i = 0; i < points * 2; ++i) dst[i] = 0.0f;
  dst[0] = src[0]; dst[1] = src[1];
  int len = lengths[b] > 0 ? 1 : 0;
  int anchor = 0;
  for (int i = 1; i < points - 1; ++i) {
    bool hit = false;
    if (i < lengths[b] - 1) {
      float ax = src[anchor*2], ay = src[anchor*2+1];
      float dx = src[(i+1)*2] - ax, dy = src[(i+1)*2+1] - ay;
      float raw_steps = ceilf(fmaxf(fabsf(dx), fabsf(dy)) * inv_half_resolution);
      if (!isfinite(dx) || !isfinite(dy) || !isfinite(raw_steps)) {
        out_length[b] = -1;
        return;
      }
      int steps = static_cast<int>(fminf(fmaxf(raw_steps, 1.0f), static_cast<float>(samples)));
      for (int j = 1; j <= steps; ++j) {
        float fraction = static_cast<float>(j) / static_cast<float>(steps);
        float x = ax + dx * fraction, y = ay + dy * fraction;
        float scaled_x = x * inv_resolution, scaled_y = y * inv_resolution;
        if (!isfinite(scaled_x) || !isfinite(scaled_y)) {
          out_length[b] = -1;
          return;
        }
        int ix = static_cast<int>(floorf(fminf(fmaxf(scaled_x, 0.0f), static_cast<float>(nx-1))));
        int iy = static_cast<int>(floorf(fminf(fmaxf(scaled_y, 0.0f), static_cast<float>(ny-1))));
        if (grid[ix*ny+iy]) { hit = true; break; }
      }
    }
    int slot = min(len, points-1);
    dst[slot*2] = hit ? src[i*2] : 0.0f;
    dst[slot*2+1] = hit ? src[i*2+1] : 0.0f;
    if (hit) { anchor = i; ++len; }
  }
  bool has_route = lengths[b] > 1;
  int last = max(static_cast<int>(lengths[b])-1, 0);
  int slot = min(len, points-1);
  dst[slot*2] = has_route ? src[last*2] : 0.0f;
  dst[slot*2+1] = has_route ? src[last*2+1] : 0.0f;
  len = has_route ? min(len+1, points) : 1;
  float x = dst[(len-1)*2], y = dst[(len-1)*2+1];
  for (int i = len; i < points; ++i) { dst[i*2] = x; dst[i*2+1] = y; }
  out_length[b] = len;
}

void shortcut_launch(const float* path, const int64_t* lengths, const bool* blocked,
                     float* output, int64_t* out_length, int batch, int points,
                     int nx, int ny, int samples, float inv_resolution, float inv_half_resolution, hipStream_t stream) {
  hipLaunchKernelGGL(shortcut_kernel, dim3((batch+63)/64), dim3(64), 0, stream,
                     path, lengths, blocked, output, out_length, batch, points,
                     nx, ny, samples, inv_resolution, inv_half_resolution);
}
