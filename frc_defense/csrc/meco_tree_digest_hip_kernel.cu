#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t sha_rotr(uint32_t value, uint32_t shift) {
  return (value >> shift) | (value << (32 - shift));
}

__global__ void plan_a_sha256_bytes_kernel(const uint8_t* input, int64_t length,
                                           uint8_t* output, int64_t chunk_size) {
  const int64_t chunk = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t count = (length + chunk_size - 1) / chunk_size;
  if (chunk >= count) return;
  const int64_t offset = chunk * chunk_size;
  input += offset;
  length = min(chunk_size, length - offset);
  output += chunk * 32;
  const uint32_t k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
  uint32_t state[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                       0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  const int64_t padded = ((length + 9 + 63) / 64) * 64;
  const uint64_t bit_length = static_cast<uint64_t>(length) * 8;
  for (int64_t base = 0; base < padded; base += 64) {
    uint32_t words[64];
    for (int i = 0; i < 16; ++i) {
      uint32_t value = 0;
      for (int j = 0; j < 4; ++j) {
        const int64_t position = base + i * 4 + j;
        uint8_t byte = 0;
        if (position < length) byte = input[position];
        else if (position == length) byte = 0x80;
        else if (position >= padded - 8) {
          const int shift = static_cast<int>((padded - 1 - position) * 8);
          byte = static_cast<uint8_t>((bit_length >> shift) & 0xff);
        }
        value = (value << 8) | byte;
      }
      words[i] = value;
    }
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = sha_rotr(words[i - 15], 7) ^ sha_rotr(words[i - 15], 18) ^ (words[i - 15] >> 3);
      const uint32_t s1 = sha_rotr(words[i - 2], 17) ^ sha_rotr(words[i - 2], 19) ^ (words[i - 2] >> 10);
      words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }
    uint32_t a=state[0], b=state[1], c=state[2], d=state[3];
    uint32_t e=state[4], f=state[5], g=state[6], h=state[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t s1 = sha_rotr(e, 6) ^ sha_rotr(e, 11) ^ sha_rotr(e, 25);
      const uint32_t choice = (e & f) ^ ((~e) & g);
      const uint32_t temp1 = h + s1 + choice + k[i] + words[i];
      const uint32_t s0 = sha_rotr(a, 2) ^ sha_rotr(a, 13) ^ sha_rotr(a, 22);
      const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t temp2 = s0 + majority;
      h=g; g=f; f=e; e=d+temp1; d=c; c=b; b=a; a=temp1+temp2;
    }
    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
  }
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 4; ++j)
      output[i * 4 + j] = static_cast<uint8_t>((state[i] >> (24 - j * 8)) & 0xff);
}

void plan_a_sha256_chunks_launch(const uint8_t* input, int64_t length,
    int64_t chunk_size, uint8_t* output, hipStream_t stream) {
  const int64_t count = (length + chunk_size - 1) / chunk_size;
  hipLaunchKernelGGL(plan_a_sha256_bytes_kernel, dim3((count + 127) / 128),
      dim3(128), 0, stream, input, length, output, chunk_size);
}
