#include <hip/hip_runtime.h>

namespace {

__global__ void piece_occlusion_kernel(const float* pose_xy,
                                       const float* segment,
                                       const float* obstacles,
                                       const bool* eligible,
                                       bool* output,
                                       int pieces,
                                       int obstacle_count,
                                       int64_t total) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  if (!eligible[index]) {
    output[index] = false;
    return;
  }

  const int world = static_cast<int>(index / pieces);
  const float* pose = pose_xy + static_cast<int64_t>(world) * 2;
  const float* delta = segment + index * 2;
  // Match Torch's square-then-sum denominator order. Explicit round-to-nearest
  // operations plus --ffp-contract=off avoid contraction across the predicate.
  const float dx = delta[0];
  const float dy = delta[1];
  float denominator = __fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy));
  if (denominator < 1.0e-8f) denominator = 1.0e-8f;

  bool blocked = false;
  for (int obstacle = 0; obstacle < obstacle_count; ++obstacle) {
    const float* circle = obstacles + static_cast<int64_t>(obstacle) * 3;
    const float radius = __fadd_rn(circle[2], 0.03f);
    if (!(radius > 0.0f)) continue;

    const float rel_x = __fadd_rn(circle[0], -pose[0]);
    const float rel_y = __fadd_rn(circle[1], -pose[1]);
    const float dot = __fadd_rn(__fmul_rn(rel_x, dx), __fmul_rn(rel_y, dy));
    const float t = __fdiv_rn(dot, denominator);
    if (!(t > 0.02f && t < 0.98f)) continue;
    const float clamped_t = fminf(1.0f, fmaxf(0.0f, t));
    const float closest_x = __fadd_rn(pose[0], __fmul_rn(clamped_t, dx));
    const float closest_y = __fadd_rn(pose[1], __fmul_rn(clamped_t, dy));
    const float distance_x = __fadd_rn(closest_x, -circle[0]);
    const float distance_y = __fadd_rn(closest_y, -circle[1]);
    const float distance_sq = __fadd_rn(__fmul_rn(distance_x, distance_x),
                                        __fmul_rn(distance_y, distance_y));
    if (sqrtf(distance_sq) <= radius) {
      blocked = true;
      break;
    }
  }
  output[index] = blocked;
}

__global__ void visibility_kernel(const float* pose, const float* pieces_xy,
                                  const bool* piece_active,
                                  const int64_t* piece_owner, const bool* active,
                                  const float* obstacles, const float* other_xy,
                                  const float* other_radius, bool* output,
                                  int pieces, int obstacle_count, float range_m,
                                  float half_fov, bool full_fov, int64_t total) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int world = static_cast<int>(index / pieces);
  if (!active[world] || !piece_active[index] || piece_owner[index] >= 0) {
    output[index] = false;
    return;
  }
  const float* robot = pose + static_cast<int64_t>(world) * 3;
  const float* point = pieces_xy + index * 2;
  const float dx = __fadd_rn(point[0], -robot[0]);
  const float dy = __fadd_rn(point[1], -robot[1]);
  const float distance_sq = __fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy));
  if (!(sqrtf(distance_sq) <= range_m)) {
    output[index] = false;
    return;
  }
  if (!full_fov) {
    const float bearing = atan2f(dy, dx);
    const float difference = __fadd_rn(bearing, -robot[2]);
    const float angle = fabsf(atan2f(sinf(difference), cosf(difference)));
    if (!(angle <= half_fov)) {
      output[index] = false;
      return;
    }
  }

  const float denominator_raw = __fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy));
  const float denominator = fmaxf(denominator_raw, 1.0e-8f);
  for (int obstacle = 0; obstacle < obstacle_count; ++obstacle) {
    const float* circle = obstacles + static_cast<int64_t>(obstacle) * 3;
    const float radius = __fadd_rn(circle[2], 0.03f);
    if (!(radius > 0.0f)) continue;
    const float rel_x = __fadd_rn(circle[0], -robot[0]);
    const float rel_y = __fadd_rn(circle[1], -robot[1]);
    const float dot = __fadd_rn(__fmul_rn(rel_x, dx), __fmul_rn(rel_y, dy));
    const float t = __fdiv_rn(dot, denominator);
    if (!(t > 0.02f && t < 0.98f)) continue;
    const float clamped_t = fminf(1.0f, fmaxf(0.0f, t));
    const float closest_x = __fadd_rn(robot[0], __fmul_rn(clamped_t, dx));
    const float closest_y = __fadd_rn(robot[1], __fmul_rn(clamped_t, dy));
    const float distance_x = __fadd_rn(closest_x, -circle[0]);
    const float distance_y = __fadd_rn(closest_y, -circle[1]);
    const float closest_sq = __fadd_rn(__fmul_rn(distance_x, distance_x),
                                       __fmul_rn(distance_y, distance_y));
    if (sqrtf(closest_sq) <= radius) {
      output[index] = false;
      return;
    }
  }

  const float* opponent = other_xy + static_cast<int64_t>(world) * 2;
  const float rel_x = __fadd_rn(opponent[0], -robot[0]);
  const float rel_y = __fadd_rn(opponent[1], -robot[1]);
  const float dot = __fadd_rn(__fmul_rn(rel_x, dx), __fmul_rn(rel_y, dy));
  const float t = __fdiv_rn(dot, denominator);
  if (t > 0.02f && t < 0.98f) {
    const float clamped_t = fminf(1.0f, fmaxf(0.0f, t));
    const float closest_x = __fadd_rn(robot[0], __fmul_rn(clamped_t, dx));
    const float closest_y = __fadd_rn(robot[1], __fmul_rn(clamped_t, dy));
    const float distance_x = __fadd_rn(closest_x, -opponent[0]);
    const float distance_y = __fadd_rn(closest_y, -opponent[1]);
    const float closest_sq = __fadd_rn(__fmul_rn(distance_x, distance_x),
                                       __fmul_rn(distance_y, distance_y));
    if (sqrtf(closest_sq) <= other_radius[world]) {
      output[index] = false;
      return;
    }
  }
  output[index] = true;
}

__global__ void visibility_3v3_kernel(const float* pose, const float* pieces_xy,
                                      const bool* piece_active,
                                      const int64_t* piece_owner, const bool* active,
                                      const float* obstacles,
                                      const float* robot_radius, bool* output,
                                      int robots, int pieces, int obstacle_count,
                                      float range_m, float half_fov, bool full_fov,
                                      int64_t total) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int64_t per_world = static_cast<int64_t>(robots) * pieces;
  const int world = static_cast<int>(index / per_world);
  const int within_world = static_cast<int>(index - static_cast<int64_t>(world) * per_world);
  const int robot_id = within_world / pieces;
  const int piece_id = within_world - robot_id * pieces;
  const int64_t piece_index = static_cast<int64_t>(world) * pieces + piece_id;
  if (!active[world] || !piece_active[piece_index] || piece_owner[piece_index] >= 0) {
    output[index] = false;
    return;
  }

  const float* robot = pose + static_cast<int64_t>(world) * robots * 3 + robot_id * 3;
  const float* point = pieces_xy + piece_index * 2;
  const float dx = __fadd_rn(point[0], -robot[0]);
  const float dy = __fadd_rn(point[1], -robot[1]);
  const float distance_sq = __fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy));
  if (!(sqrtf(distance_sq) <= range_m)) {
    output[index] = false;
    return;
  }
  if (!full_fov) {
    const float bearing = atan2f(dy, dx);
    const float difference = __fadd_rn(bearing, -robot[2]);
    const float angle = fabsf(atan2f(sinf(difference), cosf(difference)));
    if (!(angle <= half_fov)) {
      output[index] = false;
      return;
    }
  }

  const float denominator_raw = __fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy));
  const float denominator = fmaxf(denominator_raw, 1.0e-8f);
  for (int obstacle = 0; obstacle < obstacle_count; ++obstacle) {
    const float* circle = obstacles + static_cast<int64_t>(obstacle) * 3;
    const float radius = __fadd_rn(circle[2], 0.03f);
    if (!(radius > 0.0f)) continue;
    const float rel_x = __fadd_rn(circle[0], -robot[0]);
    const float rel_y = __fadd_rn(circle[1], -robot[1]);
    const float dot = __fadd_rn(__fmul_rn(rel_x, dx), __fmul_rn(rel_y, dy));
    const float t = __fdiv_rn(dot, denominator);
    if (!(t > 0.02f && t < 0.98f)) continue;
    const float clamped_t = fminf(1.0f, fmaxf(0.0f, t));
    const float closest_x = __fadd_rn(robot[0], __fmul_rn(clamped_t, dx));
    const float closest_y = __fadd_rn(robot[1], __fmul_rn(clamped_t, dy));
    const float distance_x = __fadd_rn(closest_x, -circle[0]);
    const float distance_y = __fadd_rn(closest_y, -circle[1]);
    const float closest_sq = __fadd_rn(__fmul_rn(distance_x, distance_x),
                                       __fmul_rn(distance_y, distance_y));
    if (sqrtf(closest_sq) <= radius) {
      output[index] = false;
      return;
    }
  }

  for (int peer = 0; peer < robots; ++peer) {
    if (peer == robot_id) continue;
    const float* other = pose + static_cast<int64_t>(world) * robots * 3 + peer * 3;
    const float rel_x = __fadd_rn(other[0], -robot[0]);
    const float rel_y = __fadd_rn(other[1], -robot[1]);
    const float dot = __fadd_rn(__fmul_rn(rel_x, dx), __fmul_rn(rel_y, dy));
    const float t = __fdiv_rn(dot, denominator);
    if (!(t > 0.02f && t < 0.98f)) continue;
    const float clamped_t = fminf(1.0f, fmaxf(0.0f, t));
    const float closest_x = __fadd_rn(robot[0], __fmul_rn(clamped_t, dx));
    const float closest_y = __fadd_rn(robot[1], __fmul_rn(clamped_t, dy));
    const float distance_x = __fadd_rn(closest_x, -other[0]);
    const float distance_y = __fadd_rn(closest_y, -other[1]);
    const float closest_sq = __fadd_rn(__fmul_rn(distance_x, distance_x),
                                       __fmul_rn(distance_y, distance_y));
    const float radius = robot_radius[static_cast<int64_t>(world) * robots + peer];
    if (sqrtf(closest_sq) <= radius) {
      output[index] = false;
      return;
    }
  }
  output[index] = true;
}

// One block handles one camera view. Threads project balls and calculate
// adjacent depth-ordered overlap in parallel; thread zero emits connected
// angular clusters in one segmented pass without Torch scatter dispatches.
__global__ void angular_cluster_kernel(
    const float* pose,const float* positions,const float* velocities,
    const bool* visible,const float* input_fraction,const float* input_spatial,
    const float* input_angular,const bool* input_merged,bool fuse_only,
    float* out_pos,float* out_vel,bool* out_valid,
    float* out_fraction,float* out_spatial,float* out_angular,bool* out_merged,
    int pieces,float diameter,float half_fov,int64_t rows) {
  const int row=static_cast<int>(blockIdx.x);
  if(row>=rows) return;
  const int i=static_cast<int>(threadIdx.x);
  extern __shared__ float work[];
  float* distance=work;
  float* lo=distance+pieces;
  float* hi=lo+pieces;
  float* fraction=hi+pieces;
  float* joins=fraction+pieces;
  constexpr int bins=256;
  float* bin_depth=joins+pieces;
  const int padded=1<<static_cast<int>(ceilf(log2f(static_cast<float>(pieces))));
  float* sort_key=bin_depth+bins;
  int* sorted_index=reinterpret_cast<int*>(sort_key+padded);
  const float* camera=pose+static_cast<int64_t>(row)*3;
  const int64_t base=static_cast<int64_t>(row)*pieces;
  for(int j=i;j<pieces;j+=blockDim.x) {
    const int64_t k=base+j;
    const float dx=positions[k*2]-camera[0];
    const float dy=positions[k*2+1]-camera[1];
    const float d=sqrtf(dx*dx+dy*dy);
    const float bearing=atan2f(dy,dx)-camera[2];
    const float wrapped=atan2f(sinf(bearing),cosf(bearing));
    const float half=fuse_only?input_angular[k]*.5f:
        atan2f(diameter*.5f,fmaxf(d,1.0e-4f));
    distance[j]=d;
    lo[j]=fmaxf(-half_fov,wrapped-half);
    hi[j]=fminf(half_fov,wrapped+half);
    fraction[j]=visible[k]?1.0f:0.0f; joins[j]=0.0f;
    sort_key[j]=visible[k]?wrapped:half_fov+1.0f; sorted_index[j]=j;
  }
  for(int j=i+pieces;j<padded;j+=blockDim.x) {
    sort_key[j]=INFINITY; sorted_index[j]=j;
  }
  __syncthreads();
  for(int k=2;k<=padded;k<<=1) {
    for(int stride=k>>1;stride>0;stride>>=1) {
      for(int j=i;j<padded;j+=blockDim.x) {
        const int peer=j^stride;
        if(peer<=j) continue;
        const bool ascending=(j&k)==0;
        const bool swap=ascending?(sort_key[j]>sort_key[peer]):
                                   (sort_key[j]<sort_key[peer]);
        if(swap) {
          const float key=sort_key[j]; sort_key[j]=sort_key[peer];
          sort_key[peer]=key;
          const int index=sorted_index[j]; sorted_index[j]=sorted_index[peer];
          sorted_index[peer]=index;
        }
      }
      __syncthreads();
    }
  }
  for(int b=i;b<bins;b+=blockDim.x) bin_depth[b]=INFINITY;
  __syncthreads();
  const float bin_width=(2.0f*half_fov)/bins;
  for(int j=i;j<pieces && !fuse_only;j+=blockDim.x) {
    const int src=sorted_index[j];
    if(!visible[base+src]) continue;
    const float a=fmaxf(-half_fov,lo[src]);
    const float z=fminf(half_fov,hi[src]);
    int first=static_cast<int>(floorf((a+half_fov)/bin_width));
    int last=static_cast<int>(floorf((z+half_fov)/bin_width));
    first=max(0,min(bins-1,first)); last=max(0,min(bins-1,last));
    for(int b=first;b<=last;b++) atomicMin(
        reinterpret_cast<unsigned int*>(&bin_depth[b]),__float_as_uint(distance[src]));
  }
  __syncthreads();
  for(int j=i;j<pieces;j+=blockDim.x) {
    const int src=sorted_index[j];
    const int64_t k=base+src;
    if(!visible[k]) { fraction[j]=0.0f; continue; }
    if(fuse_only) { fraction[j]=input_fraction[k]; continue; }
    const float a=fmaxf(-half_fov,lo[src]);
    const float z=fminf(half_fov,hi[src]);
    int first=static_cast<int>(floorf((a+half_fov)/bin_width));
    int last=static_cast<int>(floorf((z+half_fov)/bin_width));
    first=max(0,min(bins-1,first)); last=max(0,min(bins-1,last));
    int seen=0,total=0;
    for(int b=first;b<=last;b++) {
      ++total;
      if(bin_depth[b]>=distance[src]-1.0e-4f) ++seen;
    }
    fraction[j]=total?static_cast<float>(seen)/total:0.0f;
  }
  __syncthreads();
  for(int j=i;j<pieces;j+=blockDim.x) {
    const int src=sorted_index[j];
    const int prev=j?sorted_index[j-1]:src;
    if(j==0 || !visible[base+src] || !visible[base+prev]) { joins[j]=0.0f; continue; }
    const float overlap=fminf(hi[src],hi[prev])-fmaxf(lo[src],lo[prev]);
    const bool same_depth=fabsf(distance[src]-distance[prev])<=.20f;
    joins[j]=(overlap>0.0f && same_depth && fraction[j]>.05f &&
              fraction[j-1]>.05f)?1.0f:0.0f;
  }
  __syncthreads();
  for(int j=i;j<pieces;j+=blockDim.x) {
    const int64_t k=base+j;
    out_valid[k]=false; out_merged[k]=false; out_fraction[k]=0.0f;
    out_pos[k*2]=out_pos[k*2+1]=0.0f;
    out_vel[k*2]=out_vel[k*2+1]=0.0f;
    out_spatial[k*2]=out_spatial[k*2+1]=0.0f; out_angular[k]=0.0f;
  }
  __syncthreads();
  if(i!=0) return;
  int group=-1; bool in_group=false; float denom=0.0f, members=0.0f;
  bool inherited_merged=false; float group_confidence=0.0f;
  float sx=0.0f,sy=0.0f,vx=0.0f,vy=0.0f;
  float minx=INFINITY,miny=INFINITY,maxx=-INFINITY,maxy=-INFINITY;
  float minang=INFINITY,maxang=-INFINITY;
  auto emit=[&]() {
    if(!in_group) return;
    if(denom>0.0f && group<pieces) {
      const int64_t out=base+group;
      out_pos[out*2]=sx/denom; out_pos[out*2+1]=sy/denom;
      out_vel[out*2]=vx/denom; out_vel[out*2+1]=vy/denom;
      out_valid[out]=true;
      out_spatial[out*2]=maxx-minx+(fuse_only?0.0f:diameter);
      out_spatial[out*2+1]=maxy-miny+(fuse_only?0.0f:diameter);
      out_angular[out]=fmaxf(0.0f,maxang-minang);
      out_merged[out]=inherited_merged || members>1.0f;
      out_fraction[out]=fuse_only?fminf(1.0f,group_confidence):
          fminf(1.0f,denom/fmaxf(members,1.0f));
    }
  };
  for(int j=0;j<pieces;j++) {
    const int src=sorted_index[j];
    const int64_t k=base+src;
    if(!visible[k]) { emit(); in_group=false; continue; }
    if(!in_group || (j>0 && joins[j]==0.0f)) {
      emit(); ++group; in_group=true; denom=members=sx=sy=vx=vy=0.0f;
      minx=miny=minang=INFINITY; maxx=maxy=maxang=-INFINITY;
      inherited_merged=false; group_confidence=0.0f;
    }
    const float f=fraction[j];
    if(f<=.05f) continue;
    const float x=positions[k*2],y=positions[k*2+1];
    denom+=f; members+=1.0f;
    sx+=x*f; sy+=y*f; vx+=velocities[k*2]*f; vy+=velocities[k*2+1]*f;
    if(fuse_only) {
      minx=fminf(minx,x-input_spatial[k*2]*.5f);
      miny=fminf(miny,y-input_spatial[k*2+1]*.5f);
      maxx=fmaxf(maxx,x+input_spatial[k*2]*.5f);
      maxy=fmaxf(maxy,y+input_spatial[k*2+1]*.5f);
      const float center=(lo[src]+hi[src])*.5f;
      const float half=input_angular[k]*.5f;
      minang=fminf(minang,center-half); maxang=fmaxf(maxang,center+half);
      group_confidence=fmaxf(group_confidence,input_fraction[k]);
      inherited_merged |= input_merged[k];
    } else {
      minx=fminf(minx,x); miny=fminf(miny,y);
      maxx=fmaxf(maxx,x); maxy=fmaxf(maxy,y);
      minang=fminf(minang,lo[src]); maxang=fmaxf(maxang,hi[src]);
      group_confidence=fmaxf(group_confidence,f);
    }
  }
  emit();
}

struct FusedDetection {
  float x,y,vx,vy,confidence,extent_x,extent_y,angular,bearing;
  int merged;
};

__global__ void stereo_fusion_pack_kernel(
    const float* pose,const float* positions,const float* velocities,
    const bool* visible,const float* confidence,const float* spatial,
    const float* angular,const bool* merged,float* out_pos,float* out_vel,
    bool* out_valid,float* out_confidence,float* out_spatial,
    float* out_angular,bool* out_merged,int pieces,int padded,
    int64_t rows) {
  const int row=static_cast<int>(blockIdx.x);
  if(row>=rows) return;
  const int lane=threadIdx.x;
  extern __shared__ unsigned char storage[];
  __shared__ int group_count[2];
  float* keys=reinterpret_cast<float*>(storage);
  int* sources=reinterpret_cast<int*>(keys+padded);
  FusedDetection* front=reinterpret_cast<FusedDetection*>(sources+padded);
  FusedDetection* rear=front+pieces;
  const float* observer=pose+static_cast<int64_t>(row)*3;
  const int64_t input_row=static_cast<int64_t>(row)*4*pieces;
  const int64_t output_row=static_cast<int64_t>(row)*pieces;

  for(int i=lane;i<pieces;i+=blockDim.x) {
    const int64_t k=output_row+i;
    out_valid[k]=false; out_merged[k]=false; out_confidence[k]=0.f;
    out_pos[k*2]=out_pos[k*2+1]=0.f;
    out_vel[k*2]=out_vel[k*2+1]=0.f;
    out_spatial[k*2]=out_spatial[k*2+1]=0.f; out_angular[k]=0.f;
  }
  __syncthreads();

  if(lane==0) { group_count[0]=0; group_count[1]=0; }
  __syncthreads();
  for(int side=0;side<2;side++) {
    const float heading=observer[2]+(side?3.14159265358979323846f:0.f);
    for(int i=lane;i<2*pieces;i+=blockDim.x) {
      int camera=side*2+(i>=pieces?1:0);
      int slot=i%pieces;
      int source=camera*pieces+slot;
      int64_t k=input_row+source;
      const float dx=positions[k*2]-observer[0];
      const float dy=positions[k*2+1]-observer[1];
      float b=atan2f(dy,dx)-heading;
      b=atan2f(sinf(b),cosf(b));
      keys[i]=visible[k]?b:INFINITY;
      sources[i]=source;
    }
    for(int i=lane+2*pieces;i<padded;i+=blockDim.x) {
      keys[i]=INFINITY; sources[i]=i;
    }
    __syncthreads();
    for(int k=2;k<=padded;k<<=1) {
      for(int stride=k>>1;stride>0;stride>>=1) {
        for(int i=lane;i<padded;i+=blockDim.x) {
          const int peer=i^stride;
          if(peer<=i) continue;
          const bool ascending=(i&k)==0;
          const bool swap=ascending?(keys[i]>keys[peer]):(keys[i]<keys[peer]);
          if(swap) {
            const float key=keys[i]; keys[i]=keys[peer]; keys[peer]=key;
            const int source=sources[i]; sources[i]=sources[peer];
            sources[peer]=source;
          }
        }
        __syncthreads();
      }
    }
    if(lane==0) {
      FusedDetection* output=side?rear:front;
      int groups=0; bool in_group=false; int members=0; bool was_merged=false;
      float weight_sum=0.f,sx=0.f,sy=0.f,svx=0.f,svy=0.f,conf=0.f;
      float minx=INFINITY,miny=INFINITY,maxx=-INFINITY,maxy=-INFINITY;
      float minang=INFINITY,maxang=-INFINITY,previous_hi=-INFINITY;
      float previous_distance=INFINITY;
      auto emit=[&]() {
        if(!in_group || weight_sum<=0.f || groups>=pieces) return;
        FusedDetection& d=output[groups++];
        d.x=sx/weight_sum; d.y=sy/weight_sum;
        d.vx=svx/weight_sum; d.vy=svy/weight_sum;
        d.confidence=conf; d.extent_x=maxx-minx; d.extent_y=maxy-miny;
        d.angular=maxang-minang; d.bearing=(minang+maxang)*.5f;
        d.merged=was_merged || members>1;
      };
      for(int i=0;i<2*pieces;i++) {
        const int source=sources[i];
        const int camera=source/pieces;
        const int slot=source%pieces;
        const int64_t k=input_row+source;
        if(keys[i]==INFINITY) break;
        const float d=keys[i];
        const float half=angular[k]*.5f;
        const float lo=d-half,hi=d+half;
        const float range=hypotf(positions[k*2]-observer[0],
                                 positions[k*2+1]-observer[1]);
        const bool joins=in_group && lo<=previous_hi &&
            fabsf(range-previous_distance)<=.20f;
        if(!joins) {
          emit(); in_group=true; members=0; was_merged=false;
          weight_sum=sx=sy=svx=svy=conf=0.f;
          minx=miny=minang=INFINITY; maxx=maxy=maxang=-INFINITY;
        }
        const float w=fmaxf(confidence[k],.001f);
        ++members; weight_sum+=w;
        sx+=positions[k*2]*w; sy+=positions[k*2+1]*w;
        svx+=velocities[k*2]*w; svy+=velocities[k*2+1]*w;
        conf=fmaxf(conf,confidence[k]); was_merged|=merged[k];
        minx=fminf(minx,positions[k*2]-spatial[k*2]*.5f);
        miny=fminf(miny,positions[k*2+1]-spatial[k*2+1]*.5f);
        maxx=fmaxf(maxx,positions[k*2]+spatial[k*2]*.5f);
        maxy=fmaxf(maxy,positions[k*2+1]+spatial[k*2+1]*.5f);
        minang=fminf(minang,lo); maxang=fmaxf(maxang,hi);
        previous_hi=hi; previous_distance=range;
      }
      emit(); group_count[side]=groups;
    }
    __syncthreads();
  }
  if(lane==0) {
    int out=0;
    auto write=[&](const FusedDetection& d) {
      if(out>=pieces) return;
      const int64_t k=output_row+out++;
      out_pos[k*2]=d.x; out_pos[k*2+1]=d.y;
      out_vel[k*2]=d.vx; out_vel[k*2+1]=d.vy;
      out_valid[k]=true; out_confidence[k]=d.confidence;
      out_spatial[k*2]=d.extent_x; out_spatial[k*2+1]=d.extent_y;
      out_angular[k]=d.angular; out_merged[k]=d.merged!=0;
    };
    // Rear-camera positive bearings wrap to the negative robot-bearing edge.
    for(int i=0;i<group_count[1];i++) if(rear[i].bearing>=0.f) write(rear[i]);
    for(int i=0;i<group_count[0];i++) write(front[i]);
    // Rear-camera negative bearings wrap to the positive robot-bearing edge.
    for(int i=0;i<group_count[1];i++) if(rear[i].bearing<0.f) write(rear[i]);
  }
}

__global__ void stereo_pair_fusion_kernel(
    const float* pose,const float* positions,const float* velocities,
    const bool* visible,const float* confidence,const float* spatial,
    const float* angular,const bool* merged,float* out_pos,float* out_vel,
    bool* out_valid,float* out_confidence,float* out_spatial,
    float* out_angular,bool* out_merged,int pieces,int padded,
    int64_t rows) {
  const int row=static_cast<int>(blockIdx.x);
  if(row>=rows) return;
  const int lane=threadIdx.x;
  extern __shared__ unsigned char storage[];
  float* keys=reinterpret_cast<float*>(storage);
  int* sources=reinterpret_cast<int*>(keys+padded);
  const int world=row/12, robot=(row/2)%6, side=row%2;
  const int64_t pose_index=(static_cast<int64_t>(world)*6+robot)*3;
  const int64_t input_base=(static_cast<int64_t>(world)*24+robot*4+side*2)*pieces;
  const int64_t output_base=static_cast<int64_t>(row)*pieces;
  const float* own=pose+pose_index;
  const float heading=own[2]+(side?3.14159265358979323846f:0.f);
  for(int i=lane;i<pieces;i+=blockDim.x) {
    const int64_t k=output_base+i;
    out_valid[k]=false; out_merged[k]=false; out_confidence[k]=0.f;
    out_pos[k*2]=out_pos[k*2+1]=0.f;
    out_vel[k*2]=out_vel[k*2+1]=0.f;
    out_spatial[k*2]=out_spatial[k*2+1]=0.f; out_angular[k]=0.f;
  }
  for(int i=lane;i<2*pieces;i+=blockDim.x) {
    const int camera=i/pieces, slot=i%pieces;
    const int source=camera*pieces+slot;
    const int64_t k=input_base+source;
    const float dx=positions[k*2]-own[0],dy=positions[k*2+1]-own[1];
    float b=atan2f(dy,dx)-heading; b=atan2f(sinf(b),cosf(b));
    keys[i]=visible[k]?b:INFINITY; sources[i]=source;
  }
  for(int i=lane+2*pieces;i<padded;i+=blockDim.x) {
    keys[i]=INFINITY; sources[i]=i;
  }
  __syncthreads();
  for(int k=2;k<=padded;k<<=1) {
    for(int stride=k>>1;stride>0;stride>>=1) {
      for(int i=lane;i<padded;i+=blockDim.x) {
        const int peer=i^stride;
        if(peer<=i) continue;
        const bool ascending=(i&k)==0;
        const bool swap=ascending?(keys[i]>keys[peer]):(keys[i]<keys[peer]);
        if(swap) {
          const float key=keys[i]; keys[i]=keys[peer]; keys[peer]=key;
          const int source=sources[i]; sources[i]=sources[peer]; sources[peer]=source;
        }
      }
      __syncthreads();
    }
  }
  if(lane!=0) return;
  int group=0,members=0; bool in_group=false,was_merged=false;
  float weight_sum=0.f,sx=0.f,sy=0.f,svx=0.f,svy=0.f,conf=0.f;
  float minx=INFINITY,miny=INFINITY,maxx=-INFINITY,maxy=-INFINITY;
  float minang=INFINITY,maxang=-INFINITY,previous_hi=-INFINITY;
  float previous_distance=INFINITY;
  auto emit=[&]() {
    if(!in_group || weight_sum<=0.f || group>=pieces) return;
    const int64_t out=output_base+group++;
    out_pos[out*2]=sx/weight_sum; out_pos[out*2+1]=sy/weight_sum;
    out_vel[out*2]=svx/weight_sum; out_vel[out*2+1]=svy/weight_sum;
    out_valid[out]=true; out_confidence[out]=conf;
    out_spatial[out*2]=maxx-minx; out_spatial[out*2+1]=maxy-miny;
    out_angular[out]=maxang-minang;
    out_merged[out]=was_merged || members>1;
  };
  for(int i=0;i<2*pieces;i++) {
    if(keys[i]==INFINITY) break;
    const int64_t k=input_base+sources[i];
    const float center=keys[i],half=angular[k]*.5f;
    const float lo=center-half,hi=center+half;
    const float range=hypotf(positions[k*2]-own[0],positions[k*2+1]-own[1]);
    const bool joins=in_group && lo<=previous_hi &&
                     fabsf(range-previous_distance)<=.20f;
    if(!joins) {
      emit(); in_group=true; members=0; was_merged=false;
      weight_sum=sx=sy=svx=svy=conf=0.f;
      minx=miny=minang=INFINITY; maxx=maxy=maxang=-INFINITY;
    }
    const float w=fmaxf(confidence[k],.001f);
    ++members; weight_sum+=w;
    sx+=positions[k*2]*w; sy+=positions[k*2+1]*w;
    svx+=velocities[k*2]*w; svy+=velocities[k*2+1]*w;
    conf=fmaxf(conf,confidence[k]); was_merged|=merged[k];
    minx=fminf(minx,positions[k*2]-spatial[k*2]*.5f);
    miny=fminf(miny,positions[k*2+1]-spatial[k*2+1]*.5f);
    maxx=fmaxf(maxx,positions[k*2]+spatial[k*2]*.5f);
    maxy=fmaxf(maxy,positions[k*2+1]+spatial[k*2+1]*.5f);
    minang=fminf(minang,lo); maxang=fmaxf(maxang,hi);
    previous_hi=hi; previous_distance=range;
  }
  emit();
}

__global__ void pack_stereo_detections_kernel(
    const float* pose,const float* pair_pos,const float* pair_vel,
    const bool* pair_valid,const float* pair_confidence,
    const float* pair_spatial,const float* pair_angular,
    const bool* pair_merged,float* out_pos,float* out_vel,bool* out_valid,
    float* out_confidence,float* out_spatial,float* out_angular,
    bool* out_merged,int pieces,int64_t rows) {
  const int row=blockIdx.x;
  if(row>=rows) return;
  const int lane=threadIdx.x;
  const int64_t out_base=static_cast<int64_t>(row)*pieces;
  const int64_t front_base=static_cast<int64_t>(row)*2*pieces;
  const int64_t rear_base=front_base+pieces;
  for(int i=lane;i<pieces;i+=blockDim.x) {
    const int64_t k=out_base+i;
    out_valid[k]=false; out_merged[k]=false; out_confidence[k]=0.f;
    out_pos[k*2]=out_pos[k*2+1]=0.f;
    out_vel[k*2]=out_vel[k*2+1]=0.f;
    out_spatial[k*2]=out_spatial[k*2+1]=0.f; out_angular[k]=0.f;
  }
  __syncthreads();
  if(lane!=0) return;
  const float* own=pose+static_cast<int64_t>(row)*3;
  int out=0;
  auto copy=[&](int64_t src) {
    if(out>=pieces) return;
    const int64_t dst=out_base+out++;
    out_pos[dst*2]=pair_pos[src*2]; out_pos[dst*2+1]=pair_pos[src*2+1];
    out_vel[dst*2]=pair_vel[src*2]; out_vel[dst*2+1]=pair_vel[src*2+1];
    out_valid[dst]=true; out_confidence[dst]=pair_confidence[src];
    out_spatial[dst*2]=pair_spatial[src*2]; out_spatial[dst*2+1]=pair_spatial[src*2+1];
    out_angular[dst]=pair_angular[src]; out_merged[dst]=pair_merged[src];
  };
  for(int i=0;i<pieces;i++) {
    const int64_t src=rear_base+i;
    if(!pair_valid[src]) break;
    float b=atan2f(pair_pos[src*2+1]-own[1],pair_pos[src*2]-own[0])-own[2];
    b=atan2f(sinf(b-3.14159265358979323846f),
             cosf(b-3.14159265358979323846f));
    if(b>=0.f) copy(src);
  }
  for(int i=0;i<pieces;i++) {
    const int64_t src=front_base+i;
    if(!pair_valid[src]) break;
    copy(src);
  }
  for(int i=0;i<pieces;i++) {
    const int64_t src=rear_base+i;
    if(!pair_valid[src]) break;
    float b=atan2f(pair_pos[src*2+1]-own[1],pair_pos[src*2]-own[0])-own[2];
    b=atan2f(sinf(b-3.14159265358979323846f),
             cosf(b-3.14159265358979323846f));
    if(b<0.f) copy(src);
  }
}

__device__ __forceinline__ uint32_t perception_hash(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  value ^= value >> 16;
  return value;
}

__device__ __forceinline__ float perception_uniform(uint32_t seed,
                                                     uint32_t tick,
                                                     uint32_t world,
                                                     uint32_t robot,
                                                     uint32_t piece,
                                                     uint32_t lane) {
  uint32_t key = seed + tick * 0x9e3779b9u + world * 0x85ebca6bu +
                 robot * 0xc2b2ae35u + piece * 0x27d4eb2fu +
                 lane * 0x165667b1u;
  const uint32_t bits = perception_hash(key);
  return (static_cast<float>(bits) + 0.5f) * 2.3283064365386963e-10f;
}

__global__ void perception_commit_3v3_kernel(
    const bool* visible, const bool* active, const int64_t* ticks,
    const float* piece_pos, const float* piece_vel, float* track_pos,
    float* track_vel, float* track_age, bool* track_mask,
    uint32_t seed, int robots, int pieces, float dt, float dropout,
    float position_noise, float velocity_noise, float track_timeout,
    int64_t total) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int64_t per_world = static_cast<int64_t>(robots) * pieces;
  const int world = static_cast<int>(index / per_world);
  if (!active[world]) return;
  const int within_world = static_cast<int>(index - static_cast<int64_t>(world) * per_world);
  const int robot = within_world / pieces;
  const int piece = within_world - robot * pieces;
  const int64_t piece_index = static_cast<int64_t>(world) * pieces + piece;
  const int64_t pair_index = index * 2;

  float age = track_age[index];
  if (track_mask[index]) age = __fadd_rn(age, dt);
  bool observed = visible[index];
  const uint32_t tick = static_cast<uint32_t>(ticks[world]);
  if (observed && dropout > 0.0f) {
    observed = perception_uniform(seed, tick, world, robot, piece, 0u) >= dropout;
  }
  if (observed) {
    const float p0 = perception_uniform(seed, tick, world, robot, piece, 1u);
    const float p1 = perception_uniform(seed, tick, world, robot, piece, 2u);
    const float v0 = perception_uniform(seed, tick, world, robot, piece, 3u);
    const float v1 = perception_uniform(seed, tick, world, robot, piece, 4u);
    const float p_radius = sqrtf(-2.0f * logf(p0));
    const float p_angle = 6.283185307179586f * p1;
    const float v_radius = sqrtf(-2.0f * logf(v0));
    const float v_angle = 6.283185307179586f * v1;
    track_pos[pair_index] = __fadd_rn(piece_pos[piece_index * 2],
        __fmul_rn(p_radius * cosf(p_angle), position_noise));
    track_pos[pair_index + 1] = __fadd_rn(piece_pos[piece_index * 2 + 1],
        __fmul_rn(p_radius * sinf(p_angle), position_noise));
    track_vel[pair_index] = __fadd_rn(piece_vel[piece_index * 2],
        __fmul_rn(v_radius * cosf(v_angle), velocity_noise));
    track_vel[pair_index + 1] = __fadd_rn(piece_vel[piece_index * 2 + 1],
        __fmul_rn(v_radius * sinf(v_angle), velocity_noise));
    track_age[index] = 0.0f;
    track_mask[index] = true;
  } else {
    track_age[index] = age;
    if (age > track_timeout) track_mask[index] = false;
  }
}

__device__ __forceinline__ float perception_normal(uint32_t seed,
                                                   uint32_t tick,
                                                   uint32_t world,
                                                   uint32_t robot,
                                                   uint32_t lane_a,
                                                   uint32_t lane_b) {
  const float u0 = perception_uniform(seed, tick, world, robot, 0u, lane_a);
  const float u1 = perception_uniform(seed, tick, world, robot, 0u, lane_b);
  return sqrtf(-2.0f * logf(u0)) * cosf(6.283185307179586f * u1);
}

__device__ __forceinline__ float perception_detection_normal(
    uint32_t seed,uint32_t tick,uint32_t world,uint32_t robot,
    uint32_t detection,uint32_t lane) {
  const float u0=perception_uniform(seed,tick,world,robot,detection,lane*2u);
  const float u1=perception_uniform(seed,tick,world,robot,detection,lane*2u+1u);
  return sqrtf(-2.0f*logf(u0))*cosf(6.283185307179586f*u1);
}

__global__ void anonymous_track_update_kernel(
    const float* observer,const float* detections,const float* velocities,
    const bool* detected,const float* confidence,const float* spatial,
    const float* angular,const bool* merged,const bool* active,
    const int64_t* ticks,float* track_pos,float* track_vel,float* track_age,
    bool* track_mask,float* quality_state,float* track_confidence,
    float* track_spatial,float* track_angular,bool* track_merged,
    bool* current_visibility,uint32_t seed,int slots,float dt,float dropout,
    float position_noise,float velocity_noise,float timeout,float gate,
    float ball_diameter,int64_t rows) {
  const int row=blockIdx.x;
  if(row>=rows) return;
  const int lane=threadIdx.x;
  const int padded=1<<static_cast<int>(ceilf(log2f(static_cast<float>(slots))));
  extern __shared__ unsigned char shared_bytes[];
  float* keys=reinterpret_cast<float*>(shared_bytes);
  int* sorted_slots=reinterpret_cast<int*>(keys+padded);
  int* candidate=sorted_slots+padded;
  int* winner=candidate+slots;
  int* target=winner+slots;
  bool* observed=reinterpret_cast<bool*>(target+slots);
  bool* accepted=observed+slots;
  float* candidate_distance=reinterpret_cast<float*>(accepted+slots);
  float* quality=candidate_distance+slots;
  const int world=row/(6);
  const int robot=row%6;
  const int64_t track_base=static_cast<int64_t>(row)*slots;
  const int64_t det_base=static_cast<int64_t>(row)*slots;
  const float* own=observer+static_cast<int64_t>(row)*3;
  if(!active[world]) return;

  for(int i=lane;i<slots;i+=blockDim.x) {
    float age=track_age[track_base+i];
    if(track_mask[track_base+i]) age+=dt;
    track_age[track_base+i]=age;
    if(age>timeout) track_mask[track_base+i]=false;
    current_visibility[track_base+i]=false;
    const float dx=track_pos[(track_base+i)*2]-own[0];
    const float dy=track_pos[(track_base+i)*2+1]-own[1];
    float b=atan2f(dy,dx)-own[2];
    b=atan2f(sinf(b),cosf(b));
    keys[i]=track_mask[track_base+i]?b:INFINITY;
    sorted_slots[i]=i;
  }
  for(int i=lane+slots;i<padded;i+=blockDim.x) {
    keys[i]=INFINITY; sorted_slots[i]=i;
  }
  __syncthreads();
  for(int k=2;k<=padded;k<<=1) {
    for(int j=k>>1;j>0;j>>=1) {
      for(int i=lane;i<padded;i+=blockDim.x) {
        const int peer=i^j;
        if(peer<=i) continue;
        const bool ascending=(i&k)==0;
        const bool swap=ascending?(keys[i]>keys[peer]):(keys[i]<keys[peer]);
        if(swap) {
          const float key=keys[i]; keys[i]=keys[peer]; keys[peer]=key;
          const int slot=sorted_slots[i]; sorted_slots[i]=sorted_slots[peer];
          sorted_slots[peer]=slot;
        }
      }
      __syncthreads();
    }
  }
  const uint32_t tick=static_cast<uint32_t>(ticks[world]);
  for(int i=lane;i<slots;i+=blockDim.x) {
    const int64_t idx=det_base+i;
    const float dx=detections[idx*2]-own[0];
    const float dy=detections[idx*2+1]-own[1];
    const float distance=fmaxf(sqrtf(dx*dx+dy*dy),.05f);
    const float width=2.0f*atanf((ball_diameter*.5f)/distance);
    float instant=expf(-distance/8.0f)*fminf(width/.012f,1.0f);
    instant=fminf(fmaxf(instant,0.0f),1.0f)*confidence[idx];
    const float old=quality_state[idx];
    quality[i]=detected[idx]?(0.8f*old+0.2f*instant)*confidence[idx]:old;
    if(detected[idx]) quality_state[idx]=quality[i];
    observed[i]=detected[idx] && perception_uniform(seed,tick,world,robot,i,0u)<
        quality[i]*(1.0f-dropout);
    candidate[i]=-1; candidate_distance[i]=INFINITY; winner[i]=slots+1;
    target[i]=-1; accepted[i]=false;
    if(!observed[i]) continue;
    int best=-1; float best_d=INFINITY;
    for(int rank=max(0,i-1);rank<=min(slots-1,i+1);++rank) {
      const int old_slot=sorted_slots[rank];
      if(old_slot>=slots || !track_mask[track_base+old_slot]) continue;
      const float tx=track_pos[(track_base+old_slot)*2]-detections[idx*2];
      const float ty=track_pos[(track_base+old_slot)*2+1]-detections[idx*2+1];
      const float d=sqrtf(tx*tx+ty*ty);
      if(d<best_d) { best_d=d; best=old_slot; }
    }
    if(best>=0 && best_d<=gate) {
      candidate[i]=best; candidate_distance[i]=best_d;
      atomicMin(&winner[best],i);
    }
  }
  __syncthreads();
  if(lane==0) {
    bool* claimed=reinterpret_cast<bool*>(quality+slots);
    for(int s=0;s<slots;s++) claimed[s]=false;
    for(int i=0;i<slots;i++) {
      if(candidate[i]>=0 && winner[candidate[i]]==i && observed[i]) {
        target[i]=candidate[i]; accepted[i]=true; claimed[target[i]]=true;
      }
    }
    int next_free=0;
    for(int i=0;i<slots;i++) {
      if(!observed[i] || accepted[i]) continue;
      while(next_free<slots && (claimed[next_free] || track_mask[track_base+next_free]))
        ++next_free;
      if(next_free<slots) {
        target[i]=next_free; accepted[i]=true; claimed[next_free]=true;
        ++next_free;
      }
    }
  }
  __syncthreads();
  for(int i=lane;i<slots;i+=blockDim.x) {
    if(!accepted[i]) continue;
    const int slot=target[i];
    const int64_t idx=det_base+i;
    const float distance=fmaxf(hypotf(detections[idx*2]-own[0],
                                      detections[idx*2+1]-own[1]),.05f);
    const float width=2.0f*atanf((ball_diameter*.5f)/distance);
    const float size_quality=fmaxf(.15f,fminf(1.0f,width/.012f));
    const float scale=position_noise*(1.0f+distance/8.0f)/sqrtf(size_quality);
    const float nx=perception_detection_normal(seed,tick,world,robot,i,0u)*scale;
    const float ny=perception_detection_normal(seed,tick,world,robot,i,1u)*scale;
    const float vx=perception_detection_normal(seed,tick,world,robot,i,2u)*velocity_noise;
    const float vy=perception_detection_normal(seed,tick,world,robot,i,3u)*velocity_noise;
    const int64_t out=track_base+slot;
    track_pos[out*2]=detections[idx*2]+nx;
    track_pos[out*2+1]=detections[idx*2+1]+ny;
    track_vel[out*2]=velocities[idx*2]+vx;
    track_vel[out*2+1]=velocities[idx*2+1]+vy;
    track_age[out]=0.0f; track_mask[out]=true;
    track_confidence[out]=quality[i];
    track_spatial[out*2]=spatial[idx*2];
    track_spatial[out*2+1]=spatial[idx*2+1];
    track_angular[out]=angular[idx]; track_merged[out]=merged[idx];
    current_visibility[out]=true;
  }
}

__device__ __forceinline__ bool segment_blocked_by_circles(
    const float* origin, float dx, float dy, const float* obstacles,
    int obstacle_count, float denominator) {
  for (int obstacle = 0; obstacle < obstacle_count; ++obstacle) {
    const float* circle = obstacles + static_cast<int64_t>(obstacle) * 3;
    const float radius = __fadd_rn(circle[2], 0.03f);
    if (!(radius > 0.0f)) continue;
    const float rel_x = __fadd_rn(circle[0], -origin[0]);
    const float rel_y = __fadd_rn(circle[1], -origin[1]);
    const float dot = __fadd_rn(__fmul_rn(rel_x, dx), __fmul_rn(rel_y, dy));
    const float t = __fdiv_rn(dot, denominator);
    if (!(t > 0.02f && t < 0.98f)) continue;
    const float clamped_t = fminf(1.0f, fmaxf(0.0f, t));
    const float closest_x = __fadd_rn(origin[0], __fmul_rn(clamped_t, dx));
    const float closest_y = __fadd_rn(origin[1], __fmul_rn(clamped_t, dy));
    const float distance_x = __fadd_rn(closest_x, -circle[0]);
    const float distance_y = __fadd_rn(closest_y, -circle[1]);
    const float distance_sq = __fadd_rn(__fmul_rn(distance_x, distance_x),
                                        __fmul_rn(distance_y, distance_y));
    if (sqrtf(distance_sq) <= radius) return true;
  }
  return false;
}

__global__ void opponent_tracks_3v3_kernel(
    const float* pose, const float* length, const float* width,
    const float* acceleration, const float* robot_radius,
    const float* obstacles, const bool* active, const bool* controlled,
    const bool* defense_role, const int64_t* ticks, float* opponent_pose,
    float* opponent_velocity, float* opponent_size, float* opponent_age,
    bool* opponent_valid, uint32_t seed, int worlds, int obstacle_count,
    float range_m, float half_fov, float dropout, float position_noise,
    float velocity_noise, float dt, int64_t total) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int world = static_cast<int>(index / 6);
  const int robot_id = static_cast<int>(index - static_cast<int64_t>(world) * 6);
  if (!active[world]) return;
  const int64_t pose_base = static_cast<int64_t>(world) * 18;
  const int64_t track_index = static_cast<int64_t>(world) * 6 + robot_id;
  const float* own = pose + pose_base + robot_id * 3;
  float age = opponent_age[track_index];
  if (opponent_valid[track_index]) age = __fadd_rn(age, dt);

  int selected = -1;
  float selected_distance = HUGE_VALF;
  for (int enemy_slot = 0; enemy_slot < 3; ++enemy_slot) {
    const int enemy_id = robot_id < 3 ? enemy_slot + 3 : enemy_slot;
    if (defense_role[robot_id] && !controlled[enemy_id]) continue;
    const float* enemy = pose + pose_base + enemy_id * 3;
    const float dx = __fadd_rn(enemy[0], -own[0]);
    const float dy = __fadd_rn(enemy[1], -own[1]);
    const float distance_sq = __fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy));
    const float distance = sqrtf(distance_sq);
    if (!(distance <= range_m)) continue;
    const float bearing = atan2f(dy, dx);
    const float difference = __fadd_rn(bearing, -own[2]);
    const float angle = fabsf(atan2f(sinf(difference), cosf(difference)));
    if (!(angle <= half_fov)) continue;
    const float denominator = fmaxf(distance_sq, 1.0e-8f);
    if (segment_blocked_by_circles(own, dx, dy, obstacles,
                                   obstacle_count, denominator)) continue;

    bool occluded = false;
    for (int peer = 0; peer < 6; ++peer) {
      if (peer == robot_id || peer == enemy_id) continue;
      const float* other = pose + pose_base + peer * 3;
      const float rel_x = __fadd_rn(other[0], -own[0]);
      const float rel_y = __fadd_rn(other[1], -own[1]);
      const float dot = __fadd_rn(__fmul_rn(rel_x, dx), __fmul_rn(rel_y, dy));
      const float t = __fdiv_rn(dot, denominator);
      if (!(t > 0.02f && t < 0.98f)) continue;
      const float clamped_t = fminf(1.0f, fmaxf(0.0f, t));
      const float closest_x = __fadd_rn(own[0], __fmul_rn(clamped_t, dx));
      const float closest_y = __fadd_rn(own[1], __fmul_rn(clamped_t, dy));
      const float distance_x = __fadd_rn(closest_x, -other[0]);
      const float distance_y = __fadd_rn(closest_y, -other[1]);
      const float closest_sq = __fadd_rn(__fmul_rn(distance_x, distance_x),
                                         __fmul_rn(distance_y, distance_y));
      const float radius = robot_radius[static_cast<int64_t>(world) * 6 + peer];
      if (sqrtf(closest_sq) <= radius) {
        occluded = true;
        break;
      }
    }
    if (!occluded && distance < selected_distance) {
      selected = enemy_id;
      selected_distance = distance;
    }
  }

  bool observed = selected >= 0;
  const uint32_t tick = static_cast<uint32_t>(ticks[world]);
  if (observed && dropout > 0.0f) {
    observed = perception_uniform(seed, tick, world, robot_id, 0u, 16u) >= dropout;
  }
  if (observed) {
    const float* enemy = pose + pose_base + selected * 3;
    const float px = perception_normal(seed, tick, world, robot_id, 17u, 18u) * position_noise;
    const float py = perception_normal(seed, tick, world, robot_id, 19u, 20u) * position_noise;
    const float heading = perception_normal(seed, tick, world, robot_id, 21u, 22u) *
                          fminf(0.05f, position_noise);
    const float vx = perception_normal(seed, tick, world, robot_id, 23u, 24u) * velocity_noise;
    const float vy = perception_normal(seed, tick, world, robot_id, 25u, 26u) * velocity_noise;
    const float omega = perception_normal(seed, tick, world, robot_id, 27u, 28u) * velocity_noise;
    const int64_t base = track_index * 3;
    opponent_pose[base] = __fadd_rn(enemy[0], px);
    opponent_pose[base + 1] = __fadd_rn(enemy[1], py);
    opponent_pose[base + 2] = __fadd_rn(enemy[2], heading);
    opponent_velocity[base] = __fadd_rn(pose[pose_base + selected * 3], vx);
    opponent_velocity[base + 1] = __fadd_rn(pose[pose_base + selected * 3 + 1], vy);
    opponent_velocity[base + 2] = __fadd_rn(pose[pose_base + selected * 3 + 2], omega);
    opponent_size[base] = length[track_index - robot_id + selected];
    opponent_size[base + 1] = width[track_index - robot_id + selected];
    opponent_size[base + 2] = acceleration[track_index - robot_id + selected] / 10.0f;
    opponent_age[track_index] = 0.0f;
    opponent_valid[track_index] = true;
  } else {
    opponent_age[track_index] = age;
    if (age > 1.0f) opponent_valid[track_index] = false;
  }
}

}  // namespace

void piece_occlusion_launch(const float* pose_xy, const float* segment,
                            const float* obstacles, const bool* eligible,
                            bool* output,
                            int worlds, int pieces, int obstacle_count,
                            hipStream_t stream) {
  const int64_t total = static_cast<int64_t>(worlds) * pieces;
  if (total == 0) return;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(piece_occlusion_kernel, dim3(blocks), dim3(threads), 0,
                     stream, pose_xy, segment, obstacles, eligible, output, pieces,
                     obstacle_count, total);
}

void visibility_launch(const float* pose, const float* pieces_xy,
                       const bool* piece_active, const int64_t* piece_owner,
                       const bool* active, const float* obstacles,
                       const float* other_xy, const float* other_radius,
                       bool* output, int worlds, int pieces, int obstacle_count,
                       float range_m, float half_fov, bool full_fov,
                       hipStream_t stream) {
  const int64_t total = static_cast<int64_t>(worlds) * pieces;
  if (total == 0) return;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(visibility_kernel, dim3(blocks), dim3(threads), 0,
                     stream, pose, pieces_xy, piece_active, piece_owner, active,
                     obstacles, other_xy, other_radius, output, pieces,
                     obstacle_count, range_m, half_fov, full_fov, total);
}

void visibility_3v3_launch(const float* pose, const float* pieces_xy,
                           const bool* piece_active, const int64_t* piece_owner,
                           const bool* active, const float* obstacles,
                           const float* robot_radius, bool* output, int worlds,
                           int robots, int pieces, int obstacle_count, float range_m,
                           float half_fov, bool full_fov, hipStream_t stream) {
  const int64_t total = static_cast<int64_t>(worlds) * robots * pieces;
  if (total == 0) return;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(visibility_3v3_kernel, dim3(blocks), dim3(threads), 0,
                     stream, pose, pieces_xy, piece_active, piece_owner, active,
                     obstacles, robot_radius, output, robots, pieces,
                     obstacle_count, range_m, half_fov, full_fov, total);
}

void angular_cluster_launch(const float* pose,const float* positions,
                            const float* velocities,const bool* visible,
                            const float* input_fraction,const float* input_spatial,
                            const float* input_angular,const bool* input_merged,
                            bool fuse_only,
                            float* out_pos,float* out_vel,bool* out_valid,
                            float* out_fraction,float* out_spatial,
                            float* out_angular,bool* out_merged,int rows,
                            int pieces,float diameter,float half_fov,
                            hipStream_t stream) {
  if(rows==0 || pieces==0) return;
  constexpr int threads=256;
  int padded=1; while(padded<pieces) padded<<=1;
  const size_t shared_bytes=(static_cast<size_t>(pieces)*5+256+padded)*sizeof(float)+
      static_cast<size_t>(padded)*sizeof(int);
  hipLaunchKernelGGL(angular_cluster_kernel,dim3(rows),dim3(threads),shared_bytes,
      stream,pose,positions,velocities,visible,input_fraction,input_spatial,
      input_angular,input_merged,fuse_only,out_pos,out_vel,out_valid,
      out_fraction,out_spatial,out_angular,out_merged,pieces,diameter,half_fov,
      rows);
}

__global__ void angular_cluster_cameras_kernel(
    const float* poses,const float* positions,const float* velocities,
    const bool* visible,float* out_pos,float* out_vel,bool* out_valid,
    float* out_fraction,float* out_spatial,float* out_angular,
    bool* out_merged,int robots,int pieces,float diameter,float half_fov,
    int64_t rows) {
  const int row=static_cast<int>(blockIdx.x);
  if(row>=rows) return;
  const int world=row/(robots*4);
  const int robot=(row/4)%robots;
  const int camera_id=row%4;
  const int lane=threadIdx.x;
  const int64_t robot_base=static_cast<int64_t>(world)*robots*4*pieces+
      static_cast<int64_t>(robot)*4*pieces+static_cast<int64_t>(camera_id)*pieces;
  const int64_t piece_base=static_cast<int64_t>(world)*pieces;
  const int64_t pose_base=static_cast<int64_t>(world)*robots*12+
      static_cast<int64_t>(robot)*12+camera_id*3;
  const int padded=1<<static_cast<int>(ceilf(log2f(static_cast<float>(pieces))));
  extern __shared__ float work[];
  float* distance=work;
  float* lo=distance+pieces;
  float* hi=lo+pieces;
  float* fraction=hi+pieces;
  float* joins=fraction+pieces;
  constexpr int bins=256;
  float* bin_depth=joins+pieces;
  float* sort_key=bin_depth+bins;
  int* sorted_index=reinterpret_cast<int*>(sort_key+padded);
  const float* camera=poses+pose_base;
  for(int j=lane;j<pieces;j+=blockDim.x) {
    const int64_t k=piece_base+j;
    const float dx=positions[k*2]-camera[0];
    const float dy=positions[k*2+1]-camera[1];
    const float d=sqrtf(dx*dx+dy*dy);
    const float b=atan2f(dy,dx)-camera[2];
    const float wrapped=atan2f(sinf(b),cosf(b));
    const float half=atan2f(diameter*.5f,fmaxf(d,1.0e-4f));
    distance[j]=d; lo[j]=fmaxf(-half_fov,wrapped-half);
    hi[j]=fminf(half_fov,wrapped+half); fraction[j]=0.f; joins[j]=0.f;
    const int64_t vk=robot_base+j;
    sort_key[j]=visible[vk]?wrapped:half_fov+1.0f; sorted_index[j]=j;
  }
  for(int j=lane+pieces;j<padded;j+=blockDim.x) {
    sort_key[j]=INFINITY; sorted_index[j]=j;
  }
  __syncthreads();
  for(int k=2;k<=padded;k<<=1) for(int stride=k>>1;stride>0;stride>>=1) {
    for(int j=lane;j<padded;j+=blockDim.x) {
      const int peer=j^stride; if(peer<=j) continue;
      const bool ascending=(j&k)==0;
      const bool swap=ascending?(sort_key[j]>sort_key[peer]):(sort_key[j]<sort_key[peer]);
      if(swap) { const float a=sort_key[j]; sort_key[j]=sort_key[peer]; sort_key[peer]=a;
        const int x=sorted_index[j]; sorted_index[j]=sorted_index[peer]; sorted_index[peer]=x; }
    }
    __syncthreads();
  }
  for(int b=lane;b<bins;b+=blockDim.x) bin_depth[b]=INFINITY;
  __syncthreads();
  const float bin_width=(2.f*half_fov)/bins;
  for(int j=lane;j<pieces;j+=blockDim.x) {
    const int src=sorted_index[j];
    const int64_t vk=robot_base+src;
    if(!visible[vk]) continue;
    const int first=max(0,min(bins-1,static_cast<int>(floorf((fmaxf(-half_fov,lo[src])+half_fov)/bin_width))));
    const int last=max(0,min(bins-1,static_cast<int>(floorf((fminf(half_fov,hi[src])+half_fov)/bin_width))));
    for(int b=first;b<=last;b++) atomicMin(reinterpret_cast<unsigned int*>(&bin_depth[b]),__float_as_uint(distance[src]));
  }
  __syncthreads();
  for(int j=lane;j<pieces;j+=blockDim.x) {
    const int src=sorted_index[j]; const int64_t vk=robot_base+src;
    if(!visible[vk]) { fraction[j]=0.f; continue; }
    const int first=max(0,min(bins-1,static_cast<int>(floorf((fmaxf(-half_fov,lo[src])+half_fov)/bin_width))));
    const int last=max(0,min(bins-1,static_cast<int>(floorf((fminf(half_fov,hi[src])+half_fov)/bin_width))));
    int seen=0,total=0; for(int b=first;b<=last;b++) { ++total; if(bin_depth[b]>=distance[src]-1.e-4f) ++seen; }
    fraction[j]=total?static_cast<float>(seen)/total:0.f;
  }
  __syncthreads();
  for(int j=lane;j<pieces;j+=blockDim.x) {
    const int src=sorted_index[j],prev=j?sorted_index[j-1]:src;
    const int64_t vk=robot_base+src;
    const int64_t pk=robot_base+prev;
    if(j==0 || !visible[vk] || !visible[pk]) { joins[j]=0.f; continue; }
    joins[j]=(fminf(hi[src],hi[prev])-fmaxf(lo[src],lo[prev])>0.f &&
        fabsf(distance[src]-distance[prev])<=.20f && fraction[j]>.05f && fraction[j-1]>.05f)?1.f:0.f;
  }
  __syncthreads();
  const int64_t out_base=robot_base;
  for(int j=lane;j<pieces;j+=blockDim.x) {
    const int64_t k=out_base+j; out_valid[k]=false; out_merged[k]=false; out_fraction[k]=0.f;
    out_pos[k*2]=out_pos[k*2+1]=out_vel[k*2]=out_vel[k*2+1]=0.f;
    out_spatial[k*2]=out_spatial[k*2+1]=out_angular[k]=0.f;
  }
  __syncthreads(); if(lane!=0) return;
  int group=-1; bool in_group=false; float denom=0.f,members=0.f,sx=0.f,sy=0.f,vx=0.f,vy=0.f;
  float minx=INFINITY,miny=INFINITY,maxx=-INFINITY,maxy=-INFINITY,minang=INFINITY,maxang=-INFINITY;
  auto emit=[&]() { if(!in_group) return; if(denom>0.f && group<pieces) {
    const int64_t o=out_base+group; out_pos[o*2]=sx/denom; out_pos[o*2+1]=sy/denom;
    out_vel[o*2]=vx/denom; out_vel[o*2+1]=vy/denom; out_valid[o]=true;
    out_spatial[o*2]=maxx-minx+diameter; out_spatial[o*2+1]=maxy-miny+diameter;
    out_angular[o]=fmaxf(0.f,maxang-minang); out_merged[o]=members>1.f;
    out_fraction[o]=fminf(1.f,denom/fmaxf(members,1.f)); } };
  for(int j=0;j<pieces;j++) { const int src=sorted_index[j];
    const int64_t k=robot_base+src;
    if(!visible[k]) { emit(); in_group=false; continue; }
    if(!in_group || (j>0 && joins[j]==0.f)) { emit(); ++group; in_group=true;
      denom=members=sx=sy=vx=vy=0.f; minx=miny=minang=INFINITY; maxx=maxy=maxang=-INFINITY; }
    const float f=fraction[j]; if(f<=.05f) continue;
    const float x=positions[(piece_base+src)*2],y=positions[(piece_base+src)*2+1];
    denom+=f; members+=1.f; sx+=x*f; sy+=y*f;
    vx+=velocities[(piece_base+src)*2]*f; vy+=velocities[(piece_base+src)*2+1]*f;
    minx=fminf(minx,x); miny=fminf(miny,y); maxx=fmaxf(maxx,x); maxy=fmaxf(maxy,y);
    minang=fminf(minang,lo[src]); maxang=fmaxf(maxang,hi[src]);
  }
  emit();
}

void angular_cluster_cameras_launch(const float* camera_pose,const float* positions,
    const float* velocities,const bool* visible,float* out_pos,float* out_vel,
    bool* out_valid,float* out_fraction,float* out_spatial,float* out_angular,
    bool* out_merged,int worlds,int robots,int cameras,int pieces,float diameter,
    float half_fov,hipStream_t stream) {
  if(worlds==0 || pieces==0) return;
  const int rows=worlds*robots*cameras;
  const int padded=1<<static_cast<int>(ceilf(log2f(static_cast<float>(pieces))));
  const size_t shared=(static_cast<size_t>(pieces)*5+256+padded)*sizeof(float)+static_cast<size_t>(padded)*sizeof(int);
  hipLaunchKernelGGL(angular_cluster_cameras_kernel,dim3(rows),dim3(256),shared,stream,
      camera_pose,positions,velocities,visible,out_pos,out_vel,out_valid,out_fraction,
      out_spatial,out_angular,out_merged,robots,pieces,diameter,half_fov,rows);
}

void stereo_pair_fusion_launch(const float* pose,const float* positions,
    const float* velocities,const bool* visible,const float* confidence,
    const float* spatial,const float* angular,const bool* merged,
    float* out_pos,float* out_vel,bool* out_valid,float* out_confidence,
    float* out_spatial,float* out_angular,bool* out_merged,int worlds,
    int pieces,hipStream_t stream) {
  if(worlds==0 || pieces==0) return;
  int padded=1; while(padded<2*pieces) padded<<=1;
  const size_t shared_bytes=static_cast<size_t>(2*padded)*
      (sizeof(float)+sizeof(int));
  constexpr int threads=256;
  const int rows=worlds*12;
  hipLaunchKernelGGL(stereo_pair_fusion_kernel,dim3(rows),dim3(threads),shared_bytes,
      stream,pose,positions,velocities,visible,confidence,spatial,angular,merged,
      out_pos,out_vel,out_valid,out_confidence,out_spatial,out_angular,out_merged,
      pieces,padded,rows);
}

void pack_stereo_detections_launch(const float* pose,const float* pair_pos,
    const float* pair_vel,const bool* pair_valid,const float* pair_confidence,
    const float* pair_spatial,const float* pair_angular,const bool* pair_merged,
    float* out_pos,float* out_vel,bool* out_valid,float* out_confidence,
    float* out_spatial,float* out_angular,bool* out_merged,int worlds,
    int pieces,hipStream_t stream) {
  const int rows=worlds*6;
  if(rows==0 || pieces==0) return;
  constexpr int threads=256;
  hipLaunchKernelGGL(pack_stereo_detections_kernel,dim3(rows),dim3(threads),0,
      stream,pose,pair_pos,pair_vel,pair_valid,pair_confidence,pair_spatial,
      pair_angular,pair_merged,out_pos,out_vel,out_valid,out_confidence,
      out_spatial,out_angular,out_merged,pieces,rows);
}

void stereo_fusion_pack_launch(const float* pose,const float* camera_pos,
    const float* camera_vel,const bool* camera_valid,const float* camera_conf,
    const float* camera_spatial,const float* camera_angular,const bool* camera_merged,
    float* pair_pos,float* pair_vel,bool* pair_valid,float* pair_conf,
    float* pair_spatial,float* pair_angular,bool* pair_merged,float* out_pos,
    float* out_vel,bool* out_valid,float* out_conf,float* out_spatial,
    float* out_angular,bool* out_merged,int worlds,int pieces,hipStream_t stream) {
  if(worlds==0 || pieces==0) return;
  constexpr int threads=256;
  if(pieces<=512) {
    int padded=1; while(padded<2*pieces) padded<<=1;
    const size_t shared_bytes=static_cast<size_t>(2*padded)*
        (sizeof(float)+sizeof(int))+static_cast<size_t>(2*pieces)*sizeof(FusedDetection);
    hipLaunchKernelGGL(stereo_fusion_pack_kernel,dim3(worlds*6),dim3(threads),
        shared_bytes,stream,pose,camera_pos,camera_vel,camera_valid,camera_conf,
        camera_spatial,camera_angular,camera_merged,out_pos,out_vel,out_valid,
        out_conf,out_spatial,out_angular,out_merged,pieces,padded,
        static_cast<int64_t>(worlds)*6);
    return;
  }
  stereo_pair_fusion_launch(pose,camera_pos,camera_vel,camera_valid,camera_conf,
      camera_spatial,camera_angular,camera_merged,pair_pos,pair_vel,pair_valid,
      pair_conf,pair_spatial,pair_angular,pair_merged,worlds,pieces,stream);
  pack_stereo_detections_launch(pose,pair_pos,pair_vel,pair_valid,pair_conf,
      pair_spatial,pair_angular,pair_merged,out_pos,out_vel,out_valid,out_conf,
      out_spatial,out_angular,out_merged,worlds,pieces,stream);
}

void perception_commit_3v3_launch(const bool* visible, const bool* active,
                                  const int64_t* ticks, const float* piece_pos,
                                  const float* piece_vel, float* track_pos,
                                  float* track_vel, float* track_age,
                                  bool* track_mask, uint32_t seed, int worlds,
                                  int robots, int pieces, float dt, float dropout,
                                  float position_noise, float velocity_noise,
                                  float track_timeout,
                                  hipStream_t stream) {
  const int64_t total = static_cast<int64_t>(worlds) * robots * pieces;
  if (total == 0) return;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(perception_commit_3v3_kernel, dim3(blocks), dim3(threads), 0,
      stream, visible, active, ticks, piece_pos, piece_vel, track_pos,
      track_vel, track_age, track_mask, seed, robots, pieces, dt, dropout,
      position_noise, velocity_noise, track_timeout, total);
}

void anonymous_track_update_launch(
    const float* observer,const float* detections,const float* velocities,
    const bool* detected,const float* confidence,const float* spatial,
    const float* angular,const bool* merged,const bool* active,
    const int64_t* ticks,float* track_pos,float* track_vel,float* track_age,
    bool* track_mask,float* quality_state,float* track_confidence,
    float* track_spatial,float* track_angular,bool* track_merged,
    bool* current_visibility,uint32_t seed,int worlds,int slots,float dt,
    float dropout,float position_noise,float velocity_noise,float timeout,
    float gate,float ball_diameter,hipStream_t stream) {
  if(worlds==0 || slots==0) return;
  int padded=1; while(padded<slots) padded<<=1;
  const size_t shared_bytes=static_cast<size_t>(3*padded+5*slots)*sizeof(int)+
      static_cast<size_t>(padded)*sizeof(float)+
      static_cast<size_t>(3*slots)*sizeof(bool)+
      static_cast<size_t>(2*slots)*sizeof(float);
  constexpr int threads=256;
  hipLaunchKernelGGL(anonymous_track_update_kernel,dim3(worlds*6),dim3(threads),
      shared_bytes,stream,observer,detections,velocities,detected,confidence,
      spatial,angular,merged,active,ticks,track_pos,track_vel,track_age,
      track_mask,quality_state,track_confidence,track_spatial,track_angular,
      track_merged,current_visibility,seed,slots,dt,dropout,position_noise,
      velocity_noise,timeout,gate,ball_diameter,static_cast<int64_t>(worlds)*6);
}

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
                                float velocity_noise, float dt, hipStream_t stream) {
  const int64_t total = static_cast<int64_t>(worlds) * 6;
  if (total == 0) return;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(opponent_tracks_3v3_kernel, dim3(blocks), dim3(threads), 0,
      stream, pose, length, width, acceleration, robot_radius, obstacles, active,
      controlled, defense_role, ticks, opponent_pose, opponent_velocity,
      opponent_size, opponent_age, opponent_valid, seed, worlds, obstacle_count,
      range_m, half_fov, dropout, position_noise, velocity_noise, dt, total);
}
