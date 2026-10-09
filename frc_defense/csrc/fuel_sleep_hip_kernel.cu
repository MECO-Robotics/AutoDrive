#include <hip/hip_runtime.h>
#include <math.h>

struct SleepConfig {int n,p,r,capacity,allpairs;float radius,gravity,dt,time,speed,spin,height;};

__device__ int find_root(int* parent,int i) {
 int next=atomicAdd(parent+i,0);
 while(next!=i){i=next;next=atomicAdd(parent+i,0);}
 return i;
}

__device__ void unite(int* parent,int i,int j) {
 while(true) {
  int a=find_root(parent,i),b=find_root(parent,j);
  if(a==b)return;
  int high=max(a,b),low=min(a,b);
  if(atomicCAS(parent+high,high,low)==high)return;
 }
}

__device__ bool touching(const float* pos,int i,int j,float cutoff) {
 float x=pos[3*i]-pos[3*j],y=pos[3*i+1]-pos[3*j+1],z=pos[3*i+2]-pos[3*j+2];
 return x*x+y*y+z*z<=cutoff*cutoff;
}

__global__ void island_kernel(float* pos,float* vel,float* angular,bool* sleeping,float* clock,int* parent,int* quiet,int* supported,float* minimum_clock,const bool* free,const float* pose,const float* robotvel,const float* length,const float* width,const int* neighbors,const int* counts,const int* overflow,float* public_clock,const float* support_height,SleepConfig c) {
 int w=blockIdx.x,base=w*c.p;
 for(int i=threadIdx.x;i<c.p;i+=blockDim.x){int k=base+i;parent[k]=i;quiet[k]=1;supported[k]=0;minimum_clock[k]=INFINITY;}
 __syncthreads();
 for(int i=threadIdx.x;i<c.p;i+=blockDim.x) {
  int k=base+i;if(!free[k])continue;
  if(c.allpairs||overflow[w]) {
   for(int j=i+1;j<c.p;++j)if(free[base+j]&&touching(pos,k,base+j,2*c.radius))unite(parent+base,i,j);
  } else {
   for(int slot=0;slot<min(counts[k],c.capacity);++slot){int j=neighbors[k*c.capacity+slot];if(j>=0&&j<c.p&&free[base+j]&&touching(pos,k,base+j,2*c.radius))unite(parent+base,i,j);}
  }
 }
 __syncthreads();
 for(int i=threadIdx.x;i<c.p;i+=blockDim.x) {
  int k=base+i;if(!free[k])continue;
  int root=base+find_root(parent+base,i);
  float vx=vel[3*k],vy=vel[3*k+1],vz=vel[3*k+2]-(sleeping[k]?0:c.gravity*c.dt);
  float wx=angular[3*k],wy=angular[3*k+1],wz=angular[3*k+2];
  bool still=vx*vx+vy*vy+vz*vz<=c.speed*c.speed&&wx*wx+wy*wy+wz*wz<=c.spin*c.spin;
  for(int r=0;r<c.r&&still;++r) {
   int robot=w*c.r+r;float theta=pose[3*robot+2],co=cosf(theta),si=sinf(theta);
   float dx=pos[3*k]-pose[3*robot],dy=pos[3*k+1]-pose[3*robot+1];
   float x=co*dx+si*dy,y=-si*dx+co*dy;
   float ex=fmaxf(fabsf(x)-length[robot]*.5f,0.f),ey=fmaxf(fabsf(y)-width[robot]*.5f,0.f);
   if(ex*ex+ey*ey<=(c.radius+.002f)*(c.radius+.002f)&&pos[3*k+2]<c.height+c.radius)still=false;
  }
  atomicAnd(quiet+root,still?1:0);
  if(pos[3*k+2]<=support_height[k]+c.radius+.005f)atomicOr(supported+root,1);
  atomicMin(reinterpret_cast<unsigned int*>(minimum_clock+root),__float_as_uint(fminf(clock[k],public_clock[k])));
 }
 __syncthreads();
 for(int i=threadIdx.x;i<c.p;i+=blockDim.x) {
  int k=base+i;if(!free[k])continue;
  int root=base+find_root(parent+base,i);
  bool stable=quiet[root]&&supported[root];
  float elapsed=stable?minimum_clock[root]+c.dt:0.f;
  clock[k]=public_clock[k]=elapsed;
  bool asleep=stable&&elapsed>=c.time;
  sleeping[k]=asleep;
  if(asleep){for(int axis=0;axis<3;++axis){vel[3*k+axis]=0;angular[3*k+axis]=0;}}
 }
}

void fuel_sleep_launch(float* pos,float* vel,float* angular,bool* sleeping,float* clock,int* parent,int* quiet,int* supported,float* minimum_clock,const bool* free,const float* pose,const float* robotvel,const float* length,const float* width,const int* neighbors,const int* counts,const int* overflow,float* public_clock,const float* support_height,const float* p,const int* d,hipStream_t stream) {
 SleepConfig c{d[0],d[1],d[2],d[3],d[4],p[0],p[1],p[2],p[3],p[4],p[5],p[6]};
 island_kernel<<<c.n,256,0,stream>>>(pos,vel,angular,sleeping,clock,parent,quiet,supported,minimum_clock,free,pose,robotvel,length,width,neighbors,counts,overflow,public_clock,support_height,c);
}
