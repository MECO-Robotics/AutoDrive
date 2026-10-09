#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>

struct V { float x,y,z; __device__ V(float a=0,float b=0,float c=0):x(a),y(b),z(c){} };
__device__ V operator+(V a,V b){return V(a.x+b.x,a.y+b.y,a.z+b.z);}
__device__ V operator-(V a,V b){return V(a.x-b.x,a.y-b.y,a.z-b.z);}
__device__ V operator*(V a,float b){return V(a.x*b,a.y*b,a.z*b);}
__device__ float dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}
__device__ V cross(V a,V b){return V(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);}
__device__ float norm(V a){return sqrtf(dot(a,a));}
__device__ V get(const float*p,int i){return V(p[3*i],p[3*i+1],p[3*i+2]);}
__device__ void put(float*p,int i,V v){p[3*i]=v.x;p[3*i+1]=v.y;p[3*i+2]=v.z;}
__device__ void add(float*p,int i,V v){atomicAdd(p+3*i,v.x);atomicAdd(p+3*i+1,v.y);atomicAdd(p+3*i+2,v.z);}
struct C {float r,m,g,k,d,mu,roll,height,skin,dt,L,W,st,sv,sw,compression,panel,rise;int N,P,R,K,M,X,Y,Z,B,capacity;bool cache,allpairs,compact,colored,robotgrid;int sleep;};
__device__ int cell(V p,C c){int x=max(0,min(c.X-1,(int)floorf(p.x/(2*c.r+c.skin))+1));int y=max(0,min(c.Y-1,(int)floorf(p.y/(2*c.r+c.skin))+1));int z=max(0,min(c.Z-1,(int)floorf(p.z/(2*c.r+c.skin))+1));return (x*c.Y+y)*c.Z+z;}
__global__ void integrate(float*pos,float*vel,const bool*free,bool*sleep,const float*angular,C c){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=c.N*c.P||!free[i])return;
 if(sleep[i]&&(norm(get(vel,i))>=c.sv||norm(get(angular,i))>=c.sw))sleep[i]=false;
 if(!sleep[i]){V v=get(vel,i);v.z-=c.g*c.dt;put(vel,i,v);put(pos,i,get(pos,i)+v*c.dt);}}
__global__ void detect(const float*pos,const float*ref,const bool*free,const bool*oldfree,int*rebuild,C c){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=c.N*c.P)return;if(!c.cache||free[i]!=oldfree[i]||dot(get(pos,i)-get(ref,i),get(pos,i)-get(ref,i))>.25f*c.skin*c.skin)atomicExch(rebuild+i/c.P,1);}
__global__ void clear_grid(int*heads,int*overflow,const int*rebuild,const float*reference,C c){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=c.N*c.P)return;int w=i/c.P;if(!rebuild[w])return;
 heads[w*c.X*c.Y*c.Z+cell(get(reference,i),c)]=-1;
 if(i%c.P==0)overflow[w]=0;
}
__global__ void bin(const float*pos,const bool*free,int*heads,int*next,const int*rebuild,C c){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=c.N*c.P||!free[i]||!rebuild[i/c.P])return;int h=(i/c.P)*c.X*c.Y*c.Z+cell(get(pos,i),c);next[i]=atomicExch(heads+h,i%c.P);}
__global__ void neighbors(const float*pos,const bool*free,const int*heads,const int*next,int*list,int*counts,int*overflow,const int*rebuild,C c){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=c.N*c.P||!rebuild[i/c.P])return;int w=i/c.P,pi=i%c.P,total=0;counts[i]=0;if(!free[i])return;int cellid=cell(get(pos,i),c),z=cellid%c.Z,y=(cellid/c.Z)%c.Y,x=cellid/(c.Y*c.Z);float cutoff=2*c.r+c.skin;
 for(int dx=-1;dx<=1;dx++)for(int dy=-1;dy<=1;dy++)for(int dz=-1;dz<=1;dz++){int xx=x+dx,yy=y+dy,zz=z+dz;if(xx<0||xx>=c.X||yy<0||yy>=c.Y||zz<0||zz>=c.Z)continue;int j=heads[w*c.X*c.Y*c.Z+(xx*c.Y+yy)*c.Z+zz];while(j>=0){if(j>pi){V d=get(pos,i)-get(pos,w*c.P+j);if(dot(d,d)<=cutoff*cutoff){if(total<c.M)list[i*c.M+total]=j;total++;}}j=next[w*c.P+j];}}
 counts[i]=min(total,c.M);if(total>c.M)atomicExch(overflow+w,1);}
__global__ void save_ref(const float*pos,float*ref,const bool*free,bool*oldfree,int*rebuild,int*metrics,const int*counts,const int*overflow,C c){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=c.N*c.P)return;int w=i/c.P;if(rebuild[w]){put(ref,i,get(pos,i));oldfree[i]=free[i];if(i%c.P==0){atomicAdd(metrics,1);atomicAdd(metrics+1,overflow[w]);}}}
__global__ void finish_rebuild(int*rebuild,C c){int w=blockIdx.x*blockDim.x+threadIdx.x;if(w<c.N)rebuild[w]=0;}
__device__ V impulse(V n,float depth,V relative,float inv,C c,float tangent_inv=-1){float vn=dot(relative,n);float jn=fmaxf(0.f,c.dt*(c.k*depth-c.d*vn)/(1+c.dt*c.d*inv+c.dt*c.dt*c.k*inv));if(depth>=2*c.r*c.compression)jn=fmaxf(jn,fmaxf(0.f,-vn/inv));V tangent=relative-n*vn;float speed=norm(tangent);float jt=fminf(speed/(tangent_inv<0?inv+2.5f/c.m:tangent_inv),c.mu*jn);return n*jn-tangent*(jt/fmaxf(speed,1.e-12f));}
__device__ void pair(int i,int j,const float*pos,const float*vel,const float*angular,bool*sleep,float*dv,float*dw,C c){V delta=get(pos,i)-get(pos,j);float distance=norm(delta);if(distance>=2*c.r||(sleep[i]&&sleep[j]))return;V n=distance>1.e-12f?delta*(1/distance):V(1,0,0);V ri=n*(-.5f*distance),rj=n*(.5f*distance);V relative=get(vel,i)+cross(get(angular,i),ri)-get(vel,j)-cross(get(angular,j),rj);V jimp=impulse(n,2*c.r-distance,relative,2/c.m,c,2/c.m+.5f*distance*distance/(.4f*c.m*c.r*c.r));add(dv,i,jimp*(1/c.m));add(dv,j,jimp*(-1/c.m));float ii=.4f*c.m*c.r*c.r;add(dw,i,cross(ri,jimp)*(1/ii));add(dw,j,cross(rj,jimp*(-1))*(1/ii));sleep[i]=false;sleep[j]=false;}
__global__ void fused_pairs(const float*pos,const float*vel,const float*angular,const bool*free,bool*sleep,const int*list,const int*counts,const int*overflow,float*dv,float*dw,C c){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=c.N*c.P||!free[i])return;int w=i/c.P,pi=i%c.P;if(c.allpairs||overflow[w]){for(int j=pi+1;j<c.P;j++)if(free[w*c.P+j])pair(i,w*c.P+j,pos,vel,angular,sleep,dv,dw,c);}else{for(int k=0;k<counts[i];k++){int j=w*c.P+list[i*c.M+k];if(free[j])pair(i,j,pos,vel,angular,sleep,dv,dw,c);}}}
__global__ void compact_pairs(const float*pos,const float*vel,const float*angular,const bool*free,bool*sleep,const int*list,const int*counts,const int*overflow,int*contacts,int*count,float*dv,float*dw,C c){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=c.N*c.P||!free[i])return;int w=i/c.P,pi=i%c.P;if(!c.allpairs&&overflow[w]){for(int j=pi+1;j<c.P;j++)if(free[w*c.P+j])pair(i,w*c.P+j,pos,vel,angular,sleep,dv,dw,c);return;}int candidates=c.allpairs?c.P-pi-1:counts[i];for(int k=0;k<candidates;k++){int j=w*c.P+((c.allpairs||overflow[w])?pi+1+k:list[i*c.M+k]);if(!free[j]||norm(get(pos,i)-get(pos,j))>=2*c.r)continue;int q=atomicAdd(count,1);contacts[2*q]=i;contacts[2*q+1]=j;}}
__global__ void solve_compact(const float*pos,const float*vel,const float*angular,bool*sleep,const int*contacts,const int*count,float*dv,float*dw,int capacity,C c){int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=capacity||q>=*count)return;pair(contacts[2*q],contacts[2*q+1],pos,vel,angular,sleep,dv,dw,c);}
// Round-robin edge coloring of the complete particle graph. Every color
// contains disjoint pairs, so direct velocity writes have no atomic conflicts.
// Sparse grid Jacobi is expected to win when most graph edges are absent.
__global__ void colored_pairs(const float*pos,float*vel,float*angular,const bool*free,bool*sleep,C c){
 int w=blockIdx.x,even=c.P+(c.P%2),rotating=even-1;
 for(int color=0;color<rotating;color++){
  for(int slot=threadIdx.x;slot<even/2;slot+=blockDim.x){
   int a=slot==0?even-1:(color+slot)%rotating;
   int b=slot==0?color:(color-slot+rotating)%rotating;
   if(a>=c.P||b>=c.P)continue;
   int i=w*c.P+a,j=w*c.P+b;if(!free[i]||!free[j]||(sleep[i]&&sleep[j]))continue;
   V delta=get(pos,i)-get(pos,j);float distance=norm(delta);if(distance>=2*c.r)continue;
   V n=distance>1.e-12f?delta*(1/distance):V(1,0,0),ri=n*(-.5f*distance),rj=n*(.5f*distance);
   V relative=get(vel,i)+cross(get(angular,i),ri)-get(vel,j)-cross(get(angular,j),rj);
   V ji=impulse(n,2*c.r-distance,relative,2/c.m,c,2/c.m+.5f*distance*distance/(.4f*c.m*c.r*c.r));float inertia=.4f*c.m*c.r*c.r;
   put(vel,i,get(vel,i)+ji*(1/c.m));put(vel,j,get(vel,j)-ji*(1/c.m));
   put(angular,i,get(angular,i)+cross(ri,ji)*(1/inertia));put(angular,j,get(angular,j)+cross(rj,ji*(-1))*(1/inertia));
   sleep[i]=false;sleep[j]=false;
  }
  __syncthreads();
 }
}
__global__ void apply(float*vel,float*angular,const float*dv,const float*dw,C c){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<c.N*c.P){put(vel,i,get(vel,i)+get(dv,i));put(angular,i,get(angular,i)+get(dw,i));}}
__device__ V boxnormal(float x,float y,float hx,float hy,float&distance){float cx=fmaxf(-hx,fminf(hx,x)),cy=fmaxf(-hy,fminf(hy,y));V delta(x-cx,y-cy,0);distance=norm(delta);if(distance>1.e-12f)return delta*(1/distance);float gx=hx-fabsf(x),gy=hy-fabsf(y);distance=-fminf(gx,gy);return gx<=gy?V(x>=0?1:-1,0,0):V(0,y>=0?1:-1,0);}
__global__ void robot_candidates(const float*pose,const float*length,const float*width,const bool*free,const int*heads,const int*next,bool*mask,C c){
 int robot=blockIdx.x*blockDim.x+threadIdx.x;if(robot>=c.N*c.R)return;
 int w=robot/c.R;V p=get(pose,robot);float co=cosf(p.z),si=sinf(p.z),margin=c.r+.5f*c.skin;
 float hx=.5f*length[robot],hy=.5f*width[robot],ex=fabsf(co)*hx+fabsf(si)*hy+margin,ey=fabsf(si)*hx+fabsf(co)*hy+margin;
 float cellsize=2*c.r+c.skin;
 int xlo=max(0,min(c.X-1,(int)floorf((p.x-ex)/cellsize)+1)),xhi=max(0,min(c.X-1,(int)floorf((p.x+ex)/cellsize)+1));
 int ylo=max(0,min(c.Y-1,(int)floorf((p.y-ey)/cellsize)+1)),yhi=max(0,min(c.Y-1,(int)floorf((p.y+ey)/cellsize)+1));
 int zlo=0,zhi=max(0,min(c.Z-1,(int)floorf((c.height+margin)/cellsize)+1));
 for(int x=xlo;x<=xhi;x++)for(int y=ylo;y<=yhi;y++)for(int z=zlo;z<=zhi;z++){
  int j=heads[w*c.X*c.Y*c.Z+(x*c.Y+y)*c.Z+z];
  while(j>=0){if(free[w*c.P+j])mask[robot*c.P+j]=true;j=next[w*c.P+j];}
 }
}
__global__ void robot_contacts(float*pos,float*vel,float*angular,const bool*free,bool*sleep,const float*pose,const float*rv,const float*length,const float*width,const float*mass,const float*inertia_scale,float*rdv,const bool*mask,C c){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=c.N*c.P||!free[i])return;int w=i/c.P;V p=get(pos,i),v=get(vel,i),ang=get(angular,i);for(int r=0;r<c.R;r++){int robot=w*c.R+r;if((c.robotgrid&&!mask[robot*c.P+i%c.P])||p.z>=c.height+c.r)continue;V rp=get(pose,robot),rvel=get(rv,robot);float dx=p.x-rp.x,dy=p.y-rp.y;float bound=.5f*sqrtf(length[robot]*length[robot]+width[robot]*width[robot])+c.r;if(dx*dx+dy*dy>bound*bound)continue;float co=cosf(rp.z),si=sinf(rp.z),distance;V localn=boxnormal(co*dx+si*dy,-si*dx+co*dy,length[robot]*.5f,width[robot]*.5f,distance);if(distance>=c.r||p.z>=c.height+c.r)continue;V n(co*localn.x-si*localn.y,si*localn.x+co*localn.y,0);V arm(dx-c.r*n.x,dy-c.r*n.y,0),contact(rvel.x-rvel.z*arm.y,rvel.y+rvel.z*arm.x,0);V relative=v+cross(ang,n*(-c.r))-contact;float inertia=mass[robot]*(length[robot]*length[robot]+width[robot]*width[robot])/12*inertia_scale[robot];float rn=arm.x*n.y-arm.y*n.x;V tangent=relative-n*dot(relative,n);V unit=tangent*(1/fmaxf(norm(tangent),1.e-12f));float rt=arm.x*unit.y-arm.y*unit.x;V j=impulse(n,c.r-distance,relative,1/c.m+1/mass[robot]+rn*rn/inertia,c,3.5f/c.m+1/mass[robot]+rt*rt/inertia);v=v+j*(1/c.m);ang=ang+cross(n*(-c.r),j)*(1/(.4f*c.m*c.r*c.r));add(rdv,robot,V(-j.x/mass[robot],-j.y/mass[robot],-(arm.x*j.y-arm.y*j.x)/inertia));sleep[i]=false;}put(vel,i,v);put(angular,i,ang);}
__global__ void apply_robot(float*rv,const float*rdv,C c){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<c.N*c.R)put(rv,i,get(rv,i)+get(rdv,i));}
__device__ void staticcontact(V n,float depth,V&v,V&ang,C c){if(depth<=0)return;V arm=n*(-c.r);V j=impulse(n,depth,v+cross(ang,arm),1/c.m,c);v=v+j*(1/c.m);ang=ang+cross(arm,j)*(1/(.4f*c.m*c.r*c.r));float speed=norm(ang),loss=c.roll*norm(j)*c.r/(.4f*c.m*c.r*c.r);ang=ang*fmaxf(0.f,1-loss/fmaxf(speed,1.e-12f));}
__global__ void static_contacts(float*pos,float*vel,float*angular,const bool*free,bool*sleep,float*clock,const float*boxes,const float*bumps,float*support_height,C c){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=c.N*c.P||!free[i])return;V p=get(pos,i),v=get(vel,i),ang=get(angular,i);float height=0;V floor_normal(0,0,1);for(int b=0;b<c.B;b++){float dx=p.x-bumps[4*b];if(fabsf(dx)<=bumps[4*b+2]&&fabsf(p.y-bumps[4*b+1])<=bumps[4*b+3]){float h=c.panel+c.rise*(1-fabsf(dx)/bumps[4*b+2]);if(h>height){height=h;float slope=(dx>0?-1:dx<0?1:0)*c.rise/bumps[4*b+2];floor_normal=V(-slope,0,1)*(1/sqrtf(1+slope*slope));}}}support_height[i]=height;if(!sleep[i]){staticcontact(floor_normal,c.r-(p.z-height)*floor_normal.z,v,ang,c);staticcontact(V(1,0,0),c.r-p.x,v,ang,c);staticcontact(V(-1,0,0),p.x+c.r-c.L,v,ang,c);staticcontact(V(0,1,0),c.r-p.y,v,ang,c);staticcontact(V(0,-1,0),p.y+c.r-c.W,v,ang,c);if(p.z<c.height+c.r)for(int k=0;k<c.K;k++){float distance;V normal=boxnormal(p.x-boxes[4*k],p.y-boxes[4*k+1],boxes[4*k+2],boxes[4*k+3],distance);staticcontact(normal,c.r-distance,v,ang,c);}}
 if(c.sleep==1){V support=v; if(p.z<height+c.r+.005f&&!sleep[i])support.z-=c.g*c.dt;bool quiet=norm(support)<c.sv&&norm(ang)<c.sw&&p.z<height+c.r+.005f;clock[i]=quiet?clock[i]+c.dt:0;sleep[i]=quiet&&clock[i]>=c.st;if(sleep[i]){v=V();ang=V();}}else if(c.sleep==0)sleep[i]=false;put(vel,i,v);put(angular,i,ang);}

void fuel_substep_launch(float*pos,float*vel,float*angular,bool*sleep,float*clock,const bool*free,const float*pose,float*rv,const float*length,const float*width,const float*mass,const float*inertia,const float*boxes,float*reference,bool*oldfree,int*heads,int*next,int*list,int*counts,int*overflow,int*rebuild,int*metrics,float*dv,float*dw,float*rdv,int*contacts,int*contact_count,bool*robot_mask,const float*bumps,float*support_height,const float*params,const int*dims,hipStream_t stream){C c;

 c.r=params[0];
c.m=params[1];
c.g=params[2];
c.k=params[3];
c.d=params[4];
c.mu=params[5];
c.roll=params[6];
c.height=params[7];
c.skin=params[8];
c.dt=params[9];
c.L=params[10];
c.W=params[11];
c.st=params[12];
c.sv=params[13];
c.sw=params[14];c.compression=params[15];
c.panel=params[16];
c.rise=params[17];
c.N=dims[0];
c.P=dims[1];
c.R=dims[2];
c.K=dims[3];
c.M=dims[4];
c.X=dims[5];
c.Y=dims[6];
c.Z=dims[7];
c.cache=dims[8];
c.sleep=dims[9];
c.allpairs=dims[10];
c.compact=dims[11];
c.colored=dims[12];
c.robotgrid=dims[13];
c.B=dims[14];
c.capacity=dims[15];
int threads=128,blocks=(c.N*c.P+threads-1)/threads;

 integrate<<<blocks,threads,0,stream>>>(pos,vel,free,sleep,angular,c);

 if(!c.allpairs||c.robotgrid){
detect<<<blocks,threads,0,stream>>>(pos,reference,free,oldfree,rebuild,c);
int cells=c.N*c.X*c.Y*c.Z;
clear_grid<<<blocks,threads,0,stream>>>(heads,overflow,rebuild,reference,c);
bin<<<blocks,threads,0,stream>>>(pos,free,heads,next,rebuild,c);
if(!c.allpairs)
neighbors<<<blocks,threads,0,stream>>>(pos,free,heads,next,list,counts,overflow,rebuild,c);
save_ref<<<blocks,threads,0,stream>>>(pos,reference,free,oldfree,rebuild,metrics,counts,overflow,c);
finish_rebuild<<<(c.N+threads-1)/threads,threads,0,stream>>>(rebuild,c);
}
 hipMemsetAsync(dv,0,c.N*c.P*3*sizeof(float),stream);
hipMemsetAsync(dw,0,c.N*c.P*3*sizeof(float),stream);
hipMemsetAsync(rdv,0,c.N*c.R*3*sizeof(float),stream);
if(c.robotgrid&&c.R>0){
hipMemsetAsync(robot_mask,0,c.N*c.R*c.P*sizeof(bool),stream);
robot_candidates<<<(c.N*c.R+threads-1)/threads,threads,0,stream>>>(pose,length,width,free,heads,next,robot_mask,c);
}


 if(c.colored){
colored_pairs<<<c.N,128,0,stream>>>(pos,vel,angular,free,sleep,c);
}
else if(c.compact){
hipMemsetAsync(contact_count,0,sizeof(int),stream);
compact_pairs<<<blocks,threads,0,stream>>>(pos,vel,angular,free,sleep,list,counts,overflow,contacts,contact_count,dv,dw,c);
int capacity=c.capacity;
if(capacity>0)
solve_compact<<<(capacity+threads-1)/threads,threads,0,stream>>>(pos,vel,angular,sleep,contacts,contact_count,dv,dw,capacity,c);
}else
 fused_pairs<<<blocks,threads,0,stream>>>(pos,vel,angular,free,sleep,list,counts,overflow,dv,dw,c);

 apply<<<blocks,threads,0,stream>>>(vel,angular,dv,dw,c);
robot_contacts<<<blocks,threads,0,stream>>>(pos,vel,angular,free,sleep,pose,rv,length,width,mass,inertia,rdv,robot_mask,c);
if(c.R>0)
apply_robot<<<(c.N*c.R+threads-1)/threads,threads,0,stream>>>(rv,rdv,c);
static_contacts<<<blocks,threads,0,stream>>>(pos,vel,angular,free,sleep,clock,boxes,bumps,support_height,c);

}

__global__ void spawn_height_kernel(const float* xy,const float* bumps,float* out,
                                  int count,int bump_count,float radius,float panel,float rise) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=count)return;
 float x=xy[2*i],y=xy[2*i+1],height=radius;
 for(int b=0;b<bump_count;b++) {
  float dx=x-bumps[4*b],half_x=bumps[4*b+2];
  if(fabsf(dx)>half_x||fabsf(y-bumps[4*b+1])>bumps[4*b+3])continue;
  float slope=(dx>0?-1:dx<0?1:0)*rise/half_x;
  float surface=panel+rise*(1-fabsf(dx)/half_x)+radius*sqrtf(1+slope*slope);
  height=fmaxf(height,surface);
 }
 out[i]=height;
}

void fuel_spawn_height_launch(const float* xy,const float* bumps,float* out,
                              int count,int bump_count,float radius,float panel,float rise,
                              hipStream_t stream) {
 if(count<=0)return;
 spawn_height_kernel<<<(count+127)/128,128,0,stream>>>(xy,bumps,out,count,bump_count,radius,panel,rise);
}
