#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <c10/core/DeviceGuard.h>
#include <hip/hip_runtime.h>
#include <vector>

void fuel_substep_launch(float*,float*,float*,bool*,float*,const bool*,const float*,float*,const float*,const float*,const float*,const float*,const float*,float*,bool*,int*,int*,int*,int*,int*,int*,int*,float*,float*,float*,int*,int*,bool*,const float*,float*,const float*,const int*,hipStream_t);

void substep(std::vector<torch::Tensor> t,std::vector<double> params,std::vector<int64_t> dims){
 TORCH_CHECK(t.size()==30 && params.size()==18 && dims.size()==16,"invalid fuel substep argument count");
 const auto device=t[0].device();
 c10::DeviceGuard device_guard(device);
 for(const auto& a:t)TORCH_CHECK(a.is_cuda()&&a.is_contiguous()&&a.device()==device,"fuel tensors must be contiguous on same HIP device");
 for(int i:{0,1,2,4,6,7,8,9,10,11,12,13,22,23,24,28,29})TORCH_CHECK(t[i].scalar_type()==at::kFloat,"fuel float dtype required");
 for(int i:{3,5,14,27})TORCH_CHECK(t[i].scalar_type()==at::kBool,"fuel bool dtype required");
 for(int i:{15,16,17,18,19,20,21,25,26})TORCH_CHECK(t[i].scalar_type()==at::kInt,"fuel index dtype int32 required");
 int n=dims[0],p=dims[1],r=dims[2];
 TORCH_CHECK(t[0].sizes()==torch::IntArrayRef({n,p,3}) && t[1].sizes()==t[0].sizes() && t[2].sizes()==t[0].sizes(),"fuel state shape mismatch");
 TORCH_CHECK(t[6].sizes()==torch::IntArrayRef({n,r,3})&&t[7].sizes()==t[6].sizes(),"robot state shape mismatch");
 float values[18];int shape[16];for(int i=0;i<18;i++)values[i]=params[i];for(int i=0;i<16;i++)shape[i]=dims[i];
 fuel_substep_launch(t[0].data_ptr<float>(),t[1].data_ptr<float>(),t[2].data_ptr<float>(),t[3].data_ptr<bool>(),t[4].data_ptr<float>(),t[5].data_ptr<bool>(),t[6].data_ptr<float>(),t[7].data_ptr<float>(),t[8].data_ptr<float>(),t[9].data_ptr<float>(),t[10].data_ptr<float>(),t[11].data_ptr<float>(),t[12].data_ptr<float>(),t[13].data_ptr<float>(),t[14].data_ptr<bool>(),t[15].data_ptr<int>(),t[16].data_ptr<int>(),t[17].data_ptr<int>(),t[18].data_ptr<int>(),t[19].data_ptr<int>(),t[20].data_ptr<int>(),t[21].data_ptr<int>(),t[22].data_ptr<float>(),t[23].data_ptr<float>(),t[24].data_ptr<float>(),t[25].data_ptr<int>(),t[26].data_ptr<int>(),t[27].data_ptr<bool>(),t[28].data_ptr<float>(),t[29].data_ptr<float>(),values,shape,c10::cuda::getCurrentCUDAStream(t[0].get_device()));
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void fuel_spawn_height_launch(const float*,const float*,float*,int,int,float,float,float,hipStream_t);

void spawn_height(torch::Tensor xy,torch::Tensor bumps,torch::Tensor out,
                  double radius,double panel,double rise) {
 TORCH_CHECK(xy.is_cuda()&&xy.scalar_type()==at::kFloat&&xy.is_contiguous()&&
             xy.dim()>=2&&xy.size(-1)==2,"spawn positions must be contiguous HIP float [...,2]");
 const auto device=xy.device();
 c10::DeviceGuard device_guard(device);
 TORCH_CHECK(bumps.is_cuda()&&bumps.device()==device&&bumps.scalar_type()==at::kFloat&&
             bumps.is_contiguous()&&bumps.dim()==2&&bumps.size(1)==4,"invalid spawn terrain boxes");
 TORCH_CHECK(out.is_cuda()&&out.device()==device&&out.scalar_type()==at::kFloat&&
             out.is_contiguous()&&out.numel()==xy.numel()/2,"invalid spawn height output");
 fuel_spawn_height_launch(xy.data_ptr<float>(),bumps.data_ptr<float>(),out.data_ptr<float>(),
                          static_cast<int>(out.numel()),static_cast<int>(bumps.size(0)),
                          radius,panel,rise,c10::cuda::getCurrentCUDAStream(xy.get_device()));
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
 m.def("substep",&substep,"Coupled fuel substep");
 m.def("spawn_height",&spawn_height,"Analytic terrain height for ground fuel");
}
