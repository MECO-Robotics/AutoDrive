#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <vector>

void fuel_sleep_launch(float*,float*,float*,bool*,float*,int*,int*,int*,float*,const bool*,const float*,const float*,const float*,const float*,const int*,const int*,const int*,float*,const float*,const float*,const int*,hipStream_t);

void update(std::vector<torch::Tensor> t,std::vector<double> p,std::vector<int64_t> d) {
 TORCH_CHECK(t.size()==19&&p.size()==7&&d.size()==5,"invalid island arguments");
 const auto device=t[0].device();
 c10::DeviceGuard guard(device);
 for(const auto& a:t) TORCH_CHECK(a.is_cuda()&&a.is_contiguous()&&a.device()==device,"island buffers must be contiguous on one GPU");
 for(int i:{0,1,2,4,8,10,11,12,13,17,18})TORCH_CHECK(t[i].scalar_type()==at::kFloat,"island float32 required");
 for(int i:{3,9})TORCH_CHECK(t[i].scalar_type()==at::kBool,"island bool required");
 for(int i:{5,6,7,14,15,16})TORCH_CHECK(t[i].scalar_type()==at::kInt,"island int32 required");
 float params[7];int dims[5];
 for(int i=0;i<7;++i)params[i]=p[i];
 for(int i=0;i<5;++i)dims[i]=d[i];
 fuel_sleep_launch(t[0].data_ptr<float>(),t[1].data_ptr<float>(),t[2].data_ptr<float>(),t[3].data_ptr<bool>(),t[4].data_ptr<float>(),t[5].data_ptr<int>(),t[6].data_ptr<int>(),t[7].data_ptr<int>(),t[8].data_ptr<float>(),t[9].data_ptr<bool>(),t[10].data_ptr<float>(),t[11].data_ptr<float>(),t[12].data_ptr<float>(),t[13].data_ptr<float>(),t[14].data_ptr<int>(),t[15].data_ptr<int>(),t[16].data_ptr<int>(),t[17].data_ptr<float>(),t[18].data_ptr<float>(),params,dims,c10::cuda::getCurrentCUDAStream(t[0].get_device()));
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("update",&update);}
