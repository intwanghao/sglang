#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <c10/xpu/XPUStream.h>
#include <ATen/xpu/XPUContext.h>
#include <torch/all.h>
#include <sycl/sycl.hpp>

void VectorAddKernel(float* A, float* B, float* C, int n)
{

    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        if(item_ct1.get_local_id(2) >= n) {
        return;
    }
    C[item_ct1.get_local_id(2)] =
        A[item_ct1.get_local_id(2)] + B[item_ct1.get_local_id(2)];
}


void
vec_add_usm(torch::Tensor& a,
            torch::Tensor& b,
            torch::Tensor& c, int n) {
  //const size_t n = static_cast<size_t>(a.numel());

  auto pa = static_cast<float*>(a.data_ptr<float>());
  auto pb = static_cast<float*>(b.data_ptr<float>());
  auto pc = static_cast<float*>(c.data_ptr<float>());

  //sycl::queue q;

  // USM device（也可改为 malloc_shared 省掉 memcpy）
  //float* dA = sycl::malloc_device<float>(n, q);
  //float* dB = sycl::malloc_device<float>(n, q);
  //float* dC = sycl::malloc_device<float>(n, q);
  sycl::queue* stream = &c10::xpu::getCurrentXPUStream().queue();
  // H2D
  //q.memcpy(dA, pa, n*sizeof(float));
  //q.memcpy(dB, pb, n*sizeof(float));

  // kernel
  //stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> i){
  //  pc[i] = pa[i] + pb[i];
  //});
  stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, n),
                                       sycl::range<3>(1, 1, n)),
                     [=](sycl::nd_item<3> item_ct1) {
                         VectorAddKernel(pa, pb, pc, n);
                     });
  // D2H + 同步
  //q.memcpy(pc, dC, n*sizeof(float)).wait();
  stream->wait();
  //sycl::free(dA,q); sycl::free(dB,q); sycl::free(dC,q);
  //return out;
}