#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <c10/xpu/XPUStream.h>
#include <c10/xpu/XPUStream.h>

#include <torch/all.h>

#include "cutlass/bfloat16.h"
#include "cutlass/float8.h"
#include <c10/xpu/XPUStream.h>

template <typename ElementA, typename ElementB, typename ElementC, typename ElementAccumulator>
void int4_fp8_get_group_gemm_starts(
    int32_t* expert_offsets,
    ElementA** a_offsets,
    ElementB** b_offsets,
    ElementC** out_offsets,
    ElementAccumulator** a_scales_offsets,
    cutlass::bfloat16_t** b_scales_offsets,
    ElementA* a_base_as_int,
    ElementB* b_base_as_int,
    ElementC* out_base_as_int,
    ElementAccumulator* a_scales_base_as_int,
    cutlass::bfloat16_t* b_scales_base_as_int,
    int64_t n,
    int64_t k,
    bool per_act_token,
    bool per_out_ch) {
  int expert_id = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
  int32_t expert_offset = expert_offsets[expert_id];

  a_offsets[expert_id] = a_base_as_int + expert_offset * k;
  b_offsets[expert_id] = b_base_as_int + expert_id * k * n / 2;
  out_offsets[expert_id] = out_base_as_int + expert_offset * n;
  a_scales_offsets[expert_id] = a_scales_base_as_int + (per_act_token ? expert_offset : 0);
  b_scales_offsets[expert_id] = b_scales_base_as_int + (per_out_ch ? expert_id * n * 4 * k / 512 : expert_id);
}

/*
DPCT1049:594: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
info::device::max_work_group_size. Adjust the work-group size if needed.
*/
#define __CALL_W4A8_GET_STARTS_KERNEL(TENSOR_C_TYPE, C_TYPE)                                                      \
  else if (out_tensors.dtype() == TENSOR_C_TYPE) {                                                                \
    auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync}; \
    dpct::has_capability_or_fail(c10::xpu::getCurrentXPUStream().queue().get_device(), {sycl::aspect::fp16});     \
                                                                                                                  \
    ((sycl::queue*)(stream))->submit([&](sycl::handler& cgh) {                                                    \
      auto expert_offsets_data_ptr_ct0 = static_cast<int32_t*>(expert_offsets.data_ptr());                        \
      auto a_ptrs_data_ptr_ct1 = static_cast<cutlass::float_e4m3_t**>(a_ptrs.data_ptr());                         \
      auto b_ptrs_data_ptr_ct2 = static_cast<cutlass::int8_t**>(b_ptrs.data_ptr());                               \
      auto out_ptrs_data_ptr_ct3 = static_cast<C_TYPE**>(out_ptrs.data_ptr());                                    \
      auto a_scales_ptrs_data_ptr_ct4 = static_cast<float**>(a_scales_ptrs.data_ptr());                           \
      auto b_scales_ptrs_data_ptr_ct5 = static_cast<cutlass::bfloat16_t**>(b_scales_ptrs.data_ptr());             \
      auto a_tensors_data_ptr_ct6 = static_cast<cutlass::float_e4m3_t*>(a_tensors.data_ptr());                    \
      auto b_tensors_data_ptr_ct7 = static_cast<cutlass::int8_t*>(b_tensors.data_ptr());                          \
      auto out_tensors_data_ptr_ct8 = static_cast<C_TYPE*>(out_tensors.data_ptr());                               \
      auto a_scales_data_ptr_ct9 = static_cast<float*>(a_scales.data_ptr());                                      \
      auto b_scales_data_ptr_ct10 = static_cast<cutlass::bfloat16_t*>(b_scales.data_ptr());                       \
      auto out_tensors_size_ct11 = out_tensors.size(1);                                                           \
      auto a_tensors_size_ct12 = a_tensors.size(1);                                                               \
      auto per_act_token_ct13 = per_act_token;                                                                    \
      auto per_out_ch_ct14 = per_out_ch;                                                                          \
                                                                                                                  \
      auto last_event = dpct::get_default_queue().ext_oneapi_get_last_event();                                    \
      [&](auto&& _e) {                                                                                            \
        if constexpr (std::is_same_v<std::remove_reference_t<decltype(last_event)>, sycl::event>)                 \
          cgh.depends_on(_e);                                                                                     \
        else if (_e.has_value())                                                                                  \
          cgh.depends_on(_e.value());                                                                             \
      }(last_event);                                                                                              \
                                                                                                                  \
      cgh.parallel_for(                                                                                           \
          sycl::nd_range<3>(sycl::range<3>(1, 1, num_experts), sycl::range<3>(1, 1, num_experts)),                \
          exp_props,                                                                                              \
          [=](sycl::nd_item<3> item_ct1) {                                                                        \
            int4_fp8_get_group_gemm_starts<cutlass::float_e4m3_t, cutlass::int8_t, C_TYPE, float>(                \
                expert_offsets_data_ptr_ct0,                                                                      \
                a_ptrs_data_ptr_ct1,                                                                              \
                b_ptrs_data_ptr_ct2,                                                                              \
                out_ptrs_data_ptr_ct3,                                                                            \
                a_scales_ptrs_data_ptr_ct4,                                                                       \
                b_scales_ptrs_data_ptr_ct5,                                                                       \
                a_tensors_data_ptr_ct6,                                                                           \
                b_tensors_data_ptr_ct7,                                                                           \
                out_tensors_data_ptr_ct8,                                                                         \
                a_scales_data_ptr_ct9,                                                                            \
                b_scales_data_ptr_ct10,                                                                           \
                out_tensors_size_ct11,                                                                            \
                a_tensors_size_ct12,                                                                              \
                per_act_token_ct13,                                                                               \
                per_out_ch_ct14);                                                                                 \
          });                                                                                                     \
    });                                                                                                           \
  }

namespace {

void run_int4_fp8_get_group_gemm_starts(
    torch::Tensor const& expert_offsets,
    torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs,
    torch::Tensor& out_ptrs,
    torch::Tensor& a_scales_ptrs,
    torch::Tensor& b_scales_ptrs,
    torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors,
    torch::Tensor& out_tensors,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales) {
  TORCH_CHECK(a_tensors.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b_tensors.dtype() == torch::kInt8);
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kBFloat16);

  int num_experts = static_cast<int>(expert_offsets.size(0));
  bool per_act_token = a_scales.numel() != 1;
  bool per_out_ch = b_scales.numel() != num_experts;

  auto stream = &c10::xpu::getCurrentXPUStream(expert_offsets.device().index()).queue();

  if (false) {
  }
  __CALL_W4A8_GET_STARTS_KERNEL(torch::kBFloat16, cutlass::bfloat16_t)
  __CALL_W4A8_GET_STARTS_KERNEL(torch::kFloat16, sycl::half)
  else {
    TORCH_CHECK(false, "Invalid output type (must be float16 or bfloat16)");
  }
}

}  // namespace
