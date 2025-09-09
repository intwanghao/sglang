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

template <
    typename ElementAB,
    typename ElementC,
    typename ElementAccumulator,
    typename LayoutSFA,
    typename LayoutSFB,
    typename ScaleConfig>
void get_group_gemm_starts(
    int32_t* expert_offsets,
    ElementAB** a_offsets,
    ElementAB** b_offsets,
    ElementC** out_offsets,
    ElementAccumulator** a_scales_offsets,
    ElementAccumulator** b_scales_offsets,
    ElementAB* a_base_as_int,
    ElementAB* b_base_as_int,
    ElementC* out_base_as_int,
    ElementAccumulator* a_scales_base_as_int,
    ElementAccumulator* b_scales_base_as_int,
    LayoutSFA* layout_sfa_base_as_int,
    LayoutSFB* layout_sfb_base_as_int,
    int* problem_sizes,
    int* problem_sizes_transpose,
    bool transpose = false) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int expert_id = item_ct1.get_local_id(2);

  if (expert_id >= item_ct1.get_group_range(2) * item_ct1.get_local_range(2)) {
    return;
  }

  int m = problem_sizes[expert_id * 3];
  int n = problem_sizes[expert_id * 3 + 1];
  int k = problem_sizes[expert_id * 3 + 2];
  if (transpose) {
    problem_sizes_transpose[expert_id * 3] = n;
    problem_sizes_transpose[expert_id * 3 + 1] = m;
    problem_sizes_transpose[expert_id * 3 + 2] = k;
  }

  int32_t expert_offset = expert_offsets[expert_id];
  int a_stride = 0;
  int b_stride = 0;
  int a_scale_stride = 0;
  int b_scale_stride = 0;
  if (!transpose) {
    a_stride = expert_offset * k;
    b_stride = expert_id * k * n;
    a_scale_stride = expert_offset * k / 128;
    b_scale_stride = expert_id * k * n / 128 / 128;
  } else {
    a_stride = expert_id * k * n;
    b_stride = expert_offset * k;
    a_scale_stride = expert_id * k * n / 128 / 128;
    b_scale_stride = expert_offset * k / 128;
  }
  a_offsets[expert_id] = a_base_as_int + a_stride;
  b_offsets[expert_id] = b_base_as_int + b_stride;
  out_offsets[expert_id] = out_base_as_int + expert_offset * n;
  a_scales_offsets[expert_id] = a_scales_base_as_int + a_scale_stride;
  b_scales_offsets[expert_id] = b_scales_base_as_int + b_scale_stride;

  LayoutSFA* layout_sfa_ptr = layout_sfa_base_as_int + expert_id;
  LayoutSFB* layout_sfb_ptr = layout_sfb_base_as_int + expert_id;

  if (!transpose) {
    *layout_sfa_ptr = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    *layout_sfb_ptr = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
  } else {
    *layout_sfa_ptr = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(n, m, k, 1));
    *layout_sfb_ptr = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(n, m, k, 1));
  }
}

/*
DPCT1049:613: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
info::device::max_work_group_size. Adjust the work-group size if needed.
*/
#define __CALL_GET_STARTS_KERNEL(TENSOR_C_TYPE, C_TYPE, LayoutSFA, LayoutSFB, ScaleConfig)                        \
  else if (out_tensors.dtype() == TENSOR_C_TYPE) {                                                                \
    auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync}; \
                                                                                                                  \
    ((sycl::queue*)(stream))->submit([&](sycl::handler& cgh) {                                                    \
      auto expert_offsets_data_ptr_ct0 = static_cast<int32_t*>(expert_offsets.data_ptr());                        \
      auto a_ptrs_data_ptr_ct1 = static_cast<cutlass::float_e4m3_t**>(a_ptrs.data_ptr());                         \
      auto b_ptrs_data_ptr_ct2 = static_cast<cutlass::float_e4m3_t**>(b_ptrs.data_ptr());                         \
      auto out_ptrs_data_ptr_ct3 = static_cast<C_TYPE**>(out_ptrs.data_ptr());                                    \
      auto a_scales_ptrs_data_ptr_ct4 = static_cast<float**>(a_scales_ptrs.data_ptr());                           \
      auto b_scales_ptrs_data_ptr_ct5 = static_cast<float**>(b_scales_ptrs.data_ptr());                           \
      auto a_tensors_data_ptr_ct6 = static_cast<cutlass::float_e4m3_t*>(a_tensors.data_ptr());                    \
      auto b_tensors_data_ptr_ct7 = static_cast<cutlass::float_e4m3_t*>(b_tensors.data_ptr());                    \
      auto out_tensors_data_ptr_ct8 = static_cast<C_TYPE*>(out_tensors.data_ptr());                               \
      auto a_scales_data_ptr_ct9 = static_cast<float*>(a_scales.data_ptr());                                      \
      auto b_scales_data_ptr_ct10 = static_cast<float*>(b_scales.data_ptr());                                     \
      auto layout_sfa_data_ptr_ct11 = reinterpret_cast<LayoutSFA*>(layout_sfa.data_ptr());                        \
      auto layout_sfb_data_ptr_ct12 = reinterpret_cast<LayoutSFB*>(layout_sfb.data_ptr());                        \
      auto problem_sizes_data_ptr_ct13 = static_cast<int*>(problem_sizes.data_ptr());                             \
      auto problem_sizes_transpose_data_ptr_ct14 = static_cast<int*>(problem_sizes_transpose.data_ptr());         \
      auto transpose_ct15 = transpose;                                                                            \
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
            get_group_gemm_starts<cutlass::float_e4m3_t, C_TYPE, float, LayoutSFA, LayoutSFB, ScaleConfig>(       \
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
                layout_sfa_data_ptr_ct11,                                                                         \
                layout_sfb_data_ptr_ct12,                                                                         \
                problem_sizes_data_ptr_ct13,                                                                      \
                problem_sizes_transpose_data_ptr_ct14,                                                            \
                transpose_ct15);                                                                                  \
          });                                                                                                     \
    });                                                                                                           \
  }

namespace {
template <typename LayoutSFA, typename LayoutSFB, typename ScaleConfig>
void run_get_group_gemm_starts(
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
    torch::Tensor const& b_scales,
    torch::Tensor const& layout_sfa,
    torch::Tensor const& layout_sfb,
    torch::Tensor const& problem_sizes,
    torch::Tensor& problem_sizes_transpose,
    bool transpose = false) {
  TORCH_CHECK(a_tensors.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b_tensors.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(out_tensors.size(1) % 128 == 0 or out_tensors.size(0) % 128 == 0);
  TORCH_CHECK(a_tensors.size(1) % 128 == 0 or a_tensors.size(0) % 128 == 0);

  int num_experts = (int)expert_offsets.size(0);
  auto stream = &c10::xpu::getCurrentXPUStream(a_tensors.device().index()).queue();

  if (false) {
  }
  __CALL_GET_STARTS_KERNEL(torch::kBFloat16, cutlass::bfloat16_t, LayoutSFA, LayoutSFB, ScaleConfig)
  __CALL_GET_STARTS_KERNEL(torch::kFloat16, sycl::half, LayoutSFA, LayoutSFB, ScaleConfig)
  else {
    TORCH_CHECK(false, "Invalid output type (must be float16 or bfloat16)");
  }
}
}  // namespace
