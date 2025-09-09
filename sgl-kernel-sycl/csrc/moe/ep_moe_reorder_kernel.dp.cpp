#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <c10/xpu/XPUStream.h>
#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>

#include <c10/core/DeviceGuard.h>

#include <flashinfer/vec_dtypes.dp.hpp>

#include "utils.h"

template <typename scalar_t>
void ep_pre_reorder_cuda_kernel(
    const scalar_t* __restrict__ input_ptr,
    scalar_t* __restrict__ gateup_input_ptr,
    const int* __restrict__ src2dst_ptr,
    const int* __restrict__ topk_ids_ptr,
    const float* __restrict__ a1_scales_ptr,
    int start_expert_id,
    int end_expert_id,
    int topk,
    int hidden_size,
    bool use_per_token_if_dynamic) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int token_idx = item_ct1.get_group(2);
  int tid = item_ct1.get_local_id(2);

  const scalar_t* src_ptr = input_ptr + int64_t(token_idx) * hidden_size;
  const int* token_src2dst = src2dst_ptr + token_idx * topk;
  const int* token_topk_ids = topk_ids_ptr + token_idx * topk;

  float scale = 1.0f;

  if (a1_scales_ptr != nullptr and use_per_token_if_dynamic) {
    scale = 1.0f / a1_scales_ptr[token_idx];
  }

  for (int k = 0; k < topk; ++k) {
    int expert_id = token_topk_ids[k];
    if (expert_id < start_expert_id || expert_id > end_expert_id) continue;

    if (a1_scales_ptr != nullptr) {
      if (!use_per_token_if_dynamic) {
        scale = 1.0f / a1_scales_ptr[expert_id - start_expert_id];
      }
    }

    int dst_idx = token_src2dst[k];
    scalar_t* dst_ptr = gateup_input_ptr + int64_t(dst_idx) * hidden_size;

    constexpr uint32_t vec_size = 16 / sizeof(scalar_t);
    using vec_t = flashinfer::vec_t<scalar_t, vec_size>;

    int vec_elements = (hidden_size / vec_size) * vec_size;
    for (int idx = tid; idx < hidden_size / vec_size; idx += item_ct1.get_local_range(2)) {
      vec_t input_vec, output_vec;
      input_vec.cast_load(src_ptr + idx * vec_size);
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        float val = static_cast<float>(input_vec[i]);
        output_vec[i] = static_cast<scalar_t>(val * scale);
      }
      output_vec.cast_store(dst_ptr + idx * vec_size);
    }

    for (int idx = vec_elements + tid; idx < hidden_size; idx += item_ct1.get_local_range(2)) {
      float val = static_cast<float>(src_ptr[idx]);
      dst_ptr[idx] = static_cast<scalar_t>(val * scale);
    }
  }
}

template <typename scalar_t>
/*
DPCT1110:623: The total declared local variable size in device function ep_post_reorder_cuda_kernel exceeds 128 bytes
and may cause high register pressure. Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void ep_post_reorder_cuda_kernel(
    const scalar_t* __restrict__ down_output_ptr,
    scalar_t* __restrict__ output_ptr,
    const int* __restrict__ src2dst_ptr,
    const int* __restrict__ topk_ids_ptr,
    const scalar_t* __restrict__ topk_weights_ptr,
    int start_expert_id,
    int end_expert_id,
    int topk,
    int hidden_size) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int token_idx = item_ct1.get_group(2);
  const int tid = item_ct1.get_local_id(2);

  const int* token_src2dst = src2dst_ptr + token_idx * topk;
  const int* token_topk_ids = topk_ids_ptr + token_idx * topk;
  const scalar_t* token_topk_weights = topk_weights_ptr + token_idx * topk;

  scalar_t* dst_ptr = output_ptr + static_cast<int64_t>(token_idx) * hidden_size;

  constexpr uint32_t vec_size = 16 / sizeof(scalar_t);
  using vec_t = flashinfer::vec_t<scalar_t, vec_size>;

  const int vec_iters = hidden_size / vec_size;
  for (int idx = tid; idx < vec_iters; idx += item_ct1.get_local_range(2)) {
    float acc[vec_size] = {0};

    for (int k = 0; k < topk; ++k) {
      const int expert_id = token_topk_ids[k];
      if (expert_id < start_expert_id || expert_id > end_expert_id) continue;
      const int src_row = token_src2dst[k];
      const scalar_t* src_ptr = down_output_ptr + static_cast<int64_t>(src_row) * hidden_size;
      const float weight = static_cast<float>(token_topk_weights[k]);

      vec_t src_vec;
      src_vec.cast_load(src_ptr + idx * vec_size);

#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        acc[i] += static_cast<float>(src_vec[i]) * weight;
      }
    }
    vec_t out_vec;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i)
      out_vec[i] = static_cast<scalar_t>(acc[i]);

    out_vec.cast_store(dst_ptr + idx * vec_size);
  }
}

void ep_moe_pre_reorder(
    torch::Tensor input,
    torch::Tensor gateup_input,
    torch::Tensor src2dst,
    torch::Tensor topk_ids,
    torch::Tensor a1_scales,
    int64_t start_expert_id,
    int64_t end_expert_id,
    int64_t topk,
    bool use_per_token_if_dynamic) {
  const int total_blocks = input.size(0);
  const int block_size = 512;
  dpct::dim3 grid(total_blocks);
  dpct::dim3 block(block_size);
  int hidden_size = input.size(1);

  /*
  DPCT1038:624: When the kernel function name is used as a macro argument, the migration result may be incorrect. You
  need to verify the definition of the macro.
  */
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), scalar_t, [&] {
    /*
    DPCT1049:625: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
      auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
      dpct::has_capability_or_fail(c10::xpu::getCurrentXPUStream().queue().get_device(), {sycl::aspect::fp16});

      c10::xpu::getCurrentXPUStream().queue().submit([&](sycl::handler& cgh) {
        auto input_data_ptr_ct0 = static_cast<scalar_t*>(input.data_ptr());
        auto gateup_input_data_ptr_ct1 = static_cast<scalar_t*>(gateup_input.data_ptr());
        const int* src2dst_data_ptr_int_ct2 = src2dst.data_ptr<int>();
        const int* topk_ids_data_ptr_int_ct3 = topk_ids.data_ptr<int>();
        const float* a1_scales_defined_a1_scales_data_ptr_float_nullptr_ct4 =
            a1_scales.defined() ? a1_scales.data_ptr<float>() : nullptr;

        cgh.depends_on(dpct::get_current_device().get_in_order_queues_last_events());

        cgh.parallel_for(sycl::nd_range<3>(grid * block, block), exp_props, [=](sycl::nd_item<3> item_ct1) {
          ep_pre_reorder_cuda_kernel<scalar_t>(
              input_data_ptr_ct0,
              gateup_input_data_ptr_ct1,
              src2dst_data_ptr_int_ct2,
              topk_ids_data_ptr_int_ct3,
              a1_scales_defined_a1_scales_data_ptr_float_nullptr_ct4,
              start_expert_id,
              end_expert_id,
              topk,
              hidden_size,
              use_per_token_if_dynamic);
        });
      });
    }
    return true;
  });
}

void ep_moe_post_reorder(
    torch::Tensor down_output,
    torch::Tensor output,
    torch::Tensor src2dst,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    int64_t start_expert_id,
    int64_t end_expert_id,
    int64_t topk) {
  const int total_tokens = output.size(0);
  const int block_size = 512;
  dpct::dim3 grid(total_tokens);
  dpct::dim3 block(block_size);
  const int hidden_size = output.size(1);

  /*
  DPCT1038:626: When the kernel function name is used as a macro argument, the migration result may be incorrect. You
  need to verify the definition of the macro.
  */
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(down_output.scalar_type(), scalar_t, [&] {
    /*
    DPCT1049:627: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
      auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
      dpct::has_capability_or_fail(c10::xpu::getCurrentXPUStream().queue().get_device(), {sycl::aspect::fp16});

      c10::xpu::getCurrentXPUStream().queue().submit([&](sycl::handler& cgh) {
        auto down_output_data_ptr_ct0 = static_cast<scalar_t*>(down_output.data_ptr());
        auto output_data_ptr_ct1 = static_cast<scalar_t*>(output.data_ptr());
        const int* src2dst_data_ptr_int_ct2 = src2dst.data_ptr<int>();
        const int* topk_ids_data_ptr_int_ct3 = topk_ids.data_ptr<int>();
        auto topk_weights_data_ptr_ct4 = static_cast<scalar_t*>(topk_weights.data_ptr());

        cgh.depends_on(dpct::get_current_device().get_in_order_queues_last_events());

        cgh.parallel_for(sycl::nd_range<3>(grid * block, block), exp_props, [=](sycl::nd_item<3> item_ct1) {
          ep_post_reorder_cuda_kernel<scalar_t>(
              down_output_data_ptr_ct0,
              output_data_ptr_ct1,
              src2dst_data_ptr_int_ct2,
              topk_ids_data_ptr_int_ct3,
              topk_weights_data_ptr_ct4,
              static_cast<int>(start_expert_id),
              static_cast<int>(end_expert_id),
              static_cast<int>(topk),
              hidden_size);
        });
      });
    }
    return true;
  });
}
