#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <c10/xpu/XPUStream.h>
#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>

#include <c10/core/DeviceGuard.h>

#include <algorithm>
#include <flashinfer/vec_dtypes.dp.hpp>

#include "utils.h"
#include <sycl/ext/intel/math.hpp>

#include <cmath>

using namespace flashinfer;

template <typename scalar_t>
inline scalar_t silu_quantize(float x);

template <>
inline float silu_quantize<float>(float x) {
  float y = x / (1.f + sycl::native::exp(-x));
  return y;
}

template <>
inline sycl::half silu_quantize<sycl::half>(float x) {
  float y = x / (1.f + sycl::native::exp(-x));
  return sycl::ext::intel::math::float2half_rn(y);
}

template <>
inline sycl::ext::oneapi::bfloat16 silu_quantize<sycl::ext::oneapi::bfloat16>(float x) {
  float y = x / (1.f + sycl::native::exp(-x));
  return sycl::ext::intel::math::float2bfloat16_rn(y);
}

template <typename scalar_t>
/*
DPCT1110:646: The total declared local variable size in device function ep_moe_act_and_mul_cuda_kernel exceeds 128 bytes
and may cause high register pressure. Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void ep_moe_act_and_mul_cuda_kernel(
    const scalar_t* __restrict__ gateup_output,
    scalar_t* __restrict__ down_input,
    const int* __restrict__ reorder_topk_ids,
    const float* __restrict__ scales,
    int start_expert_id,
    int end_expert_id,
    int hidden_size) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  constexpr uint32_t vec_size = 16 / sizeof(scalar_t);
  using vec_t = flashinfer::vec_t<scalar_t, vec_size>;

  const int64_t token_idx = item_ct1.get_group(2);
  const int64_t thread_idx = item_ct1.get_local_id(2);
  const int64_t stride = item_ct1.get_local_range(2);

  const int half_hidden_size = hidden_size >> 1;
  const int expert_id = reorder_topk_ids[token_idx];

  if (expert_id < start_expert_id || expert_id > end_expert_id) return;
  const scalar_t* gate_output_ptr = gateup_output + static_cast<int64_t>(token_idx) * hidden_size;
  const scalar_t* up_output_ptr = gate_output_ptr + half_hidden_size;
  scalar_t* dst_ptr = down_input + static_cast<int64_t>(token_idx) * half_hidden_size;
  scalar_t scale_q = static_cast<scalar_t>(scales ? (1.f / scales[expert_id - start_expert_id]) : 1.f);

  const uint32_t vec_elements = half_hidden_size / vec_size;
#pragma unroll 1
  for (uint32_t idx = thread_idx; idx < vec_elements; idx += stride) {
    vec_t gate_vec, up_vec, out_vec;
    gate_vec.load(gate_output_ptr + idx * vec_size);
    up_vec.load(up_output_ptr + idx * vec_size);

#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      float gate_f = static_cast<float>(gate_vec[i]);
      scalar_t gate_q = silu_quantize<scalar_t>(gate_f);
      scalar_t prod = gate_q * up_vec[i] * scale_q;
      out_vec[i] = prod;
    }
    out_vec.store(dst_ptr + idx * vec_size);
  }

  const int64_t scalar_start = static_cast<int64_t>(vec_elements) * vec_size + thread_idx;
#pragma unroll 1
  for (int64_t idx = scalar_start; idx < half_hidden_size; idx += stride) {
    float gate_f = static_cast<float>(gate_output_ptr[idx]);
    scalar_t gate_q = silu_quantize<scalar_t>(gate_f);
    dst_ptr[idx] = gate_q * up_output_ptr[idx] * scale_q;
  }
}

void ep_moe_silu_and_mul(
    torch::Tensor gateup_output,
    torch::Tensor down_input,
    torch::Tensor reorder_topk_ids,
    torch::Tensor scales,
    int64_t start_expert_id,
    int64_t end_expert_id) {
  const int total_tokens = gateup_output.size(0);
  const int hidden_size = gateup_output.size(1);

  /*
  DPCT1038:647: When the kernel function name is used as a macro argument, the migration result may be incorrect. You
  need to verify the definition of the macro.
  */
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(gateup_output.scalar_type(), scalar_t, [&] {
    dpct::dim3 grid(total_tokens);
    constexpr uint32_t vec_size = 16 / sizeof(scalar_t);
    const int half_hidden_size = hidden_size >> 1;
    uint32_t threads = (half_hidden_size + vec_size - 1) / vec_size;
    threads = std::max<uint32_t>(threads, 256);
    threads = ((threads + 31) & ~31U);
    dpct::dim3 block(std::min(threads, 1024U));
    /*
    DPCT1049:648: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
      auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
      dpct::has_capability_or_fail(c10::xpu::getCurrentXPUStream().queue().get_device(), {sycl::aspect::fp16});

      c10::xpu::getCurrentXPUStream().queue().submit([&](sycl::handler& cgh) {
        auto gateup_output_data_ptr_ct0 = static_cast<scalar_t*>(gateup_output.data_ptr());
        auto down_input_data_ptr_ct1 = static_cast<scalar_t*>(down_input.data_ptr());
        const int* reorder_topk_ids_data_ptr_int_ct2 = reorder_topk_ids.data_ptr<int>();
        const float* scales_defined_scales_data_ptr_float_nullptr_ct3 =
            scales.defined() ? scales.data_ptr<float>() : nullptr;

        cgh.depends_on(dpct::get_current_device().get_in_order_queues_last_events());

        cgh.parallel_for(sycl::nd_range<3>(grid * block, block), exp_props, [=](sycl::nd_item<3> item_ct1) {
          ep_moe_act_and_mul_cuda_kernel<scalar_t>(
              gateup_output_data_ptr_ct0,
              down_input_data_ptr_ct1,
              reorder_topk_ids_data_ptr_int_ct2,
              scales_defined_scales_data_ptr_float_nullptr_ct3,
              static_cast<int>(start_expert_id),
              static_cast<int>(end_expert_id),
              hidden_size);
        });
      });
    }
    return true;
  });
}
