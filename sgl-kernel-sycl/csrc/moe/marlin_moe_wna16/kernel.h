
#ifndef MARLIN_NAMESPACE_NAME
#define MARLIN_NAMESPACE_NAME marlin_moe_wna16
#endif

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <c10/xpu/XPUStream.h>
#include "gptq_marlin/marlin.dp.hpp"
#include "gptq_marlin/marlin_dtypes.dp.hpp"
#include "scalar_type.hpp"

#define MARLIN_KERNEL_PARAMS                                                                                         \
  const sycl::int4 *__restrict__ A, const sycl::int4 *__restrict__ B, sycl::int4 *__restrict__ C,                    \
      sycl::int4 *__restrict__ C_tmp, const sycl::int4 *__restrict__ scales_ptr,                                     \
      const sycl::int4 *__restrict__ zp_ptr, const int *__restrict__ g_idx,                                          \
      const int32_t *__restrict__ sorted_token_ids_ptr, const int32_t *__restrict__ expert_ids_ptr,                  \
      const int32_t *__restrict__ num_tokens_past_padded_ptr, const float *__restrict__ topk_weights_ptr, int top_k, \
      bool mul_topk_weights, bool is_ep, int num_groups, int prob_m, int prob_n, int prob_k, int *locks,             \
      bool use_atomic_add, bool use_fp32_reduce

namespace MARLIN_NAMESPACE_NAME {
template <
    typename scalar_t,                     // compute dtype, half or nv_float16
    const sglang::ScalarTypeId w_type_id,  // weight ScalarType id
    const int threads,                     // number of threads in a threadblock
    const int thread_m_blocks,             // number of 16x16 blocks in the m
                                           // dimension (batchsize) of the
                                           // threadblock
    const int thread_n_blocks,             // same for n dimension (output)
    const int thread_k_blocks,             // same for k dimension (reduction)
    const bool m_block_size_8,             // whether m_block_size == 8
                                           // only works when thread_m_blocks == 1
    const int stages,                      // number of stages for the async global->shared
                                           // fetch pipeline
    const bool has_act_order,              // whether act_order is enabled
    const bool has_zp,                     // whether zero-points are enabled
    const int group_blocks,                // number of consecutive 16x16 blocks
                                           // with a separate quantization scale
    const bool is_zp_float                 // is zero point of float16 type?
    >
void Marlin(MARLIN_KERNEL_PARAMS, uint8_t* dpct_local);

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
template <
    typename scalar_t,
    const sglang::ScalarTypeId w_type_id,
    const int threads,
    const int thread_m_blocks,
    const int thread_n_blocks,
    const int thread_k_blocks,
    const bool m_block_size_8,
    const int stages,
    const bool has_act_order,
    const bool has_zp,
    const int group_blocks,
    const bool is_zp_float>
void Marlin_wrapper(MARLIN_KERNEL_PARAMS);
    //const sycl::int4* __restrict A,
    //const sycl::int4* __restrict B,
    //sycl::int4* __restrict C,
    //sycl::int4* __restrict C_tmp,
    //const sycl::int4* __restrict scales_ptr,
    //const sycl::int4* __restrict zp_ptr,
    //const int* __restrict g_idx,
    //const int32_t* __restrict sorted_token_ids_ptr,
    //const int32_t* __restrict expert_ids_ptr,
    //const int32_t* __restrict num_tokens_past_padded_ptr,
    //const float* __restrict topk_weights_ptr,
    //int top_k,
    //bool mul_topk_weights,
    //bool is_ep,
    //int num_groups,
    //int prob_m,
    //int prob_n,
    //int prob_k,
    //int* locks,
    //bool use_atomic_add,
    //bool use_fp32_reduce);
}
