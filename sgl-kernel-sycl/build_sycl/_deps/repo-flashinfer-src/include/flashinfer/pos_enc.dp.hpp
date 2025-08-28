/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_POS_ENC_CUH_
#define FLASHINFER_POS_ENC_CUH_

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <c10/xpu/XPUStream.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

#include "layout.dp.hpp"
//#include "math.dp.hpp"
#include "utils.dp.hpp"
#include "vec_dtypes.dp.hpp"

namespace flashinfer {

/*!
 * \brief An enumeration class that defines different modes for applying RoPE
 *   (Rotary Positional Embeddings).
 */
enum class PosEncodingMode {
  // No rotary positional embeddings
  kNone = 0U,
  // Apply Llama-style rope.
  kRoPELlama = 1U,
  // Apply ALiBi bias
  kALiBi = 2U
};

/*!
 * \brief Convert PosEncodingMode to string
 * \param pos_encoding_mode A PosEncodingMode value
 */
inline std::string PosEncodingModeToString(const PosEncodingMode& pos_encoding_mode) {
  switch (pos_encoding_mode) {
    case PosEncodingMode::kNone:
      return "None";
    case PosEncodingMode::kRoPELlama:
      return "Llama";
    case PosEncodingMode::kALiBi:
      return "ALiBi";
    default:
      return "Unknown";
  }
}

//__dpct_inline__ float get_alibi_slope(uint32_t head_idx, uint32_t num_heads) {
//  int n = math::ptx_exp2((int)math::ptx_log2(num_heads));
//  return head_idx < n ? math::ptx_exp2(-8. * float(head_idx + 1) / float(n))
//                      : math::ptx_exp2(-4. * float((head_idx + 1 - n) * 2 - 1) / float(n));
//}

/*!
 * \brief Apply RoPE (Rotary Positional Embeddings) to x[0: head_dim],
 *   return thread-local vector
 * \tparam vec_size A template integer indicates the vector size used
 *   in the kernel
 * \tparam bdx A template integer indicates the blockDim.x
 * \tparam T A template type indicates the x data type
 * \param x A pointer to the start of x data
 * \param freq A vector of float indicates the thread-local rope frequency
 * \param offset A integer indicates the offset of the position in RoPE
 */
template <uint32_t vec_size, uint32_t bdx, typename T>
__dpct_inline__ vec_t<float, vec_size> vec_apply_llama_rope(
    const T* x, const vec_t<float, vec_size>& freq, int32_t offset, const uint32_t rotary_dim = vec_size * bdx) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  vec_t<float, vec_size> permuted_vec, vec;
  vec.cast_load(x + item_ct1.get_local_id(2) * vec_size);

  if (item_ct1.get_local_id(2) * vec_size < rotary_dim) {
    permuted_vec.cast_load(
        x + ((item_ct1.get_local_id(2) * vec_size < rotary_dim / 2)
                 ? item_ct1.get_local_id(2) * vec_size + rotary_dim / 2
                 : item_ct1.get_local_id(2) * vec_size - rotary_dim / 2));
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      float embed = float(offset) * freq[i];
      float cos, sin;
      sin = sycl::sincos(
          embed,
          sycl::address_space_cast<sycl::access::address_space::generic_space, sycl::access::decorated::yes>(&cos));
      vec[i] = vec[i] * cos +
               ((item_ct1.get_local_id(2) * vec_size < rotary_dim / 2) ? -permuted_vec[i] : permuted_vec[i]) * sin;
    }
  }
  return vec;
}

template <uint32_t vec_size, uint32_t bdx, typename T>
/*
DPCT1110:268: The total declared local variable size in device function vec_apply_llama_rope_cos_sin exceeds 128 bytes
and may cause high register pressure. Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ vec_t<float, vec_size> vec_apply_llama_rope_cos_sin(
    const T* x,
    const vec_t<float, vec_size>& cos,
    const vec_t<float, vec_size>& sin,
    const uint32_t rotary_dim = vec_size * bdx) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  vec_t<float, vec_size> permuted_vec, vec;
  vec.cast_load(x + item_ct1.get_local_id(2) * vec_size);

  if (item_ct1.get_local_id(2) * vec_size < rotary_dim) {
    permuted_vec.cast_load(
        x + ((item_ct1.get_local_id(2) * vec_size < rotary_dim / 2)
                 ? item_ct1.get_local_id(2) * vec_size + rotary_dim / 2
                 : item_ct1.get_local_id(2) * vec_size - rotary_dim / 2));
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      vec[i] = vec[i] * cos[i] +
               ((item_ct1.get_local_id(2) * vec_size < rotary_dim / 2) ? -permuted_vec[i] : permuted_vec[i]) * sin[i];
    }
  }
  return vec;
}

/*!
 * \brief Apply RoPE (Rotary Positional Embeddings) to x[0: head_dim] with interleave,
 *   return thread-local vector.
 * \tparam vec_size A template integer indicates the vector size used
 *   in the kernel
 * \tparam bdx A template integer indicates the blockDim.x
 * \tparam T A template type indicates the x data type
 * \param x A pointer to the start of x data
 * \param freq A vector of float indicates the thread-local rope frequency
 * \param offset A integer indicates the offset of the position in RoPE
 */
template <uint32_t vec_size, uint32_t bdx, typename T>
__dpct_inline__ vec_t<float, vec_size> vec_apply_llama_rope_interleave(
    const T* x, const vec_t<float, vec_size>& freq, int32_t offset, const uint32_t rotary_dim = vec_size * bdx) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  vec_t<float, vec_size> vec, vec_before;
  vec.cast_load(x + item_ct1.get_local_id(2) * vec_size);

  if (item_ct1.get_local_id(2) * vec_size < rotary_dim) {
    vec_before = vec;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      float embed = float(offset) * freq[i];
      float cos, sin;
      sin = sycl::sincos(
          embed,
          sycl::address_space_cast<sycl::access::address_space::generic_space, sycl::access::decorated::yes>(&cos));
      vec[i] = vec[i] * cos + ((i % 2 == 0) ? -vec_before[i ^ 1] : vec_before[i ^ 1]) * sin;
    }
  }
  return vec;
}

template <uint32_t vec_size, uint32_t bdx, typename T>
__dpct_inline__ vec_t<float, vec_size> vec_apply_llama_rope_cos_sin_interleave(
    const T* x,
    const vec_t<float, vec_size>& cos,
    const vec_t<float, vec_size>& sin,
    const uint32_t rotary_dim = vec_size * bdx) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  vec_t<float, vec_size> vec, vec_before;
  vec.cast_load(x + item_ct1.get_local_id(2) * vec_size);

  if (item_ct1.get_local_id(2) * vec_size < rotary_dim) {
    vec_before = vec;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      vec[i] = vec[i] * cos[i] + ((i % 2 == 0) ? -vec_before[i ^ 1] : vec_before[i ^ 1]) * sin[i];
    }
  }
  return vec;
}

/*
HACK (ByronHsu): in the interleave mode with cos_sin_cache, we actually only use the first half of
cos and sin

For example,
In the below example, the vec_size is 4
the computation in the kernel is:
    [x1, x2, x3, x4...] * [cos1, cos1, cos2, cos2] + [-x2, x1, -x4, x3...] * [sin1, sin1, sin2,
sin2] the data we loaded are:
    - loaded vec = [x1, x2, x3, x4]
    - loaded cos = [cos1, cos2, cos3, cos4]
    - loaded sin = [sin1, sin2, sin3, sin4]
But only the first half of cos and sin is used in the computation.

However, we argue the additional overhead is acceptable:
    1. loading additional elements of cos and sin is not adding much overhead. The arithmetic
intensity is the same as non-interleave mode. Each elements of cos and sin is load twice
    2. we don't want two code paths of cos and sin vector for interleave and non-interleave mode.
*/
template <uint32_t vec_size, uint32_t bdx, typename T>
/*
DPCT1110:269: The total declared local variable size in device function
vec_apply_llama_rope_cos_sin_interleave_reuse_half exceeds 128 bytes and may cause high register pressure. Consult with
your hardware vendor to find the total register size available and adjust the code, or use smaller sub-group size to
avoid high register pressure.
*/
__dpct_inline__ vec_t<float, vec_size> vec_apply_llama_rope_cos_sin_interleave_reuse_half(
    const T* x,
    const vec_t<float, vec_size>& cos,
    const vec_t<float, vec_size>& sin,
    const uint32_t rotary_dim = vec_size * bdx) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  vec_t<float, vec_size> vec, vec_before;
  vec.cast_load(x + item_ct1.get_local_id(2) * vec_size);

  if (item_ct1.get_local_id(2) * vec_size < rotary_dim) {
    vec_before = vec;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      // i / 2 is to get the index of the first half of cos and sin
      vec[i] = vec[i] * cos[i / 2] +
               ((i % 2 == 0) ? -vec_before[i ^ 1] : vec_before[i ^ 1]) * sin[i / 2];
    }
  }
  return vec;
}

template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType, typename IdType>
/*
DPCT1110:270: The total declared local variable size in device function
BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust the code, or use smaller sub-group size
to avoid high register pressure.
*/
void BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel(
    DType* q,
    DType* k,
    DType* q_rope,
    DType* k_rope,
    float* __restrict__ cos_sin_cache,
    IdType* __restrict__ pos_ids,
    uint32_t nnz,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t rotary_dim,
    size_t q_stride_n,
    size_t q_stride_h,
    size_t k_stride_n,
    size_t k_stride_h,
    size_t q_rope_stride_n,
    size_t q_rope_stride_h,
    size_t k_rope_stride_n,
    size_t k_rope_stride_h) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  uint32_t bx = item_ct1.get_group(2), tx = item_ct1.get_local_id(2), ty = item_ct1.get_local_id(1);
  uint32_t by = item_ct1.get_group(1);
  const uint32_t bdy = item_ct1.get_local_range(1);

  vec_t<float, vec_size> cos, sin;
  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];

    const int half_rotary_dim = rotary_dim / 2;

    // 1. if interleave:
    //  - cos = cos_sin_cache[pos_id][tx * vec_size // 2]
    //  - sin = cos_sin_cache[pos_id][(rot_dim // 2) + tx * vec_size // 2]
    // 2. if not interleave
    //  - cos = cos_cache[pos_id][(tx * vec_size) % (rot_dim // 2)]
    //  - sin = sin_cache[pos_id][(rot_dim // 2) + (tx * vec_size) % (rot_dim // 2)]
    if (tx * vec_size < rotary_dim) {
      int sin_offset = rotary_dim / 2;
      int vec_idx;
      if constexpr (interleave) {
        vec_idx = (tx * vec_size) / 2;  // Force integer division
      } else {
        vec_idx = (tx * vec_size) % half_rotary_dim;  // Use half_rotary_dim
      }
      cos.load(cos_sin_cache + (pos * rotary_dim) + vec_idx);
      sin.load(cos_sin_cache + (pos * rotary_dim) + (sin_offset + vec_idx));
    }

    if (by < num_qo_heads) {
      uint32_t qo_head_idx = by;
      DType* q_ptr = q + get_elem_offset_impl(idx, qo_head_idx, 0, q_stride_n, q_stride_h);
      DType* q_rope_ptr =
          q_rope + get_elem_offset_impl(idx, qo_head_idx, 0, q_rope_stride_n, q_rope_stride_h);
      vec_t<float, vec_size> q_vec;
      if constexpr (interleave) {
        q_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(q_ptr, cos, sin,
                                                                                  rotary_dim);
      } else {
        q_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      }
      q_vec.cast_store(q_rope_ptr + tx * vec_size);
    } else {
      uint32_t kv_head_idx = by - num_qo_heads;
      DType* k_ptr = k + get_elem_offset_impl(idx, kv_head_idx, 0, k_stride_n, k_stride_h);
      DType* k_rope_ptr =
          k_rope + get_elem_offset_impl(idx, kv_head_idx, 0, k_rope_stride_n, k_rope_stride_h);
      vec_t<float, vec_size> k_vec;
      if constexpr (interleave) {
        k_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(k_ptr, cos, sin,
                                                                                  rotary_dim);
      } else {
        k_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      }
      k_vec.cast_store(k_rope_ptr + tx * vec_size);
    }
  }
}
// need
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType, typename IdType>
void BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel_wrapper(
    DType* q,
    DType* k,
    DType* q_rope,
    DType* k_rope,
    float* __restrict cos_sin_cache,
    IdType* __restrict pos_ids,
    uint32_t nnz,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t rotary_dim,
    size_t q_stride_n,
    size_t q_stride_h,
    size_t k_stride_n,
    size_t k_stride_h,
    size_t q_rope_stride_n,
    size_t q_rope_stride_h,
    size_t k_rope_stride_n,
    size_t k_rope_stride_h) {
   sycl::queue queue = *dpct::kernel_launcher::_que;
   unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
   sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

   //auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
   dpct::has_capability_or_fail(c10::xpu::getCurrentXPUStream().queue().get_device(), {sycl::aspect::fp16});

   queue.submit([&](sycl::handler& cgh) {
      auto last_event = dpct::get_default_queue().ext_oneapi_get_last_event();
      [&](auto&& _e) {
         if constexpr (std::is_same_v<std::remove_reference_t<decltype(last_event)>, sycl::event>)
            cgh.depends_on(_e);
         else if (_e.has_value())
            cgh.depends_on(_e.value());
      }(last_event);

      cgh.parallel_for(nr, [=](sycl::nd_item<3> item_ct1) {
         BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel<interleave, head_dim, vec_size, bdx, DType, IdType>(
             q,
             k,
             q_rope,
             k_rope,
             cos_sin_cache,
             pos_ids,
             nnz,
             num_qo_heads,
             num_kv_heads,
             rotary_dim,
             q_stride_n,
             q_stride_h,
             k_stride_n,
             k_stride_h,
             q_rope_stride_n,
             q_rope_stride_h,
             k_rope_stride_n,
             k_rope_stride_h);
      });
   });
}

template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType, typename IdType>
/*
DPCT1110:271: The total declared local variable size in device function BatchQKApplyRotaryPosIdsCosSinCacheKernel
exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total register
size available and adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void BatchQKApplyRotaryPosIdsCosSinCacheKernel(
    DType* q,
    DType* k,
    DType* q_rope,
    DType* k_rope,
    float* __restrict__ cos_sin_cache,
    IdType* __restrict__ pos_ids,
    uint32_t nnz,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t rotary_dim,
    size_t q_stride_n,
    size_t q_stride_h,
    size_t k_stride_n,
    size_t k_stride_h,
    size_t q_rope_stride_n,
    size_t q_rope_stride_h,
    size_t k_rope_stride_n,
    size_t k_rope_stride_h) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  uint32_t bx = item_ct1.get_group(2), tx = item_ct1.get_local_id(2), ty = item_ct1.get_local_id(1);
  const uint32_t bdy = item_ct1.get_local_range(1);

  vec_t<float, vec_size> cos, sin;
  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];
    const int half_rotary_dim = rotary_dim / 2;

    // 1. if interleave:
    //  - cos = cos_sin_cache[pos_id][tx * vec_size // 2]
    //  - sin = cos_sin_cache[pos_id][(rot_dim // 2) + tx * vec_size // 2]
    // 2. if not interleave
    //  - cos = cos_cache[pos_id][(tx * vec_size) % (rot_dim // 2)]
    //  - sin = sin_cache[pos_id][(rot_dim // 2) + (tx * vec_size) % (rot_dim // 2)]
    if (tx * vec_size < rotary_dim) {
      int sin_offset = rotary_dim / 2;
      int vec_idx;
      if constexpr (interleave) {
        vec_idx = (tx * vec_size) / 2;  // Force integer division
      } else {
        vec_idx = (tx * vec_size) % half_rotary_dim;  // Use half_rotary_dim
      }
      cos.load(cos_sin_cache + (pos * rotary_dim) + vec_idx);
      sin.load(cos_sin_cache + (pos * rotary_dim) + (sin_offset + vec_idx));
    }

    // not to unroll the loop, because num head might be large and might lead to worse performance
#pragma unroll 1
    for (uint32_t qo_head_idx = 0; qo_head_idx < num_qo_heads; ++qo_head_idx) {
      DType* q_ptr = q + get_elem_offset_impl(idx, qo_head_idx, 0, q_stride_n, q_stride_h);
      DType* q_rope_ptr =
          q_rope + get_elem_offset_impl(idx, qo_head_idx, 0, q_rope_stride_n, q_rope_stride_h);
      vec_t<float, vec_size> q_vec;
      if constexpr (interleave) {
        q_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(q_ptr, cos, sin,
                                                                                  rotary_dim);
      } else {
        q_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      }
      q_vec.cast_store(q_rope_ptr + tx * vec_size);
    }

#pragma unroll 1
    for (uint32_t kv_head_idx = 0; kv_head_idx < num_kv_heads; ++kv_head_idx) {
      DType* k_ptr = k + get_elem_offset_impl(idx, kv_head_idx, 0, k_stride_n, k_stride_h);
      DType* k_rope_ptr =
          k_rope + get_elem_offset_impl(idx, kv_head_idx, 0, k_rope_stride_n, k_rope_stride_h);
      vec_t<float, vec_size> k_vec;
      if constexpr (interleave) {
        k_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(k_ptr, cos, sin,
                                                                                  rotary_dim);
      } else {
        k_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      }
      k_vec.cast_store(k_rope_ptr + tx * vec_size);
    }
  }
}
// need
// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType, typename IdType>
void BatchQKApplyRotaryPosIdsCosSinCacheKernel_wrapper(
    DType* q,
    DType* k,
    DType* q_rope,
    DType* k_rope,
    float* __restrict cos_sin_cache,
    IdType* __restrict pos_ids,
    uint32_t nnz,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t rotary_dim,
    size_t q_stride_n,
    size_t q_stride_h,
    size_t k_stride_n,
    size_t k_stride_h,
    size_t q_rope_stride_n,
    size_t q_rope_stride_h,
    size_t k_rope_stride_n,
    size_t k_rope_stride_h) {
   sycl::queue queue = *dpct::kernel_launcher::_que;
   unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
   sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

   //auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
   dpct::has_capability_or_fail(c10::xpu::getCurrentXPUStream().queue().get_device(), {sycl::aspect::fp16});

   queue.submit([&](sycl::handler& cgh) {
      auto last_event = dpct::get_default_queue().ext_oneapi_get_last_event();
      [&](auto&& _e) {
         if constexpr (std::is_same_v<std::remove_reference_t<decltype(last_event)>, sycl::event>)
            cgh.depends_on(_e);
         else if (_e.has_value())
            cgh.depends_on(_e.value());
      }(last_event);

      cgh.parallel_for(nr,  [=](sycl::nd_item<3> item_ct1) {
         BatchQKApplyRotaryPosIdsCosSinCacheKernel<interleave, head_dim, vec_size, bdx, DType, IdType>(
             q,
             k,
             q_rope,
             k_rope,
             cos_sin_cache,
             pos_ids,
             nnz,
             num_qo_heads,
             num_kv_heads,
             rotary_dim,
             q_stride_n,
             q_stride_h,
             k_stride_n,
             k_stride_h,
             q_rope_stride_n,
             q_rope_stride_h,
             k_rope_stride_n,
             k_rope_stride_h);
      });
   });
}

template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType>
void BatchQKApplyRotaryPosIdsHeadParallelismKernel(
    DType* q, DType* k, DType* q_rope, DType* k_rope, IdType* __restrict__ pos_ids, uint32_t nnz,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rotary_dim, size_t q_stride_n,
    size_t q_stride_h, size_t k_stride_n, size_t k_stride_h, size_t q_rope_stride_n,
    size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h, float smooth_a,
    float smooth_b, float rope_rcp_scale, float rope_rcp_theta) {
  // NOTE: q and q_rope may be the same ptr, so do k and k_rope
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  uint32_t bx = item_ct1.get_group(2), tx = item_ct1.get_local_id(2), ty = item_ct1.get_local_id(1);
  uint32_t by = item_ct1.get_group(1);
  const uint32_t bdy = item_ct1.get_local_range(1);
  vec_t<float, vec_size> freq;
  if (tx * vec_size < rotary_dim) {
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      if constexpr (interleave) {
        freq[i] = dpct::pow(rope_rcp_theta, float(2 * ((tx * vec_size + i) / 2)) / float(rotary_dim));
      } else {
        freq[i] = dpct::pow(rope_rcp_theta, float(2 * ((tx * vec_size + i) % (rotary_dim / 2))) / float(rotary_dim));
      }

      float smooth = freq[i] * smooth_a + smooth_b;
      smooth = sycl::max(0.0f, sycl::min(1.0f, smooth));  // clamp to [0, 1]
      freq[i] = (1 - smooth) * (freq[i] * rope_rcp_scale) + smooth * freq[i];
    }
  }

  vec_t<float, vec_size> cos, sin;

  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];

    if (tx * vec_size < rotary_dim) {
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        float embed = float(pos) * freq[i];
        sin[i] = sycl::sincos(
            embed,
            sycl::address_space_cast<sycl::access::address_space::generic_space, sycl::access::decorated::yes>(
                &cos[i]));
      }
    }

    if (by < num_qo_heads) {
      uint32_t qo_head_idx = by;
      DType* q_ptr = q + get_elem_offset_impl(idx, qo_head_idx, 0, q_stride_n, q_stride_h);
      DType* q_rope_ptr =
          q_rope + get_elem_offset_impl(idx, qo_head_idx, 0, q_rope_stride_n, q_rope_stride_h);
      vec_t<float, vec_size> q_vec;
      if constexpr (interleave) {
        q_vec = vec_apply_llama_rope_cos_sin_interleave<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      } else {
        q_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      }
      q_vec.cast_store(q_rope_ptr + tx * vec_size);
    } else {
      uint32_t kv_head_idx = by - num_qo_heads;
      DType* k_ptr = k + get_elem_offset_impl(idx, kv_head_idx, 0, k_stride_n, k_stride_h);
      DType* k_rope_ptr =
          k_rope + get_elem_offset_impl(idx, kv_head_idx, 0, k_rope_stride_n, k_rope_stride_h);
      vec_t<float, vec_size> k_vec;
      if constexpr (interleave) {
        k_vec = vec_apply_llama_rope_cos_sin_interleave<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      } else {
        k_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      }
      k_vec.cast_store(k_rope_ptr + tx * vec_size);
    }
  }
}

template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType>
void BatchQKApplyRotaryPosIdsKernel(
    DType* q, DType* k, DType* q_rope, DType* k_rope, IdType* __restrict__ pos_ids, uint32_t nnz,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rotary_dim, size_t q_stride_n,
    size_t q_stride_h, size_t k_stride_n, size_t k_stride_h, size_t q_rope_stride_n,
    size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h, float smooth_a,
    float smooth_b, float rope_rcp_scale, float rope_rcp_theta) {
  // NOTE: q and q_rope may be the same ptr, so do k and k_rope
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  uint32_t bx = item_ct1.get_group(2), tx = item_ct1.get_local_id(2), ty = item_ct1.get_local_id(1);
  const uint32_t bdy = item_ct1.get_local_range(1);
  vec_t<float, vec_size> freq;
  if (tx * vec_size < rotary_dim) {
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      if constexpr (interleave) {
        freq[i] = dpct::pow(rope_rcp_theta, float(2 * ((tx * vec_size + i) / 2)) / float(rotary_dim));
      } else {
        freq[i] = dpct::pow(rope_rcp_theta, float(2 * ((tx * vec_size + i) % (rotary_dim / 2))) / float(rotary_dim));
      }

      float smooth = freq[i] * smooth_a + smooth_b;
      smooth = sycl::max(0.0f, sycl::min(1.0f, smooth));  // clamp to [0, 1]
      freq[i] = (1 - smooth) * (freq[i] * rope_rcp_scale) + smooth * freq[i];
    }
  }

  vec_t<float, vec_size> cos, sin;

  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];

    if (tx * vec_size < rotary_dim) {
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        float embed = float(pos) * freq[i];
        sin[i] = sycl::sincos(
            embed,
            sycl::address_space_cast<sycl::access::address_space::generic_space, sycl::access::decorated::yes>(
                &cos[i]));
      }
    }

#pragma unroll 1
    for (uint32_t qo_head_idx = 0; qo_head_idx < num_qo_heads; ++qo_head_idx) {
      DType* q_ptr = q + get_elem_offset_impl(idx, qo_head_idx, 0, q_stride_n, q_stride_h);
      DType* q_rope_ptr =
          q_rope + get_elem_offset_impl(idx, qo_head_idx, 0, q_rope_stride_n, q_rope_stride_h);
      vec_t<float, vec_size> q_vec;
      if constexpr (interleave) {
        q_vec = vec_apply_llama_rope_cos_sin_interleave<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      } else {
        q_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      }
      q_vec.cast_store(q_rope_ptr + tx * vec_size);
    }

#pragma unroll 1
    for (uint32_t kv_head_idx = 0; kv_head_idx < num_kv_heads; ++kv_head_idx) {
      DType* k_ptr = k + get_elem_offset_impl(idx, kv_head_idx, 0, k_stride_n, k_stride_h);
      DType* k_rope_ptr =
          k_rope + get_elem_offset_impl(idx, kv_head_idx, 0, k_rope_stride_n, k_rope_stride_h);
      vec_t<float, vec_size> k_vec;
      if constexpr (interleave) {
        k_vec = vec_apply_llama_rope_cos_sin_interleave<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      } else {
        k_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      }
      k_vec.cast_store(k_rope_ptr + tx * vec_size);
    }
  }
}

template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType>
void BatchQKApplyRotaryKernel(
    DType* q, DType* k, DType* q_rope, DType* k_rope, IdType* __restrict__ indptr,
    IdType* __restrict__ offsets, uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t rotary_dim, size_t q_stride_n, size_t q_stride_h, size_t k_stride_n, size_t k_stride_h,
    size_t q_rope_stride_n, size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h,
    float smooth_a, float smooth_b, float rope_rcp_scale, float rope_rcp_theta) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  uint32_t bx = item_ct1.get_group(2), tx = item_ct1.get_local_id(2), ty = item_ct1.get_local_id(1);
  const uint32_t bdy = item_ct1.get_local_range(1);
  vec_t<float, vec_size> freq;
  if (tx * vec_size < rotary_dim) {
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      if constexpr (interleave) {
        freq[i] = dpct::pow(rope_rcp_theta, float(2 * ((tx * vec_size + i) / 2)) / float(rotary_dim));
      } else {
        freq[i] = dpct::pow(rope_rcp_theta, float(2 * ((tx * vec_size + i) % (rotary_dim / 2))) / float(rotary_dim));
      }

      float smooth = freq[i] * smooth_a + smooth_b;
      smooth = sycl::max(0.0f, sycl::min(1.0f, smooth));  // clamp to [0, 1]
      freq[i] = (1 - smooth) * (freq[i] * rope_rcp_scale) + smooth * freq[i];
    }
  }

  if (bx < batch_size * num_qo_heads) {
    // apply rotary to q
    const uint32_t batch_idx = bx / num_qo_heads;
    const uint32_t qo_head_idx = bx % num_qo_heads;
    const uint32_t seq_len = indptr[batch_idx + 1] - indptr[batch_idx];
    const uint32_t offset = offsets[batch_idx];
#pragma unroll 2
    for (uint32_t i = 0; i < (seq_len + bdy - 1) / bdy; ++i) {
      vec_t<float, vec_size> q_vec;
      if (i * bdy + ty < seq_len) {
        DType* q_ptr = q + get_elem_offset_impl(indptr[batch_idx] + i * bdy + ty, qo_head_idx, 0,
                                                q_stride_n, q_stride_h);
        DType* q_rope_ptr =
            q_rope + get_elem_offset_impl(indptr[batch_idx] + i * bdy + ty, qo_head_idx, 0,
                                          q_rope_stride_n, q_rope_stride_h);
        if constexpr (interleave) {
          q_vec = vec_apply_llama_rope_interleave<vec_size, bdx>(q_ptr, freq, offset + i * bdy + ty,
                                                                 rotary_dim);
        } else {
          q_vec =
              vec_apply_llama_rope<vec_size, bdx>(q_ptr, freq, offset + i * bdy + ty, rotary_dim);
        }
        q_vec.cast_store(q_rope_ptr + tx * vec_size);
      }
    }
  } else {
    // apply rotary to k
    uint32_t batch_idx = (bx - batch_size * num_qo_heads) / num_kv_heads;
    uint32_t kv_head_idx = (bx - batch_size * num_qo_heads) % num_kv_heads;
    const uint32_t seq_len = indptr[batch_idx + 1] - indptr[batch_idx];
    const uint32_t offset = offsets[batch_idx];
#pragma unroll 2
    for (uint32_t i = 0; i < (seq_len + bdy - 1) / bdy; ++i) {
      vec_t<float, vec_size> k_vec;
      if (i * bdy + ty < seq_len) {
        DType* k_ptr = k + get_elem_offset_impl(indptr[batch_idx] + i * bdy + ty, kv_head_idx, 0,
                                                k_stride_n, k_stride_h);
        DType* k_rope_ptr =
            k_rope + get_elem_offset_impl(indptr[batch_idx] + i * bdy + ty, kv_head_idx, 0,
                                          k_rope_stride_n, k_rope_stride_h);
        if constexpr (interleave) {
          k_vec = vec_apply_llama_rope_interleave<vec_size, bdx>(k_ptr, freq, offset + i * bdy + ty,
                                                                 rotary_dim);
        } else {
          k_vec =
              vec_apply_llama_rope<vec_size, bdx>(k_ptr, freq, offset + i * bdy + ty, rotary_dim);
        }
        k_vec.cast_store(k_rope_ptr + tx * vec_size);
      }
    }
  }
}

#define DISPATCH_INTERLEAVE(interleave, INTERLEAVE, ...) \
  if (interleave) {                                      \
    const bool INTERLEAVE = true;                        \
    __VA_ARGS__                                          \
  } else {                                               \
    const bool INTERLEAVE = false;                       \
    __VA_ARGS__                                          \
  }

// need
template <typename DType, typename IdType>
dpct::err0 BatchQKApplyRotaryPosIdsCosSinCache(
    DType* q,
    DType* k,
    DType* q_rope,
    DType* k_rope,
    float* cos_sin_cache,
    IdType* pos_ids,
    uint32_t nnz,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t rotary_dim,
    uint32_t head_dim,
    size_t q_stride_n,
    size_t q_stride_h,
    size_t k_stride_n,
    size_t k_stride_h,
    size_t q_rope_stride_n,
    size_t q_rope_stride_h,
    size_t k_rope_stride_n,
    size_t k_rope_stride_h,
    bool interleave,
    dpct::queue_ptr stream = &c10::xpu::getCurrentXPUStream().queue()) try {
  int dev_id = 0;
  int num_sms = 0;
  FLASHINFER_CUDA_CALL(DPCT_CHECK_ERROR(dev_id = dpct::get_current_device_id()));
  FLASHINFER_CUDA_CALL(DPCT_CHECK_ERROR(num_sms = dpct::get_device(dev_id).get_max_compute_units()));

  /*
  DPCT1111:488: Please verify the input arguments of "dpct::experimental::calculate_max_active_wg_per_xecore" base on
  the target function "kernel_0".
  */
  DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      // operate on 16 Bytes at a time
      constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
      // how many threads needed per head_dim
      constexpr uint32_t bdx = HEAD_DIM / vec_size;
      // how many threads needed per block
      uint32_t num_threads = std::max(128U, bdx);
      // how many tokens can we process in a block
      uint32_t bdy = num_threads / bdx;
      // how many blocks needed to process all tokens
      uint32_t nblks_x = (nnz + bdy - 1) / bdy;
      void* args[] = {(void*)&q,
                      (void*)&k,
                      (void*)&q_rope,
                      (void*)&k_rope,
                      (void*)&cos_sin_cache,
                      (void*)&pos_ids,
                      (void*)&nnz,
                      (void*)&num_qo_heads,
                      (void*)&num_kv_heads,
                      (void*)&rotary_dim,
                      (void*)&q_stride_n,
                      (void*)&q_stride_h,
                      (void*)&k_stride_n,
                      (void*)&k_stride_h,
                      (void*)&q_rope_stride_n,
                      (void*)&q_rope_stride_h,
                      (void*)&k_rope_stride_n,
                      (void*)&k_rope_stride_h};
      using Fn1 = void (*)(
        DType*, DType*, DType*, DType*,
        float*, IdType*,
        uint32_t, uint32_t, uint32_t, uint32_t,
        size_t, size_t, size_t, size_t,
        size_t, size_t, size_t, size_t
      );

      Fn1 kernel_0 =
          dpct::wrapper_register(
              BatchQKApplyRotaryPosIdsCosSinCacheKernel_wrapper<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>)
              .get();

      int num_blocks_per_sm_0 = 0;
      FLASHINFER_CUDA_CALL(
          dpct::experimental::calculate_max_active_wg_per_xecore(
              &num_blocks_per_sm_0, num_threads, 0 /* total share local memory size */));
      uint32_t num_ctas_0 = num_blocks_per_sm_0 * num_sms;

      if ((nnz + bdy - 1) / bdy >= num_ctas_0) {
        dpct::dim3 nblks(nblks_x);
        dpct::dim3 nthrs(bdx, bdy);
        FLASHINFER_CUDA_CALL(DPCT_CHECK_ERROR(dpct::kernel_launcher::launch(kernel_0, nblks, nthrs, args, 0, stream)));
      } else {
        dpct::dim3 nblks(nblks_x, num_qo_heads + num_kv_heads);
        dpct::dim3 nthrs(bdx, bdy);
        Fn1 kernel_1 =
            dpct::wrapper_register(BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel_wrapper<
                                                           INTERLEAVE,
                                                           HEAD_DIM,
                                                           vec_size,
                                                           bdx,
                                                           DType,
                                                           IdType>)
                .get();
        FLASHINFER_CUDA_CALL(DPCT_CHECK_ERROR(dpct::kernel_launcher::launch(kernel_1, nblks, nthrs, args, 0, stream)));
      }
    });
  });

  return 0;
}
catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}




}  // namespace flashinfer

#endif  // FLASHINFER_POS_ENC_CUH_
