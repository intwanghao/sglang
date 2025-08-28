#define DPCT_COMPAT_RT_MAJOR_VERSION 12
/*
 * Copyright (c) 2024 by FlashInfer team.
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

#ifndef FLASHINFER_ACTIVATION_CUH_
#define FLASHINFER_ACTIVATION_CUH_

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <c10/xpu/XPUStream.h>
//#include "math.dp.hpp"
//#include "utils.dp.hpp"
#include "vec_dtypes.dp.hpp"

namespace flashinfer {
namespace math {
__dpct_inline__ sycl::half2 uint32_as_half2(uint32_t x) { return *(sycl::half2*)&x; }

__dpct_inline__ uint32_t half2_as_uint32(sycl::half2 x) { return *(uint32_t*)&x; }

/*!
 * \brief Wrapper of PTX tanh.approx.f32 instruction, which computes tanh(x)
 * \param x input
 */
__dpct_inline__ float tanh(float x) {
  float y;
   y = sycl::tanh(x);
  return y;
}

/*!
 * \brief Wrapper of PTX tanh.approx.f16x2 instruction, which computes tanh(x)
 * \param x input
 */
__dpct_inline__ sycl::half2 tanh(sycl::half2 x) {
  uint32_t y_u32;
  uint32_t x_u32 = half2_as_uint32(x);
   y_u32 =
       sycl::tanh(sycl::vec<uint32_t, 1>(x_u32).template as<sycl::half2>()).template as<sycl::vec<uint32_t, 1>>().x();
  return uint32_as_half2(y_u32);
}

/*!
 * \brief Wrapper of PTX tanh.approx.f16 instruction, which computes tanh(x)
 * \param x input
 */
__dpct_inline__ sycl::half tanh(sycl::half x) {
  return sycl::tanh(x);
  //ushort y_u16;
  // y_u16 = sycl::tanh(sycl::bit_cast<unsigned short, sycl::half>(x));
  //return sycl::bit_cast<sycl::half, unsigned short>(y_u16);
}


}}


namespace flashinfer {


namespace activation {

__dpct_inline__ float silu(const float& val) {
  return val / (1.0f + sycl::native::exp(-val));
}

 __dpct_inline__ float gelu(const float& val) {
  constexpr float kAlpha = M_SQRT1_2;
  return val * 0.5f * (1.0f + sycl::erf(val * kAlpha));
}

 __dpct_inline__ float gelu_tanh(const float& val) {
  const float cdf = 0.5f * (1.0f + math::tanh((0.7978845608028654f * (val + 0.044715f * val * val * val))));
  return val * cdf;
}

template<int index>
 __dpct_inline__ float activation(const float& val) {
    if(index == 0) {
      return silu(val);
    } else if(index == 2) {
      return gelu(val);
    } 
    return gelu_tanh(val);
}

template <typename T, int type>
void act_and_mul_kernel(T* __restrict__ out, const T* __restrict__ input, const int d) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  constexpr uint32_t vec_size = 16 / sizeof(T);
  const int64_t token_idx = item_ct1.get_group(2);
  const int64_t thread_idx = item_ct1.get_local_id(2);
  const int64_t stride = item_ct1.get_local_range(2);
  const int64_t offset = token_idx * 2 * d;

#if (DPCT_COMPAT_RT_MAJOR_VERSION >= 12 && defined(DPCT_COMPATIBILITY_TEMP) && (DPCT_COMPATIBILITY_TEMP >= 900))
  /*
  DPCT1053:25: Migration of device assembly code is not supported.
  */
  //asm volatile("griddepcontrol.wait;");
#endif
#pragma unroll 1
  for (uint32_t idx = thread_idx; idx < d / vec_size; idx += stride) {
    vec_t<float, vec_size> x_vec, y_vec, out_vec;
    x_vec.cast_load(input + offset + idx * vec_size);
    y_vec.cast_load(input + offset + d + idx * vec_size);
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      out_vec[i] = activation<type>(x_vec[i]) * y_vec[i];
      //out_vec[i] = 12;
    }
    out_vec.cast_store(out + token_idx * d + idx * vec_size);
  }
  const int64_t remaining_offset = d - d % (stride * vec_size);
  // process the remaining elements
#pragma unroll 1
  for (int64_t idx = thread_idx; idx < d % (stride * vec_size); idx += stride) {
    float x = input[offset + remaining_offset + idx];
    float y = input[offset + remaining_offset + d + idx];
    out[token_idx * d + remaining_offset + idx] = activation<type>(x) * y;
    //out[token_idx * d + remaining_offset + idx] = 12;
    //Activation(0);
  }
#if (DPCT_COMPAT_RT_MAJOR_VERSION >= 12 && defined(DPCT_COMPATIBILITY_TEMP) && (DPCT_COMPATIBILITY_TEMP >= 900))
  /*
  DPCT1053:26: Migration of device assembly code is not supported.
  */
  //asm volatile("griddepcontrol.launch_dependents;");
#endif
}

}  // namespace activation
}  // namespace flashinfer

#endif  // FLASHINFER_ACTIVATION_CUH_
