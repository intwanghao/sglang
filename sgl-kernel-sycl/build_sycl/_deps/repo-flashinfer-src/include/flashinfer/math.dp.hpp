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
#ifndef FLASHINFER_MATH_CUH_
#define FLASHINFER_MATH_CUH_

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <c10/xpu/XPUStream.h>

#include <cstdint>

namespace flashinfer {
namespace math {

// log2(e)
constexpr float log2e = 1.44269504088896340736f;

constexpr float loge2 = 0.693147180559945309417f;

constexpr float inf = 5e4;

__dpct_inline__ sycl::half2 uint32_as_half2(uint32_t x) { return *(sycl::half2*)&x; }

__dpct_inline__ uint32_t half2_as_uint32(sycl::half2 x) { return *(uint32_t*)&x; }

/*!
 * \brief Wrapper of PTX ex2.approx instruction, which computes 2^x
 * \param x input
 */
__dpct_inline__ float ptx_exp2(float x) {
  float y;
   y = sycl::pow(2, x);
  return y;
}

/*!
 * \brief Wrapper of PTX lg2.approx instruction, which computes log2(x)
 * \param x input
 */
__dpct_inline__ float ptx_log2(float x) {
  float y;
   y = sycl::log2(x);
  return y;
}

/*!
 * \brief Wrapper of PTX ex2.approx.f16x2 instruction, which computes 2^x
 * \param x input
 */
__dpct_inline__ sycl::half2 ptx_exp2(sycl::half2 x) {
  uint32_t y_u32;
  uint32_t x_u32 = half2_as_uint32(x);
   y_u32 =
       sycl::pow(2, sycl::vec<uint32_t, 1>(x_u32).template as<sycl::half2>()).template as<sycl::vec<uint32_t, 1>>().x();
  return uint32_as_half2(y_u32);
}

/*!
 * \brief Wrapper of PTX ex2.approx.f16 instruction, which computes 2^x
 * \param x input
 */
__dpct_inline__ sycl::half ptx_exp2(sycl::half x) {
  ushort y_u16;
   y_u16 = sycl::pow(2, sycl::bit_cast<unsigned short, sycl::half>(x));
  return sycl::bit_cast<sycl::half, unsigned short>(y_u16);
}

/*!
 * \brief Wrapper of PTX rcp.approx instruction, which computes 1/x
 * \param x input
 */
__dpct_inline__ float ptx_rcp(float x) {
  float y;
   y = 1 / x;
  return y;
}

/*!
 * \brief Wrapper of PTX shfl.sync.bfly instruction, which performs a butterfly shuffle
 *   between threads in a warp.
 * \param x The value in the source lane
 * \param lane_mask The mask to perform thread index xor with: y[i] <- x[i ^ delta]
 */
__dpct_inline__ float shfl_xor_sync(float x, int lane_mask) {
  float y;
   y = dpct::experimental::permute_sub_group_by_xor(
       0xffffffff, sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_sub_group(), x, lane_mask);
  return y;
}

/*!
 * \brief Wrapper of PTX shfl.sync.bfly instruction on half2, which performs a butterfly
 *   shuffle between threads in a warp.
 * \param x The value in the source lane
 * \param lane_mask The mask to perform thread index xor with: y[i] <- x[i ^ lane_mask]
 */
__dpct_inline__ sycl::half2 shfl_xor_sync(sycl::half2 x, int lane_mask) {
  /*
  DPCT1108:24: '__shfl_xor_sync' was migrated with the experimental feature masked sub_group function which may not be
  supported by all compilers or runtimes. You may need to adjust the code.
  */
  return dpct::experimental::permute_sub_group_by_xor(
      0xffffffff, sycl::ext::oneapi::this_work_item::get_sub_group(), x, lane_mask);
}

/*!
 * \brief Wrapper of PTX rsqrt approximation instruction, which computes 1/sqrt(x)
 * \param x input
 */
__dpct_inline__ float rsqrt(float x) {
  float y;
   y = sycl::rsqrt(x);
  return y;
}

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
  ushort y_u16;
   y_u16 = sycl::tanh(sycl::bit_cast<unsigned short, sycl::half>(x));
  return sycl::bit_cast<sycl::half, unsigned short>(y_u16);
}

}  // namespace math
}  // namespace flashinfer
#endif  // FLASHINFER_MATH_CUH_
