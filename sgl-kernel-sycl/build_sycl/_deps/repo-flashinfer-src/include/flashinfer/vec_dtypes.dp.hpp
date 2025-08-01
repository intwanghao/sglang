#define DPCT_COMPAT_RT_MAJOR_VERSION 12
#define DPCT_COMPAT_RT_MINOR_VERSION 4
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
#ifndef VEC_DTYPES_CUH_
#define VEC_DTYPES_CUH_

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <c10/xpu/XPUStream.h>

#include <type_traits>
#include <sycl/ext/intel/math.hpp>

namespace flashinfer {

#if (!defined(DPCT_COMPATIBILITY_TEMP) || (DPCT_COMPATIBILITY_TEMP >= 900))
#define FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED
#endif

#define FLASHINFER_INLINE inline __attribute__((always_inline))

#if (DPCT_COMPAT_RT_MAJOR_VERSION * 10000 + DPCT_COMPAT_RT_MINOR_VERSION * 100 < 120200) && \
    (defined(DPCT_COMPATIBILITY_TEMP) && (DPCT_COMPATIBILITY_TEMP < 800))
// CUDA version < 12.2 and GPU architecture < 80
FLASHINFER_INLINE __nv_bfloat162 make_bfloat162(const __nv_bfloat16 x, const __nv_bfloat16 y) {
  __nv_bfloat162 t;
  t.x = x;
  t.y = y;
  return t;
}

FLASHINFER_INLINE __nv_bfloat16 __hmul(const __nv_bfloat16 a, const __nv_bfloat16 b) {
  __nv_bfloat16 val;
  const float fa = __bfloat162float(a);
  const float fb = __bfloat162float(b);
  // avoid ftz in device code
  val = __float2bfloat16(__fmaf_ieee_rn(fa, fb, -0.0f));
  return val;
}

FLASHINFER_INLINE __nv_bfloat162 __hmul2(const __nv_bfloat162 a, const __nv_bfloat162 b) {
  __nv_bfloat162 val;
  val.x = __hmul(a.x, b.x);
  val.y = __hmul(a.y, b.y);
  return val;
}

FLASHINFER_INLINE __nv_bfloat162 __floats2bfloat162_rn(const float a, const float b) {
  __nv_bfloat162 val;
  val = __nv_bfloat162(__float2bfloat16_rn(a), __float2bfloat16_rn(b));
  return val;
}

FLASHINFER_INLINE __nv_bfloat162 __float22bfloat162_rn(const float2 a) {
  __nv_bfloat162 val = __floats2bfloat162_rn(a.x, a.y);
  return val;
}
FLASHINFER_INLINE float2 __bfloat1622float2(const __nv_bfloat162 a) {
  float hi_float;
  float lo_float;
  lo_float = __internal_bfloat162float(((__nv_bfloat162_raw)a).x);
  hi_float = __internal_bfloat162float(((__nv_bfloat162_raw)a).y);
  return make_float2(lo_float, hi_float);
}
#endif

/******************* vec_t type cast *******************/

template <typename dst_t, typename src_t>
struct vec_cast {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(dst_t* dst, const src_t* src) {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      dst[i] = (dst_t)src[i];
    }
  }
};

template <>
struct vec_cast<float, sycl::half> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(float* dst, const sycl::half* src) {
    if constexpr (vec_size == 1) {
      dst[0] = (float)src[0];
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size / 2; ++i) {
        ((sycl::float2*)dst)[i] = sycl::float2(
            sycl::ext::intel::math::half2float(((sycl::half2*)src)[i].x()),
            sycl::ext::intel::math::half2float(((sycl::half2*)src)[i].y()));
      }
    }
  }
};

template <>
struct vec_cast<sycl::half, float> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(sycl::half* dst, const float* src) {
    if constexpr (vec_size == 1) {
      dst[0] = sycl::ext::intel::math::float2half_rn(src[0]);
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size / 2; ++i) {
        ((sycl::half2*)dst)[i] = sycl::half2(
            sycl::ext::intel::math::float2half_rn(((sycl::float2*)src)[i].x()),
            sycl::ext::intel::math::float2half_rn(((sycl::float2*)src)[i].y()));
      }
    }
  }
};

template <typename T>
constexpr FLASHINFER_INLINE int get_exponent_bits() {
 if constexpr (std::is_same_v<T, sycl::half>) {
    return 5;
  } else if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
    return 8;
  }
}

template <typename T>
constexpr FLASHINFER_INLINE int get_mantissa_bits() {
 if constexpr (std::is_same_v<T, sycl::half>) {
    return 11;
  } else if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
    return 7;
  }
}

/*!
 * \brief Fallback to software fast dequant implementation if hardware dequantization is not
 * available.
 * \note Inspired by Marlin's fast dequantization, but here we don't have to permute
 * weights order.
 * \ref
 * https://github.com/vllm-project/vllm/blob/6dffa4b0a6120159ef2fe44d695a46817aff65bc/csrc/quantization/fp8/fp8_marlin.cu#L120
 */
template <typename fp8_dtype, typename fp16_dtype>
void fast_dequant_f8f16x4(uint32_t* input, sycl::uint2* output) {
  uint32_t q = *input;
  if constexpr (0) {
    output->x() = dpct::byte_level_permute(0U, q, 0x5140);
    output->y() = dpct::byte_level_permute(0U, q, 0x7362);
  } else {
    constexpr int FP8_EXPONENT = get_exponent_bits<fp8_dtype>();
    constexpr int FP8_MANTISSA = get_mantissa_bits<fp8_dtype>();
    constexpr int FP16_EXPONENT = get_exponent_bits<fp16_dtype>();

    constexpr int RIGHT_SHIFT = FP16_EXPONENT - FP8_EXPONENT;
    // Calculate MASK for extracting mantissa and exponent
    constexpr int MASK1 = 0x80000000;
    constexpr int MASK2 = MASK1 >> (FP8_EXPONENT + FP8_MANTISSA);
    constexpr int MASK3 = MASK2 & 0x7fffffff;
    constexpr int MASK = MASK3 | (MASK3 >> 16);
    q = dpct::byte_level_permute(q, q, 0x1302);

    // Extract and shift FP8 values to FP16 format
    uint32_t Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
    uint32_t Out2 = ((q << 8) & 0x80008000) | (((q << 8) & MASK) >> RIGHT_SHIFT);

    constexpr int BIAS_OFFSET = (1 << (FP16_EXPONENT - 1)) - (1 << (FP8_EXPONENT - 1));
    // Construct and apply exponent bias
    if constexpr (std::is_same_v<fp16_dtype, sycl::half>) {
      const sycl::half2 bias_reg = sycl::half2(sycl::ext::intel::math::float2half_rn(float(1 << BIAS_OFFSET)));

      // Convert to half2 and apply bias
      *(sycl::half2*)&(output->x()) =
          sycl::ext::intel::math::hmul2(*reinterpret_cast<const sycl::half2*>(&Out1), bias_reg);
      *(sycl::half2*)&(output->y()) =
          sycl::ext::intel::math::hmul2(*reinterpret_cast<const sycl::half2*>(&Out2), bias_reg);
    } else {
      constexpr uint32_t BIAS = (BIAS_OFFSET + 127) << 23;
      const sycl::vec<sycl::ext::oneapi::bfloat16, 2> bias_reg = sycl::vec<sycl::ext::oneapi::bfloat16, 2>(
          *reinterpret_cast<const float*>(&BIAS), *reinterpret_cast<const float*>(&BIAS));
      // Convert to bfloat162 and apply bias
      *(sycl::vec<sycl::ext::oneapi::bfloat16, 2>*)&(output->x()) =
          *reinterpret_cast<const sycl::vec<sycl::ext::oneapi::bfloat16, 2>*>(&Out1) * bias_reg;
      *(sycl::vec<sycl::ext::oneapi::bfloat16, 2>*)&(output->y()) =
          *reinterpret_cast<const sycl::vec<sycl::ext::oneapi::bfloat16, 2>*>(&Out2) * bias_reg;
    }
  }
}

/* template <>
struct vec_cast<sycl::ext::oneapi::bfloat16, __nv_fp8_e4m3> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(sycl::ext::oneapi::bfloat16* dst, const __nv_fp8_e4m3* src) {
    if constexpr (vec_size == 1) {
      dst[0] = sycl::ext::oneapi::bfloat16(src[0]);
    } else if constexpr (vec_size == 2) {
      dst[0] = sycl::ext::oneapi::bfloat16(src[0]);
      dst[1] = sycl::ext::oneapi::bfloat16(src[1]);
    } else {
      static_assert(vec_size % 4 == 0, "vec_size must be a multiple of 4");
#pragma unroll
      for (uint32_t i = 0; i < vec_size / 4; ++i) {
        fast_dequant_f8f16x4<__nv_fp8_e4m3, sycl::ext::oneapi::bfloat16>(
            (uint32_t*)&src[i * 4], (sycl::uint2*)&dst[i * 4]);
      }
    }
  }
}; */

/* template <>
struct vec_cast<sycl::ext::oneapi::bfloat16, __nv_fp8_e5m2> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(sycl::ext::oneapi::bfloat16* dst, const __nv_fp8_e5m2* src) {
    if constexpr (vec_size == 1) {
      dst[0] = sycl::ext::oneapi::bfloat16(src[0]);
    } else if constexpr (vec_size == 2) {
      dst[0] = sycl::ext::oneapi::bfloat16(src[0]);
      dst[1] = sycl::ext::oneapi::bfloat16(src[1]);
    } else {
      static_assert(vec_size % 4 == 0, "vec_size must be a multiple of 4");
#pragma unroll
      for (uint32_t i = 0; i < vec_size / 4; ++i) {
        fast_dequant_f8f16x4<__nv_fp8_e5m2, sycl::ext::oneapi::bfloat16>(
            (uint32_t*)&src[i * 4], (sycl::uint2*)&dst[i * 4]);
      }
    }
  }
};
 */
/* template <>
struct vec_cast<__nv_fp8_e4m3, sycl::half> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(__nv_fp8_e4m3* dst, const sycl::half* src) {
#ifdef FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED
    if constexpr (vec_size == 1) {
      dst[0] = __nv_fp8_e4m3(src[0]);
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size / 2; ++i) {
        uint16_t y;
        uint32_t x = *(uint32_t*)&src[i * 2];

        asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(y) : "r"(x));
        *(uint16_t*)&dst[i * 2] = y;
      }
    }
#else
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      dst[i] = __nv_fp8_e4m3(src[i]);
    }
#endif  // FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED
  }
}; */

/* template <>
struct vec_cast<__nv_fp8_e5m2, sycl::half> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(__nv_fp8_e5m2* dst, const sycl::half* src) {
#ifdef FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED
    if constexpr (vec_size == 1) {
      dst[0] = __nv_fp8_e5m2(src[0]);
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size / 2; ++i) {
        uint16_t y;
        uint32_t x = *(uint32_t*)&src[i * 2];
        asm volatile("cvt.rn.satfinite.e5m2x2.f16x2 %0, %1;" : "=h"(y) : "r"(x));
        *(uint16_t*)&dst[i * 2] = y;
      }
    }
#else
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      dst[i] = __nv_fp8_e5m2(src[i]);
    }
#endif  // FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED
  }
}; */

/* template <>
struct vec_cast<sycl::half, __nv_fp8_e4m3> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(sycl::half* dst, const __nv_fp8_e4m3* src) {
#ifdef FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED
    if constexpr (vec_size == 1) {
      dst[0] = sycl::half(src[0]);
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size / 2; ++i) {
        uint32_t y;
        uint16_t x = *(uint16_t*)&src[i * 2];

        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(y) : "h"(x));
        *(uint32_t*)&dst[i * 2] = y;
      }
    }
#else
    if constexpr (vec_size == 1) {
      dst[0] = half(src[0]);
    } else if constexpr (vec_size == 2) {
      dst[0] = half(src[0]);
      dst[1] = half(src[1]);
    } else {
      static_assert(vec_size % 4 == 0, "vec_size must be a multiple of 4");
#pragma unroll
      for (uint32_t i = 0; i < vec_size / 4; ++i) {
        fast_dequant_f8f16x4<__nv_fp8_e4m3, half>((uint32_t*)&src[i * 4], (uint2*)&dst[i * 4]);
      }
    }
#endif  // FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED
  }
}; */

/* template <>
struct vec_cast<sycl::half, __nv_fp8_e5m2> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(sycl::half* dst, const __nv_fp8_e5m2* src) {
#ifdef FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED
    if constexpr (vec_size == 1) {
      dst[0] = sycl::half(src[0]);
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size / 2; ++i) {
        uint32_t y;
        uint16_t x = *(uint16_t*)&src[i * 2];
        asm volatile("cvt.rn.f16x2.e5m2x2 %0, %1;" : "=r"(y) : "h"(x));
        *(uint32_t*)&dst[i * 2] = y;
      }
    }
#else
    if constexpr (vec_size == 1) {
      dst[0] = half(src[0]);
    } else if constexpr (vec_size == 2) {
      dst[0] = half(src[0]);
      dst[1] = half(src[1]);
    } else {
      static_assert(vec_size % 4 == 0, "vec_size must be a multiple of 4");
#pragma unroll
      for (uint32_t i = 0; i < vec_size / 4; ++i) {
        fast_dequant_f8f16x4<__nv_fp8_e5m2, half>((uint32_t*)&src[i * 4], (uint2*)&dst[i * 4]);
      }
    }
#endif  // FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED
  }
}; */

template <>
struct vec_cast<float, sycl::ext::oneapi::bfloat16> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(float* dst, const sycl::ext::oneapi::bfloat16* src) {
    if constexpr (vec_size == 1) {
      dst[0] = (float)src[0];
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size / 2; ++i) {
        ((sycl::float2*)dst)[i] = sycl::float2(
            sycl::ext::intel::math::bfloat162float(((sycl::vec<sycl::ext::oneapi::bfloat16, 2>*)src)[i].x()),
            sycl::ext::intel::math::bfloat162float(((sycl::vec<sycl::ext::oneapi::bfloat16, 2>*)src)[i].y()));
      }
    }
  }
};

template <>
struct vec_cast<sycl::ext::oneapi::bfloat16, float> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(sycl::ext::oneapi::bfloat16* dst, const float* src) {
    if constexpr (vec_size == 1) {
      dst[0] = sycl::ext::oneapi::bfloat16(src[0]);
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size / 2; ++i) {
        ((sycl::vec<sycl::ext::oneapi::bfloat16, 2>*)dst)[i] =
            sycl::vec<sycl::ext::oneapi::bfloat16, 2>(((sycl::float2*)src)[i].x(), ((sycl::float2*)src)[i].y());
      }
    }
  }
};

template <typename float_t, size_t vec_size>
struct vec_t {
  FLASHINFER_INLINE float_t& operator[](size_t i);
  FLASHINFER_INLINE const float_t& operator[](size_t i) const;
  FLASHINFER_INLINE void fill(float_t val);
  FLASHINFER_INLINE void load(const float_t* ptr);
  FLASHINFER_INLINE void store(float_t* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size>& src);
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr);
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const;
  FLASHINFER_INLINE static void memcpy(float_t* dst, const float_t* src);
  FLASHINFER_INLINE float_t* ptr();
};

template <typename src_float_t, typename tgt_float_t, size_t vec_size>
FLASHINFER_INLINE void cast_from_impl(vec_t<tgt_float_t, vec_size>& dst,
                                      const vec_t<src_float_t, vec_size>& src) {
  vec_cast<tgt_float_t, src_float_t>::template cast<vec_size>(
      dst.ptr(), const_cast<vec_t<src_float_t, vec_size>*>(&src)->ptr());
}

template <typename src_float_t, typename tgt_float_t, size_t vec_size>
FLASHINFER_INLINE void cast_load_impl(vec_t<tgt_float_t, vec_size>& dst,
                                      const src_float_t* src_ptr) {
  if constexpr (std::is_same_v<src_float_t, tgt_float_t>) {
    dst.load(src_ptr);
  } else {
    vec_t<src_float_t, vec_size> tmp;
    tmp.load(src_ptr);
    dst.cast_from(tmp);
  }
}

template <typename src_float_t, typename tgt_float_t, size_t vec_size>
FLASHINFER_INLINE void cast_store_impl(tgt_float_t* dst_ptr,
                                       const vec_t<src_float_t, vec_size>& src) {
  if constexpr (std::is_same_v<src_float_t, tgt_float_t>) {
    src.store(dst_ptr);
  } else {
    vec_t<tgt_float_t, vec_size> tmp;
    tmp.cast_from(src);
    tmp.store(dst_ptr);
  }
}

/*
template <>
struct vec_t<__nv_fp8_e4m3, 1> {
  __nv_fp8_e4m3 data;

  FLASHINFER_INLINE __nv_fp8_e4m3& operator[](size_t i) { return ((__nv_fp8_e4m3*)(&data))[i]; }
  FLASHINFER_INLINE const __nv_fp8_e4m3& operator[](size_t i) const {
    return ((const __nv_fp8_e4m3*)(&data))[i];
  }
  FLASHINFER_INLINE __nv_fp8_e4m3* ptr() { return reinterpret_cast<__nv_fp8_e4m3*>(&data); }
  FLASHINFER_INLINE void fill(__nv_fp8_e4m3 val);
  FLASHINFER_INLINE void load(const __nv_fp8_e4m3* ptr);
  FLASHINFER_INLINE void store(__nv_fp8_e4m3* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 1>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }

  FLASHINFER_INLINE static void memcpy(__nv_fp8_e4m3* dst, const __nv_fp8_e4m3* src);
};

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 1>::fill(__nv_fp8_e4m3 val) { data = val; }

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 1>::load(const __nv_fp8_e4m3* ptr) { data = *ptr; }

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 1>::store(__nv_fp8_e4m3* ptr) const { *ptr = data; }

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 1>::memcpy(__nv_fp8_e4m3* dst,
                                                       const __nv_fp8_e4m3* src) {
  *dst = *src;
}


template <>
struct vec_t<__nv_fp8_e4m3, 2> {
  __nv_fp8x2_e4m3 data;

  FLASHINFER_INLINE __nv_fp8_e4m3& operator[](size_t i) { return ((__nv_fp8_e4m3*)(&data))[i]; }
  FLASHINFER_INLINE const __nv_fp8_e4m3& operator[](size_t i) const {
    return ((const __nv_fp8_e4m3*)(&data))[i];
  }
  FLASHINFER_INLINE __nv_fp8_e4m3* ptr() { return reinterpret_cast<__nv_fp8_e4m3*>(&data); }
  FLASHINFER_INLINE void fill(__nv_fp8_e4m3 val);
  FLASHINFER_INLINE void load(const __nv_fp8_e4m3* ptr);
  FLASHINFER_INLINE void store(__nv_fp8_e4m3* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 2>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(__nv_fp8_e4m3* dst, const __nv_fp8_e4m3* src);
};

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 2>::fill(__nv_fp8_e4m3 val) {
  data.__x = (__nv_fp8x2_storage_t(val.__x) << 8) | __nv_fp8x2_storage_t(val.__x);
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 2>::load(const __nv_fp8_e4m3* ptr) {
  data = *((__nv_fp8x2_e4m3*)ptr);
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 2>::store(__nv_fp8_e4m3* ptr) const {
  *((__nv_fp8x2_e4m3*)ptr) = data;
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 2>::memcpy(__nv_fp8_e4m3* dst,
                                                       const __nv_fp8_e4m3* src) {
  *((__nv_fp8x2_e4m3*)dst) = *((__nv_fp8x2_e4m3*)src);
}


template <>
struct vec_t<__nv_fp8_e4m3, 4> {
  __nv_fp8x4_e4m3 data;

  FLASHINFER_INLINE __nv_fp8_e4m3& operator[](size_t i) { return ((__nv_fp8_e4m3*)(&data))[i]; }
  FLASHINFER_INLINE const __nv_fp8_e4m3& operator[](size_t i) const {
    return ((const __nv_fp8_e4m3*)(&data))[i];
  }
  FLASHINFER_INLINE __nv_fp8_e4m3* ptr() { return reinterpret_cast<__nv_fp8_e4m3*>(&data); }
  FLASHINFER_INLINE void fill(__nv_fp8_e4m3 val);
  FLASHINFER_INLINE void load(const __nv_fp8_e4m3* ptr);
  FLASHINFER_INLINE void store(__nv_fp8_e4m3* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 4>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }

  FLASHINFER_INLINE static void memcpy(__nv_fp8_e4m3* dst, const __nv_fp8_e4m3* src);
};

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 4>::fill(__nv_fp8_e4m3 val) {
  data.__x = (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
             (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 4>::load(const __nv_fp8_e4m3* ptr) {
  data = *((__nv_fp8x4_e4m3*)ptr);
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 4>::store(__nv_fp8_e4m3* ptr) const {
  *((__nv_fp8x4_e4m3*)ptr) = data;
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 4>::memcpy(__nv_fp8_e4m3* dst,
                                                       const __nv_fp8_e4m3* src) {
  *((__nv_fp8x4_e4m3*)dst) = *((__nv_fp8x4_e4m3*)src);
}
*/
// __nv_fp8_e4m3 x 8
/*
template <>
struct vec_t<__nv_fp8_e4m3, 8> {
  sycl::uint2 data;

  FLASHINFER_INLINE __nv_fp8_e4m3& operator[](size_t i) { return ((__nv_fp8_e4m3*)(&data))[i]; }
  FLASHINFER_INLINE const __nv_fp8_e4m3& operator[](size_t i) const {
    return ((const __nv_fp8_e4m3*)(&data))[i];
  }
  FLASHINFER_INLINE __nv_fp8_e4m3* ptr() { return reinterpret_cast<__nv_fp8_e4m3*>(&data); }
  FLASHINFER_INLINE void fill(__nv_fp8_e4m3 val);
  FLASHINFER_INLINE void load(const __nv_fp8_e4m3* ptr);
  FLASHINFER_INLINE void store(__nv_fp8_e4m3* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 8>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }

  FLASHINFER_INLINE static void memcpy(__nv_fp8_e4m3* dst, const __nv_fp8_e4m3* src);
};

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 8>::fill(__nv_fp8_e4m3 val) {
  ((__nv_fp8x4_e4m3*)(&data.x()))->__x = (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
                                         (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
  ((__nv_fp8x4_e4m3*)(&data.y()))->__x = (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
                                         (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 8>::load(const __nv_fp8_e4m3* ptr) {
  data = *((sycl::uint2*)ptr);
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 8>::store(__nv_fp8_e4m3* ptr) const {
  *((sycl::uint2*)ptr) = data;
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 8>::memcpy(__nv_fp8_e4m3* dst,
                                                       const __nv_fp8_e4m3* src) {
  *((sycl::uint2*)dst) = *((sycl::uint2*)src);
}
*/
// __nv_fp8_e4m3 x 16 or more
/*
template <size_t vec_size>
struct vec_t<__nv_fp8_e4m3, vec_size> {
  sycl::uint4 data[vec_size / 16];

  FLASHINFER_INLINE __nv_fp8_e4m3& operator[](size_t i) { return ((__nv_fp8_e4m3*)data)[i]; }
  FLASHINFER_INLINE const __nv_fp8_e4m3& operator[](size_t i) const {
    return ((const __nv_fp8_e4m3*)data)[i];
  }
  FLASHINFER_INLINE __nv_fp8_e4m3* ptr() { return reinterpret_cast<__nv_fp8_e4m3*>(&data); }
  FLASHINFER_INLINE void fill(__nv_fp8_e4m3 val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 16; ++i) {
      ((__nv_fp8x4_e4m3*)(&(data[i].x)))->__x =
          (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
          (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
      ((__nv_fp8x4_e4m3*)(&(data[i].y)))->__x =
          (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
          (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
      ((__nv_fp8x4_e4m3*)(&(data[i].z)))->__x =
          (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
          (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
      ((__nv_fp8x4_e4m3*)(&(data[i].w)))->__x =
          (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
          (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
    }
  }
  FLASHINFER_INLINE void load(const __nv_fp8_e4m3* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 16; ++i) {
      data[i] = ((sycl::uint4*)ptr)[i];
    }
  }
  FLASHINFER_INLINE void store(__nv_fp8_e4m3* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 16; ++i) {
      ((sycl::uint4*)ptr)[i] = data[i];
    }
  }
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }

  FLASHINFER_INLINE static void memcpy(__nv_fp8_e4m3* dst, const __nv_fp8_e4m3* src) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 16; ++i) {
      ((sycl::uint4*)dst)[i] = ((sycl::uint4*)src)[i];
    }
  }
};
*/
/******************* vec_t<__nv_fp8_e5m2> *******************/

// __nv_fp8_e5m2 x 1
/*
template <>
struct vec_t<__nv_fp8_e5m2, 1> {
  __nv_fp8_e5m2 data;

  FLASHINFER_INLINE __nv_fp8_e5m2& operator[](size_t i) { return ((__nv_fp8_e5m2*)(&data))[i]; }
  FLASHINFER_INLINE const __nv_fp8_e5m2& operator[](size_t i) const {
    return ((const __nv_fp8_e5m2*)(&data))[i];
  }
  FLASHINFER_INLINE __nv_fp8_e5m2* ptr() { return reinterpret_cast<__nv_fp8_e5m2*>(&data); }
  FLASHINFER_INLINE void fill(__nv_fp8_e5m2 val);
  FLASHINFER_INLINE void load(const __nv_fp8_e5m2* ptr);
  FLASHINFER_INLINE void store(__nv_fp8_e5m2* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 1>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }

  FLASHINFER_INLINE static void memcpy(__nv_fp8_e5m2* dst, const __nv_fp8_e5m2* src);
};

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 1>::fill(__nv_fp8_e5m2 val) { data = val; }

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 1>::load(const __nv_fp8_e5m2* ptr) { data = *ptr; }

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 1>::store(__nv_fp8_e5m2* ptr) const { *ptr = data; }

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 1>::memcpy(__nv_fp8_e5m2* dst,
                                                       const __nv_fp8_e5m2* src) {
  *dst = *src;
}
*/
// __nv_fp8_e5m2 x 2
/*
template <>
struct vec_t<__nv_fp8_e5m2, 2> {
  __nv_fp8x2_e5m2 data;

  FLASHINFER_INLINE __nv_fp8_e5m2& operator[](size_t i) { return ((__nv_fp8_e5m2*)(&data))[i]; }
  FLASHINFER_INLINE const __nv_fp8_e5m2& operator[](size_t i) const {
    return ((const __nv_fp8_e5m2*)(&data))[i];
  }
  FLASHINFER_INLINE __nv_fp8_e5m2* ptr() { return reinterpret_cast<__nv_fp8_e5m2*>(&data); }
  FLASHINFER_INLINE void fill(__nv_fp8_e5m2 val);
  FLASHINFER_INLINE void load(const __nv_fp8_e5m2* ptr);
  FLASHINFER_INLINE void store(__nv_fp8_e5m2* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 2>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }

  FLASHINFER_INLINE static void memcpy(__nv_fp8_e5m2* dst, const __nv_fp8_e5m2* src);
};

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 2>::fill(__nv_fp8_e5m2 val) {
  data.__x = (__nv_fp8x2_storage_t(val.__x) << 8) | __nv_fp8x2_storage_t(val.__x);
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 2>::load(const __nv_fp8_e5m2* ptr) {
  data = *((__nv_fp8x2_e5m2*)ptr);
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 2>::store(__nv_fp8_e5m2* ptr) const {
  *((__nv_fp8x2_e5m2*)ptr) = data;
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 2>::memcpy(__nv_fp8_e5m2* dst,
                                                       const __nv_fp8_e5m2* src) {
  *((__nv_fp8x2_e5m2*)dst) = *((__nv_fp8x2_e5m2*)src);
}
*/
// __nv_fp8_e5m2 x 4
/*
template <>
struct vec_t<__nv_fp8_e5m2, 4> {
  __nv_fp8x4_e5m2 data;

  FLASHINFER_INLINE __nv_fp8_e5m2& operator[](size_t i) { return ((__nv_fp8_e5m2*)(&data))[i]; }
  FLASHINFER_INLINE const __nv_fp8_e5m2& operator[](size_t i) const {
    return ((const __nv_fp8_e5m2*)(&data))[i];
  }
  FLASHINFER_INLINE __nv_fp8_e5m2* ptr() { return reinterpret_cast<__nv_fp8_e5m2*>(&data); }
  FLASHINFER_INLINE void fill(__nv_fp8_e5m2 val);
  FLASHINFER_INLINE void load(const __nv_fp8_e5m2* ptr);
  FLASHINFER_INLINE void store(__nv_fp8_e5m2* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 4>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }

  FLASHINFER_INLINE static void memcpy(__nv_fp8_e5m2* dst, const __nv_fp8_e5m2* src);
};

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 4>::fill(__nv_fp8_e5m2 val) {
  data.__x = (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
             (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 4>::load(const __nv_fp8_e5m2* ptr) {
  data = *((__nv_fp8x4_e5m2*)ptr);
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 4>::store(__nv_fp8_e5m2* ptr) const {
  *((__nv_fp8x4_e5m2*)ptr) = data;
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 4>::memcpy(__nv_fp8_e5m2* dst,
                                                       const __nv_fp8_e5m2* src) {
  *((__nv_fp8x4_e5m2*)dst) = *((__nv_fp8x4_e5m2*)src);
}
*/
// __nv_fp8_e5m2 x 8
/*
template <>
struct vec_t<__nv_fp8_e5m2, 8> {
  sycl::uint2 data;

  FLASHINFER_INLINE __nv_fp8_e5m2& operator[](size_t i) { return ((__nv_fp8_e5m2*)(&data))[i]; }
  FLASHINFER_INLINE const __nv_fp8_e5m2& operator[](size_t i) const {
    return ((const __nv_fp8_e5m2*)(&data))[i];
  }
  FLASHINFER_INLINE __nv_fp8_e5m2* ptr() { return reinterpret_cast<__nv_fp8_e5m2*>(&data); }
  FLASHINFER_INLINE void fill(__nv_fp8_e5m2 val);
  FLASHINFER_INLINE void load(const __nv_fp8_e5m2* ptr);
  FLASHINFER_INLINE void store(__nv_fp8_e5m2* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 8>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(__nv_fp8_e5m2* dst, const __nv_fp8_e5m2* src);
};

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 8>::fill(__nv_fp8_e5m2 val) {
  ((__nv_fp8x4_e5m2*)(&data.x()))->__x = (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
                                         (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
  ((__nv_fp8x4_e5m2*)(&data.y()))->__x = (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
                                         (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 8>::load(const __nv_fp8_e5m2* ptr) {
  data = *((sycl::uint2*)ptr);
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 8>::store(__nv_fp8_e5m2* ptr) const {
  *((sycl::uint2*)ptr) = data;
}

FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 8>::memcpy(__nv_fp8_e5m2* dst,
                                                       const __nv_fp8_e5m2* src) {
  *((sycl::uint2*)dst) = *((sycl::uint2*)src);
}
*/
// __nv_fp8_e5m2 x 16 or more
/*
template <size_t vec_size>
struct vec_t<__nv_fp8_e5m2, vec_size> {
  sycl::uint4 data[vec_size / 16];

  FLASHINFER_INLINE __nv_fp8_e5m2& operator[](size_t i) { return ((__nv_fp8_e5m2*)data)[i]; }
  FLASHINFER_INLINE const __nv_fp8_e5m2& operator[](size_t i) const {
    return ((const __nv_fp8_e5m2*)data)[i];
  }
  FLASHINFER_INLINE __nv_fp8_e5m2* ptr() { return reinterpret_cast<__nv_fp8_e5m2*>(&data); }
  FLASHINFER_INLINE void fill(__nv_fp8_e5m2 val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 16; ++i) {
      ((__nv_fp8x4_e5m2*)(&(data[i].x)))->__x =
          (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
          (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
      ((__nv_fp8x4_e5m2*)(&(data[i].y)))->__x =
          (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
          (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
      ((__nv_fp8x4_e5m2*)(&(data[i].z)))->__x =
          (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
          (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
      ((__nv_fp8x4_e5m2*)(&(data[i].w)))->__x =
          (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
          (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
    }
  }
  FLASHINFER_INLINE void load(const __nv_fp8_e5m2* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 16; ++i) {
      data[i] = ((sycl::uint4*)ptr)[i];
    }
  }
  FLASHINFER_INLINE void store(__nv_fp8_e5m2* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 16; ++i) {
      ((sycl::uint4*)ptr)[i] = data[i];
    }
  }
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(__nv_fp8_e5m2* dst, const __nv_fp8_e5m2* src) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 16; ++i) {
      ((sycl::uint4*)dst)[i] = ((sycl::uint4*)src)[i];
    }
  }
};
*/
/******************* vec_t<half> *******************/

// half x 1
template <>
struct vec_t<sycl::half, 1> {
  sycl::half data;

  FLASHINFER_INLINE sycl::half& operator[](size_t i) { return ((sycl::half*)(&data))[i]; }
  FLASHINFER_INLINE const sycl::half& operator[](size_t i) const { return ((const sycl::half*)(&data))[i]; }
  FLASHINFER_INLINE sycl::half* ptr() { return reinterpret_cast<sycl::half*>(&data); }
  FLASHINFER_INLINE void fill(sycl::half val);
  FLASHINFER_INLINE void load(const sycl::half* ptr);
  FLASHINFER_INLINE void store(sycl::half* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 1>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }

  FLASHINFER_INLINE static void memcpy(sycl::half* dst, const sycl::half* src);
};

FLASHINFER_INLINE void vec_t<sycl::half, 1>::fill(sycl::half val) { data = val; }

FLASHINFER_INLINE void vec_t<sycl::half, 1>::load(const sycl::half* ptr) { data = *ptr; }

FLASHINFER_INLINE void vec_t<sycl::half, 1>::store(sycl::half* ptr) const {* ptr = data; }

FLASHINFER_INLINE void vec_t<sycl::half, 1>::memcpy(sycl::half* dst, const sycl::half* src) {* dst = *src; }

// half x 2
template <>
struct vec_t<sycl::half, 2> {
  sycl::half2 data;

  FLASHINFER_INLINE sycl::half& operator[](size_t i) { return ((sycl::half*)(&data))[i]; }
  FLASHINFER_INLINE const sycl::half& operator[](size_t i) const { return ((const sycl::half*)(&data))[i]; }
  FLASHINFER_INLINE sycl::half* ptr() { return reinterpret_cast<sycl::half*>(&data); }
  FLASHINFER_INLINE void fill(sycl::half val);
  FLASHINFER_INLINE void load(const sycl::half* ptr);
  FLASHINFER_INLINE void store(sycl::half* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 2>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }

  FLASHINFER_INLINE static void memcpy(sycl::half* dst, const sycl::half* src);
};

FLASHINFER_INLINE void vec_t<sycl::half, 2>::fill(sycl::half val) { data = sycl::half2(val, val); }

FLASHINFER_INLINE void vec_t<sycl::half, 2>::load(const sycl::half* ptr) { data = *((sycl::half2*)ptr); }

FLASHINFER_INLINE void vec_t<sycl::half, 2>::store(sycl::half* ptr) const { *((sycl::half2*)ptr) = data; }

FLASHINFER_INLINE void vec_t<sycl::half, 2>::memcpy(sycl::half* dst, const sycl::half* src) {
  *((sycl::half2*)dst) = *((sycl::half2*)src);
}

// half x 4

template <>
struct vec_t<sycl::half, 4> {
  sycl::uint2 data;

  FLASHINFER_INLINE sycl::half& operator[](size_t i) { return ((sycl::half*)(&data))[i]; }
  FLASHINFER_INLINE const sycl::half& operator[](size_t i) const { return ((const sycl::half*)(&data))[i]; }
  FLASHINFER_INLINE sycl::half* ptr() { return reinterpret_cast<sycl::half*>(&data); }
  FLASHINFER_INLINE void fill(sycl::half val);
  FLASHINFER_INLINE void load(const sycl::half* ptr);
  FLASHINFER_INLINE void store(sycl::half* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 4>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(sycl::half* dst, const sycl::half* src);
};

FLASHINFER_INLINE void vec_t<sycl::half, 4>::fill(sycl::half val) {
  *(sycl::half2*)(&data.x()) = sycl::half2(val, val);
  *(sycl::half2*)(&data.y()) = sycl::half2(val, val);
}

FLASHINFER_INLINE void vec_t<sycl::half, 4>::load(const sycl::half* ptr) { data = *((sycl::uint2*)ptr); }

FLASHINFER_INLINE void vec_t<sycl::half, 4>::store(sycl::half* ptr) const { *((sycl::uint2*)ptr) = data; }

FLASHINFER_INLINE void vec_t<sycl::half, 4>::memcpy(sycl::half* dst, const sycl::half* src) {
  *((sycl::uint2*)dst) = *((sycl::uint2*)src);
}

// half x 8 or more

template <size_t vec_size>
struct vec_t<sycl::half, vec_size> {
  sycl::uint4 data[vec_size / 8];
  FLASHINFER_INLINE sycl::half& operator[](size_t i) { return ((sycl::half*)data)[i]; }
  FLASHINFER_INLINE const sycl::half& operator[](size_t i) const { return ((const sycl::half*)data)[i]; }
  FLASHINFER_INLINE sycl::half* ptr() { return reinterpret_cast<sycl::half*>(&data); }
  FLASHINFER_INLINE void fill(sycl::half val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      *(sycl::half2*)(&(data[i].x)) = sycl::half2(val, val);
      *(sycl::half2*)(&(data[i].y)) = sycl::half2(val, val);
      *(sycl::half2*)(&(data[i].z)) = sycl::half2(val, val);
      *(sycl::half2*)(&(data[i].w)) = sycl::half2(val, val);
    }
  }
  FLASHINFER_INLINE void load(const sycl::half* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      data[i] = ((sycl::uint4*)ptr)[i];
    }
  }
  FLASHINFER_INLINE void store(sycl::half* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((sycl::uint4*)ptr)[i] = data[i];
    }
  }
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(sycl::half* dst, const sycl::half* src) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((sycl::uint4*)dst)[i] = ((sycl::uint4*)src)[i];
    }
  }
};

/******************* vec_t<nv_bfloat16> *******************/

// nv_bfloat16 x 1
template <>
struct vec_t<sycl::ext::oneapi::bfloat16, 1> {
  sycl::ext::oneapi::bfloat16 data;
  FLASHINFER_INLINE sycl::ext::oneapi::bfloat16& operator[](size_t i) {
    return ((sycl::ext::oneapi::bfloat16*)(&data))[i];
  }
  FLASHINFER_INLINE const sycl::ext::oneapi::bfloat16& operator[](size_t i) const {
    return ((const sycl::ext::oneapi::bfloat16*)(&data))[i];
  }
  FLASHINFER_INLINE sycl::ext::oneapi::bfloat16* ptr() { return reinterpret_cast<sycl::ext::oneapi::bfloat16*>(&data); }
  FLASHINFER_INLINE void fill(sycl::ext::oneapi::bfloat16 val);
  FLASHINFER_INLINE void load(const sycl::ext::oneapi::bfloat16* ptr);
  FLASHINFER_INLINE void store(sycl::ext::oneapi::bfloat16* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 1>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(sycl::ext::oneapi::bfloat16* dst, const sycl::ext::oneapi::bfloat16* src);
};

FLASHINFER_INLINE void vec_t<sycl::ext::oneapi::bfloat16, 1>::fill(sycl::ext::oneapi::bfloat16 val) { data = val; }

FLASHINFER_INLINE void vec_t<sycl::ext::oneapi::bfloat16, 1>::load(const sycl::ext::oneapi::bfloat16* ptr) {
  data = *ptr;
}

FLASHINFER_INLINE void vec_t<sycl::ext::oneapi::bfloat16, 1>::store(sycl::ext::oneapi::bfloat16* ptr) const { *ptr = data; }

FLASHINFER_INLINE void vec_t<sycl::ext::oneapi::bfloat16, 1>::memcpy(
    sycl::ext::oneapi::bfloat16* dst, const sycl::ext::oneapi::bfloat16* src) {
  *dst = *src;
}

// nv_bfloat16 x 2
template <>
struct vec_t<sycl::ext::oneapi::bfloat16, 2> {
  sycl::vec<sycl::ext::oneapi::bfloat16, 2> data;

  FLASHINFER_INLINE sycl::ext::oneapi::bfloat16& operator[](size_t i) {
    return ((sycl::ext::oneapi::bfloat16*)(&data))[i];
  }
  FLASHINFER_INLINE const sycl::ext::oneapi::bfloat16& operator[](size_t i) const {
    return ((const sycl::ext::oneapi::bfloat16*)(&data))[i];
  }
  FLASHINFER_INLINE sycl::ext::oneapi::bfloat16* ptr() { return reinterpret_cast<sycl::ext::oneapi::bfloat16*>(&data); }
  FLASHINFER_INLINE void fill(sycl::ext::oneapi::bfloat16 val);
  FLASHINFER_INLINE void load(const sycl::ext::oneapi::bfloat16* ptr);
  FLASHINFER_INLINE void store(sycl::ext::oneapi::bfloat16* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 2>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(sycl::ext::oneapi::bfloat16* dst, const sycl::ext::oneapi::bfloat16* src);
};

FLASHINFER_INLINE void vec_t<sycl::ext::oneapi::bfloat16, 2>::fill(sycl::ext::oneapi::bfloat16 val) {
  data = sycl::vec<sycl::ext::oneapi::bfloat16, 2>(val, val);
}

FLASHINFER_INLINE void vec_t<sycl::ext::oneapi::bfloat16, 2>::load(const sycl::ext::oneapi::bfloat16* ptr) {
  data = *((sycl::vec<sycl::ext::oneapi::bfloat16, 2>*)ptr);
}

FLASHINFER_INLINE void vec_t<sycl::ext::oneapi::bfloat16, 2>::store(sycl::ext::oneapi::bfloat16* ptr) const {
  *((sycl::vec<sycl::ext::oneapi::bfloat16, 2>*)ptr) = data;
}

FLASHINFER_INLINE void vec_t<sycl::ext::oneapi::bfloat16, 2>::memcpy(
    sycl::ext::oneapi::bfloat16* dst, const sycl::ext::oneapi::bfloat16* src) {
  *((sycl::vec<sycl::ext::oneapi::bfloat16, 2>*)dst) = *((sycl::vec<sycl::ext::oneapi::bfloat16, 2>*)src);
}

// nv_bfloat16 x 4

template <>
struct vec_t<sycl::ext::oneapi::bfloat16, 4> {
  sycl::uint2 data;

  FLASHINFER_INLINE sycl::ext::oneapi::bfloat16& operator[](size_t i) {
    return ((sycl::ext::oneapi::bfloat16*)(&data))[i];
  }
  FLASHINFER_INLINE const sycl::ext::oneapi::bfloat16& operator[](size_t i) const {
    return ((const sycl::ext::oneapi::bfloat16*)(&data))[i];
  }
  FLASHINFER_INLINE sycl::ext::oneapi::bfloat16* ptr() { return reinterpret_cast<sycl::ext::oneapi::bfloat16*>(&data); }
  FLASHINFER_INLINE void fill(sycl::ext::oneapi::bfloat16 val);
  FLASHINFER_INLINE void load(const sycl::ext::oneapi::bfloat16* ptr);
  FLASHINFER_INLINE void store(sycl::ext::oneapi::bfloat16* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 4>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(sycl::ext::oneapi::bfloat16* dst, const sycl::ext::oneapi::bfloat16* src);
};

FLASHINFER_INLINE void vec_t<sycl::ext::oneapi::bfloat16, 4>::fill(sycl::ext::oneapi::bfloat16 val) {
  *(sycl::vec<sycl::ext::oneapi::bfloat16, 2>*)(&data.x()) = sycl::vec<sycl::ext::oneapi::bfloat16, 2>(val, val);
  *(sycl::vec<sycl::ext::oneapi::bfloat16, 2>*)(&data.y()) = sycl::vec<sycl::ext::oneapi::bfloat16, 2>(val, val);
}

FLASHINFER_INLINE void vec_t<sycl::ext::oneapi::bfloat16, 4>::load(const sycl::ext::oneapi::bfloat16* ptr) {
  data = *((sycl::uint2*)ptr);
}

FLASHINFER_INLINE void vec_t<sycl::ext::oneapi::bfloat16, 4>::store(sycl::ext::oneapi::bfloat16* ptr) const {
  *((sycl::uint2*)ptr) = data;
}

FLASHINFER_INLINE void vec_t<sycl::ext::oneapi::bfloat16, 4>::memcpy(
    sycl::ext::oneapi::bfloat16* dst, const sycl::ext::oneapi::bfloat16* src) {
  *((sycl::uint2*)dst) = *((sycl::uint2*)src);
}

// nv_bfloat16 x 8 or more

template <size_t vec_size>
struct vec_t<sycl::ext::oneapi::bfloat16, vec_size> {
  sycl::uint4 data[vec_size / 8];

  FLASHINFER_INLINE sycl::ext::oneapi::bfloat16& operator[](size_t i) {
    return ((sycl::ext::oneapi::bfloat16*)data)[i];
  }
  FLASHINFER_INLINE const sycl::ext::oneapi::bfloat16& operator[](size_t i) const {
    return ((const sycl::ext::oneapi::bfloat16*)data)[i];
  }
  FLASHINFER_INLINE sycl::ext::oneapi::bfloat16* ptr() { return reinterpret_cast<sycl::ext::oneapi::bfloat16*>(&data); }
  FLASHINFER_INLINE void fill(sycl::ext::oneapi::bfloat16 val) {
#pragma unoll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      *(sycl::vec<sycl::ext::oneapi::bfloat16, 2>*)(&(data[i].x)) = sycl::vec<sycl::ext::oneapi::bfloat16, 2>(val, val);
      *(sycl::vec<sycl::ext::oneapi::bfloat16, 2>*)(&(data[i].y)) = sycl::vec<sycl::ext::oneapi::bfloat16, 2>(val, val);
      *(sycl::vec<sycl::ext::oneapi::bfloat16, 2>*)(&(data[i].z)) = sycl::vec<sycl::ext::oneapi::bfloat16, 2>(val, val);
      *(sycl::vec<sycl::ext::oneapi::bfloat16, 2>*)(&(data[i].w)) = sycl::vec<sycl::ext::oneapi::bfloat16, 2>(val, val);
    }
  }
  FLASHINFER_INLINE void load(const sycl::ext::oneapi::bfloat16* ptr) {
#pragma unoll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      data[i] = ((sycl::uint4*)ptr)[i];
    }
  }
  FLASHINFER_INLINE void store(sycl::ext::oneapi::bfloat16* ptr) const {
#pragma unoll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((sycl::uint4*)ptr)[i] = data[i];
    }
  }
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(sycl::ext::oneapi::bfloat16* dst, const sycl::ext::oneapi::bfloat16* src) {
#pragma unoll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((sycl::uint4*)dst)[i] = ((sycl::uint4*)src)[i];
    }
  }
};

/******************* vec_t<float> *******************/

// float x 1

template <>
struct vec_t<float, 1> {
  float data;

  FLASHINFER_INLINE float& operator[](size_t i) { return ((float*)(&data))[i]; }
  FLASHINFER_INLINE const float& operator[](size_t i) const { return ((const float*)(&data))[i]; }
  FLASHINFER_INLINE float* ptr() { return reinterpret_cast<float*>(&data); }
  FLASHINFER_INLINE void fill(float val);
  FLASHINFER_INLINE void load(const float* ptr);
  FLASHINFER_INLINE void store(float* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 1>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(float* dst, const float* src);
};

FLASHINFER_INLINE void vec_t<float, 1>::fill(float val) { data = val; }

FLASHINFER_INLINE void vec_t<float, 1>::load(const float* ptr) { data = *ptr; }

FLASHINFER_INLINE void vec_t<float, 1>::store(float* ptr) const { *ptr = data; }

FLASHINFER_INLINE void vec_t<float, 1>::memcpy(float* dst, const float* src) { *dst = *src; }

// float x 2

template <>
struct vec_t<float, 2> {
  sycl::float2 data;

  FLASHINFER_INLINE float& operator[](size_t i) { return ((float*)(&data))[i]; }
  FLASHINFER_INLINE const float& operator[](size_t i) const { return ((const float*)(&data))[i]; }
  FLASHINFER_INLINE float* ptr() { return reinterpret_cast<float*>(&data); }
  FLASHINFER_INLINE void fill(float val);
  FLASHINFER_INLINE void load(const float* ptr);
  FLASHINFER_INLINE void store(float* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 2>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(float* dst, const float* src);
};

FLASHINFER_INLINE void vec_t<float, 2>::fill(float val) { data = sycl::float2(val, val); }

FLASHINFER_INLINE void vec_t<float, 2>::load(const float* ptr) { data = *((sycl::float2*)ptr); }

FLASHINFER_INLINE void vec_t<float, 2>::store(float* ptr) const { *((sycl::float2*)ptr) = data; }

FLASHINFER_INLINE void vec_t<float, 2>::memcpy(float* dst, const float* src) {
  *((sycl::float2*)dst) = *((sycl::float2*)src);
}

// float x 4 or more
template <size_t vec_size>
struct vec_t<float, vec_size> {
  sycl::float4 data[vec_size / 4];

  FLASHINFER_INLINE float& operator[](size_t i) { return ((float*)(data))[i]; }
  FLASHINFER_INLINE const float& operator[](size_t i) const { return ((const float*)(data))[i]; }
  FLASHINFER_INLINE float* ptr() { return reinterpret_cast<float*>(&data); }
  FLASHINFER_INLINE void fill(float val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = sycl::float4(val, val, val, val);
    }
  }
  FLASHINFER_INLINE void load(const float* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = ((sycl::float4*)ptr)[i];
    }
  }
  FLASHINFER_INLINE void store(float* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      ((sycl::float4*)ptr)[i] = data[i];
    }
  }
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(float* dst, const float* src) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      ((sycl::float4*)dst)[i] = ((sycl::float4*)src)[i];
    }
  }
};

}  // namespace flashinfer

#endif  // VEC_DTYPES_CUH_
