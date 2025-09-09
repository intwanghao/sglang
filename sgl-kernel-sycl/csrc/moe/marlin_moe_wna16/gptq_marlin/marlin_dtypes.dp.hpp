
#ifndef _data_types_cuh
#define _data_types_cuh
#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <c10/xpu/XPUStream.h>

#include "marlin.dp.hpp"
#include <sycl/ext/intel/math.hpp>

#ifndef MARLIN_NAMESPACE_NAME
#define MARLIN_NAMESPACE_NAME marlin_moe_wna16
#endif

namespace MARLIN_NAMESPACE_NAME {

template <typename scalar_t>
class ScalarType {};

template <>
class ScalarType<sycl::half> {
 public:
  using scalar_t = sycl::half;
  using scalar_t2 = sycl::half2;

  // Matrix fragments for tensor core instructions; their precise layout is
  // documented here:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
  using FragA = Vec<sycl::half2, 4>;
  using FragB = Vec<sycl::half2, 2>;
  using FragC = Vec<float, 4>;
  using FragS = Vec<sycl::half2, 1>;
  using FragZP = Vec<sycl::half2, 4>;

  static float inline num2float(const sycl::half x) {
    return sycl::ext::intel::math::half2float(x);
  }

  static sycl::half2 inline num2num2(const sycl::half x) {
    return sycl::half2(x);
  }

  static sycl::half2 inline nums2num2(const sycl::half x1, const sycl::half x2) {
    return sycl::half2(x1, x2);
  }

  static sycl::half inline float2num(const float x) {
    return sycl::ext::intel::math::float2half_rn(x);
  }
};

template <>
class ScalarType<sycl::ext::oneapi::bfloat16> {
 public:
  using scalar_t = sycl::ext::oneapi::bfloat16;
  using scalar_t2 = sycl::vec<sycl::ext::oneapi::bfloat16, 2>;

  using FragA = Vec<sycl::vec<sycl::ext::oneapi::bfloat16, 2>, 4>;
  using FragB = Vec<sycl::vec<sycl::ext::oneapi::bfloat16, 2>, 2>;
  using FragC = Vec<float, 4>;
  using FragS = Vec<sycl::vec<sycl::ext::oneapi::bfloat16, 2>, 1>;
  using FragZP = Vec<sycl::vec<sycl::ext::oneapi::bfloat16, 2>, 4>;

#if !defined(DPCT_COMPATIBILITY_TEMP) || DPCT_COMPATIBILITY_TEMP >= 800
  static float inline num2float(const sycl::ext::oneapi::bfloat16 x) {
    return sycl::ext::intel::math::bfloat162float(x);
  }

  static sycl::vec<sycl::ext::oneapi::bfloat16, 2> inline num2num2(const sycl::ext::oneapi::bfloat16 x) {
    return sycl::vec<sycl::ext::oneapi::bfloat16, 2>(x, x);
  }

  static sycl::vec<sycl::ext::oneapi::bfloat16, 2> inline nums2num2(
      const sycl::ext::oneapi::bfloat16 x1, const sycl::ext::oneapi::bfloat16 x2) {
    return sycl::vec<sycl::ext::oneapi::bfloat16, 2>(x1, x2);
  }

  static sycl::ext::oneapi::bfloat16 inline float2num(const float x) {
    return sycl::ext::intel::math::float2bfloat16(x);
  }
#endif
};

}  // namespace MARLIN_NAMESPACE_NAME

#endif
