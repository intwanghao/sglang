#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <c10/xpu/XPUStream.h>
#include <ATen/xpu/XPUContext.h>

#include <c10/core/DeviceGuard.h>

#include <torch/all.h>

#include <iostream>

#ifndef MARLIN_NAMESPACE_NAME
#define MARLIN_NAMESPACE_NAME marlin_moe_wna16
#endif

namespace MARLIN_NAMESPACE_NAME {

// Marlin params

// 8 warps are a good choice since every SM has 4 schedulers and having more
// than 1 warp per schedule allows some more latency hiding. At the same time,
// we want relatively few warps to have many registers per warp and small tiles.
static constexpr int default_threads = 256;

static constexpr int pipe_stages = 4;  // 4 pipeline stages fit into shared memory

static constexpr int min_thread_n = 64;
static constexpr int min_thread_k = 64;
static constexpr int max_thread_n = 256;

static constexpr int tile_size = 16;
static constexpr int max_par = 16;

// Repack params
static constexpr int repack_stages = 8;

static constexpr int repack_threads = 256;

static constexpr int tile_k_size = tile_size;
static constexpr int tile_n_size = tile_k_size * 4;

// Helpers
template <typename T, int n>
struct Vec {
  T elems[n];
  T& operator[](int i) {
    return elems[i];
  }
};

using I4 = Vec<int, 4>;

constexpr int div_ceil(int a, int b) {
  return (a + b - 1) / b;
}

#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 800
// No support for async
#else

inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr, bool pred = true) {
  const int BYTES = 16;
  auto smem = smem_ptr;
  /*
  DPCT1137:0: ASM instruction "cp.async" is asynchronous copy, currently it is migrated to synchronous copy operation.
  You may need to adjust the code to tune the performance.
  */
  {
    bool p;
    p = (int)pred != 0;
    if (p) {
      *(((uint32_t*)(uintptr_t)smem)) = *(((uint32_t*)(uintptr_t)glob_ptr));
      if (BYTES > 4)
        *(((uint32_t*)(uintptr_t)smem) + 1) = *(((uint32_t*)(uintptr_t)glob_ptr) + 1);
      if (BYTES > 8)
        *(((uint32_t*)(uintptr_t)smem) + 2) = *(((uint32_t*)(uintptr_t)glob_ptr) + 2);
      if (BYTES > 12)
        *(((uint32_t*)(uintptr_t)smem) + 3) = *(((uint32_t*)(uintptr_t)glob_ptr) + 3);
    }
  }
}

inline void cp_async4(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  auto smem = smem_ptr;
  /*
  DPCT1137:1: ASM instruction "cp.async" is asynchronous copy, currently it is migrated to synchronous copy operation.
  You may need to adjust the code to tune the performance.
  */
  {
    *(((uint32_t*)(uintptr_t)smem)) = *(((uint32_t*)(uintptr_t)glob_ptr));
    if (BYTES > 4)
      *(((uint32_t*)(uintptr_t)smem) + 1) = *(((uint32_t*)(uintptr_t)glob_ptr) + 1);
    if (BYTES > 8)
      *(((uint32_t*)(uintptr_t)smem) + 2) = *(((uint32_t*)(uintptr_t)glob_ptr) + 2);
    if (BYTES > 12)
      *(((uint32_t*)(uintptr_t)smem) + 3) = *(((uint32_t*)(uintptr_t)glob_ptr) + 3);
  }
}

inline void cp_async_fence() {
  /*
  DPCT1026:2: The call to "cp.async.commit_group;
" was removed because current "cp.async" is migrated to synchronous copy operation. You may need to adjust the code to
tune the performance.
  */
}

template <int n>
inline void cp_async_wait() {
  /*
  DPCT1026:3: The call to "cp.async.wait_group %0;
" was removed because current "cp.async" is migrated to synchronous copy operation. You may need to adjust the code to
tune the performance.
  */
}
#endif

}  // namespace MARLIN_NAMESPACE_NAME
