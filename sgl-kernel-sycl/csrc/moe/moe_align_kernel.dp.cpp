/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <c10/xpu/XPUStream.h>
#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>

#include <c10/core/DeviceGuard.h>

#include "utils.h"
#include <c10/xpu/XPUStream.h>

#include <cmath>

#define WARP_SIZE 32

#define VEC_SIZE 4
using Vec = sycl::int4;

template <typename scalar_t>
void count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum_buffer,
    size_t numel) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const size_t tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
  const size_t stride = item_ct1.get_local_range(2) * item_ct1.get_group_range(2);

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    int32_t rank_post_pad =
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&cumsum_buffer[expert_id], 1);
    sorted_token_ids[rank_post_pad] = i;
  }
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
template <typename scalar_t>
void count_and_sort_expert_tokens_kernel_wrapper(
    const scalar_t* __restrict topk_ids,
    int32_t* __restrict sorted_token_ids,
    int32_t* __restrict cumsum_buffer,
    size_t numel) {
  sycl::queue queue = *dpct::kernel_launcher::_que;
  unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
  sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

  auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};

  queue.submit([&](sycl::handler& cgh) {
    auto last_event = dpct::get_default_queue().ext_oneapi_get_last_event();
    [&](auto&& _e) {
      if constexpr (std::is_same_v<std::remove_reference_t<decltype(last_event)>, sycl::event>)
        cgh.depends_on(_e);
      else if (_e.has_value())
        cgh.depends_on(_e.value());
    }(last_event);

    cgh.parallel_for(nr, exp_props, [=](sycl::nd_item<3> item_ct1) {
      count_and_sort_expert_tokens_kernel<scalar_t>(topk_ids, sorted_token_ids, cumsum_buffer, numel);
    });
  });
}

template <typename scalar_t>
/*
DPCT1110:615: The total declared local variable size in device function moe_align_block_size_kernel exceeds 128 bytes
and may cause high register pressure. Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void moe_align_block_size_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    size_t numel,
    int32_t* __restrict__ cumsum,
    bool pad_sorted_token_ids,
    const int32_t scan_size,
    uint8_t* dpct_local) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto smem = (int32_t*)dpct_local;
  int32_t* shared_counts = smem;                  // [num_experts]
  int32_t* prefix = shared_counts + num_experts;  // [num_experts + 1]
  int32_t* scan_buf = prefix + num_experts + 1;   // [scan_size]
  auto& s_total_tokens_post_pad = *sycl::ext::oneapi::group_local_memory_for_overwrite<int32_t>(
      sycl::ext::oneapi::this_work_item::get_work_group<3>());

  const size_t tid = item_ct1.get_local_id(2);
  const size_t stride = item_ct1.get_local_range(2);

  if (tid < num_experts) {
    shared_counts[tid] = 0;
  }

  /*
  DPCT1113:778: Consider replacing sycl::nd_item::barrier(sycl::access::fence_space::local_space) with
  sycl::nd_item::barrier() if function "moe_align_block_size_kernel" is called in a multidimensional kernel.
  */
  item_ct1.barrier(sycl::access::fence_space::local_space);

  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&shared_counts[expert_id], 1);
  }

  /*
  DPCT1113:779: Consider replacing sycl::nd_item::barrier(sycl::access::fence_space::local_space) with
  sycl::nd_item::barrier() if function "moe_align_block_size_kernel" is called in a multidimensional kernel.
  */
  item_ct1.barrier(sycl::access::fence_space::local_space);

  int32_t padded_count = 0;
  if (tid < num_experts) {
    int32_t count = shared_counts[tid];
    padded_count = (count + block_size - 1) / block_size * block_size;
    scan_buf[tid] = padded_count;
  }

  if (tid >= num_experts && tid < scan_size) {
    scan_buf[tid] = 0;
  }

  /*
  DPCT1113:780: Consider replacing sycl::nd_item::barrier(sycl::access::fence_space::local_space) with
  sycl::nd_item::barrier() if function "moe_align_block_size_kernel" is called in a multidimensional kernel.
  */
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // Blelloch scan
  int offset = 1;
#pragma unroll
  for (int d = scan_size >> 1; d > 0; d >>= 1) {
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      scan_buf[bi] += scan_buf[ai];
    }
    offset <<= 1;
    /*
    DPCT1118:616: SYCL group functions and algorithms must be encountered in converged control flow. You may need to
    adjust the code.
    */
    /*
    DPCT1113:783: Consider replacing sycl::nd_item::barrier(sycl::access::fence_space::local_space) with
    sycl::nd_item::barrier() if function "moe_align_block_size_kernel" is called in a multidimensional kernel.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
  }

  // down-sweep
  if (tid == 0) {
    prefix[num_experts] = scan_buf[scan_size - 1];
    scan_buf[scan_size - 1] = 0;
  }
  /*
  DPCT1113:781: Consider replacing sycl::nd_item::barrier(sycl::access::fence_space::local_space) with
  sycl::nd_item::barrier() if function "moe_align_block_size_kernel" is called in a multidimensional kernel.
  */
  item_ct1.barrier(sycl::access::fence_space::local_space);

#pragma unroll
  for (int d = 1; d < scan_size; d <<= 1) {
    offset >>= 1;
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      if (bi < scan_size) {
        int temp = scan_buf[ai];
        scan_buf[ai] = scan_buf[bi];
        scan_buf[bi] += temp;
      }
    }
    /*
    DPCT1118:617: SYCL group functions and algorithms must be encountered in converged control flow. You may need to
    adjust the code.
    */
    /*
    DPCT1113:784: Consider replacing sycl::nd_item::barrier(sycl::access::fence_space::local_space) with
    sycl::nd_item::barrier() if function "moe_align_block_size_kernel" is called in a multidimensional kernel.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
  }

  if (tid < num_experts) {
    prefix[tid] = scan_buf[tid];
  }

  if (tid == 0) {
    s_total_tokens_post_pad = prefix[num_experts];
    *total_tokens_post_pad = s_total_tokens_post_pad;
  }

  /*
  DPCT1113:782: Consider replacing sycl::nd_item::barrier(sycl::access::fence_space::local_space) with
  sycl::nd_item::barrier() if function "moe_align_block_size_kernel" is called in a multidimensional kernel.
  */
  item_ct1.barrier(sycl::access::fence_space::local_space);

  if (tid <= num_experts) {
    cumsum[tid] = prefix[tid];
  }

  // fill expert_ids
  const int32_t num_blocks = s_total_tokens_post_pad / block_size;
  for (int32_t i = tid; i < num_blocks; i += stride) {
    int32_t block_start = i * block_size;
    int left = 0, right = num_experts;
    while (left < right) {
      int mid = (left + right) >> 1;
      if (prefix[mid] <= block_start) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
    expert_ids[i] = left - 1;
  }

  if (pad_sorted_token_ids) {
    Vec fill_vec;
    fill_vec.x() = fill_vec.y() = fill_vec.z() = fill_vec.w() = numel;
    int32_t total_vecs = (s_total_tokens_post_pad + VEC_SIZE - 1) / VEC_SIZE;
    Vec* out_ptr = reinterpret_cast<Vec*>(sorted_token_ids);
    for (int32_t i = tid; i < total_vecs; i += stride) {
      out_ptr[i] = fill_vec;
    }
  }
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
template <typename scalar_t>
void moe_align_block_size_kernel_wrapper(
    const scalar_t* __restrict topk_ids,
    int32_t* __restrict sorted_token_ids,
    int32_t* __restrict expert_ids,
    int32_t* __restrict total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    size_t numel,
    int32_t* __restrict cumsum,
    bool pad_sorted_token_ids,
    const int32_t scan_size) {
  sycl::queue queue = *dpct::kernel_launcher::_que;
  unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
  sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

  auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};

  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(localMemSize), cgh);

    auto last_event = dpct::get_default_queue().ext_oneapi_get_last_event();
    [&](auto&& _e) {
      if constexpr (std::is_same_v<std::remove_reference_t<decltype(last_event)>, sycl::event>)
        cgh.depends_on(_e);
      else if (_e.has_value())
        cgh.depends_on(_e.value());
    }(last_event);

    cgh.parallel_for(nr, exp_props, [=](sycl::nd_item<3> item_ct1) {
      moe_align_block_size_kernel<scalar_t>(
          topk_ids,
          sorted_token_ids,
          expert_ids,
          total_tokens_post_pad,
          num_experts,
          block_size,
          numel,
          cumsum,
          pad_sorted_token_ids,
          scan_size,
          dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
    });
  });
}

template <typename scalar_t>
void moe_align_block_size_small_batch_expert_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    size_t numel,
    bool pad_sorted_token_ids,
    uint8_t *dpct_local) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const size_t tid = item_ct1.get_local_id(2);
  const size_t stride = item_ct1.get_local_range(2);

  auto shared_mem = (int32_t*)dpct_local;
  int32_t* cumsum = shared_mem;
  int32_t* tokens_cnts = (int32_t*)(shared_mem + num_experts + 1);

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[(item_ct1.get_local_id(2) + 1) * num_experts + i] = 0;
  }

  for (size_t i = tid; i < numel; i += stride) {
    ++tokens_cnts[(item_ct1.get_local_id(2) + 1) * num_experts + topk_ids[i]];
  }

  /*
  DPCT1065:785: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global
  memory.
  */
  item_ct1.barrier();

  if (item_ct1.get_local_id(2) < num_experts) {
    tokens_cnts[item_ct1.get_local_id(2)] = 0;
    for (int i = 1; i <= item_ct1.get_local_range(2); ++i) {
      tokens_cnts[i * num_experts + item_ct1.get_local_id(2)] +=
          tokens_cnts[(i - 1) * num_experts + item_ct1.get_local_id(2)];
    }
  }

  /*
  DPCT1065:786: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global
  memory.
  */
  item_ct1.barrier();

  if (item_ct1.get_local_id(2) == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] =
          cumsum[i - 1] +
          CEILDIV(
              tokens_cnts[sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_range(2) * num_experts + i - 1],
              block_size) *
              block_size;
    }
    *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
  }

  /*
  DPCT1065:787: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global
  memory.
  */
  item_ct1.barrier();

  if (item_ct1.get_local_id(2) < num_experts) {
    for (int i = cumsum[item_ct1.get_local_id(2)]; i < cumsum[item_ct1.get_local_id(2) + 1]; i += block_size) {
      expert_ids[i / block_size] = item_ct1.get_local_id(2);
    }
  }

  if (pad_sorted_token_ids) {
    Vec fill_vec;
    fill_vec.x() = fill_vec.y() = fill_vec.z() = fill_vec.w() = numel;
    int32_t total_vecs = (*total_tokens_post_pad + VEC_SIZE - 1) / VEC_SIZE;
    Vec* out_ptr = reinterpret_cast<Vec*>(sorted_token_ids);
    for (int32_t i = tid; i < total_vecs; i += stride) {
      out_ptr[i] = fill_vec;
    }
  }

  /*
  DPCT1065:788: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global
  memory.
  */
  item_ct1.barrier();

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    int32_t rank_post_pad = tokens_cnts[item_ct1.get_local_id(2) * num_experts + expert_id] + cumsum[expert_id];
    sorted_token_ids[rank_post_pad] = i;
    ++tokens_cnts[item_ct1.get_local_id(2) * num_experts + expert_id];
  }
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
template <typename scalar_t>
void moe_align_block_size_small_batch_expert_kernel_wrapper(
    const scalar_t* __restrict topk_ids,
    int32_t* __restrict sorted_token_ids,
    int32_t* __restrict expert_ids,
    int32_t* __restrict total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    size_t numel,
    bool pad_sorted_token_ids) {
  sycl::queue queue = *dpct::kernel_launcher::_que;
  unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
  sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

  auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};

  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(localMemSize), cgh);

    auto last_event = dpct::get_default_queue().ext_oneapi_get_last_event();
    [&](auto&& _e) {
      if constexpr (std::is_same_v<std::remove_reference_t<decltype(last_event)>, sycl::event>)
        cgh.depends_on(_e);
      else if (_e.has_value())
        cgh.depends_on(_e.value());
    }(last_event);

    cgh.parallel_for(nr, exp_props, [=](sycl::nd_item<3> item_ct1) {
      moe_align_block_size_small_batch_expert_kernel<scalar_t>(
          topk_ids,
          sorted_token_ids,
          expert_ids,
          total_tokens_post_pad,
          num_experts,
          block_size,
          numel,
          pad_sorted_token_ids,
          dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
    });
  });
}

void moe_align_block_size(
    torch::Tensor topk_ids,
    int64_t num_experts,
    int64_t block_size,
    torch::Tensor sorted_token_ids,
    torch::Tensor experts_ids,
    torch::Tensor num_tokens_post_pad,
    torch::Tensor token_cnts_buffer,
    torch::Tensor cumsum_buffer,
    bool pad_sorted_token_ids) {
  const dpct::queue_ptr stream = &c10::xpu::getCurrentXPUStream().queue();

  int64_t padded_num_experts = ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

  int experts_per_warp = WARP_SIZE;
  int threads = 1024;

  threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

  /*
  DPCT1038:618: When the kernel function name is used as a macro argument, the migration result may be incorrect. You
  need to verify the definition of the macro.
  */
  DISPATCH_INTEGRAL_TYPES(topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
    bool small_batch_expert_mode = (topk_ids.numel() < 1024) && (num_experts <= 64);

    if (small_batch_expert_mode) {
      const int32_t threads = dpct::max((int32_t)num_experts, WARP_SIZE);
      const int32_t shared_mem_size = ((threads + 1) * num_experts + (num_experts + 1)) * sizeof(int32_t);

      auto small_batch_expert_kernel = moe_align_block_size_small_batch_expert_kernel_wrapper<scalar_t>;
      dpct::kernel_launcher::launch(
          small_batch_expert_kernel,
          1,
          threads,
          shared_mem_size,
          stream,
          topk_ids.data_ptr<scalar_t>(),
          sorted_token_ids.data_ptr<int32_t>(),
          experts_ids.data_ptr<int32_t>(),
          num_tokens_post_pad.data_ptr<int32_t>(),
          num_experts,
          block_size,
          topk_ids.numel(),
          pad_sorted_token_ids);
    } else {
      auto align_kernel = moe_align_block_size_kernel_wrapper<scalar_t>;

      const size_t scan_size = next_pow2(num_experts);
      const size_t shared_mem_size = (num_experts + (num_experts + 1) + scan_size) * sizeof(int32_t);

      dpct::kernel_launcher::launch(
          align_kernel,
          1,
          threads,
          shared_mem_size,
          stream,
          topk_ids.data_ptr<scalar_t>(),
          sorted_token_ids.data_ptr<int32_t>(),
          experts_ids.data_ptr<int32_t>(),
          num_tokens_post_pad.data_ptr<int32_t>(),
          num_experts,
          block_size,
          topk_ids.numel(),
          cumsum_buffer.data_ptr<int32_t>(),
          pad_sorted_token_ids,
          scan_size);

      const int block_threads = std::min(256, (int)threads);
      const int num_blocks = (topk_ids.numel() + block_threads - 1) / block_threads;
      const int max_blocks = 65535;
      const int actual_blocks = std::min(num_blocks, max_blocks);

      auto sort_kernel = count_and_sort_expert_tokens_kernel_wrapper<scalar_t>;
      dpct::kernel_launcher::launch(
          sort_kernel,
          actual_blocks,
          block_threads,
          0,
          stream,
          topk_ids.data_ptr<scalar_t>(),
          sorted_token_ids.data_ptr<int32_t>(),
          cumsum_buffer.data_ptr<int32_t>(),
          topk_ids.numel());
    }
  });
}
