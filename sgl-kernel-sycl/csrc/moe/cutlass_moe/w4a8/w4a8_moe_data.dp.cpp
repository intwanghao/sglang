#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <c10/xpu/XPUStream.h>
#include <c10/core/DeviceGuard.h>

#include <torch/all.h>

#include <iostream>
#include <c10/xpu/XPUStream.h>

constexpr uint64_t THREADS_PER_EXPERT = 512;

void compute_problem_sizes_w4a8(
    const int32_t* __restrict__ topk_ids,
    int32_t* problem_sizes1,
    int32_t* problem_sizes2,
    int32_t* atomic_buffer,
    const int topk_length,
    const int n,
    const int k) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int expert_id = item_ct1.get_group(2);

  int occurrences = 0;
  for (int i = item_ct1.get_local_id(2); i < topk_length; i += THREADS_PER_EXPERT) {
    occurrences += (topk_ids[i] == expert_id);
  }
  dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&atomic_buffer[expert_id], occurrences);
  /*
  DPCT1065:749: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global
  memory.
  */
  item_ct1.barrier();

  if (item_ct1.get_local_id(2) == 0) {
    int final_occurrences = atomic_buffer[expert_id];
    problem_sizes1[expert_id * 3] = 2 * n;
    problem_sizes1[expert_id * 3 + 1] = final_occurrences;
    problem_sizes1[expert_id * 3 + 2] = k;
    problem_sizes2[expert_id * 3] = k;
    problem_sizes2[expert_id * 3 + 1] = final_occurrences;
    problem_sizes2[expert_id * 3 + 2] = n;
  }
}

void compute_expert_offsets_w4a8(
    const int32_t* __restrict__ problem_sizes1,
    int32_t* expert_offsets,
    int32_t* atomic_buffer,
    const int num_experts) {
  int32_t tot_offset = 0;
  expert_offsets[0] = 0;
  for (int i = 0; i < num_experts; ++i) {
    atomic_buffer[i] = tot_offset;
    tot_offset += problem_sizes1[i * 3 + 1];
    expert_offsets[i + 1] = tot_offset;
  }
}

void get_cutlass_w4a8_moe_mm_data_caller(
    const torch::Tensor& topk_ids,
    torch::Tensor& expert_offsets,
    torch::Tensor& problem_sizes1,
    torch::Tensor& problem_sizes2,
    torch::Tensor& input_permutation,
    torch::Tensor& output_permutation,
    const int64_t num_experts,
    const int64_t n,
    const int64_t k) {
  auto stream = &c10::xpu::getCurrentXPUStream(topk_ids.device().index()).queue();
  auto options_int32 = torch::TensorOptions().dtype(torch::kInt32).device(topk_ids.device());
  torch::Tensor atomic_buffer = torch::zeros(num_experts, options_int32);

  int num_threads = dpct::min(THREADS_PER_EXPERT, topk_ids.numel());
  /*
  DPCT1049:595: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
  info::device::max_work_group_size. Adjust the work-group size if needed.
  */
  {
    auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};

    ((sycl::queue*)(stream))->submit([&](sycl::handler& cgh) {
      auto topk_ids_data_ptr_ct0 = static_cast<const int32_t*>(topk_ids.data_ptr());
      auto problem_sizes1_data_ptr_ct1 = static_cast<int32_t*>(problem_sizes1.data_ptr());
      auto problem_sizes2_data_ptr_ct2 = static_cast<int32_t*>(problem_sizes2.data_ptr());
      auto atomic_buffer_data_ptr_ct3 = static_cast<int32_t*>(atomic_buffer.data_ptr());
      const int topk_ids_numel_ct4 = topk_ids.numel();

      auto last_event = dpct::get_default_queue().ext_oneapi_get_last_event();
      [&](auto&& _e) {
        if constexpr (std::is_same_v<std::remove_reference_t<decltype(last_event)>, sycl::event>)
          cgh.depends_on(_e);
        else if (_e.has_value())
          cgh.depends_on(_e.value());
      }(last_event);

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(1, 1, num_experts) * sycl::range<3>(1, 1, num_threads), sycl::range<3>(1, 1, num_threads)),
          exp_props,
          [=](sycl::nd_item<3> item_ct1) {
            compute_problem_sizes_w4a8(
                topk_ids_data_ptr_ct0,
                problem_sizes1_data_ptr_ct1,
                problem_sizes2_data_ptr_ct2,
                atomic_buffer_data_ptr_ct3,
                topk_ids_numel_ct4,
                n,
                k);
          });
    });
  }
  {
    auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};

    ((sycl::queue*)(stream))->submit([&](sycl::handler& cgh) {
      auto problem_sizes1_data_ptr_ct0 = static_cast<const int32_t*>(problem_sizes1.data_ptr());
      auto expert_offsets_data_ptr_ct1 = static_cast<int32_t*>(expert_offsets.data_ptr());
      auto atomic_buffer_data_ptr_ct2 = static_cast<int32_t*>(atomic_buffer.data_ptr());

      auto last_event = dpct::get_default_queue().ext_oneapi_get_last_event();
      [&](auto&& _e) {
        if constexpr (std::is_same_v<std::remove_reference_t<decltype(last_event)>, sycl::event>)
          cgh.depends_on(_e);
        else if (_e.has_value())
          cgh.depends_on(_e.value());
      }(last_event);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          exp_props,
          [=](sycl::nd_item<3> item_ct1) {
            compute_expert_offsets_w4a8(
                problem_sizes1_data_ptr_ct0, expert_offsets_data_ptr_ct1, atomic_buffer_data_ptr_ct2, num_experts);
          });
    });
  }
}
