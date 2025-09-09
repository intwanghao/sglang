#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <c10/xpu/XPUStream.h>
#include <c10/core/DeviceGuard.h>

#include <torch/all.h>

#include <flashinfer/vec_dtypes.dp.hpp>
#include <iostream>

#include "cutlass/array.h"
#include "utils.h"
#include <c10/xpu/XPUStream.h>

#include <cmath>

constexpr uint64_t THREADS_PER_EXPERT = 512;

void compute_problem_sizes(
    const int* __restrict__ topk_ids,
    int32_t* problem_sizes1,
    int32_t* problem_sizes2,
    int32_t* atomic_buffer,
    const int64_t topk_length,
    const int64_t n,
    const int64_t k) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int expert_id = item_ct1.get_group(2);

  int occurrences = 0;
  for (int i = item_ct1.get_local_id(2); i < topk_length; i += THREADS_PER_EXPERT) {
    occurrences += (topk_ids[i] == expert_id);
  }
  dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&atomic_buffer[expert_id], occurrences);
  /*
  DPCT1065:805: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global
  memory.
  */
  item_ct1.barrier();

  if (item_ct1.get_local_id(2) == 0) {
    int final_occurrences = atomic_buffer[expert_id];
    problem_sizes1[expert_id * 3] = final_occurrences;
    problem_sizes1[expert_id * 3 + 1] = static_cast<int32_t>(2 * n);
    problem_sizes1[expert_id * 3 + 2] = static_cast<int32_t>(k);
    problem_sizes2[expert_id * 3] = final_occurrences;
    problem_sizes2[expert_id * 3 + 1] = static_cast<int32_t>(k);
    problem_sizes2[expert_id * 3 + 2] = static_cast<int32_t>(n);
  }
}

void compute_expert_offsets(
    const int32_t* __restrict__ problem_sizes1,
    int32_t* expert_offsets,
    int32_t* atomic_buffer,
    const int64_t num_experts) {
  int32_t tot_offset = 0;
  expert_offsets[0] = 0;
  for (int i = 0; i < num_experts; ++i) {
    atomic_buffer[i] = tot_offset;
    tot_offset += problem_sizes1[i * 3];
    expert_offsets[i + 1] = tot_offset;
  }
}

void compute_expert_blockscale_offsets(
    const int32_t* __restrict__ problem_sizes1,
    int32_t* expert_offsets,
    int32_t* blockscale_offsets,
    int32_t* atomic_buffer,
    const int64_t num_experts) {
  int32_t tot_offset = 0;
  int32_t tot_rounded_offset = 0;
  expert_offsets[0] = 0;
  blockscale_offsets[0] = 0;
  for (int i = 0; i < num_experts; ++i) {
    atomic_buffer[i] = tot_offset;
    int num_tokens = problem_sizes1[i * 3];
    int rounded_num_tokens = (num_tokens + (128 - 1)) / 128 * 128;
    tot_offset += num_tokens;
    tot_rounded_offset += rounded_num_tokens;
    expert_offsets[i + 1] = tot_offset;
    blockscale_offsets[i + 1] = tot_rounded_offset;
  }
}

void compute_arg_sorts(
    const int32_t* __restrict__ topk_ids,
    int32_t* input_permutation,
    int32_t* output_permutation,
    int32_t* atomic_buffer,
    const int64_t topk_length,
    const int64_t topk) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int expert_id = item_ct1.get_group(2);

  for (int i = item_ct1.get_local_id(2); i < topk_length; i += THREADS_PER_EXPERT) {
    if (topk_ids[i] == expert_id) {
      int start = dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&atomic_buffer[expert_id], 1);
      input_permutation[start] = i / topk;
      output_permutation[i] = start;
    }
  }
}

void get_moe_prepare_input_caller(
    const torch::Tensor& topk_ids,
    torch::Tensor& expert_offsets,
    const std::optional<torch::Tensor>& blockscale_offsets,
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

  uint32_t num_threads = static_cast<uint32_t>(dpct::min(THREADS_PER_EXPERT, topk_ids.numel()));
  uint32_t num_blocks = static_cast<uint32_t>(num_experts);

  /*
  DPCT1049:649: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
  info::device::max_work_group_size. Adjust the work-group size if needed.
  */
  {
    auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};

    ((sycl::queue*)(stream))->submit([&](sycl::handler& cgh) {
      auto topk_ids_data_ptr_ct0 = static_cast<const int32_t*>(topk_ids.data_ptr());
      auto problem_sizes1_data_ptr_ct1 = static_cast<int32_t*>(problem_sizes1.data_ptr());
      auto problem_sizes2_data_ptr_ct2 = static_cast<int32_t*>(problem_sizes2.data_ptr());
      auto atomic_buffer_data_ptr_ct3 = static_cast<int32_t*>(atomic_buffer.data_ptr());
      auto topk_ids_numel_ct4 = topk_ids.numel();

      auto last_event = dpct::get_default_queue().ext_oneapi_get_last_event();
      [&](auto&& _e) {
        if constexpr (std::is_same_v<std::remove_reference_t<decltype(last_event)>, sycl::event>)
          cgh.depends_on(_e);
        else if (_e.has_value())
          cgh.depends_on(_e.value());
      }(last_event);

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, num_threads), sycl::range<3>(1, 1, num_threads)),
          exp_props,
          [=](sycl::nd_item<3> item_ct1) {
            compute_problem_sizes(
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
  if (blockscale_offsets.has_value()) {
    auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};

    ((sycl::queue*)(stream))->submit([&](sycl::handler& cgh) {
      auto problem_sizes1_data_ptr_ct0 = static_cast<const int32_t*>(problem_sizes1.data_ptr());
      auto expert_offsets_data_ptr_ct1 = static_cast<int32_t*>(expert_offsets.data_ptr());
      auto blockscale_offsets_value_data_ptr_ct2 = static_cast<int32_t*>(blockscale_offsets.value().data_ptr());
      auto atomic_buffer_data_ptr_ct3 = static_cast<int32_t*>(atomic_buffer.data_ptr());

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
            compute_expert_blockscale_offsets(
                problem_sizes1_data_ptr_ct0,
                expert_offsets_data_ptr_ct1,
                blockscale_offsets_value_data_ptr_ct2,
                atomic_buffer_data_ptr_ct3,
                num_experts);
          });
    });
  } else {
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
            compute_expert_offsets(
                problem_sizes1_data_ptr_ct0, expert_offsets_data_ptr_ct1, atomic_buffer_data_ptr_ct2, num_experts);
          });
    });
  }
  /*
  DPCT1049:650: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
  info::device::max_work_group_size. Adjust the work-group size if needed.
  */
  {
    auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};

    ((sycl::queue*)(stream))->submit([&](sycl::handler& cgh) {
      auto topk_ids_data_ptr_ct0 = static_cast<const int32_t*>(topk_ids.data_ptr());
      auto input_permutation_data_ptr_ct1 = static_cast<int32_t*>(input_permutation.data_ptr());
      auto output_permutation_data_ptr_ct2 = static_cast<int32_t*>(output_permutation.data_ptr());
      auto atomic_buffer_data_ptr_ct3 = static_cast<int32_t*>(atomic_buffer.data_ptr());
      auto topk_ids_numel_ct4 = topk_ids.numel();
      auto topk_ids_size_ct5 = topk_ids.size(1);

      auto last_event = dpct::get_default_queue().ext_oneapi_get_last_event();
      [&](auto&& _e) {
        if constexpr (std::is_same_v<std::remove_reference_t<decltype(last_event)>, sycl::event>)
          cgh.depends_on(_e);
        else if (_e.has_value())
          cgh.depends_on(_e.value());
      }(last_event);

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, num_threads), sycl::range<3>(1, 1, num_threads)),
          exp_props,
          [=](sycl::nd_item<3> item_ct1) {
            compute_arg_sorts(
                topk_ids_data_ptr_ct0,
                input_permutation_data_ptr_ct1,
                output_permutation_data_ptr_ct2,
                atomic_buffer_data_ptr_ct3,
                topk_ids_numel_ct4,
                topk_ids_size_ct5);
          });
    });
  }
}

void prepare_moe_input(
    const torch::Tensor& topk_ids,
    torch::Tensor& expert_offsets,
    const std::optional<torch::Tensor>& blockscale_offsets,
    torch::Tensor& problem_sizes1,
    torch::Tensor& problem_sizes2,
    torch::Tensor& input_permutation,
    torch::Tensor& output_permutation,
    const int64_t num_experts,
    const int64_t n,
    const int64_t k) {
  TORCH_CHECK(topk_ids.dtype() == torch::kInt32);
  get_moe_prepare_input_caller(
      topk_ids,
      expert_offsets,
      blockscale_offsets,
      problem_sizes1,
      problem_sizes2,
      input_permutation,
      output_permutation,
      num_experts,
      n,
      k);
  return;
}

template <typename T>
void shuffleRowsKernel(
    const T* input,
    const int32_t* dst2src_map,
    T* output,
    int64_t num_src_rows,
    int64_t num_dst_rows,
    int64_t num_cols) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int64_t dest_row_idx = item_ct1.get_group(2);
  int64_t const source_row_idx = dst2src_map[dest_row_idx];

  if (item_ct1.get_group(2) < num_dst_rows) {
    // Load 128-bits per thread
    constexpr uint64_t ELEM_PER_THREAD = 128 / sizeof(T) / 8;
    using DataElem = cutlass::Array<T, ELEM_PER_THREAD>;

    // Duplicate and permute rows
    auto const* source_row_ptr = reinterpret_cast<DataElem const*>(input + source_row_idx * num_cols);
    auto* dest_row_ptr = reinterpret_cast<DataElem*>(output + dest_row_idx * num_cols);

    auto const start_offset = item_ct1.get_local_id(2);
    auto const stride = item_ct1.get_local_range(2);
    auto const num_elems_in_col = num_cols / ELEM_PER_THREAD;

    for (auto elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
      dest_row_ptr[elem_index] = source_row_ptr[elem_index];
    }
  }
}

#define DECLARE_SHUFFLE_ROWS(T)      \
  void shuffleRowsKernel( \
      const T* input,                \
      const int32_t* dst2src_map,    \
      T* output,                     \
      int64_t num_src_rows,          \
      int64_t num_dest_rows,         \
      int64_t num_cols);

DECLARE_SHUFFLE_ROWS(float);
DECLARE_SHUFFLE_ROWS(sycl::half);
DECLARE_SHUFFLE_ROWS(sycl::ext::oneapi::bfloat16);
//DECLARE_SHUFFLE_ROWS(__nv_fp8_e4m3);
DECLARE_SHUFFLE_ROWS(uint8_t);

/*
DPCT1049:651: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
info::device::max_work_group_size. Adjust the work-group size if needed.
*/
#define SHUFFLE_ROWS(T)                                                                                             \
  {                                                                                                                 \
    auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};   \
    dpct::has_capability_or_fail(c10::xpu::getCurrentXPUStream().queue().get_device(), {sycl::aspect::fp16});       \
                                                                                                                    \
    stream->submit([&](sycl::handler& cgh) {                                                                        \
      auto input_ct0 = reinterpret_cast<const T*>(input);                                                           \
      auto dst2src_map_data_ptr_ct1 = static_cast<const int32_t*>(dst2src_map.data_ptr());                          \
      auto output_ct2 = reinterpret_cast<T*>(output);                                                               \
      auto num_src_rows_ct3 = num_src_rows;                                                                         \
      auto num_dst_rows_ct4 = num_dst_rows;                                                                         \
      auto num_cols_ct5 = num_cols;                                                                                 \
                                                                                                                    \
      auto last_event = dpct::get_default_queue().ext_oneapi_get_last_event();                                      \
      [&](auto&& _e) {                                                                                              \
        if constexpr (std::is_same_v<std::remove_reference_t<decltype(last_event)>, sycl::event>)                   \
          cgh.depends_on(_e);                                                                                       \
        else if (_e.has_value())                                                                                    \
          cgh.depends_on(_e.value());                                                                               \
      }(last_event);                                                                                                \
                                                                                                                    \
      cgh.parallel_for(                                                                                             \
          sycl::nd_range<3>(                                                                                        \
              sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)),         \
          exp_props,                                                                                                \
          [=](sycl::nd_item<3> item_ct1) {                                                                          \
            shuffleRowsKernel<T>(                                                                                   \
                input_ct0, dst2src_map_data_ptr_ct1, output_ct2, num_src_rows_ct3, num_dst_rows_ct4, num_cols_ct5); \
          });                                                                                                       \
    });                                                                                                             \
  }

#define DTYPE_DISPATCH_CASE(T, CUDA_T) \
  case T:                              \
    SHUFFLE_ROWS(CUDA_T);              \
    break;

void shuffle_rows_caller(
    const torch::Tensor& input_tensor, const torch::Tensor& dst2src_map, torch::Tensor& output_tensor) {
  TORCH_CHECK(
      input_tensor.scalar_type() == output_tensor.scalar_type(),
      "Input and output tensors must have the same data type");
  auto stream = &(c10::xpu::getCurrentXPUStream().queue());
  uint32_t blocks = static_cast<uint32_t>(output_tensor.size(0));
  uint32_t threads = 256;
  int64_t num_dst_rows = output_tensor.size(0);
  int64_t num_src_rows = input_tensor.size(0);
  int64_t num_cols = input_tensor.size(1);
  const void* input = input_tensor.data_ptr();
  void* output = output_tensor.data_ptr();
  switch (input_tensor.scalar_type()) {
    DTYPE_DISPATCH_CASE(torch::kFloat16, sycl::half);
    DTYPE_DISPATCH_CASE(torch::kBFloat16, sycl::ext::oneapi::bfloat16);
    DTYPE_DISPATCH_CASE(torch::kFloat32, float);
    //DTYPE_DISPATCH_CASE(torch::kFloat8_e4m3fn, __nv_fp8_e4m3);
    DTYPE_DISPATCH_CASE(torch::kUInt8, uint8_t);
    default:
      TORCH_CHECK(false, "[moe replicate input] data type dispatch fail!");
  }
  return;
}

void shuffle_rows(const torch::Tensor& input_tensor, const torch::Tensor& dst2src_map, torch::Tensor& output_tensor) {
  shuffle_rows_caller(input_tensor, dst2src_map, output_tensor);
  return;
}

template <typename scalar_t>
/*
DPCT1110:652: The total declared local variable size in device function apply_shuffle_mul_sum_kernel exceeds 128 bytes
and may cause high register pressure. Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void apply_shuffle_mul_sum_kernel(
    const scalar_t* __restrict__ input_tensor,  // [m * topk, k] (expert-major layout)
    scalar_t* __restrict__ output_tensor,       // [m, k] (token-major layout)
    const int32_t* __restrict__ permutation,    // [m * topk] (c_map: token-major-idx -> expert-major-idx)
    int m,
    int topk,
    int row_stride,
    const scalar_t* __restrict__ factors)  // [m * topk] (topk_weights, token-major layout)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int i = item_ct1.get_group(2);
  if (i >= m) {
    return;
  }

  constexpr uint32_t vec_size = 16 / sizeof(scalar_t);
  using t = float;
  using vec_t = flashinfer::vec_t<t, vec_size>;
  int thread_idx = item_ct1.get_local_id(2);
  int stride = item_ct1.get_local_range(2);

  for (int d_vec_idx = thread_idx; d_vec_idx < row_stride / vec_size; d_vec_idx += stride) {
    int d = d_vec_idx * vec_size;
    vec_t sum_vec;
    sum_vec.fill(0.0f);

    for (int j = 0; j < topk; ++j) {
      int token_major_idx = i * topk + j;
      int src_row = permutation[token_major_idx];

      vec_t val_vec;
      val_vec.cast_load(input_tensor + src_row * row_stride + d);

      t factor = 1.0;
      if (factors != nullptr) {
        factor = factors[token_major_idx];
      }

#pragma unroll
      for (int k = 0; k < vec_size; ++k) {
        sum_vec[k] += factor * val_vec[k];
      }
    }
    sum_vec.cast_store(output_tensor + i * row_stride + d);
  }

  //  remainder part
  int remainder_start = (row_stride / vec_size) * vec_size;
  for (int d = remainder_start + thread_idx; d < row_stride; d += stride) {
    t sum_val = 0.0;
    for (int j = 0; j < topk; ++j) {
      int token_major_idx = i * topk + j;
      int src_row = permutation[token_major_idx];
      t val = input_tensor[src_row * row_stride + d];

      t factor = 1.0;
      if (factors != nullptr) {
        factor = factors[token_major_idx];
      }
      sum_val += factor * val;
    }
    output_tensor[i * row_stride + d] = sum_val;
  }
}

void get_apply_shuffle_mul_sum_caller(
    const torch::Tensor& input_tensor,                // [m * topk, row_stride], bf16/f16
    torch::Tensor& output_tensor,                     // [m, row_stride], bf16/f16
    const torch::Tensor& permutation,                 // [m * topk], int32
    const std::optional<torch::Tensor>& factors_opt)  // optional [m * topk], bf16/f16
{
  TORCH_CHECK(input_tensor.dim() == 2, "input_tensor must be 2D [m * topk, row_stride]");
  TORCH_CHECK(output_tensor.dim() == 2, "output_tensor must be 2D [m, row_stride]");
  TORCH_CHECK(permutation.dim() == 1, "permutation must be 1D [m * topk]");

  int m = output_tensor.size(0);
  int topk = int(permutation.size(0) / m);
  int row_stride = output_tensor.size(1);

  TORCH_CHECK(permutation.size(0) == m * topk, "permutation size must match m * topk");

  auto scalar_type = output_tensor.scalar_type();
  uint32_t vec_size = 16 / sizeof(scalar_type);
  auto blockDim = std::min(row_stride / vec_size, 1024U);
  dpct::dim3 block(blockDim);

  dpct::dim3 grid(m);  // blockIdx.x = j, blockIdx.y = i
  auto stream = &c10::xpu::getCurrentXPUStream(input_tensor.device().index()).queue();

  const int32_t* perm_ptr = permutation.data_ptr<int32_t>();

  void* factors_ptr = nullptr;
  if (factors_opt.has_value()) {
    TORCH_CHECK(factors_opt->dtype() == output_tensor.dtype(), "Factors must match output dtype");
    TORCH_CHECK(factors_opt->numel() == m * topk, "Factors must have shape [m * topk]");
    factors_ptr = factors_opt->data_ptr();
  }

  /*
  DPCT1038:653: When the kernel function name is used as a macro argument, the migration result may be incorrect. You
  need to verify the definition of the macro.
  */
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(output_tensor.scalar_type(), scalar_t, [&] {
    /*
    DPCT1049:654: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
      auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
      dpct::has_capability_or_fail(c10::xpu::getCurrentXPUStream().queue().get_device(), {sycl::aspect::fp16});

      ((sycl::queue*)(stream))->submit([&](sycl::handler& cgh) {
        auto input_tensor_data_ptr_ct0 = static_cast<const scalar_t*>(input_tensor.data_ptr());
        auto output_tensor_data_ptr_ct1 = static_cast<scalar_t*>(output_tensor.data_ptr());

        auto last_event = dpct::get_default_queue().ext_oneapi_get_last_event();
        [&](auto&& _e) {
          if constexpr (std::is_same_v<std::remove_reference_t<decltype(last_event)>, sycl::event>)
            cgh.depends_on(_e);
          else if (_e.has_value())
            cgh.depends_on(_e.value());
        }(last_event);

        cgh.parallel_for(sycl::nd_range<3>(grid * block, block), exp_props, [=](sycl::nd_item<3> item_ct1) {
          apply_shuffle_mul_sum_kernel<scalar_t>(
              input_tensor_data_ptr_ct0,
              output_tensor_data_ptr_ct1,
              perm_ptr,
              m,
              topk,
              row_stride,
              static_cast<const scalar_t*>(factors_ptr));
        });
      });
    }
    return true;
  });
}

/**
 * @brief Applies a permutation-based shuffle, element-wise multiplication, and reduction over the second dimension.
 *
 * This function performs the equivalent of the following PyTorch expression:
 *
 *     (c2[c_map].view(m, topk, k) * topk_weights.view(m, topk, 1).to(out_dtype)).sum(dim=1)
 *
 * Specifically:
 * - `input` is shuffled using the `permutation` tensor.
 * - The shuffled tensor is reshaped and multiplied element-wise with `factors` (e.g., top-k weights).
 * - The result is summed along dimension 1 (the top-k dimension), and stored in `output`.
 *
 * @param input        Input tensor of shape (m * topk, k), representing c2.
 * @param output       Output tensor of shape (m, k), where the final reduced results are stored.
 * @param permutation  Index tensor (e.g., c_map) that maps positions in `input` to shuffled layout.
 * @param factors      Optional scaling factors (e.g., top-k weights), shape (m * topk) or (m, topk).
 */
void apply_shuffle_mul_sum(
    const torch::Tensor& input,
    torch::Tensor& output,
    const torch::Tensor& permutation,
    const std::optional<torch::Tensor>& factors) {
  get_apply_shuffle_mul_sum_caller(input, output, permutation, factors);
}
