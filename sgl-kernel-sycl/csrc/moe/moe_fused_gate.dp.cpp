#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <c10/xpu/XPUStream.h>
#include <ATen/xpu/XPUContext.h>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <stdio.h>
#include <torch/all.h>

#include <cfloat>
#include <type_traits>
#include <c10/xpu/XPUStream.h>

#include <cmath>

template <typename T, int N>
using AlignedArray = cutlass::AlignedArray<T, N>;
using bfloat16_t = cutlass::bfloat16_t;
using float16_t = cutlass::half_t;
using float32_t = float;

// QQ NOTE: to handle the case for at::Half, error: more than one operator ">" matches these operands: built-in operator
// "arithmetic > arithmetic" function "operator>(const __half &, const __half &)"
template <typename T>
inline bool cmp_gt(const T& a, const T& b) {
  if constexpr (std::is_same<T, at::Half>::value) {
    // at::Half (or float16_t in our native case) causes ambiguity, so we cast to float.
    return static_cast<float>(a) > static_cast<float>(b);
  } else {
    // For types like float, at::BFloat16, or cutlass::half_t / cutlass::bfloat16_t, assume operator> works as expected.
    return a > b;
  }
}

template <typename T>
inline bool cmp_eq(const T& a, const T& b) {
  if constexpr (std::is_same<T, at::Half>::value) {
    return static_cast<float>(a) == static_cast<float>(b);
  } else {
    return a == b;
  }
}

// Fixed constants common to both dynamic and static template versions:
static constexpr int WARP_SIZE = 32;
static constexpr int WARPS_PER_CTA = 6;
static constexpr int MAX_VPT = 32;  // maximum VPT we support, > params.VPT = num_expert / num_expert_group

// Create an alias for Array using AlignedArray
template <typename T, int N>
using Array = AlignedArray<T, N>;
// QQ: NOTE expression must have a constant value, this has to be > params.VPT
template <typename T>
using AccessType = AlignedArray<T, MAX_VPT>;

template <typename T, typename Params>
/*
DPCT1110:628: The total declared local variable size in device function moe_fused_gate_impl exceeds 128 bytes and may
cause high register pressure. Consult with your hardware vendor to find the total register size available and adjust the
code, or use smaller sub-group size to avoid high register pressure.
*/
void moe_fused_gate_impl(
    void* input,
    void* bias,
    float* output_ptr,
    int32_t* indices_ptr,
    int64_t num_rows,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor,
    Params params) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int tidx = item_ct1.get_local_id(2);
  int64_t thread_row = item_ct1.get_group(2) * params.ROWS_PER_CTA + item_ct1.get_local_id(1) * params.ROWS_PER_WARP +
                       tidx / params.THREADS_PER_ROW;
  //if (thread_row >= num_rows) {
  //  return;
  //}
  // Calculate topk_excluding_share_expert_fusion from topk
  int64_t topk_excluding_share_expert_fusion = topk - num_fused_shared_experts;

  // Cast pointers to type T:
  auto* input_ptr = reinterpret_cast<T*>(input);
  auto* bias_ptr = reinterpret_cast<T*>(bias);
  auto* thread_row_ptr = input_ptr + thread_row * params.NUM_EXPERTS;

  int thread_group_idx = tidx % params.THREADS_PER_ROW;
  int first_elt_read_by_thread = thread_group_idx * params.VPT;

  // Create local arrays for the row chunk and bias chunk and then reinterpret the address of row_chunk as a pointer to
  // AccessType.
  T* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;
  Array<T, MAX_VPT> row_chunk;
  AccessType<T> const* vec_thread_read_ptr = reinterpret_cast<AccessType<T> const*>(thread_read_ptr);

  T* bias_thread_read_ptr = bias_ptr + first_elt_read_by_thread;
  Array<T, MAX_VPT> bias_chunk;
  AccessType<T> const* vec_bias_thread_read_ptr = reinterpret_cast<AccessType<T> const*>(bias_thread_read_ptr);

// QQ NOTE: doing the follow will be slower than loop assign and more importantly
// have misaligned address issue when params.VPT < 8 and mismatch with MAX_VPT
// AccessType<T>* row_chunk_vec_ptr = reinterpret_cast<AccessType<T>*>(&row_chunk);
// row_chunk_vec_ptr[0] = vec_thread_read_ptr[0];
if (thread_row < num_rows) {
#pragma unroll
  for (int ii = 0; ii < params.VPT; ++ii) {
    row_chunk[ii] = vec_thread_read_ptr[0][ii];
    bias_chunk[ii] = vec_bias_thread_read_ptr[0][ii];
  }
}
  /*
  DPCT1065:789: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global
  memory.
  */
  sycl::group_barrier(item_ct1.get_group());
if (thread_row < num_rows) {
////////////////////// Sigmoid //////////////////////
#pragma unroll
  for (int ii = 0; ii < params.VPT; ++ii) {
    row_chunk[ii] = static_cast<T>(1.0f / (1.0f + sycl::native::exp(-float(row_chunk[ii]))));
  }
}
  /*
  DPCT1065:790: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global
  memory.
  */
  sycl::group_barrier(item_ct1.get_group());
if (thread_row < num_rows) {
////////////////////// Add Bias //////////////////////
#pragma unroll
  for (int ii = 0; ii < params.VPT; ++ii) {
    bias_chunk[ii] = row_chunk[ii] + bias_chunk[ii];
  }

////////////////////// Exclude Groups //////////////////////
#pragma unroll
  for (int k_idx = 0; k_idx < params.THREADS_PER_ROW - topk_group;
       ++k_idx) {  // QQ NOTE Here params.THREADS_PER_ROW = num_expert_group
    int expert = first_elt_read_by_thread;
    // local argmax
    T max_val = static_cast<T>(-FLT_MAX);
    T max_val_second = static_cast<T>(-FLT_MAX);
#pragma unroll
    for (int ii = 0; ii < params.VPT; ++ii) {
      T val = bias_chunk[ii];

      if (cmp_gt(val, max_val)) {
        max_val_second = max_val;
        max_val = val;
      } else if (cmp_gt(val, max_val_second)) {
        max_val_second = val;
      }
    }

    // QQ NOTE: currently fixed to pick top2 sigmoid weight value in each expert group and sum them as the group weight
    // to select expert groups
    T max_sum = max_val + max_val_second;

// argmin reduce
#pragma unroll
    for (int mask = params.THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      T other_max_sum =
          /*
          DPCT1108:630: '__shfl_xor_sync' was migrated with the experimental feature masked sub_group function which may
          not be supported by all compilers or runtimes. You may need to adjust the code.
          */
          static_cast<T>(dpct::experimental::permute_sub_group_by_xor(
              0xFFFFFFFF,
              sycl::ext::oneapi::this_work_item::get_sub_group(),
              static_cast<float>(max_sum),
              mask,
              params.THREADS_PER_ROW));
      /*
      DPCT1108:631: '__shfl_xor_sync' was migrated with the experimental feature masked sub_group function which may not
      be supported by all compilers or runtimes. You may need to adjust the code.
      */
      int other_expert = dpct::experimental::permute_sub_group_by_xor(
          0xFFFFFFFF, sycl::ext::oneapi::this_work_item::get_sub_group(), expert, mask, params.THREADS_PER_ROW);

      // higher indices win
      if (cmp_gt(max_sum, other_max_sum) || (cmp_eq(other_max_sum, max_sum) && other_expert > expert)) {
        max_sum = other_max_sum;
        expert = other_expert;
      }
    }

    // clear the max value in the thread
    if (k_idx < params.THREADS_PER_ROW - topk_group) {
      int const thread_to_clear_in_group = expert / params.VPT;

      if (thread_group_idx == thread_to_clear_in_group) {
#pragma unroll
        for (int ii = 0; ii < params.VPT; ++ii) {
          bias_chunk[ii] = static_cast<T>(FLT_MAX);
        }
      }
    }
  }
}
  /*
  DPCT1065:791: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global
  memory.
  */
  sycl::group_barrier(item_ct1.get_group());

  ////////////////////// Topk //////////////////////
  float output_sum = 0.0f;
  for (int k_idx = 0; k_idx < topk_excluding_share_expert_fusion; ++k_idx) {
    if (thread_row < num_rows) {
    // local argmax
    T max_val = bias_chunk[0];
    int expert = first_elt_read_by_thread;

    if (!cmp_eq(max_val, static_cast<T>(FLT_MAX))) {
#pragma unroll
      for (int ii = 1; ii < params.VPT; ++ii) {
        T val = bias_chunk[ii];
        if (cmp_gt(val, max_val)) {
          max_val = val;
          expert = first_elt_read_by_thread + ii;
        }
      }
    } else {
      max_val = static_cast<T>(-FLT_MAX);
    }

    // argmax reduce
#pragma unroll
    for (int mask = params.THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      T other_max =
          /*
          DPCT1108:632: '__shfl_xor_sync' was migrated with the experimental feature masked sub_group function which may
          not be supported by all compilers or runtimes. You may need to adjust the code.
          */
          static_cast<T>(dpct::experimental::permute_sub_group_by_xor(
              0xFFFFFFFF,
              sycl::ext::oneapi::this_work_item::get_sub_group(),
              static_cast<float>(max_val),
              mask,
              params.THREADS_PER_ROW));
      /*
      DPCT1108:633: '__shfl_xor_sync' was migrated with the experimental feature masked sub_group function which may not
      be supported by all compilers or runtimes. You may need to adjust the code.
      */
      int other_expert = dpct::experimental::permute_sub_group_by_xor(
          0xFFFFFFFF, sycl::ext::oneapi::this_work_item::get_sub_group(), expert, mask, params.THREADS_PER_ROW);

      // lower indices to win
      if (cmp_gt(other_max, max_val) || (cmp_eq(other_max, max_val) && other_expert < expert)) {
        max_val = other_max;
        expert = other_expert;
      }
    }

    int thread_to_clear_in_group = expert / params.VPT;
    int64_t idx = topk * thread_row + k_idx;

    if (thread_group_idx == thread_to_clear_in_group) {
      int expert_to_clear_in_thread = expert % params.VPT;

      // clear the max value in the thread
      bias_chunk[expert_to_clear_in_thread] = static_cast<T>(-FLT_MAX);

      // store output
      output_ptr[idx] = static_cast<float>(row_chunk[expert_to_clear_in_thread]);
      indices_ptr[idx] = static_cast<int32_t>(expert);
    }

    // accumulate sum for all elements
    if (thread_group_idx == 0) {
      output_sum += output_ptr[idx];
    }

    /*
    DPCT1118:629: SYCL group functions and algorithms must be encountered in converged control flow. You may need to
    adjust the code.
    */
    /*
    DPCT1065:793: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to
    global memory.
    */
    }
    sycl::group_barrier(item_ct1.get_group());
  }
if (thread_row < num_rows) {
  if (thread_group_idx == 0 && num_fused_shared_experts > 0) {
    int64_t last_idx = topk * thread_row + topk_excluding_share_expert_fusion;
    int64_t expert_offset = 0;
    indices_ptr[last_idx] = static_cast<int32_t>(params.NUM_EXPERTS + expert_offset);

    // Set the weight to the sum of all weights divided by routed_scaling_factor
    output_ptr[last_idx] = output_sum / routed_scaling_factor;

    if (num_fused_shared_experts > 1) {
      for (int i = 1; i < num_fused_shared_experts; ++i) {
        ++last_idx;
        ++expert_offset;
        indices_ptr[last_idx] = static_cast<int32_t>(params.NUM_EXPERTS + expert_offset);
        // Set the weight to the sum of all weights divided by routed_scaling_factor
        output_ptr[last_idx] = output_sum / routed_scaling_factor;
      }
    }
  }
}
  /*
  DPCT1065:792: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global
  memory.
  */
  sycl::group_barrier(item_ct1.get_group());
if (thread_row < num_rows) {
  ////////////////////// Rescale Output //////////////////////
  if (thread_group_idx == 0) {
#pragma unroll
    for (int ii = 0; ii < topk; ++ii) {
      int64_t const idx = topk * thread_row + ii;
      output_ptr[idx] = output_ptr[idx] / output_sum;
    }
  }
}
}

//------------------------------------------------------------------------------
// Templated Kernel Version (using compile-time constants)
//------------------------------------------------------------------------------
template <int VPT_, int NUM_EXPERTS_, int THREADS_PER_ROW_, int ROWS_PER_WARP_, int ROWS_PER_CTA_, int WARPS_PER_CTA_>
struct KernelParams {
  static constexpr int VPT = VPT_;
  static constexpr int NUM_EXPERTS = NUM_EXPERTS_;
  static constexpr int THREADS_PER_ROW = THREADS_PER_ROW_;
  static constexpr int ROWS_PER_WARP = ROWS_PER_WARP_;
  static constexpr int ROWS_PER_CTA = ROWS_PER_CTA_;
  static constexpr int WARPS_PER_CTA = WARPS_PER_CTA_;
};

template <
    typename T,
    int VPT,
    int NUM_EXPERTS,
    int THREADS_PER_ROW,
    int ROWS_PER_WARP,
    int ROWS_PER_CTA,
    int WARPS_PER_CTA>
void moe_fused_gate_kernel(
    void* input,
    void* bias,
    float* output_ptr,
    int32_t* indices_ptr,
    int64_t num_rows,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor) {
  KernelParams<VPT, NUM_EXPERTS, THREADS_PER_ROW, ROWS_PER_WARP, ROWS_PER_CTA, WARPS_PER_CTA> params;
  moe_fused_gate_impl<T>(
      input,
      bias,
      output_ptr,
      indices_ptr,
      num_rows,
      topk_group,
      topk,
      num_fused_shared_experts,
      routed_scaling_factor,
      params);
}

// Macro to compute compile-time constants and launch the kernel.
/*
DPCT1049:634: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
info::device::max_work_group_size. Adjust the work-group size if needed.
*/
#define LAUNCH_MOE_GATE_CONFIG(T, EXPERTS, EXPERT_GROUP)                                                            \
  do {                                                                                                              \
    constexpr int VPT = (EXPERTS) / (EXPERT_GROUP);                                                                 \
    /* If EXPERT_GROUP > WARP_SIZE, fall back to 1 row per warp */                                                  \
    constexpr int ROWS_PER_WARP = ((EXPERT_GROUP) <= WARP_SIZE) ? (WARP_SIZE / (EXPERT_GROUP)) : 1;                 \
    constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;                                                     \
    {                                                                                                               \
      auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync}; \
      dpct::has_capability_or_fail(c10::xpu::getCurrentXPUStream().queue().get_device(), {sycl::aspect::fp64});     \
                                                                                                                    \
      stream->submit([&](sycl::handler& cgh) {                                                                      \
        auto input_data_ptr_ct0 = input.data_ptr();                                                                 \
        auto bias_data_ptr_ct1 = bias.data_ptr();                                                                   \
        auto output_data_ptr_float_ct2 = output.data_ptr<float>();                                                  \
        auto indices_data_ptr_int32_t_ct3 = indices.data_ptr<int32_t>();                                            \
        auto num_rows_ct4 = num_rows;                                                                               \
        auto topk_group_ct5 = topk_group;                                                                           \
        auto topk_ct6 = topk;                                                                                       \
        auto num_fused_shared_experts_ct7 = num_fused_shared_experts;                                               \
        auto routed_scaling_factor_ct8 = routed_scaling_factor;                                                     \
                                                                                                                    \
        auto last_event = dpct::get_default_queue().ext_oneapi_get_last_event();                                    \
        [&](auto&& _e) {                                                                                            \
          if constexpr (std::is_same_v<std::remove_reference_t<decltype(last_event)>, sycl::event>)                 \
            cgh.depends_on(_e);                                                                                     \
          else if (_e.has_value())                                                                                  \
            cgh.depends_on(_e.value());                                                                             \
        }(last_event);                                                                                              \
                                                                                                                    \
        cgh.parallel_for(                                                                                           \
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * block_dim, block_dim),                             \
            exp_props,                                                                                              \
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {                                      \
              moe_fused_gate_kernel<T, VPT, (EXPERTS), (EXPERT_GROUP), ROWS_PER_WARP, ROWS_PER_CTA, WARPS_PER_CTA>( \
                  input_data_ptr_ct0,                                                                               \
                  bias_data_ptr_ct1,                                                                                \
                  output_data_ptr_float_ct2,                                                                        \
                  indices_data_ptr_int32_t_ct3,                                                                     \
                  num_rows_ct4,                                                                                     \
                  topk_group_ct5,                                                                                   \
                  topk_ct6,                                                                                         \
                  num_fused_shared_experts_ct7,                                                                     \
                  routed_scaling_factor_ct8);                                                                       \
            });                                                                                                     \
      });                                                                                                           \
    }                                                                                                               \
    dispatched = true;                                                                                              \
  } while (0)

//------------------------------------------------------------------------------
// Dynamic Kernel Version (parameters computed at runtime)
//------------------------------------------------------------------------------
struct KernelParamsDynamic {
  int VPT;
  int NUM_EXPERTS;
  int THREADS_PER_ROW;
  int ROWS_PER_WARP;
  int ROWS_PER_CTA;
  int WARPS_PER_CTA;
};

template <typename T>
void moe_fused_gate_kernel_dynamic(
    void* input,
    void* bias,
    float* output_ptr,
    int32_t* indices_ptr,
    int64_t num_rows,
    int64_t num_experts,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor) {
  KernelParamsDynamic params;
  params.NUM_EXPERTS = num_experts;             // e.g, for deepseek v3, this is 256
  params.VPT = num_experts / num_expert_group;  // e.g., for deepseek v3, this is 256 / 8 = 32
  params.THREADS_PER_ROW = num_expert_group;    // fixed as num_expert_group, e.g., for deepseek v3, this is 8
  params.WARPS_PER_CTA = WARPS_PER_CTA;         // fixed as 6
  params.ROWS_PER_WARP = std::max<int64_t>(1, WARP_SIZE / num_expert_group);  // WARP_SIZE is fixed as 32
  params.ROWS_PER_CTA = params.WARPS_PER_CTA * params.ROWS_PER_WARP;

  moe_fused_gate_impl<T>(
      input,
      bias,
      output_ptr,
      indices_ptr,
      num_rows,
      topk_group,
      topk,
      num_fused_shared_experts,
      routed_scaling_factor,
      params);
}

//------------------------------------------------------------------------------
// Host Launcher Function
//------------------------------------------------------------------------------
std::vector<at::Tensor> moe_fused_gate(
    at::Tensor& input,
    at::Tensor& bias,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor) {
  int64_t num_rows = input.size(0);
  int32_t num_experts = input.size(1);
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kXPU);
  auto output = torch::empty({num_rows, topk}, options);
  auto indices = torch::empty({num_rows, topk}, options.dtype(torch::kInt32));

  // Compute grid dimensions based on runtime value for num_expert_group.
  int64_t rows_per_warp = std::max<int64_t>(1, WARP_SIZE / num_expert_group);
  int64_t num_warps = (num_rows + rows_per_warp - 1) / rows_per_warp;
  int64_t num_blocks = (num_warps + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
  const dpct::queue_ptr stream = &c10::xpu::getCurrentXPUStream().queue();
  dpct::dim3 block_dim(WARP_SIZE, WARPS_PER_CTA);

  // Check 1: Ensure that num_experts is a power of 2.
  TORCH_CHECK((num_experts & (num_experts - 1)) == 0, "num_experts must be a power of 2, but got ", num_experts);

  // Check 2: Ensure that num_experts is divisible by num_expert_group. (this also means num_expert_group is power of 2)
  TORCH_CHECK(
      num_experts % num_expert_group == 0,
      "num_experts must be divisible by num_expert_group, but got ",
      num_experts,
      " / ",
      num_expert_group);

  int computed_vpt = num_experts / num_expert_group;
  // Check 3: Ensure that num_experts/num_expert_group does not exceed MAX_VPT=32. Maximum VPT indicate max value per
  // threads we can process.
  TORCH_CHECK(
      computed_vpt <= MAX_VPT,
      "Per group experts: num_experts / num_expert_group = (",
      computed_vpt,
      ") exceeds the maximum supported (",
      MAX_VPT,
      ")");

  // Dispatch to templated kernel for known compile-time configurations.
  // We currently only support for:
  //   Case 1: 256 experts, with 8 or 16 groups.
  //   Case 2: 128 experts, with 4 or 8 groups.
  //   Case 3: other cases, require 8 <= num_experts / num_expert_group <= 32
  bool dispatched = false;
  switch (num_experts) {
    case 256:
      if (num_expert_group == 8)
        // This is deepseek v3 case. Here VPT = 256/8 = 32, ROWS_PER_WARP = 32/8 = 4, ROWS_PER_CTA = 6 * 4 = 24.
        if (input.scalar_type() == at::kBFloat16) {
          LAUNCH_MOE_GATE_CONFIG(bfloat16_t, 256, 8);
        } else if (input.scalar_type() == at::kHalf) {
          LAUNCH_MOE_GATE_CONFIG(float16_t, 256, 8);
        } else if (input.scalar_type() == at::kFloat) {
          LAUNCH_MOE_GATE_CONFIG(float32_t, 256, 8);
        } else if (num_expert_group == 16)
          // Here VPT = 256/16 = 16, ROWS_PER_WARP = 32/16 = 2, ROWS_PER_CTA = 6 * 2 = 12.
          if (input.scalar_type() == at::kBFloat16) {
            LAUNCH_MOE_GATE_CONFIG(bfloat16_t, 256, 16);
          } else if (input.scalar_type() == at::kHalf) {
            LAUNCH_MOE_GATE_CONFIG(float16_t, 256, 16);
          } else if (input.scalar_type() == at::kFloat) {
            LAUNCH_MOE_GATE_CONFIG(float32_t, 256, 16);
          }
      break;
    case 128:
      if (num_expert_group == 4)
        // VPT = 128/4 = 32, ROWS_PER_WARP = 32/16 = 2, ROWS_PER_CTA = 6 * 2 = 12.
        if (input.scalar_type() == at::kBFloat16) {
          LAUNCH_MOE_GATE_CONFIG(bfloat16_t, 128, 4);
        } else if (input.scalar_type() == at::kHalf) {
          LAUNCH_MOE_GATE_CONFIG(float16_t, 128, 4);
        } else if (input.scalar_type() == at::kFloat) {
          LAUNCH_MOE_GATE_CONFIG(float32_t, 128, 4);
        } else if (num_expert_group == 8)
          // VPT = 128/8 = 16, ROWS_PER_WARP = 32/8 = 4, ROWS_PER_CTA = 6 * 4 = 24.
          if (input.scalar_type() == at::kBFloat16) {
            LAUNCH_MOE_GATE_CONFIG(bfloat16_t, 128, 8);
          } else if (input.scalar_type() == at::kHalf) {
            LAUNCH_MOE_GATE_CONFIG(float16_t, 128, 8);
          } else if (input.scalar_type() == at::kFloat) {
            LAUNCH_MOE_GATE_CONFIG(float32_t, 128, 8);
          }
      break;
    default:
      break;
  }
  if (!dispatched) {
    // Fallback to the dynamic kernel if none of the supported combinations match.
    // currently only support num_experts / num_expert_group <= 32 for dynamic kernels
    if (input.scalar_type() == at::kBFloat16) {
      /*
      DPCT1049:635: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
      info::device::max_work_group_size. Adjust the work-group size if needed.
      */
      auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
      dpct::has_capability_or_fail(c10::xpu::getCurrentXPUStream().queue().get_device(), {sycl::aspect::fp64});

      stream->submit([&](sycl::handler& cgh) {
        auto input_data_ptr_ct0 = input.data_ptr();
        auto bias_data_ptr_ct1 = bias.data_ptr();
        auto output_data_ptr_float_ct2 = output.data_ptr<float>();
        auto indices_data_ptr_int32_t_ct3 = indices.data_ptr<int32_t>();

        auto last_event = dpct::get_default_queue().ext_oneapi_get_last_event();
        [&](auto&& _e) {
          if constexpr (std::is_same_v<std::remove_reference_t<decltype(last_event)>, sycl::event>)
            cgh.depends_on(_e);
          else if (_e.has_value())
            cgh.depends_on(_e.value());
        }(last_event);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * block_dim, block_dim),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
              moe_fused_gate_kernel_dynamic<bfloat16_t>(
                  input_data_ptr_ct0,
                  bias_data_ptr_ct1,
                  output_data_ptr_float_ct2,
                  indices_data_ptr_int32_t_ct3,
                  num_rows,
                  num_experts,
                  num_expert_group,
                  topk_group,
                  topk,
                  num_fused_shared_experts,
                  routed_scaling_factor);
            });
      });
    } else if (input.scalar_type() == at::kHalf) {
      /*
      DPCT1049:636: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
      info::device::max_work_group_size. Adjust the work-group size if needed.
      */
      auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
      dpct::has_capability_or_fail(c10::xpu::getCurrentXPUStream().queue().get_device(), {sycl::aspect::fp64});

      stream->submit([&](sycl::handler& cgh) {
        auto input_data_ptr_ct0 = input.data_ptr();
        auto bias_data_ptr_ct1 = bias.data_ptr();
        auto output_data_ptr_float_ct2 = output.data_ptr<float>();
        auto indices_data_ptr_int32_t_ct3 = indices.data_ptr<int32_t>();

        auto last_event = dpct::get_default_queue().ext_oneapi_get_last_event();
        [&](auto&& _e) {
          if constexpr (std::is_same_v<std::remove_reference_t<decltype(last_event)>, sycl::event>)
            cgh.depends_on(_e);
          else if (_e.has_value())
            cgh.depends_on(_e.value());
        }(last_event);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * block_dim, block_dim),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
              moe_fused_gate_kernel_dynamic<float16_t>(
                  input_data_ptr_ct0,
                  bias_data_ptr_ct1,
                  output_data_ptr_float_ct2,
                  indices_data_ptr_int32_t_ct3,
                  num_rows,
                  num_experts,
                  num_expert_group,
                  topk_group,
                  topk,
                  num_fused_shared_experts,
                  routed_scaling_factor);
            });
      });
    } else if (input.scalar_type() == at::kFloat) {
      /*
      DPCT1049:637: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
      info::device::max_work_group_size. Adjust the work-group size if needed.
      */
      auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
      dpct::has_capability_or_fail(c10::xpu::getCurrentXPUStream().queue().get_device(), {sycl::aspect::fp64});

      stream->submit([&](sycl::handler& cgh) {
        auto input_data_ptr_ct0 = input.data_ptr();
        auto bias_data_ptr_ct1 = bias.data_ptr();
        auto output_data_ptr_float_ct2 = output.data_ptr<float>();
        auto indices_data_ptr_int32_t_ct3 = indices.data_ptr<int32_t>();

        auto last_event = dpct::get_default_queue().ext_oneapi_get_last_event();
        [&](auto&& _e) {
          if constexpr (std::is_same_v<std::remove_reference_t<decltype(last_event)>, sycl::event>)
            cgh.depends_on(_e);
          else if (_e.has_value())
            cgh.depends_on(_e.value());
        }(last_event);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * block_dim, block_dim),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
              moe_fused_gate_kernel_dynamic<float32_t>(
                  input_data_ptr_ct0,
                  bias_data_ptr_ct1,
                  output_data_ptr_float_ct2,
                  indices_data_ptr_int32_t_ct3,
                  num_rows,
                  num_experts,
                  num_expert_group,
                  topk_group,
                  topk,
                  num_fused_shared_experts,
                  routed_scaling_factor);
            });
      });
    } else {
      TORCH_CHECK(false, "Unsupported data type for moe_fused_gate");
    }
  }
  return {output, indices};
}
