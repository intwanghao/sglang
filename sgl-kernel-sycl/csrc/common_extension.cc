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
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>

#include "sgl_kernel_ops.h"

PYBIND11_MODULE(common_ops, m) {
  m.def("moe_align_block_size", &moe_align_block_size);
  m.def("topk_softmax", &topk_softmax);
  m.def("moe_fused_gate", &moe_fused_gate);
  m.def("ep_moe_pre_reorder", &ep_moe_pre_reorder);
  m.def("ep_moe_silu_and_mul", &ep_moe_silu_and_mul);
  m.def("ep_moe_post_reorder", &ep_moe_post_reorder);
  m.def("gptq_marlin_repack", &marlin_moe_wna16::gptq_marlin_repack);
  m.def("awq_marlin_repack", &marlin_moe_wna16::awq_marlin_repack);
  m.def("moe_wna16_marlin_gemm", &moe_wna16_marlin_gemm);
  m.def("vec_add_usm", &vec_add_usm);
  m.def("silu_and_mul", &silu_and_mul);
  m.def("apply_rope_pos_ids_cos_sin_cache", &apply_rope_pos_ids_cos_sin_cache);
}

TORCH_LIBRARY_FRAGMENT(sgl_kernel_sycl, m) {

  /*
   * From csrc/moe
   */
  m.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts, int block_size, Tensor! sorted_token_ids, Tensor! "
      "experts_ids, Tensor! num_tokens_post_pad, Tensor! token_cnts_buffer, Tensor! cumsum_buffer, bool "
      "pad_sorted_token_ids) -> ()");
  m.impl("moe_align_block_size", torch::kXPU, &moe_align_block_size);

  m.def("topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor gating_output, bool renormalize) -> ()");
  m.impl("topk_softmax", torch::kXPU, &topk_softmax);

  m.def(
      "moe_fused_gate(Tensor input, Tensor bias, int num_expert_group, int topk_group, int topk, int "
      "num_fused_shared_experts, float routed_scaling_factor) -> "
      "(Tensor[])");
  m.impl("moe_fused_gate", torch::kXPU, &moe_fused_gate);
  m.def(
      "ep_moe_pre_reorder(Tensor input, Tensor gateup_input, Tensor src2dst, Tensor topk_ids, Tensor "
      "a1_scales, int start_expert_id, int end_expert_id, int topk, bool use_per_token_if_dynamic) -> ()");
  m.impl("ep_moe_pre_reorder", torch::kXPU, &ep_moe_pre_reorder);
  m.def(
      "ep_moe_silu_and_mul(Tensor gateup_output, Tensor down_input, Tensor reorder_topk_ids, Tensor scales, int "
      "start_expert_id, int end_expert_id) -> ()");
  m.impl("ep_moe_silu_and_mul", torch::kXPU, &ep_moe_silu_and_mul);
  m.def(
      "ep_moe_post_reorder(Tensor down_output, Tensor output, Tensor src2dst, Tensor topk_ids, Tensor "
      "topk_weights, int start_expert_id, int end_expert_id, int topk) -> ()");
  m.impl("ep_moe_post_reorder", torch::kXPU, &ep_moe_post_reorder);
  m.def(
      "fp8_blockwise_scaled_grouped_mm(Tensor output, Tensor a_ptrs, Tensor b_ptrs, Tensor out_ptrs, Tensor "
      "a_scales_ptrs, Tensor b_scales_ptrs, Tensor a, Tensor b, Tensor scales_a, Tensor scales_b, Tensor "
      "stride_a, Tensor stride_b, Tensor stride_c, Tensor layout_sfa, Tensor layout_sfb, Tensor problem_sizes, Tensor "
      "expert_offsets, Tensor workspace) -> ()");
  m.impl("fp8_blockwise_scaled_grouped_mm", torch::kXPU, &fp8_blockwise_scaled_grouped_mm);
  m.def(
      "prepare_moe_input(Tensor topk_ids, Tensor expert_offsets, Tensor? blockscale_offsets, Tensor problem_sizes1,"
      " Tensor problem_sizes2, Tensor input_permutation, Tensor output_permutation, int num_experts, int n, int k) -> "
      "()");
  m.impl("prepare_moe_input", torch::kXPU, &prepare_moe_input);

  m.def("shuffle_rows(Tensor input, Tensor dst2src_map, Tensor output) -> ()");
  m.impl("shuffle_rows", torch::kXPU, &shuffle_rows);
  m.def("apply_shuffle_mul_sum(Tensor input, Tensor output, Tensor permutation, Tensor? factors) -> ()");
  m.impl("apply_shuffle_mul_sum", torch::kXPU, &apply_shuffle_mul_sum);

  m.def("gptq_marlin_repack(Tensor! b_q_weight, Tensor! perm, int size_k, int size_n, int num_bits) -> Tensor");
  m.impl("gptq_marlin_repack", torch::kXPU, &marlin_moe_wna16::gptq_marlin_repack);

  m.def("awq_marlin_repack(Tensor! b_q_weight, int size_k, int size_n, int num_bits) -> Tensor");
  m.impl("awq_marlin_repack", torch::kXPU, &marlin_moe_wna16::awq_marlin_repack);

  /*
   * From csrc/moe/cutlass_moe/w4a8
   */
  m.def(
      "get_cutlass_w4a8_moe_mm_data(Tensor topk_ids, Tensor! expert_offsets, "
      "                        Tensor! problem_sizes1, Tensor! problem_sizes2, "
      "                        Tensor! input_permutation, "
      "                        Tensor! output_permutation, int num_experts, "
      "                        int n, int k) -> ()");
  m.impl("get_cutlass_w4a8_moe_mm_data", torch::kXPU, &get_cutlass_w4a8_moe_mm_data);

  m.def(
      "cutlass_w4a8_moe_mm(Tensor! d, Tensor a, Tensor b, "
      "               Tensor a_scales, Tensor b_scales, Tensor expert_offsets, "
      "               Tensor problem_sizes, Tensor a_strides, "
      "               Tensor b_strides, Tensor d_strides, Tensor s_strides,"
      "               int chunk_size, int topk) -> ()");
  m.impl("cutlass_w4a8_moe_mm", torch::kXPU, &cutlass_w4a8_moe_mm);

  m.def(
      "moe_wna16_marlin_gemm(Tensor! a, Tensor? c_or_none,"
      "Tensor! b_q_weight, Tensor! b_scales, Tensor? b_zeros_or_none,"
      "Tensor? g_idx_or_none, Tensor? perm_or_none, Tensor! workspace,"
      "Tensor sorted_token_ids,"
      "Tensor! expert_ids, Tensor! num_tokens_past_padded,"
      "Tensor! topk_weights, int moe_block_size, int top_k, "
      "bool mul_topk_weights, bool is_ep, int b_q_type_id,"
      "int size_m, int size_n, int size_k,"
      "bool is_full_k, bool use_atomic_add,"
      "bool use_fp32_reduce, bool is_zp_float) -> Tensor");
  m.impl("moe_wna16_marlin_gemm", torch::kXPU, &moe_wna16_marlin_gemm);
}

//REGISTER_EXTENSION(common_ops)
