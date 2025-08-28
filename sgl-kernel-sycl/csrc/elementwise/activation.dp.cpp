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
#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <c10/xpu/XPUStream.h>
#include <flashinfer/activation.dp.hpp>

#include "pytorch_extension_utils.h"
#include <cmath>

using namespace flashinfer;

#include <iostream>

void silu_and_mul(at::Tensor& out, at::Tensor& input, int64_t cuda_stream) {
      
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dpct::dim3 grid(num_tokens);

  //dpct::queue_ptr stream = dpct::int_as_queue_ptr(cuda_stream);
  dpct::queue_ptr stream = &c10::xpu::getCurrentXPUStream().queue();
  /*
  DPCT1038:28: When the kernel function name is used as a macro argument, the migration result may be incorrect. You
  need to verify the definition of the macro.
  */
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    dpct::dim3 block(std::min(d / vec_size, 1024U));
    /*
    DPCT1049:29: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
      {
         //auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
         //dpct::has_capability_or_fail(c10::xpu::getCurrentXPUStream().queue().get_device(), {sycl::aspect::fp16});

         stream->submit([&](sycl::handler& cgh) {
            auto out_data_ptr_ct0 = static_cast<c_type*>(out.data_ptr());
            auto input_data_ptr_ct1 = static_cast<c_type*>(input.data_ptr());
            
            auto last_event = dpct::get_default_queue().ext_oneapi_get_last_event();
            [&](auto&& _e) {
               if constexpr (std::is_same_v<std::remove_reference_t<decltype(last_event)>, sycl::event>)
                  cgh.depends_on(_e);
               else if (_e.has_value())
                  cgh.depends_on(_e.value());
            }(last_event);

            cgh.parallel_for(sycl::nd_range<3>(grid * block, block),  [=](sycl::nd_item<3> item_ct1) {
               //flashinfer::activation::act_and_mul_kernel<c_type, silu>(out_data_ptr_ct0, input_data_ptr_ct1, d);
               flashinfer::activation::act_and_mul_kernel<c_type, 0>(out_data_ptr_ct0, input_data_ptr_ct1, d);
            });
         });
      }

    return true;
  });
  stream->wait();
}

void gelu_tanh_and_mul(at::Tensor& out, at::Tensor& input, int64_t cuda_stream) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dpct::dim3 grid(num_tokens);

  //dpct::queue_ptr stream = dpct::int_as_queue_ptr(cuda_stream);
  dpct::queue_ptr stream = &c10::xpu::getCurrentXPUStream().queue();
  /*
  DPCT1038:30: When the kernel function name is used as a macro argument, the migration result may be incorrect. You
  need to verify the definition of the macro.
  */
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    dpct::dim3 block(std::min(d / vec_size, 1024U));
    /*
    DPCT1049:31: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
      {
         //auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
         dpct::has_capability_or_fail(c10::xpu::getCurrentXPUStream().queue().get_device(), {sycl::aspect::fp16});

         stream->submit([&](sycl::handler& cgh) {
            auto out_data_ptr_ct0 = static_cast<c_type*>(out.data_ptr());
            auto input_data_ptr_ct1 = static_cast<c_type*>(input.data_ptr());

            auto last_event = dpct::get_default_queue().ext_oneapi_get_last_event();
            [&](auto&& _e) {
               if constexpr (std::is_same_v<std::remove_reference_t<decltype(last_event)>, sycl::event>)
                  cgh.depends_on(_e);
               else if (_e.has_value())
                  cgh.depends_on(_e.value());
            }(last_event);

            cgh.parallel_for(sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1) {
               //flashinfer::activation::act_and_mul_kernel<c_type, gelu_tanh>(out_data_ptr_ct0, input_data_ptr_ct1, d);
               flashinfer::activation::act_and_mul_kernel<c_type, 1>(out_data_ptr_ct0, input_data_ptr_ct1, d);
            });
         });
      }

    return true;
  });
  stream->wait();
}

void gelu_and_mul(at::Tensor& out, at::Tensor& input, int64_t cuda_stream) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dpct::dim3 grid(num_tokens);

  //dpct::queue_ptr stream = dpct::int_as_queue_ptr(cuda_stream);
  dpct::queue_ptr stream = &c10::xpu::getCurrentXPUStream().queue();
  /*
  DPCT1038:32: When the kernel function name is used as a macro argument, the migration result may be incorrect. You
  need to verify the definition of the macro.
  */
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    dpct::dim3 block(std::min(d / vec_size, 1024U));
    /*
    DPCT1049:33: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
      {
        // auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
         dpct::has_capability_or_fail(c10::xpu::getCurrentXPUStream().queue().get_device(), {sycl::aspect::fp16});

         stream->submit([&](sycl::handler& cgh) {
            auto out_data_ptr_ct0 = static_cast<c_type*>(out.data_ptr());
            auto input_data_ptr_ct1 = static_cast<c_type*>(input.data_ptr());

            auto last_event = dpct::get_default_queue().ext_oneapi_get_last_event();
            [&](auto&& _e) {
               if constexpr (std::is_same_v<std::remove_reference_t<decltype(last_event)>, sycl::event>)
                  cgh.depends_on(_e);
               else if (_e.has_value())
                  cgh.depends_on(_e.value());
            }(last_event);

            cgh.parallel_for(sycl::nd_range<3>(grid * block, block),  [=](sycl::nd_item<3> item_ct1) {
               //flashinfer::activation::act_and_mul_kernel<c_type, gelu>(out_data_ptr_ct0, input_data_ptr_ct1, d);
               flashinfer::activation::act_and_mul_kernel<c_type, 2>(out_data_ptr_ct0, input_data_ptr_ct1, d);
            });
         });
      }

    return true;
  });
  stream->wait();
}
