// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION. 
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor threshold_backward_out_npu(
    Tensor& result,
    const Tensor& grad_output,
    const Tensor& self,
    Scalar threshold) {
  OpCommand cmd;

  // The performance of the ReluGrad operator is better than that of ThresholdGradV2D. 
  // However, ReluGrad does not support the scenario where threshold is not 0.
  if (CalcuOpUtil::get_scalar_float_value(threshold) != 0) {
    cmd.Name("ThresholdGradV2D")
          .Input(grad_output)
          .Input(self)
          .Output(result)
          .Attr("threshold", threshold)
          .Run();
  } else {
    cmd.Name("ReluGrad")
          .Input(grad_output)
          .Input(self)
          .Output(result)
          .Run();
  }
  
  return result;
}

Tensor threshold_backward_npu(
    const Tensor& grad_output,
    const Tensor& self,
    Scalar threshold) {
  // calculate the output size
  auto outputSize = input_same_output_size(self);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  // use 5HD in Relu
  if ((grad_output.storage().unsafeGetStorageImpl()->npu_desc_.npu_format_ ==
       ACL_FORMAT_NCHW) &&
      (self.storage().unsafeGetStorageImpl()->npu_desc_.npu_format_ ==
       ACL_FORMAT_NC1HWC0)) {
    Tensor grad_output_5HD =
        at::npu_format_cast(grad_output, ACL_FORMAT_NC1HWC0);
    threshold_backward_out_npu(result, grad_output_5HD, self, threshold);
    return result;
  } else {
    threshold_backward_out_npu(result, grad_output, self, threshold);
    return result;
  }

}

} // namespace native
} // namespace at