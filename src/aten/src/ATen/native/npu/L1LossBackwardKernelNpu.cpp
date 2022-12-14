// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

Tensor& l1_loss_backward_out_npu(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  Tensor gradOutputBroadcast = grad_output;
  Tensor targetBroadcast = target; 
  if (grad_output.sizes() != self.sizes()) {
    gradOutputBroadcast = broadcast_npu(grad_output, self.sizes());
  }
  if (target.sizes() != self.sizes()) {
    targetBroadcast = broadcast_npu(target, self.sizes());
  }
  
  auto reductionStr = CalcuOpUtil::get_reduction_str(reduction);

  OpCommand cmd;
  cmd.Name("L1LossGrad")
      .Input(gradOutputBroadcast)
      .Input(self)
      .Input(targetBroadcast)
      .Attr("reduction", reductionStr)
      .Output(grad_input)
      .Run();

  return grad_input;
}

Tensor l1_loss_backward_npu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  // construct the output tensor of the NPU
  Tensor grad_input =  OpPreparation::ApplyTensor(self);

  // calculate the output result of the NPU
  l1_loss_backward_out_npu(grad_input, grad_output, self, target, reduction);

  return grad_input;
}
} // namespace native
} // namespace at