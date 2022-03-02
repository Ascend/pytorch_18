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

Tensor& soft_margin_loss_backward_out_npu(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    Tensor& grad_input) {
  string reductionStr(CalcuOpUtil::get_reduction_str(reduction));

  OpPreparation::CheckMemory({grad_output, input, target}, {grad_input});
  OpCommand cmd;
  cmd.Name("SoftMarginLossGrad")
      .Input(input)
      .Input(target)
      .Input(grad_output)
      .Output(grad_input)
      .Attr("reduction", reductionStr)
      .Run();
  return grad_input;
}

Tensor soft_margin_loss_backward_npu(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    int64_t reduction) {
  Tensor grad_input = OpPreparation::ApplyTensor(input);

  soft_margin_loss_backward_out_npu(
      grad_output, input, target, reduction, grad_input);
  return grad_input;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("soft_margin_loss_backward", TORCH_FN(soft_margin_loss_backward_npu));
  m.impl("soft_margin_loss_backward.grad_input", TORCH_FN(soft_margin_loss_backward_out_npu));
}
} // namespace native
} // namespace at