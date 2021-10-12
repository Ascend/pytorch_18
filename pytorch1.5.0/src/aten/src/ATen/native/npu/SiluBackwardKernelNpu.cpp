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

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& silu_backward_out_npu_nocheck(
    Tensor& result,
    const Tensor& grad_output,
    const Tensor& x0, 
    const Tensor& x1) {

  OpCommand cmd;
  cmd.Name("SwishGrad")
    .Input(grad_output)
    .Input(x0)
    .Input(x1)
    .Output(result)
    .Run();

  return result;
}

Tensor silu_backward_npu(const Tensor& grad_output, const Tensor& x0, const Tensor& x1) {
  // construct the output tensor of the NPU
  Tensor grad_input = OpPreparation::ApplyTensor(grad_output);

  // calculate the output result of the NPU
  silu_backward_out_npu_nocheck(grad_input, grad_output, x0, x1);

  return grad_input;
}

} // namespace native
} // namespace at