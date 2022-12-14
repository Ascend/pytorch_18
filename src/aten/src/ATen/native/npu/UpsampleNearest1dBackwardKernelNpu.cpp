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

Tensor& upsample_nearest1d_backward_out_npu(
    Tensor& y,
    const Tensor& grads,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales) {
  OpCommand cmd;
  cmd.Name("UpsampleNearest1dGrad")
      .Input(grads)
      .Output(y)
      .Attr("output_size", output_size)
      .Attr("input_size", input_size);
      if (scales.has_value()) {
        cmd.Attr("scales", static_cast<float>(scales.value()));
      }
      cmd.Run();

   return y;
}

Tensor upsample_nearest1d_backward_npu(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales) {
  Tensor grads = grad_output;
  if (grad_output.scalar_type() != at::ScalarType::Float) {
    grads = grad_output.to(at::kFloat);
  }

  Tensor grad_input = OpPreparation::ApplyTensor(input_size, grads.options(), grad_output);

  upsample_nearest1d_backward_out_npu(
      grad_input, grads, output_size, input_size, scales);
  return grad_input;
}

} // namespace native
} // namespace at
