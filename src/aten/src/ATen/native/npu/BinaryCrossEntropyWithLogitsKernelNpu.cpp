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

Tensor binary_cross_entropy_with_logits_npu(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    const Tensor& pos_weight,
    int64_t reduction) {
  // calculate the output size
  IntArrayRef outputSize;
  int64_t resultformat = CalcuOpUtil::get_tensor_npu_format(self);

  if (reduction == Reduction::None) {
    outputSize = input_same_output_size(self);
  } else {
    outputSize = ArrayRef<int64_t>();
    resultformat = ACL_FORMAT_ND;
  }

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize,
      target.options(),
      resultformat);

  // construct the input tensor of the NPU
  Tensor weightTensor;
  if (weight.defined()) {
    weightTensor = NpuUtils::format_contiguous(weight);
  } else {
    weightTensor = at::ones(target.sizes(), target.options());
  }

  Tensor posWeightTensor;
  if (pos_weight.defined()) {
    posWeightTensor = NpuUtils::format_contiguous(pos_weight);
  } else {
    posWeightTensor = at::ones(target.sizes(), target.options());
  }

  // constructs the attr of the NPUAttrDesc
  string reductionStr;
  if (reduction == Reduction::None) {
    reductionStr = "none";
  } else if (reduction == Reduction::Mean) {
    reductionStr = "mean";
  } else if (reduction == Reduction::Sum) {
    reductionStr = "sum";
  }

  // executing the NPU operator
  OpCommand cmd;
  cmd.Name("SigmoidCrossEntropyWithLogitsV2")
      .Input(self.to(target.dtype()))
      .Input(target)
      .Input(weightTensor)
      .Input(posWeightTensor)
      .Output(result)
      .Attr("reduction", reductionStr)
      .Run();

  return result;
}
} // namespace native
} // namespace at
