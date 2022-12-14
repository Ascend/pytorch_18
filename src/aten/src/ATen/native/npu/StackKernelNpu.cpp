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

SmallVector<int64_t, SIZE> stack_npu_output_size(
    TensorList tensors,
    int64_t dim) {
  dim = make_wrap_dim(dim, tensors[0].dim() + 1);
  SmallVector<int64_t, SIZE> shape;
  for (int i = 0; i < dim; i++) {
    shape.emplace_back(tensors[0].size(i));
  }
  shape.emplace_back(tensors.size());
  for (int i = dim; i < tensors[0].dim(); i++) {
    shape.emplace_back(tensors[0].size(i));
  }
  return shape;
}

Tensor& stack_out_npu_nocheck(Tensor& result, TensorList tensors, int64_t dim) {
  // constructs the input and output NPUTensorDesc
  auto inputTensors = CalcuOpUtil::ConvertTensorListToSmallVector(tensors);

  OpCommand cmd;
  cmd.Name("Pack");
  for (int i = 0; i < inputTensors.size(); i++) {
    string inputName = "x" + to_string(i);
    cmd.Input(inputTensors[i],inputName);
  }
  cmd.Output(result)
    .Attr("N", (int64_t)tensors.size())
    .Attr("axis", dim)
    .Run();

  return result;
}

Tensor& stack_out_npu(Tensor& result, TensorList tensors, int64_t dim) {
  auto outputSize = stack_npu_output_size(tensors, dim);
  OpPreparation::CheckOut(
    {tensors[0]}, 
    result, 
    ACL_FORMAT_ND, 
    tensors[0].scalar_type(), 
    outputSize); 
  stack_out_npu_nocheck(result, tensors, dim); 
  return result;
}

Tensor stack_npu(TensorList tensors, int64_t dim) {
  // calculate the output size
  auto outputSize = stack_npu_output_size(tensors, dim);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize,
      tensors[0].options(),
      CalcuOpUtil::get_tensor_npu_format(tensors[0]));

  // calculate the output result of the NPU
  stack_out_npu_nocheck(result, tensors, dim);

  return result;
}

} // namespace native
} // namespace at