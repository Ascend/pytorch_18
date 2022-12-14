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

SmallVector<int64_t, SIZE> ger_npu_output_size(
    const Tensor& self,
    const Tensor& vec2) {
  int64_t outputsize_0 = self.size(0);
  int64_t outputsize_1 = vec2.size(0);
  SmallVector<int64_t, SIZE> outputsize = {outputsize_0, outputsize_1};

  return outputsize;
}

Tensor& ger_out_npu_nocheck(Tensor& result, const Tensor& self , const Tensor& vec2) {
  OpCommand cmd;
  cmd.Name("Ger")
      .Input(self)
      .Input(vec2)
      .Output(result)
      .Run();

  return result;
}

Tensor& ger_out_npu(Tensor& result, const Tensor& self , const Tensor& vec2) {
  // check shape
  TORCH_CHECK(
      self.dim() == 1, "Input1 must have only1 dims."); 
  TORCH_CHECK(
      vec2.dim() == 1, "Input2 must have only1 dims.");

  // calculate the output size
  auto outputSize = ger_npu_output_size(self, vec2);

  OpPreparation::CheckOut(
      {self},
      result,
      self,
      outputSize);

  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self, vec2}, {result})
      .Func([&self, &vec2](Tensor& result){ger_out_npu_nocheck(result, self, vec2);})
      .Call(result);
}

Tensor ger_npu(const Tensor& self, const Tensor& vec2) {
  // check shape
  TORCH_CHECK(
      self.dim() == 1, "Input1 must have only1 dims."); 
  TORCH_CHECK(
      vec2.dim() == 1, "Input2 must have only1 dims.");

  // calculate the output size
  auto outputSize = ger_npu_output_size(self, vec2);

  // construct the output tensor of the NPU 
  Tensor result = OpPreparation::ApplyTensor(self, outputSize);

  // calculate the output result of the NPU
  ger_out_npu_nocheck(result, self, vec2);

  return result;
}

} // namespace native
} // namespace at
