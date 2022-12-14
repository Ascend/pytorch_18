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

Tensor& exp_out_npu_nocheck(Tensor& result, const Tensor& self) {
  OpCommand cmd;
  cmd.Name("Exp")
    .Input(self)
    .Output(result)
    .Run();

  return result;
}

Tensor& exp_out_npu(Tensor& result, const Tensor& self) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);

  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {result})
   .Func([&self](Tensor& result){exp_out_npu_nocheck(result, self);})
   .Call(result);
}

Tensor& exp_npu_(Tensor& self) {
  exp_out_npu(self, self);

  return self;
}

Tensor exp_npu(const Tensor& self) {
  Tensor result = OpPreparation::ApplyTensor(self);
  exp_out_npu_nocheck(result, self);
  return result;
}

} // namespace native
} // namespace at
