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

Tensor& sqrt_out_npu_nocheck(const Tensor& self, Tensor& result) {
  OpCommand cmd;
  cmd.Name("Sqrt")
    .Input(self)
    .Output(result)
    .Run();

  return result;
}

Tensor& sqrt_out_npu(const Tensor& self, Tensor& result) {
  OpPreparation::CheckOut({self}, result, self);

  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {result})
      .Func([&self](Tensor& result){sqrt_out_npu_nocheck(self, result);})
      .Call(result);
}

Tensor sqrt_npu(const Tensor& self) {
  Tensor result = OpPreparation::ApplyTensor(self);

  sqrt_out_npu_nocheck(self, result);
  return result;
}

Tensor& sqrt_npu_(Tensor& self) {
  sqrt_out_npu(self, self);

  return self;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("sqrt", TORCH_FN(sqrt_npu));
  m.impl("sqrt_", TORCH_FN(sqrt_npu_));
  m.impl("sqrt.out", TORCH_FN(sqrt_out_npu));
}
} // namespace native
} // namespace at