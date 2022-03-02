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

Tensor& inverse_out_npu(
    Tensor& result,
    const Tensor& self) {
  Tensor selfCast = self;
  if(self.scalar_type() == at::kHalf) {
    selfCast = self.to(at::kFloat);
  }

  OpCommand cmd;
  cmd.Name("MatrixInverse")
      .Input(selfCast)
      .Output(result)
      .Attr("adjoint", false)
      .Run();
  if (result.scalar_type() != self.scalar_type()) {
    result = result.to(self.scalar_type());
  }
  return result;
}

Tensor inverse_npu(const Tensor& self) {
  Tensor selfCast = self;
  if(self.scalar_type() == at::kHalf) {
    selfCast = self.to(at::kFloat);
  }
  Tensor result = OpPreparation::ApplyTensor(selfCast);

  inverse_out_npu(result, selfCast);

  if (result.scalar_type() != self.scalar_type()) {
    result = result.to(self.scalar_type());
  }
  return result;
}

} // namespace native
} // namespace at