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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/XLANativeFunctions.h"

namespace at_npu {
namespace native {

bool XLANativeFunctions::equal(const at::Tensor& self, const at::Tensor& other) {
  // check the shape of self and other
  if(self.sizes() != other.sizes()) {
    return false;
  }

  TORCH_CHECK(
      self.scalar_type() == other.scalar_type(),
      "Expected object of scalar type ",
      self.scalar_type(),
      ", but got ",
      other.scalar_type(),
      " for argument #2 'other' in call to equal_npu");

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      {1},
      self.options().dtype(at::kBool),
      ACL_FORMAT_ND);

  // calculate the output result of the NPU
  OpCommand cmd;
  cmd.Name("TensorEqual")
      .Input(self)
      .Input(other)
      .Output(result)
      .Run();

  return result.item().to<bool>();
}
} // namespace native
} // namespace at_npu