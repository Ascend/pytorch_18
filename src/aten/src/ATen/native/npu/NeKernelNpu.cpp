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

Tensor& ne_out_npu_nocheck(Tensor& result, const Tensor& self, const Tensor& other) {
  Tensor selfCast = self;
  Tensor otherCast = other;
  if(self.dtype() == ScalarType::Int || other.dtype() == ScalarType::Int){
    selfCast = self.to(ScalarType::Float);
    otherCast = other.to(ScalarType::Float);
  }
  auto unified_result = OpPreparation::comparison_op_check(result, selfCast, otherCast, true);
  if(self.scalar_type() == at::kLong) {
    TORCH_WARN_ONCE("The oprator of ne is executed, Currently High Accuracy but Low Performance OP with 64-bit has been used,"
      "Please Do Some Cast at Python Functions with 32-bit for Better Performance!");
  }
  OpCommand cmd;
  cmd.Name("NotEqual")
    .Expect(unified_result)
    .Input(selfCast)
    .Input(otherCast)
    .Output(result)   
    .Run();
  
  return result;
}

Tensor& ne_out_npu_nocheck(Tensor& result, const Tensor& self, Scalar other) {
  Tensor selfCast = self;
  if(self.dtype() == ScalarType::Int){
    selfCast = self.to(ScalarType::Float);
  }
  if(self.scalar_type() == at::kLong) {
    TORCH_WARN_ONCE("The oprator of ne is executed, Currently High Accuracy but Low Performance OP with 64-bit has been used,"
      "Please Do Some Cast at Python Functions with 32-bit for Better Performance!");
  }
  OpCommand cmd;
  cmd.Name("NotEqual")
    .Input(selfCast)
    .Input(other, selfCast.scalar_type())
    .Output(result)   
    .Run();

  return result;
}

Tensor& ne_out_npu(Tensor& result, const Tensor& self, const Tensor& other) {
  Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  Tensor formatCastOfOther = OpPreparation::CastBackToOriFormat(other);
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  OpPreparation::CheckOut(
    {self, other}, 
    result, 
    ACL_FORMAT_ND,
    result.scalar_type(), 
    IntArrayRef(outputSize));
  ne_out_npu_nocheck(result, formatCastOfSelf, formatCastOfOther);
  return result;
}

Tensor& ne_out_npu(Tensor& result, const Tensor& self, Scalar other) {
  Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  auto outputSize = formatCastOfSelf.sizes();
  OpPreparation::CheckOut(
    {self}, 
    result, 
    ACL_FORMAT_ND,
    result.scalar_type(), 
    outputSize);
  ne_out_npu_nocheck(result, formatCastOfSelf, other);
  return result;
}

Tensor ne_npu(const Tensor& self, const Tensor& other) {
  Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  Tensor formatCastOfOther = OpPreparation::CastBackToOriFormat(other);
  // calculate the output size
  auto outputSize = broadcast_ops_npu_output_size(formatCastOfSelf, formatCastOfOther);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
    outputSize,
    formatCastOfSelf.options().dtype(kBool),
    ACL_FORMAT_ND);

  // calculate the output result of the NPU
  ne_out_npu_nocheck(result, formatCastOfSelf, formatCastOfOther);
  return result;
}

Tensor ne_npu(const Tensor& self, Scalar other) {
  Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  // calculate the output size
  auto outputSize = input_same_output_size(formatCastOfSelf);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
    outputSize,
    formatCastOfSelf.options().dtype(kBool),
    ACL_FORMAT_ND);

  // calculate the output result of the NPU
  ne_out_npu_nocheck(result, formatCastOfSelf, other);
  return result;
}

Tensor& ne_npu_(Tensor& self, const Tensor& other) {
  OpPreparation::CastBackToOriFormat(self);
  OpPreparation::CastBackToOriFormat(other);
  OpPreparation::CheckMemory({self, other}, {self});

  Tensor result = at::empty_with_format(
    self.sizes(),
    self.options().dtype(ScalarType::Byte),
    ACL_FORMAT_ND);

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    ne_out_npu_nocheck(result, contiguousSelf, other);
  } else {
    ne_out_npu_nocheck(result, self, other);
  }

  // uint8 to self dtype
  self.copy_(result);

  return self;
}

Tensor& ne_npu_(Tensor& self, Scalar other) {
  OpPreparation::CastBackToOriFormat(self);
  OpPreparation::CheckMemory({self}, {self});
  Tensor result = at::empty_with_format(
    self.sizes(),
    self.options().dtype(ScalarType::Byte),
    ACL_FORMAT_ND);

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    ne_out_npu_nocheck(result, contiguousSelf, other);
  } else {
    ne_out_npu_nocheck(result, self, other);
  }

  // uint8 to self dtype
  self.copy_(result);

  return self;
}

} // namespace native
} // namespace at
