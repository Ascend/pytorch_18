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

Tensor& lt_out_npu_nocheck(Tensor& result, const Tensor& self, const Tensor& other) {
  Tensor selfCast = self;
  Tensor otherCast = other;
  if(self.dtype() == ScalarType::Int || other.dtype() == ScalarType::Int
      || self.dtype() == ScalarType::Bool || other.dtype() == ScalarType::Bool
      || self.dtype() == ScalarType::Long || other.dtype() == ScalarType::Long){
    selfCast = self.to(ScalarType::Float);
    otherCast = other.to(ScalarType::Float);
  }
  auto unified_result = OpPreparation::comparison_op_check(result, selfCast, otherCast, true);
  OpCommand cmd;
  cmd.Name("Less")
      .Expect(unified_result)
      .Input(selfCast)
      .Input(otherCast)
      .Output(result)
      .Run();

  return result;
}

Tensor& lt_out_npu(Tensor& result, const Tensor& self, const Tensor& other) {
  Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  Tensor formatCastOfOther = OpPreparation::CastBackToOriFormat(other);
  auto outputSize = broadcast_ops_npu_output_size(formatCastOfSelf, formatCastOfOther);

  OpPreparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      kBool,
      outputSize);

  lt_out_npu_nocheck(result, formatCastOfSelf, formatCastOfOther);
  return result;
}

Tensor& lt_out_npu_nocheck(Tensor& result, const Tensor& self, Scalar other) {
  Tensor selfCast = self;
  if(self.dtype() == ScalarType::Int || self.dtype() == ScalarType::Bool || self.dtype() == ScalarType::Long){
    selfCast = self.to(ScalarType::Float);
  }
  OpCommand cmd;
  cmd.Name("Less")
      .Input(selfCast)
      .Input(other, selfCast.scalar_type())
      .Output(result)
      .Run();

  return result;
}

Tensor& lt_out_npu(Tensor& result, const Tensor& self, Scalar other) {
  Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  auto outputSize = formatCastOfSelf.sizes();
  OpPreparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      kBool,
      outputSize);

  lt_out_npu_nocheck(result, formatCastOfSelf, other);
  return result;
}

Tensor lt_npu(const Tensor& self, const Tensor& other) {
  Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  Tensor formatCastOfOther = OpPreparation::CastBackToOriFormat(other);
  // calculate the output size
  auto outputSize = broadcast_ops_npu_output_size(formatCastOfSelf, formatCastOfOther);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize,
      formatCastOfSelf.options().dtype(kBool));

  // calculate the output result of the NPU
  lt_out_npu_nocheck(result, formatCastOfSelf, formatCastOfOther);
  return result;
}

Tensor lt_npu(const Tensor& self, Scalar other) {
  Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  // calculate the output size
  auto outputSize = input_same_output_size(formatCastOfSelf);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize,
      formatCastOfSelf.options().dtype(kBool));

  // calculate the output result of the NPU
  lt_out_npu_nocheck(result, formatCastOfSelf, other);
  return result;
}

Tensor& lt_npu_(Tensor& self, const Tensor& other) {
  OpPreparation::CastBackToOriFormat(self);
  OpPreparation::CastBackToOriFormat(other);
  SmallVector<Tensor, N> inputs = {self, other};
  SmallVector<Tensor, N> outputs = {self};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);

  Tensor result = at::empty_with_format(
      self.sizes(),
      self.options().dtype(ScalarType::Byte),
      CalcuOpUtil::get_tensor_npu_format(self));

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    lt_out_npu_nocheck(result, contiguousSelf, other);
  } else {
    lt_out_npu_nocheck(result, self, other);
  }

  // uint8 to self dtype
  self.copy_(result);

  return self;
}

Tensor& lt_npu_(Tensor& self, Scalar other) {
  OpPreparation::CastBackToOriFormat(self);
  SmallVector<Tensor, N> inputs = {self};
  SmallVector<Tensor, N> outputs = {self};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);

  Tensor result = at::empty_with_format(
      self.sizes(),
      self.options().dtype(ScalarType::Byte),
      CalcuOpUtil::get_tensor_npu_format(self));

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    lt_out_npu_nocheck(result, contiguousSelf, other);
  } else {
    lt_out_npu_nocheck(result, self, other);
  }

  // uint8 to self dtype
  self.copy_(result);

  return self;
}

} // namespace native
} // namespace at
