// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

#include "OpPreparation.h"
#include "ATen/native/npu/frame/FormatHelper.h"
#include "ATen/native/npu/frame/InferFormat.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
namespace npu {

UnifiedResult OpPreparation::binary_op_check(
    Tensor& out,
    const Tensor& a,
    const Tensor& b,
    bool check_mem_overlap) {
  UnifiedResult unified_result;
  if (a.dtype() != b.dtype()) {
    std::tuple<ScalarType, IntArrayRef> binary_op = NPUTensorIterator::binary_op(out, a, b, check_mem_overlap);
    unified_result.common_type = std::get<0>(binary_op);
    unified_result.common_shape = std::get<1>(binary_op);
  }
  return unified_result;
}

UnifiedResult OpPreparation::binary_op_check(
    Tensor& out,
    const Tensor& a,
    const Scalar b,
    bool check_mem_overlap) {
  UnifiedResult unified_result;
  std::tuple<ScalarType, IntArrayRef> binary_op = NPUTensorIterator::binary_op(a, b);
  unified_result.common_type = std::get<0>(binary_op);
  unified_result.common_shape = std::get<1>(binary_op);
  return unified_result;
}


UnifiedResult OpPreparation::comparison_op_check(
    Tensor& out,
    const Tensor& a,
    const Tensor& b,
    bool check_mem_overlap) {
  UnifiedResult unified_result;
  if (a.dtype() != b.dtype()) {
    std::tuple<ScalarType, IntArrayRef> comparison_op = NPUTensorIterator::comparison_op(out, a, b, check_mem_overlap);
    unified_result.common_type = std::get<0>(comparison_op);
    unified_result.common_shape = std::get<1>(comparison_op);
  }
  if(out.dtype() != a.dtype() && out.dtype() != b.dtype()) {
    unified_result.result_type_defined = true;
  }
  return unified_result;
}

UnifiedResult OpPreparation::unary_op_check(
    Tensor& out,
    const Tensor& a,
    bool check_mem_overlap) {
  UnifiedResult unified_result;
  std::tuple<ScalarType, IntArrayRef> unary_op = NPUTensorIterator::unary_op(out, a, check_mem_overlap);
  unified_result.common_type = std::get<0>(unary_op);
  unified_result.common_shape = std::get<1>(unary_op);
  return unified_result;
}

void OpPreparation::nullary_op(Tensor& out) {
  NPUTensorIterator::nullary_op(out);
}

UnifiedResult OpPreparation::reduce_op_check(Tensor& out, const Tensor& a) {
  UnifiedResult unified_result;
  std::tuple<ScalarType, IntArrayRef> reduce_op = NPUTensorIterator::reduce_op(out, a);
  unified_result.common_type = std::get<0>(reduce_op);
  unified_result.common_shape = std::get<1>(reduce_op);
  return unified_result;
}

UnifiedResult OpPreparation::reduce_op_check(Tensor& out1, Tensor& out2, const Tensor& a) {
  UnifiedResult unified_result;
  std::tuple<ScalarType, IntArrayRef> reduce_op = NPUTensorIterator::reduce_op(out1, out2, a);
  unified_result.common_type = std::get<0>(reduce_op);
  unified_result.common_shape = std::get<1>(reduce_op);
  return unified_result;
}

// OpPreparation part
void OpPreparation::CheckOut(
    const std::initializer_list<Tensor>& inputs,
    Tensor& output,
    Tensor dst) {
  CheckOut(
      inputs,
      output, 
      CalcuOpUtil::get_tensor_npu_format(dst),
      dst.scalar_type(),
      dst.sizes());
}

void OpPreparation::CheckOut(
    const std::initializer_list<Tensor>& inputs,
    Tensor& output,
    Tensor dst,
    IntArrayRef shape) {
  CheckOut(
      inputs,
      output, 
      CalcuOpUtil::get_tensor_npu_format(dst),
      dst.scalar_type(),
      shape);
}

void OpPreparation::CheckOut(
    const std::initializer_list<Tensor>& input,
    Tensor& output,
    int64_t format,
    ScalarType dtype,
    IntArrayRef shape) {
  // Check that the outputs have no internal overlap
  // and do not share memory with inputs.
  SmallVector<Tensor, N> inputs{input};
  SmallVector<Tensor, N> outputs = {output};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);
  TORCH_CHECK(output.is_npu(), "output with device ",
      output.device(), " doesn't match the desired device NPU");  
  TORCH_CHECK(output.scalar_type() == dtype, "expected dtype ",
      dtype, " but got dtype ", output.scalar_type());
  
  bool is_read_write = false;
  // check if output is also an input
  for (const auto& input : inputs) {
    if (output.is_same(input)) {
      is_read_write = true;
      break;
      }
  }

  // Preserve legacy resizing behavior of out=... arguments
  if (!output.sizes().equals(shape)) {
    TORCH_CHECK(!is_read_write, "output with shape ", output.sizes(),
        " doesn't match the broadcast shape ", shape);
    output.resize_(shape);
  }

  if (!output.is_contiguous()) {
    Tensor tmp = output.contiguous();
    output.copy_(tmp);
  }

  if (CalcuOpUtil::get_tensor_npu_format(output) != format) {
    if (output.scalar_type() == ScalarType::Float || output.scalar_type() == ScalarType::Half) {
      TORCH_CHECK(!is_read_write, "can not cast format when output is input");
      output.npu_format_cast_(format);
    } else {
      TORCH_CHECK(FormatHelper::IsBaseFormatType(output) && FormatHelper::IsBaseFormatType(static_cast<aclFormat>(format)),
        "can not cast format to un-base format when output has bool dtype");
      output.npu_format_cast_(format);
    }
  }
}

Tensor OpPreparation::CastBackToOriFormat(const Tensor& tensor) {
  auto &tensor_desc = tensor.storage().unsafeGetStorageImpl()->npu_desc_; 
  auto ret = tensor.npu_format_cast(tensor_desc.origin_format_);
  return ret;
}

Tensor& OpPreparation::CastBackToOriFormat(Tensor& tensor) {
  auto &tensor_desc = tensor.storage().unsafeGetStorageImpl()->npu_desc_; 
  tensor.npu_format_cast_(tensor_desc.origin_format_);
  return tensor;
}

Tensor OpPreparation::ApplyTensor(const Tensor& src) {
  return ApplyTensor(src, src.sizes());
}

Tensor OpPreparation::ApplyTensor(const Tensor& src, IntArrayRef sizes) {
  return ApplyTensorWithFormat(sizes, src.options(), CalcuOpUtil::get_tensor_npu_format(src));
}

Tensor OpPreparation::ApplyTensor(const Tensor& src, const TensorOptions& options) {
  return ApplyTensorWithFormat(src.sizes(), options, CalcuOpUtil::get_tensor_npu_format(src));
}

Tensor OpPreparation::ApplyTensor(IntArrayRef sizes, const TensorOptions& options, const Tensor& src) {
  return ApplyTensorWithFormat(sizes, options, CalcuOpUtil::get_tensor_npu_format(src));
}

Tensor OpPreparation::ApplyTensorWithFormat(const Tensor& src, int64_t format) {
  return ApplyTensorWithFormat(src, src.sizes(), format);
}

Tensor OpPreparation::ApplyTensorWithFormat(const Tensor& src, IntArrayRef sizes, int64_t format) {
  return ApplyTensorWithFormat(sizes, src.options(), format);
}

Tensor OpPreparation::ApplyTensorWithFormat(IntArrayRef sizes, const TensorOptions& options, int64_t format) {
  auto fixFormat = InferFormat::GuessStorageFormat(sizes, (aclFormat)format);
  return at::empty_with_format(sizes, options, fixFormat);
}

Tensor OpPreparation::ApplyTensorWithSizes(IntArrayRef sizes, const TensorOptions& options) {
  auto format = InferFormat::GuessBaseFormat(sizes);
  return at::empty_with_format(sizes, options, format);
}

void OpPreparation::CheckMemory(const std::initializer_list<Tensor>& inputs, const std::initializer_list<Tensor>& outputs) {
  SmallVector<Tensor, N> in = inputs;
  SmallVector<Tensor, N> out = outputs;
  CalcuOpUtil::check_memory_over_laps(in, out);
}

} // namespace npu
} // namespace native
} // namespace at