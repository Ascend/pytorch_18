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

#include "NPUTypeProperties.h"
#include "NPUTensorIterator.h"

namespace at {
namespace native {
namespace npu {

std::tuple<ScalarType, IntArrayRef> NPUTensorIterator::binary_op(
    Tensor& out, 
    const Tensor& a,
    const Tensor& b, 
    bool check_mem_overlap) {
  auto iter = NPUTensorIterator();
  iter.add_output(out);
  iter.add_input(a);
  iter.add_input(b);
  iter.promote_common_dtype();
  iter.compute_types();
  auto common_type = iter.common_dtype();
  auto common_shape = a.sizes();
  return std::tie(common_type, common_shape);
}

std::tuple<ScalarType, IntArrayRef> NPUTensorIterator::binary_op(
    const Tensor& a,
    const Scalar b) {
  ScalarType scalar_type;
  if (b.isFloatingPoint()) {
    scalar_type = ScalarType::Float;
  } else if (b.isBoolean()) {
    scalar_type = ScalarType::Bool;
  } else if (b.isComplex()) {
    scalar_type = ScalarType::ComplexFloat;
  } else {
    AT_ASSERT(b.isIntegral(false));
    scalar_type = ScalarType::Int;
  }
  if (a.scalar_type() == ScalarType::Half) {
    scalar_type = ScalarType::Half;
  }
  if (a.scalar_type() != scalar_type) {
    scalar_type = result_type(a.scalar_type(), scalar_type);
  }
  auto common_shape = a.sizes();
  return std::tie(scalar_type, common_shape);
}

std::tuple<ScalarType, IntArrayRef> NPUTensorIterator::comparison_op(
    Tensor& out, 
    const Tensor& a,
    const Tensor& b, 
    bool check_mem_overlap) {
  auto iter = NPUTensorIterator();
  iter.add_output(out);
  iter.add_input(a);
  iter.add_input(b);
  iter.compute_common_dtype_only_for_inputs();
  iter.compute_types();
  auto common_type = iter.common_dtype();
  auto common_shape = a.sizes();
  return std::tie(common_type, common_shape);
}

std::tuple<ScalarType, IntArrayRef> NPUTensorIterator::unary_op(
    Tensor& out, 
    const Tensor& a,
    bool check_mem_overlap) {
  auto iter = NPUTensorIterator();
  iter.add_output(out);
  iter.add_input(a);
  iter.num_outputs_ = 1;
  iter.compute_types();
  auto common_type = iter.common_dtype();
  auto common_shape = a.sizes();
  return std::tie(common_type, common_shape);
}

void NPUTensorIterator::nullary_op(Tensor& out) {
  auto iter = NPUTensorIterator();
  iter.add_output(out);
  iter.compute_types();
}

std::tuple<ScalarType, IntArrayRef> NPUTensorIterator::reduce_op(Tensor& out, const Tensor& a) {
  TORCH_INTERNAL_ASSERT(out.defined());
  auto iter = NPUTensorIterator();
  iter.add_output(out);
  iter.add_input(a);
  iter.promote_npu_output_dtypes_ = true;
  iter.is_reduction_ = true;
  // TODO: This is only really necessary for arg{min,max}
  iter.compute_common_dtype_only_for_inputs();
  iter.compute_types();
  auto common_type = iter.common_dtype();
  auto common_shape = a.sizes();
  return std::tie(common_type, common_shape);
}

std::tuple<ScalarType, IntArrayRef> NPUTensorIterator::reduce_op(
    Tensor& out1, 
    Tensor& out2, 
    const Tensor& a) {
  TORCH_INTERNAL_ASSERT(out1.defined());
  TORCH_INTERNAL_ASSERT(out2.defined());
  TORCH_CHECK(out1.dim() == out2.dim(), "reduce_op(): expected both outputs to have same number of dims, but output1 has ", out1.dim(),
      " and output2 has ", out2.dim());
  TORCH_CHECK(out1.sizes() == out2.sizes(), "reduce_op(): expected both outputs to have same sizes, but output1 has ", out1.sizes(),
      " and output2 has ", out2.sizes());
  TORCH_CHECK(out1.strides() == out2.strides(), "reduce_op(): expected both outputs to have same strides, but output1 has ", out1.strides(),
           " and output2 has ", out2.strides());
  auto iter = NPUTensorIterator();
  iter.add_output(out1);
  iter.add_output(out2);
  iter.add_input(a);
  iter.promote_npu_output_dtypes_ = true;
  iter.is_reduction_ = true;
  iter.compute_types();
  auto common_type = iter.common_dtype();
  auto common_shape = a.sizes();
  return std::tie(common_type, common_shape);
}

static std::tuple<ScalarType, bool> compute_common_type_(at::ArrayRef<NPUOperandInfo> operands) {
  // See [Result type computation] in NPUTensorIterator.h
  auto common_type = ScalarType::Undefined;
  bool all_same_type = true;
  for (const auto& op: operands) {
    if (!op.tensor.defined()) 
      continue;
    //don't handle scalars
    if (op.tensor.dim() > 0) {
      ScalarType current = op.current_dtype;
      if (current == ScalarType::Undefined) {
        all_same_type = false;
        break;
      }
      if (common_type == ScalarType::Undefined) {
        common_type = current;
      }
      if (common_type != current) {
        all_same_type = false;
        break;
      }
    } else {
      all_same_type = false;
      break;
    }
  }
  if (all_same_type) {
    return std::make_tuple(common_type, true);
  }

  ResultTypeState state = {};
  for (const auto& op : operands) {
    state = update_result_type_state(op.tensor, state);
  }
  auto dtype = result_type(state);

  auto result = std::make_tuple(dtype, false);
  TORCH_INTERNAL_ASSERT(dtype != ScalarType::Undefined);
  return result;
}

std::tuple<ScalarType, bool> NPUTensorIterator::compute_common_type() {
  return compute_common_type_(operands_);
}

void NPUTensorIterator::compute_types() {
  bool missing_dtypes = false;
  bool missing_output_dtypes = false;
  common_dtype_ = dtype();
  for (auto& op : operands_) {
    if (!op.tensor.defined() && !op.is_type_defined()) {
      missing_dtypes = true;
      if (op.is_output) {
        missing_output_dtypes = true;
      }
    }
  }

  if (common_dtype_strategy_ == CommonDTypeStrategy::PROMOTE_INPUTS) {
    TORCH_CHECK(!missing_output_dtypes, "unable to compute and promote common dtype based only on inputs if there are missing dtypes for outputs");
  }
  bool compute_common_dtype = (common_dtype_strategy_ != CommonDTypeStrategy::NONE);
  bool compute_common_dtype_only_for_inputs = (common_dtype_strategy_ == CommonDTypeStrategy::PROMOTE_INPUTS);
  if (missing_dtypes || compute_common_dtype) {
    auto operands = compute_common_dtype_only_for_inputs ? at::ArrayRef<NPUOperandInfo>(operands_).slice(noutputs()) : operands_;
    auto common_type = compute_common_type_(operands);
    common_dtype_ = std::get<0>(common_type);
  }
}

} // namespace npu
} // namespace native
} // namespace at
