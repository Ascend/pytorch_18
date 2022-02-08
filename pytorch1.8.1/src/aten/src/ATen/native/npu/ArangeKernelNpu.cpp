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

// bool inputs are considered integral
static inline bool allIntegral(
    std::initializer_list<std::reference_wrapper<Scalar>> l) {
  for (Scalar& s : l) {
    if (!s.isIntegral(true)) {
      return false;
    }
  }
  return true;
}


Tensor& arange_out_npu_nocheck(
    Tensor& result,
    Scalar start,
    Scalar end,
    Scalar step) {
  OpCommand cmd;
  cmd.Name("Range")
     .Input(start, result.scalar_type())
     .Input(end, result.scalar_type())
     .Input(step, result.scalar_type())
     .Output(result)
     .Run();

  return result;
}

Tensor arange_start_step_npu(
    Scalar start,
    Scalar end,
    Scalar step,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {

  auto device =  device_or_default(device_opt);
  TensorOptions option;
  option = option.dtype(dtype_opt)
                 .layout(layout_opt)
                 .device(device)
                 .pinned_memory(pin_memory_opt);

  float start_value = CalcuOpUtil::get_scalar_float_value(start);
  float end_value = CalcuOpUtil::get_scalar_float_value(end);
  float step_value = CalcuOpUtil::get_scalar_float_value(step);

  // Check step start end
  TORCH_CHECK(step_value != 0, "step must be nonzero");
  TORCH_CHECK(((step_value > 0) && (end_value >= start_value)) || ((step_value < 0) && (end_value <= start_value)),
      "upper bound and larger bound inconsistent with step sign");

  bool set_to_integral_dtype =
      !option.has_dtype() && allIntegral({start, end, step});

  // check start == end
  if (set_to_integral_dtype) {
    option = option.dtype(at::ScalarType::Int);
  }
  Tensor result_check = OpPreparation::ApplyTensorWithFormat({0}, option, ACL_FORMAT_ND);

  if (start_value == end_value) {
    return result_check;
  }

  // calculate the output size
  double size_arange = std::ceil(static_cast<double>(end.toDouble() - start.toDouble())
                                 / step.toDouble());
  int64_t size_value = static_cast<int64_t>(size_arange);
  SmallVector<int64_t, SIZE> outputSize = {size_value};
  Tensor result = OpPreparation::ApplyTensorWithFormat(outputSize, option, ACL_FORMAT_ND);

  if(option.dtype() == at::kHalf) {
    result = result.to(at::kFloat);
  }

  arange_out_npu_nocheck(result, start, end, step);

  if(option.dtype() == at::kHalf) {
    result = result.to(at::kHalf);
  }

  return result;
}

Tensor arange_start_npu(
    Scalar start, 
    Scalar end, 
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {

  return arange_start_step_npu(start, end, 1, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}


Tensor arange_npu(
    Scalar end, 
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {

  return arange_start_npu(0, end, dtype_opt, layout_opt, device_opt, pin_memory_opt);  // start = 0
}

Tensor& arange_start_out_npu(
    Scalar start,
    Scalar end,
    Scalar step,
    Tensor& result) {
  float start_value = CalcuOpUtil::get_scalar_float_value(start);
  float end_value = CalcuOpUtil::get_scalar_float_value(end);
  float step_value = CalcuOpUtil::get_scalar_float_value(step);

  // Check step start end
  TORCH_CHECK(step_value != 0, "step must be nonzero");
  TORCH_CHECK(((step_value > 0) && (end_value >= start_value)) || ((step_value < 0) && (end_value <= start_value)),
      "upper bound and larger bound inconsistent with step sign");

  // calculate the output size
  double size_arange = std::ceil(static_cast<double>(end.toDouble() - start.toDouble())
                                 / step.toDouble());
  int64_t size_value = static_cast<int64_t>(size_arange);
  SmallVector<int64_t, SIZE> outputSize = {size_value};
  result.resize_(outputSize);

  arange_out_npu_nocheck(result, start, end, step);

  return result;
}

Tensor& arange_other_out_npu(Scalar start, Scalar end, Tensor& result) {
  return arange_start_out_npu(start, end, 1, result);
}

Tensor& arange_out_npu(Scalar end, Tensor& result) {
  return arange_other_out_npu(0, end, result);
}

Tensor _dim_arange_npu(const Tensor& self, int64_t dim) {
  c10::optional<ScalarType> dtype_opt(at::kInt);
  c10::optional<Layout> layout_opt(self.options().layout());
  c10::optional<Device> device_opt(self.options().device());
  c10::optional<bool> pin_memory_opt(self.options().pinned_memory());

  Tensor result = arange_npu(self.size(dim), dtype_opt, layout_opt, device_opt, pin_memory_opt);
  return result;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("arange", TORCH_FN(arange_npu));
  m.impl("arange.start", TORCH_FN(arange_start_npu));
  m.impl("arange.start_step", TORCH_FN(arange_start_step_npu));
  m.impl("arange.out", TORCH_FN(arange_out_npu));
  m.impl("arange.start_out", TORCH_FN(arange_start_out_npu));
  m.impl("_dim_arange", TORCH_FN(_dim_arange_npu));
}
} // namespace native
} // namespace at