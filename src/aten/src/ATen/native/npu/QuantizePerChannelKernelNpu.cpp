// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
using namespace at::native::npu;

SmallVector<int64_t, SIZE> quantize_reshape_size(
    const Tensor& self,
    int64_t axis) {
  SmallVector<int64_t, SIZE> outSize;
  for(int64_t i=0; i < self.dim(); i++) {
    if(i != axis) {
      outSize.emplace_back(1);
    } else {
      outSize.emplace_back(self.sizes()[i]);
    }
  }
  return outSize;
}

SmallVector<NPUTensorDesc, N> quantize_per_channel_npu_input (
    const SmallVector<Tensor, N>& self) {
  return CalcuOpUtil::create_npu_input_tensor_desc(self);
}

SmallVector<NPUTensorDesc, N> quantize_per_channel_npu_output (
    const SmallVector<Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUAttrDesc, N> quantize_per_channel_npu_attr (int64_t axis, ScalarType dtype) {
  NPUAttrDesc npuAttrAxis = NPUAttrDesc("axis", axis);
  string dtypeStr;
  if (dtype == ScalarType::QInt8) {
    dtypeStr = "torch.qint8";
  } else if (dtype == ScalarType::QUInt8) {
    dtypeStr = "torch.quint8";
  } else if (dtype == ScalarType::QInt32) {
    dtypeStr = "torch.qint32";
  }
  NPUAttrDesc npuAttrDtype = NPUAttrDesc("dtype", dtypeStr);
  SmallVector<NPUAttrDesc, N> attrs = {npuAttrAxis, npuAttrDtype};

  return attrs;
}

Tensor& quantize_per_channel_out_npu_after_broadcast (
    Tensor& result,
    const Tensor& self,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis,
    ScalarType dtype) {
  // constructs the input and output NPUTensorDesc
  auto inputs = quantize_per_channel_npu_input({self, scales, zero_points});
  auto outputs = quantize_per_channel_npu_output({result});

  // constructs the attr of the NPUAttrDesc
  auto attrs = quantize_per_channel_npu_attr(axis, dtype);

  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("Quantize", inputs, outputs, attrs);

  return result;
}

Tensor& quantize_per_channel_out_npu (
    Tensor& result,
    const Tensor& self,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis,
    ScalarType dtype) {
  auto reshapeSize = quantize_reshape_size(self, axis);
  Tensor scales_reshape = scales.reshape(reshapeSize);
  Tensor zp_reshape = zero_points.reshape(reshapeSize);
  Tensor scales_broadcast = at::npu_broadcast(scales_reshape, self.sizes());
  Tensor zp_broadcast = at::npu_broadcast(zp_reshape, self.sizes());
  quantize_per_channel_out_npu_after_broadcast(
    result,
    self,
    scales_broadcast,
    zp_broadcast,
    axis,
    dtype
  );
  return result;
}

Tensor quantize_per_channel_npu (
    const Tensor& self,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis,
    ScalarType dtype) {
  axis = CalcuOpUtil::make_wrap_dim(axis, self.dim());
  TORCH_CHECK(scales.dim() == 1, "Scales' dim should be equal to 1.");
  TORCH_CHECK(zero_points.dim() == 1, "Zero points' dim should be equal to 1.");
  TORCH_CHECK(scales.sizes()[0] == zero_points.sizes()[0], "Scales' size should be equal to zero points' size.");
  TORCH_CHECK(scales.sizes()[0] == self.sizes()[axis], "length of scales must equal to the specified dimension.");
  // constructs the input and output NPUTensorDesc
  auto outputSize = input_same_output_size(self);
  auto outputDtype = kInt;
  if (dtype == ScalarType::QInt8) {
    outputDtype = kChar;
  } else if (dtype == ScalarType::QUInt8) {
    outputDtype = kByte;
  } else if (dtype == ScalarType::QInt32) {
    outputDtype = kInt;
  }
  Tensor result = at::empty_with_format(
      outputSize,
      self.options().dtype(outputDtype),
      CalcuOpUtil::get_tensor_npu_format(self));
  quantize_per_channel_out_npu(result, self, scales, zero_points, axis, dtype);
  return result;
}

} // namespace native
} // namespace at