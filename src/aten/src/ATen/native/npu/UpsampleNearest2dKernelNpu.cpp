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

SmallVector<int64_t, SIZE> upsample_nearest2d_npu_output_size(
    const Tensor& input,
    IntArrayRef output_size){
  int64_t N = input.size(0);
  int64_t C = input.size(1);
  int64_t H = output_size[0];
  int64_t W = output_size[1];
  SmallVector<int64_t, SIZE> outputSize = {N, C, H, W};

  return outputSize;
}

Tensor& upsample_nearest2d_out_npu(
    Tensor& result,
    const Tensor& self,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  SmallVector<int64_t, SIZE> outputSize = upsample_nearest2d_npu_output_size(self, output_size);
  if (!result.sizes().equals(outputSize)){
    result.resize_(outputSize);
  }
  SmallVector<int64_t,N> outputSizeVec = array_to_small_vector(output_size);
  if (!c10::npu::OptionsManager::CheckDynamicEnable()) {
    OpCommand cmd;
    cmd.Name("ResizeNearestNeighborV2")
      .Input(self)
      .Input(outputSizeVec, at::kInt)
      .Output(result)
      .Attr("align_corners", false)
      .Attr("half_pixel_centers", false)
      .Run();
  } else {
    OpDynamicCommand cmd;
    cmd.Name("ResizeNearestNeighborV2D")
      .Input(self)
      .Output(result)
      .Attr("align_corners", false)
      .Attr("size", output_size)
      .Attr("half_pixel_centers", false);
    
    Tensor outputSizeTensorCpu = from_blob((void*)outputSizeVec.data(), {outputSizeVec.size()}, at::kLong).to(at::kInt);
    Tensor outputSizeTensorNpu = CalcuOpUtil::copy_tensor_host_to_device(outputSizeTensorCpu);
    cmd.DynamicName("ResizeNearestNeighborV2")
        .DynamicInput(self)
        .DynamicInput(outputSizeTensorNpu, "size")
        .DynamicOutput(result)
        .DynamicAttr("align_corners", false)
        .DynamicAttr("half_pixel_centers", false)
        .DynamicOpRun();
  }
  return result;
}

Tensor upsample_nearest2d_npu(
    const Tensor& self,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  // calculate the output size
  SmallVector<int64_t, SIZE> outputSize = upsample_nearest2d_npu_output_size(self, output_size);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  upsample_nearest2d_out_npu(result, self, output_size, scales_h, scales_w);

  return result;
}

} // namespace native
} // namespace at