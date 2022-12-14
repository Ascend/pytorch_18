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

namespace {
bool isSpecialConv1d(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  if (stride[1] > 63 &&
      stride[1] == weight.size(3) &&
      padding[1] == 0 &&
      dilation[1] == 1 &&
      groups == 1 &&
      input.size(1) == 1) {
    return true;
  } else {
    return false;
  }
} // isSpecialConv1d
} // namespace

Tensor conv2d_backward_input_out_npu(
    Tensor& gradInput,
    const Tensor& input,
    const Tensor& grad,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  // support special scenario
  if (isSpecialConv1d(input,
                      weight,
                      stride,
                      padding,
                      dilation,
                      groups)) {
    Tensor mmInput = grad.permute({0, 2, 1});
    Tensor mmOther = weight.reshape({weight.size(0), weight.size(3)});
    Tensor mmResult = at::matmul(mmInput, mmOther);
    gradInput = mmResult.reshape({grad.size(0), 1, 1, grad.size(2)*weight.size(3)});
    return gradInput;
  }

  SmallVector<int64_t, N> dimList = array_to_small_vector(input.sizes());
  SmallVector<int64_t, N> stridesSize = {1, 1, stride[0], stride[1]};
  SmallVector<int64_t, N> paddings = {
      padding[0], padding[0], padding[1], padding[1]};
  SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};
  string dataFormat = "NCHW";
  string sizeName = "input_size";

  // executing the NPU operator
  if (!c10::npu::OptionsManager::CheckDynamicEnable()) {
    OpCommand cmd;
    cmd.Name("Conv2DBackpropInput")
        .Input(dimList, at::kInt)
        .Input(weight)
        .Input(grad)
        .Output(gradInput)
        .Attr("strides", stridesSize)
        .Attr("pads", paddings)
        .Attr("dilations", dilations)
        .Attr("groups", groups)
        .Attr("data_format", dataFormat)
        .Run();

  } else {
    OpCommand cmd;
    cmd.Name("Conv2DBackpropInputD")
        .Input(weight)
        .Input(grad)
        .Output(gradInput)
        .Attr(sizeName, dimList)
        .Attr("strides", stridesSize)
        .Attr("pads", paddings)
        .Attr("dilations", dilations)
        .Attr("groups", groups)
        .Attr("data_format", dataFormat)
        .Run();
  }
  
  return gradInput;
}

Tensor conv2d_backward_weight_out_npu(
    Tensor& gradWeight,
    const Tensor& input,
    const Tensor& grad,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  // support special scenario
  if (isSpecialConv1d(input,
                      weight,
                      stride,
                      padding,
                      dilation,
                      groups)) {
    Tensor mmInput = grad.permute({1, 0, 2})
                         .reshape({grad.size(1), grad.size(0)*grad.size(2)});
    Tensor mmOther = input.reshape({input.size(0), grad.size(2), input.size(3)/grad.size(2)})
                          .permute({2, 0, 1})
                          .reshape({weight.size(3), input.size(0)*input.size(3)/weight.size(3)})
                          .permute({1, 0});
    Tensor mmResult = at::matmul(mmInput, mmOther);
    gradWeight = mmResult.reshape({grad.size(1), 1, 1, weight.size(3)});
    return gradWeight;
  }
  
  SmallVector<int64_t, N> dimList = array_to_small_vector(weight.sizes());
  SmallVector<int64_t, N> stridesSize = {1, 1, stride[0], stride[1]};
  SmallVector<int64_t, N> paddings = {
      padding[0], padding[0], padding[1], padding[1]};
  SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};
  string dataFormat = "NCHW";
  string sizeName = "filter_size";
  
  // executing the NPU operator
  if (!c10::npu::OptionsManager::CheckDynamicEnable()) {
    OpCommand cmd;
    cmd.Name("Conv2DBackpropFilter")
        .Input(input)
        .Input(dimList, at::kInt)
        .Input(grad)
        .Output(gradWeight)
        .Attr("strides", stridesSize)
        .Attr("pads", paddings)
        .Attr("dilations", dilations)
        .Attr("groups", groups)
        .Attr("data_format", dataFormat)
        .Run();

  } else {
    OpCommand cmd;
    cmd.Name("Conv2DBackpropFilterD")
        .Input(input)
        .Input(grad)
        .Output(gradWeight)
        .Attr(sizeName, dimList)
        .Attr("strides", stridesSize)
        .Attr("pads", paddings)
        .Attr("dilations", dilations)
        .Attr("groups", groups)
        .Attr("data_format", dataFormat)
        .Run();
  }
  
  return gradWeight;
}

Tensor conv2d_backward_bias_out_npu(
    Tensor& gradBias,
    const Tensor& input,
    const Tensor& grad,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  // constructs the input and output NPUTensorDesc
  if (grad.numel() == grad.size(0)*grad.size(1)) {
    Tensor gradView = grad.contiguous().view({grad.size(0), grad.size(1)});
    at::sum_out(gradBias, gradView, SmallVector<int64_t, N>{0});
  } else {
    Tensor gradView = grad.contiguous().view({grad.size(0), grad.size(1), -1});
    at::sum_out(gradBias, gradView, SmallVector<int64_t, N>{0, 2});
  }

  return gradBias;
}

tuple<Tensor&, Tensor&, Tensor&> conv2d_backward_out_npu(
    Tensor& gradInput,
    Tensor& gradWeight,
    Tensor& gradBias,
    const Tensor& input,
    const Tensor& grad,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    std::array<bool, 3> grad_input_mask) {
  // calculate the output result of the NPU
  if (grad_input_mask[0]) {
    conv2d_backward_input_out_npu(
        gradInput, input, grad, weight, stride, padding, dilation, groups);
  }

  if (grad_input_mask[1]) {
    conv2d_backward_weight_out_npu(
        gradWeight, input, grad, weight, stride, padding, dilation, groups);
  }

  if (grad_input_mask[2]) {
    conv2d_backward_bias_out_npu(
        gradBias, input, grad, weight, stride, padding, dilation, groups);
  }

  return tuple<Tensor&, Tensor&, Tensor&>(gradInput, gradWeight, gradBias);
}

tuple<Tensor, Tensor, Tensor> conv2d_backward_npu(
    const Tensor& input,
    const Tensor& grad,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    std::array<bool, 3> grad_input_mask) {
  // calculate the output size
  auto outputSizes = conv2d_backward_npu_output_size(
      input, grad, weight, stride, padding, dilation, groups);

  Tensor gradInput;
  Tensor gradWeight;
  Tensor gradBias;
  // construct the output tensor of the NPU
  if (grad_input_mask[0]) {
    gradInput = at::empty_with_format(
        std::get<0>(outputSizes), input.options(), ACL_FORMAT_NC1HWC0);
  }

  if (grad_input_mask[1]) {
    // For group conv2d: keep consistent with weight to avoid allreduce accuracy problem.
    // For more info: https://gitee.com/ascend/pytorch-develop/pulls/2255
    if (groups > 1) {
      gradWeight = at::empty_with_format(
          std::get<1>(outputSizes),
          weight.options().dtype(kFloat),
          ACL_FORMAT_NCHW);      
    } else {
      gradWeight = at::empty_with_format(
          std::get<1>(outputSizes),
          weight.options().dtype(kFloat),
          ACL_FORMAT_FRACTAL_Z);      
    }
  }

  if (grad_input_mask[2]) {
    gradBias = at::empty_with_format(
        std::get<2>(outputSizes), grad.options(), ACL_FORMAT_NCHW);
  }

  // calculate the output result of the NPU
  conv2d_backward_out_npu(
      gradInput,
      gradWeight,
      gradBias,
      input,
      grad,
      weight,
      stride,
      padding,
      dilation,
      groups,
      grad_input_mask);

  return std::make_tuple(
      std::move(gradInput), std::move(gradWeight), std::move(gradBias));
}

} // namespace native
} // namespace at
