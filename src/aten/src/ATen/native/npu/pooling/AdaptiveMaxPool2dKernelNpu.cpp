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

#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

tuple<Tensor&, Tensor&> adaptive_max_pool2d_out_npu(
    Tensor& output,
    Tensor& indices,
    const Tensor& self,
    IntArrayRef output_size) {

  auto inputsize = self.sizes();

  SmallVector<int64_t, N> input_size;
  if (inputsize.size() == 3) {
      SmallVector<int64_t, N> size = { inputsize[1], inputsize[2] };
      input_size = IntArrayRef(size);
  } else if (inputsize.size() == 4) {
      SmallVector<int64_t, N> size = { inputsize[2], inputsize[3] };
      input_size = IntArrayRef(size);
  }

  if (input_size[0] % output_size[0] == 0 && input_size[1] % output_size[1] == 0) {
      int64_t kernel_size[2];
      int64_t stride[2];
      int64_t padding[2];
      int64_t strideH = input_size[0] / output_size[0];
      int64_t strideW = input_size[1] / output_size[1];
      int64_t kernel_sizeH = input_size[0] - (output_size[0] - 1) * strideH;
      int64_t kernel_sizeW = input_size[1] - (output_size[1] - 1) * strideW;
      stride[0] = strideH;
      stride[1] = strideW;
      kernel_size[0] = kernel_sizeH;
      kernel_size[1] = kernel_sizeW;
      padding[0] = padding[1] = 0;

      SmallVector<int64_t, N> kernelSize = {1, kernel_size[0], kernel_size[1], 1};
      SmallVector<int64_t, N> stridesSize = {1, stride[0], stride[1], 1};
      SmallVector<int64_t, N> paddings = {1, padding[0], padding[1], 1};
      SmallVector<int64_t, N> dilations = {1, 1, 1, 1};
      bool ceil_mode = false;

      OpCommand cmd;
      cmd.Name("MaxPoolWithArgmaxV1")
          .Input(self)
          .Output(output)
          .Output(indices,"uint16")
          .Attr("ksize", kernelSize)
          .Attr("strides", stridesSize)
          .Attr("pads", paddings)
          .Attr("dilation", dilations)
          .Attr("ceil_mode", ceil_mode)
          .Run();

    } else {
        // H and W can not be divided, temporarily reported error processing
        AT_ERROR("H and W must be divisible");
    }
    return tuple<Tensor&, Tensor&>(output, indices);
}

tuple<Tensor, Tensor> adaptive_max_pool2d_npu(
    const Tensor& self,
    IntArrayRef output_size) {

  for (int64_t i = 0; i < self.dim(); i++) {
    TORCH_CHECK(
        self.size(i) > 0,
        "adaptive_max_pooling2d(): expected input to have non-empty spatial dimensions, "
        "but input has sizes ",
        self.sizes(),
        " with dimension ",
        i,
        " being "
        "empty");
  }
  TORCH_CHECK(
      (self.dim() == 3 || self.dim() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");

  TORCH_CHECK(
      (output_size.size() == 2),
      "adaptive_max_pool2d: internal error: output_size.size() must be 2");

  // size
  int64_t N = self.size(0);
  int64_t C = self.size(1);
  int64_t H = self.size(2);
  int64_t W = self.size(3);

  int64_t strideH = H / output_size[0];
  int64_t strideW = W / output_size[1];

  int64_t kernel_sizeH = H - (output_size[0] - 1) * strideH;
  int64_t kernel_sizeW = W - (output_size[1] - 1) * strideW;

  int64_t Ho = output_size[0];
  int64_t Wo = output_size[1];

  SmallVector<int64_t, SIZE> outputSize = {N, C, Ho, Wo};

  const int64_t BLOCKSIZE = 16;
  int64_t maskH = kernel_sizeH * kernel_sizeW;
  int64_t maskW = (CeilDiv(Ho * Wo, BLOCKSIZE) + 1);

  SmallVector<int64_t, SIZE> indicesSize = {N, C, maskH, maskW};

  // construct the output tensor of the NPU
  Tensor output = OpPreparation::ApplyTensorWithFormat(self, outputSize, ACL_FORMAT_NC1HWC0);
  Tensor indices = OpPreparation::ApplyTensorWithFormat(indicesSize, self.options().dtype(kLong), ACL_FORMAT_NC1HWC0);

  // calculate the output result of the NPU
  adaptive_max_pool2d_out_npu(output, indices, self, output_size);
  return tuple<Tensor, Tensor>(output, indices);
}

} // namespace native
} // namespace at