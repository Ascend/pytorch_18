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

Tensor& hardtanh_out_npu(
    Tensor& result,
    const Tensor& self,
    Scalar min,
    Scalar max) {
  OpCommand cmd;
  cmd.Name("ClipByValue")
      .Input(self)
      .Input(min, self.scalar_type())
      .Input(max, self.scalar_type())
      .Output(result)
      .Run();
 
  return result;
}

Tensor hardtanh_npu(const Tensor& self, Scalar min, Scalar max) {
  Tensor result = OpPreparation::ApplyTensor(self);
  hardtanh_out_npu(result, self, min, max);
  return result;
}

Tensor& hardtanh_npu_(Tensor& self, Scalar min, Scalar max) {
  OpPreparation::CheckMemory({self}, {self});
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = hardtanh_out_npu(contiguousSelf, contiguousSelf, min, max);
    NpuUtils::format_fresh_view(self, result);
  } else {
    hardtanh_out_npu(self, self, min, max);
  }
 
  return self;
}

} // namespace native
} // namespace at
