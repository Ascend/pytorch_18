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

Tensor& batch_norm_elemt_nocheck(
    Tensor& result,
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& mean,
    const Tensor& invstd,
    double eps) {
  auto dimC = self.size(1);
  TORCH_CHECK(weight.dim() == 1 && bias.dim() == 1 && mean.dim() == 1 && invstd.dim() == 1,
              "weight, bias, mean, invstd: must be only one dimension.")
  TORCH_CHECK(weight.size(0) == dimC && bias.size(0) == dimC && mean.size(0) == dimC && invstd.size(0) == dimC,
              "weight, bias, mean, invstd: shape must be equal to  self's dimC.")
  auto options = self.options().dtype(at::kFloat);
  Tensor weight_ = weight.defined() ? weight : ones_npu({dimC}, options);
  Tensor bias_ = bias.defined() ? bias : ones_npu({dimC}, options);
  Tensor mean_ = mean.defined() ? mean : ones_npu({dimC}, options);
  Tensor invstd_ = invstd.defined() ? invstd : ones_npu({dimC}, options);
  Tensor one = at::ones({1}, options);
  auto variance = at::mul(invstd_, invstd_);
  variance = at::div(one, variance) - eps;
  int64_t selfDim = self.dim();
  Tensor self5d(self);
  SmallVector<int64_t, N> selfShape = array_to_small_vector(self.sizes());
  if (selfDim > 5) {
    self5d = self.reshape({self.size(0), self.size(1), self.size(2), self.size(3), -1});
  }
  std::tuple<Tensor, Tensor, Tensor> bnReult = batch_norm_npu(self5d, weight_, bias_, mean_, variance, false, 0.0, eps);
  result.copy_(std::get<0>(bnReult));
  if (selfDim > 5) {
    result = result.view(selfShape);
  }
  return result;
}

Tensor& batch_norm_elemt_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& mean,
    const Tensor& invstd,
    double eps) {
  OpPreparation::CheckOut({self}, result, self);
  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self, weight, bias, mean, invstd}, {result})
        .Func([&self, &weight, &bias, &mean, &invstd, &eps](Tensor& result)
        {batch_norm_elemt_nocheck(result, self, weight, bias, mean, invstd, eps);})
        .Call(result);
}

Tensor batch_norm_elemt_npu(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& mean,
    const Tensor& invstd,
    double eps) {
  Tensor result = OpPreparation::ApplyTensor(self);
  batch_norm_elemt_nocheck(result, self, weight, bias, mean, invstd, eps);
  return result;
}
} // namespace native
} // namespace at