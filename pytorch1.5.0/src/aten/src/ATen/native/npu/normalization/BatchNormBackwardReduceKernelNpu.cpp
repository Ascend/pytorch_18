
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

std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> batch_norm_backward_reduce_npu_impl(
    Tensor& sum_dy,
    Tensor& sum_dy_xmu,
    Tensor& grad_weight,
    Tensor& grad_bias,
    const Tensor& grad_out,
    const Tensor& self,
    const Tensor& mean,
    const Tensor& invstd,
    const Tensor& weight,
    bool input_g,
    bool weight_g,
    bool bias_g) {
  Tensor sum_dy_;
  Tensor sum_dy_xmu_;
  Tensor grad_weight_;
  Tensor grad_bias_;
  SmallVector<int64_t, N> axes;
  int dimN = self.ndimension();
  for(int i = 0; i < dimN; i++){
    if (i == 1) {
      continue;
    }
    axes.emplace_back(i);
  }
  // sum_dy_xmu
  Tensor mul_dy_dx = grad_out * self;
  sum_dy_xmu_ = at::sum(mul_dy_dx, axes, false);
  // sum_dy
  auto meanLen = mean.size(0);
  auto sumDyDxLen = sum_dy_xmu_.size(0);
  int64_t pad_dy = meanLen - sumDyDxLen;

  SmallVector<int64_t, N> nedps = {0, pad_dy};
  IntArrayRef need_pad(nedps);
  Tensor sum_dy_dx_pad = at::npu_pad(sum_dy_xmu_, need_pad);
  // grad_bais:
  grad_bias_ = at::sum(grad_out, axes, false);
  sum_dy_ = at::npu_pad(grad_bias_, need_pad);
  // grad_weight
  Tensor sum_dy_xmu_out = OpPreparation::ApplyTensor(sum_dy_);
  Tensor grad_weight_res = OpPreparation::ApplyTensor(invstd);
  OpCommand cmd;
  cmd.Name("SyncBatchNormBackwardReduce")
      .Input(sum_dy_)
      .Input(sum_dy_dx_pad)
      .Input(mean)
      .Input(invstd)
      .Output(sum_dy_xmu_out)
      .Output(grad_weight_res)
      .Run();
  if (weight_g){
    int64_t grad_biasLen = grad_bias_.size(0);
    grad_weight.resize_({grad_biasLen});
    grad_weight_ = grad_weight_res.slice(0, 0, grad_biasLen);
  }
  if (input_g){
    sum_dy_xmu.copy_(sum_dy_xmu_out);
    sum_dy.copy_(sum_dy_);
  }
  if (weight_g) {
    grad_weight.copy_(grad_weight_);
  }
  if (bias_g) {
    grad_bias.copy_(grad_bias_);
  }

  return std::tie(sum_dy, sum_dy_xmu, grad_weight, grad_bias);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> batch_norm_backward_reduce_npu(
    const Tensor& grad_out,
    const Tensor& self,
    const Tensor& mean,
    const Tensor& invstd,
    const Tensor& weight,
    bool input_g,
    bool weight_g,
    bool bias_g) {
  int64_t n_input = self.size(1);
  Tensor sum_dy_;
  Tensor sum_dy_xmu_;
  Tensor grad_weight_;
  Tensor grad_bias_;

  Tensor weight_ = weight.defined() ? weight : ones_npu({n_input}, self.options().dtype(at::kFloat));

  if (input_g) {
      sum_dy_ = OpPreparation::ApplyTensor(mean);
      sum_dy_xmu_ = OpPreparation::ApplyTensor(mean);
  }
  if (weight_g) {
      grad_weight_ = OpPreparation::ApplyTensor(weight_, {n_input});
  }
  if (bias_g) {
      grad_bias_ = OpPreparation::ApplyTensor(weight_, {n_input});
  }
  batch_norm_backward_reduce_npu_impl(
      sum_dy_,
      sum_dy_xmu_,
      grad_weight_,
      grad_bias_,
      grad_out,
      self,
      mean,
      invstd,
      weight,
      input_g,
      weight_g,
      bias_g);
  return std::tie(sum_dy_, sum_dy_xmu_, grad_weight_, grad_bias_);
}

} // namespace native
} // namespace at