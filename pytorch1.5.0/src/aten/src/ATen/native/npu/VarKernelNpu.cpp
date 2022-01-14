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
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

auto check_and_trans_dim(const Tensor& self, IntArrayRef dim) {
  int64_t dim_size = self.dim();
  int64_t ne_dim_size = dim_size * -1;
  std::vector<int64_t> result_dim;
  for(int64_t i = 0; i < dim.size(); i++) {
    if(dim[i] >= ne_dim_size && dim[i] <= (dim_size - 1)) {
      int64_t tmp_dim = CalcuOpUtil::make_wrap_dim(dim[i], self.dim());
      result_dim.emplace_back(tmp_dim);
    } else {
      AT_ERROR("dim value should be in the range of [-n, n-1], n is the dimension number of input tensor.");
    }
  }
  std::sort(result_dim.begin(), result_dim.end());
  return result_dim;
}

auto get_result_names(const Tensor& self, IntArrayRef dim, bool keepdim){
  auto names = self.names();
  std::vector<Dimname> result_names;
  for(int64_t i = 0; i < names.size(); i++){
    result_names.emplace_back(names[i]);
  }
  if(!keepdim){
    for(int64_t i = dim.size() - 1; i >= 0; i--){
      int64_t need_remove_dim = dim[i];
      result_names.erase(result_names.begin() + need_remove_dim);
    }
  }
  return result_names;
}

Tensor& var_after_npu_nocheckout(
    Tensor& var,
    const Tensor& self,
    const Tensor& mean_broadcast,
    IntArrayRef dim,
    bool unbiased,
    bool keepdim) {
  bool if_std = false;
  OpCommand cmd;
  cmd.Name("ReduceStdV2Update")
     .Input(self)
     .Input(mean_broadcast)
     .Output(var)
     .Attr("dim", dim)
     .Attr("if_std", if_std)
     .Attr("unbiased", unbiased)
     .Attr("keepdim", keepdim)
     .Run();
  return var;
}

tuple<Tensor&, Tensor&> var_mean_compute(
    Tensor& variance,
    Tensor& mean,
    const Tensor& self,
    IntArrayRef dim,
    bool unbiased,
    bool keepdim) {
  auto meanOutputSizeKeepDim = var_npu_output_size(self, dim, true);
  auto meanOutputSizeNotKeepDim = var_npu_output_size(self, dim, false);
  mean = mean_npu(self, dim, false);
  mean.resize_(meanOutputSizeKeepDim);
  Tensor mean_broadcast = at::npu_broadcast(mean, self.sizes());
  if(!keepdim){
    mean.resize_(meanOutputSizeNotKeepDim);
  }
  var_after_npu_nocheckout(variance, self, mean_broadcast, dim, unbiased, keepdim);
  return tuple<Tensor&, Tensor&>(variance, mean);
}

tuple<Tensor&, Tensor&> var_mean_out_npu(
    Tensor& variance,
    Tensor& mean,
    const Tensor& self,
    IntArrayRef dim,
    bool unbiased,
    bool keepdim) {
  auto dim_now = check_and_trans_dim(self, dim);
  auto meanOutputSizeKeepDim = var_npu_output_size(self, dim_now, true);
  auto meanOutputSizeNotKeepDim = var_npu_output_size(self, dim_now, false);
  auto ori_type = self.scalar_type();
  if(ori_type != c10::ScalarType::Half && ori_type != c10::ScalarType::Float) {
    AT_ERROR("Var Mean only support float16 or float32 type.");
  }
  if(variance.scalar_type() != mean.scalar_type() || variance.scalar_type() != ori_type) {
    AT_ERROR("mean's type and variance' type must be equal to input's type.");
  }
    var_mean_compute(
        variance,
        mean,
        self,
        dim_now,
        unbiased,
        keepdim);
 
  return tuple<Tensor&, Tensor&>(variance, mean);
}

Tensor& var_out_npu(
    Tensor& var,
    const Tensor& self,
    IntArrayRef dim,
    bool unbiased,
    bool keepdim) {
  // check and trans dim
  auto dim_now = check_and_trans_dim(self, dim);
  auto outputSize = var_npu_output_size(self, dim_now, keepdim);

  // construct the output mean tensor of the NPU
  Tensor mean = OpPreparation::ApplyTensor(self, outputSize);
  Tensor var_ = OpPreparation::ApplyTensor(self, outputSize);
  
  var_mean_out_npu(var_, mean, self, dim, unbiased, keepdim);
  OpPreparation::CheckOut(
      {var_},
      var,
      var_);
      var.copy_(var_);
   return var;
}

Tensor& var_out_npu(
    Tensor& var,
    const Tensor& self,
    DimnameList dim,
    bool unbiased,
    bool keepdim) {
  return var_out_npu(
      var, self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

Tensor var_npu(const Tensor& self, bool unbiased) {
  bool keepdim = false;
  SmallVector<int64_t, N> dim = CalcuOpUtil::get_dimlist_for_tensor(self);

  return var_npu(self, dim, unbiased, keepdim);
}

Tensor var_npu(
    const Tensor& self,
    IntArrayRef dim,
    bool unbiased,
    bool keepdim) {
  auto dim_now = check_and_trans_dim(self, dim);
  // calculate the output size
  auto outputSize = var_npu_output_size(self, dim_now, keepdim);

  // construct the output tensor of the NPU
Tensor variance = OpPreparation::ApplyTensor(self, outputSize);

  // calculate the output result of the NPU
  var_out_npu(variance, self, dim, unbiased, keepdim);

  return variance;
}

Tensor var_npu(
    const Tensor& self,
    DimnameList dim,
    bool unbiased,
    bool keepdim) {
  return var_npu(self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

Tensor _var_npu(const Tensor& self, bool unbiased) {
  return at::var(self, unbiased);
}

tuple<Tensor, Tensor> var_mean_npu(
    const Tensor& self,
    DimnameList dim,
    bool unbiased,
    bool keepdim) {
  return var_mean_npu(self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

tuple<Tensor, Tensor> var_mean_npu(
    const Tensor& self,
    IntArrayRef dim,
    bool unbiased,
    bool keepdim) {
  auto dim_now = check_and_trans_dim(self, dim);
  // calculate the output size
  auto outputSize = var_npu_output_size(self, dim_now, keepdim);

  // construct the output tensor of the NPU
  Tensor variance = OpPreparation::ApplyTensor(self, outputSize);
 
  Tensor mean = OpPreparation::ApplyTensor(self, outputSize);
  
  // calculate the output result of the NPU
  var_mean_out_npu(variance, mean, self, dim, unbiased, keepdim);

  return tuple<Tensor, Tensor>(variance, mean);
}

tuple<Tensor, Tensor> var_mean_npu(const Tensor& self, bool unbiased) {
  SmallVector<int64_t, SIZE> dim = CalcuOpUtil::get_dimlist_for_tensor(self);

  return var_mean_npu(self, dim, unbiased, false);
}

} // namespace native
} // namespace at