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

Tensor& ps_roi_pooling_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    const Tensor& rois,
    double spatial_scale,
    int64_t group_size,
    int64_t output_dim) {
  OpCommand cmd;
  cmd.Name("PSROIPoolingV2")
      .Input(self)
      .Input(rois)
      .Output(result)
      .Attr("spatial_scale", (float)spatial_scale)
      .Attr("output_dim", output_dim)
      .Attr("group_size", group_size)
      .Run();

  return result;
}

Tensor ps_roi_pooling_npu(
    const Tensor& self,
    const Tensor& rois,
    double spatial_scale,
    int64_t group_size,
    int64_t output_dim) {
  auto outputSize ={
      rois.size(0) * rois.size(2), output_dim, group_size, group_size};

  Tensor result = OpPreparation::ApplyTensor(self, outputSize);

  ps_roi_pooling_npu_nocheck(
      result,
      self,
      rois,
      spatial_scale,
      group_size,
      output_dim);

  return result;
}

TORCH_LIBRARY_IMPL(aten, NPU, m){
  m.impl("npu_ps_roi_pooling", TORCH_FN(ps_roi_pooling_npu));
}

} // namespace native
} // namespace at