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

tuple<Tensor, Tensor> ifmr_npu(
    const Tensor& data,
    const Tensor& data_min,
    const Tensor& data_max,
    const Tensor& cumsum,
    const double min_percentile=0.999999,
    const double max_percentile=0.999999,
    const double search_start=0.7,
    const double search_end=1.3,
    const double search_step=0.01,
    const bool with_offset=true) {
  Tensor scale = at::empty_with_format(
      data_min.sizes(), data_min.options(), ACL_FORMAT_NCHW);
  Tensor offset = at::empty_with_format(
      data_min.sizes(), data_min.options(), ACL_FORMAT_NCHW);

  std::vector<float> tmp;
  tmp.push_back(static_cast<float>(search_start));
  tmp.push_back(static_cast<float>(search_end));
  at::ArrayRef<float> searchRange(tmp);

  OpCommand cmd;
  cmd.Name("IFMR")
      .Input(data)
      .Input(data_min)
      .Input(data_max)
      .Input(cumsum)
      .Attr("min_percentile", static_cast<float>(min_percentile))
      .Attr("max_percentile", static_cast<float>(max_percentile))
      .Attr("search_range", searchRange)
      .Attr("search_step", static_cast<float>(search_step))
      .Attr("with_offset", with_offset)
      .Output(scale)
      .Output(offset)
      .Run();

  return std::tie(scale, offset);
}

} // namespace native
} // namespace at
