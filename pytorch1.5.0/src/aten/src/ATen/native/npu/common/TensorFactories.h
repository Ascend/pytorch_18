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

#pragma once

#include <c10/core/TensorOptions.h>

namespace at {
namespace native {

inline void check_size_nonnegative(IntArrayRef& size) {
  for (auto& x : size) {
    TORCH_CHECK(
        x >= 0,
        "Trying to create tensor with negative dimension ",
        x,
        ": ",
        size);
  }
}

inline void check_args(
    int64_t row, int64_t col, const TensorOptions& options) {
  TORCH_CHECK(row >= 0, "row must be non-negative, got", row);
  TORCH_CHECK(col >= 0, "col must be non-negative, got", col);
  if (options.has_layout()) {
    TORCH_CHECK(
        options.layout() == at::kStrided,
        "only support layout=torch.strided, got",
        options.layout());
  }
}

inline int64_t get_tril_size(int64_t row, int64_t col, int64_t offset) {
  // number of elements in the first row of the tril
  auto m_first_row = offset > 0 ?
    std::min<int64_t>(col, 1 + offset) : // upper bounded by col
    row + offset > 0; // either 0 or 1
  // number of elements in the last row of the tril, bounded by [0, col]
  auto m_last_row = std::max<int64_t>(0, std::min<int64_t>(col, row + offset));
  // number of rows, bounded by [0, row]
  auto n_row_all = std::max<int64_t>(0, std::min<int64_t>(row, row + offset));
  auto n_row_trapezoid = (m_last_row - m_first_row + 1);

  // calculate # of elements in the top trapezoid
  auto tril_size = (m_first_row + m_last_row) * n_row_trapezoid >> 1;

  // calculate # of elements in the bottom rectangle if there is any
  auto diff_row = n_row_all - n_row_trapezoid;
  if (diff_row > 0) {
    tril_size += diff_row * col;
  }

  return tril_size;
}

} // namespace native
} // namespace at
