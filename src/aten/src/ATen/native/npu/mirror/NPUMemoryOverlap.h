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


#include <ATen/ATen.h>

namespace at { namespace native { namespace npu {

// MemOverlap: Whether or not there is memory overlap
//
// NO: Absolutely no memory overlap
// YES: Absolutely yes memory overlap
// TOO_HARD: There might be memory overlap, but it was too expensive to compute.
//
// NB: Please update the python test for these if you renumber them.
enum class MemOverlap { NO, YES, TOO_HARD };
enum class MemOverlapStatus { FULL, PARTIAL, NO, TOO_HARD };

MemOverlap has_internal_overlap(const Tensor& t);
MemOverlap has_internal_overlap(TensorImpl* t);

void assert_no_internal_overlap(const Tensor& t);
void assert_no_internal_overlap(TensorImpl* t);

MemOverlapStatus get_overlap_status(const Tensor& a, const Tensor& b);
MemOverlapStatus get_overlap_status(TensorImpl* a, TensorImpl* b);

void assert_no_partial_overlap(const Tensor& a, const Tensor& b);
void assert_no_partial_overlap(TensorImpl* a, TensorImpl* b);

}}}