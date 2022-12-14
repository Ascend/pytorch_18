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

#include <limits.h>
#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;
namespace {
  constexpr int MAX_ERROR_OF_FP32_TO_FP16 = 16;
}

Tensor& random_out_npu(Tensor& result, Tensor& self, int64_t from, int64_t to, Generator* gen_) {
  OpCommand cmd;
  cmd.Name("Random")
       .Input(self)
       .Output(result)
       .Attr("from", from)
       .Attr("to", to)
       .Run();
  return result;
}

Tensor& random_npu_(Tensor& self, int64_t from, int64_t to, Generator* gen_) {
  Tensor selfCopy = self;
  if (self.scalar_type() == ScalarType::Half) {
    selfCopy = self.npu_dtype_cast(ScalarType::Float);
  }

  OpPreparation::CheckMemory({selfCopy}, {selfCopy});

  if (!NpuUtils::check_match(&selfCopy)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(selfCopy);
    Tensor result = random_out_npu(contiguousSelf, contiguousSelf, from, to, gen_);
    NpuUtils::format_fresh_view(selfCopy, result);
  } else {
    random_out_npu(selfCopy, selfCopy, from, to, gen_);
  }
  self.copy_(selfCopy);
  return self;
}

Tensor& random_npu_(Tensor& self, int64_t from, c10::optional<int64_t> to, Generator* gen_) {
  int64_t to_ = to.value();

  random_npu_(self, from, to_, gen_);

  // fp32 casting to fp16 will introduce error, so needing to counteract it.
  if (self.scalar_type() == ScalarType::Half) {
    self = at::where(self == to_, self - MAX_ERROR_OF_FP32_TO_FP16, self);
  }

  return self;
}

Tensor& random_npu_(Tensor& self, int64_t to, Generator* gen_) {
  int64_t from = 0;

  random_npu_(self, from, to, gen_);

  // fp32 casting to fp16 will introduce error, so needing to counteract it.
  if (self.scalar_type() == ScalarType::Half) {
    self = at::where(self == to, self - MAX_ERROR_OF_FP32_TO_FP16, self);
  }

  return self;
}

Tensor& random_npu_(Tensor& self, Generator* gen_) {
  // Check the dtype of input
  TORCH_CHECK(
      self.dtype() == at::kHalf ||
      self.dtype() == at::kFloat ||
      self.dtype() == at::kInt ||
      self.dtype() == at::kLong,
      "the dtype of input must be float16, float32, int32, int64");
  
  int64_t from = 0;
  int64_t to = 1;
  
  if (self.dtype() == at::kHalf) {
    to = NPU_HALF_MAX;
  } else if (self.dtype() == at::kInt) {
    to = INT_MAX;
  } else if (self.dtype() == at::kLong || self.dtype() == at::kFloat) {
    // the max of 'to' is also LONG_MAX because to's dtype is int64 though self is of fp32
    to = LONG_MAX;
  } 

  random_npu_(self, from, to, gen_);

  return self;
}
} // namespace native
} // namespace at