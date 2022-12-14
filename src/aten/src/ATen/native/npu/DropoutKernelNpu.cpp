// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/NpuUtils.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/OpTemplate.h"
#include "c10/npu/SecondaryStreamGuard.h"
#include "c10/npu/NPUCachingAllocator.h"

namespace at {
namespace native {
using namespace at::native::npu;
Tensor dropout_do_mask(
    Tensor& result,
    const Tensor& self,
    const Tensor& mask,
    const Tensor& prob) {
  OpCommand cmd;
  cmd.Name("DropOutDoMask")
      .Input(self)
      .Input(mask)
      .Input(prob)
      .Output(result)
      .Run();

  return result;
}

Tensor dropout_gen_mask(const Tensor& self, const Tensor& prob) {
  uint32_t length = (self.numel() + 128 - 1) / 128 * 128;
  Tensor mask = at::empty_with_format(
      {length / 8},
      self.options().dtype(at::kByte),
      CalcuOpUtil::get_tensor_npu_format(self));

  Tensor cpu_shape =
      from_blob((void*)self.sizes().data(), {self.dim()}, at::kLong)
          .to(at::kInt);
  Tensor npu_shape = CalcuOpUtil::copy_tensor_host_to_device(cpu_shape);

  OpCommand cmd;
  // If either seed or seed2 are set to be non-zero, the random number generator
  // is seeded by the given seed. Otherwise, it is seeded by a random seed.
  int64_t seed = 0;
  int64_t seed2 = 0;
  cmd.Name("DropOutGenMask")
      .InputPair(npu_shape, /*cpu_input=*/cpu_shape)
      .Input(prob)
      .Output(mask)
      .Attr("seed", seed)
      .Attr("seed2", seed2)
      .Run();
  return mask;
}

std::tuple<Tensor, Tensor> dropout_v1_npu_impl(
    Tensor result,
    const Tensor& self,
    double p) {
  TORCH_CHECK(
      p >= 0 && p <= 1,
      "dropout probability has to be between 0 and 1, but got ",
      p);
  TORCH_CHECK(
      at::isFloatingType(self.scalar_type()),
      "dropout only supports floating-point dtypes");

  // dropout only supports NCHW foramt(aicpu restriction)
  Tensor selfFormatCast = self.npu_format_cast(ACL_FORMAT_ND);
  
  double retain = 1. - p;
  Tensor prob;
  Tensor mask;
  auto original_stream = c10::npu::getCurrentNPUStream();
  {
    // During the life cycle of this raii instance, the calcu stream is set as the
    // secondary stream, and tasks are distributed to the secondary stream. At the 
    // same time, according to the one-stream-one-pool principle, memory is also
    // alloced from the pool of the secondary stream.
    c10::npu::SecondaryStreamGuard guard(c10::npu::getCurrentSecondaryStream());
    prob = CalcuOpUtil::CopyScalarToDevice(retain, selfFormatCast.scalar_type());
    mask = dropout_gen_mask(selfFormatCast, prob);
  }
  // When tasks on multiple streams read and write the same block of memory,
  // recordStream needs to be called to ensure the correctness of memory reuse.
  c10::npu::NPUCachingAllocator::recordStream(prob.storage().data_ptr(), original_stream);
  c10::npu::NPUCachingAllocator::recordStream(mask.storage().data_ptr(), original_stream);
  dropout_do_mask(result, selfFormatCast, mask, prob);

  return std::tie(result, mask);
}
std::tuple<Tensor, Tensor> _dropout_npu(
    const Tensor& self,
    double p) {
  Tensor selfFormatCast = self.npu_format_cast(ACL_FORMAT_ND); 
  auto outputSize = input_same_output_size(selfFormatCast);
  Tensor result = at::empty_with_format(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(selfFormatCast));
  return dropout_v1_npu_impl(result, selfFormatCast, p);
}

std::tuple<Tensor, Tensor> _dropout_npu_inplace(
    Tensor& self,
    double p) {
  return dropout_v1_npu_impl(self, self, p);
}

Tensor dropout_npu(const Tensor& self, double p, bool train) {
  if (p == 0 || !train || self.numel() == 0) {
    return self;
  }
  if (p == 1) {
    return self.mul(at::zeros(self.sizes(), self.options()));
  }
  Tensor result = std::get<0>(at::_npu_dropout(self, p));
  return result;
}

Tensor& dropout_npu_(Tensor& self, double p, bool train) {
  if (p == 0 || !train || self.numel() == 0) {
    return self;
  }
  if (p == 1) {
    return self.mul_(at::zeros(self.sizes(), self.options()));
  }
  if (!NpuUtils::check_match(&self)) {
    Tensor result = NpuUtils::format_contiguous(self);
    at::_npu_dropout_inplace(result, p);
    NpuUtils::format_fresh_view(self, result);
  } else {
    at::_npu_dropout_inplace(self, p);
  }
  return self;
}

} // namespace native
} // namespace at