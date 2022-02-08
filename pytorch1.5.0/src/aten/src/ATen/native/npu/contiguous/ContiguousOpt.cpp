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

#include <ATen/native/npu/contiguous/ContiguousOpt.h>

namespace at {
namespace native {
namespace npu {
OptimizationCases TransContiguous::optCasesDefault = {};

OptimizationCases TransContiguous::optCasesAnyFormat = {
    "reshape",
    "slice"};

ContiguousTensorDesc TransContiguous::GetTensorDescInfo(
    const Tensor& src,
    const OptimizationCases& opt_cases) {
  NPUStorageDesc src_base_info = src.storage().get_npu_desc();
  SmallVector<int64_t, MAX_DIM> src_size_inferred;
  SmallVector<int64_t, MAX_DIM> src_stride_inferred;
  SmallVector<int64_t, MAX_DIM> src_storage_size_inferred = src_base_info.storage_sizes_;
  if (src.dim() == 0) {
    // torch.tensor([x]).[0] create a tensor with 0 dim.
    TORCH_WARN(
        "Warning: You have sliced a Tensor of single element, we recommend not performing such operation to avoid data copying!");
    src_size_inferred = {1};
    src_stride_inferred = {1};
    if (src_storage_size_inferred.size() == 0) {
      src_storage_size_inferred = {1};
    }
  } else {
    src_size_inferred = array_to_small_vector(src.sizes());
    src_stride_inferred = array_to_small_vector(src.strides());  
  }
  ContiguousTensorDesc src_desc = {
      src.is_contiguous(),
      src_size_inferred,
      src_stride_inferred,
      src.storage_offset(),
      src_base_info.base_sizes_,
      src_base_info.base_strides_,
      src_storage_size_inferred,
      src_base_info.base_offset_,
      src_base_info.npu_format_,
      opt_cases};
  if (src_desc.opt_cases_.empty()) {
    src_desc.find_match_optimization_cases();
  }
  return src_desc;
}

bool TransContiguous::CheckClone(const Tensor& src, Tensor& self) {
  // self tensor may not be temporary constructed empty tensor from src, so:
  // 1. contiguous storage is needed:storage_offset and numels eq
  // 2. full memory copy: size match between src and self
  if (StorageDescHelper::OffsetAreMatch(&self) && self.is_contiguous() &&
      src.sizes().equals(self.sizes()) &&
      self.sizes().equals(self.storage().get_npu_desc().base_sizes_)) {
    return true;
  }
  return false;
}

bool TransContiguous::can_optimize_(
    ContiguousTensorDesc& tensor_desc) {
  for (auto opt_case : tensor_desc.opt_cases_) {
    bool res =
        register_opt::CopyOptRegister::GetInstance()->CanOptimize(opt_case, tensor_desc);
    if (res) {
      // refresh patterns to only keep optimized pattern
      tensor_desc.opt_cases_.clear();
      tensor_desc.opt_cases_.emplace_back(opt_case);
      return true;
    }
  }
  return false;
}

bool TransContiguous::CanOptimize(
    ContiguousTensorDesc& tensor_desc) {
  return can_optimize_(tensor_desc);
}

bool TransContiguous::CanOptimize(
    const Tensor& tensor,
    const OptimizationCases& opt_cases) {
  ContiguousTensorDesc tensor_desc = GetTensorDescInfo(tensor, opt_cases);
  return can_optimize_(tensor_desc);
}

bool TransContiguous::contiguous_optimize_with_anyformat_(
    Tensor& self,
    const Tensor& src,
    ContiguousTensorDesc& src_desc) {
  if (!CheckClone(src, self)) {
    return false;
  }
  for (auto& opt_case : src_desc.opt_cases_) {
    bool res =
        register_opt::CopyOptRegister::GetInstance()->Run(opt_case, self, src, src_desc);
    if (res) {
      return true;
    }
  }
  return false;
}

bool TransContiguous::ContiguousOptimizeWithAnyFormat(
    Tensor& self,
    const Tensor& src,
    const OptimizationCases& opt_cases) {
  ContiguousTensorDesc src_desc = GetTensorDescInfo(src, opt_cases);
  return contiguous_optimize_with_anyformat_(self, src, src_desc);
}

c10::optional<Tensor> TransContiguous::ContiguousOptimizeWithAnyFormat(
    const Tensor& src,
    const OptimizationCases& opt_cases) {
  auto self = at::native::empty_with_format_npu(
      src.sizes(), src.options(), src.storage().get_npu_desc().npu_format_);
  ContiguousTensorDesc src_desc = GetTensorDescInfo(src, opt_cases);
  if (contiguous_optimize_with_anyformat_(self, src, src_desc)) {
    return self;
  }
  return c10::nullopt;
}

bool TransContiguous::ContiguousOptimizeWithBaseFormat(
    Tensor& self,
    const Tensor& src,
    const OptimizationCases& opt_cases,
    bool OpenCombined) {
  TORCH_CHECK(
      FormatHelper::IsBaseFormatType(src),
      "ContiguousOptimizeWithBaseFormat func requires Input Tensor with base format!");
  // In non-specific cases, classify the cases and simplify judgement.
  ContiguousTensorDesc src_desc = GetTensorDescInfo(src, opt_cases);
  if (OpenCombined &&
      c10::npu::OptionsManager::CheckCombinedOptimizerEnable()) {
    src_desc.add_optimization_case("combined");
  }
  return contiguous_optimize_with_anyformat_(self, src, src_desc);
}

} // namespace npu
} // namespace native
} // namespace at