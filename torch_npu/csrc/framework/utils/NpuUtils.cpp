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

#include <mutex>
#include <set>

#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/framework/StorageDescHelper.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/contiguous/ContiguousOpt.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"

namespace at_npu
{
  namespace native
  {

    void NpuUtils::format_fresh_view(
        at::Tensor &x,
        const at::Tensor &y)
    {
      // x:NPU before inplace_op, y: NPU computed
      // now we fresh x according to y
      RECORD_FUNCTION("format_fresh_view", vector<c10::IValue>({x, y}));
      x.copy_(y);
    }

    // NOTE [Check Match for Npu at::Tensor]
    // check_match is used to ensure that npu tensor satisfies the
    // calculation requirements of npu operators.
    // The rules are as follows,
    // 1、tensor should be contiguous
    // Not contiguous means the operator needs to read and write memory
    // at intervals according to strides and sizes. Npu operators has
    // no such ability for the time being
    // 2、metadata should be match
    // Resize_ a contiguous cpu tensor from [1,2,3,4] to [4,3,2,1] no
    // need to change the physical memory. However, for a contiguous npu
    // tensor whose npu_format_ is 5HD, storage shape should be change
    // from [1,1,3,4,16] to [4,1,2,1,16]. So metadata not match often
    // results in unexpected physical memory. format_contiguous will be
    // called preparing correct memory of operand in these case.
    bool NpuUtils::check_match(const at::Tensor *tensor)
    {
      // case1:uncontiguous tensor
      if (!tensor->is_contiguous())
      {
        return false;
      }

      // case2:meta data not match, sizes or strides of presentation
      // layer is different from that of storage layer
      if (!StorageDescHelper::MetaDataAreMatch(tensor))
      {
        return false;
      }

      // case3:meta data not match, storage_offset of presentation layer
      // is different from that of storage layer
      bool isPadding = FormatHelper::IsPadded(tensor);
      if (isPadding && (!StorageDescHelper::OffsetAreMatch(tensor)))
      {
        return false;
      }
      return true;
    }

    bool NpuUtils::check_5d_5d_match(const at::Tensor &tensor)
    {
      // (1) NC1HWC0 format in storage, NCHW format in des.
      // (2) 4d format situation, only uncontiguous in Channel size
      // (3) size and start point must be 16*, make sure the memory be contiguous
      const c10::Storage storage = tensor.storage();
      const torch_npu::NPUStorageDesc npuDesc = torch_npu::NPUBridge::GetNpuStorageImpl(storage.unsafeGetStorageImpl())->get_npu_desc();

      if (tensor.is_contiguous())
      {
        return false;
      }

      if (npuDesc.npu_format_ != ACL_FORMAT_NC1HWC0)
      {
        return false;
      }

      if (tensor.sizes().size() != 4)
      {
        return false;
      }

      bool is_c_channel_slice = true;
      int64_t z = 1;
      for (int64_t d = tensor.dim() - 1; d >= 1; d--)
      {
        if (tensor.size(d) != 1)
        {
          if (tensor.stride(d) == z)
          {
            z *= tensor.size(d);
          }
          else
          {
            is_c_channel_slice = false;
            break;
          }
        }
      }
      if (!is_c_channel_slice)
      {
        return false;
      }

      int64_t contiguous_len = 16;
      int64_t c0_len = 16;
      for (int i = 2; i < npuDesc.base_sizes_.size(); i++)
      {
        contiguous_len *= npuDesc.base_sizes_[i];
      }
      bool is_offset_match = (tensor.storage_offset() % contiguous_len == 0);
      bool is_length_match = (tensor.size(1) % c0_len == 0);

      return is_offset_match && is_length_match;
    }

    void NpuUtils::RefreshFormat(const at::Tensor &tensor)
    {
      auto &tensor_desc = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_;
      if (tensor_desc.storage_sizes_.size() == 4 && tensor_desc.npu_format_ == ACL_FORMAT_ND)
      {
        tensor_desc.npu_format_ = ACL_FORMAT_NCHW;
        tensor_desc.origin_format_ = ACL_FORMAT_NCHW;
      }
      else if (tensor_desc.storage_sizes_.size() != 4 && tensor_desc.npu_format_ == ACL_FORMAT_NCHW)
      {
        tensor_desc.npu_format_ = ACL_FORMAT_ND;
        tensor_desc.origin_format_ = ACL_FORMAT_ND;
      }
    }

    at::Tensor metadata_convert_match(const at::Tensor &src)
    {
      auto &src_desc = torch_npu::NPUBridge::GetNpuStorageImpl(src)->npu_desc_;
      bool numelEq = (src.numel() == at::prod_intlist(src_desc.base_sizes_));
      // Only when a tensor monopolizes a storage can NpuStorageDesc be refreshed.
      // When the original format is not NCHW, the npu_format_cast to NCHW will generate
      // a temporary tensor, which always monopolizes its own storage.
      if (numelEq && (!FormatHelper::IsBaseFormatType(src)))
      {
        at::Tensor tempTensor = NPUNativeFunctions::npu_format_cast(src, FormatHelper::GetBaseFormat(src));
        auto &temp_desc =
            torch_npu::NPUBridge::GetNpuStorageImpl(tempTensor)->npu_desc_;
        temp_desc.base_sizes_ = tempTensor.sizes();
        temp_desc.base_strides_ = tempTensor.strides();
        temp_desc.storage_sizes_ = tempTensor.sizes();
        NpuUtils::RefreshFormat(tempTensor);
        return tempTensor;
      }
      else
      {
        at::Tensor contiguous_view = at::empty(src.sizes(), src.options());
        contiguous_view.copy_(src);
        NpuUtils::RefreshFormat(contiguous_view);
        return contiguous_view;
      }
    }

    at::Tensor metadata_with_offset_padding_convert_match(const at::Tensor &src)
    {
      at::Tensor contiguous_view = at::empty(src.sizes(), src.options());
      contiguous_view.copy_(src);
      NpuUtils::RefreshFormat(contiguous_view);
      return contiguous_view;
    }

    at::Tensor NpuUtils::format_contiguous(const at::Tensor &src)
    {
      // case1:tensor src is not contiguous
      if (!src.is_contiguous())
      {
        RECORD_FUNCTION("format_contiguous", vector<c10::IValue>({src}));
        return src.contiguous();
      }
      // case2:meta data not match, sizes or strides of presentation
      // layer is different from that of storage layer
      if (!StorageDescHelper::MetaDataAreMatch(&src))
      {
        // Fix not match case2, tensor should have matched metadata and NPUStorageDesc.
        RECORD_FUNCTION("format_contiguous", vector<c10::IValue>({src}));
        return metadata_convert_match(src);
      }

      // case3:meta data not match, storage_offset of presentation layer
      // is different from that of storage layer
      if (FormatHelper::IsPadded(&src) && (!StorageDescHelper::OffsetAreMatch(&src)))
      {
        // Fix not match case3, tensor with padding should not have storage-offset.
        RECORD_FUNCTION("format_contiguous", vector<c10::IValue>({src}));
        return metadata_with_offset_padding_convert_match(src);
      }

      return src;
    }

    at::Tensor NpuUtils::format_contiguous_add_copy_optimize(const at::Tensor &src)
    {
      // case1:tensor src is not contiguous
      if (!src.is_contiguous())
      {
        RECORD_FUNCTION("format_contiguousV2", vector<c10::IValue>({src}));
        return src.contiguous();
      }
      // case2:meta data not match, sizes or strides of presentation
      // layer is different from that of storage layer
      if (!StorageDescHelper::MetaDataAreMatch(&src))
      {
        // Fix not match case2, tensor should have matched metadata and NPUStorageDesc.
        RECORD_FUNCTION("format_contiguousV2", vector<c10::IValue>({src}));
        // copy optimize for reshape cases with 3 choices
        // [1] memory-repoint: base format or NZ[1. key dims keep matched; 2. no padding]
        // [2] d2dCopyAsync: base format or NZ[key dims keep matched]
        // [3] copy_: Universal method
        static OptimizationCases optimizations_reshape{"reshapeV2"};
        auto reshapeTensor =
            TransContiguous::ContiguousOptimizeWithAnyFormat(src, optimizations_reshape);
        if (reshapeTensor.has_value())
        {
          return reshapeTensor.value();
        }
        return metadata_convert_match(src);
      }

      // case3:meta data not match, storage_offset of presentation layer
      // is different from that of storage layer
      if (FormatHelper::IsPadded(&src) && (!StorageDescHelper::OffsetAreMatch(&src)))
      {
        // Fix not match case3, tensor with padding should not have storage-offset.
        RECORD_FUNCTION("format_contiguousV2", vector<c10::IValue>({src}));
        return metadata_with_offset_padding_convert_match(src);
      }

      return src;
    }

    bool NpuUtils::IsOomError(aclError ret, int index)
    {
      if (ret == ACL_ERROR_GE_DEVICE_MEMORY_ALLOCATION_FAILED)
      {
        int deviceId = 0;
        // free devcie cached memory when return value of the first op execution is oom
        if (index == 1)
        {
          C10_NPU_CHECK(aclrtGetDevice(&deviceId));
          c10_npu::NPUCachingAllocator::FreeDeviceCachedMemory(deviceId);
          return true;
        }
        AT_ERROR("NPU out of memory. device id: ", deviceId);
      }
      return false;
    }

    void NpuUtils::check_1d(const at::Tensor &t, const char *arg, const char *fn)
    {
      TORCH_CHECK(t.dim() == 1, fn, ": Expected 1-D argument ", arg, ", but got ", t.dim(), "-D");
    }
  } // namespace native
} // namespace at_npu
