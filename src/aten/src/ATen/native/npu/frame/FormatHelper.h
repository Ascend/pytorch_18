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

#ifndef __NATIVE_NPU_UTILS_FORMAT_HELPER__
#define __NATIVE_NPU_UTILS_FORMAT_HELPER__

#include <ATen/ATen.h>
#include <unordered_map>
#include <ATen/native/npu/utils/NPUDefinition.h>

namespace at {
namespace native {
namespace npu {

using baseFormatConverter = std::function<FormatShape(IntArrayRef storage_dims, IntArrayRef base_dims)>;
// helper function of storage format
class FormatHelper {
public:
  // helper function of copy, because of padding will change the physical size.
  static bool IsPadded(const Tensor* tensor);
  static char* GetFormatName(const Tensor& tensor);
  static aclFormat GetBaseFormat(const Tensor& tensor);
  static aclFormat GetBaseFormat(aclFormat format);
  static aclFormat GetFormat(const Tensor& tensor);

  static bool IsBaseFormatType(aclFormat format);
  static bool IsBaseFormatType(const Tensor& tensor);

  // Default assumption: the original format are ND, NCHW or NDHWC.
  // So, if original size are 4D, it maybe NCHW or ND and so on.
  // The format can be split into two parts:
  // 1. The storage size can be infered between NC1HWC0, NHWC, NC1HWC0_C04, NCHW.
  // 2. The storage size can be infered between NDC1HWC0 and NDHWC/NCDHW.
  // The storage size can not be infered between different groups.
  template<typename sizeType>
  static FormatShape GetStorageSizes(aclFormat format, sizeType ori_size);
  // GetStorageSizes used to calculate the storage sizes of op at npu device at different format.
  static FormatShape GetStorageSizes(NPUStorageDesc desc);
  static FormatShape GetSizeOfBaseFormat(const Tensor& src, aclFormat dst_format);

private:
  static bool IsPadded(aclFormat format);
  static char* GetFormatName(aclFormat format);

private:
  using shapeInfer = std::function<FormatShape(IntArrayRef dims)>;
  typedef struct FormatInfo_ {
    aclFormat format = ACL_FORMAT_ND;
    aclFormat baseFormat = ACL_FORMAT_ND;
    shapeInfer func = nullptr;
    char formatName[30] = {0};
    bool isPadded = false;
  } FormatInfo;
  static std::unordered_map<aclFormat, FormatInfo> info;
  static std::unordered_map<aclFormat, std::unordered_map<aclFormat, baseFormatConverter>> base_format_convert_info;
}; // class FormatHelper

// template impl
template<typename sizeType>
FormatShape FormatHelper::GetStorageSizes(aclFormat format, sizeType ori_size) {
  auto itr = info.find(format);
  if (itr != info.end()) {
    if (itr->second.func) {
      return itr->second.func(ori_size);
    }
  }
  AT_ERROR("unsupport InferShape with format ", GetFormatName(format), "with shape", ori_size);
  return {};
}

} // npu
} // native
} // at
#endif