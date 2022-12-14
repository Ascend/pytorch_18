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

#ifndef __NATIVE_NPU_UTILS_FORMAT_INFER__
#define __NATIVE_NPU_UTILS_FORMAT_INFER__

#include <ATen/ATen.h>
#include <ATen/native/npu/utils/NPUDefinition.h>

namespace at {
namespace native {
namespace npu {

// Format is the property of tensor storage. Format is the way to tell an
// operator how the result should be organized in memory and nothing more.
// Storage format collect the helper functions of npu's format. It tell the 
// relationship between format and storage.
// 
class InferFormat {
public:
  // Feature: The function is used to guess base format
  // The base formats are NCHW, NCDHW, ND, who is not padding.
  // The format transform between other formats should be based
  // on these base formats.(their should convert to base format first.)
  // This function will be called at new, reset, set and so on.
  static std::tuple<aclFormat, aclFormat> GuessFormatUnit(IntArrayRef size, aclFormat format);
  // GuessBaseFormat is the base of the format assumption
  // this function is called when apply the new tensor
  static aclFormat GuessBaseFormat(IntArrayRef size);
  // this function used to fix format when format and size is not match
  static aclFormat GuessStorageFormat(IntArrayRef size, aclFormat format);
  // Features: guess the format of tensor after it called format_contiguous().
  // According to the law of continuity, the output format is same as input format,
  // this function is called to guess the input format, so it also the output format.
  // NOTE: The caller should make sure that the tensor is non-contigous
  static aclFormat GuessFormatWhenContiguous(const Tensor& tensor);
  // This api is used to infer storage size when called transdata
  // fix: ND->NZ when dim < 2
  // not effect the storage data.
  static FormatShape GuessStorageSizeWhenConvertFormat(const Tensor& tensor);
  // This api is used to judge if tensor is reasonable when size changes.
  // solution: tranform to base format to fix it.
  // fix: NCHW | 5HD -> NCDHW | NCDHW or ND | ND
  // unsqueeze/squeeze/select/flatten/view will change meta data, they will call
  // as_strided and view
  static bool IsDefiniteTensorWhenMetaDataChanges(const Tensor& tensor, IntArrayRef size);
}; //class InferFormat


} // namespace npu
} // namespace native
} // namespace at

#endif // __NATIVE_NPU_UTILS_FORMAT_INFER__