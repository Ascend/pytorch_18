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

#ifndef __NATIVE_NPU_UTILS_OP_TEMPLATE__
#define __NATIVE_NPU_UTILS_OP_TEMPLATE__

#include <functional>
#include "ATen/native/npu/frame/OpCommandBase.h"
#include "ATen/native/npu/utils/OpPreparation.h"

namespace at {
namespace native {
namespace npu {

class OpCommand : public OpCommandBase<OpCommand> {
public:
  // Usage:
  //  auto func = [&self]() {  //
  //  请直接将所有都通过引用或者赋值的方式导入lambda函数中
  //    bool pass = true; // true时，不进入acl处理
  //    return std::tie(pass, self);
  //  };
  //
  //  OpCommand cmd;
  //  cmd.Name("xxx")
  //   .InputWithFunc(func);
  //   .Run();
  using FUNC_TYPE = std::function<std::tuple<bool, Tensor&>(void)>;
  OpCommand& InputWithFunc(const FUNC_TYPE& func);
  OpCommand& Inputs(const TensorList& inputs);
  OpCommand& InputPair(const Tensor& npu_input, const Tensor& cpu_input);
}; // class OpCommand

// only for transData now
class TransDataOpCommand : public OpCommandBase<TransDataOpCommand> {
public:
  TransDataOpCommand& InputAndOutput(const Tensor& input, const Tensor& output);
private:
  TransDataOpCommand& AddInputAndOutput(const Tensor& input, const Tensor& output);
}; // class TransDataOpCommand

} // namespace npu
} // namespace native
} // namespace at

#endif