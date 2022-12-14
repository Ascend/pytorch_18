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

#ifndef __NATIVE_NPU_UTILS_OP_PIPE__
#define __NATIVE_NPU_UTILS_OP_PIPE__

#include <ATen/ATen.h>

namespace at {
namespace native {
namespace npu {

// 
template<class Derived>
class OpPipe {
public:
  using PROCESS_FUNC = std::function<void(Tensor&)>;
  Derived& Func(const PROCESS_FUNC& func) {
    this->func = func;
    return static_cast<Derived&>(*this);
  }
protected:
  PROCESS_FUNC func = nullptr;
};

//
class OpPipeWithDefinedOut : public OpPipe<OpPipeWithDefinedOut> {
public:
  OpPipeWithDefinedOut& CheckMemory(const std::initializer_list<Tensor>& inputs, const std::initializer_list<Tensor>& outputs);
  Tensor& Call(Tensor& dst);
};

// 
class OpPipeWithApplyOut : public OpPipe<OpPipeWithApplyOut> {
public:
  using PROCESS_FUNC = std::function<void(Tensor&)>;
  OpPipeWithApplyOut& ApplyOutputSameAs(const Tensor& src);
  Tensor& Call();
private:
  Tensor dst;
};

namespace deprecated {
  // TODO(ascend): add
} // deprecated

} // namespace npu
} // namespace native
} // namespace at

#endif
