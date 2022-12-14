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

#include "HcclInterface.h"
#include "c10/npu/register/FunctionLoader.h"
#include "c10/util/Exception.h"

namespace c10 {
namespace npu {
namespace hccl {

#undef LOAD_FUNCTION
#define LOAD_FUNCTION(funcName) \
  REGISTER_FUNCTION(libhccl, funcName)
#undef GET_FUNC
#define GET_FUNC(funcName)              \
  GET_FUNCTION(libhccl, funcName)

REGISTER_LIBRARY(libhccl)
LOAD_FUNCTION(HcclBarrier)

HcclResult hccl_barrier(HcclComm comm, aclrtStream stream) { 
  typedef HcclResult(*HcclBarrierFunc)(HcclComm, aclrtStream);
  static HcclBarrierFunc func = nullptr;
  if (func == nullptr) {
    func = (HcclBarrierFunc)GET_FUNC(HcclBarrier);
  }
  if (func == nullptr) {
    // can not find HcclBarrier API
    return HcclResult::HCCL_E_NOT_SUPPORT;
  }
  return func(comm, stream);
}

} // namespace hccl
} // namespace npu
} // namespace c10
