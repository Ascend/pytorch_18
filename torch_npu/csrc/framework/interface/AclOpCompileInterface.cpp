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

#include <c10/npu/register/FunctionLoader.h>
#include <c10/util/Exception.h>

#include "torch_npu/csrc/framework/interface/AclOpCompileInterface.h"

namespace at_npu
{
  namespace native
  {

#undef LOAD_FUNCTION
#define LOAD_FUNCTION(funcName) \
  REGISTER_FUNCTION(libacl_op_compiler, funcName)
#undef GET_FUNC
#define GET_FUNC(funcName) \
  GET_FUNCTION(libacl_op_compiler, funcName)

    REGISTER_LIBRARY(libacl_op_compiler)
    LOAD_FUNCTION(aclopSetCompileFlag)

    aclError AclopSetCompileFlag(aclOpCompileFlag flag)
    {
      typedef aclError (*aclopSetCompileFlagFunc)(aclOpCompileFlag);
      static aclopSetCompileFlagFunc func = nullptr;
      if (func == nullptr)
      {
        func = (aclopSetCompileFlagFunc)GET_FUNC(aclopSetCompileFlag);
      }
      TORCH_CHECK(func, "Failed to find function ", "aclopSetCompileFlag");
      auto ret = func(flag);
      return ret;
    }

  } // namespace native
} // namespace at_npu