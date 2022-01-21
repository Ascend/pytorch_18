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

#include <c10/npu/register/OptionRegister.h>
#include <c10/util/Exception.h>
#include <ATen/native/npu/profiler/NpuProfiling.h>

#include "torch_npu/csrc/framework/utils/NpuFuzzyBlacklist.h"
#include "torch_npu/csrc/framework/utils/NpuProfilingDispatch.h"
#include "third_party/acl/inc/acl/acl_mdl.h"
#include "torch_npu/csrc/framework/interface/AclOpCompileInterface.h"

namespace at_npu
{
  namespace native
  {
    namespace env
    {

      REGISTER_OPTION(autotune)
      REGISTER_OPTION_BOOL_FUNCTION(AutoTuneEnabled, autotune, "disable", "enable")

      REGISTER_OPTION_INIT_BY_ENV(bmmv2_enable)
      REGISTER_OPTION_BOOL_FUNCTION(CheckBmmV2Enable, bmmv2_enable, "0", "1")

      REGISTER_OPTION_HOOK(mdldumpswitch, [](const std::string &val)
                           {
                             if (val == "enable")
                             {
                               aclmdlInitDump();
                             }
                             else
                             {
                               aclmdlFinalizeDump();
                             }
                           })
      REGISTER_OPTION_HOOK(mdldumpconfigpath, [](const std::string &val)
                           { aclmdlSetDump(val.c_str()); })

      REGISTER_OPTION_HOOK(fuzzycompileswitch, [](const std::string &val)
                           {
                             if (val == "enable")
                             {
                               AclopSetCompileFlag(aclOpCompileFlag::ACL_OP_COMPILE_FUZZ);
                             }
                             else
                             {
                               AclopSetCompileFlag(aclOpCompileFlag::ACL_OP_COMPILE_DEFAULT);
                             }
                           })
      REGISTER_OPTION_BOOL_FUNCTION(CheckFuzzyEnable, fuzzycompileswitch, "disable", "enable")

      REGISTER_OPTION_HOOK(ACL_OP_DEBUG_LEVEL, [](const std::string &val)
                           { aclSetCompileopt(aclCompileOpt::ACL_OP_DEBUG_LEVEL, val.c_str()); })
      REGISTER_OPTION_HOOK(ACL_DEBUG_DIR, [](const std::string &val)
                           { aclSetCompileopt(aclCompileOpt::ACL_DEBUG_DIR, val.c_str()); })
      REGISTER_OPTION_HOOK(ACL_OP_COMPILER_CACHE_MODE, [](const std::string &val)
                           { aclSetCompileopt(aclCompileOpt::ACL_OP_COMPILER_CACHE_MODE, val.c_str()); })
      REGISTER_OPTION_HOOK(ACL_OP_COMPILER_CACHE_DIR, [](const std::string &val)
                           { aclSetCompileopt(aclCompileOpt::ACL_OP_COMPILER_CACHE_DIR, val.c_str()); })
      REGISTER_OPTION_HOOK(ACL_OP_SELECT_IMPL_MODE, [](const std::string &val)
                           {
                             if (val == "high_precision" || val == "high_performance")
                             {
                               aclSetCompileopt(aclCompileOpt::ACL_OP_SELECT_IMPL_MODE, val.c_str());
                             }
                             else
                             {
                               TORCH_CHECK(0, "ACL_OP_SELECT_IMPL_MODE only support `high_precision` or "
                                              " `high_performance`, but got ",
                                           val);
                             }
                           })
      REGISTER_OPTION_HOOK(ACL_OPTYPELIST_FOR_IMPLMODE, [](const std::string &val)
                           { aclSetCompileopt(aclCompileOpt::ACL_OPTYPELIST_FOR_IMPLMODE, val.c_str()); })
      REGISTER_OPTION_HOOK(NPU_FUZZY_COMPILE_BLACKLIST, [](const std::string &val)
                           { FuzzyCompileBlacklist::GetInstance().RegisterBlacklist(val); })

      REGISTER_OPTION_INIT_BY_ENV(PROFILING_MODE)
      REGISTER_OPTION_BOOL_FUNCTION(CheckProfilingEnable, PROFILING_MODE, "false", "true");

      REGISTER_OPTION_HOOK(deliverswitch, [](const std::string &val)
                           {
                             TORCH_CHECK(
                                 CheckProfilingEnable(),
                                 "before you prepare to deliver op, ",
                                 "you should be enture profiling mode is on correctly!");
                             if (val == "enable")
                             {
                               NpuProfilingDispatch::Instance().start();
                             }
                             else
                             {
                               NpuProfilingDispatch::Instance().stop();
                             }
                           })

      REGISTER_OPTION_HOOK(profilerResultPath, [](const std::string &val)
                           { at::native::npu::NpuProfiling::Instance().Init(val); })

      REGISTER_OPTION_HOOK(profiling, [](const std::string &val)
                           {
                             if (val.compare("start") == 0)
                             {
                               at::native::npu::NpuProfiling::Instance().Start();
                             }
                             else if (val.compare("stop") == 0)
                             {
                               at::native::npu::NpuProfiling::Instance().Stop();
                             }
                             else if (val.compare("finalize") == 0)
                             {
                               at::native::npu::NpuProfiling::Instance().Finalize();
                             }
                             else
                             {
                               TORCH_CHECK(false, "profiling input: (", val, " ) error!")
                             }
                           })

    } // namespace env
  }   // namespace native
} // namespace at_npu