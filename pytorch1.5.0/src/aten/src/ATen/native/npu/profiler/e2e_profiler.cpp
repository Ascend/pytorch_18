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

#include <ATen/native/npu/profiler/e2e_profiler.h>
#include <ATen/native/npu/interface/MsProfilerInterface.h>
#include <c10/npu/interface/AclInterface.h>
#include <third_party/acl/inc/acl/acl_prof.h>
#include <mutex>

namespace at { 
namespace native {
namespace npu {
namespace profiler {

class CallbackManager {
public:
  void pushCallback(
      E2ERecordFunctionCallback start,
      E2ERecordFunctionCallback end) {
    start_callbacks.push_back(std::move(start));
    end_callbacks.push_back(std::move(end));
  }

  void popCallback() {
    if (start_callbacks.empty()) {
      throw std::runtime_error("Empty callbacks stack");
    }
    start_callbacks.pop_back();
    end_callbacks.pop_back();
  }

  bool hasCallbacks() {
    return !start_callbacks.empty();
  }

  std::vector<E2ERecordFunctionCallback> start_callbacks;
  std::vector<E2ERecordFunctionCallback> end_callbacks;
};

std::mutex next_thread_id_mutex_;
uint16_t next_thread_id_ = 0;
thread_local uint16_t current_thread_id_ = 0;
void * local_stamp = nullptr;
aclprofConfig* local_profCfg = nullptr;

CallbackManager& manager() {
  static CallbackManager instance;
  return instance;
}

void pushCallback(
    E2ERecordFunctionCallback start,
    E2ERecordFunctionCallback end) {
  manager().pushCallback(
      std::move(start), std::move(end));
}

void popCallback() {
  manager().popCallback();
}

bool hasCallbacks() {
  return manager().hasCallbacks();
}

void initMsPorf(const std::string dump_path, uint64_t npu_event,
    uint64_t aicore_metrics) {
  // to init MsProf, there are 4 steps:
  // 1. create profile config, configure option, 
  //    such as type of aicore metrics and 
  //    which modules(ACL, TASK, AICORE, AICORE, L2CACHE) need profiling
  // 2. set msprof switch to be true and set profiling result path.
  // 3. create `stamp` used to record time info.  
  // 4. configure the option of `stamp`.

  int deviceIndex = 0;
  aclError ret = aclrtGetDevice(&deviceIndex);
  if(ret){
    NPU_LOGE("In npu e2e profiling, aclrtGetDevice fail, error code: %d", ret);
    C10_NPU_SHOW_ERR_MSG();
    return;
  }
  const uint32_t deviceNum = 1;
  uint32_t deviceIdList[deviceNum] = {deviceIndex};
    local_profCfg = c10::npu::acl::AclProfilingCreateConfig(
        deviceIdList,
        deviceNum,
        (aclprofAicoreMetrics)aicore_metrics,
        nullptr,
        npu_event);
  if (local_profCfg == nullptr) {
    NPU_LOGE("In npu e2e profiling, create_config fail, error profCfg is null.");
    C10_NPU_SHOW_ERR_MSG();
    (void)c10::npu::acl::AclProfilingFinalize();
    return;
  }
  ret  = c10::npu::acl::AclProfilingInit(dump_path.c_str(), dump_path.length());
  if (ret != ACL_ERROR_NONE) {
    NPU_LOGE("In npu e2e profiling, AclProfilingInit failed.");
    C10_NPU_SHOW_ERR_MSG();
    (void)c10::npu::acl::AclProfilingFinalize();
    return;
  }
  ret = c10::npu::acl::AclProfilingStart(local_profCfg);
  if(ret){
    NPU_LOGE("In npu e2e profiling, AclProfStart fail, error code: %d", ret);
    C10_NPU_SHOW_ERR_MSG();
    (void)c10::npu::acl::AclProfilingFinalize();
    return;
  }
  ret = at::native::npu::profiler::AclprofMsproftxSwitch(true);
  if (ret != ACL_ERROR_NONE) {
    NPU_LOGE("In npu e2e profiling, profMsproftxSwitch open failed.");
    C10_NPU_SHOW_ERR_MSG();
    (void)c10::npu::acl::AclProfilingFinalize();
    return;
  }
  local_stamp = at::native::npu::profiler::AclprofCreateStamp();
  if (local_stamp == nullptr) {
    NPU_LOGE("In npu e2e profiling, aclprofCreateStamp failed, created stamp is nullptr.");
    return;
  }
}

void init_e2e_profiler(const std::string dump_path, uint64_t npu_event,
    uint64_t aicore_metrics) {
  initMsPorf(dump_path, npu_event, aicore_metrics);
  pushCallback(
      [](const E2ERecordFunction& fn) {
        fn.push();
      },
      [](const E2ERecordFunction& fn) {
        fn.pop();
      });
}

void finalize_e2e_profiler() {
  auto ret = c10::npu::acl::AclProfilingStop(local_profCfg);
  if (ret) {
    NPU_LOGE("In npu e2e profiling, AclProfStop fail, error code: %d", ret);
    C10_NPU_SHOW_ERR_MSG();
  }
  at::native::npu::profiler::AclprofDestroyStamp(local_stamp);
  c10::npu::acl::AclProfilingFinalize();
}

/* static */
uint16_t E2ERecordFunction::getCurrentThreadId() {
  if (!current_thread_id_) {
    // happens only once per thread
    std::lock_guard<std::mutex> guard(next_thread_id_mutex_);
    current_thread_id_ = ++next_thread_id_;
  }
  return current_thread_id_;
}

inline void E2ERecordFunction::push() const {
    auto ret = at::native::npu::profiler::AclprofSetStampTraceMessage(
        local_stamp, name_.c_str(), name_.length());
    if (ret != ACL_ERROR_NONE) {
      NPU_LOGE("In npu e2e profiling, AclprofSetStampTraceMessage set failed.");
      C10_NPU_SHOW_ERR_MSG();
      (void)c10::npu::acl::AclProfilingFinalize();
      return;
    }
    ret = at::native::npu::profiler::AclprofPush(local_stamp);
    if (ret != ACL_ERROR_NONE) {
      NPU_LOGE("In npu e2e profiling, AclprofPush set failed.");
      C10_NPU_SHOW_ERR_MSG();
      (void)c10::npu::acl::AclProfilingFinalize();
      return;
    }
}

inline void E2ERecordFunction::pop() const {
    auto ret =at::native::npu::profiler::AclprofPop();
    if (ret != ACL_ERROR_NONE) {
      NPU_LOGE("In npu e2e profiling, AclprofPop set failed.");
      C10_NPU_SHOW_ERR_MSG();
      (void)c10::npu::acl::AclProfilingFinalize();
      return;
    }
}

void E2ERecordFunction::before(const char* name) {
  if (!hasCallbacks()) {
    return;
  }
  AT_ASSERT(!initialized_);
  name_ = std::string(name);
  initialized_ = true;
  processCallbacks();
}

void E2ERecordFunction::before(std::string name) {
  if (!hasCallbacks()) {
    return;
  }
  AT_ASSERT(!initialized_);
  name_ = name;
  initialized_ = true;
  processCallbacks();
}

void E2ERecordFunction::processCallbacks() {
  for (size_t idx = 0; idx < manager().start_callbacks.size(); ++idx) {
    try {
      manager().start_callbacks[idx](*this);
    } catch (const std::exception &e) {
      NPU_LOGE("Exception in E2ERecordFunction start observer: %s" , e.what());
    }
  }
}

E2ERecordFunction::~E2ERecordFunction() {
  end();
}

void E2ERecordFunction::end() {
  if (initialized_) {
    for (size_t idx = 0; idx < manager().end_callbacks.size(); ++idx) {
      try {
        manager().end_callbacks[idx](*this);
      } catch (const std::exception &e) {
        NPU_LOGE("Exception in RecordFunction end observer: %s", e.what());
      }
    }
  }
  initialized_ = false;
}

}
}
}
}