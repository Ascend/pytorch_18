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

#ifndef __C10_NPU_INTERFACE_ACLINTERFACE__
#define __C10_NPU_INTERFACE_ACLINTERFACE__

#include "third_party/acl/inc/acl/acl_rt.h"
#include <third_party/acl/inc/acl/acl_base.h>

namespace c10 {
namespace npu {
namespace acl {
typedef enum aclrtEventWaitStatus {
    ACL_EVENT_WAIT_STATUS_COMPLETE  = 0,
    ACL_EVENT_WAIT_STATUS_NOT_READY = 1,
    ACL_EVENT_WAIT_STATUS_RESERVED  = 0xffff,
} aclrtEventWaitStatus;

/**
  aclprofStepInfo is provide by acl, it used to be store dispatch op info.
 */
using aclprofStepInfoPtr = aclprofStepInfo *;
/**
 NpdStatus is provide by acl, it used to store the return value.
 */
using NpdStatus = int;

/** 
  This Api is used to init npd, it need to be called once at process.
 */
aclprofStepInfoPtr init_stepinfo();
/** 
  This Api is used to destroy npd, it need to be called once at process.
 */
NpdStatus destroy_stepinfo(aclprofStepInfoPtr stepInfo);
/** 
  This Api is used to start dispatch op, this operation should be called after init.
 */
NpdStatus start_deliver_op(aclprofStepInfoPtr stepInfo, aclprofStepTag stepTag, aclrtStream stream);
/** 
  This Api is used to stop dispatch op, this operation should be called after start dispatch op.
 */
NpdStatus stop_deliver_op(aclprofStepInfoPtr stepInfo, aclprofStepTag stepTag, aclrtStream stream);

/**
  This API is used to get error msg
  */
const char *AclGetErrMsg();

/**
 * @ingroup AscendCL
 * @brief create event instance
 *
 * @param event [OUT]   created event
 * @param flag [IN]     event flag
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
aclError AclrtCreateEventWithFlag(aclrtEvent *event, uint32_t flag);

/**
  This API is used to query status of event task
  */
aclError AclQueryEventStatus(aclrtEvent event, aclrtEventWaitStatus *waitStatus, aclrtEventStatus *recordStatus);
} // namespace acl
} // namespace npu
} // namespace c10

#endif // __C10_NPU_INTERFACE_ACLINTERFACE__