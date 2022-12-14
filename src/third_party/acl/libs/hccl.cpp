// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

#include "hccl.h"

hcclResult_t hcclCommInitUniqueId(hcclComm_t* comm, u32 nranks, hcclUniqueId commId, u32 myrank) {return HCCL_SUCCESS;}
hcclResult_t hcclGetUniqueId(hcclUniqueId* id) {return HCCL_SUCCESS;}
hcclResult_t hcclAllReduce(void *inputPtr, void *outputPtr, u64 count, hcclDataType_t dataType,
                                  hcclRedOp_t op, hcclComm_t comm, rtStream_t stream) {return HCCL_SUCCESS;}
hcclResult_t hcclBroadcast(void *ptr, u64 count, hcclDataType_t dataType, u32 root, hcclComm_t comm,
                                  rtStream_t stream) {return HCCL_SUCCESS;}
hcclResult_t hcclCommDestroy(hcclComm_t comm) {return HCCL_SUCCESS;}