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
#include "NPUTdtChannelQueue.h"
namespace c10 {
namespace npu {
bool NpuTdtChannelQue::Init() {
  std::lock_guard<std::mutex> lock(channel_mutex_);
  if (inited_) {
    TORCH_WARN("Current channel que %s has beed inited.", channel_name_);
    return true;
  }
  TORCH_CHECK(!channel_name_.empty(), "Name of channel que is empty.");
  TORCH_CHECK(capacity_ != 0, "Capacity of cur queue %s is zero.", channel_name_);
  C10_NPU_CHECK(aclrtGetDevice(&device_id_));
  channel_handle_ = acl_tdt::AcltdtCreateChannelWithCapacity(
      device_id_, channel_name_.c_str(), capacity_);
  
  TORCH_CHECK(channel_handle_ != nullptr, "Init channel queue failed.");
  inited_ = true;
  return inited_;
}

NpuTdtChannelQue::~NpuTdtChannelQue() {
  std::lock_guard<std::mutex> lock(channel_mutex_);
  if (inited_) {
    TORCH_CHECK(channel_handle_ != nullptr, "channel_handle is nullptr during ~NpuTdtChannelQue");
    auto ret = acl_tdt::AcltdtDestroyChannel(channel_handle_);
    TORCH_CHECK(ret == ACL_ERROR_NONE,
                "Destroy channel queue ", channel_name_,
                " failed, error message is ", acl::AclGetErrMsg());
    channel_handle_ = nullptr;
    inited_ = false;
  }
}

std::shared_ptr<TdtDataSet> NpuTdtChannelQue::Dequeue() {
  std::lock_guard<std::mutex> lock(channel_mutex_);
  if (!inited_) {
    return nullptr;
  }
  TORCH_CHECK(channel_handle_ != nullptr, "channel_handle is nullptr during Dequeue");
  auto ret_dataset = std::make_shared<TdtDataSet>();
  acl_tdt::AcltdtReceiveTensor(channel_handle_, ret_dataset->GetPtr().get(), time_out_);
  return ret_dataset;
}
}
}