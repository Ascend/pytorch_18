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

#ifndef THNP_EVENT_INC
#define THNP_EVENT_INC

#include <ATen/npu/NPUEvent.h>
#include <torch/csrc/python_headers.h>

struct THNPEvent {
  PyObject_HEAD
  at::npu::NPUEvent npu_event;
};
extern PyObject *THNPEventClass;

void THNPEvent_init(PyObject *module);

inline bool THNPEvent_Check(PyObject* obj) {
  return THNPEventClass && PyObject_IsInstance(obj, THNPEventClass);
}

#endif // THNP_EVENT_INC
