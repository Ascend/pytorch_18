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

#include <torch/csrc/python_headers.h>

#include <ATen/ATen.h>
#include <ATen/npu/NPUGenerator.h>
#include <TH/TH.h>
#include <acl/acl.h>
#include <c10/npu/NPUException.h>
#include <c10/npu/NPUFunctions.h>
#include <c10/npu/NPUCachingAllocator.h>
#include <c10/npu/NPUStream.h>
#include <c10/npu/sys_ctrl/npu_sys_ctrl.h>
#include <c10/npu/register/OptionRegister.h>
#include <ATen/native/npu/utils/DynamicShapeUtil.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/utils/npu_lazy_init.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/THP.h>
#include <chrono>
#include <sstream>
#include <thread>
#include <unordered_map>


using namespace torch;

static PyObject* THNPModule_initExtension(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  c10::npu::NpuSysCtrl::SysStatus status =
  c10::npu::NpuSysCtrl::GetInstance().Initialize(); if (status !=
  c10::npu::NpuSysCtrl::SysStatus::INIT_SUCC) {
    throw python_error();
  }
  auto m = THPObjectPtr(PyImport_ImportModule("torch.npu"));
  if (!m) throw python_error();

  auto set_module_attr = [&](const char* name, PyObject* v) {
    // PyObject_SetAttrString doesn't steal reference. So no need to incref.
    if (PyObject_SetAttrString(m, name, v) < 0) {
      throw python_error();
    }
  };
  auto num_npus = c10::npu::device_count();
  auto default_npu_generators = PyTuple_New(static_cast<Py_ssize_t>(num_npus));
  for(int i = 0; i < num_npus; i++) {
    auto gen = at::npu::detail::getDefaultNPUGenerator(i);
    auto cast_gen = (THPGenerator*)THPGenerator_initDefaultGenerator(gen);
    // This reference is meant to be given away, so no need to incref here.
    PyTuple_SetItem(default_npu_generators, i, (PyObject*)cast_gen);
  }
  set_module_attr("default_generators", default_npu_generators);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_set_run_yet_variable_to_false_wrap(
    PyObject* self,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch::utils::npu_set_run_yet_variable_to_false();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_npuSynchronize(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  pybind11::gil_scoped_release no_gil;
  c10::npu::npuSynchronizeDevice();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

void THNPModule_setDevice(int device) {
  C10_NPU_CHECK(aclrtSetDevice(device));
}

PyObject* THNPModule_setDevice_wrap(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  int device = THPUtils_unpackLong(arg);
  c10::npu::NpuSysCtrl::SysStatus status =
      c10::npu::NpuSysCtrl::GetInstance().Initialize(device);
  if (status != c10::npu::NpuSysCtrl::SysStatus::INIT_SUCC) {
      NPU_LOGE("Npu init fail.");
  }

  int pre_device = 0;
  auto ret = aclrtGetDevice(&pre_device);
  if (ret != ACL_ERROR_NONE){
      C10_NPU_CHECK(aclrtSetDevice(device));
  } else if (pre_device != device) {
      c10::npu::NpuSysCtrl::GetInstance().ExchangeDevice(pre_device, device);
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_getDevice_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  int device;
  torch::utils::npu_lazy_init();
  C10_NPU_CHECK(aclrtGetDevice(&device));
  return PyLong_FromLong(device);
  END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_getDeviceCount_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return PyLong_FromLong(c10::npu::device_count());
  END_HANDLE_TH_ERRORS
}

PyObject * THNPModule_getCurrentStream_wrap(
    PyObject * /* unused */, PyObject *device_index) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
    THPUtils_checkLong(device_index), "invalid argument to getCurrentStream");
  int64_t device = THPUtils_unpackLong(device_index);
  return PyLong_FromUnsignedLongLong(
    c10::npu::getCurrentNPUStream(device).pack());
  END_HANDLE_TH_ERRORS
}

PyObject * THNPModule_getDefaultStream_wrap(PyObject *self /* unused */, PyObject *device_index) {
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(device_index), "invalid argument to getDefaultStream");
  int64_t device = THPUtils_unpackLong(device_index);
  return PyLong_FromUnsignedLongLong(c10::npu::getDefaultNPUStream(device).pack());
  END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_increaseStep_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  at::native::npu::DynamicIncreaseSteps();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THNPModule_setStream_wrap(PyObject *self, PyObject *obj)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(PyLong_Check(obj), "invalid stream");
  uint64_t bits = PyLong_AsUnsignedLongLong(obj);
  if (bits == static_cast<uint64_t>(-1) && PyErr_Occurred()) {
    throw python_error();
  }
  auto stream = c10::npu::NPUStream::unpack(bits);
  int device;
  C10_NPU_CHECK(aclrtGetDevice(&device));
  if (device != stream.device_index()) {
    THNPModule_setDevice(stream.device_index());
  }
  c10::npu::setCurrentNPUStream(stream);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THNPModule_emptyCache(PyObject *_unused, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  c10::npu::NPUCachingAllocator::emptyCache();
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject * THNPModule_memoryStats(PyObject *_unused, PyObject *arg)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to memory_allocated");
  const int device = (int) THPUtils_unpackLong(arg);

  using c10::npu::NPUCachingAllocator::StatType;
  using c10::npu::NPUCachingAllocator::Stat;
  using c10::npu::NPUCachingAllocator::StatArray;
  using c10::npu::NPUCachingAllocator::DeviceStats_;

  const auto statToDict = [](const Stat& stat) {
    py::dict dict;

    dict["current"] = stat.current;
    dict["peak"] = stat.peak;
    dict["allocated"] = stat.allocated;
    dict["freed"] = stat.freed;
    return dict;
  };

  const auto statArrayToDict = [=](const StatArray& statArray) {
    const std::array<const char*, static_cast<size_t>(StatType::NUM_TYPES)> statTypeNames = {
      "all", "small_pool", "large_pool"
    };
    py::dict dict;
    for (size_t i = 0; i < statTypeNames.size(); ++i) {
      dict[statTypeNames[i]] = statToDict(statArray[i]);
    }
    return dict;
  };

  const DeviceStats_ stats = c10::npu::NPUCachingAllocator::getDeviceStats(device);

  py::dict result;
  result["num_alloc_retries"] = stats.num_alloc_retries;
  result["num_ooms"] = stats.num_ooms;
  result["allocation"] = statArrayToDict(stats.allocation);
  result["segment"] = statArrayToDict(stats.segment);
  result["active"] = statArrayToDict(stats.active);
  result["inactive_split"] = statArrayToDict(stats.inactive_split);
  result["allocated_bytes"] = statArrayToDict(stats.allocated_bytes);
  result["reserved_bytes"] = statArrayToDict(stats.reserved_bytes);
  result["active_bytes"] = statArrayToDict(stats.active_bytes);
  result["inactive_split_bytes"] = statArrayToDict(stats.inactive_split_bytes);

  return result.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject * THNPModule_resetAccumulatedMemoryStats(PyObject *_unused, PyObject *arg)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to reset_accumulated_memory_stats");
  const int device = (int) THPUtils_unpackLong(arg);
  c10::npu::NPUCachingAllocator::resetAccumulatedStats(device);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject * THNPModule_resetPeakMemoryStats(PyObject *_unused, PyObject *arg)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to reset_peak_memory_stats");
  const int device = (int) THPUtils_unpackLong(arg);
  c10::npu::NPUCachingAllocator::resetPeakStats(device);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject * THNPModule_memorySnapshot(PyObject *_unused, PyObject *noargs)
{
  HANDLE_TH_ERRORS

  using c10::npu::NPUCachingAllocator::SegmentInfo;
  using c10::npu::NPUCachingAllocator::BlockInfo;

  const auto segmentInfoToDict = [](const SegmentInfo& segmentInfo) {
    py::dict segmentDict;
    segmentDict["device"] = segmentInfo.device;
    segmentDict["address"] = segmentInfo.address;
    segmentDict["total_size"] = segmentInfo.total_size;
    segmentDict["allocated_size"] = segmentInfo.allocated_size;
    segmentDict["active_size"] = segmentInfo.active_size;
    segmentDict["segment_type"] = (segmentInfo.is_large ? "large" : "small");

    py::list blocks;
    for (const auto& blockInfo : segmentInfo.blocks) {
      py::dict blockDict;
      blockDict["size"] = blockInfo.size;
      blockDict["state"] = (blockInfo.allocated ? "active_allocated" : (blockInfo.active ? "active_pending_free" : "inactive"));
      blocks.append(blockDict);
    }
    segmentDict["blocks"] = blocks;

    return segmentDict;
  };

  const std::vector<SegmentInfo>& snapshot = c10::npu::NPUCachingAllocator::snapshot();
  py::list result;

  for (const auto& segmentInfo : snapshot) {
    result.append(segmentInfoToDict(segmentInfo));
  }

  return result.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject * THNPModule_npuCachingAllocator_raw_alloc(PyObject *_unused, PyObject *args){
  HANDLE_TH_ERRORS
  PyObject* size_o = nullptr;
  PyObject* stream_o = nullptr;
  if(!PyArg_ParseTuple(args, "OO", &size_o, &stream_o)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "caching_allocator_alloc",
        1,
        "(ssize_t size, intptr_t stream);");
    return nullptr;
  }
  ssize_t size = PyLong_AsSsize_t(size_o);
  aclrtStream stream = static_cast<aclrtStream>(PyLong_AsVoidPtr(stream_o));
  void* mem = c10::npu::NPUCachingAllocator::raw_alloc_with_stream(size, stream);
  return PyLong_FromVoidPtr(mem);
  END_HANDLE_TH_ERRORS
}

PyObject * THNPModule_npuCachingAllocator_raw_delete(PyObject *_unused, PyObject *obj){
  HANDLE_TH_ERRORS
  void* mem_ptr = PyLong_AsVoidPtr(obj);
  c10::npu::NPUCachingAllocator::raw_delete(mem_ptr);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// We need to ensure that as long as a thread will NEVER loose the GIL as long as
// it holds the NPU mutex. Otherwise another thread might be scheduled and try to
// e.g. allocate a new tensor which will cause a deadlock. It's enough to have a
// single global, because it can be only set once (npuMutex is not recursive)
// by the thread that owns the mutex (obviously there can be only one such thread).
static PyGILState_STATE npuMutexGILState;

PyObject * THNPModule_npuLockMutex(PyObject *module, PyObject *noargs)
{
  auto mutex = c10::npu::NPUCachingAllocator::getFreeMutex();
  // This has to be a busy loop because we **absolutely need to** hold the GIL
  // or it's a recipe for a deadlock otherwise (if we let other Python threads
  // run while we have the cudaMutex, but not the GIL, they might try to e.g.
  // free a CUDA tensor and acquire the cudaMutex without giving up the GIL,
  // because it happens deep within THC).
  while (true) {
    if (mutex->try_lock())
      break;
    {
      pybind11::gil_scoped_release no_gil;
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
  }

  npuMutexGILState = PyGILState_Ensure();
  Py_RETURN_NONE;
}

PyObject * THNPModule_npuUnlockMutex(PyObject *module, PyObject *noargs)
{
  auto mutex = c10::npu::NPUCachingAllocator::getFreeMutex();
  PyGILState_Release(npuMutexGILState);
  mutex->unlock();
  Py_RETURN_NONE;
}

PyObject* THNPModule_setOption_wrap(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS

  if (!PyDict_Check(arg)) {
    throw TypeError("npu option must be a dict.");
  }

  PyObject *key, *value;
  Py_ssize_t pos = 0;
  std::map<std::string, std::string> option;

  while (PyDict_Next(arg, &pos, &key, &value)) {
    if (key == nullptr || !PyUnicode_Check(key)) {
      throw TypeError("option name is nullptr or is not string.");
    }

    if (value == nullptr || !PyUnicode_Check(value)) {
      throw TypeError("option value is nullptr or is not string.");
    }

    const char *pKey = PyUnicode_AsUTF8(key);
    const char *pValue = PyUnicode_AsUTF8(value);
    option[pKey] = pValue;
  }
  
  torch::utils::npu_lazy_init();
  {
    pybind11::gil_scoped_release no_gil;
    c10::npu::SetOption(option);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static struct PyMethodDef _THNPModule_methods[] = {
    {"_npu_init", (PyCFunction)THNPModule_initExtension, METH_NOARGS, nullptr},
    {"_npu_set_run_yet_variable_to_false", (PyCFunction)THNPModule_set_run_yet_variable_to_false_wrap, METH_NOARGS, nullptr},
    {"_npu_synchronize", (PyCFunction)THNPModule_npuSynchronize, METH_NOARGS, nullptr},
    {"_npu_setDevice", (PyCFunction)THNPModule_setDevice_wrap, METH_O, nullptr},
    {"_npu_getDevice", (PyCFunction)THNPModule_getDevice_wrap, METH_NOARGS, nullptr},
    {"_npu_getDeviceCount", (PyCFunction)THNPModule_getDeviceCount_wrap, METH_NOARGS, nullptr},
    {"_npu_getCurrentStream", (PyCFunction)THNPModule_getCurrentStream_wrap, METH_O, nullptr},
    {"_npu_getDefaultStream", (PyCFunction)THNPModule_getDefaultStream_wrap, METH_O, nullptr},
    {"_npu_setStream", (PyCFunction)THNPModule_setStream_wrap,  METH_O, nullptr}, 
    {"_npu_increaseStep", (PyCFunction)THNPModule_increaseStep_wrap,METH_NOARGS,nullptr},
    {"_npu_setStream", (PyCFunction)THNPModule_setStream_wrap,  METH_O, nullptr},
    {"_npu_emptyCache", (PyCFunction) THNPModule_emptyCache, METH_NOARGS, nullptr},
    {"_npu_memoryStats", (PyCFunction) THNPModule_memoryStats, METH_O, nullptr},
    {"_npu_resetAccumulatedMemoryStats", (PyCFunction) THNPModule_resetAccumulatedMemoryStats, METH_O, nullptr},
    {"_npu_resetPeakMemoryStats", (PyCFunction) THNPModule_resetPeakMemoryStats, METH_O,  nullptr},
    {"_npu_memorySnapshot", (PyCFunction) THNPModule_memorySnapshot, METH_NOARGS, nullptr},
    {"_npu_npuCachingAllocator_raw_alloc", (PyCFunction)THNPModule_npuCachingAllocator_raw_alloc, METH_VARARGS, nullptr},
    {"_npu_npuCachingAllocator_raw_delete", (PyCFunction)THNPModule_npuCachingAllocator_raw_delete, METH_O, nullptr},
    {"_npu_lock_mutex",   (PyCFunction)THNPModule_npuLockMutex,   METH_NOARGS,  nullptr},
    {"_npu_unlock_mutex", (PyCFunction)THNPModule_npuUnlockMutex, METH_NOARGS,  nullptr},
    {"_npu_setOption", (PyCFunction)THNPModule_setOption_wrap, METH_O, nullptr},
    {nullptr}};

PyMethodDef* THNPModule_methods() {
  return _THNPModule_methods;
}

namespace torch {
namespace npu {

void initModule(PyObject* module) {
  // todo init
  // python::initCommMethods(module);
}

} // namespace npu
} // namespace torch
