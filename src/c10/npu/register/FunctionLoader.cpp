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


#include "FunctionLoader.h"
#include <dlfcn.h>
#include "c10/util/Exception.h"

namespace c10 {
namespace npu {

FunctionLoader::FunctionLoader(const std::string& name) {
  this->fileName = name + ".so";
}

FunctionLoader::~FunctionLoader() {
  if (this->handle != nullptr) {
    dlclose(this->handle);
  }
}

void FunctionLoader::Set(const std::string& name) {
  this->registry[name] = nullptr;
}

void* FunctionLoader::Get(const std::string& name) {
  if (this->handle == nullptr) {
    auto handle = dlopen(this->fileName.c_str(), RTLD_LAZY);
    if (handle == nullptr) {
      AT_ERROR(dlerror());
      return nullptr;
    }
    this->handle = handle;
  }

  auto itr = registry.find(name);
  if (itr == registry.end()) {
    AT_ERROR("function(", name, ") is not registered.");
    return nullptr;
  }

  if (itr->second != nullptr) {
    return itr->second;
  }

  auto func = dlsym(this->handle, name.c_str());
  if (func == nullptr) {
    return nullptr;
  }
  this->registry[name] = func;
  return func;
}

namespace register_function {
  FunctionRegister* FunctionRegister::GetInstance() {
    static FunctionRegister instance;
    return &instance;
  }
  void FunctionRegister::Register(const std::string& name, ::std::unique_ptr<FunctionLoader>& ptr) {
    std::lock_guard<std::mutex> lock(mu_);
    registry.emplace(name, std::move(ptr));
  }

  void FunctionRegister::Register(const std::string& name, const std::string& funcName) {
    auto itr = registry.find(name);
    if (itr == registry.end()) {
      AT_ERROR(name, " library should register first.");
      return;
    }
    itr->second->Set(funcName);
  }
  
  void* FunctionRegister::Get(const std::string& soName, const std::string& funcName) {
    auto itr = registry.find(soName);
    if (itr != registry.end()) {
      return itr->second->Get(funcName);
    }
    return nullptr;
  }

  FunctionRegisterBuilder::FunctionRegisterBuilder(const std::string& name, ::std::unique_ptr<FunctionLoader>& ptr) {
    FunctionRegister::GetInstance()->Register(name, ptr);
  }
  FunctionRegisterBuilder::FunctionRegisterBuilder(const std::string& soName, const std::string& funcName) {
    FunctionRegister::GetInstance()->Register(soName, funcName);
  }
} // namespace register_function


} // namespace npu
} // namespace at
