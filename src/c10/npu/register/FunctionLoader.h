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

#include <mutex>
#include <memory>
#include <string>
#include <unordered_map>

namespace c10 {
namespace npu {

/**
  FunctionLoader is used to store function address in the process.
  */
class FunctionLoader {
public:
  /**
    ctr
    */
  explicit FunctionLoader(const std::string& filename);
  /**
    dectr
    */
  ~FunctionLoader();
  /**
    set function name
    */
  void Set(const std::string& name);
  /**
    get function address by function name.
    */
  void* Get(const std::string& name);
private:
  mutable std::mutex mu_;
  std::string fileName;
  void* handle = nullptr;
  mutable std::unordered_map<std::string, void*> registry;
}; // class FunctionLoader


namespace register_function {
/**
  this class is used to register
  */
class FunctionRegister {
public:
  /**
    Singleton
    */
  static FunctionRegister* GetInstance();
  /**
    this API is used to store FunctionLoader class
    */
  void Register(const std::string& name, ::std::unique_ptr<FunctionLoader>& ptr);
  /**
    this API is used to associate library name and function name.
    */
  void Register(const std::string& name, const std::string& funcName);
  /**
    this API is used to get the function address by library and function name.
    */
  void* Get(const std::string& soName, const std::string& funcName);

private:
  FunctionRegister() = default;
  mutable std::mutex mu_;
  mutable std::unordered_map<std::string, ::std::unique_ptr<FunctionLoader>> registry;
}; // class FunctionRegister

/**
  FunctionRegisterBuilder is the helper of FunctionRegister.
  */
class FunctionRegisterBuilder {
public:
  /**
    ctr
    */
  FunctionRegisterBuilder(const std::string& name, ::std::unique_ptr<FunctionLoader>& ptr);
  /**
    ctr
    */
  FunctionRegisterBuilder(const std::string& soName, const std::string& funcName);
}; // class FunctionRegisterBuilder

} // namespace register_function

#define REGISTER_LIBRARY(soName)                                                \
  auto library_##soName =                                                       \
    ::std::unique_ptr<c10::npu::FunctionLoader>(new c10::npu::FunctionLoader(#soName));      \
  static c10::npu::register_function::FunctionRegisterBuilder                             \
    register_library_##soName(#soName, library_##soName);

#define REGISTER_FUNCTION(soName, funcName)                                     \
  static c10::npu::register_function::FunctionRegisterBuilder                             \
    register_function_##funcName(#soName, #funcName);

#define GET_FUNCTION(soName, funcName)                                              \
  c10::npu::register_function::FunctionRegister::GetInstance()->Get(#soName, #funcName);

} // namespace npu
} // namespace c10
