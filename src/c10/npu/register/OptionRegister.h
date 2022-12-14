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

#ifndef __C10_NPU_OPTION_REGISTER_H__
#define __C10_NPU_OPTION_REGISTER_H__

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <c10/util/Optional.h>

namespace c10 {
namespace npu {

typedef void(*OptionCallBack) (const std::string&);
/**
  This class is used to storage env value, and provide Set and Get to
  */
class OptionInterface {
 public:
  /**
    dctr
    */
    OptionInterface(OptionCallBack callback=nullptr);
  /**
    This API is used to store value.
    */
  void Set(const std::string& in);
  /**
    This API is used to load value.
    */
  std::string Get();
 private:
/**
  Its used to store hook.
  */
  OptionCallBack callback = nullptr; 
  std::string val;
};

namespace register_options {

/**
  This class is used to register OptionInterface
  */
class OptionRegister {
 public:
  /**
    dctr
    */
  ~OptionRegister() = default;
  /**
    singleton
    */
  static OptionRegister* GetInstance();
  /**
    register
    */
  void Register(const std::string& name, ::std::unique_ptr<OptionInterface>& ptr);
  /**
    This API is used to store value to special key.
    */
  void Set(const std::string& name, const std::string& val);
  /**
    This API is used to load value from special key.
    */
  c10::optional<std::string> Get(const std::string& name);
 private:
  OptionRegister() {}
  mutable std::mutex mu_;
  mutable std::unordered_map<std::string, ::std::unique_ptr<OptionInterface>> registry;
};

/**
  This class is the helper to construct class OptionRegister
  */
class OptionInterfaceBuilder {
 public:
  OptionInterfaceBuilder(const std::string& name, ::std::unique_ptr<OptionInterface>& ptr, const std::string& type = "cli");
};

} // namespace register_options

/**
  This API is used to store key-value pairs
  */
void SetOption(const std::map<std::string, std::string>& options);
/**
  This API is used to store key-value pair
  */
void SetOption(const std::string& key, const std::string& val);
/**
  This API is used to load value by key
  */
c10::optional<std::string> GetOption(const std::string& key);

#define REGISTER_OPTION(name)                                       \
  REGISTER_OPTION_UNIQ(name, name, cli)

#define REGISTER_OPTION_INIT_BY_ENV(name)                           \
  REGISTER_OPTION_UNIQ(name, name, env)

#define REGISTER_OPTION_UNIQ(id, name, type)                        \
  auto options_interface_##id =                                     \
      ::std::unique_ptr<c10::npu::OptionInterface>(new c10::npu::OptionInterface());    \
  static c10::npu::register_options::OptionInterfaceBuilder                             \
      register_options_interface_##id(#name, options_interface_##id, #type);

#define REGISTER_OPTION_HOOK(name, ...)                                       \
  REGISTER_OPTION_HOOK_UNIQ(name, name, __VA_ARGS__)

#define REGISTER_OPTION_HOOK_UNIQ(id, name, ...)                                \
  auto options_interface_##id =                                                 \
      ::std::unique_ptr<c10::npu::OptionInterface>(                             \
        new c10::npu::OptionInterface(c10::npu::OptionCallBack(__VA_ARGS__)));  \
  static c10::npu::register_options::OptionInterfaceBuilder                     \
      register_options_interface_##id(#name, options_interface_##id);

#define REGISTER_OPTION_BOOL_FUNCTION(func, key, defaultVal, trueVal)  \
  bool func() {                                                     \
    auto val = c10::npu::GetOption(#key);                           \
    if (val.value_or(defaultVal) == trueVal) {                      \
      return true;                                                  \
    }                                                               \
    return false;                                                   \
  }

} // namespace npu
} // namespace c10

#endif // __C10_NPU_OPTION_REGISTER_H__