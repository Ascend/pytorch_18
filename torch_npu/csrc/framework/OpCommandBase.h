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
#ifndef __PULGIN_NATIVE_UTILS_COMMAND_BASE__
#define __PULGIN_NATIVE_UTILS_COMMAND_BASE__

#include <c10/npu/OptionsManager.h>
#include <ATen/npu/Exceptions.h>

#include "torch_npu/csrc/aten/mirror/NPUTensorIterator.h"
#include "torch_npu/csrc/framework/OpCmdHelper.h"
#include "torch_npu/csrc/framework/OpParamMaker.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/framework/allocator/THNPUCachingHostAllocator.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu
{
  namespace native
  {

    // get common dtype and shape from op adapter layer
    struct UnifiedResult
    {
      c10::optional<at::ScalarType> common_type = c10::nullopt;
      c10::optional<c10::IntArrayRef> common_shape = c10::nullopt;
      // judge result tensor's dtype is defined or not.
      // if result's dtype is defined, result_type_defined is true and result's dtype remains unchanged.
      bool result_type_defined = false;
    };

    template <class Derived>
    class OpCommandBase
    {
    public:
      explicit OpCommandBase()
      {
        aclCmds = OpCommandImpls::GetInstance();
        aclCmds->Push(aclCmd);
      }
      virtual ~OpCommandBase() {}

      Derived &Name(string name)
      {
        aclCmd->SetName(name);
        return static_cast<Derived &>(*this);
      }

      Derived &Expect(UnifiedResult unified_result)
      {
        commonType = unified_result.common_type;
        resultTypeDefined = unified_result.result_type_defined;
        commonShape = unified_result.common_shape;
        return static_cast<Derived &>(*this);
      }

      template <typename dataType>
      Derived &Attr(string name, dataType value)
      {
        aclCmd->AddAttr(name, value);
        return static_cast<Derived &>(*this);
      }

      Derived &Input()
      {
        return AddNoneTensor();
      }

      Derived &Input(
          const at::Tensor &input,
          const string &descName = "",
          const string &realData = "")
      {
        return AddTensorInput(Contiguous(input), at::ScalarType::Undefined, descName, realData);
      }

      Derived &Input(
          const at::Tensor &cpuTensor,
          c10::SmallVector<int64_t, N> dimList,
          const string &descName = "")
      {
        at::Tensor npuTensor = CopyHostToDevice(cpuTensor);
        aclCmd->AddConst(dimList);
        return AddTensorInput(npuTensor, at::ScalarType::Undefined, descName, "", cpuTensor);
      }

      Derived &Input(c10::SmallVector<int64_t, N> &dimList,
                     at::ScalarType toType = at::kLong)
      {

        at::Tensor &cpuTensor = CreateHostTensor((void *)dimList.data(),
                                                 dimList.size(),
                                                 c10::TensorOptions(at::kCPU).dtype(at::kLong),
                                                 toType);
        return AddHostTensorInput(cpuTensor);
      }

      Derived &Input(c10::IntArrayRef &dimListRef,
                     at::ScalarType toType = at::kLong)
      {

        at::Tensor &cpuTensor = CreateHostTensor((void *)dimListRef.data(),
                                                 dimListRef.size(),
                                                 c10::TensorOptions(at::kCPU).dtype(at::kLong),
                                                 toType);
        return AddHostTensorInput(cpuTensor);
      }

      Derived &Input(const c10::Scalar &input, const at::ScalarType type, MemoryType memoryType = MemoryType::MEMORY_DEVICE)
      {
        if (memoryType == MemoryType::MEMORY_DEVICE)
        {
          return AddScalarInput(input, type);
        }
        else
        {
          auto scalarTensor = CreateScalarTensor(input, type);
          return AddHostTensorInput(scalarTensor);
        }
      }

      Derived &Output(at::Tensor &output, const string &realType = "")
      {
        return AddOutput(output, realType);
      }

      void Run()
      {
        if (c10::npu::OptionsManager::CheckQueueEnable())
        {
          ExecuteParas params;
          aclCmd->ExportParams(params);
          c10::npu::enCurrentNPUStream(&params);
          aclCmd->releaseSource(false);
        }
        else
        {
          aclCmd->Run();
          aclCmd->releaseSource();
        }
        aclCmds->Pop();
      }

    protected:
      Derived &AddTensorInput(at::Tensor &tensor,
                              at::ScalarType forceScaleType = at::ScalarType::Undefined,
                              const string &descName = "", const string &realData = "",
                              c10::optional<at::Tensor> cpu_tensor = c10::nullopt)
      {
        std::tuple<aclTensorDesc *, aclDataBuffer *, int64_t, aclFormat> res;
        if (commonType.has_value() && commonType.value() != tensor.scalar_type())
        {
          tensor = NPUNativeFunctions::npu_dtype_cast(tensor, commonType.value());
        }
        // 针对dim=0的场景，绝对不会有输入为uint16的情况，因为这个是TBE引入的，TBE没有dim=0的情况
        if (tensor.dim() == 0)
        {
          if (tensor.is_npu())
          {
            res = OpCmdHelper::CovertNPUTensorWithZeroDimToAclInput(tensor, descName);
          }
          else
          {
            res = OpCmdHelper::CovertTensorWithZeroDimToAclInput(tensor, forceScaleType);
          }
        }
        else
        {
          res = OpCmdHelper::CovertTensorToAclInput(tensor, cpu_tensor, descName, realData);
        }
        aclCmd->AddInput(
            std::get<0>(res), std::get<1>(res), std::get<2>(res), std::get<3>(res));
        return static_cast<Derived &>(*this);
      }
      Derived &AddHostTensorInput(const at::Tensor &tensor)
      {
        std::tuple<aclTensorDesc *, aclDataBuffer *, int64_t, aclFormat> res;
        res = OpCmdHelper::CovertHostTensorToAclInput(tensor, tensor.scalar_type());
        aclCmd->AddInput(
            std::get<0>(res), std::get<1>(res), std::get<2>(res), std::get<3>(res), tensor);
        return static_cast<Derived &>(*this);
      }
      Derived &AddNoneTensor()
      {
        AclTensorDescMaker desc;
        auto aclDesc = desc.Create(ACL_DT_UNDEFINED, ACL_FORMAT_UNDEFINED).Get();
        AclTensorBufferMaker buffer(nullptr, 0);
        aclCmd->AddInput(aclDesc, buffer.Get(), 0, ACL_FORMAT_UNDEFINED);
        return static_cast<Derived &>(*this);
      }
      Derived &AddScalarInput(const c10::Scalar &input,
                              at::ScalarType type)
      {
        at::ScalarType type_bk = type;
        if (commonType.has_value())
        {
          type_bk = commonType.value();
        }
        at::Tensor aclInput = CopyHostToDevice(input, type_bk);
        auto res = OpCmdHelper::CovertScalarToAclInput(aclInput, type_bk);
        aclCmd->AddInput(
            std::get<0>(res), std::get<1>(res), std::get<2>(res), std::get<3>(res));
        return static_cast<Derived &>(*this);
      }
      Derived &AddOutput(at::Tensor &output, const string &realType = "")
      {
        if (resultTypeDefined == false && commonType.has_value() && commonType.value() != output.scalar_type())
        {
          output = NPUNativeFunctions::npu_dtype_cast(output, commonType.value());
        }
        const at::Tensor *tensor = &output;
        auto res = OpCmdHelper::CovertToAclOutput(tensor, realType);
        aclCmd->AddOutput(
            std::get<0>(res), std::get<1>(res), std::get<2>(res), std::get<3>(res));
        return static_cast<Derived &>(*this);
      }

    protected:
      // 由于format_contiguous会生成新Tensor，为了保证其在生命周期内有效，故而放到对象中存储
      // 同下，CopyScalarToDevice也有同样问题
      at::Tensor &Contiguous(const at::Tensor &input)
      {
        storage.emplace_back(NpuUtils::format_contiguous_add_copy_optimize(input));
        return storage.back();
      }
      at::Tensor CopyHostToDevice(const c10::Scalar &scalar, at::ScalarType type)
      {
        auto tensor = scalar_to_tensor(scalar).to(type);
        return CopyHostToDevice(tensor);
      }
      at::Tensor CopyHostToDevice(const at::Tensor &cpuTensor)
      {
        at::Tensor cpuPinMemTensor = cpuTensor.pin_memory();
        int deviceIndex = 0;
        AT_NPU_CHECK(aclrtGetDevice(&deviceIndex));
        auto tensor = cpuPinMemTensor.to(
            c10::Device(c10::DeviceType::NPU, deviceIndex),
            cpuPinMemTensor.scalar_type(),
            true,
            true);
        storage.emplace_back(tensor);
        return storage.back();
      }

      at::Tensor &CreateHostTensor(void *data, size_t size,
                                   const c10::TensorOptions &options, at::ScalarType toType)
      {

        AT_ASSERT(options.dtype() == at::kLong);
        auto cpuTensor = at::empty(size, options);
        AT_ASSERT(cpuTensor.is_contiguous());
        std::memcpy(cpuTensor.data_ptr(), data, sizeof(int64_t) * cpuTensor.numel());
        if (toType != at::kLong)
        {
          cpuTensor = cpuTensor.to(toType);
        }

        storage.emplace_back(std::move(cpuTensor));
        return storage.back();
      }
      at::Tensor CreateScalarTensor(const c10::Scalar &scalar, const at::ScalarType type)
      {
        storage.emplace_back(scalar_to_tensor(scalar).to(type));
        return storage.back();
      }
      c10::SmallVector<at::Tensor, N> storage; // tensor's life cycle should maintain when Run() is called

    protected:
      OpCommandImpls *aclCmds = nullptr; // owned
      OpCommandImpl *aclCmd = nullptr;

    private:
      c10::optional<at::ScalarType> commonType = c10::nullopt;
      c10::optional<c10::IntArrayRef> commonShape = c10::nullopt;
      bool resultTypeDefined = false;

    }; // class OpCommandBase

  } // namespace native
} // namespace at_npu

#endif