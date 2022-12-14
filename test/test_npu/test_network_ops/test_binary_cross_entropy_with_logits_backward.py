# Copyright (c) 2020, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import copy
import torch.nn as nn
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


def generate_data(min, max, shape, dtype):
    input1 = np.random.uniform(min, max, shape).astype(dtype)
    # modify from numpy.ndarray to torch.tensor
    output = torch.from_numpy(input1)
    # generate target: target.size == input1.size
    label = torch.randint(shape[1], size=(shape[0],), dtype=torch.long)
    target = torch.zeros(shape[0], shape[1])
    target[range(target.shape[0]), label] = 1
    target = target.to(output.dtype)
    return output, target


class TestBinaryCrossEntropyWithLogitsBackward(TestCase):
    def cpu_op_exec(self, input1, target):
        input1.requires_grad_(True)
        output = torch.nn.functional.binary_cross_entropy_with_logits(input1, target)
        input_cpu = output.detach().numpy()
        output.backward()
        res = input1.grad
        res = res.numpy()
        return input_cpu, res

    def npu_op_exec(self, input1, target):
        target = target.to("npu")
        input1 = input1.to("npu")
        input1.requires_grad_(True)
        output = torch.nn.functional.binary_cross_entropy_with_logits(input1, target)
        input_npu = output.cpu()
        input_npu = input_npu.detach().numpy()
        output.backward()
        res = input1.grad.cpu()
        res = res.numpy()
        return input_npu, res

    def test_binary_cross_entropy_with_logits_backward_fp32(self, device):
        npu_input1, npu_target = generate_data(0, 100, (5, 3), np.float32)
        cpu_input1 = copy.deepcopy(npu_input1)
        cpu_target = copy.deepcopy(npu_target)
        cpu_output, cpu_grad_output = self.cpu_op_exec(cpu_input1, cpu_target)
        npu_output, npu_grad_output = self.npu_op_exec(npu_input1, npu_target)
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpu_grad_output, npu_grad_output)

    def test_binary_cross_entropy_with_logits_backward_fp16(self, device):
        npu_input1, npu_target = generate_data(0, 100, (5, 3), np.float16)
        cpu_input1 = copy.deepcopy(npu_input1)
        cpu_target = copy.deepcopy(npu_target)
        cpu_input1 = cpu_input1.to(torch.float32)
        cpu_target = cpu_target.to(torch.float32)
        cpu_output, cpu_grad_output = self.cpu_op_exec(cpu_input1, cpu_target)
        npu_output, npu_grad_output = self.npu_op_exec(npu_input1, npu_target)
        cpu_output = cpu_output.astype(npu_output.dtype)
        cpu_grad_output = cpu_grad_output.astype(npu_grad_output.dtype)
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpu_grad_output, npu_grad_output)


instantiate_device_type_tests(TestBinaryCrossEntropyWithLogitsBackward, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()

