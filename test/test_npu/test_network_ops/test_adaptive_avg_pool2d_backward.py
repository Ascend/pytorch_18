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
import numpy as np
import sys
from torch.nn import functional as F
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestAdaptiveAvgPool2dBackward(TestCase):

    def cpu_op_exec(self, input_x, input_grad):
        input_x.requires_grad_(True)
        m = torch.nn.AdaptiveAvgPool2d(input_grad)
        output = m(input_x)
        output.backward(output)
        out = input_x.grad
        return out

    def npu_op_exec(self, input_x, input_grad):
        input_x.requires_grad_(True)
        m = torch.nn.AdaptiveAvgPool2d(input_grad)
        output = m(input_x)
        output.backward(output)
        out = input_x.grad.cpu()
        return out

    def test_adaptiveAvgPool2d_backward_1(self, device):
        cpu_input = torch.randn((1, 8, 9), dtype=torch.float32)
        npu_input = cpu_input
        output_size = np.array((2, 3))
        cpu_output = self.cpu_op_exec(cpu_input, output_size)
        npu_output = self.npu_op_exec(npu_input, output_size)
        self.assertRtolEqual(cpu_output.numpy(), npu_output.numpy())
        
    def test_adaptiveAvgPool2d_backward_2(self, device):
        cpu_input = torch.randn((1, 3, 3, 3), dtype=torch.float32)
        npu_input = cpu_input
        output_size = np.array((2, 2))
        cpu_output = self.cpu_op_exec(cpu_input, output_size)
        npu_output = self.npu_op_exec(npu_input, output_size)
        
        self.assertRtolEqual(cpu_output.numpy(), npu_output.numpy())

    def test_adaptiveAvgPool2d_backward_fp16(self, device):
        input_x = np.random.uniform(0, 1, (1, 3, 6, 6)).astype(np.float16)
        cpu_input = torch.from_numpy(input_x)
        npu_input = cpu_input
        output_size = np.array((5, 5))
        cpu_input = cpu_input.to(torch.float32)
        cpu_output = self.cpu_op_exec(cpu_input, output_size)
        npu_output = self.npu_op_exec(npu_input, output_size)
        cpu_output = cpu_output.to(torch.float16)
        self.assertRtolEqual(cpu_output.numpy(), npu_output.numpy())
        
instantiate_device_type_tests(TestAdaptiveAvgPool2dBackward, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
