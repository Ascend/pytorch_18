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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestSoftShrink(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input_x = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input = torch.from_numpy(input_x)
        return npu_input

    def cpu_op_exec(self, input_x, lambd):
        output = torch.nn.functional.softshrink(input_x, lambd=lambd)
        output = output.numpy()
        return output.astype(np.float32)

    def npu_op_exec(self, input_x, lambd):
        input1 = input_x.to("npu")
        output = torch.nn.functional.softshrink(input1, lambd=lambd)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input_x, lambd, output_y):
        input_x = input_x.to("npu")
        output = output_y.to("npu")
        torch.nn.functional.softshrink(input_x, lambd=lambd, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_softshrink_3_3_float32(self, device):
        input_x1 = self.generate_data(-1, 1, (3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 0.5)
        npu_output1 = self.npu_op_exec(input_x1, 0.5)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_softshrink_100_100_float32(self, device):
        input_x1 = self.generate_data(-1, 1, (100, 100), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 0.5)
        npu_output1 = self.npu_op_exec(input_x1, 0.5)
        self.assertRtolEqual(cpu_output1, npu_output1)
    
    def test_softshrink_3_3_float16(self, device):
        input_x1 = self.generate_data(-1, 1, (3, 3), np.float16)
        input_x1_cpu = input_x1.float()
        cpu_output1 = self.cpu_op_exec(input_x1_cpu, 0.5).astype(np.float16)
        npu_output1 = self.npu_op_exec(input_x1, 0.5)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_softshrink_100_100_float16(self, device):
        input_x1 = self.generate_data(-1, 1, (100, 100), np.float16)
        input_x1_cpu = input_x1.float()
        cpu_output1 = self.cpu_op_exec(input_x1_cpu, 0.5).astype(np.float16)
        npu_output1 = self.npu_op_exec(input_x1, 0.5)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_softshrink_10_10_10_10_float32(self, device):
        input_x1 = self.generate_data(-1, 1, (10, 10, 10, 10), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 0.5)
        npu_output1 = self.npu_op_exec(input_x1, 0.5)
        self.assertRtolEqual(cpu_output1, npu_output1)
    
    def test_softshrink_3_3_out_float32(self, device):
        input_x1 = self.generate_data(-1, 1, (3, 3), np.float32)
        output_y = self.generate_data(-1, 1, (3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 0.5)
        npu_output1 = self.npu_op_exec_out(input_x1, 0.5, output_y)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_softshrink_100_100_out_float32(self, device):
        input_x1 = self.generate_data(-1, 1, (100, 100), np.float32)
        output_y = self.generate_data(-1, 1, (100, 100), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 0.5)
        npu_output1 = self.npu_op_exec_out(input_x1, 0.5, output_y)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_softshrink_10_10_10_10_out_float32(self, device):
        input_x1 = self.generate_data(-1, 1, (10, 10, 10, 10), np.float32)
        output_y = self.generate_data(-1, 1, (10, 10, 10, 10), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 0.5)
        npu_output1 = self.npu_op_exec_out(input_x1, 0.5, output_y)
        self.assertRtolEqual(cpu_output1, npu_output1)
        
instantiate_device_type_tests(TestSoftShrink, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()