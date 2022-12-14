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
import copy

from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class Testcast_Half(TestCase):

    def generate_single_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        # modify from numpy.ndarray to torch.tensor
        npu_input1 = torch.from_numpy(input1)
        return npu_input1
    def cpu_op_exec(self, input1):
        output = torch._cast_Half(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        input1 = input1.to("npu")
        output = torch._cast_Half(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output
    def test_cast_Half_float16(self, device):
        def cpu_op_exec_fp16(input1):
            input1 = input1.to(torch.float32)
            output = torch._cast_Half(input1)
            output = output.numpy()
            output = output.astype(np.float16)
            return output
        npu_input1 = self.generate_single_data(0, 100, (5,3), np.float16)
        cpu_output = cpu_op_exec_fp16(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cast_Half_float32(self, device):
        npu_input1 = self.generate_single_data(0, 100, (4,3), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cast_Half_int32(self, device):
        npu_input1 = self.generate_single_data(0, 100, (4,3), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)
        
    def test_cast_Half_int8(self, device):
        npu_input1 = self.generate_single_data(0, 100, (4,3,2), np.int8)
        cpu_output = self.cpu_op_exec(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cast_Half_uint8(self, device):
        npu_input1 = self.generate_single_data(0, 100, (4,3,2), np.uint8)
        cpu_output = self.cpu_op_exec(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)
        
instantiate_device_type_tests(Testcast_Half, globals(), except_for='cpu')
if __name__ == '__main__':
    run_tests()