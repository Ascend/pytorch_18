# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION. 
# All rights reserved.
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


class TestTan(TestCase):

    def generate_single_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        
        return npu_input1
    
    def cpu_op_exec(self, input):
        output = torch.tan(input)
        output = output.numpy()
        return output

    def cpu_op_exec_out(self, input1, input2):
        torch.tan(input1, out=input2)
        output = input2.numpy()
        return output

    def cpu_op_exec_self(self, input):
        torch.tan_(input)
        output = input.numpy()
        return output

    def npu_op_exec(self, input):
        input = input.to("npu")
        output = torch.tan(input)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2):
        input1 = input1.to("npu")
        output = input2.to("npu")
        torch.tan(input1, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_self(self, input):
        input = input.to("npu")
        torch.tan_(input)
        output = input.to("cpu")
        output = output.numpy()
        return output

    def test_tan_float32(self, device):
        input = self.generate_single_data(0, 6, (1,3), np.float32)
        cpu_output = self.cpu_op_exec(input)
        npu_output = self.npu_op_exec(input)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_tan_out_float32(self, device):
        input1 = self.generate_single_data(0, 6, (1, 3), np.float32)
        input2 = self.generate_single_data(0, 6, (1, 3), np.float32)
        cpu_output = self.cpu_op_exec_out(input1, input2)
        npu_output = self.npu_op_exec_out(input1, input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_tan_self_float32(self, device):
        input = self.generate_single_data(0, 6, (1, 3), np.float32)
        input2 = copy.deepcopy(input)
        cpu_output = self.cpu_op_exec_self(input)
        npu_output = self.npu_op_exec_self(input2)
        self.assertRtolEqual(cpu_output, npu_output)
        

instantiate_device_type_tests(TestTan, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
