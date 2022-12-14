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

class TestMkldnnMaxPool2D(TestCase):
    def generate_single_data(self, min_val, max_val, shape, dtype): 
        input1 = np.random.uniform(min_val, max_val, shape).astype(dtype) 
        npu_input1 = torch.from_numpy(input1) 
        return npu_input1 
     
    def cpu_op_exec(self, input_data): 
        input_cpu = input_data.to('cpu')
        output = torch.max_pool2d(input_cpu,2)
        #print("cpu data:   ", input1, '\n')
        return output 
        
    def npu_op_exec(self, input_data): 
        input_npu = input_data.to('npu')
        output = torch.max_pool2d(input_npu, 2)
        output = output.to("cpu")
        return output

    def test_maxpool_float32_1(self,device):
        input_data = self.generate_single_data(0,100,(1, 5, 5, 5),np.float32)
        cpu_output = self.cpu_op_exec(input_data)
        npu_output = self.npu_op_exec(input_data)

        self.assertRtolEqual(cpu_output, npu_output)
        
    def test_maxpool_float32_2(self,device):
        input_data = self.generate_single_data(-1000, 1000,(1, 20, 30, 30),np.float32)
        cpu_output = self.cpu_op_exec(input_data)
        npu_output = self.npu_op_exec(input_data)
        self.assertRtolEqual(cpu_output, npu_output)
        
    def test_maxpool_float32_3(self,device):
        input_data = self.generate_single_data(-10, 10,(20, 100, 30, 50),np.float32)
        cpu_output = self.cpu_op_exec(input_data)
        npu_output = self.npu_op_exec(input_data)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_maxpool_float32_4(self,device):
        input_data = self.generate_single_data(-1000, 10,(50, 10, 1000, 60),np.float32)
        cpu_output = self.cpu_op_exec(input_data)
        npu_output = self.npu_op_exec(input_data)
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test_maxpool_float16_1(self,device):
        input_data = self.generate_single_data(-100, 10, (200, 10, 100, 60), np.float16)
        input_data = input_data.to(torch.float32)
        cpu_output = self.cpu_op_exec(input_data)
        npu_output = self.npu_op_exec(input_data)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_maxpool_float16_2(self,device):
        input_data = self.generate_single_data(-100, 50, (20, 10, 5, 10), np.float16)
        input_data = input_data.to(torch.float32)
        cpu_output = self.cpu_op_exec(input_data)
        npu_output = self.npu_op_exec(input_data)
        self.assertRtolEqual(cpu_output, npu_output)
        
instantiate_device_type_tests(TestMkldnnMaxPool2D, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:2")
    run_tests()
