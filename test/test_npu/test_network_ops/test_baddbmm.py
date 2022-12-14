# Copyright (c) 2020 Huawei Technologies Co., Ltd
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

class TestBaddBmm(TestCase):
    def generate_scalar(self, dtype, min, max):
        if dtype == "float32":
            scalar = np.random.uniform(min, max)
        if dtype == "float16":
            scalar = np.random.uniform(min, max)
        if dtype == "int32":
            scalar = np.random.randint(min, max)
        return scalar

    def cpu_op_exec(self, input1, input2, input3, scalar1, scalar2):
        output = torch.baddbmm(input1, input2, input3, beta=scalar1, alpha=scalar2)
        output = output.numpy()
        return output
        
    def cpu_op_exec_(self, input1, input2, input3, scalar1, scalar2):
        input1.baddbmm_(input2, input3, beta=scalar1, alpha=scalar2)
        input1 = input1.numpy()
        return input1

    def npu_op_exec(self, input1, input2, input3, scalar1, scalar2):
        output = torch.baddbmm(input1, input2, input3, beta=scalar1, alpha=scalar2)
        output = output.to("cpu")
        output = output.numpy()
        return output 

    def npu_op_exec_(self, input1, input2, input3, scalar1, scalar2):
        input1.baddbmm_(input2, input3, beta=scalar1, alpha=scalar2)
        input1 = input1.to("cpu")
        input1 = input1.numpy()
        return input1   

    def test_baddbmm_common_shape_format(self, device):
        shape_format = [
                [[np.float32, -1, (1, 3, 5)], [np.float32, -1, (1, 3, 4)], [np.float32, -1, (1, 4, 5)], "float32"],
                [[np.float32, -1, (6, 4, 3)], [np.float32, -1, (6, 4, 5)], [np.float32, -1, (6, 5, 3)], "float32"],
                [[np.float32, -1, (175, 455, 22)], [np.float32, -1, (175, 455, 116)], [np.float32, -1, (175, 116, 22)], "float32"],
                [[np.float32, -1, (25, 56, 12)], [np.float32, -1, (25, 56, 51)], [np.float32, -1, (25, 51, 12)], "float32"]
        ] 

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 1, 10)
            cpu_input3, npu_input3 = create_common_tensor(item[2], 1, 10)
            scalar1 = self.generate_scalar(item[3], 0, 10)
            scalar2 = self.generate_scalar(item[3], 0, 10)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, cpu_input3, scalar1, scalar2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_input3, scalar1, scalar2)
            self.assertRtolEqual(cpu_output, npu_output)
            cpu_output_ = self.cpu_op_exec_(cpu_input1, cpu_input2, cpu_input3, scalar1, scalar2)
            npu_output_ = self.npu_op_exec_(npu_input1, npu_input2, npu_input3, scalar1, scalar2)
            self.assertRtolEqual(cpu_output_, npu_output_)
            
    def test_baddbmm_float16_shape_format(self, device):
        def cpu_op_exec_fp16(input1, input2, input3, scalar1, scalar2):
            input1 = input1.to(torch.float32)
            input2 = input2.to(torch.float32)
            input3 = input3.to(torch.float32)
            output = torch.baddbmm(input1, input2, input3, beta=scalar1, alpha=scalar2)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        shape_format = [
                [[np.float16, -1, (1, 3, 5)], [np.float16, -1, (1, 3, 4)], [np.float16, -1, (1, 4, 5)], "float16"],
                [[np.float16, -1, (500, 40, 300)], [np.float16, -1, (500, 40, 500)], [np.float16, -1, (500, 500, 300)], "float16"],
                [[np.float16, -1, (175, 455, 22)], [np.float16, -1, (175, 455, 116)], [np.float16, -1, (175, 116, 22)], "float16"],
                [[np.float16, -1, (25, 21, 11)], [np.float16, -1, (25, 21, 34)], [np.float16, -1, (25, 34, 11)], "float16"],
        ] 

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 1, 10)
            cpu_input3, npu_input3 = create_common_tensor(item[2], 1, 10)
            scalar1 = self.generate_scalar(item[3], 0, 10)
            scalar2 = self.generate_scalar(item[3], 0, 10)
            cpu_output = cpu_op_exec_fp16(cpu_input1, cpu_input2, cpu_input3, scalar1, scalar2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_input3, scalar1, scalar2)
            self.assertRtolEqual(cpu_output, npu_output)  
      

instantiate_device_type_tests(TestBaddBmm, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
