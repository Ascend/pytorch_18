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
import sys
import torch
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestCross(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = torch.cross(input1, input2)
        output = output.numpy()
        return output
    
    def cpu_op_exec_dim(self, input1, input2, dim):
        output = torch.cross(input1, input2, dim)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.cross(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_dim(self, input1, input2, dim):
        output = torch.cross(input1, input2, dim)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_median_shape_format(self, device):
        shape_format_dim = [
            [[np.float32, -1, (4, 3)], 1],
            [[np.float32, -1, (4, 3, 3)], 1],
            [[np.float32, -1, (3, 2)], 0],
        ]
        for item in shape_format_dim:
            cpu_input_left, npu_input_left = create_common_tensor(item[0], -2, 2)
            cpu_input_right, npu_input_right = create_common_tensor(item[0], -2, 2)
           
            cpu_output = self.cpu_op_exec_dim(cpu_input_left, cpu_input_right, item[1])
            npu_output = self.npu_op_exec_dim(npu_input_left, npu_input_right, item[1])
            self.assertRtolEqual(cpu_output, npu_output)
        
        shape_format = [
            [[np.float32, -1, (5, 3)]],
            [[np.float32, -1, (5, 3, 4)]],
        ]
        for item in shape_format:
            cpu_input_left, npu_input_left = create_common_tensor(item[0], -2, 2)
            cpu_input_right, npu_input_right = create_common_tensor(item[0], -2, 2)
           
            cpu_output = self.cpu_op_exec(cpu_input_left, cpu_input_right)
            npu_output = self.npu_op_exec(npu_input_left, npu_input_right)
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestCross, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
