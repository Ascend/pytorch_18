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

class TestBincount(TestCase):
    def cpu_op_exec(self, x): 
        output = torch.bincount(x) 
        return output.numpy()
     
    def npu_op_exec(self, x): 
        output = torch.bincount(x)
        output = output.cpu()
        return output.numpy()

    def cpu_op_exec_with_weights(self, x, weights): 
        output = torch.bincount(x, weights) 
        return output.numpy() 
     
    def npu_op_exec_with_weights(self, x, weights): 
        output = torch.bincount(x, weights)
        output = output.cpu()
        return output.numpy()
     
    def test_bincount(self, device):
        input_param = [
            [np.int8, -1, [0]],
            [np.int8, -1, [np.random.randint(1, 65536)]],
            [np.int16, -1, [0]],
            [np.int16, -1, [np.random.randint(1, 65536)]],
            [np.int32, -1, [0]],
            [np.int32, -1, [np.random.randint(1, 65536)]],
            [np.int64, -1, [0]],
            [np.int64, -1, [np.random.randint(1, 65536)]],
        ]
        weight_dtype = [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32]
        shape_format = [[x, y] for x in input_param for y in weight_dtype]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

            cpu_weight, npu_weight = create_common_tensor([item[1]] + item[0][1:], 0, 100)
            cpu_output = self.cpu_op_exec_with_weights(cpu_input, cpu_weight)
            npu_output = self.npu_op_exec_with_weights(npu_input, npu_weight)
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestBincount, globals(), except_for='cpu')     
if __name__ == '__main__': 
    run_tests() 