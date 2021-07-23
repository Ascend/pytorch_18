# Copyright (c) 2020, Huawei Technologies.
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

class TestRreluWithNoise(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = torch._C._nn.rrelu_with_noise(input1, input2, 0.1, 0.3)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch._C._nn.rrelu_with_noise(input1, input2, 0.1, 0.3)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_leaky_relu_shape_format(self, device):
        shape_format = [
               [ [np.float32, 0, (1, 6, 4)], [np.float32, 0, (1, 4, 8)], [np.float32, 0, (1, 6, 8)]],
               [ [np.float32, 3, (2, 4, 5)], [np.float32, 3, (2, 5, 10)],  [np.float32, 3, (2, 4, 10)]]
               ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_leaky_relu_shape_format_fp16(self, device):
        shape_format = [
               [ [np.float16, 0, (1, 6, 4)]],
               [ [np.float16, 3, (2, 4, 5)]]
               ]
        def cpu_op_exec_fp16(input1, input2):
            input1 = input1.to(torch.float32)
            input2 = input2.to(torch.float32)
            output = torch._C._nn.rrelu_with_noise(input1, input2, 0.1, 0.3)
            output = output.numpy()
            output = output.astype(np.float16)
            return output
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 1, 100)
            cpu_output = cpu_op_exec_fp16(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestRreluWithNoise, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()


