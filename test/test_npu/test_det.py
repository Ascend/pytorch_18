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

# pylint: disable=unused-variable, unused-argument

class TestDet(TestCase):
    def generate_data(self, min_val, max_val, shape, dtype):
        input1 = np.random.uniform(min_val, max_val, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)

        return npu_input1

    def cpu_op_exec(self, input1):
        output = torch.det(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        input1 = input1.to("npu")
        output = torch.det(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_det_float32(self, device):
        npu_input1 = self.generate_data(-9.313225746154785e-10, 9.313225746154785e-10, (1, 1, 64, 64), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_det_float16(self, device):
        npu_input1 = self.generate_data(0, 0, (2, 2, 32, 32), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1.float()).astype(np.float16)
        npu_output = self.npu_op_exec(npu_input1.float()).astype(np.float16)
        print(cpu_output,npu_output,'123')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_big_scale_float32(self, device):
        npu_input1 = self.generate_data(0, 10, (32, 32), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestDet, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:7")
    run_tests()
