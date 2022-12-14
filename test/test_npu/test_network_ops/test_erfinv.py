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

from torch import device
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestErfinv(TestCase):
    def cpu_op_exec(self, input_data):
        output = torch.erfinv(input_data)
        output = output.numpy()
        return output

    def npu_op_exec(self, input_data):
        output = torch.erfinv(input_data)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_out(self,input1,cpu_out):
        torch.erfinv(input1, out = cpu_out)
        output = cpu_out.numpy()
        return output

    def npu_op_exec_out(self,input1,npu_out):
        torch.erfinv(input1, out = npu_out)
        output = npu_out.to("cpu")
        output = output.numpy()
        return output
    
    def cpu_op_exec_(self, input1):
        input1.erfinv_()
        output = input1.numpy()
        return output

    def npu_op_exec_(self, input1):
        input1 = input1.to("npu")
        input1.erfinv_()
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def test_erfinv_shape_format(self, device):
        shape_format = [
            [np.float32, -1, (2, 3, 4, 5)],
            [np.float32, -1, (4, 5, 6, 7)],
            [np.float32, -1, (2, 3, 4, 5, 6)],
            [np.float16, -1, (2, 3, 4, 5)],
            [np.float16, -1, (4, 5, 6, 7)],
            [np.float16, -1, (2, 3, 4, 5, 6)]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -0.5, 0.5)
            if item[0] == np.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            if item[0] == np.float16:
                cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output, prec=1e-3)
    
    def test_erfinv_out_shape_format(self, device):
        shape_format = [
            [np.float32, -1, (2, 3, 4, 5)],
            [np.float32, -1, (4, 5, 6, 7)],
            [np.float32, -1, (2, 3, 4, 5, 6)],
            [np.float16, -1, (2, 3, 4, 5)],
            [np.float16, -1, (4, 5, 6, 7)],
            [np.float16, -1, (2, 3, 4, 5, 6)]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -0.5, 0.5)
            cpu_out, npu_out = create_common_tensor(item, -0.5, 0.5)
            if item[0] == np.float16:
                cpu_input = cpu_input.to(torch.float32)
                cpu_out = cpu_out.to(torch.float32)
            cpu_output = self.cpu_op_exec_out(cpu_input, cpu_out)
            npu_output = self.npu_op_exec_out(npu_input, npu_out)
            if item[0] == np.float16:
                cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output, prec=1e-3)

    def test_erfinv__shape_format(self, device):
        shape_format = [
            [np.float32, -1, (2, 3, 4, 5)],
            [np.float32, -1, (4, 5, 6, 7)],
            [np.float32, -1, (2, 3, 4, 5, 6)],
            [np.float16, -1, (2, 3, 4, 5)],
            [np.float16, -1, (4, 5, 6, 7)],
            [np.float16, -1, (2, 3, 4, 5, 6)]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -0.5, 0.5)
            if item[0] == np.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_exec_(cpu_input)
            npu_output = self.npu_op_exec_(npu_input)
            if item[0] == np.float16:
                cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output, prec=1e-3)
    

instantiate_device_type_tests(TestErfinv, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
