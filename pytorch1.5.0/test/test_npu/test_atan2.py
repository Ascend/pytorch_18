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

class TestAtan2(TestCase):

    def cpu_op_exec(self,input1, input2):
        output = torch.atan2(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec(self,input1, input2):
        output = torch.atan2(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self,input1, input2, out):
        torch.atan2(input1, input2, out=out)
        output = out.to("cpu")
        output = output.numpy()
        return output
    def test_atan2_common_shape_format(self, device):
        shape_format = [
                [[np.float32, 0, (4)],   [np.float32, 0, (4)]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -1, 1)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -1, 1)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_atan2_out_common_shape_format(self, device):
        shape_format = [
                [[np.float32, 0, (4)],    [np.float32, 0, (4)]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -1, 1)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -1, 1)
            cpu_out, npu_out = create_common_tensor(item[0], -1, 1)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec_out(npu_input1, npu_input2, npu_out)
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestAtan2, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:6")
    run_tests()