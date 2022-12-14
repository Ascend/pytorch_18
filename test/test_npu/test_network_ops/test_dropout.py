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
sys.path.append('..')
import torch
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestDropOutDoMask(TestCase):
    def cpu_op_exec(self, input):
        out = torch.nn.Dropout(0.5)(input)
        out = out.numpy()
        return out

    def npu_op_exec(self, input):
        out = torch.nn.Dropout(0.5)(input)
        out = out.to("cpu")
        out = out.numpy()
        return out

    def dropout_list_exec(self, list):
        epsilon = 1e-3
        for item in list:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            cpu_output = cpu_output.astype(npu_output.dtype)
            # 该算子随机结果的比较方式
            for a, b in zip(cpu_output.flatten(), npu_output.flatten()):
                if abs(a) > 0 and abs(b) > 0 and abs(a - b) > epsilon:
                    print(f'input = {item}, ERROR!')
                    break
            else:
                print(f'input = {item}, Successfully!')

    def test_op_shape_format_fp16(self, device):
        format_list = [0, 3, 29]
        shape_list = [1, (256, 1280), (32, 3, 3), (256, 2048, 7, 7)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        self.dropout_list_exec(shape_format)

    def test_op_shape_format_fp32(self, device):
        format_list = [0, 3, 29]
        shape_list = [1, (256, 1280), (32, 3, 3), (256, 2048, 7, 7)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        self.dropout_list_exec(shape_format)

instantiate_device_type_tests(TestDropOutDoMask, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()