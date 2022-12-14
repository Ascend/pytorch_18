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

from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestFill_(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = torch.fill_(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.fill_(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_fills_shape_format_fp16(self, device):
        format_list = [0, 3]
        shape_list = [[1024], [32, 1024], [32, 8, 1024], [128, 32, 8, 1024]]
        value_list = [0.8, 1.25, torch.tensor(0.8), torch.tensor(1.25)]
        shape_format = [
            [[np.float16, i, j], v] for i in format_list for j in shape_list for v in value_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1])
            npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_fill_shape_format_fp32(self, device):
        format_list = [0, 3]
        shape_list = [[1024], [32, 1024], [32, 8, 1024], [128, 32, 8, 1024]]
        value_list = [0.8, 1.25, torch.tensor(0.8), torch.tensor(1.25)]
        shape_format = [
            [[np.float32, i, j], v] for i in format_list for j in shape_list for v in value_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1])
            npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestFill_, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
