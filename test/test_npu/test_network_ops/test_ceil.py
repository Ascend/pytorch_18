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

class TestCeil(TestCase):
    @dtypes(torch.float)
    def test_ceil(self, device, dtype):
        cpu_input = torch.randn(10, 10, dtype=torch.float, device="cpu").to(dtype=dtype)
        npu_input = cpu_input.to("npu")
        cpu_output = torch.ceil_(cpu_input)
        npu_output = torch.ceil_(npu_input)
        npu_output = npu_output.to("cpu")

        self.assertRtolEqual(cpu_output, npu_output)
       
    def cpu_op_exec(self, input):
        output = torch.ceil(input)
        return output

    def npu_op_exec(self, input):
        output = torch.ceil(input)
        output = output.to("cpu")
        return output

    def test_ceil_shape_format(self, device):
        shape_format = [
                [np.float32, 0,  10               ],
                [np.float32, 0,  (64, 10)         ],
                [np.float32, 3,  (256, 2048, 7, 7)],
                [np.float32, 4,  (32, 1, 3, 3)    ], 
                [np.float32, 29, (10, 128)        ],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestCeil, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
