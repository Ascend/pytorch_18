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

class TestIsfinite(TestCase):
    def test_isfinite(self, device):
        x = torch.Tensor([1, 2, -10]).to("npu")
        self.assertEqual(torch.isfinite(x).to("cpu"), torch.BoolTensor([True, True, True]))
    
    
    def cpu_op_exec(self, input):
        output = torch.isfinite(input)
        output = output.numpy()
        return output
    
    def npu_op_exec(self, input):
        output = torch.isfinite(input)
        output = output.to("cpu")
        output = output.numpy()
        return output
    
    def test_isfinite_shape_format(self, device):
        shape_format = [
                [np.int16, 0, (1, 2, 2, 5)],
                [np.int32, 0, (1, 4, 3)],
                [np.int64, 0, (2, 3)],
                [np.float32, 0, (8, 4, 3, 9)],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, -100, 100)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestIsfinite, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
