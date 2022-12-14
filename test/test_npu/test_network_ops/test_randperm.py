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
from torch.testing._internal.common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestRandperm(TestCase):
    def cpu_op_exec(self, input, dtype):
        output = torch.randperm(input, dtype=dtype, device='cpu')
        output = output.sum()
        return output.numpy()

    def npu_op_exec(self, input, dtype):
        output = torch.randperm(input, dtype=dtype, device='npu')
        output = output.sum()
        output = output.cpu()
        return output.numpy()

    def test_randperm_shape_format(self, device):
        for n in (10, 25, 123, 1000):
            for dtype in (torch.long, torch.float):
                cpu_output = self.cpu_op_exec(n, dtype)
                npu_output = self.npu_op_exec(n, dtype)
                self.assertEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestRandperm, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()