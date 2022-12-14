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
import copy
import torch.nn as nn
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class Testcdist(TestCase):
    def generate_data(self, min_n, max_n, shape_x, shape_y, src_type):
        np.random.seed(10086)
        x1 = np.random.uniform(min_n, max_n, shape_x).astype(src_type)
        x2 = np.random.uniform(min_n, max_n, shape_y).astype(src_type)
        return x1, x2

    def op_exec(self, x1, x2, p, device='cpu'):
        is_fp16 = x1.dtype == np.float16
        if device == 'cpu' and is_fp16:
            x1 = x1.astype(np.float32)
            x2 = x2.astype(np.float32)

        x1 = torch.from_numpy(x1)
        x2 = torch.from_numpy(x2)

        x1 = x1.to(device)
        x2 = x2.to(device)

        y = torch.cdist(x1, x2, p)
        y = y.cpu().numpy()

        if device == 'cpu' and is_fp16:
            y = y.astype(np.float16)
        return y

    def test_cdist_float16_1(self, device):
        npu_input1, npu_input2 = self.generate_data(-1, 1,
                                                    (5, 64), (4, 64), np.float16)
        cpu_output = self.op_exec(npu_input1, npu_input2, 0.0, 'cpu')
        npu_output = self.op_exec(npu_input1, npu_input2, 0.0, 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cdist_float16_2(self, device):
        npu_input1, npu_input2 = self.generate_data(-1, 1,
                                                    (5, 10), (4, 10), np.float16)
        cpu_output = self.op_exec(npu_input1, npu_input2, 0.5, 'cpu')
        npu_output = self.op_exec(npu_input1, npu_input2, 0.5, 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cdist_float16_3(self, device):
        npu_input1, npu_input2 = self.generate_data(-1, 1,
                                                    (5, 10), (4, 10), np.float16)
        cpu_output = self.op_exec(npu_input1, npu_input2, 1.0, 'cpu')
        npu_output = self.op_exec(npu_input1, npu_input2, 1.0, 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cdist_float16_4(self, device):
        npu_input1, npu_input2 = self.generate_data(-1, 1,
                                                    (5, 10), (4, 10), np.float16)
        cpu_output = self.op_exec(npu_input1, npu_input2, 1.5, 'cpu')
        npu_output = self.op_exec(npu_input1, npu_input2, 1.5, 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cdist_float16_5(self, device):
        npu_input1, npu_input2 = self.generate_data(-1, 1,
                                                    (5, 10), (4, 10), np.float16)
        cpu_output = self.op_exec(npu_input1, npu_input2, 2.0, 'cpu')
        npu_output = self.op_exec(npu_input1, npu_input2, 2.0, 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cdist_float16_6(self, device):
        npu_input1, npu_input2 = self.generate_data(-1, 1,
                                                    (5, 10), (4, 10), np.float16)
        cpu_output = self.op_exec(npu_input1, npu_input2, 2.5, 'cpu')
        npu_output = self.op_exec(npu_input1, npu_input2, 2.5, 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cdist_float16_7(self, device):
        npu_input1, npu_input2 = self.generate_data(-1, 1,
                                                    (3, 5, 500), (4, 500), np.float16)
        cpu_output = self.op_exec(npu_input1, npu_input2, 2.0, 'cpu')
        npu_output = self.op_exec(npu_input1, npu_input2, 2.0, 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cdist_float32_1(self, device):
        npu_input1, npu_input2 = self.generate_data(-1, 1,
                                                    (5, 10), (4, 10), np.float32)
        cpu_output = self.op_exec(npu_input1, npu_input2, 0.0, 'cpu')
        npu_output = self.op_exec(npu_input1, npu_input2, 0.0, 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cdist_float32_2(self, device):
        npu_input1, npu_input2 = self.generate_data(-1, 1,
                                                    (5, 10), (4, 10), np.float32)
        cpu_output = self.op_exec(npu_input1, npu_input2, 0.5, 'cpu')
        npu_output = self.op_exec(npu_input1, npu_input2, 0.5, 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cdist_float32_3(self, device):
        npu_input1, npu_input2 = self.generate_data(-1, 1,
                                                    (5, 10), (4, 10), np.float32)
        cpu_output = self.op_exec(npu_input1, npu_input2, 1.0, 'cpu')
        npu_output = self.op_exec(npu_input1, npu_input2, 1.0, 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cdist_float32_4(self, device):
        npu_input1, npu_input2 = self.generate_data(-1, 1,
                                                    (5, 10), (4, 10), np.float32)
        cpu_output = self.op_exec(npu_input1, npu_input2, 1.5, 'cpu')
        npu_output = self.op_exec(npu_input1, npu_input2, 1.5, 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cdist_float32_5(self, device):
        npu_input1, npu_input2 = self.generate_data(-1, 1,
                                                    (5, 10), (4, 10), np.float32)
        cpu_output = self.op_exec(npu_input1, npu_input2, 2.0, 'cpu')
        npu_output = self.op_exec(npu_input1, npu_input2, 2.0, 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cdist_float32_6(self, device):
        npu_input1, npu_input2 = self.generate_data(-1, 1,
                                                    (5, 10), (4, 10), np.float32)
        cpu_output = self.op_exec(npu_input1, npu_input2, 2.5, 'cpu')
        npu_output = self.op_exec(npu_input1, npu_input2, 2.5, 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cdist_float32_7(self, device):
        npu_input1, npu_input2 = self.generate_data(-1, 1,
                                                    (5, 500), (3, 4, 500), np.float32)
        cpu_output = self.op_exec(npu_input1, npu_input2, 2.0, 'cpu')
        npu_output = self.op_exec(npu_input1, npu_input2, 2.0, 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cdist_float32_8(self, device):
        npu_input1, npu_input2 = self.generate_data(-100, 100,
                                                    (5, 100), (3, 4, 100), np.float32)
        cpu_output = self.op_exec(npu_input1, npu_input2, 2.5, 'cpu')
        npu_output = self.op_exec(npu_input1, npu_input2, 2.5, 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cdist_float32_9(self, device):
        npu_input1, npu_input2 = self.generate_data(-1000, 1000,
                                                    (5, 100), (3, 4, 100), np.float32)
        cpu_output = self.op_exec(npu_input1, npu_input2, 1.5, 'cpu')
        npu_output = self.op_exec(npu_input1, npu_input2, 1.5, 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cdist_float32_10(self, device):
        npu_input1, npu_input2 = self.generate_data(-0.1, 0.1,
                                                    (5, 100), (3, 4, 100), np.float32)
        cpu_output = self.op_exec(npu_input1, npu_input2, 2.5, 'cpu')
        npu_output = self.op_exec(npu_input1, npu_input2, 2.5, 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cdist_float32_11(self, device):
        npu_input1, npu_input2 = self.generate_data(-0.1, 0.1,
                                                    (5, 100), (3, 4, 100), np.float32)
        cpu_output = self.op_exec(npu_input1, npu_input2, 0.5, 'cpu')
        npu_output = self.op_exec(npu_input1, npu_input2, 0.5, 'npu')
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test_cdist_float32_12(self, device):
        npu_input1, npu_input2 = self.generate_data(-0.1, 0.1,
                                                    (16, 11, 17, 5, 84, 2), (16, 11, 17, 5, 84, 2), np.float32)
        cpu_output = self.op_exec(npu_input1, npu_input2, 2.0, 'cpu')
        npu_output = self.op_exec(npu_input1, npu_input2, 2.0, 'npu')
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test_cdist_float32_13(self, device):
        npu_input1, npu_input2 = self.generate_data(-0.1, 0.1,
                                                    (2, 2, 13, 39, 97, 14, 2, 7), (2, 2, 13, 39, 97, 14, 12, 7), np.float32)
        cpu_output = self.op_exec(npu_input1, npu_input2, 2.0, 'cpu')
        npu_output = self.op_exec(npu_input1, npu_input2, 2.0, 'npu')
        self.assertRtolEqual(cpu_output, npu_output)
    

instantiate_device_type_tests(Testcdist, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
