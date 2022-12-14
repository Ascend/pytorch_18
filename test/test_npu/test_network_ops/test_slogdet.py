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

class TestSlogdet(TestCase):
    def cpu_op_exec(self, input):
        sign, logabsdet = torch.slogdet(input)
        sign = sign.numpy()
        logabsdet = logabsdet.numpy()
        return sign, logabsdet

    def npu_op_exec(self, input):
        sign, logabsdet = torch.slogdet(input)
        sign = sign.cpu()
        logabsdet = logabsdet.cpu()
        sign = sign.numpy()
        logabsdet = logabsdet.numpy()
        return sign, logabsdet

    def test_slogdet_shape_format(self, device):
        shape_format = [
                [np.float32, -1, (3, 3)],
                [np.float32, -1, (4, 3, 3)],
                [np.float32, -1, (5, 5, 5, 5)],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -100, 100)
            cpu_output, cpu_indices = self.cpu_op_exec(cpu_input)
            npu_output, npu_indices = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_indices, npu_indices)



instantiate_device_type_tests(TestSlogdet, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
