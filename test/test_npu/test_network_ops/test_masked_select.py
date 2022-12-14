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
import sys
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestMaskedSelect(TestCase):
    def cpu_op_exec(self, input, mask):
        output = torch.masked_select(input, mask)
        output = output.numpy()
        return output

    def npu_op_exec(self, input, mask):
        mask = mask.to("npu")
        output = torch.masked_select(input, mask)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_maskedselect_shape_format_maskdiff(self, device):
        dtype_list = [np.int64, np.int32, np.float32]
        format_list = [0]
        shape_list = [[3, 4, 5]]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            mask_cpu, mask_npu = create_common_tensor((np.int32, 0, (3, 4, 1)), 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input, mask_cpu > 50)
            npu_output = self.npu_op_exec(npu_input, mask_npu > 50)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_maskedselect_shape_format_fp32(self, device):
        format_list = [0, 3]
        shape_list = [[3, 4, 5]]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        mask = torch.tensor([[
         [ True, False,  True,  True, False],
         [ True, False, False,  True, False],
         [False, False, False, False, False],
         [ True, False, False, False, False]],

        [[ True, False, False, False,  True],
         [False,  True, False,  True,  True],
         [False,  True, False,  True,  True],
         [False, False, False, False, False]],

        [[False,  True,  True, False,  True],
         [False,  True,  True,  True,  True],
         [False,  True, False,  True, False],
         [False,  True,  True, False, False]]])

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input, mask)
            npu_output = self.npu_op_exec(npu_input, mask)
            self.assertRtolEqual(cpu_output, npu_output)
            
    def test_maskedselect_shape_format_int(self, device):
        dtype_list = [np.int32, np.int64]
        format_list = [0]
        shape_list = [[3, 4, 5]]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
        mask = torch.tensor([[
         [ True, False,  True,  True, False],
         [ True, False, False,  True, False],
         [False, False, False, False, False],
         [ True, False, False, False, False]],

        [[ True, False, False, False,  True],
         [False,  True, False,  True,  True],
         [False,  True, False,  True,  True],
         [False, False, False, False, False]],

        [[False,  True,  True, False,  True],
         [False,  True,  True,  True,  True],
         [False,  True, False,  True, False],
         [False,  True,  True, False, False]]])

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input, mask)
            npu_output = self.npu_op_exec(npu_input, mask)
            self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestMaskedSelect, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()