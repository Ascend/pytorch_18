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

class TestNonzero(TestCase):
    def cpu_op_exec(self, input):
        output = torch.nonzero(input)
        output = output.numpy().astype(np.int32)
        return output

    def npu_op_exec(self, input):
        output = torch.nonzero(input)
        output = output.to("cpu")
        output = output.numpy().astype(np.int32)
        return output

    def test_nonzero_shape_format(self, device):
        dtype_list = [np.float32, np.float16, np.int32, np.int64]
        format_list = [0]
        shape_list = [[256,10], [256,256,100],[5,256,256,100]]

        shape_format = [
                [[i, j, k]] for i in dtype_list for j in format_list for k in shape_list
        ]
        for item in shape_format:
            print(item)
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestNonzero, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()