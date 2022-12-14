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

class TestCol2ImBackward(TestCase):

    def cpu_op_exec(self,input1, output_size, ksizes, strides, dilates, padding):
        output = torch._C._nn.col2im(input1, output_size, ksizes, dilates, padding, strides)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1,output_size, ksizes, strides, dilates,padding):
        output = torch._C._nn.col2im(input1, output_size, ksizes, dilates, padding, strides)
        output = output.to("cpu") 
        output = output.numpy()       
        return output

    def test_col2im_shape_format(self, device):
        shape_format = [
               [ [np.float32, 0, (4,4)], (4,5), (2,2), (2,2), (1,1), (0,0)],
               [ [np.float32, 3, (2, 8,30 )], (4,5), (2,2), (1,1), (1,1), (1,1)],
               [ [np.float32, 4, ( 12, 20)], (12,6), (2,3), (1,1), (2,2), (0,0)],
               [ [np.float32, 29, ( 1,12, 12)], (4,5), (2,2), (1,1), (1,1), (0,0)],
               [ [np.float16, 29, ( 1,12, 12)], (4,5), (2,2), (1,1), (1,1), (0,0)],
        ]
         
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 20)
            cpu_output = self.cpu_op_exec(cpu_input, item[1], item[2], item[3], item[4], item[5])
            npu_output = self.npu_op_exec(npu_input, item[1], item[2], item[3], item[4], item[5])
            self.assertRtolEqual(cpu_output, npu_output)
           


instantiate_device_type_tests(TestCol2ImBackward, globals(), except_for="cpu")
if __name__ == "__main__":
    torch.npu.set_device("npu:5")
    run_tests()