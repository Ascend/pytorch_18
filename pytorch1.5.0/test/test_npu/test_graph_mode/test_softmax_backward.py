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
from graph_utils import graph_mode


def input_grad_hook(grad):
    global input_grad
    input_grad = grad


def npu_input_grad_hook(grad):
    global npu_input_grad
    npu_input_grad = grad.to("cpu")


class TestSoftmaxBackward(TestCase):

    def cpu_op_exec(self, input_data, is_contiguous=True, dim=-1):
        if is_contiguous is False:
            input_data = input_data.as_strided([2, 2], [1, 2], 1)
        input_data.requires_grad = True
        input_data.register_hook(input_grad_hook)

        output = torch.softmax(input_data, dim=dim)
        z = output.sum()
        z.backward()

    def npu_op_exec(self, input_data, is_contiguous=True, dim=-1):
        if is_contiguous is False:
            input_data = input_data.as_strided([2, 2], [1, 2], 1)
        input_data.requires_grad = True
        input_data.register_hook(npu_input_grad_hook)

        output = torch.softmax(input_data, dim=dim)
        z = output.sum()
        z.backward()
        input_data = input_data.cpu()

    @graph_mode
    def test_softmax_backward_shape_format(self, device):
        shape_format = [
            [np.float32, 0, 5],
            [np.float32, 3, (64, 10)],
            [np.float32, 3, (256, 2048, 7, 7)],
            [np.float32, 3, (32, 1, 3, 3)],
            [np.float32, 0, (10, 128)]
        ]
        for item in shape_format:
            input1, npu_input1 = create_common_tensor(item, 10, 100)
            input2, npu_input2 = create_common_tensor(item, 10, 100)

            self.cpu_op_exec(input1)
            self.npu_op_exec(npu_input1)
            self.assertRtolEqual(input_grad.numpy(), npu_input_grad.numpy())

            '''
            self.cpu_op_exec(input2, False)
            self.npu_op_exec(npu_input2, False)
            self.assertRtolEqual(input_grad.numpy(), npu_input_grad.numpy())
            '''

    @graph_mode
    def test_softmax_backward_shape_format_fp16(self, device):
        shape_format = [
            [np.float16, 0, 5],
            [np.float16, 3, (64, 10)],
            [np.float16, 3, (256, 2048, 7, 7)],
            [np.float16, 3, (32, 1, 3, 3)],
            [np.float16, 0, (10, 128)]
        ]
        for item in shape_format:
            input1, npu_input1 = create_common_tensor(item, 10, 100)
            input2, npu_input2 = create_common_tensor(item, 10, 100)

            input1 = input1.to(torch.float32)
            input2 = input2.to(torch.float32)

            self.cpu_op_exec(input1)
            self.npu_op_exec(npu_input1)

            self.assertRtolEqual(input_grad.numpy().astype(np.float16), npu_input_grad.numpy())

            '''
            self.cpu_op_exec(input2, False)
            self.npu_op_exec(npu_input2, False)
            self.assertRtolEqual(input_grad.numpy().astype(np.float16), npu_input_grad.numpy())
            '''


instantiate_device_type_tests(TestSoftmaxBackward, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()