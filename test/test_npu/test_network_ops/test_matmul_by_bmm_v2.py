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

import sys
import torch
import numpy as np
import torch.nn as nn
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor
import os
os.environ["BMMV2_ENABLE"] = "1"

class TestMatMulByBmmV2(TestCase):
    def op_exec(self, npu_flag, mat1, mat2):
        input1 = mat1
        input2 = mat2
        input1.requires_grad = True
        input2.requires_grad = True

        output = torch.matmul(input1, input2)
        tmp = torch.ones_like(output)
        output.backward(tmp)
        if npu_flag:
            npuOutput = output.cpu()
            return npuOutput.detach().cpu().numpy(), input1.grad.cpu().numpy(), input2.grad.cpu().numpy()
        return output.detach().numpy(), input1.grad.numpy(), input2.grad.numpy()

    def matmul_backward_result(self, shape_format):
        for item in shape_format:
            mat1_cpu, mat1_npu = create_common_tensor(item[0], -10, 10)
            if mat1_cpu.dtype == torch.float16:
                mat1_cpu = mat1_cpu.to(torch.float32)
            mat2_cpu, mat2_npu = create_common_tensor(item[1], -10, 10)
            if mat2_cpu.dtype == torch.float16:
                mat2_cpu = mat2_cpu.to(torch.float32)
            cpu_output, cpu_mat1_grad, cpu_mat2_grad = self.op_exec(0, mat1_cpu, mat2_cpu)
            npu_output, npu_mat1_grad, npu_mat2_grad = self.op_exec(1, mat1_npu, mat2_npu)

            self.assertRtolEqual(cpu_output.astype(npu_output.dtype), npu_output)
            self.assertRtolEqual(cpu_mat1_grad.astype(npu_mat1_grad.dtype), npu_mat1_grad)
            self.assertRtolEqual(cpu_mat2_grad.astype(npu_mat2_grad.dtype), npu_mat2_grad)

    def test_matmul_backward_shape_format_fp16_case1(self, device):
        shape_format = [  # mat1 1dim, mat2 1dim       
            [[np.float16, 2, [5]], [np.float16, 2, [5]]],
            [[np.float16, 2, [2560]], [np.float16, 2, [2560]]]
        ]
        self.matmul_backward_result(shape_format)

    def test_matmul_backward_shape_format_fp16_case3(self, device):
        shape_format = [  # mat1 1dim, mat2 2dim       
            [[np.float16, 2, [5]], [np.float16, 2, [5,6]]],
            [[np.float16, 2, [2560]], [np.float16, 2, [2560,4680]]],
            [[np.float16, 2, [10]], [np.float16, 2, [10,20]]],
            [[np.float16, 2, [5]], [np.float16, 2, [5,5]]]
        ]
        self.matmul_backward_result(shape_format)
        
instantiate_device_type_tests(TestMatMulByBmmV2, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
