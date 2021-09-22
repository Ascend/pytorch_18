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

import copy
import sys
import torch
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestBatchNormBackwardReduce(TestCase):
    def cuda_op_exec(self, *args):
        cuda_sum_dy, cuda_sum_dy_xmu, cuda_grad_weight, cuda_grad_bias = \
                                        torch.batch_norm_backward_reduce(*args)
        return (cuda_sum_dy.cpu().numpy(), cuda_sum_dy_xmu.cpu().numpy(),
                cuda_grad_weight.cpu().numpy(), cuda_grad_bias.cpu().numpy())

    def cuda_expect_result(self):
        cpu_output0 = np.array([449.18185, 464.78906, 471.87485], dtype=np.float32)
        cpu_output1 = np.array([831.08484, 2112.0908, 259.91568], dtype=np.float32)
        cpu_output2 = np.array([6091.88, 3367.45, 1824.8948], dtype=np.float32)
        cpu_output3 = np.array([449.18185, 464.78906, 471.87485], dtype=np.float32)

        return cpu_output0, cpu_output1, cpu_output2, cpu_output3

    def npu_op_exec(self, *args):
        npu_sum_dy, npu_sum_dy_xmu, npu_grad_weight, npu_grad_bias = \
                                        torch.batch_norm_backward_reduce(*args)
        return (npu_sum_dy.cpu().numpy(), npu_sum_dy_xmu.cpu().numpy(),
                npu_grad_weight.cpu().numpy(), npu_grad_bias.cpu().numpy())

    def test_batch_norm_backward_reduce(self, device):
        shape_format = [
            [[np.float32, -1, [1, 3, 9, 9]], [np.float32, -1, [3]],
                                 True, True, True],
        ]
        for item in shape_format:
            cpu_grad_output, npu_grad_output = create_common_tensor(item[0], 1, 10)
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 10)
            cpu_mean, npu_mean = create_common_tensor(item[1], 1, 10)
            cpu_invstd, npu_invstd = create_common_tensor(item[1], 1, 10)
            cpu_weight, npu_weight = create_common_tensor(item[1], 1, 10)

            if torch.cuda.is_available():
                cpu_output = self.cuda_op_exec(cpu_grad_output.cuda(),
                                cpu_input1.cuda(), cpu_mean.cuda(),
                                cpu_invstd.cuda(), cpu_weight.cuda(),
                                *item[-3:])
            else:
                cpu_output = self.cuda_expect_result()
            npu_output = self.cuda_op_exec(npu_grad_output,
                                npu_input1, npu_mean,
                                npu_invstd, npu_weight,
                                *item[-3:])

            for cpu_out, npu_out in zip(cpu_output, npu_output):
                self.assertRtolEqual(cpu_out, npu_out)

instantiate_device_type_tests(TestBatchNormBackwardReduce, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.global_step_inc()
    run_tests()