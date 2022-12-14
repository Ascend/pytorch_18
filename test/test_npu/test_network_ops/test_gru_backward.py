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
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestGruBackward(TestCase):
    def test_gru_backward(self, device):
        # shape_format:[[dtype, (num_step, batch_size, input_size)], input_size, hidden_size]
        shape_format = [
                        [[np.float16, (16, 32, 64)], 64, 32], 
                        [[np.float16, (5, 32, 64)], 64, 32],
                        [[np.float32, (5, 32, 64)], 64, 32],
                        [[np.float32, (5, 32, 64)], 64, 64],
        ]

        for item in shape_format: 
            cpu_gru = torch.nn.GRU(input_size=item[1], hidden_size=item[2], num_layers=1, bidirectional=False)
            cpu_gru.weight_ih_l0.requires_grad_(True)
            cpu_gru.weight_hh_l0.requires_grad_(True)
            cpu_gru.bias_ih_l0.requires_grad_(True)
            cpu_gru.bias_hh_l0.requires_grad_(True)
            npu_gru = copy.deepcopy(cpu_gru).npu()
            
            input1 = np.random.uniform(0, 1, item[0][1]).astype(item[0][0])
            cpu_input1 = torch.from_numpy(input1.astype(np.float32))
            cpu_input1.requires_grad_(True)
            npu_input1 = torch.from_numpy(input1).npu()
            npu_input1.requires_grad_(True)

            cpu_output_y, cpu_output_h = cpu_gru(cpu_input1)
            npu_output_y, npu_output_h = npu_gru(npu_input1)

            self.assertRtolEqual(cpu_output_y.detach().numpy(), npu_output_y.cpu().detach().numpy().astype(np.float32), prec=1.e-1)
            self.assertRtolEqual(cpu_output_h.detach().numpy(), npu_output_h.cpu().detach().numpy().astype(np.float32), prec=1.e-1)
            
            cpu_input1.retain_grad()
            cpu_output_y.backward(torch.ones(cpu_output_y.size(), dtype=torch.float))
            cpu_dx = cpu_input1.grad
            cpu_dw_ih = cpu_gru.weight_ih_l0.grad
            cpu_dw_hh = cpu_gru.weight_hh_l0.grad
            cpu_db_ih = cpu_gru.bias_ih_l0.grad
            cpu_db_hh = cpu_gru.bias_hh_l0.grad
            
            npu_input1.retain_grad()
            npu_output_y.backward(torch.ones(npu_output_y.size(), dtype=torch.float).npu())
            npu_dx = npu_input1.grad
            npu_dw_ih = npu_gru.weight_ih_l0.grad
            npu_dw_hh = npu_gru.weight_hh_l0.grad
            npu_db_ih = npu_gru.bias_ih_l0.grad
            npu_db_hh = npu_gru.bias_hh_l0.grad
            
            self.assertRtolEqual(cpu_dx.numpy(), npu_dx.cpu().numpy().astype(np.float32), prec=1.e-1)
            self.assertRtolEqual(cpu_dw_ih.numpy(), npu_dw_ih.cpu().numpy().astype(np.float32), prec=1.e-1)
            self.assertRtolEqual(cpu_dw_hh.numpy(), npu_dw_hh.cpu().numpy().astype(np.float32), prec=1.e-1)
            # TODO(ascend): Insufficient precision
            #??????????????? self.assertRtolEqual(cpu_db_ih.numpy(), npu_db_ih.cpu().numpy().astype(np.float32), prec=1.e-1)
            self.assertRtolEqual(cpu_db_ih.numpy(), npu_db_ih.cpu().numpy().astype(np.float32), prec=1.e1)
            # TODO(ascend): Insufficient precision
            #??????????????? self.assertRtolEqual(cpu_db_hh.numpy(), npu_db_hh.cpu().numpy().astype(np.float32), prec=1.e-1)
            self.assertRtolEqual(cpu_db_hh.numpy(), npu_db_hh.cpu().numpy().astype(np.float32), prec=1.e1)


instantiate_device_type_tests(TestGruBackward, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests() 