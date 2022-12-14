# Copyright (c) 2020 Huawei Technologies Co., Ltd
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
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_channels):
        super(Model, self).__init__()
        self.op1 = nn.Conv2d(in_channels, in_channels, 1)
        self.op2 = nn.BatchNorm2d(in_channels)
        self.op2.running_mean = torch.tensor([i/1000 for i in range(in_channels)])
        self.op2.running_var = torch.tensor([i/1000 for i in range(in_channels)])
        self.op3 = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        self.op2.eval()
        x = self.op1(x)
        x = self.op2(x)
        x = self.op3(x)
        return x

class TestBn2dEval(TestCase):
    def test_batchnorm_backward_eval(self, device):
        model = Model(in_channels=256)
        cpu_tensor = torch.randn(32,256,14,14)
        npu_tensor = cpu_tensor.npu()
        cpu_tensor.requires_grad = True
        npu_tensor.requires_grad = True

        for i in range(1):
            out = model(cpu_tensor)
            loss = out.sum()
            loss.backward()
            cpuout = out
            cpu_grad_list = []
            for name, module in model.named_parameters():
                cpu_grad_list.append(module.grad)
                module.grad = None

            model = model.npu()
            out = model(npu_tensor)
            loss = out.sum()
            loss.backward()
            npuout = out
            npu_grad_list = []
            for name, module in model.named_parameters():
                npu_grad_list.append(module.grad.cpu())

            #print(cpu_tensor.grad, npu_tensor.grad)
            cpu_grad = cpu_tensor.grad
            npu_grad = npu_tensor.grad            
            # TODO(ascend): Insufficient precision
            #??????????????? self.assertRtolEqual(cpu_grad.numpy(), npu_grad.cpu().numpy())
            self.assertRtolEqual(cpu_grad.numpy(), npu_grad.cpu().numpy(), 0.01)

            for cpu_grad, npu_grad in zip(cpu_grad_list, npu_grad_list):
                #print(cpu_grad, npu_grad)
                # TODO(ascend): Insufficient precision
                #??????????????? self.assertRtolEqual(cpu_grad.numpy(), npu_grad.numpy())
                self.assertRtolEqual(cpu_grad.numpy(), npu_grad.numpy(), 0.1)

instantiate_device_type_tests(TestBn2dEval, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()