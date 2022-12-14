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

import torch.nn.functional as F
import copy
import torch.nn as nn
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor




class Testcdist(TestCase):
    def generate_data(self, min_n, max_n, shape_predict, shape_label, src_type):
        np.random.seed(10086)
        predict = np.random.uniform(min_n, max_n, 
                                    shape_predict).astype(src_type)
        label = np.random.uniform(min_n, max_n, shape_label).astype(src_type)
        label[label < 0] = -1
        label[label >= 0] = 1
        dout = np.ones(shape_predict).astype(src_type)
        return predict, label, dout

    def op_exec(self, predict, label, dout, reduction, device='cpu'):
        is_fp16 = predict.dtype == np.float16
        if device == 'cpu' and is_fp16:
            predict = predict.astype(np.float32)
            label = label.astype(np.float32)
            dout = dout.astype(np.float32)

        predict = torch.from_numpy(predict)
        label = torch.from_numpy(label)
        dout = torch.from_numpy(dout)
        
        predict = predict.to(device)
        label = label.to(device)
        dout = dout.to(device)

        predict.requires_grad = True

        output_forward = F.soft_margin_loss(predict, label, reduction=reduction)
        if reduction == 'none':
            output_forward.backward(dout)
        else:
            output_forward.backward()

        gradient = predict.grad.cpu().numpy()
        

        if device == 'cpu' and is_fp16:
            gradient = gradient.astype(np.float16)
        return gradient

    def test_soft_margin_loss_backward_float16_1(self, device):
        input1, input2, input3 = self.generate_data(-1, 1,
                                                    (100,), (100,), np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'none','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'none','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_2(self, device):
        input1, input2, input3 = self.generate_data(-1, 1,
                                                    (100,), (100,), np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'mean','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'mean','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_3(self, device):
        input1, input2, input3 = self.generate_data(-1, 1,
                                                    (100,), (100,), np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'sum','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'sum','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_4(self, device):
        input1, input2, input3 = self.generate_data(-0.1, 0.1,
                                                    (100, 200), (100, 200),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'none','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'none','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_5(self, device):
        input1, input2, input3 = self.generate_data(-0.1, 0.1,
                                                    (100, 200), (100, 200),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'mean','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'mean','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_6(self, device):
        input1, input2, input3 = self.generate_data(-0.1, 0.1,
                                                    (100, 200), (100, 200),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'sum','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'sum','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_7(self, device):
        input1, input2, input3 = self.generate_data(-10, 10,
                                                    (100, 20, 30), 
                                                    (100, 20, 1),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'none','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'none','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_8(self, device):
        input1, input2, input3 = self.generate_data(-10, 10,
                                                    (100, 20, 30),
                                                    (100, 20, 1),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'mean','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'mean','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_9(self, device):
        input1, input2, input3 = self.generate_data(-10, 10,
                                                    (100, 20, 30),
                                                    (100, 20, 1),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'sum','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'sum','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_10(self, device):
        input1, input2, input3 = self.generate_data(-0.01, 0.01,
                                                    (100, 20, 30), 
                                                    (100, 20, 30),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'none','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'none','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_11(self, device):
        input1, input2, input3 = self.generate_data(-0.01, 0.01,
                                                    (100, 20, 30),
                                                    (100, 20, 30),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'mean','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'mean','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_12(self, device):
        input1, input2, input3 = self.generate_data(-0.01, 0.01,
                                                    (100, 20, 30),
                                                    (100, 20, 30),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'sum','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'sum','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_13(self, device):
        input1, input2, input3 = self.generate_data(-0.001, 0.001,
                                                    (10, 20, 30, 4), 
                                                    (10, 20, 30, 4),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'none','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'none','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_14(self, device):
        input1, input2, input3 = self.generate_data(-0.001, 0.001,
                                                    (10, 20, 30, 4),
                                                    (10, 20, 30, 4),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'mean','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'mean','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_15(self, device):
        input1, input2, input3 = self.generate_data(-0.001, 0.001,
                                                    (10, 20, 30, 4),
                                                    (10, 20, 30, 4),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'sum','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'sum','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_16(self, device):
        input1, input2, input3 = self.generate_data(-0.001, 0.001,
                                                    (100, 20, 3, 4, 5), 
                                                    (100, 20, 3, 4, 5),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'none','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'none','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_17(self, device):
        input1, input2, input3 = self.generate_data(-0.001, 0.001,
                                                    (100, 20, 3, 4, 5),
                                                    (100, 20, 3, 4, 5),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'mean','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'mean','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_18(self, device):
        input1, input2, input3 = self.generate_data(-0.001, 0.001,
                                                    (10, 20, 3, 4, 5),
                                                    (10, 20, 3, 4, 5),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'sum','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'sum','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_1(self, device):
        input1, input2, input3 = self.generate_data(-1, 1,
                                                    (100,), (100,), np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'none','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'none','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_2(self, device):
        input1, input2, input3 = self.generate_data(-1, 1,
                                                    (100,), (100,), np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'mean','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'mean','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_3(self, device):
        input1, input2, input3 = self.generate_data(-1, 1,
                                                    (100,), (100,), np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'sum','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'sum','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_4(self, device):
        input1, input2, input3 = self.generate_data(-0.1, 0.1,
                                                    (100, 200), (100, 200),
                                                    np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'none','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'none','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_5(self, device):
        input1, input2, input3 = self.generate_data(-0.1, 0.1,
                                                    (100, 200), (100, 200),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'mean','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'mean','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_6(self, device):
        input1, input2, input3 = self.generate_data(-0.1, 0.1,
                                                    (100, 200), (100, 200),
                                                    np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'sum','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'sum','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_7(self, device):
        input1, input2, input3 = self.generate_data(-10, 10,
                                                    (100, 20, 30), 
                                                    (100, 20, 30),
                                                    np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'none','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'none','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_8(self, device):
        input1, input2, input3 = self.generate_data(-10, 10,
                                                    (100, 20, 30),
                                                    (100, 20, 30),
                                                    np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'mean','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'mean','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_9(self, device):
        input1, input2, input3 = self.generate_data(-10, 10,
                                                    (100, 20, 30),
                                                    (100, 20, 30),
                                                    np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'sum','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'sum','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_10(self, device):
        input1, input2, input3 = self.generate_data(-0.01, 0.01,
                                                    (100, 20, 30), 
                                                    (100, 20, 30),
                                                    np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'none','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'none','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_11(self, device):
        input1, input2, input3 = self.generate_data(-0.01, 0.01,
                                                    (100, 20, 30),
                                                    (100, 20, 30),
                                                    np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'mean','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'mean','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_12(self, device):
        input1, input2, input3 = self.generate_data(-0.01, 0.01,
                                                    (100, 20, 30),
                                                    (100, 20, 30),
                                                    np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'sum','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'sum','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_13(self, device):
        input1, input2, input3 = self.generate_data(-0.001, 0.001,
                                                    (10, 20, 30, 4), 
                                                    (10, 20, 30, 4),
                                                    np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'none','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'none','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_14(self, device):
        input1, input2, input3 = self.generate_data(-0.001, 0.001,
                                                    (10, 20, 30, 4),
                                                    (10, 20, 30, 4),
                                                    np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'mean','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'mean','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_15(self, device):
        input1, input2, input3 = self.generate_data(-0.001, 0.001,
                                                    (10, 20, 30, 4),
                                                    (10, 20, 30, 4),
                                                    np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'sum','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'sum','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_16(self, device):
        input1, input2, input3 = self.generate_data(-0.001, 0.001,
                                                    (100, 20, 3, 4, 5), 
                                                    (100, 20, 3, 4, 5),
                                                    np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'none','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'none','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_17(self, device):
        input1, input2, input3 = self.generate_data(-0.001, 0.001,
                                                    (100, 20, 3, 4, 5),
                                                    (100, 20, 3, 4, 5),
                                                    np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'mean','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'mean','npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_18(self, device):
        input1, input2, input3 = self.generate_data(-0.001, 0.001,
                                                    (100, 20, 3, 4, 5),
                                                    (100, 20, 3, 4, 5),
                                                    np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'sum','cpu')
        npu_output = self.op_exec(input1, input2, input3, 'sum','npu')
        self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(Testcdist, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()