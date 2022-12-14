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
import copy
import torch.nn as nn
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

LOWER = 0
UPPER = 1

class TestBinaryCrossEntropy(TestCase):

    def generate_input(self, lower, upper, shape, dtype):
        temp = np.random.uniform(lower, upper, shape).astype(dtype)
        npu_input = torch.from_numpy(temp)
        return npu_input

    def cpu_op_exec(self, predict, target, weight=None, reduction="mean"):
        res = torch.nn.functional.binary_cross_entropy(predict, target, weight=weight, reduction=reduction)
        return res.numpy()

    def cpu_op_exec_half(self, predict, target, weight=None, reduction="mean"):
        res = torch.nn.functional.binary_cross_entropy(predict, target, weight=weight, reduction=reduction)
        return res.type(torch.float16).numpy()

    def npu_op_exec(self, predict, target, weight=None, reduction="mean"):
        predict = predict.to("npu")
        target = target.to("npu")
        if weight is not None:
            weight = weight.to("npu")
        res = torch.nn.functional.binary_cross_entropy(predict, target, weight=weight, reduction=reduction)
        res = res.to("cpu")
        return res.numpy()

    def test_binary_cross_entropy_float32(self, device):
        for shape, weight_shape, reduction in [
           ((10, 64), None,     "mean"),
           ((10, 64), (10, 1),  "mean"),
           ((10, 64), None,     "mean"),
           ((10, 64), (10, 64), "mean"),
           ((10, 64), (10, 64), "sum" ),
           ((10, 64), (10, 64), "none")
        ]:
            predict = self.generate_input(LOWER, UPPER, shape, np.float32)
            target  = torch.empty(shape, dtype=torch.float32).random_(2)
            weight  = None
            if weight_shape is not None:
                weight = self.generate_input(LOWER, UPPER, weight_shape, np.float32)
            cpu_output = self.cpu_op_exec(predict, target, weight=weight, reduction=reduction)
            npu_output = self.npu_op_exec(predict, target, weight=weight, reduction=reduction)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_binary_cross_entropy_float16(self, device):
        for shape, weight_shape, reduction in [
           ((10, 64), (10, 64), "sum"),
           ((10, 64), (10, 64), "mean"),
           ((10, 64), (10, 64), "none")
        ]:
            predict = self.generate_input(LOWER, UPPER, shape, np.float16)
            target = torch.empty(shape, dtype=torch.float16).random_(2)
            predict_32 = predict.type(torch.float32)
            target_32 = target.type(torch.float32)
            weight = None
            weight_32 = None
            if weight_shape is not None:
                weight = self.generate_input(LOWER, UPPER, weight_shape, np.float16)
                weight_32 = weight.type(torch.float32)

            npu_output = self.npu_op_exec(predict, target, weight=weight, reduction=reduction)
            cpu_output = self.cpu_op_exec_half(predict_32, target_32, weight=weight_32, reduction=reduction)
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestBinaryCrossEntropy, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
