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
import torch.nn as nn
import numpy as np
from torch.testing._internal.common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestEmbeddingBagBackward(TestCase):
    def cpu_op_exec(self, input, embedding_matrix, offsets):
        m = nn.functional.embedding_bag(input, embedding_matrix, offsets)
        grads = torch.ones_like(m)
        m.backward(grads)

    def npu_op_exec(self, input, embedding_matrix, offsets):
        embedding_matrix.requires_grad = True
        m = nn.functional.embedding_bag(input, embedding_matrix, offsets).npu()
        grads = torch.ones_like(m)
        m.backward(grads)

    def test_embedding_bag_backward_shape_format(self, device):
        test_cases = [
            [torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]), torch.tensor([0, 4]), torch.randn(10, 3, requires_grad=True),
             torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]).to("npu"), torch.tensor([0, 4]).to("npu"),
             torch.randn(10, 3, requires_grad=True).to("npu")],
            [torch.tensor([1, 2, 4, 5, 4, 3, 2, 9, 6]), torch.tensor([0, 4]), torch.randn(10, 3, requires_grad=True),
             torch.tensor([1, 2, 4, 5, 4, 3, 2, 9, 6]).to("npu"), torch.tensor([0, 4]).to("npu"),
             torch.randn(10, 3, requires_grad=True).to("npu")],
            [torch.tensor([0, 2, 0, 5]), torch.tensor([0, 2]), torch.randn(10, 3, requires_grad=True),
             torch.tensor([0, 2, 0, 5]).to("npu"), torch.tensor([0, 2]).to("npu"),
             torch.randn(10, 3, requires_grad=True).to("npu")],
            [torch.tensor([0, 1, 2, 3]), torch.tensor([0, 3]), torch.randn(4, 3, requires_grad=True),
             torch.tensor([0, 1, 2, 3]).to("npu"), torch.tensor([0, 3]).to("npu"),
             torch.randn(4, 3, requires_grad=True).to("npu")],
            [torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]), torch.tensor([0, 4]), torch.randn(10, 4, requires_grad=True),
             torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]).to("npu"), torch.tensor([0, 4]).to("npu"),
             torch.randn(10, 4, requires_grad=True).to("npu")],
            [torch.tensor([0, 1, 2, 3]), torch.tensor([0, 3]), torch.randn(10, 4, requires_grad=True),
             torch.tensor([0, 1, 2, 3]).to("npu"), torch.tensor([0, 3]).to("npu"),
             torch.randn(10, 4, requires_grad=True).to("npu")]
        ]
        for item in test_cases:
            cpu_input, npu_input = item[0], item[3]
            cpu_offsets, npu_offsets = item[1], item[4]
            cpu_embedding_matrix, npu_embedding_matrix = item[2], item[5]
            self.cpu_op_exec(cpu_input, cpu_embedding_matrix, cpu_offsets)
            self.npu_op_exec(npu_input, npu_embedding_matrix, npu_offsets)
            self.assertEqual(cpu_embedding_matrix.grad, npu_embedding_matrix.grad)


instantiate_device_type_tests(TestEmbeddingBagBackward, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()