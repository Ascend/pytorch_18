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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor
import torch.nn.functional as F

class TestEmbedding(TestCase):
    def cpu_op_exec(self, weight, indices):
        weight.requires_grad_(True)
        out = F.embedding(indices, weight, scale_grad_by_freq=True, padding_idx=37)
        return out.detach().numpy()

    def npu_op_exec(self, weight, indices):
        weight.requires_grad_(True)
        out = F.embedding(indices, weight, scale_grad_by_freq=True, padding_idx=37)
        out_npu = out.to("cpu")
        return out_npu.detach().numpy()

    def test_shape_format(self, device):
        shape_format = [
                        [[np.float32, 0, [40,32]], [np.int64, 0, [40]]],
                        [[np.float32, 0, [40,1024]], [np.int64, 0, [40]]],
                        [[np.float32, 0, [40000,1024]], [np.int64, 0, [3125]]],
                        [[np.float32, 0, [40000,1024]], [np.int64, 0, [128,8]]],
                        [[np.float16, 0, [40,32]], [np.int64, 0, [40]]],
                        [[np.float16, 0, [40,1024]], [np.int64, 0, [128,8]]],
                        [[np.float16, 0, [33712,1024]], [np.int64, 0, [64,7]]],
                        [[np.float32, 3, [40,32]], [np.int64, 0, [40]]],
                        [[np.float32, 4, [40,1024]], [np.int64, 0, [40]]],
                        [[np.float32, 2, [40000,1024]], [np.int64, 0, [3125]]],
                        [[np.float32, 29, [40000,1024]], [np.int64, 0, [128,8]]],
                        [[np.float16, 3, [40,32]], [np.int64, 0, [40]]],
                        [[np.float16, 3, [40,1024]], [np.int64, 0, [128,8]]],
                        [[np.float16, 3, [33712,1024]], [np.int64, 0, [64,7]]]
                        ]
        for item in shape_format:
            weight_cpu, weight_npu = create_common_tensor(item[0], 1, 1)
            indices_cpu, indices_npu = create_common_tensor(item[1], 0, 1)

            if weight_cpu.dtype == torch.float16:
                weight_cpu = weight_cpu.to(torch.float32)

            cpu_out = self.cpu_op_exec(weight_cpu, indices_cpu)
            npu_out = self.npu_op_exec(weight_npu, indices_npu)
            cpu_out = cpu_out.astype(npu_out.dtype)

            self.assertRtolEqual(cpu_out, npu_out)

instantiate_device_type_tests(TestEmbedding, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
