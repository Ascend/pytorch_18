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
from common_utils import TestCase, run_tests
from common_device_type import instantiate_device_type_tests

class TestSymeig(TestCase):
    def op_exec(self, input1, eigenvectorsflag):
        npu_input = input1.npu()
        en, vn = torch.symeig(npu_input, eigenvectors = eigenvectorsflag)
        if eigenvectorsflag:
            ret = torch.matmul(vn, torch.matmul(en.diag_embed(), vn.transpose(-2, -1)))
            self.assertRtolEqual(ret.cpu(), input1, prec = 1e-3)
        else:
            e, v = torch.symeig(input1, eigenvectors = eigenvectorsflag)
            self.assertEqual(e, en.cpu())
            self.assertEqual(v, vn.cpu())
    
    def case_exec(self, input1):
        input1 = input1 + input1.transpose(-2, -1)
        self.op_exec(input1, False)
        self.op_exec(input1, True)

    def test_symeig_null(self, device):
        a = torch.randn(0, 0)
        self.op_exec(a, False)
        self.op_exec(a, True)

    def test_symeig_2d(self, device):
        a = torch.randn(5, 5, dtype = torch.float32)
        self.case_exec(a)

    def test_symeig_3d(self, device):
        a = torch.randn(10, 5, 5, dtype = torch.float32)
        self.case_exec(a)

    def test_symeig_4d(self, device):
        a = torch.randn(10, 3, 5, 5, dtype = torch.float32)
        self.case_exec(a)

    def test_symeig_5d(self, device):
        a = torch.randn(2, 10, 3, 5, 5, dtype = torch.float32)
        self.case_exec(a)

instantiate_device_type_tests(TestSymeig, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()