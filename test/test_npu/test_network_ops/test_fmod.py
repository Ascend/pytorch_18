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


class TestFmod(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = torch.fmod(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.fmod(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2, input3):
        torch.fmod(input1, input2, out=input3)
        output = input3.to("cpu")
        output = output.numpy()
        return output

    def case_exec_tensor(self, shape):
        for item in shape:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 1, 100)
            npu_input3 = torch.empty(0).npu().to(cpu_input1.dtype)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            npu_output_out = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3)
            self.assertEqual(cpu_output, npu_output)
            self.assertEqual(npu_output_out, npu_output)

    def case_exec_scalar(self, shape):
        for item in shape:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            npu_input3 = torch.empty(0).npu().to(cpu_input1.dtype)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1])
            npu_output = self.npu_op_exec(npu_input1, item[1])
            npu_output_out = self.npu_op_exec_out(npu_input1, item[1], npu_input3)
            self.assertEqual(cpu_output, npu_output)
            self.assertEqual(npu_output_out, npu_output)

    def case_exec_tensor_fp16(self, shape):
        for item in shape:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 1, 100)
            npu_input3 = torch.empty(0).npu().to(cpu_input1.dtype)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            npu_output_out = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3)
            cpu_output = cpu_output.astype(np.float16)
            self.assertEqual(cpu_output, npu_output)
            self.assertEqual(npu_output_out, npu_output)

    def case_exec_scalar_fp32(self, shape):
        for item in shape:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            npu_input3 = torch.empty(0).npu().to(cpu_input1.dtype)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1])
            npu_output = self.npu_op_exec(npu_input1, item[1])
            npu_output_out = self.npu_op_exec_out(npu_input1, item[1], npu_input3)
            cpu_output = cpu_output.astype(np.float16)
            self.assertEqual(cpu_output, npu_output)
            self.assertEqual(npu_output_out, npu_output)

    def test_fmod_shape_format_fp32(self, device):
        format_list = [0, 3]
        shape_list = [[5, 6], [3, 4, 5]]
        shape_format_tensor = [[[np.float32, i, j], [np.float32, i, j]]
                               for i in format_list for j in shape_list]
        shape_format_scalar_tensor = [
            [[np.float32, i, j], 5] for i in format_list for j in shape_list
        ]
        self.case_exec_tensor(shape_format_tensor)
        self.case_exec_scalar(shape_format_scalar_tensor)

    def test_fmod_shape_format_fp16(self, device):
        format_list = [0, 3]
        shape_list = [[5, 6], [3, 4, 5]]
        shape_format_tensor = [[[np.float16, i, j], [np.float16, i, j]]
                               for i in format_list for j in shape_list]
        shape_format_scalar_tensor = [
            [[np.float16, i, j], 5] for i in format_list for j in shape_list
        ]
        self.case_exec_tensor_fp16(shape_format_tensor)
        self.case_exec_scalar_fp32(shape_format_scalar_tensor)

    def test_fmod_mix_dtype(self, device):
        npu_input1, npu_input2 = create_common_tensor([np.float32, 0, (2, 3)], 1, 100)
        npu_input3, npu_input4 = create_common_tensor([np.float16, 0, (2, 3)], 1, 100)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input3)
        npu_output = self.npu_op_exec(npu_input2, npu_input4)
        self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestFmod, globals(), except_for="cpu")

if __name__ == "__main__":
    run_tests()
