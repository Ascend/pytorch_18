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


class TestClampMin(TestCase):
    def generate_data(self, data):
        input1 = np.random.uniform(data[0], data[1], data[2]).astype(data[3])

        #modify from numpy.ndarray to torch.tensor
        input1 = torch.from_numpy(input1)
        
        return input1

    def npu_op_exec(self, input1, min_val):
        input1 = input1.to("npu")
        output = torch.clamp_min(input1, min_val)
        output = output.to("cpu")
        output = output.numpy()

        return output

    def cpu_op_exec(self, input1, min_val):
        output = torch.clamp_min(input1, min_val)
        output = output.numpy()

        return output

    def cpu_op_exec_float16(self, input1, min_val):
        input1 = input1.to(torch.float32)
        output = torch.clamp_min(input1, min_val).to(torch.float16)
        output = output.numpy()

        return output

    def npu_inp_op_exec(self, input1, min_val):
        input1 = input1.to("npu")
        output = torch.clamp_min_(input1, min_val)
        output = input1.to("cpu")
        output = output.numpy()

        return output

    def cpu_inp_op_exec(self, input1, min_val):
        output = torch.clamp_min_(input1, min_val)
        output = output.numpy()

        return output

    def cpu_inp_op_exec_float16(self, input1, min_val):
        input1 = input1.to(torch.float32)
        output = torch.clamp_min_(input1, min_val).to(torch.float16)
        output = output.numpy()

        return output

    def npu_op_exec_out(self, input1, min_val, input2):
        input1 = input1.to("npu")
        output = input2.to("npu")
        torch.clamp_min(input1, min_val, out=output)
        output = output.to("cpu")
        output = output.numpy()

        return output

    def npu_inp_uncon_op_exec(self, input1, min_val):
        input1 = input1.to("npu")
        input1 = input1.as_strided([2, 2], [1, 2], 2)
        output = torch.clamp_min_(input1, min_val)
        output = input1.to("cpu")
        output = output.numpy()

        return output

    def cpu_inp_uncon_op_exec(self, input1, min_val):
        input1 = input1.as_strided([2, 2], [1, 2], 2)
        output = torch.clamp_min(input1, min_val)
        output = output.numpy()

        return output

    def cpu_inp_uncon_op_exec_float16(self, input1, min_val):
        input1 = input1.to(torch.float32).as_strided([2, 2], [1, 2], 2)
        output = torch.clamp_min(input1, min_val).to(torch.float16)
        output = output.numpy()

        return output

    def test_clamp_min_common(self, device):
        shape_format = [
                [1, 100, (4, 3), np.float32],
                [1, 100, (4, 3), np.int32],
        ]
        for item in shape_format:
            input1 = self.generate_data(item)

            cpu_output = self.cpu_op_exec(input1, 50)
            npu_output = self.npu_op_exec(input1, 50)

            cpu_inp_output = self.cpu_inp_op_exec(input1, 50)
            npu_inp_output = self.npu_inp_op_exec(input1, 50)

            input2 = self.generate_data(item)
            npu_out_output = self.npu_op_exec_out(input1, 50, input2)

            cpu_inp_uncon_output = self.cpu_inp_uncon_op_exec(input1, 50)
            npu_inp_uncon_output = self.npu_inp_uncon_op_exec(input1, 50)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_inp_output, npu_inp_output)
            self.assertRtolEqual(cpu_output, npu_out_output)
            self.assertRtolEqual(cpu_inp_uncon_output, npu_inp_uncon_output)

    def test_clamp_min_common_out(self, device):
        shape_format = [
                [[1, 100, (4, 3), np.float32], [1, 100, (10, 10), np.float32]],
                [[1, 100, (4, 3), np.float32], [1, 100, (1, 1), np.float32]],
                [[1, 100, (4, 3), np.int32], [1, 100, (10, 10), np.int32]],
                [[1, 100, (4, 3), np.int32], [1, 100, (1, 1), np.int32]]
        ]
        for item in shape_format:
            print(item)
            input1 = self.generate_data(item[0])
            cpu_output = self.cpu_op_exec(input1, 50)
            input2 = self.generate_data(item[0])
            npu_out_output = self.npu_op_exec_out(input1, 50, input2)
            input3 = self.generate_data(item[1])
            npu_out_output1 = self.npu_op_exec_out(input1, 50, input3)

            self.assertRtolEqual(cpu_output, npu_out_output)
            self.assertRtolEqual(cpu_output, npu_out_output1)

    def test_clamp_min_float16(self, device):
        shape_format = [
                [1, 100, (4, 3), np.float16],
        ]
        for item in shape_format:
            input1 = self.generate_data(item)

            cpu_output = self.cpu_op_exec_float16(input1, 50)
            npu_output = self.npu_op_exec(input1, 50)

            cpu_inp_output = self.cpu_inp_op_exec_float16(input1, 50)
            npu_inp_output = self.npu_inp_op_exec(input1, 50)

            input2 = self.generate_data(item)
            npu_out_output = self.npu_op_exec_out(input1, 50, input2)

            cpu_inp_uncon_output = self.cpu_inp_uncon_op_exec_float16(input1, 50)
            npu_inp_uncon_output = self.npu_inp_uncon_op_exec(input1, 50)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_inp_output, npu_inp_output)
            self.assertRtolEqual(cpu_output, npu_out_output)
            self.assertRtolEqual(cpu_inp_uncon_output, npu_inp_uncon_output)

    def test_clamp_min_float16_out(self, device):
        shape_format = [
                [[1, 100, (4, 3), np.float16], [1, 100, (10, 10), np.float16]],
                [[1, 100, (4, 3), np.float16], [1, 100, (1, 1), np.float16]],
        ]
        for item in shape_format:
            print(item)
            input1 = self.generate_data(item[0])
            cpu_output = self.cpu_op_exec_float16(input1, 50)
            input2 = self.generate_data(item[0])
            npu_out_output = self.npu_op_exec_out(input1, 50, input2)
            input3 = self.generate_data(item[1])
            npu_out_output1 = self.npu_op_exec_out(input1, 50, input3)

            self.assertRtolEqual(cpu_output, npu_out_output)
            self.assertRtolEqual(cpu_output, npu_out_output1)

instantiate_device_type_tests(TestClampMin, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
