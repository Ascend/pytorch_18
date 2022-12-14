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
import sys
import copy
import torch.nn as nn
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestNmsWithMask(TestCase):
    def npu_op_exec(self, input1, iou_threshold):
        npu_output1, npu_output2, npu_output3, = torch.npu_nms_with_mask(input1, iou_threshold)
        npu_output1 = npu_output1.to("cpu")
        npu_output2 = npu_output2.to("cpu")
        npu_output3 = npu_output3.to("cpu")

        return npu_output1, npu_output2, npu_output3

    def test_nms_with_mask_float32(self, device):
        input1 = torch.tensor([[0.0, 1.0, 2.0, 3.0, 0.6], [6.0, 7.0, 8.0, 9.0, 0.4]]).npu()
        iou_threshold = 0.5

        eq_output1 = torch.tensor([[0.0000, 1.0000, 2.0000, 3.0000, 0.6001],
                                   [6.0000, 7.0000, 8.0000, 9.0000, 0.3999]])
        eq_output2 = torch.tensor([0, 1], dtype=torch.int32)
        eq_output3 = torch.tensor([1, 1], dtype=torch.uint8)

        npu_output1, npu_output2, npu_output3 = self.npu_op_exec(input1, iou_threshold)

        self.assertRtolEqual(eq_output1, npu_output1)
        self.assertRtolEqual(eq_output2, npu_output2)
        self.assertRtolEqual(eq_output3, npu_output3)


instantiate_device_type_tests(TestNmsWithMask, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests() 