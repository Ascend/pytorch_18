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
from common_utils import TestCase, run_tests
from common_device_type import instantiate_device_type_tests

class TestRandomChoiceWithMask(TestCase):
    def test_random_choice_with_mask_fp32(self, device):
        input_bool = torch.tensor([1, 0, 1, 0], dtype = torch.bool).npu()
        expect_ret = torch.tensor([[0],[2]], dtype = torch.int32)
        expect_mask = torch.tensor([True, True])
        result, mask = torch.npu_random_choice_with_mask(input_bool, 2, 1, 0)
        self.assertRtolEqual(expect_ret, result.cpu())
        self.assertRtolEqual(expect_mask, mask.cpu())
        
instantiate_device_type_tests(TestRandomChoiceWithMask, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
