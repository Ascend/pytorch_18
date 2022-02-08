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

import os
import torch
import numpy as np

from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from util_test import create_common_tensor, check_operators_in_prof

os.environ["COMBINED_ENABLE"] = "1"  # Open combined-view cases optimization
os.environ["PTCOPY_ENABLE"] = "1"

# Note: NPU only support trans-contiguous with base format, so format_list uses -1
class CombinedSqueezeXCopyToContiguous(TestCase):
    def test_squeeze_permute_copy_contiguous(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        shape_list = [
                      [2, 1, 3, 4],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: squeeze+permute ==> can be optimized as single permute(npuCombined should not be called)
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out1 = npu_input.squeeze(1).transpose(0,1).contiguous()
            self.assertEqual(check_operators_in_prof(['npuTranspose'], prof, ['npuCombined']), True, "Error operators called!")
            cpu_out1 = cpu_input.squeeze(1).transpose(0,1).contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())

            # case 2: permute+squeeze ==> can be optimized as single permute(npuCombined should not be called)
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out2 = npu_input.permute(1,0,3,2).squeeze(0).contiguous()
            self.assertEqual(check_operators_in_prof(['npuTranspose'], prof, ['npuCombined']), True, "Error operators called!")
            cpu_out2 = cpu_input.permute(1,0,3,2).squeeze(0).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy()) 
    
    def test_squeeze_narrow_copy_contiguous(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        shape_list = [
                      [20, 1, 30, 40, 16],
                      [20, 1, 30, 40]
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: squeeze + narrow 
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out1 = npu_input.squeeze(1)[:,1:10,:].contiguous()
            self.assertEqual(check_operators_in_prof(['npuMatch', 'narrow_npuSlice'], prof), True, "Error operators called!")
            cpu_out1 = cpu_input.squeeze(1)[:,1:10,:].contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: narrow + squeeze
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out2 = npu_input[:,:,:,10:19].squeeze(1).contiguous()
            self.assertEqual(check_operators_in_prof(['npuMatch', 'narrow_npuSlice'], prof), True, "Error operators called!")
            cpu_out2 = cpu_input[:,:,:,10:19].squeeze(1).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())

    def test_squeeze_select_copy_contiguous(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        shape_list = [
                      [20, 1, 40, 16],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: squeeze+select
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out1 = npu_input.squeeze().select(2,1).contiguous()
            self.assertEqual(check_operators_in_prof(['npuMatch', 'select_npuStridedSlice'], prof), True, "Error operators called!")
            cpu_out1 = cpu_input.squeeze().select(2,1).contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: select+squeeze
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out2 = npu_input.select(2,1).squeeze().contiguous()
            self.assertEqual(check_operators_in_prof(['npuMatch', 'select_npuStridedSlice'], prof), True, "Error operators called!")
            cpu_out2 = cpu_input.select(2,1).squeeze().contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())

    def test_squeeze_strideslice_copy_contiguous(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        shape_list = [
                      [20, 1, 200, 40, 10],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: squeeze + strideslice ==> cannot be optimized(npuCombined should not called)
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out1 = npu_input.squeeze(1)[:,20:150:3].contiguous()
            self.assertEqual(check_operators_in_prof(['d2dCopyWithPTCopy'], prof, ['npuCombined']), True, "Error operators called!")
            cpu_out1 = cpu_input.squeeze(1)[:,20:150:3].contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: strideslice + squeeze ==> cannot be optimized(npuCombined should not called)
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out2 = npu_input[:,:,10:19:3].squeeze(1).contiguous()
            self.assertEqual(check_operators_in_prof(['d2dCopyWithPTCopy'], prof, ['npuCombined']), True, "Error operators called!")
            cpu_out2 = cpu_input[:,:,10:19:3].squeeze(1).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy()) 

         
instantiate_device_type_tests(CombinedSqueezeXCopyToContiguous, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()