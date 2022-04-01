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

# Note: NPU only support trans-contiguous with base format, so format_list uses -1
class CombinedReshapeXCopyToContiguous(TestCase):
    def test_view_permute_copy_contiguous(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        shape_list = [
                      [200, 30, 40, 16],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: view+permute
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out1 = npu_input.view(npu_input.size(0) * npu_input.size(1), npu_input.size(2), npu_input.size(3)).transpose(0, 1).contiguous()
            self.assertEqual(check_operators_in_prof(['npuMatch', 'npuTranspose'], prof), True, "Error operators called!")
            cpu_out1 = cpu_input.view(cpu_input.size(0) * cpu_input.size(1), cpu_input.size(2), cpu_input.size(3)).transpose(0, 1).contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())

            # case 2: permute+view
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out2 = npu_input.permute(1, 0, 2, 3).view(npu_input.size(1), npu_input.size(0), npu_input.size(2)*npu_input.size(3)).contiguous()
            self.assertEqual(check_operators_in_prof(['npuMatch', 'npuTranspose'], prof), True, "Error operators called!")
            cpu_out2 = cpu_input.permute(1, 0, 2, 3).view(cpu_input.size(1), cpu_input.size(0), cpu_input.size(2)*cpu_input.size(3)).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy()) 
    
    def test_view_select_copy_contiguous(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        shape_list = [
                      [2, 3, 4, 5],
                     ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: view+select
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out1 = npu_input.view(npu_input.size(0), npu_input.size(1) * npu_input.size(2), npu_input.size(3)).select(2, 1).contiguous()
            self.assertEqual(check_operators_in_prof(['npuMatch', 'select_npuStridedSlice'], prof), True, "Error operators called!")
            cpu_out1 = cpu_input.view(npu_input.size(0), npu_input.size(1) * npu_input.size(2), npu_input.size(3)).select(2, 1).contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: select+view ==> can be optimized as reshape+narrow
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out2 = npu_input.select(2, 1).view(npu_input.size(1), npu_input.size(0), -1).contiguous()
            self.assertEqual(check_operators_in_prof(['npuMatch', 'narrow_npuSlice'], prof), True, "Error operators called!")
            cpu_out2 = cpu_input.select(2, 1).view(npu_input.size(1), npu_input.size(0), -1).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())
    
    def test_view_narrow_copy_contiguous(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        shape_list = [
                      [20, 30, 40, 16],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: view + narrow 
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out1 = npu_input.view(20, 1200, 16)[:,20:150,:].contiguous()
            self.assertEqual(check_operators_in_prof(['npuMatch', 'narrow_npuSlice'], prof), True, "Error operators called!")
            cpu_out1 = cpu_input.view(20, 1200, 16)[:,20:150,:].contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: narrow + view 
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out2 = npu_input[:,10:19,:,:].view(20, 360, 16).contiguous()
            self.assertEqual(check_operators_in_prof(['npuMatch', 'narrow_npuSlice'], prof), True, "Error operators called!")
            cpu_out2 = cpu_input[:,10:19,:,:].view(20, 360, 16).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())
    
    def test_view_strideslice_copy_contiguous(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        shape_list = [
                      [20, 30, 40, 10],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: view + strideslice 
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out1 = npu_input.view(20, 1200, 10)[:,20:150:3,:].contiguous()
            self.assertEqual(check_operators_in_prof(['npuMatch', 'narrow_npuSlice'], prof), True, "Error operators called!")
            cpu_out1 = cpu_input.view(20, 1200, 10)[:,20:150:3,:].contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: strideslice + view 
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out2 = npu_input[10:19:3,:,:].view(3, 2400, 5).contiguous()
            self.assertEqual(check_operators_in_prof(['npuAsStrided'], prof), True, "Error operators called!")
            cpu_out2 = cpu_input[10:19:3,:,:].view(3, 2400, 5).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())

         
instantiate_device_type_tests(CombinedReshapeXCopyToContiguous, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()