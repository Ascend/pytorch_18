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

from typing import Sequence, List
from functools import partial

import torch
from torch.testing._internal.common_methods_invocations import SampleInput

import torch_npu
from torch_npu.testing.common_methods_invocations import op_db
from torch_npu.testing.decorator import Dtypes, Formats, instantiate_ops_tests
from torch_npu.testing.testcase import TestCase, run_tests


def trans_device_and_dtype(sample, dtype, to_npu=False):
    def _trans_helper(x):
        if isinstance(x, torch.Tensor):
            x = x.to(dtype)
            if to_npu:
                x = x.to('npu')
        return x
    
    sample_helper = sample.transform(_trans_helper)
    sample = SampleInput(input=sample_helper[0], 
                         args=sample_helper[1], 
                         kwargs=sample_helper[2], 
                         broadcasts_input=sample.broadcasts_input,
    )
    return sample


@instantiate_ops_tests(op_db)
class TestOps(TestCase):

    @Formats(0)
    def test_correctness(self, dtype, op, npu_format=0):
        unsupported_dtypes_cpu = {dtype for dtype in op.dtypesIfNPU if dtype not in op.dtypes}

        def _generate_sample_inputs_requried_grad(sample_input, args):
            res = []

            if isinstance(sample_input, torch.Tensor):
                res.append(sample_input)
            elif isinstance(sample_input, Sequence) and isinstance(sample_input[0], torch.Tensor):
                res.extend(sample_input)
            
            if isinstance(args, torch.Tensor):
                res.append(args)
            elif args and isinstance(args, Sequence) and isinstance(args[0], torch.Tensor):                
                for arg in args:
                    if arg.grad_fn or arg.requires_grad:
                        res.append(arg)
            
            return res
               
        for sample in op.sample_inputs('cpu', dtype, requires_grad=op.supports_autograd):
            _trans_flag = False
            
            if dtype in unsupported_dtypes_cpu and dtype == torch.half:
                sample = trans_device_and_dtype(sample, torch.float32)
                _trans_flag = True

            expected = op(sample.input, *sample.args, **sample.kwargs)

            if _trans_flag and isinstance(expected, torch.Tensor):
                expected = expected.to(torch.half)

            sample_npu = trans_device_and_dtype(sample, dtype, to_npu=True)
            actual = op(sample_npu.input, *sample_npu.args, **sample_npu.kwargs)

            self.assertEqual(actual, expected)

            if not op.supports_autograd:
                continue

            expected = sample.output_process_fn_grad(expected)
            actual = sample_npu.output_process_fn_grad(actual)

            if isinstance(expected, torch.Tensor):
                backward_cpu_tensor = expected
                backward_npu_tensor = actual
            elif isinstance(expected, Sequence) and isinstance(expected[0], torch.Tensor):
                backward_cpu_tensor = expected[0]
                backward_npu_tensor = actual[0]

            sample_input_required_grad_cpu = _generate_sample_inputs_requried_grad(sample.input, sample.args)
            sample_input_required_grad_npu = _generate_sample_inputs_requried_grad(sample_npu.input, sample_npu.args)

            grads_cpu = torch.autograd.grad(outputs=backward_cpu_tensor, 
                                            inputs=sample_input_required_grad_cpu, 
                                            grad_outputs=torch.ones_like(backward_cpu_tensor),
            )
            grads_npu = torch.autograd.grad(outputs=backward_npu_tensor, 
                                            inputs=sample_input_required_grad_npu, 
                                            grad_outputs=torch.ones_like(backward_npu_tensor),
            )

            self.assertEqual(grads_cpu, grads_npu)


if __name__ == "__main__":
    run_tests()
