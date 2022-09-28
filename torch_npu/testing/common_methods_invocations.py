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

from typing import List
from functools import partial

import torch
from torch.testing._internal import common_methods_invocations
from torch.testing._internal.common_dtype import _dispatch_dtypes
from torch.testing._internal.common_methods_invocations import (
    OpInfo as Of_OpInfo,
    UnaryUfuncInfo as Of_UnaryUfuncInfo,
    BinaryUfuncInfo as Of_BinaryUfuncInfo,
)

import torch_npu


class OpInfo(Of_OpInfo):
        def __init__(
            self,
            name,  # the string name of the function
            dtypes=_dispatch_dtypes((torch.float32,)),
            formats=0,
            dtypesIfNPU=(torch.float16,),
            backward_dtypes=None,
            backward_dtypesIfNPU=None,
            **kwargs,
        ):
            super().__init__(
                name,
                dtypes=dtypes,
                backward_dtypes=backward_dtypes,
                **kwargs,
            )
            self.dtypesIfNPU = set(dtypesIfNPU) if dtypesIfNPU is not None else self.dtypes
            self.backward_dtypesIfNPU = set(backward_dtypesIfNPU) if backward_dtypesIfNPU is not None else (
                backward_dtypes if backward_dtypes is not None
                else dtypesIfNPU if dtypesIfNPU is not None
                else dtypes
            )
            self.formats = formats


class UnaryUfuncInfo(Of_UnaryUfuncInfo):
        def __init__(
            self,
            name,  # the string name of the function
            ref=None,
            sample_inputs_func=common_methods_invocations.sample_inputs_unary,
            dtypes=_dispatch_dtypes((torch.float32,)),
            formats=0,
            dtypesIfNPU=(torch.float16,),
            backward_dtypesIfNPU=None,
            backward_dtypes=None,
            **kwargs,
        ):
            super().__init__(
                name,
                ref=ref,
                sample_inputs_func=sample_inputs_func,
                dtypes=dtypes,
                backward_dtypes=backward_dtypes,
                **kwargs,
            )
            self.dtypesIfNPU = set(dtypesIfNPU) if dtypesIfNPU is not None else self.dtypes
            self.backward_dtypesIfNPU = set(backward_dtypesIfNPU) if backward_dtypesIfNPU is not None else (
                backward_dtypes if backward_dtypes is not None
                else dtypesIfNPU if dtypesIfNPU is not None
                else dtypes
            )
            self.formats = formats


class BinaryUfuncInfo(Of_BinaryUfuncInfo):
        def __init__(
            self,
            name,  # the string name of the function
            dtypes=_dispatch_dtypes((torch.float32,)),
            formats=0,
            dtypesIfNPU=(torch.float16,),
            backward_dtypesIfNPU=None,
            backward_dtypes=None,
            **kwargs,
        ):
            super().__init__(
                name,
                dtypes=dtypes,
                backward_dtypes=backward_dtypes,
                **kwargs,
            )
            self.dtypesIfNPU = set(dtypesIfNPU) if dtypesIfNPU is not None else self.dtypes
            self.backward_dtypesIfNPU = set(backward_dtypesIfNPU) if backward_dtypesIfNPU is not None else (
                backward_dtypes if backward_dtypes is not None
                else dtypesIfNPU if dtypesIfNPU is not None
                else dtypes
            )
            self.formats = formats

op_db: List[OpInfo] = [
    UnaryUfuncInfo(
        'abs',
        aliases=('absolute', ),
        dtypes=_dispatch_dtypes((torch.float16, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, )),
    ),
    BinaryUfuncInfo(
        'add',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=partial(common_methods_invocations.sample_inputs_add_sub, alpha=2),
    ),
    OpInfo(
        'addcdiv',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_addcmul_addcdiv,
    ),
    OpInfo(
        'addcmul',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_addcmul_addcdiv,
    ),
    OpInfo(
        'addmm',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_addmm,
    ),
    OpInfo(
        'as_strided',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_as_strided,
    ),
    BinaryUfuncInfo(
        'bitwise_and',
        dtypes=_dispatch_dtypes((torch.bool, )),
        dtypesIfNPU=_dispatch_dtypes((torch.bool, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_binary_pwise,
        supports_autograd=False,
    ),
    OpInfo(
        'cat',
        aliases=('concat',),
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_cat_concat,
    ),
    UnaryUfuncInfo(
        'ceil',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
    ),
    UnaryUfuncInfo(
        'isnan',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
        supports_autograd=False,
    ),
    BinaryUfuncInfo(
        'div',
        aliases=('divide',),
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=partial(common_methods_invocations.sample_inputs_binary_pwise, python_scalars=True),
    ),
    BinaryUfuncInfo(
        'eq',
        aliases=('equal',),
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_comparison_ops,
        supports_autograd=False,
    ),
    UnaryUfuncInfo(
        'erf',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
    ),   
    UnaryUfuncInfo(
        'exp',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
    ),  
    BinaryUfuncInfo(
        'ge',
        aliases=('greater_equal',),
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_comparison_ops,
        supports_autograd=False,
    ),
    BinaryUfuncInfo(
        'gt',
        aliases=('greater',),
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_comparison_ops,
        supports_autograd=False,
    ),
    OpInfo(
        'index_put',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_index_put,
    ),
    BinaryUfuncInfo(
        'le',
        aliases=('less_equal',),
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_comparison_ops,
        supports_autograd=False,
    ),
    OpInfo(
        'log_softmax',
        aliases=('special.log_softmax', 'nn.functional.log_softmax',),
        dtypes=_dispatch_dtypes((torch.float32,)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32,)),
        sample_inputs_func=common_methods_invocations.sample_inputs_softmax_variant,
    ),
    BinaryUfuncInfo(
        'lt',
        aliases=('less',),
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_comparison_ops,
        supports_autograd=False,
    ),
    OpInfo(
        'masked_select',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_masked_select,
    ),
    OpInfo(
        'matmul',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_matmul,
    ),
    BinaryUfuncInfo(
        'max',
        aliases=('maximum',),
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_max_min_binary,
    ),
    BinaryUfuncInfo(
        'min',
        aliases=('minimum',),
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_max_min_binary,
    ),
    OpInfo(
        'mm',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_mm,
    ),
    OpInfo(
        'nn.functional.mse_loss',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_mse_loss,
    ),
    BinaryUfuncInfo(
        'mul',
        aliases=('multiply',),
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=partial(common_methods_invocations.sample_inputs_binary_pwise, python_scalars=True),
    ),
    BinaryUfuncInfo(
        'ne',
        aliases=('not_equal',),
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_comparison_ops,
        supports_autograd=False,
    ),
    UnaryUfuncInfo(
        'neg',
        aliases=('negative',),
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
    ),  
    OpInfo(
        'nonzero',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_nonzero,
        supports_autograd=False,
    ),
    # normal has first and second test, imp first only 
    OpInfo(
        'normal',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_normal_tensor_second,
        supports_autograd=False,
    ),
    OpInfo(
        'ones_like',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_like_fns,
        supports_autograd=False,
    ),
    BinaryUfuncInfo(
        'pow',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_pow,
    ),
    UnaryUfuncInfo(
        'reciprocal',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
    ),  
    OpInfo(
        'nn.functional.relu',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_nn_activation_relu,
    ),
    OpInfo(
        'reshape',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_view_reshape,
    ),
    BinaryUfuncInfo(
        'rsub',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=partial(common_methods_invocations.sample_inputs_rsub, other_scalar=False),
    ),   
    OpInfo(
        'softmax',
        aliases=('special.softmax', 'nn.functional.softmax',),
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_softmax_variant,
    ),  
    UnaryUfuncInfo(
        'sqrt',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
    ), 
    BinaryUfuncInfo(
        'sub',
        aliases=('subtract',),
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=partial(common_methods_invocations.sample_inputs_add_sub, alpha=2, other_scalar=True),
    ),  
    OpInfo(
        'topk',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_topk,
    ),    
    OpInfo(
        'transpose',
        aliases=('swapdims', 'swapaxes'),
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_transpose_swapdims,
    ), 
    OpInfo(
        'zeros_like',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_like_fns,
        supports_autograd=False,
    ), 
]
