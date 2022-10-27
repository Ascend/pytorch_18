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
    UnaryUfuncInfo(
        'acos',
        aliases=('arccos', ),
        dtypes=_dispatch_dtypes((torch.float16, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, )),
    ),
    UnaryUfuncInfo(
        'acosh',
        aliases=('arccosh', ),
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
        'addbmm',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_addbmm,
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
        'addmv',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_addmv,
    ),
    OpInfo(
        'addr',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_addr,
    ),
    OpInfo(
        'argsort',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_argsort,
        supports_autograd=False,
    ),
    OpInfo(
        'as_strided',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_as_strided,
    ),
    UnaryUfuncInfo(
        'asin',
        aliases=('arcsin', ),
        dtypes=_dispatch_dtypes((torch.float16, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, )),
    ),
    UnaryUfuncInfo(
        'asinh',
        aliases=('arcsinh', ),
        dtypes=_dispatch_dtypes((torch.float16, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, )),
    ),
    UnaryUfuncInfo(
        'atan',
        aliases=('arctan', ),
        dtypes=_dispatch_dtypes((torch.float16, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, )),
    ),
    BinaryUfuncInfo(
        'atan2',
        aliases=('arctan2', ),
        dtypes=_dispatch_dtypes((torch.float16, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_atan2,
    ),
    UnaryUfuncInfo(
        'atanh',
        aliases=('arctanh', ),
        dtypes=_dispatch_dtypes((torch.float16, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, )),
    ),
    OpInfo(
        'baddbmm',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_baddbmm,
    ),
    OpInfo(
        'bernoulli',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_bernoulli,
    ),
    OpInfo(
        'bincount',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_bincount,
    ),
    BinaryUfuncInfo(
        'bitwise_and',
        dtypes=_dispatch_dtypes((torch.bool, )),
        dtypesIfNPU=_dispatch_dtypes((torch.bool, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_binary_pwise,
        supports_autograd=False,
    ),
    UnaryUfuncInfo(
        'bitwise_not',
        dtypes=_dispatch_dtypes((torch.float16, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, )),
        supports_autograd=False,
    ),
    OpInfo(
        'bmm',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_bmm,
    ),
    OpInfo(
        'cdist',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_cdist,
    ),
    OpInfo(
        'clamp',
        aliases=('clip',),
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_clamp,
    ),
    UnaryUfuncInfo(
        'clamp',
        aliases=('clip',),
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_clamp_scalar,
    ),
    UnaryUfuncInfo(
        'ceil',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
    ),
    UnaryUfuncInfo(
        'cos',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
    ),
    UnaryUfuncInfo(
        'cosh',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
    ),
    OpInfo(
        'cross',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_cross,
    ),
    OpInfo(
        'nn.functional.ctc_loss',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_ctc_loss,
    ),
    OpInfo(
        'cross',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_cross,
    ),
    UnaryUfuncInfo(
        'isnan',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
        supports_autograd=False,
    ),
    OpInfo(
        'diag',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_diag,
    ),
    BinaryUfuncInfo(
        'div',
        aliases=('divide',),
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=partial(common_methods_invocations.sample_inputs_binary_pwise, python_scalars=True),
    ),
    OpInfo(
        'dot',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_dot_vdot,
    ),
    OpInfo(
        'nn.functional.dropout',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_dropout,
    ),
    OpInfo(
        'nn.functional.embedding',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_embedding,
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
        'erfc',
        aliases=('special.erfc', ),
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
    ),  
    UnaryUfuncInfo(
        'erfinv',
        aliases=('special.erfinv', ),
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
    ),  
    UnaryUfuncInfo(
        'exp',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
    ),  
    UnaryUfuncInfo(
        'exp2',
        aliases=('special.exp2', ),
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
    ),  
    UnaryUfuncInfo(
        'expm1',
        aliases=('special.expm1', ),
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
    ), 
    OpInfo(
        'flip',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_flip,
    ),
    UnaryUfuncInfo(
        'floor',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
    ), 
    BinaryUfuncInfo(
        'fmod',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_fmod_remainder,
    ),
    UnaryUfuncInfo(
        'frac',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
    ), 
    OpInfo(
        'gather',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_gather,
    ),
    BinaryUfuncInfo(
        'ge',
        aliases=('greater_equal',),
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_comparison_ops,
        supports_autograd=False,
    ),
    OpInfo(
        'nn.functional.gelu',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_gelu,
    ),
    OpInfo(
        'nn.functional.glu',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_glu,
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
        'nn.functional.hardshrink',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_softshrink_hardshrink_hardtanh,
    ),
    OpInfo(
        'nn.functional.hardswish',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_hardswish,
    ),
    UnaryUfuncInfo(
        'nn.functional.hardsigmoid',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
    ), 
    OpInfo(
        'nn.functional.hardtanh',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_softshrink_hardshrink_hardtanh,
    ),
    OpInfo(
        'index_add',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_index,
    ),
    OpInfo(
        'index_copy',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_index,
    ),
    OpInfo(
        'index_put',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_index_put,
    ),
    OpInfo(
        'inverse',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_linalg_invertible,
    ),
    BinaryUfuncInfo(
        'isclose',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_isclose,
    ),
    UnaryUfuncInfo(
        'isfinite',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
    ), 
    OpInfo(
        'kthvalue',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_kthvalue,
    ),
    OpInfo(
        'nn.functional.layer_norm',
        aliases=('layer_norm', ),
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_layer_norm,
    ),
    OpInfo(
        'nn.functional.leaky_relu',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_leaky_relu,
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
        'lerp',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_lerp,
    ),
    OpInfo(
        'linalg.svd',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_svd,
    ),
    OpInfo(
        'log_softmax',
        aliases=('special.log_softmax', 'nn.functional.log_softmax',),
        dtypes=_dispatch_dtypes((torch.float32,)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32,)),
        sample_inputs_func=common_methods_invocations.sample_inputs_softmax_variant,
    ),
    OpInfo(
        'nn.functional.linear',
        dtypes=_dispatch_dtypes((torch.float32,)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32,)),
        sample_inputs_func=common_methods_invocations.sample_inputs_linear,
    ),
    UnaryUfuncInfo(
        'log10',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
    ), 
    UnaryUfuncInfo(
        'log1p',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
    ), 
    UnaryUfuncInfo(
        'log2',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
    ), 
    UnaryUfuncInfo(
        'log',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
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
