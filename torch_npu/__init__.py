# Copyright (c) 2020 Huawei Technologies Co., Ltd
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

import sys
import types
import atexit
import torch

import torch_npu
import torch_npu.npu
import torch_npu.npu.amp
import torch_npu.distributed
import torch_npu._C

from torch_npu.utils import apply_module_patch

from .version import __version__ as __version__

__all__ = []


for name in dir(torch_npu._C._VariableFunctions):
    if name.startswith('__'):
        continue
    globals()[name] = getattr(torch_npu._C._VariableFunctions, name)
    __all__.append(name)
    setattr(torch, name, getattr(torch_npu._C._VariableFunctions, name))

all_monkey_patches = [
    ["npu", torch_npu.npu],
    ["npu.amp", torch_npu.npu.amp],
    ["autograd.profiler", torch_npu.npu.profiler],
    ["distributed", torch_npu.distributed],
    ["distributed.distributed_c10d", torch_npu.distributed.distributed_c10d],
    ["nn.parallel.distributed._get_default_group", torch_npu.distributed.distributed_c10d._get_default_group]
]


def _apply_patches(monkey_patches):
    
    def _getattr(module_list, root_module=torch):
        if len(module_list) <= 1:
            return root_module

        if hasattr(root_module, module_list[0]):
            return _getattr(module_list[1:], getattr(root_module, module_list[0]))
        else:
            empty_module_name = f'{root_module.__name__}.{module_list[0]}'
            sys.modules[empty_module_name] = types.ModuleType(empty_module_name)
            setattr(root_module, module_list[0], sys.modules.get(empty_module_name))
            return _getattr(module_list[1:], getattr(root_module, module_list[0]))

    for patch_pair in monkey_patches:
        dest, patch = patch_pair
        dest_module = _getattr(dest.split('.'), root_module=torch)
        last_module_level = dest.split(".")[-1]
        if not isinstance(patch, types.ModuleType):
            setattr(dest_module, last_module_level, patch)
            continue

        if not hasattr(dest_module, last_module_level) or not hasattr(patch, '__all__'):
            setattr(dest_module, last_module_level, patch)
            sys.modules[f'{dest_module.__name__}.{last_module_level}'] = patch
            continue

        assert hasattr(patch, '__all__'), "Patch module must have __all__ definition."
        dest_module = getattr(dest_module, last_module_level)
        for attr in patch.__all__:
            setattr(dest_module, attr, getattr(patch, attr))

# Apply monkey-patches.
_apply_patches(all_monkey_patches)
apply_module_patch()

# NPU exit, need to synchronize devices
def _npu_shutdown():
    torch_npu._C._npu_shutdown()

#register npu shutdown hook on exit
atexit.register(_npu_shutdown)