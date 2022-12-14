# Copyright (c) 2021 Huawei Technologies Co., Ltd
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
from os import path, makedirs
import datetime
from enum import Enum, unique

@unique
class DumpMode(Enum):
    OFF = 0
    DUMP = 1
    LOAD = 2

def get_time_stamp():
    time_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S.%f')
    return time_stamp


class dumper(object):
    """Context manager that manages dumper mode.

    Arguments:
        enabled (bool, optional): Setting this to False makes this context manager a no-op.
            Default: ``True``.

        use_dump (bool, optional): Dump all the ir in the context and assign director from dump_path.
            Default: ``False``

        use_load (bool, optional): Enables timing of NPU events as well using the npuEvent API.
            Default: ``False``

        dump_path (string, optional): The dirctory that used to store dump file.

        load_path (string, optional): The dirctory that used to load file.

    .. warning:
        This context managers should not be called recursively, i.e. at most one
        instance should be enabled at any given time.

    Example 1:
        dump ir file to current directory:
        >>> x = torch.randn((1, 1), requires_grad=True)
        >>> with torch.utils.dumper(use_dump=True) as dump:
        >>>     for _ in range(100):
        >>>         y = x ** 2
        >>>         y.backward()

    Example 2:
        load dumped file from load_file_path, then dump file to dump_path:
        >>> x = torch.randn((1, 1), requires_grad=True)
        >>> with torch.utils.dumper(use_load=True, dump_path="/home", load_file_path="/home/dumpfile.h5") as dump:
        >>>     for _ in range(100):
        >>>         y = x ** 2
        >>>         y.backward()
    """

    def __init__(self, enabled=True, use_dump=False, use_load=False, dump_path=None, load_file_path=None):
        self.enabled = enabled
        self.use_dump = use_dump
        self.use_load = use_load
        self.dump_path = dump_path
        self.load_file_path = load_file_path
        if not self.enabled:
            return
        self.entered = False
    def __enter__(self):
        if not self.enabled:
            return
        if self.entered:
            raise RuntimeError("utils dumper are not reentrant")
        self.entered = True
        if self.use_dump and self.use_load:
            raise RuntimeError("dump mode and load mode can not run together!")

        if not self.use_dump and not self.use_load:
            raise RuntimeError("you should choose dump mode or load mode!")

        if self.use_dump:
            self._set_dump_path(self.dump_path)
            torch._C._set_dumper_mode(DumpMode.DUMP.value)
        if self.use_load:
            if path.isfile(self.load_file_path):
                if path.isfile(self.dump_path) and \
                    path.abspath(self.load_file_path) == path.abspath(self.dump_path):
                    raise RuntimeError("dump_path and load_file_path can not be same!")
                torch._C._set_loader_path(self.load_file_path)
                self._set_dump_path(self.dump_path)
            else:
                raise RuntimeError("please input a file path")
            torch._C._set_dumper_mode(DumpMode.LOAD.value)
        return self

    @staticmethod
    def _set_dump_path(paths):
        if paths is not None:
            filename = path.basename(paths)
            if len(filename) == 0:
                filename = get_time_stamp() + ".h5"
            dirname = path.dirname(paths)
            if len(dirname) != 0 and not path.isdir(dirname):
                raise RuntimeError("the directory '{}' does not exist, please input a valid one".format(dirname))
            new_paths = path.join(dirname, filename)
            torch._C._set_dumper_path(new_paths)
        else:
            torch._C._set_dumper_path(get_time_stamp() + ".h5")


    def __exit__(self, exc_type, exc_value, traceback):
        torch._C._set_dumper_mode(DumpMode.OFF.value)

