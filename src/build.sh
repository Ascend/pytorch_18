#!/bin/bash

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

CUR_DIR=$(dirname $(readlink -f $0))

function main()
{
    cd ${CUR_DIR}/third_party/acl/libs
    # stub
    dos2unix build_stub.sh
    chmod +x build_stub.sh
    ./build_stub.sh

    cd ${CUR_DIR}
    # if you add or delete file/files in the project, you need to remove the following comment
    # make clean
    export TORCH_PACKAGE_NAME=torch
    export PYTORCH_BUILD_VERSION='1.5.0+ascend'
    export PYTORCH_BUILD_NUMBER=2
    #for build GPU torch:DEBUG=0 USE_DISTRIBUTED=0 USE_HCCL=0 USE_NCCL=0 USE_MKLDNN=0 USE_CUDA=1 USE_NPU=0 BUILD_TEST=0 USE_NNPACK=0 python3.7 setup.py build bdist_wheel
    DEBUG=0 USE_DISTRIBUTED=1 USE_HCCL=1 USE_MKLDNN=0 USE_CUDA=0 USE_NPU=1 BUILD_TEST=0 USE_NNPACK=0 python3.7 setup.py build bdist_wheel
    if [ $? != 0 ]; then
        echo "Failed to compile the wheel file. Please check the source code by yourself."
        exit 1
    fi

    exit 0
}

main $1