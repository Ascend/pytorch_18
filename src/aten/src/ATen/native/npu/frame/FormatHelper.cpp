// Copyright (c) 2020 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "FormatHelper.h"
#include "c10/npu/npu_log.h"

namespace at {
namespace native {
namespace npu {

namespace {

constexpr int BLOCKSIZE = 16;

// base format is ND/NCHW
FormatShape InferShapeLessTo4(IntArrayRef dims);
FormatShape InferShape4To5(IntArrayRef dims);
FormatShape InferShape5To4(IntArrayRef dims);
FormatShape InferShapeNDToNZ(IntArrayRef dims);
FormatShape InferShapeNDToZ(IntArrayRef dims);
FormatShape InferShapeofNCHW(IntArrayRef dims);
FormatShape InferShapeofND(IntArrayRef dims);

// converter between base format
FormatShape InferShapeNCHWToND(IntArrayRef storage_dims, IntArrayRef base_dims);
FormatShape InferShapeNCDHWToND(IntArrayRef storage_dims, IntArrayRef base_dims);
FormatShape InferShapeNDToNCHW(IntArrayRef storage_dims, IntArrayRef base_dims);
FormatShape InferShapeNDToNCDHW(IntArrayRef storage_dims, IntArrayRef base_dims);


// base format is NCDHW
FormatShape InferShapeOfNDHWC(IntArrayRef dims);
FormatShape InferShapeOfNCDHW(IntArrayRef dims);
FormatShape InferShapeOfNDC1HWC0(IntArrayRef dims);
FormatShape InferShapeOfFZ3D(IntArrayRef dims);
}

std::unordered_map<aclFormat, FormatHelper::FormatInfo> FormatHelper::info = {
  {ACL_FORMAT_NC1HWC0,      (FormatInfo){ACL_FORMAT_NC1HWC0,    ACL_FORMAT_NCHW,    InferShape4To5,         "NC1HWC0",      true}},
  {ACL_FORMAT_ND,           (FormatInfo){ACL_FORMAT_ND,         ACL_FORMAT_ND,      InferShapeofND,         "ND",           false}},
  {ACL_FORMAT_NCHW,         (FormatInfo){ACL_FORMAT_NCHW,       ACL_FORMAT_NCHW,    InferShapeofNCHW,       "NCHW",         false}},
  {ACL_FORMAT_FRACTAL_NZ,   (FormatInfo){ACL_FORMAT_FRACTAL_NZ, ACL_FORMAT_ND,      InferShapeNDToNZ,       "FRACTAL_NZ",   true}},
  {ACL_FORMAT_FRACTAL_Z,    (FormatInfo){ACL_FORMAT_FRACTAL_Z,  ACL_FORMAT_NCHW,    InferShapeNDToZ,        "FRACTAL_Z",    true}},
  {ACL_FORMAT_NDHWC,        (FormatInfo){ACL_FORMAT_NDHWC,      ACL_FORMAT_NCDHW,   InferShapeOfNDHWC,      "NDHWC",        false}},
  {ACL_FORMAT_NCDHW,        (FormatInfo){ACL_FORMAT_NCDHW,      ACL_FORMAT_NCDHW,   InferShapeOfNCDHW,      "NCDHW",        false}},
  {ACL_FORMAT_NDC1HWC0,     (FormatInfo){ACL_FORMAT_NDC1HWC0,   ACL_FORMAT_NCDHW,   InferShapeOfNDC1HWC0,   "NDC1HWC0",     true}},
  {ACL_FRACTAL_Z_3D,        (FormatInfo){ACL_FRACTAL_Z_3D,      ACL_FORMAT_NCDHW,   InferShapeOfFZ3D,       "FRACTAL_Z_3D", true}},
};

std::unordered_map<aclFormat, std::unordered_map<aclFormat, baseFormatConverter>> FormatHelper::base_format_convert_info = {
  {ACL_FORMAT_ND,      {
                          {ACL_FORMAT_NCHW,    InferShapeNDToNCHW},
                          {ACL_FORMAT_NCDHW,    InferShapeNDToNCDHW},
                       }
  },
  {ACL_FORMAT_NCHW,    {
                          {ACL_FORMAT_ND,      InferShapeNCHWToND},
                       }
  },
  {ACL_FORMAT_NCDHW,    {
                          {ACL_FORMAT_ND,      InferShapeNCDHWToND},
                       }
  },
};

bool FormatHelper::IsPadded(const Tensor* tensor) {
  auto format = tensor->storage().unsafeGetStorageImpl()->npu_desc_.npu_format_;
  return IsPadded(format);
}

bool FormatHelper::IsPadded(aclFormat format) {
  auto itr = info.find(format);
  if (itr != info.end()) {
    return itr->second.isPadded;
  }
  AT_ERROR("unknown format type:", format);
  return true;
}

char* FormatHelper::GetFormatName(aclFormat format) {
  auto itr = info.find(format);
  if (itr == info.end()) {
    AT_ERROR("unknown format type:", format);
    return nullptr;
  }
  return itr->second.formatName;
}

char* FormatHelper::GetFormatName(const Tensor& tensor) {
  auto format = tensor.storage().get_npu_desc().npu_format_;
  return GetFormatName(format);
}

aclFormat FormatHelper::GetBaseFormat(const Tensor& tensor) {
  auto format = GetFormat(tensor);
  return GetBaseFormat(format);
}

aclFormat FormatHelper::GetBaseFormat(aclFormat format) {
  auto itr = info.find(format);
  if (itr == info.end()) {
    AT_ERROR("unknown format type:", format);
    return ACL_FORMAT_ND;
  }
  return itr->second.baseFormat;
}

aclFormat FormatHelper::GetFormat(const Tensor& tensor) {
  return tensor.storage().get_npu_desc().npu_format_;
}

bool FormatHelper::IsBaseFormatType(aclFormat format) {
  return GetBaseFormat(format) == format;
}

bool FormatHelper::IsBaseFormatType(const Tensor& tensor) {
  auto format = tensor.storage().get_npu_desc().npu_format_;
  return IsBaseFormatType(format);
}

FormatShape FormatHelper::GetStorageSizes(NPUStorageDesc desc) {
  auto ori_size = desc.base_sizes_;
  auto format = desc.npu_format_;
  return GetStorageSizes(format, ori_size);
}

FormatShape FormatHelper::GetSizeOfBaseFormat(const Tensor& src, aclFormat dst_format) {
  auto src_format = GetBaseFormat(src);
  auto itr = base_format_convert_info.find(src_format);
  if (itr != base_format_convert_info.end()) {
    auto next_itr = itr->second.find(dst_format);
    if (next_itr != itr->second.end()) {
      auto src_desc = src.storage().unsafeGetStorageImpl()->npu_desc_;
      return next_itr->second(src_desc.storage_sizes_, src_desc.base_sizes_);
    }
  }
  AT_ERROR("unsupport InferShape from ", GetFormatName(src_format), " to ", GetFormatName(dst_format));
  return {};
}

// 
namespace {
FormatShape InferShapeLessTo4(IntArrayRef dims) {
  FormatShape res;
  res.resize(4);
  AT_ASSERT(dims.size() <= 4, "input dim > 4 when InferShapeLessTo4");
  switch (dims.size()) {
    case 0:
      res[0] = 1;
      res[1] = 1;
      res[2] = 1;
      res[3] = 1;
      break;
    case 1:
      // RESHAPE_TYPE_C;
      res[0] = 1;
      res[1] = dims[0];
      res[2] = 1;
      res[3] = 1;
      break;
    case 2:
      // RESHAPE_TYPE_CH;
      res[0] = 1;
      res[1] = dims[0];
      res[2] = dims[1];
      res[3] = 1;
      break;
    case 3:
      // RESHAPE_TYPE_CHW;
      res[0] = 1;
      res[1] = dims[0];
      res[2] = dims[1];
      res[3] = dims[2];
      break;
    case 4:
      res[0] = dims[0];
      res[1] = dims[1];
      res[2] = dims[2];
      res[3] = dims[3];
      break;
    default:
      AT_ERROR("dims of NCHW shape should not be greater than 4, which is ",
               dims.size());
  }
  return res;
}

FormatShape InferShape4To5(IntArrayRef dims) {
  FormatShape res;
  res.resize(5);
  if (dims.size() < 4) {
    NPU_LOGD("infershape4to5 but input dim < 4");
    return InferShape4To5(InferShapeLessTo4(dims));
  } else if (dims.size() > 4) {
    NPU_LOGE("infershape4to5 but input dim > 4");
  }
  res[0] = dims[0];
  res[1] = (dims[1] + 15) / 16;
  res[2] = dims[2];
  res[3] = dims[3];
  res[4] = BLOCKSIZE;
  return res;
}


FormatShape InferShape5To4(IntArrayRef dims) {
  FormatShape res;
  res.emplace_back(dims[0]);
  res.emplace_back(((dims[1] + 15) / 16) * 16);
  res.emplace_back(dims[2]);
  res.emplace_back(dims[3]);
  return res;
}

FormatShape InferShapeNDToNZ(IntArrayRef dims) {
  FormatShape res;
  // sum(keepdim = false) may make tensor dim = 0
  FormatShape dim;
  for (int i = 0; i < dims.size(); i++) {
    dim.emplace_back(dims[i]);
  }

  // TODO(ascend): this expand code can be remove now
  // this action will move to GuessStorageSizeWhenConvertFormat
  if (dim.size() == 0) {
    dim.emplace_back(1);
  }
  if (dim.size() == 1) {
    dim.emplace_back(1);
  }

  int i = 0;
  for (; i < dim.size() - 2; i++) {
    res.emplace_back(dim[i]);
  }

  res.emplace_back((dim[i + 1] + 15) / BLOCKSIZE);
  res.emplace_back((dim[i] + 15) / BLOCKSIZE);
  res.emplace_back(BLOCKSIZE);
  res.emplace_back(BLOCKSIZE);

  return res;
}

FormatShape InferShapeNDToZ(
    IntArrayRef dims) {
  FormatShape res;
  if (dims.size() < 4) {
    return InferShapeNDToZ(InferShapeLessTo4(dims));
  }

  res.emplace_back((dims[1] + 15) / BLOCKSIZE * dims[2] * dims[3]);
  res.emplace_back((dims[0] + 15) / BLOCKSIZE);
  res.emplace_back(BLOCKSIZE);
  res.emplace_back(BLOCKSIZE);

  return res;
}

FormatShape InferShapeNCHWToND(IntArrayRef storage_dims, IntArrayRef base_dims) {
  FormatShape res;
  res.resize(4);
  auto cur_storage_dims = storage_dims;
  if (storage_dims.size() != 4) {
    cur_storage_dims = InferShapeLessTo4(storage_dims);
  }
  AT_ASSERT(cur_storage_dims.size() == 4, "input dim num not equal 4 when InferShapeNCHWToND");


  if (base_dims.size() == 0) {
    FormatShape temp_dims;
    temp_dims.emplace_back(1);
    return InferShapeLessTo4(temp_dims);
  }
  switch (base_dims.size()) {
    case 1:
      // reshape_type = RESHAPE_TYPE_C;
      res.resize(1);
      res[0] = cur_storage_dims[1];
      AT_ASSERT(cur_storage_dims[0] == 1, "reshape type RESHAPE_TYPE_C erase dim N must be 1");
      AT_ASSERT(cur_storage_dims[2] == 1, "reshape type RESHAPE_TYPE_C erase dim H must be 1");
      AT_ASSERT(cur_storage_dims[3] == 1, "reshape type RESHAPE_TYPE_C erase dim W must be 1");
      break;
    case 2:
      // reshape_type = RESHAPE_TYPE_CH;
      res.resize(2);
      res[0] = cur_storage_dims[1];
      res[1] = cur_storage_dims[2];
      AT_ASSERT(cur_storage_dims[0] == 1, "reshape type RESHAPE_TYPE_CH erase dim N must be 1");
      AT_ASSERT(cur_storage_dims[3] == 1, "reshape type RESHAPE_TYPE_CH erase dim W must be 1");
      break;
    case 3:
      // reshape_type = RESHAPE_TYPE_CHW;
      res.resize(3);
      res[0] = cur_storage_dims[1];
      res[1] = cur_storage_dims[2];
      res[2] = cur_storage_dims[3];
      AT_ASSERT(cur_storage_dims[0] == 1, "reshape type RESHAPE_TYPE_CHW erase dim N must be 1");
      break;
    case 4:
      res = cur_storage_dims;
      return res;
    default:
      AT_ERROR("unknown reshape type:");
  }
  return res;
}

FormatShape InferShapeNDToNCHW(IntArrayRef storage_dims, IntArrayRef base_dims) {
  AT_ASSERT(storage_dims.size() <= 4, "input storage dim not less than 4");
  AT_ASSERT(base_dims.size() <= 4, "input storage dim not less than 4");
  return InferShapeLessTo4(base_dims);
}

FormatShape InferShapeNDToNCDHW(IntArrayRef storage_dims, IntArrayRef base_dims) {
  AT_ASSERT(storage_dims.size() == 5, "ND [", storage_dims, "] failed to convert to NCDHW");
  FormatShape res;
  res.resize(5);
  res = storage_dims;
  return res;
}

FormatShape InferShapeNCDHWToND(IntArrayRef storage_dims, IntArrayRef base_dims) {
  FormatShape res;
  res.resize(5);
  res = storage_dims;
  AT_ASSERT(res.size() == 5, "input dim num not equal 5 when InferShapeNCDHWToND");
  return res;
}

// NCDHW -> NDHWC
FormatShape InferShapeOfNDHWC(IntArrayRef dims) {
  if (dims.size() < 5) {
    AT_ERROR("dim (", dims, ") cannot convert to NDHWC");
  }
  FormatShape res;
  res.resize(5);
  res[0] = dims[0];
  res[1] = dims[2];
  res[2] = dims[3];
  res[3] = dims[4];
  res[4] = dims[1];
  return res;
}

// NCDHW to NCDHW
FormatShape InferShapeOfNCDHW(IntArrayRef dims) {
  if (dims.size() < 5) {
    AT_ERROR("dim (", dims, ") cannot convert to NCDHW");
  }
  FormatShape res;
  res.resize(5);
  res[0] = dims[0];
  res[1] = dims[1];
  res[2] = dims[2];
  res[3] = dims[3];
  res[4] = dims[4];
  return res;
}

// NCDHW to NDC1HWC0
FormatShape InferShapeOfNDC1HWC0(IntArrayRef dims){
  if (dims.size() < 5) {
    AT_ERROR("dim (", dims, ") cannot convert to NDC1HWC0");
  }
  FormatShape res;
  res.resize(6);
  res[0] = dims[0];
  res[1] = dims[2];
  res[2] = (dims[1] + BLOCKSIZE - 1) / BLOCKSIZE;
  res[3] = dims[3];
  res[4] = dims[4];
  res[5] = BLOCKSIZE;
  return res;
}

// NCDHW to FZ_3D
FormatShape InferShapeOfFZ3D(IntArrayRef dims) {
  if (dims.size() < 5) {
    AT_ERROR("dim (", dims, ") cannot convert to FZ_3D");
  }

  int64_t d1 = dims[2];
  int64_t d2 = (dims[1] + BLOCKSIZE - 1) / BLOCKSIZE;
  int64_t d3 = dims[3];
  int64_t d4 = dims[4];
  int64_t d5 = (dims[0] + BLOCKSIZE - 1) / BLOCKSIZE;
  int64_t d6 = BLOCKSIZE;
  int64_t d7 = BLOCKSIZE;

  // The shape of FZ3D is 7D, but the CANN only accept 4D
  // so we should merge 1st, 2nd, 3rd, 4th dimension.
  FormatShape res;
  res.resize(4);
  res[0] = d1 * d2 * d3 * d4;
  res[1] = d5;
  res[2] = d6;
  res[3] = d7;
  return res;
}

FormatShape InferShapeofNCHW(IntArrayRef dims) {
  return InferShapeLessTo4(dims);
}

FormatShape InferShapeofND(IntArrayRef dims) {
  FormatShape res;
  res.resize(dims.size());
  for (int j = 0; j < dims.size(); j++) {
    res[j] = dims[j];
  }
  return res;
}
} // namespace


} // namespace npu
} // namespace native
} // namespace at
