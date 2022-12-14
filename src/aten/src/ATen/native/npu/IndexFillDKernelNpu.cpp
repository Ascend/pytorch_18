// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include <vector>

namespace at{
namespace native{
using namespace at::native::npu;

namespace {
  
void index_fill_d_check_index(IntArrayRef shape, const Tensor &index, int64_t dim)
{
  TORCH_CHECK(index.dim() == 1,"Index should be a one-dimensional tensor");
  int index_temp = INT_MAX;
  for (int i = 0; i < index.sizes()[0]; i++)
  {
    index_temp = static_cast<int>(CalcuOpUtil::get_scalar_float_value(index[i].item()));
    TORCH_CHECK(shape[dim] > index_temp,
                "Index out of range, it should be in [0,", shape[dim], ")");  
  }
}

SmallVector<float, N> index_fill_d_assist_help_init(
    int64_t dim,
    IntArrayRef sizes,
    vector<int> index,
    bool flag,
    float value)
{
  //
  int blocksize = 0;
  int blocknum = 1;
  int n = 1;

  for (int i = 0; i < sizes.size(); i++)
  {
    if (i <= dim)
    {
      blocknum *= sizes[i];
    }
    n *= sizes[i];
  }
  blocksize = n / blocknum;

  SmallVector<float, N> ast;
  ast.resize(n);

  if (flag) {
    ast = SmallVector<float, N>(n, 1);
  } else {
    ast = SmallVector<float, N>(n, 0);
  }
  for (int i = 0; i < index.size(); i++)
  {
    int start = 0, end = 0;
    int idx = index[i];
    int k = idx, count = 0;
    while (k < blocknum)
    {
      start = blocksize * k;
      end = start + blocksize;
      for (int j = start; j < end; j++)
      {
        ast[j] = value;
      }
      count++;
      k = idx + sizes[dim] * count;
    }
  }
  return ast;
}

Tensor index_fill_d_assist_help(
    const Tensor &self,
    const Tensor &index,
    int64_t dim,
    Scalar value,
    bool flag)
{
  SmallVector<float, N> assist;
  IntArrayRef size = self.sizes();
  vector<int> index_vector;
  for (int i = 0; i < index.sizes()[0]; i++)
  {
    int index_temp = static_cast<int>(CalcuOpUtil::get_scalar_float_value(index[i].item()));
    index_vector.push_back(index_temp);
  }
  // input
  // index is a 1-D tensor
  // value is a tensor which has only one item
  float value_float = CalcuOpUtil::get_scalar_float_value(value);
  assist = index_fill_d_assist_help_init(dim, size, index_vector, flag, value_float);
  Tensor assistHelp = from_blob(assist.data(), size, dtype(ScalarType::Float));
  return CalcuOpUtil::copy_tensor_host_to_device(assistHelp);
}

Tensor& index_fill_d_nocheck(
    Tensor& result,
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    Scalar value) {
  // Special case
  // There is a zero in shape
  // example : shape = [1,3,4,0] return itself else return
  // processed_data(result)
  if (self.numel() == 0) {
    return result;
  }
  Scalar value_zeros = Scalar(0.0);
  const Tensor* aclInput = &self;
  Tensor assistHelp1 =
      index_fill_d_assist_help(self, index, dim, value_zeros, true);
  Tensor assistHelp2 = index_fill_d_assist_help(self, index, dim, value, false);
  if (aclInput->scalar_type() == ScalarType::Int) // int32
  {
    assistHelp1 = assistHelp1.to(ScalarType::Int);
    assistHelp2 = assistHelp2.to(ScalarType::Int);
  } else if (aclInput->scalar_type() == ScalarType::Half) // fp16
  {
    assistHelp1 = assistHelp1.to(ScalarType::Half);
    assistHelp2 = assistHelp2.to(ScalarType::Half);
  }

  OpCommand cmd;
  cmd.Name("IndexFillD")
      .Input(self)
      .Input(assistHelp1)
      .Input(assistHelp2)
      .Attr("dim", dim)
      .Output(result)
      .Run();
  return result;
}
} // namespace

Tensor index_fill_npu(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    Scalar value) {
  Tensor result = OpPreparation::ApplyTensor(self);
  index_fill_d_nocheck(result, self, dim, index, value);

  return result;
}

Tensor& index_fill_npu_(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    Scalar value) {
  // In-Place-Scalar
  IntArrayRef shape_self = self.sizes();
  index_fill_d_check_index(shape_self, index, dim);
  index_fill_d_nocheck(self, self, dim, index, value);
  return self;
}

Tensor index_fill_npu(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& value) {
  // Out-Place-Tensor
  IntArrayRef shape_self = self.sizes();
  index_fill_d_check_index(shape_self, index, dim);
  TORCH_CHECK(value.dim() == 0, 
              "Value should be a 0-dimensional tensor,but got ",value.dim());
  Scalar value_scalar = value.item();
  Tensor result = OpPreparation::ApplyTensor(self);
  index_fill_d_nocheck(result, self, dim, index, value_scalar);
  return result;
}

Tensor& index_fill_npu_(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& value) {
  // In-Place-Tensor
  IntArrayRef shape_self = self.sizes();
  index_fill_d_check_index(shape_self, index, dim);
  TORCH_CHECK(value.dim() == 0, 
              "Value should be a 0-dimensional tensor,but got ",value.dim());
  Scalar value_scalar = value.item();
  index_fill_d_nocheck(self, self, dim, index, value_scalar);
  return self;
}
} // namespace native
} // namespace at