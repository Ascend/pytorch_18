// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

bool is_transpose_last_two_dims_v2(const Tensor& tensor) {
  if (tensor.dim() < 2) {
    return false;
  }
  auto storage_size = tensor.storage().get_npu_desc().storage_sizes_;
  int64_t numel = at::prod_intlist(storage_size);

  int64_t dim1 = tensor.dim() - 1;
  int64_t dim2 = tensor.dim() - 2;

  auto tensor_desc = tensor.storage().get_npu_desc();
  if (tensor_desc.base_sizes_.size() == tensor.dim() &&
      tensor.stride(dim2) == 1 && tensor.stride(dim1) == tensor.size(dim2) &&
      tensor.size(dim1) == tensor_desc.base_sizes_[dim2] &&
      tensor.size(dim2) == tensor_desc.base_sizes_[dim1] &&
      tensor.storage().size() == numel) {
    return true;
  } else {
    return false;
  }
}

SmallVector<int64_t, SIZE> bmm_v2_output_size(const Tensor& mat1, const Tensor& mat2) {
  auto dim_tensor1 = mat1.dim();
  auto dim_tensor2 = mat2.dim();

  int64_t m = dim_tensor1 == 1 ? 1 : mat1.size(-2);
  int64_t n = dim_tensor2 == 1 ? 1 : mat2.size(-1);

  auto batch_a = array_to_small_vector(IntArrayRef(mat1.sizes().data(), std::max<int64_t>(dim_tensor1 - 2, 0)));
  auto batch_b = array_to_small_vector(IntArrayRef(mat2.sizes().data(), std::max<int64_t>(dim_tensor2 - 2, 0)));

  batch_a.insert(batch_a.begin(), std::max<int64_t>(batch_a.size(), batch_b.size()) - batch_a.size(), 1);
  batch_b.insert(batch_b.begin(), std::max<int64_t>(batch_a.size(), batch_b.size()) - batch_b.size(), 1);

  SmallVector<int64_t, SIZE> output_size;
  for (size_t i = 0; i < batch_a.size(); ++i) {
    if (batch_a[i] == 1) {
      output_size.emplace_back(batch_b[i]);
    } else if (batch_b[i] == 1) {
      output_size.emplace_back(batch_a[i]);
    } else if (batch_a[i] != batch_b[i]) {
      AT_ERROR("mat1 and mat2 cannot broadcast, but they are mat1 ",
          mat1.sizes().data(), " mat2 ", mat2.sizes().data());
    } else {
      output_size.emplace_back(batch_a[i]);
    }
  }
  output_size.emplace_back(m);
  output_size.emplace_back(n);

  return output_size;
}

Tensor pure_bmm_v2_npu(const Tensor& self, const Tensor& mat2, const SmallVector<int64_t, SIZE>& output_size) {
  auto tensor1 = self.dim() == 1 ? self.view({1, self.size(0)}) : self;
  auto tensor2 = mat2.dim() == 1 ? mat2.view({mat2.size(0), 1}) : mat2;

  Tensor result;

  if ((tensor1.scalar_type() == ScalarType::Half)) {
    result = at::empty_with_format(output_size, tensor1.options(), ACL_FORMAT_FRACTAL_NZ);
  } else {
    result = at::empty_with_format(output_size, tensor1.options(), ACL_FORMAT_ND);
  }

  Tensor contiguous_self = tensor1;
  Tensor contiguous_mat2 = tensor2;
  bool is_self_t = is_transpose_last_two_dims_v2(tensor1);
  bool is_mat2_t = is_transpose_last_two_dims_v2(tensor2);

  if(!is_self_t) {
    contiguous_self = NpuUtils::format_contiguous(tensor1);
  }
  if(!is_mat2_t) {
    contiguous_mat2 = NpuUtils::format_contiguous(tensor2);
  }

  auto func1 = [&contiguous_self]() {
      bool pass = false;
      return std::tie(pass, contiguous_self);
  };
  auto func2 = [&contiguous_mat2]() {
      bool pass = false;
      return std::tie(pass, contiguous_mat2);
  };

  // executing the NPU operator
  OpCommand cmd;
  cmd.Name("BatchMatMul")
      .InputWithFunc(func1)
      .InputWithFunc(func2)
      .Output(result)
      .Attr("adj_x1", is_self_t)
      .Attr("adj_x2", is_mat2_t)
      .Run();

  return result;
}

Tensor reshape_tensor_self(const Tensor& self, SmallVector<int64_t, SIZE>& expect_output_size) {
  // self, expect_output: [5,6,7,17], [1,6,7,65]
  // self permute + reshape: [5,6,7,17] -> [6,7,5,17] -> [6,7,85]
  SmallVector<int64_t, SIZE> self_permute_idx;
  SmallVector<int64_t, SIZE> self_batch_idx;

  for (int64_t i = 0; i < self.dim(); ++i) {
    if (i < self.dim() - 2) {
      if (expect_output_size[i] == 1) {
        self_batch_idx.emplace_back(i);
        continue;
      }
    } else if (i == self.dim() - 1) {
      for (int64_t j = 0; j < self_batch_idx.size(); ++j) {
        self_permute_idx.emplace_back(self_batch_idx[j]);
      }
    }
    self_permute_idx.emplace_back(i);
  }
  Tensor tmp_self = self.permute(self_permute_idx);

  int64_t m_idx = 0;
  SmallVector<int64_t, SIZE> tmp_self_size;
  SmallVector<int64_t, SIZE> tmp_self_size_low;

  m_idx = self.dim() - self_batch_idx.size() - 1;
  tmp_self_size = array_to_small_vector(tmp_self.sizes());
  tmp_self_size_low.insert(tmp_self_size_low.end(), tmp_self_size.begin(), tmp_self_size.begin() + m_idx);
  tmp_self_size_low.emplace_back(-1);
  tmp_self = tmp_self.reshape(tmp_self_size_low);
  return tmp_self;
}

Tensor reshape_tensor_mat2(const Tensor& mat2, SmallVector<int64_t, SIZE>& expect_output_size) {
  // mat2, expect_output_size: [5,6,17,65], [1,6,7,65]
  // mat2 permute + reshape: [5,6,17,65] -> [6,5,17,65] -> [6,85,65]
  SmallVector<int64_t, SIZE> mat2_permute_idx;
  SmallVector<int64_t, SIZE> mat2_batch_idx;

  for (int64_t i = 0; i < mat2.dim(); ++i) {
    if (i < mat2.dim() - 2) {
      if (expect_output_size[i] == 1) {
        mat2_batch_idx.emplace_back(i);
        continue;
      }
    } else if (i == mat2.dim() - 2) {
      for (int64_t j = 0; j < mat2_batch_idx.size(); ++j) {
        mat2_permute_idx.emplace_back(mat2_batch_idx[j]);
      }
    }
    mat2_permute_idx.emplace_back(i);
  }
  Tensor tmp_mat2 = mat2.permute(mat2_permute_idx);

  int64_t k_idx = 0;
  SmallVector<int64_t, SIZE> tmp_mat2_size;
  SmallVector<int64_t, SIZE> tmp_mat2_size_low;

  k_idx = mat2.dim() - mat2_batch_idx.size() - 2;
  tmp_mat2_size = array_to_small_vector(tmp_mat2.sizes());
  tmp_mat2_size_low.insert(tmp_mat2_size_low.end(), tmp_mat2_size.begin(), tmp_mat2_size.begin() + k_idx);
  tmp_mat2_size_low.insert(tmp_mat2_size_low.end(), {-1, mat2.size(-1)});
  tmp_mat2 = tmp_mat2.reshape(tmp_mat2_size_low);
  return tmp_mat2;
}

SmallVector<int64_t, SIZE> align_small_vector(SmallVector<int64_t, SIZE> svec,
                                              SmallVector<int64_t, SIZE> golden_svec) {
  // svec, golden: [6,7,65], [5,6,7,65]
  // expect: [6,7,65] -> [1,6,7,65]
  SmallVector<int64_t, SIZE> tmp_svec;
  tmp_svec = svec;
  int64_t size_to_fill = golden_svec.size() - svec.size();
  if (size_to_fill > 0) {
    tmp_svec.insert(tmp_svec.begin(), size_to_fill, 1);
  }
  return tmp_svec;
}

void expand_tensor(Tensor& self, Tensor& mat2, SmallVector<int64_t, SIZE>& expand_output_size) {
  self = self.dim() == 1 ? self.view({1, self.size(0)}) : self;
  mat2 = mat2.dim() == 1 ? mat2.view({mat2.size(0), 1}) : mat2;
  int64_t m = self.size(-2);
  int64_t k1 = self.size(-1);
  int64_t k2 = mat2.size(-2);
  int64_t n = mat2.size(-1);

  std::vector<int64_t> expand_batch_portion(expand_output_size.begin(), expand_output_size.end() - 2);
  std::vector<int64_t> self_expand_size(expand_batch_portion);
  std::vector<int64_t> mat2_expand_size(expand_batch_portion);

  self_expand_size.insert(self_expand_size.end(), {m, k1});
  mat2_expand_size.insert(mat2_expand_size.end(), {k2, n});

  int64_t expand_batch_product = std::accumulate(expand_batch_portion.begin(), expand_batch_portion.end(),
                                                 1L, std::multiplies<int64_t>());

  std::vector<int64_t> self_bmm_view({expand_batch_product});
  std::vector<int64_t> mat2_bmm_view({expand_batch_product});
  self_bmm_view.insert(self_bmm_view.end(), {m, k1});
  mat2_bmm_view.insert(mat2_bmm_view.end(), {k2, n});

  self = self.expand(self_expand_size).reshape(self_bmm_view);
  mat2 = mat2.expand(mat2_expand_size).reshape(mat2_bmm_view);
}

Tensor bmm_v2_npu(const Tensor& self, const Tensor& mat2, IntArrayRef output_sizes) {
  auto expect_output_size = array_to_small_vector(output_sizes);
  auto infer_output_size = bmm_v2_output_size(self, mat2);
  Tensor tmp_self = self;
  Tensor tmp_mat2 = mat2;

  // forward propagation
  if (expect_output_size.empty()) {
    // avoid some accuracy error caused by transdata
    expand_tensor(tmp_self, tmp_mat2, infer_output_size);
    expect_output_size = infer_output_size;
    infer_output_size = bmm_v2_output_size(tmp_self, tmp_mat2);

    auto res = pure_bmm_v2_npu(tmp_self, tmp_mat2, infer_output_size).view(expect_output_size);
    infer_output_size = expect_output_size;

    if (self.dim() == 1) {
      // [k][b, k, n] -> [b, 1, n] -> [b, n]
      infer_output_size.erase(infer_output_size.end() - 2);
      return res.view(infer_output_size);
    } else if (mat2.dim() == 1) {
      // [b, m, k][k] -> [b, m, 1] -> [b, m]
      infer_output_size.erase(infer_output_size.end() - 1);
      return res.view(infer_output_size);
    }
    return res;
  }

  // backward propagation
  SmallVector<int64_t, SIZE> tmp_expect_output_size = expect_output_size;
  SmallVector<int64_t, SIZE> axis_reduce;
  SmallVector<int64_t, SIZE> tmp_self_size;
  SmallVector<int64_t, SIZE> tmp_mat2_size;

  tmp_expect_output_size = align_small_vector(expect_output_size, infer_output_size);
  for (int i = 0; i < tmp_expect_output_size.size(); ++i) {
    if (tmp_expect_output_size[i] != infer_output_size[i]) {
      axis_reduce.emplace_back(i);
    }
  }

  // no reduce_sum
  if (axis_reduce.empty()) {
    // avoid some accuracy error caused by transdata
    expand_tensor(tmp_self, tmp_mat2, infer_output_size);
    infer_output_size = bmm_v2_output_size(tmp_self, tmp_mat2);
    return pure_bmm_v2_npu(tmp_self, tmp_mat2, infer_output_size).view(expect_output_size);
  }

  // reduce sum without accuracy error
  tmp_self_size = align_small_vector(array_to_small_vector(self.sizes()), infer_output_size);
  tmp_mat2_size = align_small_vector(array_to_small_vector(mat2.sizes()), infer_output_size);
  tmp_self = self.reshape(tmp_self_size);
  tmp_mat2 = mat2.reshape(tmp_mat2_size);
  tmp_self = reshape_tensor_self(tmp_self, tmp_expect_output_size);
  tmp_mat2 = reshape_tensor_mat2(tmp_mat2, tmp_expect_output_size);
  infer_output_size = bmm_v2_output_size(tmp_self, tmp_mat2);
  // avoid some accuracy error caused by transdata
  expand_tensor(tmp_self, tmp_mat2, infer_output_size);
  infer_output_size = bmm_v2_output_size(tmp_self, tmp_mat2);
  return pure_bmm_v2_npu(tmp_self, tmp_mat2, infer_output_size).view(expect_output_size);
}

} // namespace native
} // namespace at
