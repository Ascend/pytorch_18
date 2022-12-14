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

#include "graph/ascend_string.h"
#include "graph/tensor.h"
#include "graph/graph.h"

namespace ge {
    AscendString::AscendString(const char* name) {}
    bool AscendString::operator<(const AscendString& d) const {return true;}
    const char* AscendString::GetString() const {return 0;}
    bool AscendString::operator==(const AscendString& d) const {return true;}

    TensorDesc::TensorDesc(const TensorDesc &desc) {}
    TensorDesc::TensorDesc(TensorDesc &&desc) {}
    TensorDesc::TensorDesc(Shape shape, Format format, DataType dt) {}
    Shape::Shape(const std::vector<int64_t> &dims) {}

    std::vector<GNode> Graph::GetDirectNode () const {return {};}

    GNode::GNode() {}
    graphStatus GNode::GetType(AscendString &type) const {return 0;}
    graphStatus GNode::SetAttr(const AscendString &name, int64_t &attr_value) const {return 0;}
    graphStatus GNode::SetAttr(const AscendString &name, int32_t &attr_value) const {return 0;}
    graphStatus GNode::SetAttr(const AscendString &name, float &attr_value) const {return 0;}
    graphStatus GNode::SetAttr(const AscendString &name, AscendString &attr_value) const {return 0;}
    graphStatus GNode::SetAttr(const AscendString &name, bool &attr_value) const {return 0;}
    graphStatus GNode::SetAttr(const AscendString &name, std::vector<int64_t> &attr_value) const {return 0;}
    graphStatus GNode::SetAttr(const AscendString &name, std::vector<float> &attr_value) const {return 0;}
    graphStatus GNode::SetAttr(const AscendString &name, std::vector<std::vector<int64_t>> &attr_value) const {return 0;}
}