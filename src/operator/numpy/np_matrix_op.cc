/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2019 by Contributors
 * \file np_matrix_op.cc
 * \brief CPU Implementation of numpy matrix operations
 */

#include "./np_matrix_op-inl.h"
#include "../nn/concat-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(NumpyTransposeParam);

bool NumpyTransposeShape(const nnvm::NodeAttrs& attrs,
                         mxnet::ShapeVector *in_attrs,
                         mxnet::ShapeVector *out_attrs) {
  const NumpyTransposeParam& param = nnvm::get<NumpyTransposeParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  mxnet::TShape& shp = (*in_attrs)[0];
  CHECK_LE(shp.ndim(), 6) << "Transpose support at most 6 dimensions";
  mxnet::TShape ret(shp.ndim(), -1);
  if (ndim_is_known(param.axes)) {
    CHECK_EQ(shp.ndim(), param.axes.ndim());
    for (int i = 0; i < shp.ndim(); ++i) {
      CHECK(param.axes[i] < static_cast<int64_t>(shp.ndim()));
      ret[i] = shp[param.axes[i]];
    }
  } else {
    for (int i = 0; i < shp.ndim(); ++i) {
      ret[i] = shp[shp.ndim()-1-i];
    }
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, ret);
  return shape_is_known(ret);
}

bool HStackShape(const nnvm::NodeAttrs& attrs,
                 mxnet::ShapeVector *in_shape,
                 mxnet::ShapeVector *out_shape) {
  using namespace mshadow;
  ConcatParam param_ = nnvm::get<ConcatParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), static_cast<size_t>(param_.num_args));
  mxnet::TShape dshape;
  dim_t size = 0;
  bool has_unknown_dim_size = false;
  int axis = (*in_shape)[0].ndim() > 1 ? 1 : 0;
  param_.dim = axis;
  for (int i = 0; i < param_.num_args; ++i) {
    // scalor tensor is treated as one dimensional vector
    if ((*in_shape)[i].ndim() == 0) {
      (*in_shape)[i] = mxnet::TShape(1, 1);
    }
    mxnet::TShape &tmp = (*in_shape)[i];
    if (tmp.ndim() > 0) {
      CheckAxis(axis, tmp.ndim());
      if (!mxnet::dim_size_is_known(tmp, axis)) {
        has_unknown_dim_size = true;
      } else {
        size += tmp[axis];
      }
      tmp[axis] = -1;
      shape_assign(&dshape, tmp);
    }
  }

  mxnet::TShape tmp = (*out_shape)[0];
  if (tmp.ndim() > 0) {
    axis = CheckAxis(param_.dim, tmp.ndim());
    tmp[axis] = -1;
    shape_assign(&dshape, tmp);
  }

  if (dshape.ndim() == -1) return false;
  CHECK_NE(dshape.ndim(), 0) << "zero-dimensional arrays cannot be concatenated";

  for (int i = 0; i < param_.num_args; ++i) {
    CHECK(shape_assign(&(*in_shape)[i], dshape))
        << "Incompatible input shape: expected " << dshape << ", got " << (*in_shape)[i];
  }

  if (!has_unknown_dim_size) {
    dshape[axis] = size;
  }
  CHECK(shape_assign(&(*out_shape)[0], dshape))
      << "Incompatible output shape: expected " << dshape << ", got " << (*out_shape)[0];

  return shape_is_known(dshape);
}

bool NumpyXSliceOpShape(const nnvm::NodeAttrs& attrs,
                         mxnet::ShapeVector* in_attrs,
                         mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const mxnet::TShape& dshape = (*in_attrs)[0];
  if (!mxnet::ndim_is_known(dshape)) return false;
  const NumpyXSliceParam& param = nnvm::get<NumpyXSliceParam>(attrs.parsed);
  mxnet::TShape oshape = dshape;

  MXNET_NDIM_SWITCH(dshape.ndim(), ndim, {
    common::StaticArray<index_t, ndim> begin, end, step;
    NumpyXGetIndexRange(dshape, param.begin, param.end, param.step, &begin, &end, &step);
    for (int i = 0; i < param.begin.ndim(); ++i) {
      const int b = begin[i], e = end[i], s = step[i];
      NumpyXSetSliceOpOutputDimSize(i, b, e, s, &oshape);
    }
  })

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  return shape_is_known(oshape);
}

bool NumpyReshapeInferShape(const mxnet::TShape& src, mxnet::TShape* dst) {
  if (shape_is_known(src) && shape_is_known(*dst)) {
    CHECK_EQ(src.Size(), dst->Size()) << "Cannot reshape array of size "
                                      << src.Size() << " into shape " << *dst;
    return true;
  } else if (!shape_is_known(src) || !ndim_is_known(*dst)) {
    return false;
  } else {
    int unknown_axis = -1;
    dim_t known_dim_size_prod = 1;
    for (int i = 0; i < dst->ndim(); ++i) {
      if (!dim_size_is_known(*dst, i)) {
        if (unknown_axis == -1) {
          unknown_axis = i;
        } else {
          return false;  // more than one unknown dim
        }
      } else {
        known_dim_size_prod *= (*dst)[i];
      }
    }
    CHECK_NE(known_dim_size_prod, 0) << "Cannot reshape array of size "
                                     << src.Size() << " into shape " << *dst;
    CHECK_EQ(src.Size() % known_dim_size_prod, 0) << "Cannot reshape array of size "
                                                  << src.Size() << " into shape " << *dst;
    (*dst)[unknown_axis] = src.Size() / known_dim_size_prod;
    return true;
  }
}

bool NumpyReshapeShape(const nnvm::NodeAttrs& attrs,
                       mxnet::ShapeVector* in_attrs,
                       mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U) << "Input: [data]";
  CHECK_EQ(out_attrs->size(), 1U);
  const NumpyReshapeParam& param = nnvm::get<NumpyReshapeParam>(attrs.parsed);
  // sanity check
  bool has_unknown_dim_size = false;
  for (int i = 0; i < param.newshape.ndim(); ++i) {
    if (param.newshape[i] < 0) {
      CHECK_EQ(param.newshape[i], -1) << "The shape dimension size to inferred must be -1";
      CHECK(!has_unknown_dim_size) << "Can only specify one unknown dimension";
      has_unknown_dim_size = true;
    }
  }

  mxnet::TShape target_shape = param.newshape;
  bool success = NumpyReshapeInferShape(in_attrs->at(0), &target_shape);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, target_shape);
  if (!success) {
    success = NumpyReshapeInferShape(out_attrs->at(0), &in_attrs->at(0));
  }
  return success;
}

bool NumpyXReshapeInferShape(const mxnet::TShape& src,
                             const mxnet::Tuple<int>& target,
                             mxnet::TShape* output) {
  bool target_shape_is_known = true;
  dim_t target_size = 1;
  for (int i = 0; i < target.ndim(); ++i) {
    if (target[i] < 0) {
      target_shape_is_known = false;
      target_size  = -1;
      break;
    } else {
      target_size *= target[i];
    }
  }
  if (shape_is_known(src) && target_shape_is_known) {
    CHECK_EQ(src.Size(), target_size) << "Cannot reshape array of size "
                                      << src.Size() << " into shape " << target;
    *output = TShape(target.begin(), target.end());
    return true;
  } else if (!shape_is_known(src) || target.ndim() == -1) {
    return false;
  } else {
    int unknown_axis = -1;
    dim_t known_dim_size_prod = 1;
    std::vector<dim_t> output_shape_vector;
    int src_inx = 0;
    for (int i = 0; i < target.ndim(); ++i) {
      dim_t proposed_dim = target[i];
      CHECK(proposed_dim >= -6)
        << "Dimension size must be greater than -6, received " << proposed_dim;
      if (proposed_dim == -1) {
        // infer the known dimension
        CHECK_LT(unknown_axis, 0)
          << "One and only one dim can be inferred";
        unknown_axis = output_shape_vector.size();
        output_shape_vector.push_back(1);
        src_inx++;
      } else if (proposed_dim == -2) {
        // copy the dimension from src to output
        CHECK_LT(src_inx, src.ndim())
          << "Unmatching dimension of proposed new shape";
        known_dim_size_prod *= src[src_inx];
        output_shape_vector.push_back(src[src_inx++]);
      } else if (proposed_dim == -3) {
        // skip the source dimension if and only if it is one
        CHECK_EQ(src[src_inx], 1)
          <<"-3 index should only be used to skip dimision size 1";
        src_inx++;
      } else if (proposed_dim == -4) {
        // copy all remaining dims from source
        while (src_inx < src.ndim()) {
          known_dim_size_prod *= src[src_inx];
          const int dn = src[src_inx++];
          output_shape_vector.push_back(dn);
        }
      } else if (proposed_dim == -5) {
        // merge two dims from source
        CHECK_LT(src_inx, src.ndim()-1)
          <<"Not enough dimensions left for the product";
        const int d1 = src[src_inx++];
        const int d2 = src[src_inx++];
        if (!mxnet::dim_size_is_known(d1) || !mxnet::dim_size_is_known(d2)) {
          CHECK_LT(unknown_axis, 0)
          << "One and only one dim can be inferred";
          unknown_axis = output_shape_vector.size();
          output_shape_vector.push_back(-1);
        } else {
          known_dim_size_prod *= d1*d2;
          output_shape_vector.push_back(d1 * d2);
        }
      } else if (proposed_dim == -6) {
        // split the source dim s into two dims
        // read the left dim and then the right dim (either can be -1)
        CHECK_LT(i + 2, target.ndim());
        CHECK_LT(src_inx, src.ndim());
        const int d0 = src[src_inx++];
        dim_t d1 = target[++i];
        dim_t d2 = target[++i];
        CHECK(d1 != -1 || d2 != -1) << "Split dims cannot both be -1.";
        if (d1 == -1 && d0 >= 0) d1 = d0 / d2;  // d0 must be known to do this
        if (d2 == -1 && d0 >= 0) d2 = d0 / d1;  // d0 must be known to do this
        CHECK(d1 * d2 == static_cast<dim_t>(d0) || static_cast<dim_t>(d0) == dim_t(-1))
          <<"Split dims " << d1 << ", " << d2 << " do not divide original dim " << d0;
        if (d1 == -1) {
          CHECK_LT(unknown_axis, 0)
          << "One and only one dim can be inferred";
          unknown_axis = output_shape_vector.size();
        } else if (d2 == -1) {
          CHECK_LT(unknown_axis, 0)
          << "One and only one dim can be inferred";
          unknown_axis = output_shape_vector.size() + 1;
        }
        known_dim_size_prod *= d0 == -1 ? 1 : d0;
        output_shape_vector.push_back(d1);
        output_shape_vector.push_back(d2);
      } else {
        // greater than 0, new shape
        known_dim_size_prod *= proposed_dim;
        output_shape_vector.push_back(proposed_dim);
        src_inx++;
      }
    }

    if (unknown_axis > -1) {
      // if the input in zero size tensor, the output must be of known shape of zero size
      CHECK_NE(known_dim_size_prod, 0) << "Cannot reshape array of size "
                                      << src.Size() << " into shape " << target;
      CHECK(src.Size() % known_dim_size_prod == 0)
        << "Cannot reshape array of size " << src.Size() << " into shape " << target;
      output_shape_vector[unknown_axis] = src.Size() / known_dim_size_prod;
    }

    *output = mxnet::TShape(output_shape_vector.begin(), output_shape_vector.end());
    CHECK_EQ((*output).Size(), src.Size())
      << "Target output shape of size " << (*output).Size()
      << " does not match the input shape of size " << src.Size();
    return true;
  }
}

bool NumpyXReshapeShape(const nnvm::NodeAttrs& attrs,
                       mxnet::ShapeVector* in_attrs,
                       mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U) << "Input: [data]";
  CHECK_EQ(out_attrs->size(), 1U);
  const NumpyXReshapeParam& param = nnvm::get<NumpyXReshapeParam>(attrs.parsed);
  // sanity check
  bool has_unknown_dim_size = false;
  for (int i = 0; i < param.newshape.ndim(); ++i) {
    if (param.newshape[i] < 0) {
      CHECK_GE(param.newshape[i], -6)
      << "Dimension size must be greater than or equal to -6";
      if (param.newshape[i] == -1) {
        CHECK(!has_unknown_dim_size) << "Can only specify one unknown dimension";
        has_unknown_dim_size = true;
      }
    }
  }

  mxnet::TShape output_shape;
  bool success = NumpyXReshapeInferShape(in_attrs->at(0), param.newshape, &output_shape);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, output_shape);
  if (!success) {
    success = ReverseReshapeInferShape(&(*in_attrs)[0], (*out_attrs)[0]);
  }
  return success;
}

NNVM_REGISTER_OP(_np_transpose)
.describe(R"code(Permute the dimensions of an array.

Examples::

  x = [[ 1, 2],
       [ 3, 4]]

  transpose(x) = [[ 1.,  3.],
                  [ 2.,  4.]]

  x = [[[ 1.,  2.],
        [ 3.,  4.]],

       [[ 5.,  6.],
        [ 7.,  8.]]]

  transpose(x) = [[[ 1.,  5.],
                   [ 3.,  7.]],

                  [[ 2.,  6.],
                   [ 4.,  8.]]]

  transpose(x, axes=(1,0,2)) = [[[ 1.,  2.],
                                 [ 5.,  6.]],

                                [[ 3.,  4.],
                                 [ 7.,  8.]]]
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyTransposeParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyTransposeShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    const NumpyTransposeParam& param = nnvm::get<NumpyTransposeParam>(n->attrs.parsed);
    if (ndim_is_known(param.axes)) {
      mxnet::TShape axes = mxnet::TShape(param.axes.ndim(), -1);
      for (int i = 0; i < axes.ndim(); ++i) {
        axes[param.axes[i]] = i;
      }
      std::ostringstream os;
      os << axes;
      return MakeNonlossGradNode("transpose", n, ograds, {}, {{"axes", os.str()}});
    } else {
      return MakeNonlossGradNode("transpose", n, ograds, {},
                                 std::unordered_map<std::string, std::string>());
    }
  })
.set_attr<FCompute>("FCompute<cpu>", NumpyTranspose<cpu>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a"};
  })
.add_argument("a", "NDArray-or-Symbol", "Source input")
.add_arguments(NumpyTransposeParam::__FIELDS__());

DMLC_REGISTER_PARAMETER(NumpyReshapeParam);

NNVM_REGISTER_OP(_np_reshape)
.describe(R"code()code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyReshapeParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyReshapeShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_reshape"})
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::IdentityCompute<cpu>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    return std::vector<bool>{true};
  })
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a"};
  })
.add_argument("a", "NDArray-or-Symbol", "Array to be reshaped.")
.add_arguments(NumpyReshapeParam::__FIELDS__());

DMLC_REGISTER_PARAMETER(NumpyXReshapeParam);

NNVM_REGISTER_OP(_npx_reshape)
.add_alias("_npi_reshape")
.describe(R"code()code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyXReshapeParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyXReshapeShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_reshape"})
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::IdentityCompute<cpu>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    return std::vector<bool>{true};
  })
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a"};
  })
.add_argument("a", "NDArray-or-Symbol", "Array to be reshaped.")
.add_arguments(NumpyXReshapeParam::__FIELDS__());

NNVM_REGISTER_OP(_npi_stack)
.describe(R"code(Join a sequence of arrays along a new axis.

The axis parameter specifies the index of the new axis in the dimensions of the
result. For example, if axis=0 it will be the first dimension and if axis=-1 it
will be the last dimension.

Examples::

  x = [1, 2]
  y = [3, 4]

  stack(x, y) = [[1, 2],
                 [3, 4]]
  stack(x, y, axis=1) = [[1, 3],
                         [2, 4]]
)code")
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    const StackParam& param = dmlc::get<StackParam>(attrs.parsed);
    return static_cast<uint32_t>(param.num_args);
  })
.set_num_outputs(1)
.set_attr_parser(ParamParser<StackParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    uint32_t num_args = dmlc::get<StackParam>(attrs.parsed).num_args;
    std::vector<std::string> ret;
    for (uint32_t i = 0; i < num_args; ++i) {
      ret.push_back(std::string("arg") + std::to_string(i));
    }
    return ret;
  })
.set_attr<std::string>("key_var_num_args", "num_args")
.set_attr<mxnet::FInferShape>("FInferShape", StackOpShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<-1, 1>)
.set_attr<FCompute>("FCompute<cpu>", StackOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_stack"})
.add_argument("data", "NDArray-or-Symbol[]", "List of arrays to stack")
.add_arguments(StackParam::__FIELDS__());

bool ConcatShape(const nnvm::NodeAttrs& attrs,
                 mxnet::ShapeVector *in_shape,
                 mxnet::ShapeVector *out_shape);

bool ConcatType(const nnvm::NodeAttrs& attrs,
                std::vector<int> *in_type,
                std::vector<int> *out_type);

struct NumpyConcatGrad {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    CHECK_EQ(ograds.size(), 1);
    std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};


NNVM_REGISTER_OP(_npi_concatenate)
.describe(R"code(Join a sequence of arrays along an existing axis.)code" ADD_FILELINE)
.set_num_inputs([](const NodeAttrs& attrs) {
  const ConcatParam& params = nnvm::get<ConcatParam>(attrs.parsed);
  return params.num_args;
})
.set_num_outputs(1)
.set_attr_parser(ParamParser<ConcatParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const ConcatParam& params = nnvm::get<ConcatParam>(attrs.parsed);
    std::vector<std::string> ret;
    for (int i = 0; i < params.num_args; ++i) {
      ret.push_back(std::string("data") + std::to_string(i));
    }
    return ret;
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"out"};
})
.set_attr<std::string>("key_var_num_args", "num_args")
.set_attr<nnvm::FInferType>("FInferType", ConcatType)
.set_attr<mxnet::FInferShape>("FInferShape", ConcatShape)
.set_attr<FCompute>("FCompute<cpu>", ConcatCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", NumpyConcatGrad{"_backward_np_concat"})
.add_argument("data", "NDArray-or-Symbol[]", "List of arrays to concatenate")
.add_arguments(ConcatParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_np_concat)
.set_num_outputs([](const NodeAttrs& attrs) {
  const ConcatParam& params = nnvm::get<ConcatParam>(attrs.parsed);
  return params.num_args;
})
.set_attr_parser(ParamParser<ConcatParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", ConcatGradCompute<cpu>);

NNVM_REGISTER_OP(_npi_hstack)
.describe(R"code(Stack tensors horizontally (in second dimension))code" ADD_FILELINE)
.set_num_inputs([](const NodeAttrs& attrs) {
  const ConcatParam& params = nnvm::get<ConcatParam>(attrs.parsed);
  return params.num_args;
})
.set_num_outputs(1)
.set_attr_parser(ParamParser<ConcatParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const ConcatParam& params = nnvm::get<ConcatParam>(attrs.parsed);
    std::vector<std::string> ret;
    for (int i = 0; i < params.num_args; ++i) {
      ret.push_back(std::string("data") + std::to_string(i));
    }
    return ret;
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"out"};
})
.set_attr<std::string>("key_var_num_args", "num_args")
.set_attr<nnvm::FInferType>("FInferType", ConcatType)
.set_attr<mxnet::FInferShape>("FInferShape", HStackShape)
.set_attr<FCompute>("FCompute<cpu>", HStackCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", NumpyConcatGrad{"_backward_np_hstack"})
.add_argument("data", "NDArray-or-Symbol[]", "List of arrays to concatenate")
.add_arguments(ConcatParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_np_hstack)
.set_num_outputs([](const NodeAttrs& attrs) {
  const ConcatParam& params = nnvm::get<ConcatParam>(attrs.parsed);
  return params.num_args;
})
.set_attr_parser(ParamParser<ConcatParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", HStackGradCompute<cpu>);

bool NumpySqueezeShape(const nnvm::NodeAttrs& attrs,
                       mxnet::ShapeVector *in_attrs,
                       mxnet::ShapeVector *out_attrs) {
  const SqueezeParam& param = nnvm::get<SqueezeParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U) << "Input: [a]";
  CHECK_EQ(out_attrs->size(), 1U);
  const mxnet::TShape& dshape = in_attrs->at(0);
  const int dndim = dshape.ndim();
  if (!shape_is_known(dshape)) return false;
  mxnet::TShape oshape = dshape;
  // special case, scalar tensor
  if (dshape.ndim() == 0) {
    if (param.axis.has_value()) {
      mxnet::Tuple<int> axes = param.axis.value();
      CHECK_EQ(axes.ndim(), 1) << "cannot specify more than one axis for a scalar tensor";
      CHECK(axes[0] == 0 || axes[0] == -1) << "axis " << axes[0]
                                           << " is out of bounds of array of dimension 0";
    }
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape(0, -1));
    return true;
  }
  if (param.axis.has_value()) {
    // preprocess axis
    mxnet::Tuple<int> axes = param.axis.value();
    for (int i = 0; i < axes.ndim(); ++i) {
      if (axes[i] < 0) {
        axes[i] += dndim;
        CHECK_GE(axes[i], 0)
            << "axis " << axes[i] - dndim << " is out of bounds for array of dimension " << dndim;
      }
      CHECK_LT(axes[i], dndim)
          << "axis " << axes[i] << " is out of bounds for array of dimension " << dndim;
      CHECK_EQ(dshape[axes[i]], 1)
          << "cannot select an axis to squeeze out which has size="
          << dshape[axes[i]] << " not equal to one";
      CHECK_NE(oshape[axes[i]], 0) << "duplicate value in axis";
      oshape[axes[i]] = -1;
    }
  } else {
    for (int i = 0; i < oshape.ndim(); ++i) {
      if (oshape[i] == 1) oshape[i] = -1;
    }
  }
  size_t oshape_size = SqueezeShapeHelper(&oshape);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape(oshape.data(), oshape.data()+oshape_size));
  return true;
}

NNVM_REGISTER_OP(_np_squeeze)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SqueezeParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", NumpySqueezeShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::IdentityCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_squeeze"})
.add_argument("a", "NDArray-or-Symbol[]", "data to squeeze")
.add_arguments(SqueezeParam::__FIELDS__());

DMLC_REGISTER_PARAMETER(NumpyXSliceParam);

NNVM_REGISTER_OP(_npx_slice)
.add_alias("_npi_slice")
// MXNET_ADD_SPARSE_OP_ALIAS(slice)
.describe(R"code(Slices a region of the array.

.. note:: ``crop`` is deprecated. Use ``slice`` instead.

This function returns a sliced array between the indices given
by `begin` and `end` with the corresponding `step`.

For an input array of ``shape=(d_0, d_1, ..., d_n-1)``,
slice operation with ``begin=(b_0, b_1...b_m-1)``,
``end=(e_0, e_1, ..., e_m-1)``, and ``step=(s_0, s_1, ..., s_m-1)``,
where m <= n, results in an array with the shape
``(|e_0-b_0|/|s_0|, ..., |e_m-1-b_m-1|/|s_m-1|, d_m, ..., d_n-1)``.

The resulting array's *k*-th dimension contains elements
from the *k*-th dimension of the input array starting
from index ``b_k`` (inclusive) with step ``s_k``
until reaching ``e_k`` (exclusive).

If the *k*-th elements are `None` in the sequence of `begin`, `end`,
and `step`, the following rule will be used to set default values.
If `s_k` is `None`, set `s_k=1`. If `s_k > 0`, set `b_k=0`, `e_k=d_k`;
else, set `b_k=d_k-1`, `e_k=-1`.

The storage type of ``slice`` output depends on storage types of inputs

- slice(csr) = csr
- otherwise, ``slice`` generates output with default storage

.. note:: When input data storage type is csr, it only supports
   step=(), or step=(None,), or step=(1,) to generate a csr output.
   For other step parameter values, it falls back to slicing
   a dense tensor.

Example::

  x = [[  1.,   2.,   3.,   4.],
       [  5.,   6.,   7.,   8.],
       [  9.,  10.,  11.,  12.]]

  slice(x, begin=(0,1), end=(2,4)) = [[ 2.,  3.,  4.],
                                     [ 6.,  7.,  8.]]
  slice(x, begin=(None, 0), end=(None, 3), step=(-1, 2)) = [[9., 11.],
                                                            [5.,  7.],
                                                            [1.,  3.]]
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<NumpyXSliceParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyXSliceOpShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
// .set_attr<FInferStorageType>("FInferStorageType", SliceForwardInferStorageType)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_npx_slice"})
.set_attr<FCompute>("FCompute<cpu>", NumpyXSliceOpForward<cpu>)
#if MXNET_USE_MKLDNN == 1
.set_attr<bool>("TIsMKLDNN", true)
#endif
.add_argument("data", "NDArray-or-Symbol", "Source input")
.add_arguments(NumpyXSliceParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_npx_slice)
.set_attr_parser(ParamParser<NumpyXSliceParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", NumpyXSliceOpBackward<cpu>);

}  // namespace op
}  // namespace mxnet
