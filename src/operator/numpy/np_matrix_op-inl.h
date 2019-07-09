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
 * \file np_matrix_op-inl.h
 * \brief Function definition of matrix related operators
 */
#ifndef MXNET_OPERATOR_NUMPY_NP_MATRIX_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_MATRIX_OP_INL_H_

#include <vector>
#include <string>
#include "../tensor/matrix_op-inl.h"
#include "np_broadcast_reduce_op.h"

namespace mxnet {
namespace op {

struct NumpyTransposeParam : public dmlc::Parameter<NumpyTransposeParam> {
  mxnet::TShape axes;
  DMLC_DECLARE_PARAMETER(NumpyTransposeParam) {
    DMLC_DECLARE_FIELD(axes).set_default(mxnet::TShape(-1, 0))
    .describe("By default, reverse the dimensions, otherwise permute "
              "the axes according to the values given.");
  }
};

struct NumpyReshapeParam : public dmlc::Parameter<NumpyReshapeParam> {
  mxnet::TShape newshape;
  std::string order;
  DMLC_DECLARE_PARAMETER(NumpyReshapeParam) {
      DMLC_DECLARE_FIELD(newshape)
          .describe("The new shape should be compatible with the original shape."
                    " If an integer, then the result will be a 1-D array of that length."
                    " One shape dimension can be -1. In this case, the value is inferred"
                    " from the length of the array and remaining dimensions.");
      DMLC_DECLARE_FIELD(order)
      .set_default("C")
      .describe("Read the elements of a using this index order, and place the elements into"
                " the reshaped array using this index order. 'C' means to read/write the elements"
                " using C-like index order, with the last axis index changing fastest, back to the"
                " first axis index changing slowest. Note that currently only C-like order is"
                " supported");
  }
};

struct NumpyXSliceParam : public dmlc::Parameter<NumpyXSliceParam> {
  mxnet::Tuple<dmlc::optional<int>> begin, end;
  mxnet::Tuple<dmlc::optional<int>> step;
  DMLC_DECLARE_PARAMETER(NumpyXSliceParam) {
    DMLC_DECLARE_FIELD(begin)
    .describe("starting indices for the slice operation, supports negative indices.");
    DMLC_DECLARE_FIELD(end)
    .describe("ending indices for the slice operation, supports negative indices.");
    DMLC_DECLARE_FIELD(step)
    .set_default(mxnet::Tuple<dmlc::optional<int>>())
    .describe("step for the slice operation, supports negative values.");
  }
  bool operator==(const NumpyXSliceParam& other) const {
    return this->begin == other.begin &&
           this->end == other.end &&
           this->step == other.step;
  }
};

struct NumpyXReshapeParam : public dmlc::Parameter<NumpyXReshapeParam> {
  mxnet::Tuple<int> newshape;
  std::string order;
  DMLC_DECLARE_PARAMETER(NumpyXReshapeParam) {
      DMLC_DECLARE_FIELD(newshape)
          .set_default(mxnet::Tuple<int>())
          .describe("The new shape should be compatible with the original shape."
                    " If an integer, then the result will be a 1-D array of that length."
                    " One shape dimension can be -1. In this case, the value is inferred"
                    " from the length of the array and remaining dimensions."
                    " -2 to -6 are used for data manipulation"
                    " -2 copy this dimension from the input to the output shape"
                    " -3 will skip current dimension if and only if the current dim size is one"
                    " -4 copy all remain of the input dimensions to the output shape"
                    " -5 use the product of two consecutive dimensions of the input"
                    " shape as the output"
                    " -6 split one dimension of the input into two dimensions passed"
                    " subsequent to -6 in the new shape");
      DMLC_DECLARE_FIELD(order)
      .set_default("C")
      .describe("Read the elements of a using this index order, and place the elements into"
                " the reshaped array using this index order. 'C' means to read/write the elements"
                " using C-like index order, with the last axis index changing fastest, back to the"
                " first axis index changing slowest. Note that currently only C-like order is"
                " supported");
  }
};

template<typename xpu>
void NumpyTranspose(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  const NumpyTransposeParam& param = nnvm::get<NumpyTransposeParam>(attrs.parsed);
  CHECK_EQ(req[0], kWriteTo) << "Transpose does not support inplace";
  if (ndim_is_known(param.axes)) {
    TransposeImpl<xpu>(ctx.run_ctx, inputs[0], outputs[0], param.axes);
  } else {
    mxnet::TShape axes(inputs[0].ndim(), -1);
    for (int i = 0; i < axes.ndim(); ++i) {
      axes[i] = axes.ndim() - 1 - i;
    }
    TransposeImpl<xpu>(ctx.run_ctx, inputs[0], outputs[0], axes);
  }
}

template<int ndim>
inline void NumpyXGetIndexRange(const mxnet::TShape& dshape,
                          const mxnet::Tuple<dmlc::optional<int>>& param_begin,
                          const mxnet::Tuple<dmlc::optional<int>>& param_end,
                          const mxnet::Tuple<dmlc::optional<int>>& param_step,
                          common::StaticArray<index_t, ndim>* begin,
                          common::StaticArray<index_t, ndim>* end,
                          common::StaticArray<index_t, ndim>* step) {
  CHECK_NE(dshape.ndim(), 0U);
  CHECK_LE(param_begin.ndim(), dshape.ndim())
    << "Slicing axis exceeds data dimensions";
  CHECK_LE(param_end.ndim(), dshape.ndim())
    << "Slicing axis exceeds data dimensions";
  CHECK_EQ(param_begin.ndim(), param_end.ndim())
    << "begin and end must have the same length";
  CHECK_EQ(ndim, dshape.ndim())
    << "Static array size=" << ndim
    << " is not equal to data shape ndim=" << dshape.ndim();

  if (param_step.ndim() > 0) {
    CHECK_EQ(param_step.ndim(), param_begin.ndim())
      << "step and begin must have the same length";
  }

  for (int i = 0; i < param_begin.ndim(); ++i) {
    index_t s = param_step.ndim() > 0 && param_step[i].has_value()?
                param_step[i].value() : 1;
    CHECK_NE(s, 0) << "slice op step[" << i << "] cannot be 0";

    index_t b = 0, e = 0;
    const index_t len = dshape[i];
    if (len > 0) {
      b = param_begin[i].has_value() ? param_begin[i].value() : (s < 0 ? len - 1 : 0);
      e = param_end[i].has_value() ? param_end[i].value() : (s < 0 ? -1 : len);

      if (b < 0) {
        b += len;
      }

      if (e < 0 && param_end[i].has_value()) {
        e += len;
      }
    }

    // move the begin and end to correct position for calculating dim size
    b = b < 0 && s > 0 ? 0 : b;
    b = b > len-1 && s < 0 ? len-1 : b;
    // if the start value lead to empty tensor under step s, use -1 for indication
    b = b < 0 || b > len-1 ? -1 : b;
    e = e > -1 ? e : -1;
    e = e > len ? len : e;
    (*begin)[i] = b;
    (*end)[i] = e;
    (*step)[i] = s;
  }

  for (index_t i = param_begin.ndim(); i < dshape.ndim(); ++i) {
    (*begin)[i] = 0;
    (*end)[i] = dshape[i];
    (*step)[i] = 1;
  }
}

inline void NumpyXSetSliceOpOutputDimSize(const index_t i, const int b,
                                    const int e, const int s,
                                    mxnet::TShape* oshape) {
  if (e != b && b >= 0) {
    if (s > 0) {
      (*oshape)[i] = e > b ? (e - b - 1) / s + 1 : 0;
    } else {
      (*oshape)[i] = e < b ? (b - e - 1) / (-s) + 1 : 0;
    }
  } else {
      (*oshape)[i] = 0;
  }
}

inline bool NumpyXSliceOpShape(const nnvm::NodeAttrs& attrs,
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

inline bool NumpyXSliceAssignOpShape(const nnvm::NodeAttrs& attrs,
                               mxnet::ShapeVector *in_attrs,
                               mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const mxnet::TShape& dshape = (*in_attrs)[0];
  if (!mxnet::ndim_is_known(dshape)) return false;
  mxnet::TShape vshape = dshape;  // vshape is the value shape on the right hand side
  const NumpyXSliceParam& param = nnvm::get<NumpyXSliceParam>(attrs.parsed);
  MXNET_NDIM_SWITCH(dshape.ndim(), ndim, {
    common::StaticArray<index_t, ndim> begin, end, step;
    NumpyXGetIndexRange(dshape, param.begin, param.end, param.step, &begin, &end, &step);
    for (int i = 0; i < param.begin.ndim(); ++i) {
      const int b = begin[i], e = end[i], s = step[i];
      NumpyXSetSliceOpOutputDimSize(i, b, e, s, &vshape);
    }
  })

  // Check if the value is broadcastable to the target slicing region
  mxnet::TShape& input_shape = (*in_attrs)[1];
  CHECK(mxnet::shape_is_known(input_shape))
    << "The input values must be of known shape.";
  CHECK(mxnet::shape_is_known(vshape))
      << "The target shape for broadcasting array must be known.";
  CHECK_LE(input_shape.ndim(), vshape.ndim())
      << "shape " << input_shape << " is not broadcastable to " << vshape;
  for (int i = vshape.ndim() - 1; i >= 0; --i) {
    int j = i - vshape.ndim() + input_shape.ndim();
    if (j < 0) break;
    CHECK(input_shape[j] == vshape[i] || input_shape[j] == 1)
        << "shape " << input_shape << " is not broadcastable to " << vshape;
  }
  // (*in_attrs)[1] = vshape;
  // SHAPE_ASSIGN_CHECK(*in_attrs, 1, vshape);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, dshape);
  return true;
}

template<typename xpu>
void NumpyXSliceOpForward(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  if (req[0] == kNullOp) return;
  using namespace mshadow;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  const TBlob& data = inputs[0];
  const TBlob& out = outputs[0];
  if (out.Size() == 0) {
    return;
  }
  const NumpyXSliceParam& param = nnvm::get<NumpyXSliceParam>(attrs.parsed);
  MXNET_NDIM_SWITCH(data.ndim(), ndim, {
    common::StaticArray<index_t, ndim> begin, end, step;
    NumpyXGetIndexRange(data.shape_, param.begin, param.end, param.step, &begin, &end, &step);
    MSHADOW_TYPE_SWITCH(out.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        size_t num_threads = out.shape_.FlatTo2D()[0];
        if (std::is_same<xpu, gpu>::value) {
          num_threads *= out.shape_.get<ndim>()[ndim - 1];
        }
        mxnet_op::Kernel<slice_forward<ndim, Req, xpu>, xpu>::Launch(s, num_threads,
            out.dptr<DType>(), data.dptr<DType>(),
            data.shape_.get<ndim>(), out.shape_.get<ndim>(), begin, step);
      })
    })
  })
}

template<typename xpu>
void NumpyXSliceOpBackward(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  if (req[0] == kNullOp) return;
  using namespace mshadow;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  const TBlob& ograd = inputs[0];
  const TBlob& igrad = outputs[0];
  const NumpyXSliceParam& param = nnvm::get<NumpyXSliceParam>(attrs.parsed);
  if (req[0] == kWriteTo) {
    Fill(s, igrad, req[0], 0);
  } else if (req[0] == kWriteInplace) {
    LOG(FATAL) << "_slice_backward does not support kWriteInplace";
  }
  if (ograd.Size() == 0) return;
  MXNET_NDIM_SWITCH(ograd.ndim(), ndim, {
    common::StaticArray<index_t, ndim> begin, end, step;
    NumpyXGetIndexRange(igrad.shape_, param.begin, param.end, param.step, &begin, &end, &step);
    MSHADOW_TYPE_SWITCH(ograd.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
      int num_threads = ograd.shape_.FlatTo2D()[0];
      if (std::is_same<xpu, gpu>::value) {
        num_threads *= ograd.shape_.get<ndim>()[ndim - 1];
      }
      mxnet_op::Kernel<slice_assign<ndim, Req, xpu>, xpu>::Launch(s, num_threads,
          igrad.dptr<DType>(), ograd.dptr<DType>(),
          igrad.shape_.get<ndim>(), ograd.shape_.get<ndim>(), begin, step);
      })
    })
  })
}

template<typename xpu>
void NumpyXSliceAssignOpForward(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 2U);  // data[index] = val, data and val are two inputs
  CHECK_EQ(outputs.size(), 1U);
  if (req[0] == kNullOp) return;

  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& data = inputs[0];
  const TBlob& val = inputs[1];
  const TBlob& out = outputs[0];
  if (req[0] == kWriteTo) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      Tensor<xpu, 1, DType> in = inputs[0].FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
      Copy(out, in, s);
    });
  } else if (req[0] != kWriteInplace) {
    LOG(FATAL) << "_npx_slice_assign only supports kWriteTo and kWriteInplace";
  }

  const NumpyXSliceParam& param = nnvm::get<NumpyXSliceParam>(attrs.parsed);
  MXNET_NDIM_SWITCH(data.ndim(), ndim, {
    common::StaticArray<index_t, ndim> begin, end, step;
    NumpyXGetIndexRange(data.shape_, param.begin, param.end, param.step, &begin, &end, &step);
    MSHADOW_TYPE_SWITCH(out.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        int num_threads = val.shape_.FlatTo2D()[0];
        if (std::is_same<xpu, gpu>::value) {
          num_threads *= val.shape_.get<ndim>()[ndim - 1];
        }
        mxnet_op::Kernel<slice_assign<ndim, Req, xpu>, xpu>::Launch(s, num_threads,
            out.dptr<DType>(), val.dptr<DType>(),
            out.shape_.get<ndim>(), val.shape_.get<ndim>(), begin, step);
      })
    })
  })
}

// template<typename xpu>
// void NumpyBroadcastToForward(const nnvm::NodeAttrs& attrs,
//                              const OpContext& ctx,
//                              const std::vector<TBlob>& inputs,
//                              const std::vector<OpReqType>& req,
//                              const std::vector<TBlob>& outputs) {
//   if (outputs[0].shape_.Size() == 0U) return;  // zero-size tensor
//   TShape expanded_ishape(outputs[0].shape_.ndim(), 1);
//   const TShape& ishape = inputs[0].shape_;
//   CHECK_LE(ishape.ndim(), expanded_ishape.ndim()) << "output ndim cannot be less than input ndim";
//   const int ndim_delta = expanded_ishape.ndim() - ishape.ndim();
//   for (int i = 0; i < ishape.ndim(); ++i) {
//     expanded_ishape[i + ndim_delta] = ishape[i];
//   }
//   BroadcastComputeImpl<xpu>(attrs, ctx, {inputs[0].reshape(expanded_ishape)},
//                             req, outputs, expanded_ishape);
// }

template<typename xpu>
void NumpyXSliceAssignScalarOpForward(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  using namespace mshadow;
  Stream<xpu> *s = ctx.get_stream<xpu>();

  const TBlob& data = inputs[0];
  const TBlob& out = outputs[0];
  if (req[0] == kWriteTo) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      Tensor<xpu, 1, DType> in = inputs[0].FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
      Copy(out, in, s);
    });
  } else if (req[0] != kWriteInplace) {
    LOG(FATAL) << "_npx_slice_assign_scalar only supports kWriteTo and kWriteInplace";
  }

  mxnet::TShape vshape = data.shape_;
  const SliceAssignScalarParam& param = nnvm::get<SliceAssignScalarParam>(attrs.parsed);
  MXNET_NDIM_SWITCH(data.ndim(), ndim, {
    common::StaticArray<index_t, ndim> begin, end, step;
    NumpyXGetIndexRange(data.shape_, param.begin, param.end, param.step, &begin, &end, &step);
    for (index_t i = 0; i < param.begin.ndim(); ++i) {
      const int b = begin[i], e = end[i], s = step[i];
      NumpyXSetSliceOpOutputDimSize(i, b, e, s, &vshape);
    }
    MSHADOW_TYPE_SWITCH(out.type_flag_, DType, {
      mxnet_op::Kernel<slice_assign_scalar<ndim>, xpu>::Launch(s, vshape.FlatTo2D()[0],
          out.dptr<DType>(), static_cast<DType>(param.scalar), req[0],
          out.shape_.get<ndim>(), vshape.get<ndim>(), begin, step);
    })
  })
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

}  // namespace op
}  // namespace mxnet

namespace std {
template<>
struct hash<mxnet::op::NumpyXSliceParam> {
  size_t operator()(const mxnet::op::NumpyXSliceParam& val) {
    size_t ret = 0;
    ret = dmlc::HashCombine(ret, val.begin);
    ret = dmlc::HashCombine(ret, val.end);
    ret = dmlc::HashCombine(ret, val.step);
    return ret;
  }
};
}  // namespace std

#endif  // MXNET_OPERATOR_NUMPY_NP_MATRIX_OP_INL_H_
