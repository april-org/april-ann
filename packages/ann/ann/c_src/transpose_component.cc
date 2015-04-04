/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2015, Francisco Zamora-Martinez
 *
 * The APRIL-ANN toolkit is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this library; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 */
#include "unused_variable.h"
#include "error_print.h"
#include "table_of_token_codes.h"
#include "token_vector.h"
#include "token_matrix.h"
#include "transpose_component.h"

using namespace Basics;
using namespace AprilUtils;
using namespace AprilMath;

namespace ANN {
  
  TransposeANNComponent::TransposeANNComponent(const int *which,
                                               const char *name) :
    VirtualMatrixANNComponent(name, 0, 0, 0) {
    if (which != 0) {
      this->which = new int[2];
      this->which[0] = which[0];
      this->which[1] = which[1];
    }
  }
  
  TransposeANNComponent::~TransposeANNComponent() {
  }

  MatrixFloat *TransposeANNComponent::
  transposeBunch(MatrixFloat *input_mat) const {
    const int N = input_mat->getNumDim();
    AprilUtils::UniquePtr<int []> dims( new int[N] );
    dims[0] = input_mat->getDimSize(0);
    if (!which.empty()) {
      for (int i=1; i<N; ++i) {
        dims[i] = input_mat->getDimSize(i);
      }
      AprilUtils::swap(dims[which[0]+1], dims[which[1]+1]);
    }
    else {
      for (int i=1; i<N; ++i) {
        dims[i] = input_mat->getDimSize(N - i);
      }
    }
    MatrixFloat *output_mat = new MatrixFloat(N, dims.get());
    AprilUtils::SharedPtr<MatrixFloat> output_row;
    AprilUtils::SharedPtr<MatrixFloat> input_row;
    for(int i=0; i<dims[0]; ++i) {
      output_row = output_mat->select(0, i, output_row.get());
      input_row  = input_mat->select(0, i, input_row.get());
      AprilUtils::UniquePtr<MatrixFloat> input_row_T;
      if (!which.empty()) {
        input_row_T = input_row->transpose(which[0], which[1]);
      }
      else {
        input_row_T = input_row->transpose();
      }
      AprilMath::MatrixExt::BLAS::
        matCopy(output_row.get(), input_row_T.get());
    }
    return output_mat;
  }
  
  MatrixFloat *TransposeANNComponent::
  privateDoForward(MatrixFloat* input_mat, bool during_training) {
    UNUSED_VARIABLE(during_training);
    return transposeBunch(input_mat);
  }

  MatrixFloat *TransposeANNComponent::
  privateDoBackprop(MatrixFloat *error_input_mat) {
    return transposeBunch(error_input_mat);
  }
  
  void TransposeANNComponent::privateReset(unsigned int it) {
    UNUSED_VARIABLE(it);
  }

  ANNComponent *TransposeANNComponent::clone() {
    TransposeANNComponent *transpose_component =
      new TransposeANNComponent(which.get(), name.c_str());
    return transpose_component;
  }
  
  char *TransposeANNComponent::toLuaString() {
    buffer_list buffer;
    if (!which.empty()) {
      buffer.printf("ann.components.transpose{ name='%s', dims={%d,%d} }",
                    name.c_str(), which[0]+1, which[1]+1);
    }
    else {
      buffer.printf("ann.components.transpose{ name='%s' }", name.c_str());
    }
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
}
