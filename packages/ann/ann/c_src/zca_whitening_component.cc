/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Francisco Zamora-Martinez
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
#include "zca_whitening_component.h"
#include "wrapper.h"
#include "utilMatrixFloat.h"

namespace ANN {
  
  ZCAWhiteningANNComponent::ZCAWhiteningANNComponent(MatrixFloat *U,
						     MatrixFloat *S,
						     float epsilon,
						     unsigned int takeN,
						     const char *name) :
    PCAWhiteningANNComponent(U,S,epsilon,takeN,name) {
  }
  
  ZCAWhiteningANNComponent::~ZCAWhiteningANNComponent() {
  }
  
  Token *ZCAWhiteningANNComponent::doForward(Token* _input,
					     bool during_training) {
    PCAWhiteningANNComponent::doForward(_input, during_training);
    MatrixFloat *input_mat  = input->getMatrix();
    MatrixFloat *output_mat = output->getMatrix();
    MatrixFloat *zca_mat    = input_mat->cloneOnlyDims();
#ifdef USE_CUDA
    zca_mat->setUseCuda(use_cuda);
#endif
    zca_mat->gemm(CblasNoTrans, CblasTrans, 1.0f, output_mat, U, 0.0f); 
    AssignRef(output, new TokenMatrixFloat(zca_mat));
    return output;
  }
  
  ANNComponent *ZCAWhiteningANNComponent::clone() {
    ZCAWhiteningANNComponent *component = new ZCAWhiteningANNComponent(U, S,
								       epsilon,
								       0,
								       name.c_str());
    return component;
  }
  
  char *ZCAWhiteningANNComponent::toLuaString() {
    buffer_list buffer;
    char *U_str, *S_str;
    int len;
    U_str = writeMatrixFloatToString(U, false, len);
    S_str = writeMatrixFloatToString(S, false, len);
    buffer.printf("ann.components.zca_whitening{ name='%s', U=%s, S=%s, epsilon=%g, takeN=0, }",
		  name.c_str(), U_str, S_str, epsilon);
    delete[] U_str;
    delete[] S_str;
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
}
