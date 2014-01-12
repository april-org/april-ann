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
#include "pca_whitening_component.h"
#include "wrapper.h"
#include "utilMatrixFloat.h"

namespace ANN {
  
  PCAWhiteningANNComponent::PCAWhiteningANNComponent(MatrixFloat *U,
						     MatrixFloat *S,
						     float epsilon,
						     unsigned int takeN,
						     const char *name) :
    ANNComponent(name, 0,
		 static_cast<unsigned int>(S->size()),
		 (takeN==0)?(static_cast<unsigned int>(S->size())):(takeN)),
    U(U), S(S), epsilon(epsilon),
    input(0),
    output(0) {
    if (U->getMajorOrder() != CblasColMajor)
      ERROR_EXIT(128, "Incorrect U matrix major order, needed col_major\n");
    if (S->getMajorOrder() != CblasColMajor)
      ERROR_EXIT(128, "Incorrect S matrix major order, needed col_major\n");
    if (U->getNumDim() != 2)
      ERROR_EXIT(128, "Needs a bi-dimensional matrix as U argument\n");
    if (S->getNumDim() != 1)
      ERROR_EXIT(128, "Needs a one-dimensional matrix as S argument\n");
    if (static_cast<int>(takeN) > S->size())
      ERROR_EXIT(128, "Taking more components than size of S matrix\n");
    if (takeN != 0) {
      int coords[2] = { 0,0 };
      int sizes[2] = { U->getDimSize(0), static_cast<int>(takeN) };
      this->U = new MatrixFloat(this->U, coords, sizes, true);
      this->S = new MatrixFloat(this->S, coords+1, sizes+1, true);
    }
    IncRef(this->U);
    IncRef(this->S);
  }
  
  PCAWhiteningANNComponent::~PCAWhiteningANNComponent() {
    if (input) DecRef(input);
    if (output) DecRef(output);
    DecRef(U);
    DecRef(S);
  }
  
  Token *PCAWhiteningANNComponent::doForward(Token* _input, bool during_training) {
    UNUSED_VARIABLE(during_training);
    if (_input->getTokenCode() != table_of_token_codes::token_matrix)
      ERROR_EXIT1(128, "Incorrect token found, only TokenMatrixFloat is "
		  "allowed [%s]\n", name.c_str());
    AssignRef(input, _input->convertTo<TokenMatrixFloat*>());    
    MatrixFloat *input_mat = input->getMatrix();
#ifdef USE_CUDA
    input_mat->setUseCuda(use_cuda);
#endif
    if (input_mat->getNumDim() != 2)
      ERROR_EXIT2(128, "A 2-dimensional matrix is expected, found %d. "
		  "[%s]", input_mat->getNumDim(), name.c_str());
    int dims[2] = { input_mat->getDimSize(0), S->size() };
    MatrixFloat *output_mat = new MatrixFloat(2, dims, CblasColMajor);
    output_mat->gemm(CblasNoTrans, CblasNoTrans, 1.0f, input_mat, U, 0.0f);
    // regularization
    if (epsilon > 0.0f) {
      MatrixFloat *aux_mat = 0;
      MatrixFloat::const_iterator Sit(S->begin());
      for (int i=0; i<S->size(); ++i, ++Sit) {
	aux_mat = output_mat->select(1, i, aux_mat);
	aux_mat->scal( 1/sqrtf( (*Sit) + epsilon ) );
      }
    }
    AssignRef(output, new TokenMatrixFloat(output_mat));
    return output;
  }

  Token *PCAWhiteningANNComponent::doBackprop(Token *_error_input) {
    UNUSED_VARIABLE(_error_input);
    return 0;
  }
  
  void PCAWhiteningANNComponent::reset(unsigned int it) {
    UNUSED_VARIABLE(it);
    if (input) DecRef(input);
    if (output) DecRef(output);
    input	 = 0;
    output	 = 0;
  }
  
  ANNComponent *PCAWhiteningANNComponent::clone() {
    PCAWhiteningANNComponent *component = new PCAWhiteningANNComponent(U, S,
								       epsilon,
								       0,
								       name.c_str());
    return component;
  }
  
  void PCAWhiteningANNComponent::build(unsigned int _input_size,
				       unsigned int _output_size,
				       MatrixFloatSet *weights_dict,
				       hash<string,ANNComponent*> &components_dict) {
    ANNComponent::build(_input_size, _output_size, weights_dict, components_dict);
    // TODO: Check that output_size == S->size()
  }
  
  char *PCAWhiteningANNComponent::toLuaString() {
    buffer_list buffer;
    char *U_str, *S_str;
    int len;
    U_str = writeMatrixFloatToString(U, false, len);
    S_str = writeMatrixFloatToString(S, false, len);
    buffer.printf("ann.components.pca_whitening{ name='%s', U=%s, S=%s, epsilon=%g, takeN=0, }",
		  name.c_str(), U_str, S_str, epsilon);
    delete[] U_str;
    delete[] S_str;
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
}
