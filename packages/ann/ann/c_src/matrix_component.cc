/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2014, Francisco Zamora-Martinez
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
#include "matrix_component.h"  
#include "unused_variable.h"
#include "token_sparse_matrix.h"
#include "wrapper.h"

using namespace basics;
using namespace april_utils;
using namespace april_math;

namespace ANN {

  VirtualMatrixANNComponent::
  VirtualMatrixANNComponent(const char *name, const char *weights_name,
                            unsigned int input_size, unsigned int output_size) :
    ANNComponent(name, weights_name, input_size, output_size),
    ComponentPropertiesAndAsserts(),
    input(0), output(0), error_input(0), error_output(0) {
  }

  VirtualMatrixANNComponent::~VirtualMatrixANNComponent() {
    if (input) DecRef(input);
    if (error_input) DecRef(error_input);
    if (output) DecRef(output);
    if (error_output) DecRef(error_output);
  }

  Token *VirtualMatrixANNComponent::doForward(Token* _input, bool during_training) {
    // error checking
    if (_input == 0)
      ERROR_EXIT1(129,"Incorrect input Token type, expected token_matrix! [%s]\n",
		  name.c_str());
    MatrixFloat *input_mat;
    switch( _input->getTokenCode() ) {
    case table_of_token_codes::token_matrix: {
      // change current input by new input
      AssignRef(input,_input->convertTo<TokenMatrixFloat*>());
      input_mat = input->getMatrix();
      break;
    }
    case table_of_token_codes::token_sparse_matrix: {
      // compute dense matrix from sparse
      TokenSparseMatrixFloat *sparse_token;
      sparse_token = _input->convertTo<TokenSparseMatrixFloat*>();
      input_mat = sparse_token->getMatrix()->toDense();
      AssignRef(input, new TokenMatrixFloat(input_mat));
#ifndef NDEBUG
      ERROR_PRINT1("Sparse to dense matrix transformation is expensive [%s]\n",
                   name.c_str());
#endif
      break;
    }
    default:
      input_mat = 0;
      ERROR_EXIT1(129,"Incorrect input Token type, expected token_matrix! [%s]\n",
		  name.c_str());
    }
    
#ifdef USE_CUDA
    input_mat->setUseCuda(use_cuda);
#endif
    ASSERT_MATRIX(input_mat);
    april_assert(input_size == 0 ||
                 input_mat->size()/input_mat->getDimSize(0) == static_cast<int>(input_size));
    if (getInputContiguousProperty() && !input_mat->getIsContiguous()) {
      input_mat = input_mat->clone();
      AssignRef(input,new TokenMatrixFloat(input_mat));
    }
    /////////////////////////
    // FORWARD COMPUTATION //
    MatrixFloat *output_mat = privateDoForward(input_mat, during_training);
    /////////////////////////
    april_assert(output_size == 0 ||
                 output_mat == 0 ||
                 output_mat->size()/output_mat->getDimSize(0) == static_cast<int>(output_size));
    AssignRef(output,new TokenMatrixFloat(output_mat));
    return output;
  }

  Token *VirtualMatrixANNComponent::doBackprop(Token *_error_input)
  {
    if ( (_error_input == 0) ||
	 (_error_input->getTokenCode() != table_of_token_codes::token_matrix))
      ERROR_EXIT1(129,"Incorrect input error Token type, expected token_matrix! [%s]\n",
		  name.c_str());
    // change current input by new input
    AssignRef(error_input,_error_input->convertTo<TokenMatrixFloat*>());
    MatrixFloat *error_input_mat  = error_input->getMatrix();
#ifdef USE_CUDA
    error_input_mat->setUseCuda(use_cuda);
#endif
    ASSERT_MATRIX(error_input_mat);
    april_assert(output_size == 0 ||
                 error_input_mat->size()/error_input_mat->getDimSize(0) == static_cast<int>(output_size));
    if (getInputContiguousProperty() && !error_input_mat->getIsContiguous()) {
      error_input_mat = error_input_mat->clone();
      AssignRef(error_input,new TokenMatrixFloat(error_input_mat));
    }
    if (! error_input_mat->sameDim(output->getMatrix()) ) {
      ERROR_EXIT1(129, "Different matrix sizes found at doForward and "
                  "doBackprop [%s]\n", name.c_str());
    }
    //////////////////////////
    // BACKPROP COMPUTATION //
    MatrixFloat *error_output_mat = privateDoBackprop(error_input_mat);
    //////////////////////////
    april_assert(input_size==0 || error_output_mat == 0 ||
                 error_output_mat->size()/error_output_mat->getDimSize(0) == static_cast<int>(input_size));
    if (error_input_mat != 0) {
      AssignRef(error_output, new TokenMatrixFloat(error_output_mat));
    }
    else {
      if (error_output) DecRef(error_output);
      error_output = 0;
    }
    return error_output;
  }

  void VirtualMatrixANNComponent::reset(unsigned int it) {
    if (input)        DecRef(input);
    if (error_input)  DecRef(error_input);
    if (output)       DecRef(output);
    if (error_output) DecRef(error_output);
    input        = 0;
    error_input  = 0;
    output       = 0;
    error_output = 0;
    // reset shared counter
    privateReset(it);
  }
}
