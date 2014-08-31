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
#include "sparse_matrix_component.h"  
#include "unused_variable.h"
#include "token_sparse_matrix.h"
#include "wrapper.h"

using namespace Basics;
using namespace AprilUtils;
using namespace AprilMath;

namespace ANN {

  VirtualSparseMatrixANNComponent::
  VirtualSparseMatrixANNComponent(const char *name, const char *weights_name,
                                  unsigned int input_size,
                                  unsigned int output_size) :
    ANNComponent(name, weights_name, input_size, output_size),
    ComponentPropertiesAndAsserts(),
    input(0), error_output(0), output(0), error_input(0) {
  }

  VirtualSparseMatrixANNComponent::~VirtualSparseMatrixANNComponent() {
    if (input) DecRef(input);
    if (error_input) DecRef(error_input);
    if (output) DecRef(output);
    if (error_output) DecRef(error_output);
  }

  Token *VirtualSparseMatrixANNComponent::doForward(Token* _input, bool during_training) {
    // error checking
    if (_input == 0)
      ERROR_EXIT1(129,"Incorrect input Token type, expected token_sparse_matrix! [%s]\n",
		  name.c_str());
    SparseMatrixFloat *input_mat;
    switch( _input->getTokenCode() ) {
    case table_of_token_codes::token_sparse_matrix: {
      // change current input by new input
      AssignRef(input,_input->convertTo<TokenSparseMatrixFloat*>());
      input_mat = input->getMatrix();      
    }
    default:
      input_mat = 0;
      ERROR_EXIT1(129,"Incorrect input Token type, expected token_matrix! [%s]\n",
		  name.c_str());
    }
    
#ifdef USE_CUDA
    input_mat->setUseCuda(use_cuda);
#endif
    if (input_mat->getSparseFormat() != CSR_FORMAT) {
      ERROR_EXIT(128, "Sparse matrix must be in csr format\n");
    }
    if (input_size > 0) {
      april_assert(input_mat->getDimSize(1) == static_cast<int>(input_size));
    }
    /////////////////////////
    // FORWARD COMPUTATION //
    MatrixFloat *output_mat = privateDoForward(input_mat, during_training);
    /////////////////////////
    AssignRef(output,new TokenMatrixFloat(output_mat));
    return output;
  }

  Token *VirtualSparseMatrixANNComponent::doBackprop(Token *_error_input)
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
    if (! error_input_mat->sameDim(output->getMatrix()) )
      ERROR_EXIT1(129, "Different matrix sizes found at doForward and "
                  "doBackprop [%s]\n", name.c_str());
    //////////////////////////
    // BACKPROP COMPUTATION //
    SparseMatrixFloat *error_output_mat = privateDoBackprop(error_input_mat);
    //////////////////////////
    if (error_output_mat != 0)
      AssignRef(error_output, new TokenSparseMatrixFloat(error_output_mat));
    else if (error_output) {
      if (error_output) DecRef(error_output);
      error_output = 0;
    }
    return error_output;
  }

  void VirtualSparseMatrixANNComponent::reset(unsigned int it) {
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
