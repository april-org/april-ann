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
#include "error_print.h"
#include "matrix_input_switch_component.h"  
#include "unused_variable.h"

using namespace Basics;
using namespace AprilUtils;
using namespace AprilMath;

namespace ANN {

  MatrixInputSwitchANNComponent::
  MatrixInputSwitchANNComponent(const char *name, const char *weights_name,
                                unsigned int input_size,
                                unsigned int output_size) :
    ANNComponent(name, weights_name, input_size, output_size),
    ComponentPropertiesAndAsserts(),
    input(0), output(0), error_input(0), error_output(0),
    sparse_input(0), sparse_error_output(0),
    is_sparse_input(false)
  {
  }

  MatrixInputSwitchANNComponent::~MatrixInputSwitchANNComponent() {
    if (input) DecRef(input);
    if (output) DecRef(output);
    if (error_input) DecRef(error_input);
    if (error_output) DecRef(error_output);
    if (sparse_input) DecRef(sparse_input);
    if (sparse_error_output) DecRef(sparse_error_output);
  }

  Token *MatrixInputSwitchANNComponent::doDenseForward(Token *_input,
                                                       bool during_training) {
    // error checking
    if (_input == 0)
      ERROR_EXIT1(129,"Incorrect input Token type, expected token_matrix! [%s]\n",
		  name.c_str());
    MatrixFloat *input_mat;
    if (_input->getTokenCode() != table_of_token_codes::token_matrix)
      ERROR_EXIT1(129,"Incorrect input Token type, expected token_matrix! [%s]\n",
		  name.c_str());      
    // change current input by new input
    AssignRef(input,_input->convertTo<TokenMatrixFloat*>());
    input_mat = input->getMatrix();
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
    MatrixFloat *output_mat = privateDoDenseForward(input_mat, during_training);
    /////////////////////////
    april_assert(output_size == 0 ||
                 output_mat == 0 ||
                 output_mat->size()/output_mat->getDimSize(0) == static_cast<int>(output_size));
    AssignRef(output,new TokenMatrixFloat(output_mat));
    return output;
  }

  Token *MatrixInputSwitchANNComponent::doSparseForward(Token *_input,
                                                        bool during_training) {
    // error checking
    if (_input == 0)
      ERROR_EXIT1(129,"Incorrect input Token type, expected token_sparse_matrix! [%s]\n",
		  name.c_str());
    if (_input->getTokenCode() != table_of_token_codes::token_sparse_matrix) {
      ERROR_EXIT1(129,"Incorrect input Token type, expected token_matrix! [%s]\n",
		  name.c_str());
    }
    // change current input by new input
    AssignRef(sparse_input, _input->convertTo<TokenSparseMatrixFloat*>());
    SparseMatrixFloat *input_mat = sparse_input->getMatrix();
#ifdef USE_CUDA
    input_mat->setUseCuda(use_cuda);
#endif
    if (input_mat->getSparseFormat() != CSR_FORMAT) {
      ERROR_EXIT(128, "Sparse matrix must be in csr format\n");
    }
    april_assert(input_size == 0 ||
                 input_mat->size()/input_mat->getDimSize(0) == static_cast<int>(input_size));
    /////////////////////////
    // FORWARD COMPUTATION //
    MatrixFloat *output_mat = privateDoSparseForward(input_mat,during_training);
    /////////////////////////
    april_assert(output_size == 0 ||
                 output_mat == 0 ||
                 output_mat->size()/output_mat->getDimSize(0) == static_cast<int>(output_size));
    AssignRef(output,new TokenMatrixFloat(output_mat));
    return output;
  }

  Token *MatrixInputSwitchANNComponent::doDenseBackprop(Token *_error_input) {
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
    MatrixFloat *error_output_mat = privateDoDenseBackprop(error_input_mat);
    //////////////////////////
    april_assert(input_size==0 || error_output_mat == 0 ||
                 error_output_mat->size()/error_output_mat->getDimSize(0) == static_cast<int>(input_size));
    if (error_output_mat != 0) {
      AssignRef(error_output, new TokenMatrixFloat(error_output_mat));
    }
    else {
      if (error_output) DecRef(error_output);
      error_output = 0;
    }
    return error_output;
  }
  
  Token *MatrixInputSwitchANNComponent::doSparseBackprop(Token *_error_input) {
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
      ERROR_EXIT1(129, "Different matrix sizes found at doForward "
                  "and doBackprop [%s]\n", name.c_str());
    //////////////////////////
    // BACKPROP COMPUTATION //
    SparseMatrixFloat *error_output_mat;
    error_output_mat = privateDoSparseBackprop(error_input_mat);
    //////////////////////////
    april_assert(input_size==0 || error_output_mat == 0 ||
                 error_output_mat->size()/error_output_mat->getDimSize(0) == static_cast<int>(input_size));
    if (error_output_mat != 0)
      AssignRef(sparse_error_output, new TokenSparseMatrixFloat(error_output_mat));
    else if (sparse_error_output) {
      if (sparse_error_output) DecRef(sparse_error_output);
      sparse_error_output = 0;
    }
    return sparse_error_output;
  }
  
  /////////////////////////////////////////////////////////////////////////////
  
  Token *MatrixInputSwitchANNComponent::doForward(Token* _input,
                                                  bool during_training) {
    Token *output;
    // error checking
    if (_input == 0)
      ERROR_EXIT1(129,"Incorrect input Token type, expected token_matrix or token_sparse_matrix! [%s]\n",
		  getName().c_str());
    switch( _input->getTokenCode() ) {
    case table_of_token_codes::token_matrix: {
      is_sparse_input = false;
      output = doDenseForward(_input, during_training);
      break;
    }
    case table_of_token_codes::token_sparse_matrix: {
      is_sparse_input = true;
      output = doSparseForward(_input, during_training);
      break;
    }
    default:
      output = 0;
      ERROR_EXIT1(129,"Incorrect input Token type, expected token_matrix or token_sparse_matrix! [%s]\n",
                  getName().c_str());
    }
    return output;
  }

  Token *MatrixInputSwitchANNComponent::doBackprop(Token *_error_input)
  {
    Token *error_output;
    if ( (_error_input == 0) ||
	 (_error_input->getTokenCode() != table_of_token_codes::token_matrix))
      ERROR_EXIT1(129,"Incorrect input error Token type, expected token_matrix! [%s]\n",
                  getName().c_str());
    if (!is_sparse_input) {
      error_output = doDenseBackprop(_error_input);
    }
    else {
      error_output = doSparseBackprop(_error_input);
    }
    return error_output;
  }

  void MatrixInputSwitchANNComponent::reset(unsigned int it) {
    if (input) DecRef(input);
    if (output) DecRef(output);
    if (error_input) DecRef(error_input);
    if (error_output) DecRef(error_output);
    if (sparse_input) DecRef(sparse_input);
    if (sparse_error_output) DecRef(sparse_error_output);
    input = 0;
    output = 0;
    error_input = 0;
    error_output = 0;
    sparse_input = 0;
    sparse_error_output = 0;
    if (!is_sparse_input) privateDenseReset(it);
    else privateSparseReset(it);
  }

  void MatrixInputSwitchANNComponent::computeGradients(AprilUtils::SharedPtr<MatrixFloat> & grads_mat) {
    if (!is_sparse_input) privateDenseComputeGradients(grads_mat);
    else privateSparseComputeGradients(grads_mat);
  }
}
