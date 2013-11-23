/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera, Adrian Palacios Corella, Francisco
 * Zamora-Martinez
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
#include "activation_function_component.h"
#include "wrapper.h"

namespace ANN {
  
  ActivationFunctionANNComponent::ActivationFunctionANNComponent(const char *name) :
    ANNComponent(name, 0, 0, 0),
    input(0),
    output(0),
    error_input(0),
    error_output(0) {
  }

  ActivationFunctionANNComponent::~ActivationFunctionANNComponent() {
    if (input)        DecRef(input);
    if (error_input)  DecRef(error_input);
    if (output)       DecRef(output);
    if (error_output) DecRef(error_output);
  }

  Token *ActivationFunctionANNComponent::doForward(Token* _input,
						   bool during_training) {
    UNUSED_VARIABLE(during_training);
    // error checking
    if ( (_input == 0) ||
	 (_input->getTokenCode() != table_of_token_codes::token_matrix))
      ERROR_EXIT1(129,"Incorrect input Token type, expected token_matrix! [%s]\n",
		  name.c_str());
    // change current input by new input
    AssignRef(input,_input->convertTo<TokenMatrixFloat*>());
    MatrixFloat *input_mat = input->getMatrix();
    april_assert(input_mat->getMajorOrder() == CblasColMajor);
    april_assert(input_mat->getNumDim() >= 2);
    if (!input_mat->getIsContiguous()) {
      input_mat = input_mat->clone();
      AssignRef(input,new TokenMatrixFloat(input_mat));
    }
#ifdef USE_CUDA
    input_mat->setUseCuda(use_cuda);
#endif
    unsigned int bunch_size = input_mat->getDimSize(0);
    unsigned int current_input_size = input_mat->size() / bunch_size;
    // new  output to fit the bunch
    MatrixFloat *output_mat = input_mat->cloneOnlyDims();
    AssignRef(output,new TokenMatrixFloat(output_mat));
    // flatten the matrices if it is necessary
    bool flattened = false;
    if (input_mat->getNumDim() > 2) {
      int dims[2] = { static_cast<int>(bunch_size),
		      static_cast<int>(current_input_size) };
      input_mat  = input_mat->rewrap(dims, 2);
      output_mat = output_mat->rewrap(dims, 2);
      flattened  = true;
    }
    // get memory blocks for tokens
    FloatGPUMirroredMemoryBlock *input_ptr  = input_mat->getRawDataAccess();
    FloatGPUMirroredMemoryBlock *output_ptr = output_mat->getRawDataAccess();
    // execute apply activations abstract method
    applyActivation(input_ptr, output_ptr, current_input_size, bunch_size);
    if (flattened) {
      delete input_mat;
      delete output_mat;
    }
    return output;
  }
    
  Token *ActivationFunctionANNComponent::doBackprop(Token *_error_input) {
    // error checking
    if ( (_error_input == 0) ||
	 (_error_input->getTokenCode() != table_of_token_codes::token_matrix))
      ERROR_EXIT1(129,"Incorrect input error Token type, expected token_matrix! [%s]\n",
		  name.c_str());
    // change current input by new input
    AssignRef(error_input,_error_input->convertTo<TokenMatrixFloat*>());
    MatrixFloat *error_input_mat = error_input->getMatrix();
    april_assert(error_input_mat->getMajorOrder() == CblasColMajor);
    april_assert(error_input_mat->getNumDim() >= 2);
    if (!error_input_mat->getIsContiguous()) {
      error_input_mat = error_input_mat->clone();
      AssignRef(error_input,new TokenMatrixFloat(error_input_mat));
    }
#ifdef USE_CUDA
    error_input_mat->setUseCuda(use_cuda);
#endif
    unsigned int bunch_size = error_input_mat->getDimSize(0);
    unsigned int current_input_size = error_input_mat->size() / bunch_size;
    // new  output to fit the bunch
    MatrixFloat *error_output_mat = error_input_mat->cloneOnlyDims();
    AssignRef(error_output,new TokenMatrixFloat(error_output_mat));
    if (!error_output_mat->sameDim(input->getMatrix()))
      ERROR_EXIT1(129, "Different bunches found at doForward and doBackprop [%s]\n",
		  name.c_str());
    // flatten the matrices if it is necessary
    bool flattened = false;
    if (error_input_mat->getNumDim() > 2) {
      int dims[2] = { static_cast<int>(bunch_size),
		      static_cast<int>(current_input_size) };
      error_input_mat  = error_input_mat->rewrap(dims, 2);
      error_output_mat = error_output_mat->rewrap(dims, 2);
      flattened  = true;
    }
    MatrixFloat *input_mat = input->getMatrix();
    MatrixFloat *output_mat = output->getMatrix();
    FloatGPUMirroredMemoryBlock *input_ptr        = input_mat->getRawDataAccess();
    FloatGPUMirroredMemoryBlock *output_ptr       = output_mat->getRawDataAccess();
    FloatGPUMirroredMemoryBlock *error_input_ptr  = error_input_mat->getRawDataAccess();
    FloatGPUMirroredMemoryBlock *error_output_ptr = error_output_mat->getRawDataAccess();
    // apply derivatives at gradients
    multiplyDerivatives(input_ptr, output_ptr,
			error_input_ptr, error_output_ptr,
			current_input_size, bunch_size);
    if (flattened) {
      delete error_input_mat;
      delete error_output_mat;
    }
    return error_output;
  }

  void ActivationFunctionANNComponent::reset(unsigned int it) {
    UNUSED_VARIABLE(it);
    if (input) DecRef(input);
    if (error_input) DecRef(error_input);
    if (output) DecRef(output);
    if (error_output) DecRef(error_output);
    input	 = 0;
    error_input	 = 0;
    output	 = 0;
    error_output = 0;
  }
  
  void ActivationFunctionANNComponent::build(unsigned int _input_size,
					     unsigned int _output_size,
					     hash<string,Connections*> &weights_dict,
					     hash<string,ANNComponent*> &components_dict) {
    ANNComponent::build(_input_size, _output_size, weights_dict, components_dict);
    if (input_size == 0) input_size = output_size;
    if (output_size == 0) output_size = input_size;
    if (input_size != output_size)
      ERROR_EXIT3(240, "ActivationFunctionANNComponent input/output "
		  "sizes must be equal (component %s): %d != %d\n",
		  name.c_str(), input_size, output_size);
  }
  
} // namespace ANN
