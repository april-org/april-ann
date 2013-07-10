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
#include "activation_function_component.h"
#include "wrapper.h"

namespace ANN {

  MTRand ActivationFunctionANNComponent::dropout_random = MTRand();
  int    ActivationFunctionANNComponent::dropout_seed   = -1;

  ActivationFunctionANNComponent::ActivationFunctionANNComponent(const char *name) :
    ANNComponent(name, 0, 0, 0),
    input(0),
    output(0),
    error_input(0),
    error_output(0),
    dropout_factor(0.0f),
    dropout_mask(0) {
  }

  ActivationFunctionANNComponent::~ActivationFunctionANNComponent() {
    if (input)        DecRef(input);
    if (error_input)  DecRef(error_input);
    if (output)       DecRef(output);
    if (error_output) DecRef(error_output);
    delete dropout_mask;
  }

  Token *ActivationFunctionANNComponent::doForward(Token* _input,
						   bool during_training) {
    // error checking
    if ( (_input == 0) ||
	 (_input->getTokenCode() != table_of_token_codes::token_matrix))
      ERROR_EXIT1(129,"Incorrect input Token type, expected token_matrix! [%s]\n",
		  name.c_str());
    // change current input by new input
    AssignRef(input,_input->convertTo<TokenMatrixFloat*>());
    MatrixFloat *input_mat = input->getMatrix();
    ASSERT_MATRIX(input_mat);
    if (!input_mat->getIsContiguous()) {
      input_mat = input_mat->clone();
      AssignRef(input,new TokenMatrixFloat(input_mat));
    }
#ifdef USE_CUDA
    input_mat->setUseCuda(use_cuda);
#endif
    unsigned int bunch_size = input_mat->getDimSize(0);
    // new  output to fit the bunch
    MatrixFloat *output_mat = input_mat->cloneOnlyDims();
    AssignRef(output,new TokenMatrixFloat(output_mat));
    // get memory blocks for tokens
    FloatGPUMirroredMemoryBlock *input_ptr  = input_mat->getRawDataAccess();
    FloatGPUMirroredMemoryBlock *output_ptr = output_mat->getRawDataAccess();
    // execute apply activations abstract method
    applyActivation(input_ptr, output_ptr, input_size, bunch_size);
    // apply dropout
    if (dropout_factor > 0.0f) {
      if (during_training) {
	delete dropout_mask;
	dropout_mask    = new FloatGPUMirroredMemoryBlock(input_mat->size());
	float *mask_ptr = dropout_mask->getPPALForWrite();
	for (unsigned int i=0; i<dropout_mask->getSize(); ++i) {
	  if (dropout_random.rand() < dropout_factor) mask_ptr[i] = 0.0f;
	  else mask_ptr[i] = 1.0f;
	}
	// apply mask
	applyMask(output_ptr, dropout_mask, 0.0f, input_size,
		  bunch_size, use_cuda);
      }
      else output_mat->scal(1.0f - dropout_factor);
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
    ASSERT_MATRIX(error_input_mat);
    if (!error_input_mat->getIsContiguous()) {
      error_input_mat = error_input_mat->clone();
      AssignRef(error_input,new TokenMatrixFloat(error_input_mat));
    }
#ifdef USE_CUDA
    input_mat->setUseCuda(use_cuda);
#endif
    unsigned int bunch_size = error_input_mat->getDimSize(0);
    // new  output to fit the bunch
    MatrixFloat *error_output_mat = error_input_mat->cloneOnlyDims();
    AssignRef(error_output,new TokenMatrixFloat(error_output_mat));
    if (!error_output_mat->sameDim(input->getMatrix()))
      ERROR_EXIT1(129, "Different bunches found at doForward and doBackprop [%s]\n",
		  name.c_str());
    //
    MatrixFloat *input_mat = input->getMatrix();
    MatrixFloat *output_mat = output->getMatrix();
    FloatGPUMirroredMemoryBlock *input_ptr        = input_mat->getRawDataAccess();
    FloatGPUMirroredMemoryBlock *output_ptr       = output_mat->getRawDataAccess();
    FloatGPUMirroredMemoryBlock *error_input_ptr  = error_input_mat->getRawDataAccess();
    FloatGPUMirroredMemoryBlock *error_output_ptr = error_output_mat->getRawDataAccess();
    // apply derivatives at gradients
    multiplyDerivatives(input_ptr, output_ptr,
			error_input_ptr, error_output_ptr,
			input_size, bunch_size);
    if (dropout_factor > 0.0f && dropout_mask != 0)
      // apply mask
      applyMask(error_output_ptr, dropout_mask, 0.0f, input_size,
		bunch_size, use_cuda);
    return error_output;
  }

  void ActivationFunctionANNComponent::reset() {
    if (input) DecRef(input);
    if (error_input) DecRef(error_input);
    if (output) DecRef(output);
    if (error_output) DecRef(error_output);
    delete dropout_mask;
    input	 = 0;
    error_input	 = 0;
    output	 = 0;
    error_output = 0;
    dropout_mask = 0;
  }

  void ActivationFunctionANNComponent::setOption(const char *name, double value) {
    mSetOption(DROPOUT_FACTOR_STRING, dropout_factor);
    if (strcmp(name, DROPOUT_SEED_STRING) == 0) {
      dropout_seed = static_cast<int>(value);
      dropout_random = MTRand(dropout_seed);
      return;
    }
    ANNComponent::setOption(name, value);
  }

  bool ActivationFunctionANNComponent::hasOption(const char *name) {
    mHasOption(DROPOUT_FACTOR_STRING);
    mHasOption(DROPOUT_SEED_STRING);
    return false;
  }
    
  double ActivationFunctionANNComponent::getOption(const char *name) {
    mGetOption(DROPOUT_FACTOR_STRING, dropout_factor);
    mGetOption(DROPOUT_SEED_STRING,   dropout_seed);
    return ANNComponent::getOption(name);
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
