/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
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
    units_order_permutation(0) {
  }

  ActivationFunctionANNComponent::~ActivationFunctionANNComponent() {
    if (input)        DecRef(input);
    if (error_input)  DecRef(error_input);
    if (output)       DecRef(output);
    if (error_output) DecRef(error_output);
    delete[] units_order_permutation;
  }

  Token *ActivationFunctionANNComponent::doForward(Token* _input,
						   bool during_training) {
    // error checking
    if ( (_input == 0) ||
	 (_input->getTokenCode() != table_of_token_codes::token_mem_block))
      ERROR_EXIT(129,"Incorrect input Token type, expected token_mem_block!\n");
    // change current input by new input
    AssignRef(input,_input->convertTo<TokenMemoryBlock*>());
    // compute bunch size
    bunch_size = input->getUsedSize() / input_size;
    // new  output to fit the bunch
    AssignRef(output,new TokenMemoryBlock(input->getUsedSize()));
    // get memory blocks for tokens
    FloatGPUMirroredMemoryBlock *input_ptr  = input->getMemBlock();
    FloatGPUMirroredMemoryBlock *output_ptr = output->getMemBlock();
    // execute apply activations abstract method
    applyActivation(input_ptr, output_ptr, input_size, bunch_size);
    // apply dropout
    if (dropout_factor > 0.0f) {
      if (during_training) {
	FloatGPUMirroredMemoryBlock *dropout_mask;
	dropout_mask    = new FloatGPUMirroredMemoryBlock(input->getUsedSize());
	float *mask_ptr = dropout_mask->getPPALForWrite();
	for (unsigned int i=0; i<dropout_mask->getSize(); ++i) {
	  if (dropout_random.rand() < dropout_factor) mask_ptr[i] = 0.0f;
	  else mask_ptr[i] = 1.0f;
	}
	/*
	  if (units_order_permutation == 0)
	  units_order_permutation = new int[input_size];
	  doVectorSetToZero(dropout_mask,
	  input->getUsedSize(), 1, 0,
	  false);
	  unsigned int length=static_cast<unsigned int>(dropout_factor*input_size);
	  for (unsigned int i=0; i<bunch_size; ++i) {
	  dropout_random.shuffle(input_size, units_order_permutation);
	  for (unsigned int j=0; j<length; ++j) {
	  unsigned int pos = units_order_permutation[j];
	  mask_ptr[i + pos*input_size] = 1.0f;
	  }
	  }
	*/
	// apply mask
	applyMask(output_ptr, dropout_mask, 0.0f, input_size,
		  bunch_size, use_cuda);
	delete dropout_mask;
      }
      else {
	float scal_factor = 1.0f - dropout_factor;
	doSscal(output_ptr->getSize(), scal_factor,
		output_ptr, 0, 1,
		use_cuda);
      }
    }
    return output;
  }
    
  Token *ActivationFunctionANNComponent::doBackprop(Token *_error_input) {
    // error checking
    if ( (_error_input == 0) ||
	 (_error_input->getTokenCode() != table_of_token_codes::token_mem_block))
      ERROR_EXIT(129,"Incorrect input error Token type, expected token_mem_block!\n");
    // change current input by new input
    AssignRef(error_input,_error_input->convertTo<TokenMemoryBlock*>());
    // compute current bunch
    unsigned int bunch_size = error_input->getUsedSize() / output_size;
    if (bunch_size != this->bunch_size)
      ERROR_EXIT(129, "Different bunches found at doForward and doBackprop\n");
    // new error output to fit the bunch
    AssignRef(error_output,new TokenMemoryBlock(error_input->getUsedSize()));
    //
    FloatGPUMirroredMemoryBlock *input_ptr        = input->getMemBlock();
    FloatGPUMirroredMemoryBlock *output_ptr       = output->getMemBlock();
    FloatGPUMirroredMemoryBlock *error_input_ptr  = error_input->getMemBlock();
    FloatGPUMirroredMemoryBlock *error_output_ptr = error_output->getMemBlock();
    // apply derivatives at gradients
    multiplyDerivatives(input_ptr, output_ptr,
			error_input_ptr, error_output_ptr,
			input_size, bunch_size);
    return error_output;
  }

  void ActivationFunctionANNComponent::reset() {
    if (input) DecRef(input);
    if (error_input) DecRef(error_input);
    if (output) DecRef(output);
    if (error_output) DecRef(error_output);
    input	 = 0;
    error_input	 = 0;
    output	 = 0;
    error_output = 0;
  }

  void ActivationFunctionANNComponent::setOption(const char *name, double value) {
    mSetOption(DROPOUT_FACTOR_STRING, dropout_factor);
    mSetOption(DROPOUT_SEED_STRING,   dropout_seed);
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
