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

// TODO: IMPLEMENT DROPOUT

ActivationFunctionANNComponent::ActivationFunctionANNComponent(const char *name) :
  ANNComponent(name, 0, 0, 0),
  output(new TokenMemoryBlock()), error_output(new TokenMemoryBlock()) { }

ActivationFunctionANNComponent::~ActivationFunctionANNComponent() { }

Token *ActivationFunctionANNComponent::doForward(Token* _input,
						 bool during_training) {
  // error checking
  if ( (_input == 0) ||
       (_input->getTokenCode() != table_of_token_codes::token_mem_block))
    ERROR_EXIT(129,"Incorrect input Token type, expected token_mem_block!\n");
  // change current input by new input
  if (input) DecRef(input);
  input = _input->convertTo<TokenMemoryBlock*>();
  IncRef(input);
  // compute bunch size
  bunch_size = input->getUsedSize() / input_size;
  // and resize the output to fit the bunch
  output->resize(bunch_size * output_size);
  // get memory blocks for tokens
  FloatGPUMirroredMemoryBlock *input_ptr  = input->getMemBlock();
  FloatGPUMirroredMemoryBlock *output_ptr = output->getMemBlock();
  // execute apply activations abstract method
  applyActivation(input_ptr, output_ptr, input_size, bunch_size);
  return output;
}
    
Token *ActivationFunctionANNComponent::doBackprop(Token *_error_input) {
  // error checking
  if ( (_error_input == 0) ||
       (_error_input->getTokenCode() != table_of_token_codes::token_mem_block))
    ERROR_EXIT(129,"Incorrect input error Token type, expected token_mem_block!\n");
  // change current input by new input
  if (error_input) DecRef(error_input);
  error_input = _error_input->convertTo<TokenMemoryBlock*>();
  IncRef(error_input);
  // compute current bunch
  unsigned int bunch_size = error_input->getUsedSize() / output_size;
  if (bunch_size != this->bunch_size)
    ERROR_EXIT(129, "Different bunches found at doForward and doBackprop\n");
  // and resize the output to fit the bunch
  error_output->resize(bunch_size * input_size);
  //
  FloatGPUMirroredMemoryBlock *input_ptr        = input->getMemBlock();
  FloatGPUMirroredMemoryBlock *output_ptr       = output->getMemBlock();
  FloatGPUMirroredMemoryBlock *error_input_ptr  = error_input->getMemBlock();
  FloatGPUMirroredMemoryBlock *error_output_ptr = error_input->getMemBlock();
  // apply derivatives at gradients
  multiplyDerivatives(input_ptr, output_ptr,
		      error_input_ptr, error_output_ptr,
		      input_size, bunch_size, false);
  return error_output;
}

void ActivationFunctionANNComponent::reset() {
  if (error_output != 0) doVectorSetToZero(error_output->getMemBlock(),
					   error_output->getMaxSize(),
					   0, 0, use_cuda);
  if (output != 0) doVectorSetToZero(output->getMemBlock(),
				     output->getMaxSize(),
				     0, 0, use_cuda);
  if (input) DecRef(input); input = 0;
  if (error_input) DecRef(error_input); error_input = 0;
}

void ActivationFunctionANNComponent::setOption(const char *name, double value) {
}

bool ActivationFunctionANNComponent::hasOption(const char *name) {
  return false;
}
    
double ActivationFunctionANNComponent::getOption(const char *name) {
  return ANNComponent::getOption(name);
}
    
void ActivationFunctionANNComponent::build(unsigned int _input_size,
					   unsigned int _output_size,
					   hash<string,Connections*> &weights_dict,
					   hash<string,ANNComponent*> &components_dict) {
  ANNComponent::build(_input_size, _output_size, weights_dict, components_dict);
  if (input_size != output_size)
    ERROR_EXIT(240, "ActivationFunctionANNComponent input/output "
	       "sizes must be equal\n");
}
