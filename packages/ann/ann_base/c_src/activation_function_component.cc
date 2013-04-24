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

// TODO: IMPLEMENT DROPOUT

namespace ANN {

  ActivationFunctionANNComponent::ActivationFunctionANNComponent(const char *name) :
    ANNComponent(name, 0, 0, 0),
    input(0),
    output(0)
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
    // new  output to fit the bunch
    if (output) DecRef(output);
    output = new TokenMemoryBlock(input->getUsedSize());
    IncRef(output);
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
    // new error output to fit the bunch
    if (error_output) DecRef(error_output);
    error_output = new TokenMemoryBlock(error_input->getUsedSize());
    IncRef(error_output);
    //
    FloatGPUMirroredMemoryBlock *input_ptr        = input->getMemBlock();
    FloatGPUMirroredMemoryBlock *output_ptr       = output->getMemBlock();
    FloatGPUMirroredMemoryBlock *error_input_ptr  = error_input->getMemBlock();
    FloatGPUMirroredMemoryBlock *error_output_ptr = error_output->getMemBlock();
    // apply derivatives at gradients
    multiplyDerivatives(input_ptr, output_ptr,
			error_input_ptr, error_output_ptr,
			input_size, bunch_size, false);
    return error_output;
  }

  void ActivationFunctionANNComponent::reset() {
    if (error_output != 0 && error_output->getMaxSize() > 0)
      doVectorSetToZero(error_output->getMemBlock(),
			error_output->getMaxSize(),
			1, 0, use_cuda);
    if (output != 0 && output->getMaxSize() > 0)
      doVectorSetToZero(output->getMemBlock(),
			output->getMaxSize(),
			1, 0, use_cuda);
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
    if (input_size == 0) input_size = output_size;
    if (output_size == 0) output_size = input_size;
    if (input_size != output_size)
      ERROR_EXIT(240, "ActivationFunctionANNComponent input/output "
		 "sizes must be equal\n");
  }

} // namespace ANN
