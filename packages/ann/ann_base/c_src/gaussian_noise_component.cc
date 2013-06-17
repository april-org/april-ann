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
#include "error_print.h"
#include "table_of_token_codes.h"
#include "token_vector.h"
#include "token_memory_block.h"
#include "gaussian_noise_component.h"
#include "wrapper.h"

namespace ANN {
  
  GaussianNoiseANNComponent::GaussianNoiseANNComponent(MTRand *random,
						       float mean,
						       float variance,
						       const char *name,
						       unsigned int size) :
    ANNComponent(name, 0, size, size),
    random(random),
    input(0),
    output(0),
    error_input(0),
    error_output(0),
    mean(mean),
    variance(variance) {
    IncRef(random);
  }
  
  GaussianNoiseANNComponent::~GaussianNoiseANNComponent() {
    if (input) DecRef(input);
    if (error_input) DecRef(error_input);
    if (output) DecRef(output);
    if (error_output) DecRef(error_output);
    DecRef(random);
  }
  
  Token *GaussianNoiseANNComponent::doForward(Token* _input, bool during_training) {
    // error checking
    if ( (_input == 0) ||
	 (_input->getTokenCode() != table_of_token_codes::token_mem_block))
      ERROR_EXIT(129,"Incorrect input Token type, expected token_mem_block!\n");
    // change current input by new input
    AssignRef(input,_input->convertTo<TokenMemoryBlock*>());
    // new  output to fit the bunch
    AssignRef(output,new TokenMemoryBlock(input->getUsedSize()));
    // get memory blocks for tokens
    FloatGPUMirroredMemoryBlock *input_ptr  = input->getMemBlock();
    FloatGPUMirroredMemoryBlock *output_ptr = output->getMemBlock();
    FloatGPUMirroredMemoryBlock *noise_ptr;
    noise_ptr = new FloatGPUMirroredMemoryBlock(input->getUsedSize());
    float *noise_mem_ptr = noise_ptr->getPPALForWrite();
    for (unsigned int i=0; i<input->getUsedSize(); ++i)
      noise_mem_ptr[i] = random->randNorm(mean, variance);
    doScopy(input->getUsedSize(),
	    input_ptr, 0, 1,
	    output_ptr, 0, 1,
	    use_cuda);
    doSaxpy(input->getUsedSize(),
	    1.0f, noise_ptr, 0, 1,
	    output_ptr, 0, 1,
	    use_cuda);
    delete noise_ptr;
    return output;
  }

  Token *GaussianNoiseANNComponent::doBackprop(Token *_error_input) {
    AssignRef(error_input,  _error_input);
    AssignRef(error_output, _error_input);
    return error_output;
  }
  
  void GaussianNoiseANNComponent::reset() {
    if (input) DecRef(input);
    if (error_input) DecRef(error_input);
    if (output) DecRef(output);
    if (error_output) DecRef(error_output);
    input	 = 0;
    error_input	 = 0;
    output	 = 0;
    error_output = 0;
  }

  ANNComponent *GaussianNoiseANNComponent::clone() {
    GaussianNoiseANNComponent *copy_component = new
      GaussianNoiseANNComponent(new MTRand(*random),
				mean, variance,
				name.c_str(),
				input_size);
    return copy_component;
  }

  void GaussianNoiseANNComponent::build(unsigned int _input_size,
					unsigned int _output_size,
					hash<string,Connections*> &weights_dict,
					hash<string,ANNComponent*> &components_dict) {
    ANNComponent::build(_input_size, _output_size,
			weights_dict, components_dict);
    if (output_size == 0) output_size = input_size;
    if (input_size  == 0) input_size  = output_size;
    if (input_size != output_size)
      ERROR_EXIT2(128, "Incorrect input/output sizes: input=%d output=%d\n",
		  input_size, output_size);
  }
  
  char *GaussianNoiseANNComponent::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.components.gaussian_noise{ name='%s',size=%d }",
		  name.c_str(), input_size);
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
}
