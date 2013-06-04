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
#include "copy_component.h"
#include "wrapper.h"

namespace ANN {
  
  CopyANNComponent::CopyANNComponent(unsigned int times, const char *name,
				     unsigned int input_size,
				     unsigned int output_size) :
    ANNComponent(name, 0, input_size, output_size),
    input(0),
    error_output(0),
    output(0),
    error_input(0),
    times(times) {
    if (times < 2)
      ERROR_EXIT(128, "CopyANNComponent for less than 2 copies is forbidden");
  }
  
  CopyANNComponent::~CopyANNComponent() {
    if (input) DecRef(input);
    if (error_input) DecRef(error_input);
    if (output) DecRef(output);
    if (error_output) DecRef(error_output);
  }
  
  Token *CopyANNComponent::doForward(Token* _input, bool during_training) {
    AssignRef(input, _input);
    AssignRef(output, new TokenBunchVector(times));
    for (unsigned int i=0; i<times; ++i) {
      (*output)[i] = input;
      IncRef(input);
    }
    return output;
  }

  Token *CopyANNComponent::doBackprop(Token *_error_input) {
    if (_error_input == 0) {
      if (error_input)  { DecRef(error_input);  error_input  = 0; }
      if (error_output) { DecRef(error_output); error_output = 0; }
      return 0;
    }
    if (_error_input->getTokenCode() != table_of_token_codes::vector_Tokens)
      ERROR_EXIT(128, "Incorrect error input token type, "
		 "expected TokenBunchVector\n");
    AssignRef(error_input, _error_input->convertTo<TokenBunchVector*>());
    if (error_input->size() != times)
      ERROR_EXIT2(128, "Incorrect error input size, found %d, expected %d\n",
		  error_input->size(), times);
    // The first is done out, scopy of input to output
    Token *current = (*error_input)[0];
    if (current->getTokenCode() != table_of_token_codes::token_mem_block)
      ERROR_EXIT(128, "Incorrect token type, expected token mem block\n");
    TokenMemoryBlock *current_mem_block;
    current_mem_block = current->convertTo<TokenMemoryBlock*>();
    unsigned int sz   = current_mem_block->getUsedSize();
    unsigned int bunch_size = sz / input_size;
    assert((bunch_size * input_size == sz) &&
	   "Incorrect input error token size, not divisible by bunch_size");
    TokenMemoryBlock *error_output_mem_block = new TokenMemoryBlock(sz);
    AssignRef(error_output, error_output_mem_block);
    doScopy(sz,
	    current_mem_block->getMemBlock(), 0, 1,
	    error_output_mem_block->getMemBlock(), 0, 1,
	    use_cuda);
    // The rest of tokens
    for (unsigned int i=1; i<times; ++i) {
      Token *current = (*error_input)[i];
      if (current->getTokenCode() != table_of_token_codes::token_mem_block)
	ERROR_EXIT(128, "Incorrect token type, expected token mem block\n");
      TokenMemoryBlock *current_mem_block;
      current_mem_block = current->convertTo<TokenMemoryBlock*>();
      if (current_mem_block->getUsedSize() != sz)
	ERROR_EXIT2(128, "Incorrect error input size, found %d, expected %d\n",
		    current_mem_block->getUsedSize(), sz);
      doSaxpy(sz,
	      1.0f, current_mem_block->getMemBlock(), 0, 1,
	      error_output_mem_block->getMemBlock(), 0, 1,
	      use_cuda);
    }
    return error_output;
  }
  
  void CopyANNComponent::reset() {
    if (input) DecRef(input);
    if (error_input) DecRef(error_input);
    if (output) DecRef(output);
    if (error_output) DecRef(error_output);
    input	 = 0;
    error_input	 = 0;
    output	 = 0;
    error_output = 0;
  }

  ANNComponent *CopyANNComponent::clone() {
    CopyANNComponent *copy_component = new CopyANNComponent(times,
							    name.c_str(),
							    input_size,
							    output_size);
    return copy_component;
  }

  void CopyANNComponent::build(unsigned int _input_size,
			       unsigned int _output_size,
			       hash<string,Connections*> &weights_dict,
			       hash<string,ANNComponent*> &components_dict) {
    ANNComponent::build(_input_size, _output_size,
			weights_dict, components_dict);
    if (output_size == 0) output_size = input_size * times;
    if (input_size  == 0) input_size  = output_size / times;
    if (input_size * times != output_size)
      ERROR_EXIT2(128, "Incorrect input/output sizes: input=%d output=%d\n",
		  input_size, output_size);
  }
  
  char *CopyANNComponent::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.components.copy{ name='%s',times=%d,input=%d,output=%d }",
		  name.c_str(), times, input_size, output_size);
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
}
