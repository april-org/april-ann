/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
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
#include "table_of_token_codes.h"
#include "error_print.h"
#include "join_component.h"

namespace ANN {
  
  JoinANNComponent::JoinANNComponent(const char *name) :
    ANNComponent(name),
    output(new TokenMemoryBlock()),
    error_output(new TokenBunchVector()) {
    IncRef(output);
    IncRef(error_output);
  }
  
  JoinANNComponent::~JoinANNComponent() {
    for (unsigned int i=0; i<components.size(); ++i)
      DecRef(components[i]);
    if (input) DecRef(input);
    if (error_input) DecRef(error_input);
    DecRef(output);
    DecRef(error_output);
  }

  void JoinANNComponent::addComponent(ANNComponent *component) {
    components.push_back(component);
    IncRef(component);
  }
    
  Token *JoinANNComponent::doForward(Token* _input, bool during_training) {
    if (_input->getTokenCode() != table_of_token_codes::vector_Tokens)
      ERROR_EXIT(128, "Incorrect Token type, expected TokenBunchVector\n");
    if (input) DecRef(input);
    input = _input->convertTo<TokenBunchVector*>();
    IncRef(input);
    if (input->size() == 0)
      ERROR_EXIT(128, "Found empty TokenBunchVector\n");
    Token **input_data = input->data();
    unsigned int total_input_size = 0;
    for (unsigned int i=0; i<input->size(); ++i) {
      if (input_data[i]->getTokenCode() != table_of_token_codes::token_memory_block)
	ERROR_EXIT(128, "Incorrect Token type, expected TokenMemoryBlock\n");
      TokenMemoryBlock *current = input_data[i]->convertTo<TokenMemoryBlock*>();
      total_input_size += current->getUsedSize();
    }
    output->resize(total_input_size);	
    unsigned int pos = 0;
    for (unsigned int i=0; i<input->size(); ++i) {
      TokenMemoryBlock *current = input_data[i]->convertTo<TokenMemoryBlock*>();
      doScopy(current->getUsedSize(),
	      current->getMemBlock(), 0, 1,
	      output->getMemBlock(), pos, 1,
	      use_cuda);
      pos += current->getUsedSize();
    }
  }

    Token *JoinANNComponent::doBackprop(Token *input_error);
    
    void JoinANNComponent::reset();
    
    ANNComponent *JoinANNComponent::clone();

  void JoinANNComponent::build(unsigned int _input_size,
			       unsigned int _output_size,
			       hash<string,Connections*> &weights_dict,
			       hash<string,ANNComponent*> &components_dict) {
  }
};
}
