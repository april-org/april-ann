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
    input(0), error_input(0), error_output(0),
    output(new TokenMemoryBlock()),
    input_vector(new TokenBunchVector()),
    error_input_vector(new TokenBunchVector()),
    output_vector(new TokenBunchVector()),
    error_output_vector(new TokenBunchVector()) {
    IncRef(output);
    IncRef(input_vector);
    IncRef(error_input_vector);
    IncRef(output_vector);
    IncRef(error_output_vector);
  }
  
  JoinANNComponent::~JoinANNComponent() {
    for (unsigned int i=0; i<components.size(); ++i)
      DecRef(components[i]);
    if (input) DecRef(input);
    if (error_input) DecRef(error_input);
    if (error_output) DecRef(error_output);
    DecRef(output);
    DecRef(input_vector);
    DecRef(error_input_vector);
    DecRef(output_vector);
    DecRef(error_output_vector);
  }

  void JoinANNComponent::addComponent(ANNComponent *component) {
    components.push_back(component);
    IncRef(component);
  }

  /**
     Two possible inputs:

       - A TokenMemoryBlock which will be segmented into pieces to feed each
         joined component

       - A TokenBunchVector of Tokens, where each position of the
         vector is the input of each component
   */  
  void JoinANNComponent::buildInputBunchVector(TokenBunchVector *&vector_token,
					       Token *token) {
    switch(token->getTokenCode()) {
    case table_of_token_codes::TokenMemoryBlock: {
      TokenMemoryBlock *mem_token = token->convertTo<TokenMemoryBlock*>();
      unsigned int bunch_size = mem_token->getUsedSize() / input_size;
      unsigned int pos=0;
      for (unsigned int i=0; i<vector_token.size(); ++i) {
	// force input token to be a TokenMemoryBlock
	if (vector_token[i]->getTokenCode() !=
	    table_of_token_codes::token_memory_block) {
	  DecRef(vector_token[i]);
	  vector_token[i] = new TokenMemoryBlock();
	  IncRef(vector_token[i]);
	}
	TokenMemoryBlock *component_mem_token;
	component_mem_token = vector_token[i]->converTo<TokenMemoryBlock*>();
	unsigned int sz = bunch_size * component[i]->getInputSize();
	component_mem_token->resize(sz);
	// copy from _input to component_mem_token
	doScopy(sz,
		mem_token->getMemBlock(), pos, 1,
		component_mem_token->getMemBlock(), 0, 1,
		use_cuda);
	pos += sz;
      }
      break;
    }
    case table_of_token_codes::TokenBunchVector: {
      TokenBunchVector *vtoken = token->convertTo<TokenBunchVector*>();
      if (token_vector.size() != vtoken.size())
	ERROR_EXIT(128, "Incorrect number of components at input vector, "
		   "expected %u and found %u\n",
		   token_vector.size(), vtoken.size());
      for (unsigned int i=0; i<token_vector.size(); ++i) {
	if (token_vector[i]) DecRef(token_vector[i]);
	token_vector[i] = vtoken[i];
	IncRef(token_vector[i]);
      }
      break;
    }
    default:
      ERROR_EXIT(129, "Incorrect token type");
    }
  }
  
  void JoinANNComponent::buildErrorInputBunchVector(TokenBunchVector *&vector_token,
						    Token *token) {
    if (token->getTokenCode() != table_of_token_codes::token_memory_block)
      ERROR_EXIT(128, "Incorrect token type\n");
    //
    TokenMemoryBlock *mem_token = token->convertTo<TokenMemoryBlock*>();
    unsigned int bunch_size = mem_token->getUsedSize() / output_size;
    unsigned int pos=0;
    for (unsigned int i=0; i<vector_token.size(); ++i) {
      // force input token to be a TokenMemoryBlock
      if (vector_token[i]->getTokenCode() !=
	  table_of_token_codes::token_memory_block) {
	DecRef(vector_token[i]);
	vector_token[i] = new TokenMemoryBlock();
      }
      TokenMemoryBlock *component_mem_token;
      component_mem_token = vector_token[i]->converTo<TokenMemoryBlock*>();
      unsigned int sz = bunch_size * component[i]->getOutputSize();
      component_mem_token->resize(sz);
      // copy from mem_token to component_mem_token
      doScopy(sz,
	      mem_token->getMemBlock(), pos, 1,
	      component_mem_token->getMemBlock(), 0, 1,
	      use_cuda);
      pos += sz;
    }
  }
  
  void JoinANNComponent::buildMemoryBlockToken(TokenMemoryBlock *&mem_block_token,
					       TokenBunchVector *token) {
    mem_block_token->clear();
    unsigned int pos = 0;
    for (unsigned int i=0; i<token.size(); ++i) {
      if (token[i]->getTokenCode() != table_of_token_codes::token_memory_block)
	ERROR_EXIT(128, "Incorrect token type\n");
      TokenMemoryBlock *component_output_mem_block;
      component_output_mem_block = token[i]->convertTo<TokenMemoryBlock*>();
      unsigned int sz = component_output_mem_block->getUsedSize();
      output->resize(pos + sz);
      // copy from component_output to output Token
      doScopy(sz,
	      component_output_mem_block->getMemBlock(), 0, 1,
	      output->getMemBlock(), pos, 1,
	      use_cuda);
      //
      pos += sz;
    }
  }

  void JoinANNComponent::buildMemoryBlockToken(TokenMemoryBlock *&mem_block_token,
					       Token *token) {
    if (token->getTokenCode() != table_of_token_codes::vector_Tokens)
      ERROR_EXIT(128, "Incorrect output token type");
    //
    TokenBunchVector *token_vector = token->convertTo<TokenBunchVector*>();
    buildMemoryBlockToken(mem_block_token, token_vector);
  }
  
  Token *JoinANNComponent::doForward(Token* _input, bool during_training) {
    if (input) DecRef(input);
    input = _input;
    IncRef(input);
    // INFO: will be possible to put this method inside next loop, but seems
    // more simpler a decoupled code
    buildInputBunchVector(input_vector, _input);
    for (unsigned int i=0; i<components->size(); ++i) {
      Token *component_output = components[i]->doForward(input_vector[i],
							 during_training);
      if (output_vector[i]) DecRef(output_vector[i]);
      output_vector[i] = component_output;
      IncRef(output_vector[i]);
    }
    // INFO: will be possible to put this method inside previous loop, but seems
    // more simpler a decoupled code
    buildMemoryBlockToken(output, output_vector);
    //
    return output;
  }

  Token *JoinANNComponent::doBackprop(Token *_error_input) {
    if (_error_input->getTokenCode() != table_of_token_codes::token_memory_block)
      ERROR_EXIT(128, "Incorrect error input token type\n");
    if (error_input) DecRef(error_input);
    error_input = _error_input;
    IncRef(error_input);
    //
    buildErrorInputBunchVector(error_input_vector, _error_input);
    for (unsigned int i=0; i<components.size(); ++i) {
      if (error_output_vector[i]) DecRef(error_output_vector[i]);
      error_output_vector[i] = component[i]->doBackprop(component_mem_token);
      IncRef(error_output_vector[i]);
    }
    return error_output_vector;
  }
  
  void JoinANNComponent::reset() {
    if (input) DecRef(input);
    if (error_input) DecRef(error_input);
  }
  
  ANNComponent *JoinANNComponent::clone();

  void JoinANNComponent::build(unsigned int _input_size,
			       unsigned int _output_size,
			       hash<string,Connections*> &weights_dict,
			       hash<string,ANNComponent*> &components_dict) {
  }
};
}
