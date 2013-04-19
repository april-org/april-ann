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
    input(0),
    error_output(new TokenMemoryBlock()),
    error_input(0),
    output(new TokenMemoryBlock()),
    input_vector(new TokenBunchVector()),
    error_input_vector(new TokenBunchVector()),
    output_vector(new TokenBunchVector()),
    error_output_vector(new TokenBunchVector()),
    segmented_input(false) {
    IncRef(input_vector);
    IncRef(error_input_vector);
    IncRef(output);
    IncRef(output_vector);
    IncRef(error_output);
    IncRef(error_output_vector);
  }
  
  JoinANNComponent::~JoinANNComponent() {
    for (unsigned int i=0; i<components.size(); ++i)
      DecRef(components[i]);
    if (input) DecRef(input);
    if (error_input) DecRef(error_input);
    DecRef(input_vector);
    DecRef(error_input_vector);
    DecRef(output);
    DecRef(output_vector);
    DecRef(error_output);
    DecRef(error_output_vector);
  }

  void JoinANNComponent::addComponent(ANNComponent *component) {
    components.push_back(component);
    IncRef(component);
  }

  void JoinANNComponent::buildInputBunchVector(TokenBunchVector *&vector_token,
					       Token *token) {
    switch(token->getTokenCode()) {
    case table_of_token_codes::TokenMemoryBlock: {
      segmented_input = false;
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
      segmented_input = true;
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
    // INFO: will be possible to put this method inside previous loop, but seems
    // more simpler a decoupled code
    buildErrorInputBunchVector(error_input_vector, _error_input);
    for (unsigned int i=0; i<components.size(); ++i) {
      if (error_output_vector[i]) DecRef(error_output_vector[i]);
      error_output_vector[i] = component[i]->doBackprop(component_mem_token);
      IncRef(error_output_vector[i]);
    }
    // error_output_vector has the gradients of each component stored as
    // array. Depending on the received input, this vector would be returned as
    // it is, or gradients will be stored as a TokenMemoryBlock joining all
    // array positions.
    if (segmented_input) {
      if (error_output) DecRef(error_output);
      error_output = error_output_vector;
      IncRef(error_output);
    }
    else {
      TokenMemoryBlock *error_output_mem_block;
      if (error_output->getTokenCode() != table_of_token_codes::token_memory_block) {
	DecRef(error_output);
	error_output = new TokenMemoryBlock();
	IncRef(error_output);
      }
      error_output_mem_block = error_output->convertTo<TokenMemoryBlock*>();
      // INFO: will be possible to put this method inside previous loop, but
      // seems more simpler a decoupled code
      buildMemoryBlockToken(error_output_mem_block, error_output_vector);
    }
    return error_output;
  }

  void JoinANNComponent::doUpdate() {
    for (unsigned int i=0; i<components.size(); ++i)
      components[i]->doUpdate();
  }
  
  void JoinANNComponent::reset() {
    if (input) DecRef(input);
    if (error_input) DecRef(error_input);
    for (unsigned int i=0; i<components.size(); ++i)
      components[i]->reset();
  }
  
  ANNComponent *JoinANNComponent::clone() {
    ANNComponent *join_component = new JoinANNComponent(name);
    for (unsigned int i=0; i<components.size(); ++i)
      join_component->addComponent(components[i]->clone());
    return join_component;
  }

  void JoinANNComponent::build(unsigned int _input_size,
			       unsigned int _output_size,
			       hash<string,Connections*> &weights_dict,
			       hash<string,ANNComponent*> &components_dict) {
    ANNComponent::build(_input_size, _output_size,
			weights_dict, components_dict);
    //
    if (components.size() == 0)
      ERROR_EXIT(128, "JoinANNComponent needs one or more components, "
		 "use addComponent method\n");
    unsigned int computed_input_size = 0, computed_output_size = 0;
    for (unsigned int i=0; i<components.size(); ++i) {
      computed_input_size  += components[i]->getInputSize();
      computed_output_size += components[i]->getOutputSize();
    }
    if (input_size == 0)  input_size  = computed_input_size;
    if (output_size == 0) output_size = computed_output_size;
    if (input_size != computed_input_size)
      ERROR_EXIT(128, "Incorrect input sizes, components inputs sum %d but "
		 " expected %d\n", computed_input_size, input_size);
    if (output_size != computed_output_size)
      ERROR_EXIT(128, "Incorrect output sizes, components outputs sum %d but "
		 " expected %d\n", computed_output_size, output_size);
    //
    for (unsigned int i=0; i<components.size(); ++i)
      components[i]->build(0, 0, weights_dict, components_dict);
  }
  
  void JoinANNComponent::setUseCuda(bool v) {
    ANNComponent::setUseCuda(v);
    for (unsigned int c=0; c<components.size(); ++c)
      components[c]->setUseCuda(v);
  }
  
  void JoinANNComponent::setOption(const char *name, double value) {
    for (unsigned int c=0; c<components.size(); ++c)
      components[c]->setOption(name, value);
  }
  
  bool JoinANNComponent::hasOption(const char *name) {
    bool ret = false;
    for (unsigned int c=0; c<components.size() && !ret; ++c)
      ret = components[c]->hasOption(name);
    return ret;
  }
  
  double JoinANNComponent::getOption(const char *name) {
    for (unsigned int c=0; c<components.size(); ++c) {
      if (components[c]->hasOption(name))
	return components[c]->getOption(name);
    }
    return ANNComponent::getOption(name);
  }

  void JoinANNComponent::copyWeights(hash<string,Connections*> &weights_dict) {
    for (unsigned int i=0; i<components.size(); ++i)
      components[i]->copyWeights(weights_dict);
  }

  void JoinANNComponent::copyComponents(hash<string,ANNComponent*> &components_dict) {
    ANNComponent::copyComponents(components_dict);
    for (unsigned int i=0; i<components.size(); ++i)
      components[i]->copyComponents(components_dict);
  }
  
  ANNComponent *JoinANNComponent::getComponent(string &name) {
    ANNComponent *c = ANNComponent::getComponent(name);
    for (unsigned int i=0; i<components.size() && c == 0; ++i)
      c = components[i]->getComponent(name);
    return c;
  }
  
};
}
