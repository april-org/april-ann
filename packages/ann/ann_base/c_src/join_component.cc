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
#include "error_print.h"
#include "table_of_token_codes.h"
#include "token_vector.h"
#include "token_memory_block.h"
#include "join_component.h"
#include "wrapper.h"

namespace ANN {
  
  JoinANNComponent::JoinANNComponent(const char *name) :
    ANNComponent(name),
    input(0),
    error_output(0),
    output(0),
    error_input(0),
    input_vector(new TokenBunchVector()),
    error_input_vector(new TokenBunchVector()),
    output_vector(new TokenBunchVector()),
    error_output_vector(new TokenBunchVector()),
    segmented_input(false),
    bunch_size(0) {
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
    DecRef(input_vector);
    DecRef(error_input_vector);
    if (output) DecRef(output);
    DecRef(output_vector);
    if (error_output) DecRef(error_output);
    DecRef(error_output_vector);
  }
  
  void JoinANNComponent::addComponent(ANNComponent *component) {
    components.push_back(component);
    IncRef(component);
  }

  void JoinANNComponent::buildInputBunchVector(TokenBunchVector *&result_vector_token,
					       Token *input_token) {
    switch(input_token->getTokenCode()) {
    case table_of_token_codes::token_mem_block: {
      segmented_input = false;
      TokenMemoryBlock *mem_input_token;
      mem_input_token = input_token->convertTo<TokenMemoryBlock*>();
      bunch_size = mem_input_token->getUsedSize() / input_size;
      assert((bunch_size*input_size == mem_input_token->getUsedSize()) &&
	     "Incorrect token size, is not divisible by bunch_size");
      unsigned int pos=0;
      for (unsigned int i=0; i<result_vector_token->size(); ++i) {
	unsigned int sz = bunch_size * components[i]->getInputSize();
	TokenMemoryBlock *component_mem_token = new TokenMemoryBlock(sz);
	AssignRef((*result_vector_token)[i], component_mem_token);
	// copy from _input to component_mem_token
	doScopy(sz,
		mem_input_token->getMemBlock(), pos, 1,
		component_mem_token->getMemBlock(), 0, 1,
		use_cuda);
	pos += sz;
      }
      break;
    }
    case table_of_token_codes::vector_Tokens: {
      segmented_input = true;
      TokenBunchVector *input_vector_token;
      input_vector_token = input_token->convertTo<TokenBunchVector*>();
      if (result_vector_token->size() != input_vector_token->size())
	ERROR_EXIT2(128, "Incorrect number of components at input vector, "
		    "expected %u and found %u\n",
		    result_vector_token->size(), input_vector_token->size());
      for (unsigned int i=0; i<result_vector_token->size(); ++i)
	AssignRef((*result_vector_token)[i], (*input_vector_token)[i]);
      bunch_size = 0;
      break;
    }
    default:
      ERROR_EXIT(129, "Incorrect token type");
    }
  }
  
  void JoinANNComponent::buildErrorInputBunchVector(TokenBunchVector *&vector_token,
						    Token *token) {
    if (token->getTokenCode() != table_of_token_codes::token_mem_block)
      ERROR_EXIT(128, "Incorrect token type\n");
    //
    TokenMemoryBlock *mem_token = token->convertTo<TokenMemoryBlock*>();
    if (bunch_size == 0) bunch_size = mem_token->getUsedSize() / output_size;
    assert((bunch_size == (mem_token->getUsedSize() / output_size)) &&
	   "Incorrect bunch size at input error");
    assert((bunch_size*output_size == mem_token->getUsedSize()) &&
	   "Incorrect input error token size, not divisible by bunch_size");
    unsigned int pos=0;
    for (unsigned int i=0; i<vector_token->size(); ++i) {
      unsigned int sz = bunch_size * components[i]->getOutputSize();
      TokenMemoryBlock *component_mem_token;
      component_mem_token = new TokenMemoryBlock(sz);
      AssignRef((*vector_token)[i], component_mem_token);
      doScopy(sz,
	      mem_token->getMemBlock(), pos, 1,
	      component_mem_token->getMemBlock(), 0, 1,
	      use_cuda);
      pos += sz;
    }
  }
  
  TokenMemoryBlock *JoinANNComponent::buildMemoryBlockToken(TokenBunchVector *token) {
    TokenMemoryBlock *mem_block_token;
    if ((*token)[0]->getTokenCode() != table_of_token_codes::token_mem_block)
      ERROR_EXIT(128, "Incorrect token type\n");
    if (bunch_size == 0) {
      TokenMemoryBlock *aux = (*token)[0]->convertTo<TokenMemoryBlock*>();
      bunch_size =aux->getUsedSize() / components[0]->getOutputSize();
    }
    mem_block_token = new TokenMemoryBlock(output_size*bunch_size);
    unsigned int pos = 0;
    for (unsigned int i=0; i<token->size(); ++i) {
      if ((*token)[i]->getTokenCode() != table_of_token_codes::token_mem_block)
	ERROR_EXIT(128, "Incorrect token type\n");
      TokenMemoryBlock *component_output_mem_block;
      component_output_mem_block = (*token)[i]->convertTo<TokenMemoryBlock*>();
      unsigned int sz = component_output_mem_block->getUsedSize();
      assert(pos + sz <= mem_block_token->getUsedSize());
      // copy from component_output to output Token
      doScopy(sz,
	      component_output_mem_block->getMemBlock(), 0, 1,
	      mem_block_token->getMemBlock(), pos, 1,
	      use_cuda);
      //
      pos += sz;
    }
    return mem_block_token;
  }

  TokenMemoryBlock *JoinANNComponent::buildMemoryBlockToken(Token *token) {
    if (token->getTokenCode() != table_of_token_codes::vector_Tokens)
      ERROR_EXIT(128, "Incorrect output token type");
    //
    TokenBunchVector *vector_token = token->convertTo<TokenBunchVector*>();
    return buildMemoryBlockToken(vector_token);
  }
  
  Token *JoinANNComponent::doForward(Token* _input, bool during_training) {
    AssignRef(input, _input);
    // INFO: will be possible to put this method inside next loop, but seems
    // more simpler a decoupled code
    buildInputBunchVector(input_vector, _input);
    for (unsigned int i=0; i<components.size(); ++i)
      AssignRef((*output_vector)[i],
		components[i]->doForward((*input_vector)[i], during_training));
    // INFO: will be possible to put this method inside previous loop, but seems
    // more simpler a decoupled code
    AssignRef(output, buildMemoryBlockToken(output_vector));
    //
    return output;
  }

  Token *JoinANNComponent::doBackprop(Token *_error_input) {
    if (_error_input == 0) {
      if (error_input)  { DecRef(error_input);  error_input  = 0; }
      if (error_output) { DecRef(error_output); error_output = 0; }
      return 0;
    }
    if (_error_input->getTokenCode() != table_of_token_codes::token_mem_block)
      ERROR_EXIT(128, "Incorrect error input token type\n");
    AssignRef(error_input, _error_input->convertTo<TokenMemoryBlock*>());
    // INFO: will be possible to put this method inside previous loop, but seems
    // more simpler a decoupled code
    buildErrorInputBunchVector(error_input_vector, _error_input);
    for (unsigned int i=0; i<components.size(); ++i)
      AssignRef((*error_output_vector)[i],
		components[i]->doBackprop((*error_input_vector)[i]));
    // error_output_vector has the gradients of each component stored as
    // array. Depending on the received input, this vector would be returned as
    // it is, or gradients will be stored as a TokenMemoryBlock joining all
    // array positions.
    if (segmented_input) AssignRef(error_output, error_output_vector);
    // INFO: will be possible to put this method inside previous loop, but
    // seems more simpler a decoupled code
    else AssignRef(error_output, buildMemoryBlockToken(error_output_vector));
    return error_output;
  }

  void JoinANNComponent::doUpdate() {
    for (unsigned int i=0; i<components.size(); ++i)
      components[i]->doUpdate();
  }
  
  void JoinANNComponent::reset() {
    if (input) DecRef(input);
    if (error_input) DecRef(error_input);
    if (output) DecRef(output);
    if (error_output) DecRef(error_output);
    input	 = 0;
    error_input	 = 0;
    output	 = 0;
    error_output = 0;
    bunch_size   = 0;
    for (unsigned int i=0; i<components.size(); ++i)
      components[i]->reset();
  }
  
  ANNComponent *JoinANNComponent::clone() {
    JoinANNComponent *join_component = new JoinANNComponent(name.c_str());
    for (unsigned int i=0; i<components.size(); ++i)
      join_component->addComponent(components[i]->clone());
    join_component->input_size  = input_size;
    join_component->output_size = output_size;
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
      components[i]->build(0, 0, weights_dict, components_dict);
      computed_input_size  += components[i]->getInputSize();
      computed_output_size += components[i]->getOutputSize();
      Token *null(0);
      input_vector->push_back(null);
      output_vector->push_back(null);
      error_input_vector->push_back(null);
      error_output_vector->push_back(null);
    }
    if (input_size == 0)  input_size  = computed_input_size;
    if (output_size == 0) output_size = computed_output_size;
    if (input_size != computed_input_size)
      ERROR_EXIT2(128, "Incorrect input sizes, components inputs sum %d but "
		  "expected %d\n", computed_input_size, input_size);
    if (output_size != computed_output_size)
      ERROR_EXIT2(128, "Incorrect output sizes, components outputs sum %d but "
		  "expected %d\n", computed_output_size, output_size);
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
  
}
