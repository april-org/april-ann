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
#include "join_component.h"
#include "wrapper.h"

namespace ANN {
  
  JoinANNComponent::JoinANNComponent(const char *name) :
    ANNComponent(name),
    input(0),
    error_output(0),
    output(0),
    error_input(0),
    input_vector(0),
    error_input_vector(0),
    output_vector(0),
    error_output_vector(0),
    segmented_input(false),
    bunch_size(0) {
  }
  
  JoinANNComponent::~JoinANNComponent() {
    for (unsigned int i=0; i<components.size(); ++i)
      DecRef(components[i]);
    if (input) DecRef(input);
    if (error_input) DecRef(error_input);
    if (input_vector) DecRef(input_vector);
    if (error_input_vector) DecRef(error_input_vector);
    if (output) DecRef(output);
    if (output_vector) DecRef(output_vector);
    if (error_output) DecRef(error_output);
    if (error_output_vector) DecRef(error_output_vector);
  }
  
  void JoinANNComponent::addComponent(ANNComponent *component) {
    components.push_back(component);
    IncRef(component);
  }

  void JoinANNComponent::buildInputBunchVector(TokenBunchVector *&result_vector_token,
					       Token *input_token) {
    switch(input_token->getTokenCode()) {
    case table_of_token_codes::token_matrix: {
      segmented_input = false;
      TokenMatrixFloat *input_mat_token;
      input_mat_token = input_token->convertTo<TokenMatrixFloat*>();
      MatrixFloat *input_mat = input_mat_token->getMatrix();
      ASSERT_MATRIX(input_mat);
#ifdef USE_CUDA
      input_mat->setUseCuda(use_cuda);
#endif
      unsigned int mat_pat_size = static_cast<unsigned int>(input_mat->getDimSize(1));
      assert(mat_pat_size==input_size && "Incorrect token matrix size");
      int sizes[2]  = { input_mat->getDimSize(0),
			input_mat->getDimSize(1) };
      int coords[2] = { 0, 0 };
      for (unsigned int i=0; i<result_vector_token->size(); ++i) {
	const unsigned int sz = components[i]->getInputSize();
	// submatrix at coords with sizes, deep copy of original matrix
	sizes[1] = sz;
	MatrixFloat *output_mat = new MatrixFloat(input_mat,coords,sizes,true);
#ifdef USE_CUDA
	output_mat->setUseCuda(use_cuda);
#endif
	coords[1] += sz;
	TokenMatrixFloat *component_mat_token = new TokenMatrixFloat(output_mat);
	AssignRef((*result_vector_token)[i], component_mat_token);
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
    if (token->getTokenCode() != table_of_token_codes::token_matrix)
      ERROR_EXIT(128, "Incorrect token type\n");
    //
    TokenMatrixFloat *mat_token = token->convertTo<TokenMatrixFloat*>();
    MatrixFloat *mat = mat_token->getMatrix();
    ASSERT_MATRIX(mat);
#ifndef USE_CUDA
    mat->setUseCuda(use_cuda);
#endif
    unsigned int mat_pat_size = static_cast<unsigned int>(mat->getDimSize(1));
    assert(mat_pat_size==output_size && "Incorrect token matrix size");
    int sizes[2]  = { mat->getDimSize(0), 0 };
    int coords[2] = { 0, 0 };
    for (unsigned int i=0; i<vector_token->size(); ++i) {
      const unsigned int sz = components[i]->getOutputSize();
      // submatrix at coords with sizes
      sizes[1] = sz;
      MatrixFloat *component_mat = new MatrixFloat(mat, coords, sizes, true);
      coords[1] += sz;
      TokenMatrixFloat *component_mat_token = new TokenMatrixFloat(component_mat);
      AssignRef((*vector_token)[i], component_mat_token);
    }
  }
  
  TokenMatrixFloat *JoinANNComponent::buildMatrixFloatToken(TokenBunchVector *token,
							    bool is_output) {
    MatrixFloat *full_mat, *aux_mat;
    if ((*token)[0]->getTokenCode() != table_of_token_codes::token_matrix)
      ERROR_EXIT1(128,"Incorrect token type at TokenBunchVector pos %d\n",0);
    aux_mat  = (*token)[0]->convertTo<TokenMatrixFloat*>()->getMatrix();
    int sizes[2]  = { aux_mat->getDimSize(0),
		      (is_output) ?
		      static_cast<int>(output_size) :
		      static_cast<int>(input_size) };
    int coords[2] = { 0, 0 };
    full_mat = new MatrixFloat(2, sizes, 0.0f, CblasColMajor);
#ifdef USE_CUDA
    full_mat->setUseCuda(use_cuda);
#endif
    for (unsigned int i=0; i<token->size(); ++i) {
      if ((*token)[i]->getTokenCode() != table_of_token_codes::token_matrix)
	ERROR_EXIT(128, "Incorrect token type\n");
      aux_mat = (*token)[i]->convertTo<TokenMatrixFloat*>()->getMatrix();
      ASSERT_MATRIX(aux_mat);
      const unsigned int sz = ( (is_output) ?
				components[i]->getOutputSize() :
				components[i]->getInputSize() );
      sizes[1] = sz;
      // Destination data sub-matrix (references original data matrix)

      // FIXME: This could be improved adding a rewrap method to matrix,
      // avoiding multiple new and delete operations
      MatrixFloat *submat = new MatrixFloat(full_mat, coords, sizes, false);
      submat->copy(aux_mat);
      delete submat;
      coords[1] += sz;
    }
    return new TokenMatrixFloat(full_mat);
  }

  TokenMatrixFloat *JoinANNComponent::buildMatrixFloatToken(Token *token,
							    bool is_output) {
    if (token->getTokenCode() != table_of_token_codes::vector_Tokens)
      ERROR_EXIT(128, "Incorrect output token type");
    //
    TokenBunchVector *vector_token = token->convertTo<TokenBunchVector*>();
    return buildMatrixFloatToken(vector_token, is_output);
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
    AssignRef(output, buildMatrixFloatToken(output_vector, true));
    //
    return output;
  }

  Token *JoinANNComponent::doBackprop(Token *_error_input) {
    if (_error_input == 0) {
      if (error_input)  { DecRef(error_input);  error_input  = 0; }
      if (error_output) { DecRef(error_output); error_output = 0; }
      return 0;
    }
    if (_error_input->getTokenCode() != table_of_token_codes::token_matrix)
      ERROR_EXIT(128, "Incorrect error input token type\n");
    AssignRef(error_input, _error_input->convertTo<TokenMatrixFloat*>());
    // INFO: will be possible to put this method inside previous loop, but seems
    // more simpler a decoupled code
    buildErrorInputBunchVector(error_input_vector, _error_input);
    for (unsigned int i=0; i<components.size(); ++i)
      AssignRef((*error_output_vector)[i],
		components[i]->doBackprop((*error_input_vector)[i]));
    // error_output_vector has the gradients of each component stored as
    // array. Depending on the received input, this vector would be returned as
    // it is, or gradients will be stored as a TokenMatrixFloat joining all
    // array positions.
    if (segmented_input) AssignRef(error_output, error_output_vector);
    // INFO: will be possible to put this method inside previous loop, but
    // seems more simpler a decoupled code
    else AssignRef(error_output, buildMatrixFloatToken(error_output_vector,
						       false));
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
    AssignRef(input_vector, new TokenBunchVector(components.size()));
    AssignRef(output_vector, new TokenBunchVector(components.size()));
    AssignRef(error_input_vector, new TokenBunchVector(components.size()));
    AssignRef(error_output_vector, new TokenBunchVector(components.size()));
    for (unsigned int i=0; i<components.size(); ++i) {
      components[i]->build(0, 0, weights_dict, components_dict);
      computed_input_size  += components[i]->getInputSize();
      computed_output_size += components[i]->getOutputSize();
      (*input_vector)[i]	= 0;
      (*output_vector)[i]	= 0;
      (*error_input_vector)[i]	= 0;
      (*error_output_vector)[i] = 0;
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

  char *JoinANNComponent::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.components.join{ name='%s' }");
    for (unsigned int i=0; i<components.size(); ++i) {
      // FIXME: please, this code could be improved freeing the aux array
      char *aux = components[i]->toLuaString();
      buffer.printf(":add(%s)", aux);
      delete[] aux;
    }
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
  
}
