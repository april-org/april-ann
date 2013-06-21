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
#include "token_matrix.h"
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
    
    // the first is only copied
    Token *current = (*error_input)[0];
    if (current->getTokenCode() != table_of_token_codes::token_matrix)
      ERROR_EXIT(128, "Incorrect token type, expected token matrix\n");
    TokenMatrixFloat *current_token;
    current_token = current->convertTo<TokenMatrixFloat*>();
    MatrixFloat *current_mat = current_token->getMatrix();
#ifdef USE_CUDA
    current_mat->setUseCuda(use_cuda);
#endif
    ASSERT_MATRIX(current_mat);
    unsigned int bunch_size = current_mat->getDimSize(0);
    
    // output token
    MatrixFloat *error_output_mat;
    int dims[2] = { static_cast<int>(bunch_size),
		    static_cast<int>(input_size) };
    error_output_mat = new MatrixFloat(2, dims, 0.0f, CblasColMajor);
#ifdef USE_CUDA
    error_output_mat->setUseCuda(use_cuda);
#endif
    TokenMatrixFloat *error_output_token = new TokenMatrixFloat(error_output_mat);
    AssignRef(error_output, error_output_token);
    error_output_mat->copy(current_mat);
    
    // The rest of tokens
    for (unsigned int i=1; i<times; ++i) {
      Token *current = (*error_input)[i];
      if (current->getTokenCode() != table_of_token_codes::token_matrix)
	ERROR_EXIT(128, "Incorrect token type, expected token matrix\n");
      current_token = current->convertTo<TokenMatrixFloat*>();
      current_mat = current_token->getMatrix();
#ifdef USE_CUDA
      current_mat->setUseCuda(use_cuda);
#endif
      ASSERT_MATRIX(current_mat);
      error_output_mat->axpy(1.0f, current_mat);
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
