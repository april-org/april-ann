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
#include "rewrap_component.h"
#include "wrapper.h"

namespace ANN {
  
  RewrapANNComponent::RewrapANNComponent(const int *rewrap_dims, int n,
					 const char *name) :
    ANNComponent(name, 0, 0, 0),
    rewrap_dims(new int[n+1]), n(n+1),
    input(0),
    output(0),
    error_input(0),
    error_output(0) {
    for (int i=1; i<this->n; ++i)
      this->rewrap_dims[i] = rewrap_dims[i-1];
  }
  
  RewrapANNComponent::~RewrapANNComponent() {
    if (input) DecRef(input);
    if (error_input) DecRef(error_input);
    if (output) DecRef(output);
    if (error_output) DecRef(error_output);
    delete[] rewrap_dims;
  }
  
  Token *RewrapANNComponent::doForward(Token* _input, bool during_training) {
    if (_input->getTokenCode() != table_of_token_codes::token_matrix)
      ERROR_EXIT1(128, "Incorrect token found, only TokenMatrixFloat is "
		  "allowed [%s]\n", name.c_str());
    AssignRef(input, _input->convertTo<TokenMatrixFloat*>());    
    MatrixFloat *input_mat = input->getMatrix();
#ifdef USE_CUDA
    input_mat->setUseCuda(use_cuda);
#endif
    rewrap_dims[0] = input_mat->getDimSize(0);
    if (input_mat->getNumDim() < 2)
      ERROR_EXIT2(128, "At 2-dimensional matrix is expected, found %d. "
		  "[%s]", input_mat->getNumDim(), name.c_str());
    if (!input_mat->getIsContiguous()) {
      input_mat = input_mat->clone();
      AssignRef(input,new TokenMatrixFloat(input_mat));
    }
    MatrixFloat *output_mat = input_mat->rewrap(rewrap_dims, n);
    AssignRef(output, new TokenMatrixFloat(output_mat));
    return output;
  }

  Token *RewrapANNComponent::doBackprop(Token *_error_input) {
    if (_error_input == 0) {
      if (error_input)  { DecRef(error_input);  error_input  = 0; }
      if (error_output) { DecRef(error_output); error_output = 0; }
      return 0;
    }
    if (_error_input->getTokenCode() != table_of_token_codes::token_matrix)
      ERROR_EXIT1(128, "Incorrect error input token type, "
		  "expected TokenMatrixFloat [%s]\n", name.c_str());
    AssignRef(error_input, _error_input->convertTo<TokenMatrixFloat*>());
    MatrixFloat *error_input_mat = error_input->getMatrix();
#ifdef USE_CUDA
    error_input_mat->setUseCuda(use_cuda);
#endif
    if (!error_input_mat->getIsContiguous()) {
      error_input_mat = error_input_mat->clone();
      AssignRef(error_input,new TokenMatrixFloat(error_input_mat));
    }
    if (!output->getMatrix()->sameDim(error_input_mat))
      ERROR_EXIT1(128, "Error input token has incorrect dimensions [%s]\n",
		  name.c_str());
    MatrixFloat *error_output_mat;
    MatrixFloat *input_mat = input->getMatrix();
    error_output_mat = error_input_mat->rewrap(input_mat->getDimPtr(),
					       input_mat->getNumDim());
    AssignRef(error_output, new TokenMatrixFloat(error_output_mat));
    return error_output;
  }
  
  void RewrapANNComponent::reset() {
    if (input) DecRef(input);
    if (error_input) DecRef(error_input);
    if (output) DecRef(output);
    if (error_output) DecRef(error_output);
    input	 = 0;
    error_input	 = 0;
    output	 = 0;
    error_output = 0;
  }

  ANNComponent *RewrapANNComponent::clone() {
    RewrapANNComponent *rewrap_component = new RewrapANNComponent(rewrap_dims+1,
								  n-1,
								  name.c_str());
    return rewrap_component;
  }
  
  void RewrapANNComponent::build(unsigned int _input_size,
				 unsigned int _output_size,
				 hash<string,Connections*> &weights_dict,
				 hash<string,ANNComponent*> &components_dict) {
    ANNComponent::build(_input_size, _output_size,
			weights_dict, components_dict);
  }
  
  char *RewrapANNComponent::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.components.rewrap{ name='%s', size={", name.c_str());
    for (int i=1; i<n; ++i) buffer.printf(" %d,", rewrap_dims[i]);
    buffer.printf("} }");
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
}
