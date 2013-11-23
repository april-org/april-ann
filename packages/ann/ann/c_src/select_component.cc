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
#include "unused_variable.h"
#include "error_print.h"
#include "table_of_token_codes.h"
#include "token_vector.h"
#include "token_matrix.h"
#include "select_component.h"
#include "wrapper.h"

namespace ANN {
  
  SelectANNComponent::SelectANNComponent(int dimension, int index,
					 const char *name) :
    ANNComponent(name, 0, 0, 0),
    dimension(dimension), index(index),
    input(0),
    output(0),
    error_input(0),
    error_output(0) {
  }
  
  SelectANNComponent::~SelectANNComponent() {
    if (input) DecRef(input);
    if (error_input) DecRef(error_input);
    if (output) DecRef(output);
    if (error_output) DecRef(error_output);
  }
  
  Token *SelectANNComponent::doForward(Token* _input, bool during_training) {
    UNUSED_VARIABLE(during_training);
    if (_input->getTokenCode() != table_of_token_codes::token_matrix)
      ERROR_EXIT1(128, "Incorrect token found, only TokenMatrixFloat is "
		  "allowed [%s]\n", name.c_str());
    AssignRef(input, _input->convertTo<TokenMatrixFloat*>());    
    MatrixFloat *input_mat = input->getMatrix();
#ifdef USE_CUDA
    input_mat->setUseCuda(use_cuda);
#endif
    if (input_mat->getNumDim() < 3)
      ERROR_EXIT2(128, "At least 3 dimensional matrix is expected, found %d. "
		  "First dimension is bunch-size, and the rest are pattern data "
		  "[%s]", input_mat->getNumDim(), name.c_str());
    MatrixFloat *aux_output_mat = input_mat->select(dimension+1, index);
    MatrixFloat *output_mat     = aux_output_mat->clone();
    delete aux_output_mat;
    AssignRef(output, new TokenMatrixFloat(output_mat));
    return output;
  }

  Token *SelectANNComponent::doBackprop(Token *_error_input) {
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
    if (!output->getMatrix()->sameDim(error_input_mat))
      ERROR_EXIT1(128, "Error input token has incorrect dimensions [%s]\n",
		  name.c_str());
    MatrixFloat *error_output_mat = input->getMatrix()->cloneOnlyDims();
    error_output_mat->zeros();
    MatrixFloat *select_error_output_mat;
    select_error_output_mat = error_output_mat->select(dimension+1, index);
    select_error_output_mat->copy(error_input_mat);
#ifdef USE_CUDA
    select_error_output_mat->setUseCuda(use_cuda);
#endif
    AssignRef(error_output, new TokenMatrixFloat(select_error_output_mat));
    return error_output;
  }
  
  void SelectANNComponent::reset(unsigned int it) {
    UNUSED_VARIABLE(it);
    if (input) DecRef(input);
    if (error_input) DecRef(error_input);
    if (output) DecRef(output);
    if (error_output) DecRef(error_output);
    input	 = 0;
    error_input	 = 0;
    output	 = 0;
    error_output = 0;
  }

  ANNComponent *SelectANNComponent::clone() {
    SelectANNComponent *select_component = new SelectANNComponent(dimension,
								  index,
								  name.c_str());
    return select_component;
  }
  
  void SelectANNComponent::build(unsigned int _input_size,
				 unsigned int _output_size,
				 hash<string,Connections*> &weights_dict,
				 hash<string,ANNComponent*> &components_dict) {
    ANNComponent::build(_input_size, _output_size,
			weights_dict, components_dict);
  }
  
  char *SelectANNComponent::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.components.select{ name='%s',dimension=%d,index=%d }",
		  name.c_str(), dimension, index);
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
}
