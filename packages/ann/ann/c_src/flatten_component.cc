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
#include "flatten_component.h"

using namespace Basics;
using namespace AprilUtils;
using namespace AprilMath;

namespace ANN {
  
  FlattenANNComponent::FlattenANNComponent(const char *name) :
    VirtualMatrixANNComponent(name, 0, 0, 0) {
    setInputContiguousProperty(true);
  }
  
  FlattenANNComponent::~FlattenANNComponent() {
  }
  
  MatrixFloat *FlattenANNComponent::
  privateDoForward(MatrixFloat* input_mat, bool during_training) {
    IncRef(input_mat);
    UNUSED_VARIABLE(during_training);
    if (input_mat->getNumDim() < 2)
      ERROR_EXIT2(128, "At 2-dimensional matrix is expected, found %d. "
		  "[%s]", input_mat->getNumDim(), name.c_str());
    flatten_dims[0] = input_mat->getDimSize(0);
    flatten_dims[1] = input_mat->size() / flatten_dims[0];
    MatrixFloat *output_mat = input_mat->rewrap(flatten_dims, 2);
    DecRef(input_mat);
    return output_mat;
  }

  MatrixFloat *FlattenANNComponent::
  privateDoBackprop(MatrixFloat *error_input_mat) {
    IncRef(error_input_mat);
    MatrixFloat *error_output_mat;
    MatrixFloat *input_mat = getInputMatrix();
    error_output_mat = error_input_mat->rewrap(input_mat->getDimPtr(),
					       input_mat->getNumDim());
    DecRef(error_input_mat);
    return error_output_mat;
  }
  
  void FlattenANNComponent::privateReset(unsigned int it) {
    UNUSED_VARIABLE(it);
  }

  ANNComponent *FlattenANNComponent::clone() {
    FlattenANNComponent *flatten_component;
    flatten_component = new FlattenANNComponent(name.c_str());
    return flatten_component;
  }
  
  void FlattenANNComponent::build(unsigned int _input_size,
				 unsigned int _output_size,
				 MatrixFloatSet *weights_dict,
				 hash<string,ANNComponent*> &components_dict) {
    ANNComponent::build(_input_size, _output_size,
			weights_dict, components_dict);
  }
  
  char *FlattenANNComponent::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.components.flatten{ name='%s' }", name.c_str());
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
}
