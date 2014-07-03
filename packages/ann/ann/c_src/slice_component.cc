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
#include "slice_component.h"
#include "wrapper.h"

namespace ANN {
  
  SliceANNComponent::SliceANNComponent(const int *slice_offset,
				       const int *slice_size,
				       int n,
				       const char *name) :
    VirtualMatrixANNComponent(name, 0, 0, mult(slice_size, n)),
    slice_offset(new int[n+1]),
    slice_size(new int[n+1]),
    n(n+1) {
    setInputContiguousProperty(false);
    for (int i=1; i<this->n; ++i) {
      this->slice_offset[i] = slice_offset[i-1];
      this->slice_size[i]   = slice_size[i-1];
    }
  }
  
  SliceANNComponent::~SliceANNComponent() {
    delete[] slice_offset;
    delete[] slice_size;
  }
  
  MatrixFloat *SliceANNComponent::
  privateDoForward(MatrixFloat* input_mat, bool during_training) {
    UNUSED_VARIABLE(during_training);
    slice_offset[0] = 0;
    slice_size[0]   = input_mat->getDimSize(0);
    if (input_mat->getNumDim() < 2)
      ERROR_EXIT2(128, "At least 2-dimensional matrix is expected, found %d. "
		  "[%s]", input_mat->getNumDim(), name.c_str());
    MatrixFloat *output_mat = new MatrixFloat(input_mat,
					      slice_offset,
					      slice_size,
					      false);
    return output_mat;
  }

  MatrixFloat *SliceANNComponent::
  privateDoBackprop(MatrixFloat *error_input_mat) {
    MatrixFloat *input_mat = getInputMatrix();
    MatrixFloat *error_output_mat;
    error_output_mat = input_mat->cloneOnlyDims();
    error_output_mat->zeros();
    MatrixFloat *error_output_mat_slice = new MatrixFloat(error_output_mat,
							  slice_offset,
							  slice_size,
							  false);
    error_output_mat_slice->copy(error_input_mat);
    delete error_output_mat_slice;
    return error_output_mat;
  }
  
  void SliceANNComponent::privateReset(unsigned int it) {
    UNUSED_VARIABLE(it);
  }
  
  ANNComponent *SliceANNComponent::clone() {
    SliceANNComponent *slice_component = new SliceANNComponent(slice_offset+1,
							       slice_size+1,
							       n-1,
							       name.c_str());
    return slice_component;
  }
  
  void SliceANNComponent::build(unsigned int _input_size,
				unsigned int _output_size,
				MatrixFloatSet *weights_dict,
				hash<string,ANNComponent*> &components_dict) {
    unsigned int sz = mult(slice_size+1, n-1);
    //
    if (_output_size != 0 && _output_size != sz)
      ERROR_EXIT2(256, "Incorrect output size, expected %d, found %d\n",
		  sz, _output_size);
    //
    ANNComponent::build(_input_size, sz, weights_dict, components_dict);
    if (input_size != 0 && input_size < output_size)
      ERROR_EXIT2(256, "Incorrect input size %d < slice size %d\n",
		  input_size, output_size);
  }
  
  char *SliceANNComponent::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.components.slice{ name='%s', size={", name.c_str());
    for (int i=1; i<n; ++i) buffer.printf(" %d,", slice_size[i]);
    buffer.printf("}, pos={");
    for (int i=1; i<n; ++i) buffer.printf(" %d,", slice_offset[i] + 1);
    buffer.printf("} }");
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
}
