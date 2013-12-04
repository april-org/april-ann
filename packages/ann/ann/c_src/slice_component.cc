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

  unsigned int mult(const int *v, int n) {
    int m = 1;
    for (int i=0; i<n; ++i) m *= v[i];
    return m;
  }
  
  SliceANNComponent::SliceANNComponent(const int *slice_offset,
				       const int *slice_size,
				       int n,
				       const char *name) :
    ANNComponent(name, 0, 0, mult(slice_size, n)),
    slice_offset(new int[n+1]),
    slice_size(new int[n+1]),
    n(n+1),
    input(0),
    output(0),
    error_input(0),
    error_output(0) {
    for (int i=1; i<this->n; ++i) {
      this->slice_offset[i] = slice_offset[i-1];
      this->slice_size[i]   = slice_size[i-1];
    }
  }
  
  SliceANNComponent::~SliceANNComponent() {
    if (input) DecRef(input);
    if (error_input) DecRef(error_input);
    if (output) DecRef(output);
    if (error_output) DecRef(error_output);
    delete[] slice_offset;
    delete[] slice_size;
  }
  
  Token *SliceANNComponent::doForward(Token* _input, bool during_training) {
    UNUSED_VARIABLE(during_training);
    if (_input->getTokenCode() != table_of_token_codes::token_matrix)
      ERROR_EXIT1(128, "Incorrect token found, only TokenMatrixFloat is "
		  "allowed [%s]\n", name.c_str());
    AssignRef(input, _input->convertTo<TokenMatrixFloat*>());    
    MatrixFloat *input_mat = input->getMatrix();
#ifdef USE_CUDA
    input_mat->setUseCuda(use_cuda);
#endif
    slice_offset[0] = 0;
    slice_size[0]   = input_mat->getDimSize(0);
    if (input_mat->getNumDim() < 2)
      ERROR_EXIT2(128, "At least 2-dimensional matrix is expected, found %d. "
		  "[%s]", input_mat->getNumDim(), name.c_str());
    MatrixFloat *output_mat = new MatrixFloat(input_mat,
					      slice_offset,
					      slice_size,
					      false);
    AssignRef(output, new TokenMatrixFloat(output_mat));
    return output;
  }

  Token *SliceANNComponent::doBackprop(Token *_error_input) {
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
    if (!output->getMatrix()->sameDim(error_input_mat))
      ERROR_EXIT1(128, "Error input token has incorrect dimensions [%s]\n",
		  name.c_str());
    MatrixFloat *error_output_mat;
    MatrixFloat *input_mat = input->getMatrix();
    error_output_mat = input_mat->cloneOnlyDims();
    error_output_mat->zeros();
    MatrixFloat *error_output_mat_slice = new MatrixFloat(error_output_mat,
							  slice_offset,
							  slice_size,
							  false);
    error_output_mat_slice->copy(error_input_mat);
    AssignRef(error_output, new TokenMatrixFloat(error_output_mat));
    delete error_output_mat_slice;
    return error_output;
  }
  
  void SliceANNComponent::reset(unsigned int it) {
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
  
  ANNComponent *SliceANNComponent::clone() {
    SliceANNComponent *slice_component = new SliceANNComponent(slice_offset+1,
							       slice_size+1,
							       n-1,
							       name.c_str());
    return slice_component;
  }
  
  void SliceANNComponent::build(unsigned int _input_size,
				unsigned int _output_size,
				hash<string,MatrixFloat*> &weights_dict,
				hash<string,ANNComponent*> &components_dict) {
    unsigned int sz = 1;
    for (int i=1; i<this->n; ++i) sz *= slice_size[i];
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
