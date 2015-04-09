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
#include "smart_ptr.h"

using namespace AprilMath;
using namespace AprilMath::MatrixExt::BLAS;
using namespace AprilMath::MatrixExt::Initializers;
using namespace AprilUtils;
using namespace Basics;

namespace ANN {
  
  SelectANNComponent::SelectANNComponent(int dimension, int index,
					 const char *name) :
    VirtualMatrixANNComponent(name, 0, 0, 0),
    dimension(dimension), index(index) {
    setInputContiguousProperty(false);
  }
  
  SelectANNComponent::~SelectANNComponent() {
  }
  
  MatrixFloat *SelectANNComponent::
  privateDoForward(MatrixFloat* input_mat, bool during_training) {
    UNUSED_VARIABLE(during_training);
    if (input_mat->getNumDim() < 3)
      ERROR_EXIT2(128, "At least 3 dimensional matrix is expected, found %d. "
		  "First dimension is bunch-size, and the rest are pattern data "
		  "[%s]", input_mat->getNumDim(), name.c_str());
    AprilUtils::SharedPtr<MatrixFloat>
      aux_output_mat( input_mat->select(dimension+1, index) );
    MatrixFloat *output_mat = aux_output_mat->clone();
    return output_mat;
  }

  MatrixFloat *SelectANNComponent::
  privateDoBackprop(MatrixFloat *error_input_mat) {
    MatrixFloat *output_mat = getOutputMatrix();
    MatrixFloat *input_mat = getInputMatrix();
    if (!output_mat->sameDim(error_input_mat))
      ERROR_EXIT1(128, "Error input token has incorrect dimensions [%s]\n",
		  name.c_str());
    MatrixFloat *error_output_mat = input_mat->cloneOnlyDims();
    matZeros(error_output_mat);
    AprilUtils::SharedPtr<MatrixFloat>
      select_error_output_mat( error_output_mat->select(dimension+1, index) );
    matCopy(select_error_output_mat.get(), error_input_mat);
#ifdef USE_CUDA
    select_error_output_mat->setUseCuda(use_cuda);
#endif
    return error_output_mat;
  }
  
  void SelectANNComponent::privateReset(unsigned int it) {
    UNUSED_VARIABLE(it);
  }

  ANNComponent *SelectANNComponent::clone() {
    SelectANNComponent *select_component = new SelectANNComponent(dimension,
								  index,
								  name.c_str());
    return select_component;
  }
  
  void SelectANNComponent::build(unsigned int _input_size,
				 unsigned int _output_size,
				 AprilUtils::LuaTable &weights_dict,
				 AprilUtils::LuaTable &components_dict) {
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
