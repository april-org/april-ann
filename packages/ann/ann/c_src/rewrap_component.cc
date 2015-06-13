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
#include "rewrap_component.h"

using namespace Basics;
using namespace AprilUtils;
using namespace AprilMath;

namespace ANN {
  
  RewrapANNComponent::RewrapANNComponent(const int *rewrap_dims, int n,
					 const char *name) :
    VirtualMatrixANNComponent(name, 0,
                              mult(rewrap_dims,n), mult(rewrap_dims,n)),
    rewrap_dims(new int[n+1]), n(n+1) {
    setInputContiguousProperty(true);
    for (int i=1; i<this->n; ++i)
      this->rewrap_dims[i] = rewrap_dims[i-1];
  }
  
  RewrapANNComponent::~RewrapANNComponent() {
    delete[] rewrap_dims;
  }
  
  MatrixFloat *RewrapANNComponent::
  privateDoForward(MatrixFloat* input_mat, bool during_training) {
    UNUSED_VARIABLE(during_training);
    rewrap_dims[0] = input_mat->getDimSize(0);
    if (input_mat->getNumDim() < 2)
      ERROR_EXIT2(128, "At least 2-dimensional matrix is expected, found %d. "
		  "[%s]", input_mat->getNumDim(), name.c_str());
    MatrixFloat *output_mat = input_mat->rewrap(rewrap_dims, n);
    return output_mat;
  }

  MatrixFloat *RewrapANNComponent::
  privateDoBackprop(MatrixFloat *error_input_mat) {
    MatrixFloat *input_mat = getInputMatrix();
    MatrixFloat *error_output_mat;
    error_output_mat = error_input_mat->rewrap(input_mat->getDimPtr(),
					       input_mat->getNumDim());
    return error_output_mat;
  }
  
  void RewrapANNComponent::privateReset(unsigned int it) {
    UNUSED_VARIABLE(it);
  }

  ANNComponent *RewrapANNComponent::clone() {
    RewrapANNComponent *rewrap_component = new RewrapANNComponent(rewrap_dims+1,
								  n-1,
								  name.c_str());
    return rewrap_component;
  }
  
  void RewrapANNComponent::build(unsigned int _input_size,
				 unsigned int _output_size,
				 AprilUtils::LuaTable &weights_dict,
				 AprilUtils::LuaTable &components_dict) {
    unsigned int sz = 1;
    for (int i=1; i<this->n; ++i) sz *= rewrap_dims[i];
    //
    if (_output_size != 0 && _output_size != sz)
      ERROR_EXIT2(256, "Incorrect output size, expected %d, found %d\n",
		  sz, _output_size);
    if (_input_size != 0 && _input_size != sz)
      ERROR_EXIT2(256, "Incorrect input size, expected %d, found %d\n",
		  sz, _input_size);
    //
    ANNComponent::build(sz, sz, weights_dict, components_dict);
  }
  
  const char *RewrapANNComponent::luaCtorName() const {
    return "ann.components.rewrap";
  }
  int RewrapANNComponent::exportParamsToLua(lua_State *L) {
    AprilUtils::LuaTable t(L);
    AprilUtils::LuaTable size(L);
    t["name"] = name.c_str();
    t["size"] = size;
    for (int i=1; i<n; ++i) size[i] = rewrap_dims[i];
    t.pushTable(L);
    return 1;
  }
}
