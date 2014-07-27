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
#ifndef REWRAPCOMPONENT_H
#define REWRAPCOMPONENT_H

#include "unused_variable.h"
#include "vector.h"
#include "matrix_component.h"
#include "token_vector.h"
#include "token_matrix.h"

using april_utils::vector;

namespace ANN {

  /// This component modifies the input matrix to be reinterpreted as the given
  /// dimension sizes array. If the input matrix is not contiguous in memory, it
  /// will be cloned. If it is contiguous, the output of this component is a
  /// reinterpretation of input matrix, but the memory pointer will be shared.
  class RewrapANNComponent : public VirtualMatrixANNComponent {
    APRIL_DISALLOW_COPY_AND_ASSIGN(RewrapANNComponent);
    
    int *rewrap_dims, n;
    
    virtual MatrixFloat *privateDoForward(MatrixFloat* input,
                                          bool during_training);
    
    virtual MatrixFloat *privateDoBackprop(MatrixFloat *input_error);
    
    virtual void privateReset(unsigned int it=0);
    
  public:
    RewrapANNComponent(const int *rewrap_dims, int n, const char *name=0);
    virtual ~RewrapANNComponent();
    
    virtual void precomputeOutputSize(const vector<unsigned int> &input_size,
				      vector<unsigned int> &output_size) {
      UNUSED_VARIABLE(input_size);
      output_size.clear();
      for (int i=0; i<n-1; ++i)
	output_size.push_back(rewrap_dims[i+1]);
    }
    
    virtual ANNComponent *clone();

    virtual void build(unsigned int _input_size,
		       unsigned int _output_size,
		       MatrixFloatSet *weights_dict,
		       hash<string,ANNComponent*> &components_dict);

    virtual char *toLuaString();
  };
}

#endif // REWRAPCOMPONENT_H
