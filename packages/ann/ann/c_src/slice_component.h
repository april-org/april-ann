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
#ifndef SLICECOMPONENT_H
#define SLICECOMPONENT_H

#include "vector.h"
#include "matrix_component.h"
#include "token_vector.h"
#include "token_matrix.h"

namespace ANN {

  /// This component modifies the input matrix to be reinterpreted as the given
  /// dimension sizes array. If the input matrix is not contiguous in memory, it
  /// will be cloned. If it is contiguous, the output of this component is a
  /// reinterpretation of input matrix, but the memory pointer will be shared.
  class SliceANNComponent : public VirtualMatrixANNComponent {
    APRIL_DISALLOW_COPY_AND_ASSIGN(SliceANNComponent);
    
    int *slice_offset, *slice_size, n;
    
    virtual Basics::MatrixFloat *privateDoForward(Basics::MatrixFloat* input,
                                                  bool during_training);
    
    virtual Basics::MatrixFloat *privateDoBackprop(Basics::MatrixFloat *input_error);
    
    virtual void privateReset(unsigned int it=0);


  public:
    SliceANNComponent(const int *slice_offset,
		      const int *slice_size,
		      int n, const char *name=0);
    virtual ~SliceANNComponent();
    
    virtual ANNComponent *clone();

    virtual void build(unsigned int _input_size,
		       unsigned int _output_size,
		       Basics::MatrixFloatSet *weights_dict,
		       AprilUtils::hash<AprilUtils::string,ANNComponent*> &components_dict);

    virtual char *toLuaString();
  };
}

#endif // SLICECOMPONENT_H
