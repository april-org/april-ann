/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2015, Francisco Zamora-Martinez
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
#ifndef TRANSPOSECOMPONENT_H
#define TRANSPOSECOMPONENT_H

#include "unused_variable.h"
#include "vector.h"
#include "matrix_component.h"
#include "token_vector.h"
#include "token_matrix.h"

namespace ANN {

  /// This component modifies the input matrix to be reinterpreted as the given
  /// dimension sizes array. If the input matrix is not contiguous in memory, it
  /// will be cloned. If it is contiguous, the output of this component is a
  /// reinterpretation of input matrix, but the memory pointer will be shared.
  class TransposeANNComponent : public VirtualMatrixANNComponent {
    APRIL_DISALLOW_COPY_AND_ASSIGN(TransposeANNComponent);
    
    virtual Basics::MatrixFloat *privateDoForward(Basics::MatrixFloat* input,
                                                  bool during_training);
    
    virtual Basics::MatrixFloat *privateDoBackprop(Basics::MatrixFloat *input_error);
    
    virtual void privateReset(unsigned int it=0);
    
    Basics::MatrixFloat *transposeBunch(Basics::MatrixFloat *input) const;
    
    AprilUtils::UniquePtr<int[]> which;
  public:
    TransposeANNComponent(const int *which=0,
                          const char *name=0);
    virtual ~TransposeANNComponent();
    
    virtual ANNComponent *clone();

    virtual char *toLuaString();
    
    virtual const char *luaCtorName() const;
    virtual int exportParamsToLua(lua_State *L);
  };
}

#endif // TRANSPOSECOMPONENT_H
