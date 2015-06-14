/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera, Adrian Palacios Corella, Francisco
 * Zamora-Martinez
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
#ifndef MULANNCOMPONENT_H
#define MULANNCOMPONENT_H  

#include "cblas_headers.h"
#include "matrix_component.h"
#include "connection.h"
#include "token_matrix.h"

namespace ANN {

  /// A component which adds a mul to the given bi-dimensional input matrix.
  class MulANNComponent : public VirtualMatrixANNComponent {
    APRIL_DISALLOW_COPY_AND_ASSIGN(MulANNComponent);
    bool scalar;
    AprilUtils::SharedPtr<Basics::MatrixFloat> mul_vector;
    
  protected:
    
    virtual Basics::MatrixFloat *privateDoForward(Basics::MatrixFloat *input,
                                                  bool during_training);
    virtual Basics::MatrixFloat *privateDoBackprop(Basics::MatrixFloat *input_error);
    virtual void privateReset(unsigned int it=0);
    virtual void computeGradients(const char *name, AprilUtils::LuaTable &weight_grads_dict);
    
    void applyCmul(Basics::MatrixFloat *dest, Basics::MatrixFloat *w);
    
  public:
    MulANNComponent(unsigned int size=0, bool scalar=false,
                    const char *name=0, const char *weights_name=0,
                    Basics::MatrixFloat *matrix=0);
    virtual ~MulANNComponent();
    virtual ANNComponent *clone();
    virtual void build(unsigned int input_size,
		       unsigned int output_size,
		       AprilUtils::LuaTable &weights_dict,
		       AprilUtils::LuaTable &components_dict);
    virtual void copyWeights(AprilUtils::LuaTable &weights_dict);
    
    virtual const char *luaCtorName() const;
    virtual int exportParamsToLua(lua_State *L);
  };
  
}

#endif // MULANNCOMPONENT_H
