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
#ifndef SELECTCOMPONENT_H
#define SELECTCOMPONENT_H

#include "matrix_component.h"
#include "token_matrix.h"

namespace ANN {

  /// This component has a dimension and index properties, and applies the
  /// matrix::select(dimension, index) method to the input matrix. The size and
  /// dimensions of the input matrix are not restricted, only the given
  /// dimension property and index must be valid at the input matrix.
  class SelectANNComponent : public VirtualMatrixANNComponent {
    APRIL_DISALLOW_COPY_AND_ASSIGN(SelectANNComponent);
    
    int dimension, index;
    
  protected:
    virtual Basics::MatrixFloat *privateDoForward(Basics::MatrixFloat* input,
                                                  bool during_training);
    
    virtual Basics::MatrixFloat *privateDoBackprop(Basics::MatrixFloat *input_error);
    
    virtual void privateReset(unsigned int it=0);
    
  public:
    SelectANNComponent(int dimension, int index, const char *name=0);
    virtual ~SelectANNComponent();
    
    virtual ANNComponent *clone();

    virtual void build(unsigned int _input_size,
		       unsigned int _output_size,
		       AprilUtils::LuaTable &weights_dict,
		       AprilUtils::LuaTable &components_dict);

    virtual char *toLuaString();

    virtual const char *luaCtorName() const;
    virtual int exportParamsToLua(lua_State *L);
  };
}

#endif // SELECTCOMPONENT_H
