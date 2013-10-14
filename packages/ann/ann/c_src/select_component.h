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

#include "vector.h"
#include "ann_component.h"
#include "token_vector.h"
#include "token_matrix.h"

using april_utils::vector;

namespace ANN {

  /// This component has a dimension and index properties, and applies the
  /// matrix::select(dimension, index) method to the input matrix. The size and
  /// dimensions of the input matrix are not restricted, only the given
  /// dimension property and index must be valid at the input matrix.
  class SelectANNComponent : public ANNComponent {
    int dimension, index;
    TokenMatrixFloat *input, *output, *error_input, *error_output;
    
  public:
    SelectANNComponent(int dimension, int index, const char *name=0);
    virtual ~SelectANNComponent();
    
    virtual Token *getInput() { return input; }
    virtual Token *getOutput() { return output; }
    virtual Token *getErrorInput() { return error_input; }
    virtual Token *getErrorOutput() { return error_output; }
    
    virtual Token *doForward(Token* input, bool during_training);
    
    virtual Token *doBackprop(Token *input_error);
    
    virtual void reset();
    
    virtual ANNComponent *clone();

    virtual void build(unsigned int _input_size,
		       unsigned int _output_size,
		       hash<string,Connections*> &weights_dict,
		       hash<string,ANNComponent*> &components_dict);

    virtual char *toLuaString();
  };
}

#endif // SELECTCOMPONENT_H
