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
#ifndef FLATTENCOMPONENT_H
#define FLATTENCOMPONENT_H

#include "vector.h"
#include "ann_component.h"
#include "token_vector.h"
#include "token_matrix.h"

using april_utils::vector;

namespace ANN {

  /// This component receives a multidimensional input matrix and reinterprets
  /// its data like to be a bi-dimensional matrix (bunch size X number of
  /// neurons).
  class FlattenANNComponent : public ANNComponent {
    int flatten_dims[2];
    TokenMatrixFloat *input, *output, *error_input, *error_output;
    
  public:
    FlattenANNComponent(const char *name=0);
    virtual ~FlattenANNComponent();
    
    virtual Token *getInput() { return input; }
    virtual Token *getOutput() { return output; }
    virtual Token *getErrorInput() { return error_input; }
    virtual Token *getErrorOutput() { return error_output; }

    virtual void precomputeOutputSize(const vector<unsigned int> &input_size,
				      vector<unsigned int> &output_size) {
      unsigned int sz = 1;
      for (unsigned int i=0; i<input_size.size(); ++i)
	sz *= input_size[i];
      output_size.clear();
      output_size.push_back(sz);
    }
    
    virtual Token *doForward(Token* input, bool during_training);
    
    virtual Token *doBackprop(Token *input_error);
    
    virtual void reset(unsigned int it=0);
    
    virtual ANNComponent *clone();

    virtual void build(unsigned int _input_size,
		       unsigned int _output_size,
		       hash<string,Connections*> &weights_dict,
		       hash<string,ANNComponent*> &components_dict);

    virtual char *toLuaString();
  };
}

#endif // FLATTENCOMPONENT_H
