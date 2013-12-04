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
#ifndef BIASANNCOMPONENT_H
#define BIASANNCOMPONENT_H  

#include "cblas_headers.h"
#include "ann_component.h"
#include "connection.h"
#include "token_matrix.h"

namespace ANN {

  /// A component which adds a bias to the given bi-dimensional input matrix.
  class BiasANNComponent : public ANNComponent {
    TokenMatrixFloat *input, *output, *error;
    MatrixFloat *bias_vector;
    
  protected:
    
    virtual void computeGradients(MatrixFloat*& grad_mat);
    
  public:
    BiasANNComponent(unsigned int size=0,
		     const char *name=0, const char *weights_name=0);
    virtual ~BiasANNComponent();
    virtual Token *getInput() { return input; }
    virtual Token *getOutput() { return output; }
    virtual Token *getErrorInput() { return error; }
    virtual Token *getErrorOutput() { return error; }
    virtual Token *doForward(Token* input, bool during_training);
    virtual Token *doBackprop(Token *input_error);
    virtual void   reset(unsigned int it=0);
    virtual ANNComponent *clone();
    virtual void build(unsigned int input_size,
		       unsigned int output_size,
		       hash<string,MatrixFloat*> &weights_dict,
		       hash<string,ANNComponent*> &components_dict);
    virtual void copyWeights(hash<string,MatrixFloat*> &weights_dict);
    
    virtual char *toLuaString();
  };
}

#endif // BIASANNCOMPONENT_H
