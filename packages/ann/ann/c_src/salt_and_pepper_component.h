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
#ifndef SALTANDPEPPERCOMPONENT_H
#define SALTANDPEPPERCOMPONENT_H

#include "ann_component.h"
#include "token_matrix.h"
#include "stochastic_component.h"

namespace ANN {
  
  /// This component adds to the input matrix Salt and Pepper noise, using a
  /// given random object, the noise probability (50% is this is for zero, and
  /// 50% for one), and the float values of zero and one. The matrix size and
  /// dimensions are not restricted.
  class SaltAndPepperANNComponent : public StochasticANNComponent {
    TokenMatrixFloat *input, *output;
    Token            *error_input, *error_output;
    float            zero, one;
    float            prob;
    
  public:
    SaltAndPepperANNComponent(MTRand *random, float zero, float one, float prob,
			      const char *name=0,
			      unsigned int size=0);
    virtual ~SaltAndPepperANNComponent();
    
    virtual Token *getInput() { return input; }
    virtual Token *getOutput() { return output; }
    virtual Token *getErrorInput() { return error_input; }
    virtual Token *getErrorOutput() { return error_output; }
    
    virtual Token *doForward(Token* input, bool during_training);
    
    virtual Token *doBackprop(Token *input_error);
    
    virtual void reset(unsigned int it=0);
    
    virtual ANNComponent *clone();

    virtual void build(unsigned int _input_size,
		       unsigned int _output_size,
		       hash<string,MatrixFloat*> &weights_dict,
		       hash<string,ANNComponent*> &components_dict);

    virtual char *toLuaString();
  };
}

#endif // SALTANDPEPPERCOMPONENT_H
