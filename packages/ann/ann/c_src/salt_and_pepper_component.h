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
    APRIL_DISALLOW_COPY_AND_ASSIGN(SaltAndPepperANNComponent);
    
    Basics::TokenMatrixFloat *input, *output;
    Basics::Token            *error_input, *error_output;
    float            zero, one;
    float            prob;
    
  public:
    SaltAndPepperANNComponent(Basics::MTRand *random, float zero,
                              float one, float prob,
			      const char *name=0,
			      unsigned int size=0);
    virtual ~SaltAndPepperANNComponent();
    
    virtual Basics::Token *getInput() { return input; }
    virtual Basics::Token *getOutput() { return output; }
    virtual Basics::Token *getErrorInput() { return error_input; }
    virtual Basics::Token *getErrorOutput() { return error_output; }
    //
    virtual void setInput(Basics::Token *tk) {
      AssignRef(input, tk->convertTo<Basics::TokenMatrixFloat*>());
    }
    virtual void setOutput(Basics::Token *tk) {
      AssignRef(output, tk->convertTo<Basics::TokenMatrixFloat*>());
    }
    virtual void setErrorInput(Basics::Token *tk) {
      AssignRef(error_input, tk);
    }
    virtual void setErrorOutput(Basics::Token *tk) {
      AssignRef(error_output, tk);
    }
    
    virtual Basics::Token *doForward(Basics::Token* input, bool during_training);
    
    virtual Basics::Token *doBackprop(Basics::Token *input_error);
    
    virtual void reset(unsigned int it=0);
    
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

#endif // SALTANDPEPPERCOMPONENT_H
