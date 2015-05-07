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
#ifndef DROPOUTCOMPONENT_H
#define DROPOUTCOMPONENT_H

#include "ann_component.h"
#include "token_matrix.h"
#include "stochastic_component.h"

#define MASK_STR "mask"

namespace ANN {
  
  /// This component adds to the input matrix mask noise, using a given random
  /// object, the noise probability, and the float value for masked units. The
  /// matrix size and dimensions are not restricted.
  class DropoutANNComponent : public StochasticANNComponent {
    APRIL_DISALLOW_COPY_AND_ASSIGN(DropoutANNComponent);
    
    Basics::TokenMatrixFloat *input, *output;
    Basics::Token            *error_input, *error_output;
    Basics::MatrixFloat      *dropout_mask;
    float            value;
    float            prob;
    bool             normalize_after_training;
    unsigned int     size;
    
  public:
    DropoutANNComponent(Basics::MTRand *random, float value, float prob,
                        bool normalize_after_training=true,
			const char *name=0,
			unsigned int size=0);
    virtual ~DropoutANNComponent();
    
    virtual Basics::Token *getInput() { return input; }
    virtual Basics::Token *getOutput() { return output; }
    virtual Basics::Token *getErrorInput() { return error_input; }
    virtual Basics::Token *getErrorOutput() { return error_output; }
    
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
    
    virtual void copyState(AprilUtils::LuaTable &dict) {
      StochasticANNComponent::copyState(dict);
      AprilUtils::LuaTable state(dict.get<AprilUtils::LuaTable>(name));
      if (dropout_mask != 0) {
        state.put(MASK_STR, dropout_mask);
      }
    }

    virtual void setState(AprilUtils::LuaTable &dict) {
      StochasticANNComponent::setState(dict);
      AprilUtils::LuaTable state(dict.get<AprilUtils::LuaTable>(name));
      Basics::MatrixFloat *mask = state.opt<Basics::MatrixFloat*>(MASK_STR, 0);
      if (mask != 0) {
        AssignRef(dropout_mask, mask);
      }
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
  };
}

#endif // DROPOUTCOMPONENT_H
