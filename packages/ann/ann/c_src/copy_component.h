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
#ifndef COPYCOMPONENT_H
#define COPYCOMPONENT_H

#include "vector.h"
#include "ann_component.h"
#include "token_vector.h"

namespace ANN {
  
  /// This component replicates its input a given number of times.
  class CopyANNComponent : public ANNComponent {
    APRIL_DISALLOW_COPY_AND_ASSIGN(CopyANNComponent);
    
    april_utils::vector<ANNComponent*> components;
    // Token pointers which contains exactly the same that was received
    basics::Token *input, *error_output;
    
    // These token are always a TokenBunchVector
    basics::TokenBunchVector *output, *error_input;
    
    unsigned int times;
    
  public:
    CopyANNComponent(unsigned int times, const char *name=0,
		     unsigned int input_size=0,
		     unsigned int output_size=0);
    virtual ~CopyANNComponent();
    
    virtual basics::Token *getInput() { return input; }
    virtual basics::Token *getOutput() { return output; }
    virtual basics::Token *getErrorInput() { return error_input; }
    virtual basics::Token *getErrorOutput() { return error_output; }
    
    virtual basics::Token *doForward(basics::Token* input, bool during_training);
    
    virtual basics::Token *doBackprop(basics::Token *input_error);
    
    virtual void reset(unsigned int it=0);
    
    virtual ANNComponent *clone();

    virtual void build(unsigned int _input_size,
		       unsigned int _output_size,
		       basics::MatrixFloatSet *weights_dict,
		       april_utils::hash<april_utils::string,ANNComponent*> &components_dict);

    virtual char *toLuaString();
  };
}

#endif // COPYCOMPONENT_H
