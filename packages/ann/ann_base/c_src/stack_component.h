/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
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
#ifndef STACKCOMPONENT_H
#define STACKCOMPONENT_H

#include "vector.h"
#include "ann_component.h"

using april_utils::vector;

namespace ANN {

  class StackANNComponent : public Referenced {
    vector<ANNComponent*> components;
  public:
    StackANNComponent(const char *name);
    virtual ~StackANNComponent();

    virtual const Token *getInput() const;
    virtual const Token *getOutput() const;
    virtual const Token *getErrorInput() const;
    virtual const Token *getErrorOutput() const;
    
    virtual Token *doForward(Token* input, bool during_training);

    virtual Token *doBackprop(Token *input_error);
    
    virtual void doUpdate();

    virtual void reset();
    
    virtual ANNComponent *clone();
    
    virtual void setOption(const char *name, double value);

    virtual bool hasOption(const char *name);
    
    virtual double getOption(const char *name);
    
    virtual void build(unsigned int input_size,
		       unsigned int output_size,
		       hash<string,Connections*> &weights_dict,
		       hash<string,ANNComponent*> &components_dict);
    
    virtual void copyWeights(hash<string,Connections*> &weights_dict);

    virtual void copyComponents(hash<string,ANNComponent*> &weights_dict);
    
    virtual ANNComponent *getComponent(string &name);

  };
}

#endif // STACKCOMPONENT_H
