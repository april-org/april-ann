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
#ifndef STACKCOMPONENT_H
#define STACKCOMPONENT_H

#include "vector.h"
#include "ann_component.h"

namespace ANN {

  class StackANNComponent : public ANNComponent {
    april_utils::vector<ANNComponent*> components;
  public:
    StackANNComponent(const char *name=0);
    virtual ~StackANNComponent();

    void pushComponent(ANNComponent *component);
    ANNComponent *topComponent();
    void popComponent();

    virtual Token *getInput();
    virtual Token *getOutput();
    virtual Token *getErrorInput();
    virtual Token *getErrorOutput();
    
    virtual Token *doForward(Token* input, bool during_training);

    virtual Token *doBackprop(Token *input_error);
    
    virtual void doUpdate();

    virtual void reset();
    
    virtual ANNComponent *clone();
    
    virtual void setUseCuda(bool v);
    
    virtual void setOption(const char *name, double value);

    virtual bool hasOption(const char *name);
    
    virtual double getOption(const char *name);
    
    virtual void build(unsigned int input_size,
		       unsigned int output_size,
		       hash<string,Connections*> &weights_dict,
		       hash<string,ANNComponent*> &components_dict);
    
    virtual void copyWeights(hash<string,Connections*> &weights_dict);

    virtual void copyComponents(hash<string,ANNComponent*> &components_dict);
    
    virtual ANNComponent *getComponent(string &name);
    virtual void resetConnections() {
      for (unsigned int i=0; i<components.size(); ++i)
	components[i]->resetConnections();
    }
    virtual void debugInfo() {
      ANNComponent::debugInfo();
      for (unsigned int i=0; i<components.size(); ++i)
	components[i]->debugInfo();
    }

    virtual char *toLuaString();
  };
}

#endif // STACKCOMPONENT_H
