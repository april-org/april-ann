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

  /// ANN component for STACK several components. The output of one component is
  /// the input of the next. The input and output size of this component depends
  /// on the input of the first stacked component and the output of the last
  /// stacked component. If it is zero, then the stack accepts (or produces) a
  /// non determined number of neurons.
  class StackANNComponent : public ANNComponent {
    /// Vector with the stack
    april_utils::vector<ANNComponent*> components;

  public:
    StackANNComponent(const char *name=0);
    virtual ~StackANNComponent();

    /// Method to push a new component at the top of the stack
    void pushComponent(ANNComponent *component);
    /// Returns the component at the top of the stack
    ANNComponent *topComponent();
    /// Removes the component at the top of the stack
    void popComponent();

    virtual Token *getInput();
    virtual Token *getOutput();
    virtual Token *getErrorInput();
    virtual Token *getErrorOutput();
    
    virtual Token *doForward(Token* input, bool during_training);

    virtual Token *doBackprop(Token *input_error);
    
    virtual void reset(unsigned int it=0);
    
    virtual ANNComponent *clone();
    
    virtual void setUseCuda(bool v);
    
    virtual void build(unsigned int input_size,
		       unsigned int output_size,
		       hash<string,Connections*> &weights_dict,
		       hash<string,ANNComponent*> &components_dict);
    
    virtual void copyWeights(hash<string,Connections*> &weights_dict);

    virtual void copyComponents(hash<string,ANNComponent*> &components_dict);
    
    virtual ANNComponent *getComponent(string &name);
    virtual void computeAllGradients(hash<string,MatrixFloat*> &weight_grads_dict);
    virtual void debugInfo() {
      ANNComponent::debugInfo();
      for (unsigned int i=0; i<components.size(); ++i)
	components[i]->debugInfo();
    }

    virtual char *toLuaString();
    
    unsigned int size() { return components.size(); }
  };
}

#endif // STACKCOMPONENT_H
