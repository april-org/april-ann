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
#ifndef HYPERPLANECOMPONENT_H
#define HYPERPLANECOMPONENT_H

#include "vector.h"
#include "ann_component.h"
#include "dot_product_component.h"
#include "bias_component.h"

namespace ANN {

  /// This component is an stack of DotProductANNComponent and BiasANNComponent,
  /// because this two components are used together in almost every ANN.
  class HyperplaneANNComponent : public ANNComponent {
    DotProductANNComponent *dot_product;
    BiasANNComponent       *bias;
    
    HyperplaneANNComponent(const char *name);
    
  public:
    HyperplaneANNComponent(const char *name,
			   const char *dot_product_name,
			   const char *bias_name,
			   const char *dot_product_weights_name,
			   const char *bias_weights_name,
			   unsigned int input_size=0,
			   unsigned int output_size=0,
			   bool transpose_weights=false);
    virtual ~HyperplaneANNComponent();

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
      dot_product->debugInfo();
      bias->debugInfo();
    }

    virtual char *toLuaString();
  };
}

#endif // HYPERPLANECOMPONENT_H
