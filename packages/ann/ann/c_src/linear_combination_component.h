/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2015, Francisco Zamora-Martinez
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
#ifndef LINCOMBANNCOMPONENT_H
#define LINCOMBANNCOMPONENT_H  

#include "cblas_headers.h"
#include "connection.h"
#include "matrix_component.h"
#include "smart_ptr.h"
#include "token_matrix.h"

namespace ANN {
  
  /// This components computes, for every I output neuron, the dot product
  /// between input neurons and the weights of the neuron I.
  class LinearCombANNComponent : public VirtualMatrixANNComponent {
    APRIL_DISALLOW_COPY_AND_ASSIGN(LinearCombANNComponent);
    
    AprilUtils::SharedPtr<Basics::MatrixFloat> weights_matrix,
      normalized_weights_mat;
    
  protected:
    
    // from MatrixANNComponentHelper
    virtual Basics::MatrixFloat *privateDoForward(Basics::MatrixFloat *input,
                                                  bool during_training);
    virtual Basics::MatrixFloat *privateDoBackprop(Basics::MatrixFloat *error_input);
    virtual void privateReset(unsigned int it=0);
    virtual void computeGradients(const char *name, AprilUtils::LuaTable &weight_grads_dict);
    Basics::MatrixFloat *initializeComputeGradients(const char *name,
                                                    AprilUtils::LuaTable &grads_mat_dict);
  
  public:
    LinearCombANNComponent(const char *name=0, const char *weights_name=0,
			   unsigned int input_size  = 0,
			   unsigned int output_size = 0);
    virtual ~LinearCombANNComponent();
    virtual ANNComponent *clone();
    virtual void build(unsigned int input_size,
		       unsigned int output_size,
		       AprilUtils::LuaTable &weights_dict,
		       AprilUtils::LuaTable &components_dict);
    virtual void copyWeights(AprilUtils::LuaTable &weights_dict);
    
    virtual char *toLuaString();
    
  };
}

#endif // LINCOMBANNCOMPONENT_H
