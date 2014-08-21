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
#ifndef DOTPRODUCTANNCOMPONENT_H
#define DOTPRODUCTANNCOMPONENT_H  

#include "token_matrix.h"
#include "cblas_headers.h"
#include "matrix_input_switch_component.h"
#include "connection.h"

namespace ANN {
  
  /// This components computes, for every I output neuron, the dot product
  /// between input neurons and the weights of the neuron I.
  class DotProductANNComponent : public MatrixInputSwitchANNComponent {
    APRIL_DISALLOW_COPY_AND_ASSIGN(DotProductANNComponent);
    
    basics::MatrixFloat *weights_matrix;
    
    /// learning parameters
    CBLAS_TRANSPOSE transpose_weights;
    
  protected:
    
    // from MatrixANNComponentHelper
    virtual basics::MatrixFloat *privateDoDenseForward(basics::MatrixFloat *input,
                                                       bool during_training);
    virtual basics::MatrixFloat *privateDoDenseBackprop(basics::MatrixFloat *error_input);
    virtual void privateDenseReset(unsigned int it=0);
    virtual void privateDenseComputeGradients(april_utils::SharedPtr<basics::MatrixFloat> & grads_mat);

    // from SparseMatrixANNComponentHelper
    virtual basics::MatrixFloat *privateDoSparseForward(basics::SparseMatrixFloat *input,
                                                        bool during_training);
    virtual basics::SparseMatrixFloat *privateDoSparseBackprop(basics::MatrixFloat *error_input);
    virtual void privateSparseReset(unsigned int it=0);
    virtual void privateSparseComputeGradients(april_utils::SharedPtr<basics::MatrixFloat> & grads_mat);
    
    //
    void initializeComputeGradients(april_utils::SharedPtr<basics::MatrixFloat> & grads_mat);
        
  public:
    DotProductANNComponent(const char *name=0, const char *weights_name=0,
			   unsigned int input_size  = 0,
			   unsigned int output_size = 0,
			   bool transpose_weights   = false);
    virtual ~DotProductANNComponent();
    virtual ANNComponent *clone();
    virtual void build(unsigned int input_size,
		       unsigned int output_size,
		       basics::MatrixFloatSet *weights_dict,
		       april_utils::hash<april_utils::string,ANNComponent*> &components_dict);
    virtual void copyWeights(basics::MatrixFloatSet *weights_dict);
    
    virtual char *toLuaString();
    
    bool transposed() { return transpose_weights == CblasTrans; }
  };
}

#endif // DOTPRODUCTANNCOMPONENT_H
