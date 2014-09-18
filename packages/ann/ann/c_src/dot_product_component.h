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
    
    Basics::MatrixFloat *weights_matrix;
    
    /// learning parameters
    CBLAS_TRANSPOSE transpose_weights;
    
  protected:
    
    // from MatrixANNComponentHelper
    virtual Basics::MatrixFloat *privateDoDenseForward(Basics::MatrixFloat *input,
                                                       bool during_training);
    virtual Basics::MatrixFloat *privateDoDenseBackprop(Basics::MatrixFloat *error_input);
    virtual void privateDenseReset(unsigned int it=0);
    virtual void privateDenseComputeGradients(AprilUtils::SharedPtr<Basics::MatrixFloat> & grads_mat);

    // from SparseMatrixANNComponentHelper
    virtual Basics::MatrixFloat *privateDoSparseForward(Basics::SparseMatrixFloat *input,
                                                        bool during_training);
    virtual Basics::SparseMatrixFloat *privateDoSparseBackprop(Basics::MatrixFloat *error_input);
    virtual void privateSparseReset(unsigned int it=0);
    virtual void privateSparseComputeGradients(AprilUtils::SharedPtr<Basics::MatrixFloat> & grads_mat);
    
    //
    void initializeComputeGradients(AprilUtils::SharedPtr<Basics::MatrixFloat> & grads_mat);
        
  public:
    DotProductANNComponent(const char *name=0, const char *weights_name=0,
			   unsigned int input_size  = 0,
			   unsigned int output_size = 0,
			   bool transpose_weights   = false);
    virtual ~DotProductANNComponent();
    virtual ANNComponent *clone();
    virtual void build(unsigned int input_size,
		       unsigned int output_size,
		       Basics::MatrixFloatSet *weights_dict,
		       AprilUtils::hash<AprilUtils::string,ANNComponent*> &components_dict);
    virtual void copyWeights(Basics::MatrixFloatSet *weights_dict);
    
    virtual char *toLuaString();
    
    bool transposed() { return transpose_weights == CblasTrans; }
  };
}

#endif // DOTPRODUCTANNCOMPONENT_H
