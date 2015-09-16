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
#ifndef BATCHSTANDARDIZATIONANNCOMPONENT_H
#define BATCHSTANDARDIZATIONANNCOMPONENT_H  

#include "cblas_headers.h"
#include "matrix_component.h"
#include "connection.h"
#include "token_matrix.h"

namespace ANN {

  /**
   * @brief A component which performs standardization of its given input.
   *
   * The standardization uses vectors which are adapted during training and
   * fixed during evaluation phase.
   */
  class BatchStandardizationANNComponent : public VirtualMatrixANNComponent {
    APRIL_DISALLOW_COPY_AND_ASSIGN(BatchStandardizationANNComponent);
    float alpha, epsilon;
    /// Mean value of current batch.
    AprilUtils::SharedPtr<Basics::MatrixFloat> mean;
    /// Inverse standard deviation of current batch.
    AprilUtils::SharedPtr<Basics::MatrixFloat> inv_std;
    /// Running mean computed during training to be used at inference stage.
    AprilUtils::SharedPtr<Basics::MatrixFloat> running_mean;
    /// Running stddev computed during training to be used at inference stage.
    AprilUtils::SharedPtr<Basics::MatrixFloat> running_inv_std;
    /// Auxiliary matrix.
    AprilUtils::SharedPtr<Basics::MatrixFloat> centered;
    
  protected:
    
    virtual Basics::MatrixFloat *privateDoForward(Basics::MatrixFloat *input,
                                                  bool during_training);
    virtual Basics::MatrixFloat *privateDoBackprop(Basics::MatrixFloat *input_error);
    virtual void privateReset(unsigned int it=0);
    virtual void computeGradients(const char *name, AprilUtils::LuaTable &weight_grads_dict);
    
  public:
    BatchStandardizationANNComponent(float alpha=0.1f, float epsilon=1e-05f,
                                     unsigned int size=0,
                                     const char *name=0,
                                     Basics::MatrixFloat *mean=0,
                                     Basics::MatrixFloat *inv_std=0);
    virtual ~BatchStandardizationANNComponent();
    virtual ANNComponent *clone(AprilUtils::LuaTable &copies);
    virtual void build(unsigned int input_size,
		       unsigned int output_size,
		       AprilUtils::LuaTable &weights_dict,
		       AprilUtils::LuaTable &components_dict);
    
    virtual const char *luaCtorName() const;
    virtual int exportParamsToLua(lua_State *L);
  };
  
}

#endif // BATCHSTANDARDIZATIONANNCOMPONENT_H
