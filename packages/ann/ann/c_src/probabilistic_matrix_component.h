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
#ifndef PROBABILISTICMATRIXANNCOMPONENT_H
#define PROBABILISTICMATRIXANNCOMPONENT_H  

#include "cblas_headers.h"
#include "connection.h"
#include "matrix_component.h"
#include "smart_ptr.h"
#include "token_matrix.h"

namespace ANN {
  
  /**
   * @brief This component computes multiplication by a probabilistic matrix.
   *
   * Weights are forced to sum 1 and to be in range [0,1]. To achieve this goal,
   * the component has a set of raw weights which are transformed by means of
   * softmax function. The transformed weights are multiplied by the given
   * input. The transformation can be LEFT (by columns) or RIGHT (by rows),
   * normalizing outgoing or incoming weights respectively.
   *
   * @note This component uses @c during_training flag at @c forward() method
   * to force weights normalization via softmax transformation.
   */
  class ProbabilisticMatrixANNComponent : public VirtualMatrixANNComponent {
  public:
    enum NormalizationSide { LEFT, RIGHT };
    
  private:
    APRIL_DISALLOW_COPY_AND_ASSIGN(ProbabilisticMatrixANNComponent);
    
    /// The raw weights of this component.
    AprilUtils::SharedPtr<Basics::MatrixFloat> weights_mat,
    /// Weights after softmax normalization
      normalized_weights_mat,
    /// Transposed version of raw weights, required for 'left' transformation
      T_weights_mat,
    /// Transposed version of normalized weights, for 'left' transformation
      T_normalized_weights_mat;
    
    bool needs_weights_normalization;
    NormalizationSide side;

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
    ProbabilisticMatrixANNComponent(NormalizationSide side,
                                    const char *name=0, const char *weights_name=0,
                                    unsigned int input_size  = 0,
                                    unsigned int output_size = 0,
                                    Basics::MatrixFloat *matrix = 0);
    virtual ~ProbabilisticMatrixANNComponent();
    virtual ANNComponent *clone();
    virtual void build(unsigned int input_size,
		       unsigned int output_size,
		       AprilUtils::LuaTable &weights_dict,
		       AprilUtils::LuaTable &components_dict);
    virtual void copyWeights(AprilUtils::LuaTable &weights_dict);
    
    Basics::MatrixFloat *getNormalizedWeights() {
      return normalized_weights_mat.get();
    }

    virtual const char *luaCtorName() const;
    virtual int exportParamsToLua(lua_State *L);
  };
}

#endif // PROBABILISTICMATRIXANNCOMPONENT_H
