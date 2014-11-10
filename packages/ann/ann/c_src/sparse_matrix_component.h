/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2014, Francisco Zamora-Martinez
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
#ifndef SPARSEMATRIXANNCOMPONENT_H
#define SPARSEMATRIXANNCOMPONENT_H  

#include "cblas_headers.h"
#include "ann_component.h"
#include "component_properties.h"
#include "connection.h"
#include "matrixFloat.h"
#include "sparse_matrixFloat.h"
#include "token_matrix.h"
#include "token_sparse_matrix.h"

namespace ANN {

  /**
   * An abstract component which defines basic interface for components which
   * expect sparse matrix as input and produce dense matrix as output.
   */
  class VirtualSparseMatrixANNComponent : public ANNComponent,
                                          public ComponentPropertiesAndAsserts {
    APRIL_DISALLOW_COPY_AND_ASSIGN(VirtualSparseMatrixANNComponent);
    
    Basics::TokenSparseMatrixFloat *input, *error_output;
    Basics::TokenMatrixFloat *output, *error_input;
    
  protected:
    // Auxiliary methods
    
    Basics::SparseMatrixFloat *getInputMatrix() {
      return input->getMatrix();
    }
    Basics::MatrixFloat *getOutputMatrix() {
      return output->getMatrix();
    }
    Basics::MatrixFloat *getErrorInputMatrix() {
      return error_input->getMatrix();
    }
    Basics::SparseMatrixFloat *getErrorOutputMatrix() {
      return error_output->getMatrix();
    }
    
    // Abstract methods
    
    /**
     * Forward computation using SparseMatrixFloat input and MatrixFloat output
     *
     * @param input - The SparseMatrixFloat received as input.
     * @param during_training - Indicates if it is training or not.
     * @return A MatrixFloat with the forward computation result.
     */
    virtual Basics::MatrixFloat *privateDoForward(Basics::SparseMatrixFloat *input,
                                                  bool during_training) = 0;
    /**
     * Backprop computation using SparseMatrixFloat and MatrixFloat
     *
     * @param input_error - The MatrixFloat received as input_error.
     * @return A SparseMatrixFloat with the backprop computation result.
     */
    virtual Basics::SparseMatrixFloat *privateDoBackprop(Basics::MatrixFloat *input_error) = 0;
    
    /**
     * Reset of intermediate data
     *
     * @param it - Current iteration of optimization algorithm.
     */
    virtual void privateReset(unsigned int it=0) = 0;
    
    // virtual void computeGradients(const char *name, AprilUtils::LuaTable &weight_grads_dict) = 0;
    
  public:
    VirtualSparseMatrixANNComponent(const char *name, const char *weights_name,
                                    unsigned int input_size,
                                    unsigned int output_size);
    virtual ~VirtualSparseMatrixANNComponent();
    virtual Basics::Token *getInput() { return input; }
    virtual Basics::Token *getOutput() { return output; }
    virtual Basics::Token *getErrorInput() { return error_input; }
    virtual Basics::Token *getErrorOutput() { return error_output; }
    virtual Basics::Token *doForward(Basics::Token* input, bool during_training);
    virtual Basics::Token *doBackprop(Basics::Token *input_error);
    virtual void   reset(unsigned int it=0);
    
    // The following methods are not implemented, derived classes had to
    //
    // virtual ANNComponent *clone() = 0;
    /*
      virtual void build(unsigned int input_size,
      unsigned int output_size,
      AprilUtils::LuaTable &weights_dict,
      AprilUtils::LuaTable &components_dict) = 0;
    */
    // virtual void copyWeights(AprilUtils::LuaTable &weights_dict) = 0;
    // virtual char *toLuaString() = 0;
  };
  
}

#endif // SPARSEMATRIXANNCOMPONENT_H
