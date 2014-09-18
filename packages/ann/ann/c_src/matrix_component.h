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
#ifndef MATRIXANNCOMPONENT_H
#define MATRIXANNCOMPONENT_H  

#include "ann_component.h"
#include "cblas_headers.h"
#include "component_properties.h"
#include "connection.h"
#include "token_matrix.h"

namespace ANN {

  /**
   * An abstract component which defines basic interface for components which
   * expect matrices and produce matrices.
   */
  class VirtualMatrixANNComponent : public ANNComponent,
                                    public ComponentPropertiesAndAsserts {
    APRIL_DISALLOW_COPY_AND_ASSIGN(VirtualMatrixANNComponent);
    
    Basics::TokenMatrixFloat *input, *output, *error_input, *error_output;
    
  protected:
    
    // Auxiliary methods
    
    Basics::MatrixFloat *getInputMatrix() {
      return input->getMatrix();
    }
    Basics::MatrixFloat *getOutputMatrix() {
      return output->getMatrix();
    }
    Basics::MatrixFloat *getErrorInputMatrix() {
      return error_input->getMatrix();
    }
    Basics::MatrixFloat *getErrorOutputMatrix() {
      return error_output->getMatrix();
    }
    
    // Abstract methods
    
    /**
     * Forward computation using MatrixFloat input/output
     *
     * @param input - The MatrixFloat received as input.
     * @param during_training - Indicates if it is training or not.
     * @return A MatrixFloat with the forward computation result.
     */
    virtual Basics::MatrixFloat *privateDoForward(Basics::MatrixFloat *input,
                                                  bool during_training) = 0;
    /**
     * Backprop computation using MatrixFloat input/output
     *
     * @param input_error - The MatrixFloat received as input_error.
     * @return A MatrixFloat with the backprop computation result.
     */
    virtual Basics::MatrixFloat *privateDoBackprop(Basics::MatrixFloat *input_error) = 0;
    
    /**
     * Reset of intermediate data
     *
     * @param it - Current iteration of optimization algorithm.
     */
    virtual void privateReset(unsigned int it=0) = 0;
    
    // virtual void computeGradients(AprilUtils::SharedPtr<MatrixFloat> &grad_mat) = 0;
    
  public:
    VirtualMatrixANNComponent(const char *name, const char *weights_name,
                              unsigned int input_size, unsigned int output_size);
    virtual ~VirtualMatrixANNComponent();
    virtual Basics::Token *getInput() { return input; }
    virtual Basics::Token *getOutput() { return output; }
    virtual Basics::Token *getErrorInput() { return error_input; }
    virtual Basics::Token *getErrorOutput() { return error_output; }
    /**
     * If a TokenSparseMatrixFloat is given as input, it will be converted to
     * TokenMatrixFloat by calling toDense method of SparseMatrixFloat.
     */
    virtual Basics::Token *doForward(Basics::Token* input, bool during_training);
    virtual Basics::Token *doBackprop(Basics::Token *input_error);
    virtual void   reset(unsigned int it=0);
    
    // The following methods are not implemented, derived classes had to
    //
    // virtual ANNComponent *clone() = 0;
    /*
      virtual void build(unsigned int input_size,
      unsigned int output_size,
      MatrixFloatSet *weights_dict,
      hash<string,ANNComponent*> &components_dict) = 0;
    */
    // virtual void copyWeights(MatrixFloatSet *weights_dict) = 0;
    // virtual char *toLuaString() = 0;
  };
  
}

#endif // MATRIXANNCOMPONENT_H
