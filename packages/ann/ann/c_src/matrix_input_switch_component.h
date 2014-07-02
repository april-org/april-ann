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
#ifndef MATRIXINPUTSWITCHANNCOMPONENT_H
#define MATRIXINPUTSWITCHANNCOMPONENT_H

#include "april_assert.h"
#include "cblas_headers.h"
#include "matrix_component.h"
#include "sparse_matrix_component.h"
#include "token_matrix.h"

namespace ANN {

  /**
   * An abstract component which defines basic interface for components which
   * receives sparse or dense matrix and produce dense matrix.
   */
  class MatrixInputSwitchANNComponent : public ANNComponent {
    TokenMatrixFloat *input, *output, *error_input, *error_output;
    TokenSparseMatrixFloat *sparse_input, *sparse_error_output;
    bool is_sparse_input;
    
    // Auxiliary methods
    Token *doDenseForward(Token *_input, bool during_training);
    Token *doSparseForward(Token *_input, bool during_training);
    Token *doDenseBackprop(Token *_error_input);
    Token *doSparseBackprop(Token *_error_input);

  protected:

    // Auxiliary methods
    
    MatrixFloat *getInputMatrix() {
      april_assert(!is_sparse_input);
      return input->getMatrix();
    }
    SparseMatrixFloat *getSparseInputMatrix() {
      april_assert(is_sparse_input);
      return sparse_input->getMatrix();
    }
    MatrixFloat *getOutputMatrix() {
      return output->getMatrix();
    }
    MatrixFloat *getErrorInputMatrix() {
      return error_input->getMatrix();
    }
    MatrixFloat *getErrorOutputMatrix() {
      april_assert(!is_sparse_input);
      return error_output->getMatrix();
    }
    SparseMatrixFloat *getSparseErrorOutputMatrix() {
      april_assert(is_sparse_input);
      return sparse_error_output->getMatrix();
    }
    
    virtual void computeGradients(MatrixFloat*& grads_mat);

    // Abstract methods

    virtual MatrixFloat *privateDoDenseForward(MatrixFloat *input, bool during_training) = 0;
    virtual MatrixFloat *privateDoDenseBackprop(MatrixFloat *input_error) = 0;
    virtual void privateDenseReset(unsigned int it=0) = 0;
    virtual void privateDenseComputeGradients(MatrixFloat*& grads_mat) = 0;

    virtual MatrixFloat *privateDoSparseForward(SparseMatrixFloat *input, bool during_training) = 0;
    virtual SparseMatrixFloat *privateDoSparseBackprop(MatrixFloat *input_error) = 0;
    virtual void privateSparseReset(unsigned int it=0) = 0;
    virtual void privateSparseComputeGradients(MatrixFloat*& grads_mat) = 0;

  public:
    MatrixInputSwitchANNComponent(const char *name, const char *weights_name,
                                  unsigned int input_size,
                                  unsigned int output_size);
    virtual ~MatrixInputSwitchANNComponent();
    virtual Token *getInput() {
      if (!is_sparse_input) return input;
      else return sparse_input;
    }
    virtual Token *getOutput() {
      return output;
    }
    virtual Token *getErrorInput() {
      return error_input;
    }
    virtual Token *getErrorOutput() {
      if (!is_sparse_input) return error_output;
      else return sparse_error_output;
    }
    
    virtual Token *doForward(Token* input, bool during_training);
    virtual Token *doBackprop(Token *input_error);
    virtual void   reset(unsigned int it=0);
    //
    
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

#endif // MATRIXINPUTSWITCHANNCOMPONENT_H
