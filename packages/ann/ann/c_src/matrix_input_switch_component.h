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

#include "ann_component.h"
#include "april_assert.h"
#include "cblas_headers.h"
#include "component_properties.h"
#include "sparse_matrixFloat.h"
#include "token_matrix.h"
#include "token_sparse_matrix.h"

namespace ANN {

  /**
   * An abstract component which defines basic interface for components which
   * receives sparse or dense matrix and produce dense matrix.
   */
  class MatrixInputSwitchANNComponent : public ANNComponent,
                                        public ComponentPropertiesAndAsserts {
    APRIL_DISALLOW_COPY_AND_ASSIGN(MatrixInputSwitchANNComponent);
    
    Basics::TokenMatrixFloat *input, *output, *error_input, *error_output;
    Basics::TokenSparseMatrixFloat *sparse_input, *sparse_error_output;
    bool is_sparse_input;
    
    // Auxiliary methods
    Basics::Token *doDenseForward(Basics::Token *_input, bool during_training);
    Basics::Token *doSparseForward(Basics::Token *_input, bool during_training);
    Basics::Token *doDenseBackprop(Basics::Token *_error_input);
    Basics::Token *doSparseBackprop(Basics::Token *_error_input);

  protected:

    // Auxiliary methods
    
    Basics::MatrixFloat *getInputMatrix() {
      april_assert(!is_sparse_input);
      return input->getMatrix();
    }
    Basics::SparseMatrixFloat *getSparseInputMatrix() {
      april_assert(is_sparse_input);
      return sparse_input->getMatrix();
    }
    Basics::MatrixFloat *getOutputMatrix() {
      return output->getMatrix();
    }
    Basics::MatrixFloat *getErrorInputMatrix() {
      return error_input->getMatrix();
    }
    Basics::MatrixFloat *getErrorOutputMatrix() {
      april_assert(!is_sparse_input);
      return error_output->getMatrix();
    }
    Basics::SparseMatrixFloat *getSparseErrorOutputMatrix() {
      april_assert(is_sparse_input);
      return sparse_error_output->getMatrix();
    }
    
    virtual void computeGradients(const char *name, AprilUtils::LuaTable &weight_grads_dict);

    // Abstract methods

    virtual Basics::MatrixFloat *privateDoDenseForward(Basics::MatrixFloat *input, bool during_training) = 0;
    virtual Basics::MatrixFloat *privateDoDenseBackprop(Basics::MatrixFloat *input_error) = 0;
    virtual void privateDenseReset(unsigned int it=0) = 0;
    virtual void privateDenseComputeGradients(const char *name,
                                              AprilUtils::LuaTable &weight_grads_dict) = 0;

    virtual Basics::MatrixFloat *privateDoSparseForward(Basics::SparseMatrixFloat *input, bool during_training) = 0;
    virtual Basics::SparseMatrixFloat *privateDoSparseBackprop(Basics::MatrixFloat *input_error) = 0;
    virtual void privateSparseReset(unsigned int it=0) = 0;
    virtual void privateSparseComputeGradients(const char *name,
                                               AprilUtils::LuaTable &weight_grads_dict) = 0;

  public:
    MatrixInputSwitchANNComponent(const char *name, const char *weights_name,
                                  unsigned int input_size,
                                  unsigned int output_size);
    virtual ~MatrixInputSwitchANNComponent();
    virtual Basics::Token *getInput() {
      if (!is_sparse_input) return input;
      else return sparse_input;
    }
    virtual Basics::Token *getOutput() {
      return output;
    }
    virtual Basics::Token *getErrorInput() {
      return error_input;
    }
    virtual Basics::Token *getErrorOutput() {
      if (!is_sparse_input) return error_output;
      else return sparse_error_output;
    }
    //
    virtual void setInput(Basics::Token *tk) {
      switch(tk->getTokenCode()) {
      case Basics::table_of_token_codes::token_matrix:
        AssignRef(input, tk->convertTo<Basics::TokenMatrixFloat*>());
        is_sparse_input = false;
        break;
      case Basics::table_of_token_codes::token_sparse_matrix:
        AssignRef(sparse_input, tk->convertTo<Basics::TokenSparseMatrixFloat*>());
        is_sparse_input = true;
        break;
      default:
        AssignRef(input, (Basics::TokenMatrixFloat*)0);
        AssignRef(sparse_input, (Basics::TokenSparseMatrixFloat*)0);
      }
    }
    virtual void setOutput(Basics::Token *tk) {
      AssignRef(output, tk->convertTo<Basics::TokenMatrixFloat*>());
    }
    virtual void setErrorInput(Basics::Token *tk) {
      AssignRef(error_input, tk->convertTo<Basics::TokenMatrixFloat*>());
    }
    virtual void setErrorOutput(Basics::Token *tk) {
      switch(tk->getTokenCode()) {
      case Basics::table_of_token_codes::token_matrix:
        AssignRef(error_output, tk->convertTo<Basics::TokenMatrixFloat*>());
        is_sparse_input = false;
        break;
      case Basics::table_of_token_codes::token_sparse_matrix:
        AssignRef(sparse_error_output,
                  tk->convertTo<Basics::TokenSparseMatrixFloat*>());
        is_sparse_input = true;
        break;
      default:
        AssignRef(error_output, (Basics::TokenMatrixFloat*)0);
        AssignRef(sparse_error_output, (Basics::TokenSparseMatrixFloat*)0);
      }
    }
    
    virtual Basics::Token *doForward(Basics::Token* input, bool during_training);
    virtual Basics::Token *doBackprop(Basics::Token *input_error);
    virtual void   reset(unsigned int it=0);
    //
    
    // The following methods are not implemented, derived classes had to
    //
    // virtual ANNComponent *clone(AprilUtils::LuaTable &copies) = 0;
    /*
      virtual void build(unsigned int input_size,
      unsigned int output_size,
      AprilUtils::LuaTable &weights_dict,
      AprilUtils::LuaTable &components_dict) = 0;
    */
    // virtual void copyWeights(AprilUtils::LuaTable &weights_dict) = 0;
    
    // virtual const char *luaCtorName() const;
    // virtual int exportParamsToLua(lua_State *L);
  };
}

#endif // MATRIXINPUTSWITCHANNCOMPONENT_H
