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
#ifndef ACTFCOMPONENT_H
#define ACTFCOMPONENT_H

#include "ann_component.h"
#include "MersenneTwister.h"
#include "smart_ptr.h"
#include "token_matrix.h"

namespace ANN {

  /**
   * @brief An abstract class that defines the basic interface that
   * the activation functions should fulfill.
   *
   * This class implements all methods of ANNComponent, and
   * declare two protected abstract methods: applyActivation() and
   * multiplyDerivatives().
   */
  class ActivationFunctionANNComponent : public ANNComponent {
    APRIL_DISALLOW_COPY_AND_ASSIGN(ActivationFunctionANNComponent);
    Basics::TokenMatrixFloat *input, *output, *error_input, *error_output;
    bool need_flatten;
    AprilUtils::SharedPtr<Basics::MatrixFloat> flat_input_mat;
    AprilUtils::SharedPtr<Basics::MatrixFloat> flat_output_mat;
    AprilUtils::SharedPtr<Basics::MatrixFloat> flat_error_input_mat;
    AprilUtils::SharedPtr<Basics::MatrixFloat> flat_error_output_mat;
  protected:
    /**
     * @brief This method receives two matrices, applies the activation over
     * the input and stores the result into the output matrix.
     *
     * @param[in] input_units - It is a matrix with the input.
     * @param[out] output_units - It is a matrix where the result must be
     *     stored. Its content should be overwritten.
     */
    virtual void applyActivation(Basics::MatrixFloat *input_units,
				 Basics::MatrixFloat *output_units) = 0;
    /**
     * @brief This method receives four matrices, two input/output with neuron
     * values before/after activation, and two input/output with
     * incoming/outgoing error deltas.
     *
     * @param[in] input_units - It is the matrix given as input in applyActivation.
     * @param[in] output_units - It is the matrix given as output at applyActivation.
     * @param[in] input_errors - It is another matrix with the incoming error deltas.
     * @param[out] output_errors -  It is a matrix where the outgoing error deltas
     * would be stored. Its content should be overwritten.
     */
    virtual void multiplyDerivatives(Basics::MatrixFloat *input_units,
				     Basics::MatrixFloat *output_units,
				     Basics::MatrixFloat *input_errors,
				     Basics::MatrixFloat *output_errors) = 0;
  public:
    ActivationFunctionANNComponent(const char *name=0, bool need_flatten=false);
    virtual ~ActivationFunctionANNComponent();
    
    virtual Basics::Token *getInput() { return input; }
    virtual Basics::Token *getOutput() { return output; }
    virtual Basics::Token *getErrorInput() { return error_input; }
    virtual Basics::Token *getErrorOutput() { return error_output; }
    //
    virtual void setInput(Basics::Token *tk) {
      AssignRef(input, tk->convertTo<Basics::TokenMatrixFloat*>());
    }
    virtual void setOutput(Basics::Token *tk) {
      AssignRef(output, tk->convertTo<Basics::TokenMatrixFloat*>());
    }
    virtual void setErrorInput(Basics::Token *tk) {
      AssignRef(error_input, tk->convertTo<Basics::TokenMatrixFloat*>());
    }
    virtual void setErrorOutput(Basics::Token *tk) {
      AssignRef(error_output, tk->convertTo<Basics::TokenMatrixFloat*>());
    }
    
    virtual Basics::Token *doForward(Basics::Token* input, bool during_training);
    
    virtual Basics::Token *doBackprop(Basics::Token *input_error);

    virtual void reset(unsigned int it=0);
    
    virtual void build(unsigned int _input_size,
		       unsigned int _output_size,
		       AprilUtils::LuaTable &weights_dict,
		       AprilUtils::LuaTable &components_dict);

    // virtual const char *luaCtorName() const;
    // virtual int exportParamsToLua(lua_State *L);
  };
}

#endif // ACTFCOMPONENT_H
