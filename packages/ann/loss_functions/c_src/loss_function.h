/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Francisco Zamora-Martinez
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
#ifndef LOSSFUNCTION_H
#define LOSSFUNCTION_H

#include "referenced.h"
#include "token_base.h"
#include "token_matrix.h"
#include "matrixFloat.h"
#include "error_print.h"
#include "mean_deviation.h"

namespace ANN {
  /// An abstract class that defines the basic interface that
  /// the loss_functions must complain.
  class LossFunction : public Referenced {
    april_utils::RunningStat acc_loss;
  protected:
    Token *error_output;
    unsigned int size;
        
    void throwErrorAndGetMatrixFromTokens(Token *input, Token *target,
					  MatrixFloat *&input_mat,
					  MatrixFloat *&target_mat,
					  bool check_target_size=true) const {
      if (input->getTokenCode() != table_of_token_codes::token_matrix)
	ERROR_EXIT(128, "Incorrect input token type, expected token matrix\n");
      if (target->getTokenCode() != table_of_token_codes::token_matrix)
	ERROR_EXIT(128, "Incorrect target token type, expected token matrix\n");
      //
      TokenMatrixFloat *input_mat_token = input->convertTo<TokenMatrixFloat*>();
      TokenMatrixFloat *target_mat_token = target->convertTo<TokenMatrixFloat*>();
      if (check_target_size && input_mat_token->size()!=target_mat_token->size())
	ERROR_EXIT2(128, "Different token sizes found: input=%d vs target=%d\n",
		    input_mat_token->size(),
		    target_mat_token->size());
      //
      input_mat  = input_mat_token->getMatrix();
      target_mat = target_mat_token->getMatrix();
      //
      april_assert(input_mat->getNumDim() == 2);
      if (check_target_size) {
	april_assert(target_mat->getNumDim() == 2);
	april_assert(input_mat->sameDim(target_mat));
      }
      april_assert(input_mat->getIsContiguous());
      april_assert(target_mat->getIsContiguous());
      april_assert(input_mat->getMajorOrder() == CblasColMajor);
      april_assert(target_mat->getMajorOrder() == CblasColMajor);
      april_assert(size==0 || input_mat->getDimSize(1)==static_cast<int>(size));
    }
    
    // To be implemented by derived classes
    virtual MatrixFloat *computeLossBunch(Token *input, Token *target) = 0;
    ////////////////////////////////////////////////////////////////

    LossFunction(LossFunction *other) :
    acc_loss(other->acc_loss),
    error_output(0),
    size(other->size) {
    }
    
  public:
    LossFunction(unsigned int size) :
    Referenced(), error_output(0), size(size) {
    }
    virtual ~LossFunction() {
      if (error_output) DecRef(error_output);
    }
    virtual float getAccumLoss() {
      return static_cast<float>(acc_loss.Mean());
    }
    virtual float getAccumLossVariance() {
      return static_cast<float>(acc_loss.Variance());
    }
    virtual void reset() {
      if (error_output) DecRef(error_output);
      error_output = 0;
      acc_loss.Clear();
    }
    virtual MatrixFloat *addLoss(Token *input, Token *target) {
      MatrixFloat *loss_data = computeLossBunch(input, target);
      april_assert(loss_data->getNumDim() == 1);
      for (MatrixFloat::iterator it(loss_data->begin());
	   it!=loss_data->end(); ++it)
	acc_loss.Push(static_cast<double>(*it));
      return loss_data;
    }
    // To be implemented by derived classes
    virtual Token *computeGradient(Token *input, Token *target) = 0;
    virtual LossFunction *clone() = 0;
    /////////////////////////////////////////////////////////////////
  };
}

#endif // LOSSFUNCTION_H
