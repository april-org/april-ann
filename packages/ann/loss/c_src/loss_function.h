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
    basics::Token *error_output;
    unsigned int size;
        
    void throwErrorAndGetMatrixFromTokens(basics::Token *input,
                                          basics::Token *target,
					  basics::MatrixFloat *&input_mat,
					  basics::MatrixFloat *&target_mat,
					  bool check_target_size=true) const {
      if (input->getTokenCode() != basics::table_of_token_codes::token_matrix)
	ERROR_EXIT(128, "Incorrect input token type, expected token matrix\n");
      if (target->getTokenCode() != basics::table_of_token_codes::token_matrix)
	ERROR_EXIT(128, "Incorrect target token type, expected token matrix\n");
      //
      basics::TokenMatrixFloat *input_mat_token = input->convertTo<basics::TokenMatrixFloat*>();
      basics::TokenMatrixFloat *target_mat_token = target->convertTo<basics::TokenMatrixFloat*>();
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
      if (!input_mat->getIsContiguous())
	ERROR_EXIT(128, "Needs a contiguous input matrix\n");
      if (!target_mat->getIsContiguous())
	ERROR_EXIT(128, "Needs a contiguous target matrix\n");
      april_assert(input_mat->getMajorOrder() == CblasColMajor);
      april_assert(target_mat->getMajorOrder() == CblasColMajor);
      april_assert(size==0 || input_mat->getDimSize(1)==static_cast<int>(size));
      //
#ifdef USE_CUDA
      target_mat->setUseCuda(input_mat->getCudaFlag());
#endif
    }
    
    // To be implemented by derived classes
    virtual basics::MatrixFloat *computeLossBunch(basics::Token *input,
                                                  basics::Token *target) = 0;
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
    virtual basics::MatrixFloat *accumLoss(basics::MatrixFloat *loss_data) {
      april_assert(loss_data->getNumDim() == 1);
      for (basics::MatrixFloat::iterator it(loss_data->begin());
	   it!=loss_data->end(); ++it)
	acc_loss.Push(static_cast<double>(*it));
      return loss_data;
    }
    virtual basics::MatrixFloat *computeLoss(basics::Token *input,
                                             basics::Token *target) {
      basics::MatrixFloat *loss_data = computeLossBunch(input, target);
      april_assert(loss_data==0 || loss_data->getNumDim() == 1);
      return loss_data;
    }
    // To be implemented by derived classes
    virtual basics::Token *computeGradient(basics::Token *input,
                                           basics::Token *target) = 0;
    virtual LossFunction *clone() = 0;
    virtual char *toLuaString() = 0;
    /////////////////////////////////////////////////////////////////
  };
}

#endif // LOSSFUNCTION_H
