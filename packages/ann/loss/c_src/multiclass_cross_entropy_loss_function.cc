/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera, Adrian Palacios, Francisco Zamora-Martinez
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
#include "token_matrix.h"
#include "multiclass_cross_entropy_loss_function.h"
#include "wrapper.h"

using namespace april_utils;
using namespace basics;

namespace ANN {

  MultiClassCrossEntropyLossFunction::MultiClassCrossEntropyLossFunction(unsigned int size) :
    LossFunction(size) {
    if (size > 0 && size < 3)
      ERROR_EXIT(128,
		 "Multi class cross entropy is only allowed for multi-class "
		 "problems (three or more output log softmax neurons). "
		 "Use cross entropy instead.\n");
  }
  
  MultiClassCrossEntropyLossFunction::~MultiClassCrossEntropyLossFunction() {
  }

  MatrixFloat *MultiClassCrossEntropyLossFunction::computeLossBunch(Token *input,
								    Token *target) {
    MatrixFloat *input_mat, *target_mat;
    throwErrorAndGetMatrixFromTokens(input, target, input_mat, target_mat);
    int dim = input_mat->getDimSize(0);
    MatrixFloat *loss_output = new MatrixFloat(1, &dim, CblasColMajor);
#ifdef USE_CUDA
    loss_output->setUseCuda(input_mat->getCudaFlag());
#endif
    MatrixFloat *t_logp = target_mat->clone();
    IncRef(t_logp);
    t_logp->cmul(input_mat);
    t_logp->sum(1, loss_output);
    DecRef(t_logp);
    loss_output->scal(-1.0);
    /*
      doMultiClassCrossEntropyLossFunction(input_mat->getRawDataAccess(),
      target_mat->getRawDataAccess(),
      loss_output->getRawDataAccess(),
      NEAR_ZERO,
      input_mat->getDimSize(1),
      input_mat->getDimSize(0),
      input_mat->getCudaFlag());
    */
    return loss_output;
  }

  Token *MultiClassCrossEntropyLossFunction::computeGradient(Token *input, Token *target) {
    MatrixFloat *input_mat, *target_mat;
    throwErrorAndGetMatrixFromTokens(input, target, input_mat, target_mat);
    MatrixFloat *error_mat = input_mat->clone(); //cloneOnlyDims();
    TokenMatrixFloat *error_mat_token = new TokenMatrixFloat(error_mat);
    AssignRef<Token>(error_output, error_mat_token);
    error_mat->clamp(logf(NEAR_ZERO), logf(1.0f - NEAR_ZERO));
    error_mat->exp();
    error_mat->axpy(-1.0f, target_mat);
    /*
      doComputeCrossEntropyGradient(input_mat->getRawDataAccess(),
      target_mat->getRawDataAccess(),
      error_mat->getRawDataAccess(),
      NEAR_ZERO,
      input_mat->getDimSize(1),
      input_mat->getDimSize(0),
      input_mat->getCudaFlag());
    */
    return error_output;
  }

  char *MultiClassCrossEntropyLossFunction::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.loss.multi_class_cross_entropy(%d)", size);
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }  

}
