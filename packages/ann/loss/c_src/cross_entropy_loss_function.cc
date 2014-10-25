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
#include "cmath_overloads.h"
#include "cross_entropy_loss_function.h"
#include "loss_kernels.h"
#include "token_matrix.h"

using namespace AprilMath::MatrixExt::LossOperations;
using namespace AprilUtils;
using namespace Basics;

namespace ANN {

  CrossEntropyLossFunction::CrossEntropyLossFunction(unsigned int size) :
    LossFunction(size) {
  }
  
  CrossEntropyLossFunction::~CrossEntropyLossFunction() {
  }
  
  MatrixFloat *CrossEntropyLossFunction::computeLossBunch(Token *input,
							  Token *target) {
    MatrixFloat *input_mat, *target_mat;
    throwErrorAndGetMatrixFromTokens(input, target, input_mat, target_mat);
    int dim = input_mat->getDimSize(0);
    MatrixFloat *loss_output = new MatrixFloat(1, &dim);
#ifdef USE_CUDA
    loss_output->setUseCuda(input_mat->getCudaFlag());
#endif
    
    matCrossEntropy(loss_output, input_mat, target_mat, NEAR_ZERO);
    return loss_output;
  }

  Token *CrossEntropyLossFunction::computeGradient(Token *input, Token *target) {
    MatrixFloat *input_mat, *target_mat;
    throwErrorAndGetMatrixFromTokens(input, target, input_mat, target_mat);
    MatrixFloat *error_mat = input_mat->cloneOnlyDims();
    TokenMatrixFloat *error_mat_token = new TokenMatrixFloat(error_mat);
    AssignRef<Token>(error_output, error_mat_token);
    matCrossEntropyGradient(error_mat, input_mat, target_mat, NEAR_ZERO);
    return error_output;
  }
  
  float CrossEntropyLossFunction::getAccumLoss() {
    float ret = LossFunction::getAccumLoss();
    if (ret < 0)
      ERROR_EXIT(128, "Found negative loss, check if output is log_logistic\n");
    return ret;
  }

  char *CrossEntropyLossFunction::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.loss.cross_entropy(%d)", size);
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }

}
