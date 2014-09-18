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
#include "mae_loss_function.h"
#include "matrix_operations.h"
#include "loss_kernels.h"
#include "token_matrix.h"

using namespace AprilMath::MatrixExt::LossOperations;
using namespace AprilMath::MatrixExt::Operations;
using namespace AprilUtils;
using namespace Basics;

namespace ANN {

  MAELossFunction::MAELossFunction(unsigned int size) :
    LossFunction(size) {
  }
  
  MAELossFunction::~MAELossFunction() {
  }
  
  MatrixFloat *MAELossFunction::computeLossBunch(Token *input, Token *target) {
    MatrixFloat *input_mat, *target_mat;
    throwErrorAndGetMatrixFromTokens(input, target, input_mat, target_mat);
    int dim = input_mat->getDimSize(0);
    MatrixFloat *loss_output = new MatrixFloat(1, &dim, CblasColMajor);
#ifdef USE_CUDA
    loss_output->setUseCuda(input_mat->getCudaFlag());
#endif
    const int N = input_mat->size() / input_mat->getDimSize(0);
    AprilUtils::SharedPtr<MatrixFloat>
      aux_output( matSubstraction(input_mat, target_mat) );
    matAbs(aux_output.get());
    matSum(aux_output.get(), 1, loss_output);
    if (N > 1) matScal(loss_output, 1.0f/N );
    return loss_output;
  }

  Token *MAELossFunction::computeGradient(Token *input, Token *target) {
    MatrixFloat *input_mat, *target_mat;
    throwErrorAndGetMatrixFromTokens(input, target, input_mat, target_mat);
    MatrixFloat *error_mat = input_mat->cloneOnlyDims();
    TokenMatrixFloat *error_mat_block = new TokenMatrixFloat(error_mat);
    AssignRef<Token>(error_output, error_mat_block);
    const int N = input_mat->size() / input_mat->getDimSize(0);
    matMAEGradient(error_mat, input_mat, target_mat, NEAR_ZERO, 1.0f/N);
    return error_output;
  }

  char *MAELossFunction::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.loss.mae(%d)", size);
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
  
}
