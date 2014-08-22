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
#include "mse_loss_function.h"
#include "wrapper.h"

using namespace april_utils;
using namespace basics;

namespace ANN {

  MSELossFunction::MSELossFunction(unsigned int size) :
    LossFunction(size) {
  }
  
  MSELossFunction::~MSELossFunction() {
  }
  
  MatrixFloat *MSELossFunction::computeLossBunch(Token *input, Token *target) {
    MatrixFloat *input_mat, *target_mat;
    throwErrorAndGetMatrixFromTokens(input, target, input_mat, target_mat);
    int dim = input_mat->getDimSize(0);
    MatrixFloat *loss_output = new MatrixFloat(1, &dim, CblasColMajor);
#ifdef USE_CUDA
    loss_output->setUseCuda(input_mat->getCudaFlag());
#endif
    doMSELossFunction(input_mat->getRawDataAccess(),
                      target_mat->getRawDataAccess(),
                      loss_output->getRawDataAccess(),
                      0.0f,
                      input_mat->getDimSize(1),
                      input_mat->getDimSize(0),
                      input_mat->getCudaFlag());
    loss_output->scal(0.5f);
    return loss_output;
  }

  Token *MSELossFunction::computeGradient(Token *input, Token *target) {
    MatrixFloat *input_mat, *target_mat;
    throwErrorAndGetMatrixFromTokens(input, target, input_mat, target_mat);
    MatrixFloat *error_mat = input_mat->cloneOnlyDims();
    TokenMatrixFloat *error_mat_token = new TokenMatrixFloat(error_mat);
    AssignRef<Token>(error_output, error_mat_token);
    doComputeMSEGradient(input_mat->getRawDataAccess(),
			 target_mat->getRawDataAccess(),
			 error_mat->getRawDataAccess(),
			 0.0f,
			 input_mat->getDimSize(1),
			 input_mat->getDimSize(0),
			 input_mat->getCudaFlag());
    return error_output;
  }

  char *MSELossFunction::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.loss.mse(%d)", size);
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
}
