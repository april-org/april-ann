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
#include "matrix_ext.h"
#include "mse_loss_function.h"
#include "loss_kernels.h"
#include "token_matrix.h"
#include "token_sparse_matrix.h"

using namespace AprilMath::MatrixExt::BLAS;
using namespace AprilMath::MatrixExt::Misc;
using namespace AprilUtils;
using namespace Basics;

namespace ANN {

  MSELossFunction::MSELossFunction(unsigned int size) :
    LossFunction(size) {
  }
  
  MSELossFunction::~MSELossFunction() {
  }

  MatrixFloat *MSELossFunction::computeLossBunch(Token *input, Token *target) {
    MatrixFloat *loss_output = 0;
    switch(target->getTokenCode()) {
    case table_of_token_codes::token_matrix:
      {
        MatrixFloat *input_mat, *target_mat;
        throwErrorAndGetMatrixFromTokens(input, target, input_mat, target_mat);
        int dim = input_mat->getDimSize(0);
        loss_output = new MatrixFloat(1, &dim);
#ifdef USE_CUDA
        loss_output->setUseCuda(input_mat->getCudaFlag());
#endif
        AprilMath::MatrixExt::LossOperations::
          matMSE(loss_output, input_mat, target_mat);
      }
      break;
    case table_of_token_codes::token_sparse_matrix:
      {
        TokenMatrixFloat *input_mat_token;
        TokenSparseMatrixFloat *target_mat_token;
        input_mat_token  = input->convertToAndCheck<TokenMatrixFloat*>();
        target_mat_token = target->convertToAndCheck<TokenSparseMatrixFloat*>();
        MatrixFloat *input_mat = input_mat_token->getMatrix();
        SparseMatrixFloat *target_mat = target_mat_token->getMatrix();
        // TODO: add error control
        int dim = input_mat->getDimSize(0);
        loss_output = new MatrixFloat(1, &dim);
#ifdef USE_CUDA
        loss_output->setUseCuda(input_mat->getCudaFlag());
#endif
        AprilMath::MatrixExt::LossOperations::
          matMSE(loss_output, input_mat, target_mat);
      } // case
      break;
    default:
      ERROR_EXIT(246, "Incorrect token type given as target\n");
    }
    return loss_output;
  }

  Token *MSELossFunction::computeGradient(Token *input, Token *target) {
    switch(target->getTokenCode()) {
    case table_of_token_codes::token_matrix:
      {
        MatrixFloat *input_mat, *target_mat;
        throwErrorAndGetMatrixFromTokens(input, target, input_mat, target_mat);
        MatrixFloat *error_mat = matSubstraction(input_mat, target_mat);
        TokenMatrixFloat *error_mat_token = new TokenMatrixFloat(error_mat);
        AssignRef<Token>(error_output, error_mat_token);
      }
      break;
    case table_of_token_codes::token_sparse_matrix:
      {
        TokenMatrixFloat *input_mat_token;
        TokenSparseMatrixFloat *target_mat_token;
        input_mat_token  = input->convertToAndCheck<TokenMatrixFloat*>();
        target_mat_token = target->convertToAndCheck<TokenSparseMatrixFloat*>();
        MatrixFloat *input_mat = input_mat_token->getMatrix();
        SparseMatrixFloat *target_mat = target_mat_token->getMatrix();
        MatrixFloat *error_mat = matAxpy(input_mat->clone(), -1.0f, target_mat);
        TokenMatrixFloat *error_mat_token = new TokenMatrixFloat(error_mat);
        AssignRef<Token>(error_output, error_mat_token);
      }
      break;
    default:
      ERROR_EXIT(246, "Incorrect token type given as target\n");
    }
    return error_output;
  }

  char *MSELossFunction::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.loss.mse(%d)", size);
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
}
