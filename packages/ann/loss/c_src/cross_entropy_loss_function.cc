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
#include "token_sparse_matrix.h"

using namespace AprilMath::MatrixExt::LossOperations;
using namespace AprilMath::MatrixExt::Initializers;
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
        matCrossEntropy(loss_output, input_mat, target_mat, NEAR_ZERO);
      } // case
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
        matCrossEntropy(loss_output, input_mat, target_mat, NEAR_ZERO);
      } // case
      break;
    default:
      ERROR_EXIT(246, "Incorrect token type given as target\n");
    } // switch
    return loss_output;
  }

  Token *CrossEntropyLossFunction::computeGradient(Token *input, Token *target) {
    switch(target->getTokenCode()) {
    case table_of_token_codes::token_matrix:
      {
        MatrixFloat *input_mat, *target_mat;
        throwErrorAndGetMatrixFromTokens(input, target, input_mat, target_mat);
        MatrixFloat *error_mat = input_mat->cloneOnlyDims();
        TokenMatrixFloat *error_mat_token = new TokenMatrixFloat(error_mat);
        AssignRef<Token>(error_output, error_mat_token);
        matCrossEntropyGradient(error_mat, input_mat, target_mat, NEAR_ZERO);
        break;
      } // case
    case table_of_token_codes::token_sparse_matrix:
      {
        TokenMatrixFloat *input_mat_token;
        TokenSparseMatrixFloat *target_mat_token;
        input_mat_token  = input->convertToAndCheck<TokenMatrixFloat*>();
        target_mat_token = target->convertToAndCheck<TokenSparseMatrixFloat*>();
        MatrixFloat *input_mat = input_mat_token->getMatrix();
        SparseMatrixFloat *target_mat = target_mat_token->getMatrix();
        if (target_mat->getSparseFormat() != CSR_FORMAT) {
          ERROR_EXIT(256, "Needs a CSR sparse matrix\n");
        }
        // TODO: add more error control
        MatrixFloat *error_mat = input_mat->cloneOnlyDims();
        TokenMatrixFloat *error_mat_token = new TokenMatrixFloat(error_mat);
        AssignRef<Token>(error_output, error_mat_token);
        matCrossEntropyGradient(error_mat, input_mat, target_mat, NEAR_ZERO);
        break;
      } // case
    default:
      ERROR_EXIT(246, "Incorrect token type given as target\n");
    } // switch
    return error_output;
  }
  
  float CrossEntropyLossFunction::getAccumLoss() {
    float ret = LossFunction::getAccumLoss();
    if (ret < 0)
      ERROR_EXIT(128, "Found negative loss, check if output is log_logistic\n");
    return ret;
  }

  //////////////////////////////////////////////////////////////////////////

  NonPairedCrossEntropyLossFunction::
  NonPairedCrossEntropyLossFunction(unsigned int size) :
    LossFunction(size) {
  }
  
  NonPairedCrossEntropyLossFunction::~NonPairedCrossEntropyLossFunction() {
  }
  
  MatrixFloat *NonPairedCrossEntropyLossFunction::
  computeLossBunch(Token *input, Token *target) {
    MatrixFloat *input_mat, *target_mat;
    throwErrorAndGetMatrixFromTokens(input, target, input_mat, target_mat);
    int dim = input_mat->getDimSize(0);
    MatrixFloat *loss_output = new MatrixFloat(1, &dim);
#ifdef USE_CUDA
    loss_output->setUseCuda(input_mat->getCudaFlag());
#endif
    
    matNonPairedCrossEntropy(loss_output, input_mat, target_mat, NEAR_ZERO);
    return loss_output;
  }

  Token *NonPairedCrossEntropyLossFunction::computeGradient(Token *input, Token *target) {
    MatrixFloat *input_mat, *target_mat;
    throwErrorAndGetMatrixFromTokens(input, target, input_mat, target_mat);
    MatrixFloat *error_mat = input_mat->cloneOnlyDims();
    TokenMatrixFloat *error_mat_token = new TokenMatrixFloat(error_mat);
    AssignRef<Token>(error_output, error_mat_token);
    matNonPairedCrossEntropyGradient(error_mat, input_mat, target_mat, NEAR_ZERO);
    return error_output;
  }
  
  float NonPairedCrossEntropyLossFunction::getAccumLoss() {
    float ret = LossFunction::getAccumLoss();
    if (ret < 0)
      ERROR_EXIT(128, "Found negative loss, check if output is log_logistic\n");
    return ret;
  }

}
