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
#include "multiclass_cross_entropy_loss_function.h"
#include "loss_kernels.h"
#include "token_matrix.h"

using namespace AprilMath;
using namespace AprilMath::MatrixExt::BLAS;
using namespace AprilMath::MatrixExt::LossOperations;
using namespace AprilMath::MatrixExt::Operations;
using namespace AprilMath::MatrixExt::Reductions;
using namespace AprilUtils;
using namespace Basics;

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
    MatrixFloat *loss_output = new MatrixFloat(1, &dim);
#ifdef USE_CUDA
    loss_output->setUseCuda(input_mat->getCudaFlag());
#endif
    matMultiClassCrossEntropy(loss_output, input_mat, target_mat, NEAR_ZERO);
    return loss_output;
  }

  Token *MultiClassCrossEntropyLossFunction::computeGradient(Token *input, Token *target) {
    MatrixFloat *input_mat, *target_mat;
    throwErrorAndGetMatrixFromTokens(input, target, input_mat, target_mat);
    MatrixFloat *error_mat = input_mat->clone();
    TokenMatrixFloat *error_mat_token = new TokenMatrixFloat(error_mat);
    AssignRef<Token>(error_output, error_mat_token);
    matClamp(error_mat, m_log(NEAR_ZERO), m_log(1.0f - NEAR_ZERO));
    matExp(error_mat);
    matAxpy(error_mat, -1.0f, target_mat);
    return error_output;
  }

  char *MultiClassCrossEntropyLossFunction::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.loss.multi_class_cross_entropy(%d)", size);
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
  
  /////////////////////////////////////////////////

  NonPairedMultiClassCrossEntropyLossFunction::
  NonPairedMultiClassCrossEntropyLossFunction(unsigned int size) :
    LossFunction(size) {
    if (size > 0 && size < 3) {
      ERROR_EXIT(128,
		 "Multi class cross entropy is only allowed for multi-class "
		 "problems (three or more output probability neurons). "
		 "Use cross entropy instead.\n");
    }
  }
  
  NonPairedMultiClassCrossEntropyLossFunction::
  ~NonPairedMultiClassCrossEntropyLossFunction() {
  }

  MatrixFloat *NonPairedMultiClassCrossEntropyLossFunction::
  computeLossBunch(Token *input, Token *target) {
    MatrixFloat *input_mat, *target_mat;
    throwErrorAndGetMatrixFromTokens(input, target, input_mat, target_mat);
    int dim = input_mat->getDimSize(0);
    MatrixFloat *loss_output = new MatrixFloat(1, &dim);
#ifdef USE_CUDA
    loss_output->setUseCuda(input_mat->getCudaFlag());
#endif
    AprilUtils::SharedPtr<MatrixFloat> aux(input_mat->clone());
    matClamp(aux.get(), NEAR_ZERO, 1.0f);
    matLog(aux.get());
    matCmul(aux.get(), target_mat);
    matSum(aux.get(), 1, loss_output);
    return matScal(loss_output, -1.0f);
  }

  Token *NonPairedMultiClassCrossEntropyLossFunction::
  computeGradient(Token *input, Token *target) {
    MatrixFloat *input_mat, *target_mat;
    throwErrorAndGetMatrixFromTokens(input, target, input_mat, target_mat);
    MatrixFloat *error_mat = input_mat->clone();
    TokenMatrixFloat *error_mat_token = new TokenMatrixFloat(error_mat);
    AssignRef<Token>(error_output, error_mat_token);
    matClamp(error_mat, NEAR_ZERO, 1.0f);
    matDiv(error_mat, -1.0f);
    matCmul(error_mat, target_mat);
    return error_output;
  }

  char *NonPairedMultiClassCrossEntropyLossFunction::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.loss.non_paired_multi_class_cross_entropy(%d)", size);
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
}
