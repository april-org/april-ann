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
#include "token_memory_block.h"
#include "mae_loss_function.h"
#include "wrapper.h"

namespace ANN {

  MAELossFunction::MAELossFunction(unsigned int size) :
    LossFunction(size), accumulated_loss(0.0f), N(0) {
  }
  
  MAELossFunction::~MAELossFunction() {
  }
  
  float MAELossFunction::addLoss(Token *input, Token *target) {
    MatrixFloat *input_mat, *target_mat;
    throwErrorAndGetMatrixFromTokens(input, target, input_mat, target_mat);
    float loss = doMAELossFunction(input_mat->getRawDataAccess(),
				   target_mat->getRawDataAccess(),
				   0.0f,
				   input_mat->getDimSize(1),
				   input_mat->getDimSize(0),
				   input_mat->getCudaFlag());
    loss = loss/input_mat->getDimSize(0);
    accumulated_loss += loss;
    ++N;
    return loss;
  }

  Token *MAELossFunction::computeGradient(Token *input, Token *target) {
    MatrixFloat *input_mat, *target_mat;
    throwErrorAndGetMatrixFromTokens(input, target, input_mat, target_mat);
    MatrixFloat *error_mat = input_mat->cloneOnlyDims();
    TokenMatrixFloat *error_mat_block = new TokenMatrixFloat(error_mat);
    AssignRef(error_output, error_mat_block);
    doComputeMAEGradient(input_mat->getRawDataAccess(),
			 target_mat->getRawDataAccess(),
			 error_mat->getRawDataAccess(),
			 NEAR_ZERO,
			 input_mat->getDimSize(1),
			 input_mat->getDimSize(0),
			 input_mat->getCudaFlag());
    return error_output;
  }
  
  float MAELossFunction::getAccumLoss() {
    return accumulated_loss/N;
  }
   
  void MAELossFunction::reset() {
    LossFunction::reset();
    accumulated_loss = 0.0f;
    N = 0;
  }
}
