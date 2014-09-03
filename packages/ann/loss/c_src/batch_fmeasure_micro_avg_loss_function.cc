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
#include "batch_fmeasure_micro_avg_loss_function.h"
#include "matrix_operations.h"
#include "token_matrix.h"

using namespace AprilMath::MatrixExt::Operations;
using namespace AprilUtils;
using namespace Basics;

namespace ANN {

  BatchFMeasureMicroAvgLossFunction::
  BatchFMeasureMicroAvgLossFunction(unsigned int size,
				    float beta,
				    bool complement_output) :
    LossFunction(size), beta(beta), beta2(beta*beta),
    G(0.0f), H(0.0f),
    complement_output(complement_output) {
  }
  
  BatchFMeasureMicroAvgLossFunction::~BatchFMeasureMicroAvgLossFunction() {
  }

  MatrixFloat *BatchFMeasureMicroAvgLossFunction::
  computeLossBunch(Token *input, Token *target) {
    IncRef(input);
    IncRef(target);
    MatrixFloat *input_mat, *target_mat;
    throwErrorAndGetMatrixFromTokens(input, target, input_mat, target_mat);
    if (complement_output) {
      input_mat  = input_mat->clone();
      target_mat = target_mat->clone();
      matComplement(input_mat);
      matComplement(target_mat);
    }
    IncRef(input_mat);
    IncRef(target_mat);
    //         (1+b^2) dot(o,t)
    // FMb = ---------------------
    //        sum(o) + b^2 sum(t)
    float dot;
    if (input_mat->getDimSize(0) == 1 || input_mat->getDimSize(1) == 1)
      // is a vector
      dot = matDot(input_mat,target_mat);
    else {
      // is a matrix
      int dim = (input_mat->getDimSize(0) > input_mat->getDimSize(1)) ? 0 : 1;
      dot = 0.0f;
      MatrixFloat *aux1=0, *aux2=0;
      for (int i=0; i<input_mat->getDimSize(dim); ++i) {
	aux1 = input_mat->select(dim,i,aux1);
	aux2 = target_mat->select(dim,i,aux2);
	dot += matDot(aux1,aux2);
      }
      delete aux1;
      delete aux2;
    }
    float input_sum  = matSum(input_mat);
    float target_sum = matSum(target_mat);
    G = (1+beta2) * dot;
    H = input_sum + beta2 * target_sum;
    MatrixFloat *loss_output;
    if ( H>0.0f || H<0.0f ) {
      int dim = 1;
      loss_output = new MatrixFloat(1, &dim, CblasColMajor);
#ifdef USE_CUDA
      loss_output->setUseCuda(input_mat->getCudaFlag());
#endif
      (*loss_output)(0) = -G/H;
    }
    else loss_output = 0;
    //
    DecRef(input);
    DecRef(target);
    DecRef(input_mat);
    DecRef(target_mat);
    return loss_output;
  }
  
  Token *BatchFMeasureMicroAvgLossFunction::
  computeGradient(Token *input,Token *target) {
    IncRef(target);
    MatrixFloat *input_mat, *target_mat;
    throwErrorAndGetMatrixFromTokens(input, target, input_mat, target_mat);
    if (complement_output) {
      target_mat = target_mat->clone();
      matComplement(target_mat);
    }
    IncRef(target_mat);
    MatrixFloat *error_mat = target_mat->clone();
    TokenMatrixFloat *error_mat_token = new TokenMatrixFloat(error_mat);
    AssignRef<Token>(error_output, error_mat_token);
#ifdef USE_CUDA
    error_mat->setUseCuda(input_mat->getCudaFlag());
#endif
    if (H > 0.0f || H < 0.0f) {
      //   grad FMb                 1 + beta^2            (1+b^2) dot(o,t) 
      // ----------- = t_ij * --------------------- - -------------------------
      //  grad o_ij            sum(o) + b^2 sum(t)     [sum(o) + b^2 sum(t)]^2
      float H2    = H*H;
      float scal  = -(1+beta2)/H;
      float add   = G/H2;
      matScal(error_mat,scal);
      matScalarAdd(error_mat,add);
    }
    else {
      matZeros(error_mat);
    }
    DecRef(target);
    DecRef(target_mat);
    return error_output;
  }

  char *BatchFMeasureMicroAvgLossFunction::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.loss.batch_fmeasure_micro_avg{ size=%d, beta=%f, "
		  "complement=%s }",
		  size, beta, (complement_output)?"true":"false");
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }

}
