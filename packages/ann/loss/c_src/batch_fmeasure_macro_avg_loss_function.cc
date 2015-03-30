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
#include "batch_fmeasure_macro_avg_loss_function.h"
#include "matrix_ext.h"
#include "token_matrix.h"

using namespace AprilMath::MatrixExt::BLAS;
using namespace AprilMath::MatrixExt::Initializers;
using namespace AprilMath::MatrixExt::Operations;
using namespace AprilMath::MatrixExt::Reductions;
using namespace AprilUtils;
using namespace Basics;

namespace ANN {

  BatchFMeasureMacroAvgLossFunction::
  BatchFMeasureMacroAvgLossFunction(unsigned int size,
				    float beta,
				    bool complement_output) :
    LossFunction(size), beta(beta), beta2(beta*beta),
    complement_output(complement_output) {
  }
  
  BatchFMeasureMacroAvgLossFunction::~BatchFMeasureMacroAvgLossFunction() {
  }

  MatrixFloat *BatchFMeasureMacroAvgLossFunction::
  computeLossBunch(Token *input_, Token *target_) {
    AprilUtils::SharedPtr<Token> input(input_);
    AprilUtils::SharedPtr<Token> target(target_);
    MatrixFloat *input_mat_, *target_mat_;
    throwErrorAndGetMatrixFromTokens(input.get(), target.get(),
                                     input_mat_, target_mat_);
    AprilUtils::SharedPtr<MatrixFloat> input_mat(input_mat_);
    AprilUtils::SharedPtr<MatrixFloat> target_mat(target_mat_);
    if (complement_output) {
      input_mat  = input_mat->clone();
      target_mat = target_mat->clone();
      matComplement(input_mat.get());
      matComplement(target_mat.get());
    }
    //         (1+b^2) dot(o,t)
    // FMb = ---------------------
    //        sum(o) + b^2 sum(t)
    int num_classes = input_mat->getDimSize(1);
    Gs = new MatrixFloat(1,&num_classes);
    Hs = Gs->clone();
    AprilUtils::SharedPtr<MatrixFloat> class_input_mat;
    AprilUtils::SharedPtr<MatrixFloat> class_target_mat;
    MatrixFloat::iterator Gs_it(Gs->begin());
    MatrixFloat::iterator Hs_it(Hs->begin());
    float FMsum = 0.0f;
    for (int i=0; i<num_classes; ++i, ++Gs_it, ++Hs_it) {
      april_assert(Gs_it != Gs->end());
      april_assert(Hs_it != Hs->end());
      class_input_mat  = input_mat->select(1,i,class_input_mat.get());
      class_target_mat = target_mat->select(1,i,class_target_mat.get());
      //
      float dot        = matDot(class_input_mat.get(), class_target_mat.get());
      float input_sum  = matSum(class_input_mat.get());
      float target_sum = matSum(class_target_mat.get());
      *Gs_it = (1+beta2) * dot;
      *Hs_it = input_sum + beta2 * target_sum;
      if (*Hs_it > 0.0f || *Hs_it < 0.0f) {
	float FM  = -(*Gs_it)/(*Hs_it);
	FMsum    += FM;
      }
      else {
	// force Hs to be a null pointer, condition used in computeGradient to
	// avoid gradient computation
        Hs.reset();
	return 0;
      }
    }
    MatrixFloat *loss_output;
    int aux = 1;
    loss_output = new MatrixFloat(1, &aux);
#ifdef USE_CUDA
    loss_output->setUseCuda(input_mat_->getCudaFlag());
#endif
    (*loss_output)(0) = FMsum/num_classes;
    //
    return loss_output;
  }
  
  Token *BatchFMeasureMacroAvgLossFunction::
  computeGradient(Token *input_,Token *target_) {
    AprilUtils::SharedPtr<Token> input(input_);
    AprilUtils::SharedPtr<Token> target(target_);
    MatrixFloat *input_mat_, *target_mat_;
    throwErrorAndGetMatrixFromTokens(input.get(), target.get(),
                                     input_mat_, target_mat_);
    AprilUtils::SharedPtr<MatrixFloat> input_mat(input_mat_);
    AprilUtils::SharedPtr<MatrixFloat> target_mat(target_mat_);
    if (complement_output) {
      target_mat = target_mat->clone();
      matComplement(target_mat.get());
    }
    MatrixFloat *error_mat = target_mat->clone();
    TokenMatrixFloat *error_mat_token = new TokenMatrixFloat(error_mat);
    AssignRef<Token>(error_output, error_mat_token);
#ifdef USE_CUDA
    error_mat->setUseCuda(input_mat->getCudaFlag());
#endif
    if (!Hs.empty()) {
      //   grad FMb                 1 + beta^2            (1+b^2) dot(o,t) 
      // ----------- = t_ij * --------------------- - -------------------------
      //  grad o_ij            sum(o) + b^2 sum(t)     [sum(o) + b^2 sum(t)]^2
      int num_classes = input_mat->getDimSize(1);
      MatrixFloat::const_iterator Gs_it(Gs->begin());
      MatrixFloat::const_iterator Hs_it(Hs->begin());
      AprilUtils::SharedPtr<MatrixFloat> class_error_mat;
      float rel = 1.0f/num_classes;
      for (int i=0; i<num_classes; ++i, ++Gs_it, ++Hs_it) {
	april_assert(Gs_it != Gs->end());
	april_assert(Hs_it != Hs->end());
	class_error_mat = error_mat->select(1,i,class_error_mat.get());
	float G=*Gs_it, H=*Hs_it;
	float H2    = H*H;
	float scal  = -(1+beta2)/H;
	float add   = G/H2;
        matScal(class_error_mat.get(),scal);
	matScalarAdd(class_error_mat.get(),add);
      }
      matScal(error_mat,rel);
    }
    else {
      matZeros(error_mat);
    }
    return error_output;
  }

  char *BatchFMeasureMacroAvgLossFunction::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.loss.batch_fmeasure_macro_avg{ size=%d, beta=%f, "
		  "complement=%s }",
		  size, beta, (complement_output)?"true":"false");
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }

}
