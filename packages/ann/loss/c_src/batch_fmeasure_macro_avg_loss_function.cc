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
#include "batch_fmeasure_macro_avg_loss_function.h"
#include "wrapper.h"

namespace ANN {

  BatchFMeasureMacroAvgLossFunction::
  BatchFMeasureMacroAvgLossFunction(unsigned int size,
				    float beta,
				    bool complement_output) :
    LossFunction(size), beta(beta), beta2(beta*beta),
    Gs(0), Hs(0),
    complement_output(complement_output) {
  }
  
  BatchFMeasureMacroAvgLossFunction::~BatchFMeasureMacroAvgLossFunction() {
    delete Gs;
    delete Hs;
  }

  MatrixFloat *BatchFMeasureMacroAvgLossFunction::
  computeLossBunch(Token *input, Token *target) {
    IncRef(input);
    IncRef(target);
    MatrixFloat *input_mat, *target_mat;
    throwErrorAndGetMatrixFromTokens(input, target, input_mat, target_mat);
    if (complement_output) {
      input_mat  = input_mat->clone();
      target_mat = target_mat->clone();
      input_mat->complement();
      target_mat->complement();
    }
    IncRef(input_mat);
    IncRef(target_mat);
    //         (1+b^2) dot(o,t)
    // FMb = ---------------------
    //        sum(o) + b^2 sum(t)
    int num_classes = input_mat->getDimSize(1);
    delete Gs;
    delete Hs;
    Gs = new MatrixFloat(1,&num_classes,CblasColMajor);
    Hs = Gs->clone();
    MatrixFloat *class_input_mat=0, *class_target_mat=0;
    MatrixFloat::iterator Gs_it(Gs->begin());
    MatrixFloat::iterator Hs_it(Hs->begin());
    float FMsum = 0.0f;
    for (int i=0; i<num_classes; ++i, ++Gs_it, ++Hs_it) {
      april_assert(Gs_it != Gs->end());
      april_assert(Hs_it != Hs->end());
      class_input_mat  = input_mat->select(1,i,class_input_mat);
      class_target_mat = target_mat->select(1,i,class_target_mat);
      //
      float dot        = class_input_mat->dot(class_target_mat);
      float input_sum  = class_input_mat->sum();
      float target_sum = class_target_mat->sum();
      *Gs_it = (1+beta2) * dot;
      *Hs_it = input_sum + beta2 * target_sum;
      if (*Hs_it > 0.0f || *Hs_it < 0.0f) {
	float FM  = -(*Gs_it)/(*Hs_it);
	FMsum    += FM;
      }
      else {
	// force Hs to be a null pointer, condition used in computeGradient to
	// avoid gradient computation
	delete Hs;
	Hs = 0;
	delete class_input_mat;
	delete class_target_mat;
	return 0;
      }
    }
    delete class_input_mat;
    delete class_target_mat;
    MatrixFloat *loss_output;
    int aux = 1;
    loss_output = new MatrixFloat(1, &aux, CblasColMajor);
    (*loss_output)(0) = FMsum/num_classes;
    //
    DecRef(input);
    DecRef(target);
    DecRef(input_mat);
    DecRef(target_mat);
    return loss_output;
  }
  
  Token *BatchFMeasureMacroAvgLossFunction::
  computeGradient(Token *input,Token *target) {
    IncRef(target);
    MatrixFloat *input_mat, *target_mat;
    throwErrorAndGetMatrixFromTokens(input, target, input_mat, target_mat);
    if (complement_output) {
      target_mat = target_mat->clone();
      target_mat->complement();
    }
    IncRef(target_mat);
    MatrixFloat *error_mat = target_mat->clone();
    TokenMatrixFloat *error_mat_token = new TokenMatrixFloat(error_mat);
    AssignRef(error_output, error_mat_token);
#ifdef USE_CUDA
    error_mat->setUseCuda(input_mat->getCudaFlag());
#endif
    if (Hs != 0) {
      //   grad FMb                 1 + beta^2            (1+b^2) dot(o,t) 
      // ----------- = t_ij * --------------------- - -------------------------
      //  grad o_ij            sum(o) + b^2 sum(t)     [sum(o) + b^2 sum(t)]^2
      int num_classes = input_mat->getDimSize(1);
      MatrixFloat::const_iterator Gs_it(Gs->begin());
      MatrixFloat::const_iterator Hs_it(Hs->begin());
      MatrixFloat *class_error_mat = 0;
      float rel = 1.0f/num_classes;
      for (int i=0; i<num_classes; ++i, ++Gs_it, ++Hs_it) {
	april_assert(Gs_it != Gs->end());
	april_assert(Hs_it != Hs->end());
	class_error_mat = error_mat->select(1,i,class_error_mat);
	float G=*Gs_it, H=*Hs_it;
	float H2    = H*H;
	float scal  = -(1+beta2)/H;
	float add   = G/H2;
	class_error_mat->scal(scal);
	class_error_mat->scalarAdd(add);
      }
      delete class_error_mat;
      error_mat->scal(rel);
    }
    else error_mat->zeros();
    DecRef(target);
    DecRef(target_mat);
    return error_output;
  }
}
