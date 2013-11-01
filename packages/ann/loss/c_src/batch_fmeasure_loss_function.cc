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
#include "batch_fmeasure_loss_function.h"
#include "wrapper.h"

namespace ANN {

  BatchFMeasureLossFunction::BatchFMeasureLossFunction(unsigned int size,
						       float beta,
						       bool complement_output) :
    LossFunction(size), beta(beta), beta2(beta*beta), dot_products(0),
    input_sums(0), target_sums(0),
    complement_output(complement_output) {
  }
  
  BatchFMeasureLossFunction::~BatchFMeasureLossFunction() {
    delete dot_products;
    delete input_sums;
    delete target_sums;
  }

  // TODO: implement it with CUDA
  MatrixFloat *BatchFMeasureLossFunction::computeLossBunch(Token *input,
							   Token *target) {
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
    int dim = 1;
    MatrixFloat *loss_output = new MatrixFloat(1, &dim, CblasColMajor);
    delete dot_products;
    delete input_sums;
    delete target_sums;
    dim = input_mat->getDimSize(1); // dim = number of classes
    dot_products = new MatrixFloat(1, &dim, CblasColMajor);
#ifdef USE_CUDA
    dot_products->setUseCuda(input_mat->getCudaFlag());
    target_mat->setUseCuda(input_mat->getCudaFlag());
    input_sums->setUseCuda(input_mat->getCudaFlag());
    target_sums->setUseCuda(input_mat->getCudaFlag());
#endif
    dot_products->zeros();
    input_sums   = input_mat->sum(0);
    target_sums  = target_mat->sum(0);
    // compute the dot products for each class, and the sums
    G1 = 0.0f;
    G2 = 0.0f;
    H  = 0.0f;
    MatrixFloat *input_sw=0, *target_sw=0;
    MatrixFloat::iterator dot_product_it(dot_products->begin());
    MatrixFloat::const_iterator input_sums_it(input_sums->begin());
    MatrixFloat::const_iterator target_sums_it(target_sums->begin());
    for (int i=0; i<dim; ++i,
	   ++dot_product_it, ++input_sums_it, ++target_sums_it) {
      //
      april_assert(dot_product_it != dot_products->end());
      april_assert(input_sums_it  != input_sums->end());
      april_assert(target_sums_it != target_sums->end());
      //
      input_sw  = input_mat->select(1,i,input_sw);
      target_sw = target_mat->select(1,i,target_sw);
      *dot_product_it = input_sw->dot(target_sw);
      float a = *dot_product_it / *input_sums_it;
      float b = *dot_product_it / *target_sums_it;
      G1 += a;
      G2 += b;
      H  += a*beta2 + b;
    }
    delete input_sw;
    delete target_sw;
    if (H > 0.0f || H < 0.0f)
      (*loss_output)(0) = -(1.0f+beta2)*G1*G2 / (dim * H);
    else (*loss_output)(0) = 0.0f;
    DecRef(input);
    DecRef(target);
    DecRef(input_mat);
    DecRef(target_mat);
    return loss_output;
  }
  
  Token *BatchFMeasureLossFunction::computeGradient(Token *input, Token *target) {
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
    MatrixFloat *error_mat = input_mat->cloneOnlyDims();
    TokenMatrixFloat *error_mat_token = new TokenMatrixFloat(error_mat);
    AssignRef(error_output, error_mat_token);
#ifdef USE_CUDA
    error_mat->setUseCuda(input_mat->getCudaFlag());
#endif
    if (H > 0.0f || H < 0.0f) {
      MatrixFloat::col_major_iterator error_mat_it(error_mat->begin());
      MatrixFloat::const_col_major_iterator input_it(input_mat->begin());
      MatrixFloat::const_col_major_iterator target_it(target_mat->begin());
      MatrixFloat::const_iterator dot_product_it(dot_products->begin());
      MatrixFloat::const_iterator input_sums_it(input_sums->begin());
      MatrixFloat::const_iterator target_sums_it(target_sums->begin());
      // for each class
      float H2 = H*H;
      float K  = 1.0f + beta2;
      for (int c=0; c<input_mat->getDimSize(1); ++c,
	     ++dot_product_it, ++input_sums_it, ++target_sums_it) {
	//
	april_assert(dot_product_it != dot_products->end());
	april_assert(input_sums_it  != input_sums->end());
	april_assert(target_sums_it != target_sums->end());
	//
	float dot_product  = *dot_product_it;
	float input_sum    = *input_sums_it;
	float target_sum   = *target_sums_it;
	float input_sum2   = input_sum * input_sum;
	float target_sum2  = target_sum * target_sum;
	// for each pattern
	for (int ipat=0; ipat<input_mat->getDimSize(0); ++ipat,
	       ++error_mat_it, ++input_it, ++target_it) {
	  //
	  april_assert(error_mat_it != error_mat->end());
	  april_assert(input_it     != target_mat->end());
	  april_assert(target_it    != input_mat->end());
	  //
	  if (input_sum2 > 0.0f && target_sum2 > 0.0f) {
	    float num = K * ( (*target_it) * input_sum - dot_product) / input_sum2;
	    num *= G2;
	    num += beta2 * G1 * ( (*target_it) * target_sum - dot_product ) / target_sum2;
	    *error_mat_it = - num / H2;
	  }
	  else *error_mat_it = 0.0f;
	}
      }
    }
    else error_mat->zeros();
    DecRef(input);
    DecRef(target);
    DecRef(input_mat);
    DecRef(target_mat);
    return error_output;
  }
  
}
