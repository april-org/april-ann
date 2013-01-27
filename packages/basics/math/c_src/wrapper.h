/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
 *
 * Copyright 2012, Salvador España-Boquera, Adrian Palacios Corella, Francisco
 * Zamora-Martinez
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
#include <cstdio>

#include "cblas_headers.h"
#include "error_print.h"

#ifdef USE_CUDA
#include <cuda.h>
#include <cublas_v2.h>
#include "cublas_error.h"
#endif

#include "gpu_mirrored_memory_block.h"
#include "gpu_helper.h"
#include "ann_configuration.h"

#define NEAR_ZERO             1e-5f
#define DERIVATIVE_SATURATION 17.0f

// ATTENTION: In 64-bit machines is better to use exp than expf
#define sigmoid(numerator,value) (numerator) / (exp(-(value))+1.0f)

#define getMatrixFlatIndex(x,lda,y) ((x)+(y)*(lda))
#define getMatrixIndex(x,lda,y) ((x)*(lda)+(y))

// ACTIVATION FUNCTIONS
void doApplyLogisticActivation(FloatGPUMirroredMemoryBlock *units,
			       unsigned int units_size,
			       const ANNConfiguration &conf,
			       bool use_gpu);

void doMultiplyLogisticDerivatives(FloatGPUMirroredMemoryBlock *units,
				   FloatGPUMirroredMemoryBlock *input_errors,
				   unsigned int units_size,
				   const ANNConfiguration &conf,
				   bool use_gpu);
     
void doApplyTanhActivation(FloatGPUMirroredMemoryBlock *units,
			   unsigned int units_size,
			   const ANNConfiguration &conf,
			   bool use_gpu);

void doMultiplyTanhDerivatives(FloatGPUMirroredMemoryBlock *units,
			       FloatGPUMirroredMemoryBlock *input_errors,
			       unsigned int units_size,
			       const ANNConfiguration &conf,
			       bool use_gpu);

void doApplySoftmaxActivation(FloatGPUMirroredMemoryBlock *units,
            FloatGPUMirroredMemoryBlock *minimums,
            FloatGPUMirroredMemoryBlock *maximums,
            FloatGPUMirroredMemoryBlock *sums,
			      unsigned int units_size,
			      const ANNConfiguration &conf,
			      bool use_gpu);

// ERROR FUNCTIONS
void doCalculateMSEErrorFunction(FloatGPUMirroredMemoryBlock *output,
				 FloatGPUMirroredMemoryBlock *target_output,
				 FloatGPUMirroredMemoryBlock *output_error,
				 FloatGPUMirroredMemoryBlock *pattern_errors,
				 float zero_epsilon_distance,
				 unsigned int output_size,
				 const ANNConfiguration &conf,
				 bool use_gpu);

void doCalculateTanhErrorFunction(FloatGPUMirroredMemoryBlock *output,
				  FloatGPUMirroredMemoryBlock *target_output,
				  FloatGPUMirroredMemoryBlock *output_error,
				  FloatGPUMirroredMemoryBlock *pattern_errors,
				  unsigned int output_size,
				  const ANNConfiguration &conf,
				  bool use_gpu);

/*
  float doCalculateMixtureCrossEntropy(FloatGPUMirroredMemoryBlock *output,
  FloatGPUMirroredMemoryBlock *target_output,
  FloatGPUMirroredMemoryBlock *output_error,
  FloatGPUMirroredMemoryBlock *pattern_errors,
  float EPSILON,
  float INF,
  unsigned int output_size,
  const ANNConfiguration &conf,
  bool use_gpu);
*/

void doCalculateLocalFMeasureErrorFunction(float alpha,
					   FloatGPUMirroredMemoryBlock *output,
					   FloatGPUMirroredMemoryBlock *target_output,
					   FloatGPUMirroredMemoryBlock *output_error,
					   FloatGPUMirroredMemoryBlock *pattern_errors,
					   unsigned int output_size,
					   const ANNConfiguration &conf,
					   bool use_gpu);

/*
  float doCalculateGA(FloatGPUMirroredMemoryBlock *output,
  FloatGPUMirroredMemoryBlock *target_output,
  FloatGPUMirroredMemoryBlock *output_error,
  FloatGPUMirroredMemoryBlock *pattern_errors,
  unsigned int output_size,
  const ANNConfiguration &conf,
  bool use_gpu);
*/

void doCalculateCrossEntropyErrorFunction(FloatGPUMirroredMemoryBlock *output,
					  FloatGPUMirroredMemoryBlock *target_output,
					  FloatGPUMirroredMemoryBlock *output_error,
					  FloatGPUMirroredMemoryBlock *pattern_errors,
					  float EPSILON,
					  float INF,
					  unsigned int output_size,
					  const ANNConfiguration &conf,
					  bool use_gpu);

void doCalculateLogisticCrossEntropyErrorFunction(FloatGPUMirroredMemoryBlock *output,
						  FloatGPUMirroredMemoryBlock *target_output,
						  FloatGPUMirroredMemoryBlock *output_error,
						  FloatGPUMirroredMemoryBlock *pattern_errors,
						  float EPSILON,
						  float INF,
						  unsigned int output_size,
						  const ANNConfiguration &conf,
						  bool use_gpu);


void doCalculateFullCrossEntropyErrorFunction(FloatGPUMirroredMemoryBlock *output,
					      FloatGPUMirroredMemoryBlock *target_output,
					      FloatGPUMirroredMemoryBlock *output_error,
					      FloatGPUMirroredMemoryBlock *pattern_errors,
					      float EPSILON,
					      float INF,
					      unsigned int output_size,
					      const ANNConfiguration &conf,
					      bool use_gpu);

// BLAS FUNCTIONS
void doSgemv(CBLAS_ORDER major_type, CBLAS_TRANSPOSE a_transpose,
	     int m, int n,
	     float alpha, FloatGPUMirroredMemoryBlock *a, unsigned int a_inc,
	     FloatGPUMirroredMemoryBlock *x, unsigned int x_inc,
	     float beta, FloatGPUMirroredMemoryBlock *y, unsigned int y_inc,
	     unsigned int a_shift, unsigned int x_shift, unsigned int y_shift,
	     bool use_gpu);

void doScopy(int N, FloatGPUMirroredMemoryBlock* x,
	     unsigned int x_shift,
	     unsigned int x_inc,
	     FloatGPUMirroredMemoryBlock* y,
	     unsigned int y_shift,
	     unsigned int y_inc,
	     bool use_gpu);

void doScopyLoop(int N, FloatGPUMirroredMemoryBlock* x, unsigned int x_inc,
		 FloatGPUMirroredMemoryBlock* y, unsigned int y_inc,
		 unsigned int times, const unsigned int stride,
		 bool use_gpu);

void doSaxpy(int N, float alpha, FloatGPUMirroredMemoryBlock* x,
	     unsigned int x_shift, unsigned int x_inc,
	     FloatGPUMirroredMemoryBlock* y, unsigned int y_shift,
	     unsigned int y_inc, bool use_gpu);

void doSaxpyLoop(int N, float alpha, FloatGPUMirroredMemoryBlock* x,
		 unsigned int x_inc, FloatGPUMirroredMemoryBlock* y,
		 unsigned int y_inc, unsigned int times,
		 const unsigned int stride, bool use_gpu);

void doSgemm(CBLAS_ORDER major_type, CBLAS_TRANSPOSE a_transpose,
	     CBLAS_TRANSPOSE b_transpose, int m, int n, int k, float alpha,
	     FloatGPUMirroredMemoryBlock* a, unsigned int a_inc,
	     FloatGPUMirroredMemoryBlock* b, unsigned int b_inc, float beta,
	     FloatGPUMirroredMemoryBlock* c, unsigned int c_inc,
	     unsigned int a_shift, unsigned int b_shift, unsigned int c_shift,
	     bool use_gpu);

void doVectorSetToZero(FloatGPUMirroredMemoryBlock *v,
		       unsigned int v_size,
		       unsigned int inc,
		       unsigned int shift,
		       bool use_gpu);

void doVectorSet(FloatGPUMirroredMemoryBlock *v,
		 float value,
		 unsigned int v_size,
		 unsigned int inc,
		 unsigned int shift,
		 bool use_gpu);

void doSscal(unsigned int size,
	     float alpha,
	     FloatGPUMirroredMemoryBlock *x,
	     unsigned int shift,
	     unsigned int inc,
	     bool use_gpu);

void doSger(CBLAS_ORDER major_type,
            unsigned int m,
            unsigned int n,
            float alpha,
            FloatGPUMirroredMemoryBlock *x,
            unsigned int x_shift,
            unsigned int x_inc,
            FloatGPUMirroredMemoryBlock *y,
            unsigned int y_shift,
            unsigned int y_inc,
            FloatGPUMirroredMemoryBlock *a,
            unsigned int a_shift,
            unsigned int a_inc,
            bool use_gpu);
