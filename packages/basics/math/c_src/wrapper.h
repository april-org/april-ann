/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
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
#ifndef WRAPPER_H
#define WRAPPER_H
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

#define NEAR_ZERO             1e-6f
#define DERIVATIVE_SATURATION 17.0f

// ATTENTION: In 64-bit machines is better to use exp than expf
#define sigmoid(numerator,value) (numerator) / (expf(-(value))+1.0f)
#define logsigmoid(value) -log1pf(expf(-(value)))

#define getMatrixFlatIndex(x,lda,y) ((x)+(y)*(lda))
#define getMatrixIndex(x,lda,y) ((x)*(lda)+(y))

// ACTIVATION FUNCTIONS
void applyMask(FloatGPUMirroredMemoryBlock *units,
	       FloatGPUMirroredMemoryBlock *mask, float mask_value,
	       unsigned int size,
	       unsigned int bunch_size,
	       bool use_gpu);

void doApplyLogisticActivation(FloatGPUMirroredMemoryBlock *input_units,
			       FloatGPUMirroredMemoryBlock *output_units,
			       unsigned int size,
			       unsigned int bunch_size,
			       bool use_gpu);

void doMultiplyLogisticDerivatives(FloatGPUMirroredMemoryBlock *output_units,
				   FloatGPUMirroredMemoryBlock *input_errors,
				   FloatGPUMirroredMemoryBlock *output_errors,
				   unsigned int size,
				   unsigned int bunch_size,
				   bool use_gpu);

void doApplyLogLogisticActivation(FloatGPUMirroredMemoryBlock *input_units,
				  FloatGPUMirroredMemoryBlock *output_units,
				  unsigned int size,
				  unsigned int bunch_size,
				  bool use_gpu);
     
void doMultiplyLogLogisticDerivatives(FloatGPUMirroredMemoryBlock *output_units,
				      FloatGPUMirroredMemoryBlock *input_errors,
				      FloatGPUMirroredMemoryBlock *output_errors,
				      unsigned int size,
				      unsigned int bunch_size,
				      bool use_gpu);

void doApplyTanhActivation(FloatGPUMirroredMemoryBlock *input_units,
			   FloatGPUMirroredMemoryBlock *output_units,
			   unsigned int size,
			   unsigned int bunch_size,
			   bool use_gpu);

void doMultiplyTanhDerivatives(FloatGPUMirroredMemoryBlock *output_units,
			       FloatGPUMirroredMemoryBlock *input_errors,
			       FloatGPUMirroredMemoryBlock *output_errors,
			       unsigned int size,
			       unsigned int bunch_size,
			       bool use_gpu);

void doApplySoftsignActivation(FloatGPUMirroredMemoryBlock *input_units,
			       FloatGPUMirroredMemoryBlock *output_units,
			       unsigned int size,
			       unsigned int bunch_size,
			       bool use_gpu);

void doMultiplySoftsignDerivatives(FloatGPUMirroredMemoryBlock *output_units,
				   FloatGPUMirroredMemoryBlock *input_errors,
				   FloatGPUMirroredMemoryBlock *output_errors,
				   unsigned int size,
				   unsigned int bunch_size,
				   bool use_gpu);

void doApplySoftplusActivation(FloatGPUMirroredMemoryBlock *input_units,
			       FloatGPUMirroredMemoryBlock *output_units,
			       unsigned int size,
			       unsigned int bunch_size,
			       bool use_gpu);

void doMultiplySoftplusDerivatives(FloatGPUMirroredMemoryBlock *input_units,
				   FloatGPUMirroredMemoryBlock *input_errors,
				   FloatGPUMirroredMemoryBlock *output_errors,
				   unsigned int size,
				   unsigned int bunch_size,
				   bool use_gpu);

void doApplyHardtanhActivation(FloatGPUMirroredMemoryBlock *input_units,
			       FloatGPUMirroredMemoryBlock *output_units,
			       unsigned int size,
			       unsigned int bunch_size,
			       bool use_gpu);

void doMultiplyHardtanhDerivatives(FloatGPUMirroredMemoryBlock *input_units,
				   FloatGPUMirroredMemoryBlock *input_errors,
				   FloatGPUMirroredMemoryBlock *output_errors,
				   unsigned int size,
				   unsigned int bunch_size,
				   bool use_gpu);

void doApplySinActivation(FloatGPUMirroredMemoryBlock *input_units,
			  FloatGPUMirroredMemoryBlock *output_units,
			  unsigned int size,
			  unsigned int bunch_size,
			  bool use_gpu);

void doMultiplySinDerivatives(FloatGPUMirroredMemoryBlock *input_units,
			      FloatGPUMirroredMemoryBlock *input_errors,
			      FloatGPUMirroredMemoryBlock *output_errors,
			      unsigned int size,
			      unsigned int bunch_size,
			      bool use_gpu);

void doApplySoftmaxActivation(FloatGPUMirroredMemoryBlock *input_units,
			      FloatGPUMirroredMemoryBlock *output_units,
			      FloatGPUMirroredMemoryBlock *minimums,
			      FloatGPUMirroredMemoryBlock *maximums,
			      FloatGPUMirroredMemoryBlock *sums,
			      unsigned int size,
			      unsigned int bunch_size,
			      bool use_gpu);

void doApplyLogSoftmaxActivation(FloatGPUMirroredMemoryBlock *input_units,
				 FloatGPUMirroredMemoryBlock *output_units,
				 FloatGPUMirroredMemoryBlock *minimums,
				 FloatGPUMirroredMemoryBlock *maximums,
				 FloatGPUMirroredMemoryBlock *sums,
				 unsigned int size,
				 unsigned int bunch_size,
				 bool use_gpu);

// LOSS FUNCTIONS
float doMSELossFunction(FloatGPUMirroredMemoryBlock *input,
			FloatGPUMirroredMemoryBlock *target,
			float zero_epsilon_distance,
			unsigned int size,
			unsigned int bunch_size,
			bool use_gpu);

void doComputeMSEGradient(FloatGPUMirroredMemoryBlock *input,
			  FloatGPUMirroredMemoryBlock *target,
			  FloatGPUMirroredMemoryBlock *error_output,
			  float zero_epsilon_distance,
			  unsigned int size,
			  unsigned int bunch_size,
			  bool use_gpu);

float doMAELossFunction(FloatGPUMirroredMemoryBlock *input,
			FloatGPUMirroredMemoryBlock *target,
			float zero_epsilon_distance,
			unsigned int size,
			unsigned int bunch_size,
			bool use_gpu);

void doComputeMAEGradient(FloatGPUMirroredMemoryBlock *input,
			  FloatGPUMirroredMemoryBlock *target,
			  FloatGPUMirroredMemoryBlock *error_output,
			  float zero_epsilon_distance,
			  unsigned int size,
			  unsigned int bunch_size,
			  bool use_gpu);

float doCrossEntropyLossFunction(FloatGPUMirroredMemoryBlock *input,
				 FloatGPUMirroredMemoryBlock *target,
				 float epsilon,
				 unsigned int size,
				 unsigned int bunch_size,
				 bool use_gpu);

float doMultiClassCrossEntropyLossFunction(FloatGPUMirroredMemoryBlock *input,
					   FloatGPUMirroredMemoryBlock *target,
					   float epsilon,
					   unsigned int size,
					   unsigned int bunch_size,
					   bool use_gpu);

void doComputeCrossEntropyGradient(FloatGPUMirroredMemoryBlock *input,
				   FloatGPUMirroredMemoryBlock *target,
				   FloatGPUMirroredMemoryBlock *error_output,
				   float epsilon,
				   unsigned int size,
				   unsigned int bunch_size,
				   bool use_gpu);

/*
  void doCalculateTanhErrorFunction(FloatGPUMirroredMemoryBlock *output,
  FloatGPUMirroredMemoryBlock *target_output,
  FloatGPUMirroredMemoryBlock *output_error,
  FloatGPUMirroredMemoryBlock *pattern_errors,
  unsigned int output_size,
  const ANNConfiguration &conf,
  bool use_gpu);
*/

float doLocalFMeasureLossFunction(FloatGPUMirroredMemoryBlock *input,
				  FloatGPUMirroredMemoryBlock *target,
				  unsigned int size,
				  unsigned int bunch_size,
				  float beta,
				  float &Gab, float &Hab,
				  bool complement_output,
				  bool use_gpu);

void doComputeLocalFMeasureGradient(FloatGPUMirroredMemoryBlock *target,
				    FloatGPUMirroredMemoryBlock *output_error,
				    unsigned int size,
				    unsigned int bunch_size,
				    float beta,
				    float Gab, float Hab,
				    bool complement_output,
				    bool use_gpu);

// BLAS FUNCTIONS
void doSgemv(CBLAS_ORDER major_type, CBLAS_TRANSPOSE a_transpose,
	     int m, int n,
	     float alpha, FloatGPUMirroredMemoryBlock *a, unsigned int a_inc,
	     FloatGPUMirroredMemoryBlock *x, unsigned int x_inc,
	     float beta, FloatGPUMirroredMemoryBlock *y, unsigned int y_inc,
	     unsigned int a_shift, unsigned int x_shift, unsigned int y_shift,
	     bool use_gpu);

void doScopy(int N, const FloatGPUMirroredMemoryBlock* x,
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

void doSaxpy(int N, float alpha, const FloatGPUMirroredMemoryBlock* x,
	     unsigned int x_shift, unsigned int x_inc,
	     FloatGPUMirroredMemoryBlock* y, unsigned int y_shift,
	     unsigned int y_inc, bool use_gpu);

void doSaxpyLoop(int N, float alpha, FloatGPUMirroredMemoryBlock* x,
		 unsigned int x_inc, FloatGPUMirroredMemoryBlock* y,
		 unsigned int y_inc, unsigned int times,
		 const unsigned int stride_x,
		 const unsigned int stride_y,
		 bool use_gpu);

void doSgemm(CBLAS_ORDER major_type, CBLAS_TRANSPOSE a_transpose,
	     CBLAS_TRANSPOSE b_transpose, int m, int n, int k, float alpha,
	     FloatGPUMirroredMemoryBlock* a, unsigned int a_inc,
	     FloatGPUMirroredMemoryBlock* b, unsigned int b_inc, float beta,
	     FloatGPUMirroredMemoryBlock* c, unsigned int c_inc,
	     unsigned int a_shift, unsigned int b_shift, unsigned int c_shift,
	     bool use_gpu);

void doSscal(unsigned int size,
	     FloatGPUMirroredMemoryBlock *x,
	     unsigned int inc,
	     unsigned int shift,
	     float alpha,
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

float doSdot(unsigned int size,
	     const FloatGPUMirroredMemoryBlock *x,
	     unsigned int x_shift,
	     unsigned int x_inc,
	     const FloatGPUMirroredMemoryBlock *y,
	     unsigned int y_shift,
	     unsigned int y_inc,
	     bool use_gpu);

float doSnrm2(unsigned int n,
	      const FloatGPUMirroredMemoryBlock *x,
	      unsigned int inc,
	      unsigned int shift,
	      bool use_gpu);

void doSsbmv(CBLAS_ORDER major_type,
	     CBLAS_UPLO uplo,
	     int n, int k,
	     float alpha, FloatGPUMirroredMemoryBlock *a, unsigned int a_lda,
	     FloatGPUMirroredMemoryBlock *x, unsigned int x_inc,
	     float beta, FloatGPUMirroredMemoryBlock *y, unsigned int y_inc,
	     unsigned int a_shift, unsigned int x_shift, unsigned int y_shift,
	     bool use_gpu);

void doClamp(unsigned int N,
	     FloatGPUMirroredMemoryBlock *v,
	     unsigned int stride,
	     unsigned int shift,
	     float lower,
	     float upper,
	     bool use_gpu);

void doFill(unsigned int N,
	    FloatGPUMirroredMemoryBlock *v,
	    unsigned int stride,
	    unsigned int shift,
	    float value,
	    bool use_gpu);

float doSum(unsigned int N,
	    const FloatGPUMirroredMemoryBlock *v,
	    unsigned int stride,
	    unsigned int shift,
	    bool use_gpu);

void doScalarAdd(unsigned int N,
		 FloatGPUMirroredMemoryBlock *v,
		 unsigned int stride,
		 unsigned int shift,
		 float value,
		 bool use_gpu);

bool doEquals(unsigned int N,
	      const FloatGPUMirroredMemoryBlock *v1,
	      const FloatGPUMirroredMemoryBlock *v2,
	      unsigned int stride1,
	      unsigned int stride2,
	      unsigned int shift1,
	      unsigned int shift2,
	      float epsilon,
	      bool use_gpu);

void doLog(unsigned int N,
	   FloatGPUMirroredMemoryBlock *v,
	   unsigned int stride,
	   unsigned int shift,
	   bool use_gpu);

void doLog1p(unsigned int N,
	     FloatGPUMirroredMemoryBlock *v,
	     unsigned int stride,
	     unsigned int shift,
	     bool use_gpu);

void doExp(unsigned int N,
	   FloatGPUMirroredMemoryBlock *v,
	   unsigned int stride,
	   unsigned int shift,
	   bool use_gpu);

void doSqrt(unsigned int N,
	    FloatGPUMirroredMemoryBlock *v,
	    unsigned int stride,
	    unsigned int shift,
	    bool use_gpu);

void doTanh(unsigned int N,
	    FloatGPUMirroredMemoryBlock *v,
	    unsigned int stride,
	    unsigned int shift,
	    bool use_gpu);

void doPow(unsigned int N,
	   FloatGPUMirroredMemoryBlock *v,
	   unsigned int stride,
	   unsigned int shift,
	   float value,
	   bool use_gpu);
#endif // WRAPPER_H
