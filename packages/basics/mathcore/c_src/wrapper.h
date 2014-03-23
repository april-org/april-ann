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

#include "lapack_headers.h"
#include "cblas_headers.h"
#include "error_print.h"

#ifdef USE_CUDA
#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include "cublas_error.h"
#include "cusparse_error.h"
#endif

#include "gpu_mirrored_memory_block.h"
#include "gpu_helper.h"
#include "complex_number.h"

#define NEAR_ZERO             1e-6f
#define DERIVATIVE_SATURATION 17.0f

// ATTENTION: In 64-bit machines is better to use exp than expf
#define sigmoid(numerator,value) (numerator) / (expf(-(value))+1.0f)
// The value of -log1p(exp(x)) when X is negative and large, is approximately X
#define logsigmoid(value) ( (value)<-10.0f) ? (value) : (-log1pf(expf(-(value))))

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

void doApplyReLUActivation(FloatGPUMirroredMemoryBlock *input_units,
			   FloatGPUMirroredMemoryBlock *output_units,
			   unsigned int size,
			   unsigned int bunch_size,
			   bool use_gpu);

void doMultiplyReLUDerivatives(FloatGPUMirroredMemoryBlock *input_units,
			       FloatGPUMirroredMemoryBlock *input_errors,
			       FloatGPUMirroredMemoryBlock *output_errors,
			       unsigned int size,
			       unsigned int bunch_size,
			       bool use_gpu);

void doApplyHardtanhActivation(FloatGPUMirroredMemoryBlock *input_units,
			       FloatGPUMirroredMemoryBlock *output_units,
			       unsigned int size,
			       unsigned int bunch_size,
			       float inf, float sup,
			       bool use_gpu);

void doMultiplyHardtanhDerivatives(FloatGPUMirroredMemoryBlock *input_units,
				   FloatGPUMirroredMemoryBlock *input_errors,
				   FloatGPUMirroredMemoryBlock *output_errors,
				   unsigned int size,
				   unsigned int bunch_size,
				   float inf, float sup,
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

void doMultiplySoftmaxDerivatives(FloatGPUMirroredMemoryBlock *output_units,
				  FloatGPUMirroredMemoryBlock *input_errors,
				  FloatGPUMirroredMemoryBlock *output_errors,
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
void doMSELossFunction(FloatGPUMirroredMemoryBlock *input,
		       FloatGPUMirroredMemoryBlock *target,
		       FloatGPUMirroredMemoryBlock *loss_output,
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

void doMAELossFunction(FloatGPUMirroredMemoryBlock *input,
		       FloatGPUMirroredMemoryBlock *target,
		       FloatGPUMirroredMemoryBlock *loss_output,
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

void doCrossEntropyLossFunction(FloatGPUMirroredMemoryBlock *input,
				FloatGPUMirroredMemoryBlock *target,
				FloatGPUMirroredMemoryBlock *loss_output,
				float epsilon,
				unsigned int size,
				unsigned int bunch_size,
				bool use_gpu);

void doMultiClassCrossEntropyLossFunction(FloatGPUMirroredMemoryBlock *input,
					  FloatGPUMirroredMemoryBlock *target,
					  FloatGPUMirroredMemoryBlock *loss_output,
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

// BLAS FUNCTIONS
template<typename T>
void doGemv(CBLAS_ORDER major_type, CBLAS_TRANSPOSE a_transpose,
	    int m, int n,
	    T alpha, GPUMirroredMemoryBlock<T> *a, unsigned int a_inc,
	    GPUMirroredMemoryBlock<T> *x, unsigned int x_inc,
	    T beta, GPUMirroredMemoryBlock<T> *y, unsigned int y_inc,
	    unsigned int a_shift, unsigned int x_shift, unsigned int y_shift,
	    bool use_gpu);

template<typename T>
void doCopy(int N, const GPUMirroredMemoryBlock<T>* x,
	    unsigned int x_shift,
	    unsigned int x_inc,
	    GPUMirroredMemoryBlock<T>* y,
	    unsigned int y_shift,
	    unsigned int y_inc,
	    bool use_gpu);

template<typename T>
void doCopyLoop(int N, GPUMirroredMemoryBlock<T>* x, unsigned int x_inc,
		GPUMirroredMemoryBlock<T>* y, unsigned int y_inc,
		unsigned int times, const unsigned int stride,
		bool use_gpu);

template<typename T>
void doAxpy(int N, T alpha, const GPUMirroredMemoryBlock<T>* x,
	    unsigned int x_shift, unsigned int x_inc,
	    GPUMirroredMemoryBlock<T>* y, unsigned int y_shift,
	    unsigned int y_inc, bool use_gpu);

template<typename T>
void doAxpyLoop(int N, T alpha,
		GPUMirroredMemoryBlock<T>* x,
		unsigned int x_inc, unsigned int x_shift,
		GPUMirroredMemoryBlock<T>* y,
		unsigned int y_inc, unsigned int y_shift,
		unsigned int times,
		const unsigned int stride_x,
		const unsigned int stride_y,
		bool use_gpu);

template<typename T>
void doSparseAxpy(int NNZ,
		  T alpha,
		  const GPUMirroredMemoryBlock<T> *x_values,
		  const Int32GPUMirroredMemoryBlock *x_indices,
		  GPUMirroredMemoryBlock<T>* y,
		  unsigned int y_shift,
		  unsigned int y_inc,
		  bool use_gpu);
  
template<typename T>
void doGemm(CBLAS_ORDER major_type, CBLAS_TRANSPOSE a_transpose,
	    CBLAS_TRANSPOSE b_transpose, int m, int n, int k, T alpha,
	    const GPUMirroredMemoryBlock<T>* a, unsigned int a_inc,
	    const GPUMirroredMemoryBlock<T>* b, unsigned int b_inc, T beta,
	    GPUMirroredMemoryBlock<T>* c, unsigned int c_inc,
	    unsigned int a_shift, unsigned int b_shift, unsigned int c_shift,
	    bool use_gpu);

template <typename T>
void doSparseMM(CBLAS_ORDER major_order,
		SPARSE_FORMAT sparse_format,
		CBLAS_TRANSPOSE a_transpose,
		int m,
		int n,
		int k,
		T alpha,
		const GPUMirroredMemoryBlock<T>* a_values,
		const Int32GPUMirroredMemoryBlock* a_indices,
		const Int32GPUMirroredMemoryBlock* a_first_index,
		const GPUMirroredMemoryBlock<T>* b,
		int b_inc,
		T beta,
		GPUMirroredMemoryBlock<T>* c,
		int c_inc,
		int b_shift,
		int c_shift,
		bool use_gpu);

template<typename T>
void doScal(unsigned int size,
	    GPUMirroredMemoryBlock<T> *x,
	    unsigned int inc,
	    unsigned int shift,
	    T alpha,
	    bool use_gpu);

template<typename T>
void doDiv(unsigned int N,
	   GPUMirroredMemoryBlock<T> *v,
	   unsigned int stride,
	   unsigned int shift,
	   T value,
	   bool use_gpu);

template<typename T>
void doGer(CBLAS_ORDER major_type,
	   unsigned int m,
	   unsigned int n,
	   T alpha,
	   GPUMirroredMemoryBlock<T> *x,
	   unsigned int x_shift,
	   unsigned int x_inc,
	   GPUMirroredMemoryBlock<T> *y,
	   unsigned int y_shift,
	   unsigned int y_inc,
	   GPUMirroredMemoryBlock<T> *a,
	   unsigned int a_shift,
	   unsigned int a_inc,
	   bool use_gpu);

template<typename T>
T doDot(unsigned int size,
	const GPUMirroredMemoryBlock<T> *x,
	unsigned int x_shift,
	unsigned int x_inc,
	const GPUMirroredMemoryBlock<T> *y,
	unsigned int y_shift,
	unsigned int y_inc,
	bool use_gpu);

template<typename T>
float doNrm2(unsigned int n,
	     const GPUMirroredMemoryBlock<T> *x,
	     unsigned int inc,
	     unsigned int shift,
	     bool use_gpu);

template<typename T>
void doSbmv(CBLAS_ORDER major_type,
	    CBLAS_UPLO uplo,
	    int n, int k,
	    T alpha, GPUMirroredMemoryBlock<T> *a, unsigned int a_lda,
	    GPUMirroredMemoryBlock<T> *x, unsigned int x_inc,
	    T beta, GPUMirroredMemoryBlock<T> *y, unsigned int y_inc,
	    unsigned int a_shift, unsigned int x_shift, unsigned int y_shift,
	    bool use_gpu);

template<typename T>
void doClamp(unsigned int N,
	     GPUMirroredMemoryBlock<T> *v,
	     unsigned int stride,
	     unsigned int shift,
	     T lower,
	     T upper,
	     bool use_gpu);  

template<typename T>
void doFill(unsigned int N,
	    GPUMirroredMemoryBlock<T> *v,
	    unsigned int stride,
	    unsigned int shift,
	    T value,
	    bool use_gpu);

template<typename T>
T doSum(unsigned int N,
	const GPUMirroredMemoryBlock<T> *v,
	unsigned int stride,
	unsigned int shift,
	bool use_gpu,
	T zero);

template<typename T>
void doScalarAdd(unsigned int N,
		 GPUMirroredMemoryBlock<T> *v,
		 unsigned int stride,
		 unsigned int shift,
		 T value,
		 bool use_gpu);

template<typename T>
bool doEquals(unsigned int N,
	      const GPUMirroredMemoryBlock<T> *v1,
	      const GPUMirroredMemoryBlock<T> *v2,
	      unsigned int stride1,
	      unsigned int stride2,
	      unsigned int shift1,
	      unsigned int shift2,
	      float epsilon,
	      bool use_gpu);

template <typename T>
void doCmul(int N,
	    const GPUMirroredMemoryBlock<T>* x,
	    unsigned int x_shift,
	    unsigned int x_inc,
	    GPUMirroredMemoryBlock<T>* y,
	    unsigned int y_shift,
	    unsigned int y_inc,
	    bool use_gpu);

void doPLogP(unsigned int N,
	     FloatGPUMirroredMemoryBlock *v,
	     unsigned int stride,
	     unsigned int shift,
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

void doTan(unsigned int N,
	   FloatGPUMirroredMemoryBlock *v,
	   unsigned int stride,
	   unsigned int shift,
	   bool use_gpu);

void doTanh(unsigned int N,
	    FloatGPUMirroredMemoryBlock *v,
	    unsigned int stride,
	    unsigned int shift,
	    bool use_gpu);

void doAtan(unsigned int N,
	    FloatGPUMirroredMemoryBlock *v,
	    unsigned int stride,
	    unsigned int shift,
	    bool use_gpu);

void doAtanh(unsigned int N,
	     FloatGPUMirroredMemoryBlock *v,
	     unsigned int stride,
	     unsigned int shift,
	     bool use_gpu);

void doAbs(unsigned int N,
	   FloatGPUMirroredMemoryBlock *v,
	   unsigned int stride,
	   unsigned int shift,
	   bool use_gpu);

void doComplement(unsigned int N,
		  FloatGPUMirroredMemoryBlock *v,
		  unsigned int stride,
		  unsigned int shift,
		  bool use_gpu);

void doSign(unsigned int N,
	    FloatGPUMirroredMemoryBlock *v,
	    unsigned int stride,
	    unsigned int shift,
	    bool use_gpu);

void doSin(unsigned int N,
	   FloatGPUMirroredMemoryBlock *v,
	   unsigned int stride,
	   unsigned int shift,
	   bool use_gpu);

void doSinh(unsigned int N,
	    FloatGPUMirroredMemoryBlock *v,
	    unsigned int stride,
	    unsigned int shift,
	    bool use_gpu);

void doAsin(unsigned int N,
	    FloatGPUMirroredMemoryBlock *v,
	    unsigned int stride,
	    unsigned int shift,
	    bool use_gpu);

void doAsinh(unsigned int N,
	     FloatGPUMirroredMemoryBlock *v,
	     unsigned int stride,
	     unsigned int shift,
	     bool use_gpu);

void doCos(unsigned int N,
	   FloatGPUMirroredMemoryBlock *v,
	   unsigned int stride,
	   unsigned int shift,
	   bool use_gpu);

void doCosh(unsigned int N,
	    FloatGPUMirroredMemoryBlock *v,
	    unsigned int stride,
	    unsigned int shift,
	    bool use_gpu);

void doAcos(unsigned int N,
	    FloatGPUMirroredMemoryBlock *v,
	    unsigned int stride,
	    unsigned int shift,
	    bool use_gpu);

void doAcosh(unsigned int N,
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

//////////////////////////////////////////////////////////////////////

int doSearchCSCSparseIndexOf(const Int32GPUMirroredMemoryBlock *indices,
			     const Int32GPUMirroredMemoryBlock *first_index,
			     const int c1, const int c2, bool use_gpu);

int doSearchCSRSparseIndexOf(const Int32GPUMirroredMemoryBlock *indices,
			     const Int32GPUMirroredMemoryBlock *first_index,
			     const int c1, const int c2, bool use_gpu);

int doSearchCSCSparseIndexOfFirst(const Int32GPUMirroredMemoryBlock *indices,
				  const Int32GPUMirroredMemoryBlock *first_index,
				  const int c1, const int c2, bool use_gpu);

int doSearchCSRSparseIndexOfFirst(const Int32GPUMirroredMemoryBlock *indices,
				  const Int32GPUMirroredMemoryBlock *first_index,
				  const int c1, const int c2, bool use_gpu);

#endif // WRAPPER_H
