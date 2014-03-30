/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
 *
 * Copyright 2012, Salvador España-Boquera, Adrian Palacios Corella, Francisco
 * Zamora-Martinez
 *
 * The APRIL-MLP toolkit is free software; you can redistribute it and/or modify it
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
#include "wrapper.h"
#include "cuda_utils.h"
#include "unused_variable.h"

#ifdef USE_CUDA
/***************************************
 ************** CUDA SECTION ***********
 ***************************************/

cublasStatus_t wrapperCublasGemv(cublasHandle_t &handle,
				 cublasOperation_t &cublas_a_transpose,
				 int m, int n,
				 const float *alpha,
				 const float *a_mem,
				 unsigned int a_inc,
				 const float *x_mem,
				 unsigned int x_inc,
				 const float *beta,
				 float *y_mem,
				 unsigned int y_inc) {
  return cublasSgemv(handle, cublas_a_transpose,
		     m, n,
		     alpha, a_mem, a_inc,
		     x_mem, x_inc,
		     beta, y_mem, y_inc);
}

cublasStatus_t wrapperCublasGemv(cublasHandle_t &handle,
				 cublasOperation_t &cublas_a_transpose,
				 int m, int n,
				 const double *alpha,
				 const double *a_mem,
				 unsigned int a_inc,
				 const double *x_mem,
				 unsigned int x_inc,
				 const double *beta,
				 double *y_mem,
				 unsigned int y_inc) {
  return cublasSgemv(handle, cublas_a_transpose,
		     m, n,
		     alpha, a_mem, a_inc,
		     x_mem, x_inc,
		     beta, y_mem, y_inc);
}

cublasStatus_t wrapperCublasGemv(cublasHandle_t &handle,
				 cublasOperation_t &cublas_a_transpose,
				 int m, int n,
				 const ComplexF *alpha,
				 const ComplexF *a_mem,
				 unsigned int a_inc,
				 const ComplexF *x_mem,
				 unsigned int x_inc,
				 const ComplexF *beta,
				 ComplexF *y_mem,
				 unsigned int y_inc) {
  return cublasCgemv(handle, cublas_a_transpose,
		     m, n,
		     reinterpret_cast<const cuComplex*>(alpha),
                     reinterpret_cast<const cuComplex*>(a_mem), a_inc,
		     reinterpret_cast<const cuComplex*>(x_mem), x_inc,
		     reinterpret_cast<const cuComplex*>(beta),
                     reinterpret_cast<cuComplex*>(y_mem), y_inc);
}

cusparseStatus_t wrapperCusparseCSRGemv(cusparseHandle_t &handle,
                                        cusparseOperation_t &cusparse_a_transpose,
                                        int m, int n, int NNZ,
                                        const float *alpha,
                                        cusparseMatDescr_t descrA,
                                        const float *a_values_mem,
                                        const int *a_indices_mem,
                                        const int *a_first_index_mem,
                                        const float *x_mem,
                                        unsigned int x_inc,
                                        const float *beta,
                                        float *y_mem,
                                        unsigned int y_inc) {
  if (x_inc != 1 || y_inc != 1)
    ERROR_EXIT(128, "Not implemented for non contiguous vectors\n");
  return cusparseScsrmv(handle, cusparse_a_transpose,
                        m, n, NNZ,
                        alpha,
                        descrA,
                        a_values_mem,
                        a_first_index_mem,
                        a_indices_mem
                        x_mem, x_inc,
                        beta, y_mem, y_inc);
}

cusparseStatus_t wrapperCusparseCSRGemv(cusparseHandle_t &handle,
                                        cusparseOperation_t &cusparse_a_transpose,
                                        int m, int n, int NNZ,
                                        const double *alpha,
                                        cusparseMatDescr_t descrA,
                                        const double *a_values_mem,
                                        const int *a_indices_mem,
                                        const int *a_first_index_mem,
                                        const double *x_mem,
                                        unsigned int x_inc,
                                        const double *beta,
                                        double *y_mem,
                                        unsigned int y_inc) {
  if (x_inc != 1 || y_inc != 1)
    ERROR_EXIT(128, "Not implemented for non contiguous vectors\n");
  return cusparseScsrmv(handle, cusparse_a_transpose,
                        m, n, NNZ,
                        alpha,
                        descrA,
                        a_values_mem,
                        a_first_index_mem,
                        a_indices_mem
                        x_mem, x_inc,
                        beta, y_mem, y_inc);
}

cusparseStatus_t wrapperCusparseCSRGemv(cusparseHandle_t &handle,
                                        cusparseOperation_t &cusparse_a_transpose,
                                        int m, int n, int NNZ,
                                        const ComplexF *alpha,
                                        cusparseMatDescr_t descrA,
                                        const ComplexF *a_values_mem,
                                        const int *a_indices_mem,
                                        const int *a_first_index_mem,
                                        const ComplexF *x_mem,
                                        unsigned int x_inc,
                                        const ComplexF *beta,
                                        ComplexF *y_mem,
                                        unsigned int y_inc) {
  if (x_inc != 1 || y_inc != 1)
    ERROR_EXIT(128, "Not implemented for non contiguous vectors\n");
  return cusparseScsrmv(handle, cusparse_a_transpose,
                        m, n, NNZ,
                        alpha,
                        descrA,
                        a_values_mem,
                        a_first_index_mem,
                        a_indices_mem
                        x_mem, x_inc,
                        beta, y_mem, y_inc);
}
#endif

/***************************************
 ************* CBLAS SECTION ***********
 ***************************************/

void wrapperCblasGemv(CBLAS_ORDER &major_order,
		      CBLAS_TRANSPOSE a_transpose,
		      int m, int n,
		      float alpha,
		      const float *a_mem, unsigned int a_inc,
		      const float *x_mem, unsigned int x_inc,
		      float beta,
		      float *y_mem, unsigned int y_inc) {
  cblas_sgemv(major_order, a_transpose, m, n, alpha, a_mem, a_inc,
	      x_mem, x_inc, beta, y_mem, y_inc);
}

void wrapperCblasGemv(CBLAS_ORDER &major_order,
		      CBLAS_TRANSPOSE a_transpose,
		      int m, int n,
		      ComplexF alpha,
		      const ComplexF *a_mem, unsigned int a_inc,
		      const ComplexF *x_mem, unsigned int x_inc,
		      ComplexF beta,
		      ComplexF *y_mem, unsigned int y_inc) {
  cblas_cgemv(major_order, a_transpose, m, n, &alpha, a_mem, a_inc,
	      x_mem, x_inc, &beta, y_mem, y_inc);
}

template <typename T>
void wrapperCblasSparseMM(SPARSE_FORMAT sparse_format,
			  CBLAS_TRANSPOSE a_transpose,
			  int m, int n,
			  T alpha,
			  const T *a_values_mem,
			  const int *a_indices_mem,
			  const int *a_first_index_mem,
			  const T *x_mem, unsigned int x_inc,
			  T beta, T *y_mem, unsigned int y_inc) {
  cblas_sparse_mv(sparse_format, a_transpose,
                  m, n,
                  alpha, a_values_mem, a_indices_mem, a_first_index_mem,
                  x_mem, static_cast<int>(x_inc),
                  beta, y_mem, static_cast<int>(y_inc));
}

/***************************************
 *********** TEMPLATE SECTION **********
 ***************************************/

template<typename T>
void doGemv(CBLAS_ORDER major_order, CBLAS_TRANSPOSE a_transpose,
	    int m, int n,
	    T alpha, GPUMirroredMemoryBlock<T> *a, unsigned int a_inc,
	    GPUMirroredMemoryBlock<T> *x, unsigned int x_inc,
	    T beta, GPUMirroredMemoryBlock<T> *y, unsigned int y_inc,
	    unsigned int a_shift, unsigned int x_shift, unsigned int y_shift,
	    bool use_gpu) {
  const T *a_mem, *x_mem;
  T *y_mem;
#ifndef USE_CUDA
  UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
  if (use_gpu) {
    cublasStatus_t status;
    cublasHandle_t handle = GPUHelper::getHandler();
    assert(major_order == CblasColMajor);
    cublasOperation_t cublas_a_transpose = getCublasOperation(a_transpose);
    a_mem = a->getGPUForRead() + a_shift;
    x_mem = x->getGPUForRead() + x_shift;
    y_mem = y->getGPUForReadAndWrite() + y_shift;

    status = cublasSetStream(handle, GPUHelper::getCurrentStream());
    checkCublasError(status);

    status = wrapperCublasGemv(handle, cublas_a_transpose,
			       m, n,
			       &alpha, a_mem, a_inc,
			       x_mem, x_inc,
			       &beta, y_mem, y_inc);
    
    checkCublasError(status);
  }
  else {
#endif
    a_mem = a->getPPALForRead() + a_shift;
    x_mem = x->getPPALForRead() + x_shift;
    y_mem = y->getPPALForReadAndWrite() + y_shift;
    wrapperCblasGemv(major_order, a_transpose,
		     m, n,
		     alpha, a_mem, a_inc,
		     x_mem, x_inc,
		     beta, y_mem, y_inc);
#ifdef USE_CUDA
  }
#endif
}

template<typename T>
void doSparseGemv(CBLAS_ORDER major_order, SPARSE_FORMAT sparse_format,
                  CBLAS_TRANSPOSE a_transpose,
                  int m, int n,
                  T alpha,
                  GPUMirroredMemoryBlock<T> *a_values,
                  Int32GPUMirroredMemoryBlock *a_indices,
                  Int32GPUMirroredMemoryBlock *a_first_index,
                  GPUMirroredMemoryBlock<T> *x, unsigned int x_inc,
                  T beta, GPUMirroredMemoryBlock<T> *y, unsigned int y_inc,
                  unsigned int x_shift, unsigned int y_shift,
                  bool use_gpu) {
  const T *a_values_mem, *x_mem;
  const int *a_indices_mem, *a_first_index_mem;
  T *y_mem;
  const int NNZ = static_cast<int>(a_values->getSize());
#ifndef USE_CUDA
  UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
  if (use_gpu) {
    cusparseStatus_t status;
    cusparseHandle_t handle = GPUHelper::getSparseHandler();
    if (major_order != CblasColMajor)
      ERROR_EXIT(128, "Column major matrices are expected\n");
    if (sparse_format != SparseMatrix<T>::CSR_FORMAT)
      a_transpose = NEGATE_CBLAS_TRANSPOSE(a_transpose);
    a_values_mem = a_values->getGPUForRead();
    a_indices_mem = a_indices->getGPUForRead();
    a_first_index_mem = a_first_index->getGPUForRead();
    x_mem = x->getGPUForRead() + x_shift;
    y_mem = y->getGPUForReadAndWrite() + y_shift;
    cusparseOperation_t cusparse_a_transpose = getCusparseOperation(a_transpose);
    
    status = cusparseSetStream(handle, GPUHelper::getCurrentStream());
    checkCusparseError(status);
    cusparseMatDescr_t descrA = {
      CUSPARSE_MATRIX_TYPE_GENERAL,
      0, // fill mode
      0, // diag type
      CUSPARSE_INDEX_BASE_ZERO
    };
    
    status = wrapperCusparseCSRGemv(handle,
                                    cusparse_a_transpose,
                                    m, n, NNZ,
                                    &alpha,
                                    descrA,
                                    a_values_mem,
                                    a_first_index_mem,
                                    a_indices_mem,
                                    x_mem, x_inc,
                                    &beta, y_mem, y_inc);
    checkCusparseError(status);

  }
  else {
#endif
    if (major_order != CblasRowMajor)
      ERROR_EXIT(128, "Row major matrices are expected\n");
    a_values_mem = a_values->getPPALForRead();
    a_indices_mem = a_indices->getPPALForRead();
    a_first_index_mem = a_first_index->getPPALForRead();
    x_mem = x->getPPALForRead() + x_shift;
    y_mem = y->getPPALForReadAndWrite() + y_shift;
    wrapperCblasSparseGemv(sparse_format,
                           a_transpose,
                           m, n,
                           alpha,
                           a_values_mem,
                           a_indices_mem,
                           a_first_index_mem,
                           x_mem, x_inc,
                           beta, y_mem, y_inc);
#ifdef USE_CUDA
  }
#endif
}

template void doGemv<float>(CBLAS_ORDER major_order, CBLAS_TRANSPOSE a_transpose,
			    int m, int n,
			    float alpha, GPUMirroredMemoryBlock<float> *a, unsigned int a_inc,
			    GPUMirroredMemoryBlock<float> *x, unsigned int x_inc,
			    float beta, GPUMirroredMemoryBlock<float> *y, unsigned int y_inc,
			    unsigned int a_shift, unsigned int x_shift, unsigned int y_shift,
			    bool use_gpu);

template void doGemv<double>(CBLAS_ORDER major_order, CBLAS_TRANSPOSE a_transpose,
                             int m, int n,
                             double alpha, GPUMirroredMemoryBlock<double> *a, unsigned int a_inc,
                             GPUMirroredMemoryBlock<double> *x, unsigned int x_inc,
                             double beta, GPUMirroredMemoryBlock<double> *y, unsigned int y_inc,
                             unsigned int a_shift, unsigned int x_shift, unsigned int y_shift,
                             bool use_gpu);

template void doGemv<ComplexF>(CBLAS_ORDER major_order, CBLAS_TRANSPOSE a_transpose,
			       int m, int n,
			       ComplexF alpha, GPUMirroredMemoryBlock<ComplexF> *a, unsigned int a_inc,
			       GPUMirroredMemoryBlock<ComplexF> *x, unsigned int x_inc,
			       ComplexF beta, GPUMirroredMemoryBlock<ComplexF> *y, unsigned int y_inc,
			       unsigned int a_shift, unsigned int x_shift, unsigned int y_shift,
			       bool use_gpu);

template void doSparseGemv<float>(CBLAS_ORDER major_order,
                                  SPARSE_FORMAT sparse_format,
                                  CBLAS_TRANSPOSE a_transpose,
                                  int m, int n,
                                  float alpha,
                                  GPUMirroredMemoryBlock<float> *a_values,
                                  Int32GPUMirroredMemoryBlock *a_indices,
                                  Int32GPUMirroredMemoryBlock *a_first_index,
                                  GPUMirroredMemoryBlock<float> *x, unsigned int x_inc,
                                  float beta, GPUMirroredMemoryBlock<float> *y, unsigned int y_inc,
                                  unsigned int x_shift, unsigned int y_shift,
                                  bool use_gpu);

template void doSparseGemv<double>(CBLAS_ORDER major_order,
                                   SPARSE_FORMAT sparse_format,
                                   CBLAS_TRANSPOSE a_transpose,
                                   int m, int n,
                                   double alpha,
                                   GPUMirroredMemoryBlock<double> *a_values,
                                   Int32GPUMirroredMemoryBlock *a_indices,
                                   Int32GPUMirroredMemoryBlock *a_first_index,
                                   GPUMirroredMemoryBlock<double> *x, unsigned int x_inc,
                                   double beta, GPUMirroredMemoryBlock<double> *y, unsigned int y_inc,
                                   unsigned int x_shift, unsigned int y_shift,
                                   bool use_gpu);

template void doSparseGemv<ComplexF>(CBLAS_ORDER major_order,
                                     SPARSE_FORMAT sparse_format,
                                     CBLAS_TRANSPOSE a_transpose,
                                     int m, int n,
                                     ComplexF alpha,
                                     GPUMirroredMemoryBlock<ComplexF> *a_values,
                                     Int32GPUMirroredMemoryBlock *a_indices,
                                     Int32GPUMirroredMemoryBlock *a_first_index,
                                     GPUMirroredMemoryBlock<ComplexF> *x, unsigned int x_inc,
                                     ComplexF beta, GPUMirroredMemoryBlock<ComplexF> *y, unsigned int y_inc,
                                     unsigned int x_shift, unsigned int y_shift,
                                     bool use_gpu);
