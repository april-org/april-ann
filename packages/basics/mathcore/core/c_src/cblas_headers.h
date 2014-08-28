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
#ifndef CBLAS_HEADERS_H
#define CBLAS_HEADERS_H

#include "aligned_memory.h"
#include "complex_number.h"
#include "gpu_mirrored_memory_block.h"

enum SPARSE_FORMAT { CSR_FORMAT=0, CSC_FORMAT=1, NONE_FORMAT=255 };

#ifdef USE_MKL
#ifdef USE_XCODE
#error "USE ONLY ONE OF: USE_MKL, USE_XCODE, NO_BLAS, or none of this"
#endif
#ifdef NO_BLAS
#error "USE ONLY ONE OF: USE_MKL, USE_XCODE, NO_BLAS, or none of this"
#endif
/////////////////////////////////// MKL ///////////////////////////////////////
extern "C" {
#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_vml.h>
#include <mkl_service.h>
}
// if you compile with MKL you do not need atlas
#define VECTOR_SSET(n, value, vec, step) for(unsigned int _i_=0,_j_=0;_j_<(n);++_j_,_i_+=(step))(vec)[_i_]=(value)
#define VECTOR_DSET(n, value, vec, step) for(unsigned int _i_=0,_j_=0;_j_<(n);++_j_,_i_+=(step))(vec)[_i_]=(value)
/*****************************************************************************/
#elif USE_XCODE
#ifdef NO_BLAS
#error "USE ONLY ONE OF: USE_MKL, USE_XCODE, NO_BLAS, or none of this"
#endif
////////////////////////////////// XCODE //////////////////////////////////////
#include <Accelerate/Accelerate.h>
#define VECTOR_SSET(n, value, vec, step) for(unsigned int _i_=0,_j_=0;_j_<(n);++_j_,_i_+=(step))(vec)[_i_]=(value)
#define VECTOR_DSET(n, value, vec, step) for(unsigned int _i_=0,_j_=0;_j_<(n);++_j_,_i_+=(step))(vec)[_i_]=(value)
/*****************************************************************************/
#else
#ifndef NO_BLAS
////////////////////////////////// ATLAS //////////////////////////////////////
extern "C" {
#include <atlas/cblas.h>
}
#define VECTOR_SSET(n, value, vec, step) catlas_sset((n), (value), (vec), (step))
#define VECTOR_DSET(n, value, vec, step) catlas_dset((n), (value), (vec), (step))
/*****************************************************************************/
#else
///////////////////////////////// AD-HOC //////////////////////////////////////
#define ADHOC_BLAS

#define VECTOR_SSET(n, value, vec, step) for(unsigned int _i_=0,_j_=0;_j_<(n);++_j_,_i_+=(step))(vec)[_i_]=(value)
#define VECTOR_DSET(n, value, vec, step) for(unsigned int _i_=0,_j_=0;_j_<(n);++_j_,_i_+=(step))(vec)[_i_]=(value)

enum CBLAS_ORDER     { CblasRowMajor = 0, CblasColMajor = 1 };
enum CBLAS_TRANSPOSE { CblasNoTrans  = 0, CblasTrans    = 1 };
enum CBLAS_UPLO      { CblasLower    = 0, CblasUpper    = 1 };

void cblas_sgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE a_transpose,
		 int m, int n,
		 float alpha, const float *a, unsigned int a_inc,
		 const float *x, unsigned int x_inc,
		 float beta, float *y, unsigned int y_inc);
void cblas_scopy(int N,
		 const float *x, unsigned int x_inc,
		 float *y, unsigned int y_inc);
void cblas_axpy(int N, float alpha,
		const float *x, unsigned int x_inc,
		float *y, unsigned int y_inc);
void cblas_sgemm(CBLAS_ORDER order,
		 CBLAS_TRANSPOSE a_transpose, CBLAS_TRANSPOSE b_transpose,
		 int m, int n, int k,
		 float alpha, const float *a, unsigned int a_inc,
		 const float *b, unsigned int b_inc,
		 float beta, float *c, unsigned int c_inc);
void cblas_sscal(unsigned int N, float alpha, float *x, unsigned int inc);
void cblas_sger(CBLAS_ORDER order,
		int m, int n,
		float alpha, const float *x, unsigned int x_inc,
		const float *y, unsigned int y_inc,
		float *a, unsigned int a_inc);
float cblas_sdot(unsigned int N,
		 const float *x, unsigned int x_inc,
		 const float *y, unsigned int y_inc);
float cblas_snrm2(unsigned int N, const float *x, unsigned int inc);
/*****************************************************************************/
#endif
#endif

#define NEGATE_CBLAS_TRANSPOSE(trans) ((trans) == CblasNoTrans)?CblasTrans:CblasNoTrans

// FIXME: MKL version is not working properly
//#ifndef USE_MKL

#if 1

// sparse BLAS is only available with CUDA or MKL
void cblas_saxpyi(int NNZ, float alpha,
		  const float *x_values_mem,
		  const int *x_indices_mem,
		  float *y_mem);
void cblas_daxpyi(int NNZ, double alpha,
		  const double *x_values_mem,
		  const int *x_indices_mem,
		  double *y_mem);
void cblas_caxpyi(int NNZ, const april_math::ComplexF *alpha,
		  const april_math::ComplexF *x_values_mem,
		  const int *x_indices_mem,
		  april_math::ComplexF *y_mem);
#endif

void cblas_sparse_mm(CBLAS_ORDER major_order,
                     SPARSE_FORMAT sparse_format,
		     CBLAS_TRANSPOSE a_transpose,
		     CBLAS_TRANSPOSE b_transpose,
		     CBLAS_TRANSPOSE c_transpose,
		     int m, int n, int k,
		     float alpha,
		     const float *a_values_mem,
		     const int *a_indices_mem,
		     const int *a_first_index_mem,
		     const float *b_mem, int b_inc,
		     float beta, float *c_mem, int c_inc);

void cblas_sparse_mm(CBLAS_ORDER major_order,
                     SPARSE_FORMAT sparse_format,
		     CBLAS_TRANSPOSE a_transpose,
		     CBLAS_TRANSPOSE b_transpose,
		     CBLAS_TRANSPOSE c_transpose,
		     int m, int n, int k,
		     double alpha,
		     const double *a_values_mem,
		     const int *a_indices_mem,
		     const int *a_first_index_mem,
		     const double *b_mem, int b_inc,
		     double beta, double *c_mem, int c_inc);

void cblas_sparse_mm(CBLAS_ORDER major_order,
                     SPARSE_FORMAT sparse_format,
		     CBLAS_TRANSPOSE a_transpose,
		     CBLAS_TRANSPOSE b_transpose,
		     CBLAS_TRANSPOSE c_transpose,
		     int m, int n, int k,
		     april_math::ComplexF alpha,
		     const april_math::ComplexF *a_values_mem,
		     const int *a_indices_mem,
		     const int *a_first_index_mem,
		     const april_math::ComplexF *b_mem, int b_inc,
		     april_math::ComplexF beta, april_math::ComplexF *c_mem, int c_inc);

void cblas_sparse_mv(SPARSE_FORMAT sparse_format,
		     CBLAS_TRANSPOSE a_transpose,
		     int m, int n,
		     float alpha,
		     const float *a_values_mem,
		     const int *a_indices_mem,
		     const int *a_first_index_mem,
		     const float *x_mem, int x_inc,
		     float beta, float *y_mem, int y_inc);

void cblas_sparse_mv(SPARSE_FORMAT sparse_format,
		     CBLAS_TRANSPOSE a_transpose,
		     int m, int n,
		     double alpha,
		     const double *a_values_mem,
		     const int *a_indices_mem,
		     const int *a_first_index_mem,
		     const double *x_mem, int x_inc,
		     double beta, double *y_mem, int y_inc);

void cblas_sparse_mv(SPARSE_FORMAT sparse_format,
		     CBLAS_TRANSPOSE a_transpose,
		     int m, int n,
		     april_math::ComplexF alpha,
		     const april_math::ComplexF *a_values_mem,
		     const int *a_indices_mem,
		     const int *a_first_index_mem,
		     const april_math::ComplexF *x_mem, int x_inc,
		     april_math::ComplexF beta, april_math::ComplexF *y_mem, int y_inc);

template<typename T>
T cblas_sparse_dot(int NNZ,
                   const T *x_values_mem,
                   const int *x_indices_mem,
                   const T *y_mem,
                   int y_inc) {
  T result = T();
  for (int i=0; i<NNZ; ++i) {
    int pos = x_indices_mem[i];
    int y_pos = pos * y_inc;
    result = result + y_mem[y_pos]*x_values_mem[i];
  }
  return result;
}

namespace april_math {
  
  ///////// CBLAS APRIL WRAPPERS ////////
  template<typename T>
  void doAxpy(int N,
              T alpha,
              const GPUMirroredMemoryBlock<T>* x,
              unsigned int x_inc,
              unsigned int x_shift,
              GPUMirroredMemoryBlock<T>* y,
              unsigned int y_inc,
              unsigned int y_shift,
              bool use_gpu);

  template<typename T>
  void doAxpyLoop(int N, T alpha,
                  GPUMirroredMemoryBlock<T>* x,
                  unsigned int x_inc,
                  unsigned int x_shift,
                  GPUMirroredMemoryBlock<T>* y,
                  unsigned int y_inc,
                  unsigned int y_shift,
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
  void doCopy(int N, const GPUMirroredMemoryBlock<T>* x,
              unsigned int x_inc,
              unsigned int x_shift,
              GPUMirroredMemoryBlock<T>* y,
              unsigned int y_inc,
              unsigned int y_shift,
              bool use_gpu);

  template<typename T>
  void doCopyBroadcast(int N,
                       GPUMirroredMemoryBlock<T>* x,
                       unsigned int x_inc,
                       GPUMirroredMemoryBlock<T>* A,
                       unsigned int A_inc,
                       unsigned int times,
                       const unsigned int A_stride,
                       bool use_gpu);

  template<typename T>
  T doDot(unsigned int size,
          const GPUMirroredMemoryBlock<T> *x,
          unsigned int x_inc,
          unsigned int x_shift,
          const GPUMirroredMemoryBlock<T> *y,
          unsigned int y_inc,
          unsigned int y_shift,
          bool use_gpu);

  template<typename T>
  T doSparseDot(int NNZ,
                const GPUMirroredMemoryBlock<T> *x_values,
                const Int32GPUMirroredMemoryBlock *x_indices,
                const GPUMirroredMemoryBlock<T> *y,
                int y_shift,
                int y_inc,
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
                  CBLAS_TRANSPOSE b_transpose,
                  CBLAS_TRANSPOSE c_transpose,
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
  
  // BLAS FUNCTIONS
  template<typename T>
  void doGemv(CBLAS_ORDER major_type, CBLAS_TRANSPOSE a_transpose,
              int m, int n,
              T alpha, const GPUMirroredMemoryBlock<T> *a, unsigned int a_inc,
              const GPUMirroredMemoryBlock<T> *x, unsigned int x_inc,
              T beta, GPUMirroredMemoryBlock<T> *y, unsigned int y_inc,
              unsigned int a_shift, unsigned int x_shift, unsigned int y_shift,
              bool use_gpu);
  
  template<typename T>
  void doSparseGemv(SPARSE_FORMAT sparse_format,
                    CBLAS_TRANSPOSE a_transpose,
                    int m, int n,
                    T alpha,
                    const GPUMirroredMemoryBlock<T> *a_values,
                    const Int32GPUMirroredMemoryBlock *a_indices,
                    const Int32GPUMirroredMemoryBlock *a_first_index,
                    const GPUMirroredMemoryBlock<T> *x, unsigned int x_inc,
                    T beta, GPUMirroredMemoryBlock<T> *y, unsigned int y_inc,
                    unsigned int x_shift, unsigned int y_shift,
                    bool use_gpu);
  
  template<typename T>
  void doGer(CBLAS_ORDER major_type,
             unsigned int m,
             unsigned int n,
             T alpha,
             const GPUMirroredMemoryBlock<T> *x,
             unsigned int x_shift,
             unsigned int x_inc,
             const GPUMirroredMemoryBlock<T> *y,
             unsigned int y_shift,
             unsigned int y_inc,
             GPUMirroredMemoryBlock<T> *a,
             unsigned int a_shift,
             unsigned int a_inc,
             bool use_gpu);

  template<typename T>
  float doNrm2(unsigned int n,
               const GPUMirroredMemoryBlock<T> *x,
               unsigned int inc,
               unsigned int shift,
               bool use_gpu);

  template<typename T>
  void doScal(unsigned int size,
              GPUMirroredMemoryBlock<T> *x,
              unsigned int inc,
              unsigned int shift,
              T alpha,
              bool use_gpu);
  
  //////// CURRIED VERSIONS /////////
  
  template<typename T>
  struct CurriedAxpy {
    const T alpha;
    CurriedAxpy(const T &alpha) : alpha(alpha) { }
    void operator()(unsigned int N,
                    const GPUMirroredMemoryBlock<T> *input,
                    unsigned int input_stride,
                    unsigned int input_shift,
                    GPUMirroredMemoryBlock<T> *output,
                    unsigned int output_stride,
                    unsigned int output_shift,
                    bool use_cuda) {
      doAxpy(static_cast<int>(N), alpha,
             input, input_stride, input_shift,
             output, output_stride, output_shift,
             use_cuda);
    }
  };

  template<typename T>
  struct CurriedScal {
    const T alpha;
    CurriedScal(const T &alpha) : alpha(alpha) { }
    void operator()(unsigned int N,
                    const GPUMirroredMemoryBlock<T> *input,
                    unsigned int input_stride,
                    unsigned int input_shift,
                    GPUMirroredMemoryBlock<T> *output,
                    unsigned int output_stride,
                    unsigned int output_shift,
                    bool use_cuda) {
      if (input != output) {
        ERROR_EXIT(128, "Scal is always an in-place operation\n");
      }
      april_assert(input_stride == output_stride);
      april_assert(input_shift == output_shift);
      UNUSED_VARIABLE(input);
      UNUSED_VARIABLE(input_stride);
      UNUSED_VARIABLE(input_shift);
      doScal(N, output, output_stride, output_shift, alpha, use_cuda);
    }
  };
  
} // namespace april_math

#endif // CBLAS_HEADERS_H
