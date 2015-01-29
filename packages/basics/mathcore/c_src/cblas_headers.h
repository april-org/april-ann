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
void cblas_caxpyi(int NNZ, const AprilMath::ComplexF *alpha,
		  const AprilMath::ComplexF *x_values_mem,
		  const int *x_indices_mem,
		  AprilMath::ComplexF *y_mem);
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
		     AprilMath::ComplexF alpha,
		     const AprilMath::ComplexF *a_values_mem,
		     const int *a_indices_mem,
		     const int *a_first_index_mem,
		     const AprilMath::ComplexF *b_mem, int b_inc,
		     AprilMath::ComplexF beta, AprilMath::ComplexF *c_mem, int c_inc);

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
		     AprilMath::ComplexF alpha,
		     const AprilMath::ComplexF *a_values_mem,
		     const int *a_indices_mem,
		     const int *a_first_index_mem,
		     const AprilMath::ComplexF *x_mem, int x_inc,
		     AprilMath::ComplexF beta, AprilMath::ComplexF *y_mem, int y_inc);

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

namespace AprilMath {
  
  ///////// CBLAS APRIL WRAPPERS ////////
  
  /**
   * @brief Template wrapper for AXPY CBLAS operation.
   *
   * It computes y = y + alpha * x, where y and x are vectors with size N.
   *
   * @tparam T - The data type. In APRIL-ANN it is available for float, double,
   * Basics::ComplexF.
   *
   * @param N - The number of elements in both vectors.
   * @param alpha - The alpha value of the AXPY operation.
   * @param x - The vector x in AXPY operation. It is given as a
   * GPUMirroredMemoryBlock pointer.
   * @param x_inc - The stride between two consecutive elements in x vector.
   * @param x_shift - The first valid position at x pointer.
   * @param[in,out] y - The vector y in AXPY operation. It is given as a
   * GPUMirroredMemoryBlock pointer.
   * @param y_inc - The stride between two consecutive elements in y vector.
   * @param y_shift - The first valid position at y pointer.
   * @param use_gpu - Indicates if to use GPU computation or not.
   */
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

  /**
   * @brief Template wrapper for sparse ºqAXPY CBLAS operation.
   *
   * It computes y = y + alpha * x, where y is a dense vector and x is a sparse
   * vector with NNZ number of non-zero elements, and y length is at least NNZ.
   *
   * @tparam T - The data type. In APRIL-ANN it is available for float, double,
   * Basics::ComplexF.
   *
   * @param NNZ - The number of non-zero elements in y sparse vector.
   * @param alpha - The alpha value of the sparse AXPY operation.
   * @param x_values - A pointer to GPUMirroredMemoryBlock with values of x
   * sparse vector.
   * @param x_indices - A pointer to a Int32GPUMirroredMemoryBlock with the indices
   * of the values in x_value parameter.
   * @param[in,out] y - The vector y in AXPY operation. It is given as a
   * GPUMirroredMemoryBlock pointer.
   * @param x_shift - The first valid position at x pointers (both of them).
   * @param y_shift - The first valid position at y pointer.
   * @param y_inc - The stride between two consecutive elements in y vector.
   * @param use_gpu - Indicates if to use GPU computation or not.
   *
   * @note The sparse vector is given by x_values and x_indices memory blocks,
   * using a stride of 1 and an offset of 0.
   */
  template<typename T>
  void doSparseAxpy(int NNZ,
                    T alpha,
                    const GPUMirroredMemoryBlock<T> *x_values,
                    const Int32GPUMirroredMemoryBlock *x_indices,
                    GPUMirroredMemoryBlock<T>* y,
                    unsigned int x_shift,
                    unsigned int y_shift,
                    unsigned int y_inc,
                    bool use_gpu);
  
  
  /**
   * @brief Template wrapper for sparse copy CBLAS operation.
   *
   * It computes y = x, where y and x are vectors with size N.
   *
   * @tparam T - The data type. In APRIL-ANN it is available for float, double,
   * Basics::ComplexF.
   *
   * @param NNZ - The number of non-zero elements in y sparse vector.
   * @param x - A pointer to GPUMirroredMemoryBlock with values of x vector.
   * @param x_inc - The stride between consecutive elements in x.
   * @param x_shift - The position of the first valid element in x pointer.
   * @param[in,out] y - The vector y in copy operation. It is given as a
   * GPUMirroredMemoryBlock pointer.
   * @param y_inc - The stride between two consecutive elements in y vector.
   * @param y_shift - The first valid position at y pointer.
   * @param use_gpu - Indicates if to use GPU computation or not.
   */
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

  /**
   * @brief Template wrapper for GEMM operation using CBLAS interface.
   *
   * GEMM computes C = alpha * op(A) * op(B) + beta * C where A, B and C are
   * matrices, op is tranposition operator, and alpha and beta are scalars. C is
   * a matrix with MxN size. op(A) will be MxK and op(B) will be KxN.
   *
   * @tparam T - The data type. In APRIl-ANN it could be float, double,
   * ComplexF.
   *
   * @param major_type - The expected major_order in the matrices.
   * @param a_transpose - Indicates if A matrix must be transposed.
   * @param b_transpose - Indicates if B matrix must be transposed.
   * @param m - The number of rows in C.
   * @param n - The number of columns in C.
   * @param k - The common dimension for A and B.
   * @param alpha - The alpha scalar of GEMM operation.
   * @param a - The A matrix given as a GPUMirroredMemoryBlock pointer.
   * @param a_inc - The stride of the leading dimension of A.
   * @param b - The B matrix given as a GPUMirroredMemoryBlock pointer.
   * @param b_inc - The stride of the leading dimension of B.
   * @param beta - The beta scalar of GEMM operation.
   * @param[in,out] c - The matrix C given as a GPUMirroredMemoryBlock pointer.
   * @param c_inc - The stride of the leading dimension of C.
   * @param a_shift - The first valid position of matrix A pointer.
   * @param b_shift - The first valid position of matrix B pointer.
   * @param c_shift - The first valid position of matrix C pointer.
   * @param use_gpu - Indicates if use GPU or not for the computation.
   */
  template<typename T>
  void doGemm(CBLAS_ORDER major_type, CBLAS_TRANSPOSE a_transpose,
              CBLAS_TRANSPOSE b_transpose, int m, int n, int k, T alpha,
              const GPUMirroredMemoryBlock<T>* a, unsigned int a_inc,
              const GPUMirroredMemoryBlock<T>* b, unsigned int b_inc, T beta,
              GPUMirroredMemoryBlock<T>* c, unsigned int c_inc,
              unsigned int a_shift, unsigned int b_shift, unsigned int c_shift,
              bool use_gpu);

  /**
   * @brief Template wrapper for sparse MM operation.
   *
   * SparseMM computes C = alpha * op(A) * op(B) + beta * C where A is a
   * sparse matrix given in CSR or CSC formats, B and C are dense matrices, op
   * is tranposition operator, and alpha and beta are scalars. C is a matrix
   * with MxN size. op(A) will be MxK and op(B) will be KxN.
   *
   * @see Basics::SparseMatrix class documentation.
   *
   * @tparam T - The data type. In APRIl-ANN it could be float, double,
   * ComplexF.
   *
   * @param major_type - The expected major_order in the dense matrices.
   * @param sparse_format - The format can be CSR_FORMAT or CSC_FORMAT.
   * @param a_transpose - Indicates if A matrix must be transposed.
   * @param b_transpose - Indicates if B matrix must be transposed.
   * @param m - The number of rows in C.
   * @param n - The number of columns in C.
   * @param k - The common dimension for A and B.
   * @param alpha - The alpha scalar of GEMM operation.
   * @param a_values - The A matrix values given as a GPUMirroredMemoryBlock pointer.
   * @param a_indices - The A matrix indices given as a Int32GPUMirroredMemoryBlock pointer.
   * @param a_first_index - The A matrix first indices given as a Int32GPUMirroredMemoryBlock pointer.
   * @param b - The B matrix given as a GPUMirroredMemoryBlock pointer.
   * @param b_inc - The stride of the leading dimension of B.
   * @param beta - The beta scalar of GEMM operation.
   * @param[in,out] c - The matrix C given as a GPUMirroredMemoryBlock pointer.
   * @param c_inc - The stride of the leading dimension of C.
   * @param b_shift - The first valid position of matrix B pointer.
   * @param c_shift - The first valid position of matrix C pointer.
   * @param use_gpu - Indicates if use GPU or not for the computation.
   */
  template <typename T>
  void doSparseMM(CBLAS_ORDER major_order,
                  SPARSE_FORMAT sparse_format,
                  CBLAS_TRANSPOSE a_transpose,
                  CBLAS_TRANSPOSE b_transpose,
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

  /**
   * @brief Template wrapper for GEMV CBLAS operation.
   *
   * GEMV computex Y = alpha * op(A)*X + beta*Y, where X and Y are vectors and A
   * is a matrix, op is tranposition operator, and alpha and beta are scalars. A
   * is of MxN size, Y is a N vector and X is a M vector.
   *
   * @tparam T - The data type. In APRIl-ANN it could be float, double,
   * ComplexF.
   *
   * @param major_type - The expected major_order in the dense matrices.
   * @param a_transpose - Indicates if A matrix must be transposed.
   * @param m - The number of rows in A.
   * @param n - The number of columns in A.
   * @param alpha - The alpha scalar of GEMV operation.
   * @param a - The A matrix given as a GPUMirroredMemoryBlock pointer.
   * @param a_inc - The stride for the leading dimension of A matrix.
   * @param x - The X vector given as a GPUMirroredMemoryBlock pointer.
   * @param x_inc - The stride between consecutive elements of X.
   * @param beta - The beta scalar of GEMV operation.
   * @param y[in,out] - The Y vector given as a GPUMirroredMemoryBlock pointer.
   * @param y_inc - The stride between consecutive elements of Y.
   * @param a_shift - The first valid position of matrix A pointer.
   * @param x_shift - The first valid position of matrix X pointer.
   * @param y_shift - The first valid position of matrix Y pointer.
   * @param use_gpu - Indicates if use GPU or not for the computation.
   */
  template<typename T>
  void doGemv(CBLAS_ORDER major_type, CBLAS_TRANSPOSE a_transpose,
              int m, int n,
              T alpha, const GPUMirroredMemoryBlock<T> *a, unsigned int a_inc,
              const GPUMirroredMemoryBlock<T> *x, unsigned int x_inc,
              T beta, GPUMirroredMemoryBlock<T> *y, unsigned int y_inc,
              unsigned int a_shift, unsigned int x_shift, unsigned int y_shift,
              bool use_gpu);

  /**
   * @brief Template wrapper for sparse GEMV CBLAS operation.
   *
   * GEMV computex Y = alpha * op(A)*X + Y, where X and Y are vectors and A is a
   * sparse matrix, op is tranposition operator, and alpha and beta are
   * scalars. A is of MxN size, Y is a N vector and X is a M vector.
   *
   * @tparam T - The data type. In APRIl-ANN it could be float, double,
   * ComplexF.
   *
   * @param sparse_format - The sparse format for A matrix, CSR_FORMAT or CSC_FORMAT.
   * @param a_transpose - Indicates if A matrix must be transposed.
   * @param m - The number of rows in A.
   * @param n - The number of columns in A.
   * @param alpha - The alpha scalar of GEMV operation.
   * @param a_values - The A matrix values given as a GPUMirroredMemoryBlock pointer.
   * @param a_indices - The A matrix indices given as a Int32GPUMirroredMemoryBlock pointer.
   * @param a_first_index - The A matrix first indices given as a Int32GPUMirroredMemoryBlock pointer.
   * @param x - The X vector given as a GPUMirroredMemoryBlock pointer.
   * @param x_inc - The stride between consecutive elements of X.
   * @param beta - The beta scalar of sparse GEMV operation.
   * @param y[in,out] - The Y vector given as a GPUMirroredMemoryBlock pointer.
   * @param y_inc - The stride between consecutive elements of Y.
   * @param x_shift - The first valid position of matrix X pointer.
   * @param y_shift - The first valid position of matrix Y pointer.
   * @param use_gpu - Indicates if use GPU or not for the computation.
   */  
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
                    bool use_cuda) const {
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
                    bool use_cuda) const {
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
  
} // namespace AprilMath

#include "dot.h"
#include "nrm2.h"


#endif // CBLAS_HEADERS_H
