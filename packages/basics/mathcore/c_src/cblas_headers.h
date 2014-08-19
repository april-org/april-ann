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

#include "complex_number.h"
#include "aligned_memory.h"

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
void cblas_ssbmv(CBLAS_ORDER order,
		 CBLAS_UPLO uplo,
		 int n, int k,
		 float alpha, const float *a, unsigned int a_lda,
		 const float *x, unsigned int x_inc,
		 float beta, float *y, unsigned int y_inc);
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

#endif // CBLAS_HEADERS_H
