/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Francisco Zamora-Martinez
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

#include "april_assert.h"
#include "unused_variable.h"
#include "cblas_headers.h"
#include "error_print.h"

#ifdef ADHOC_BLAS

void cblas_sgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE a_transpose,
		 int m, int n,
		 float alpha, const float *a, unsigned int a_inc,
		 const float *x, unsigned int x_inc,
		 float beta, float *y, unsigned int y_inc) {
  if (m == 0 || n == 0) return;
  if (alpha == 0.0f && beta == 1.0f) return;
  int cols, rows;
  if (a_transpose == CblasNoTrans) {
    cols = n;
    rows = m;
  }
  else {
    cols = m;
    rows = n;
  }
 
  // y = beta * y
  if (beta == 0.0f) {
    unsigned int y_pos = 0;
    for (int i=0; i<rows; ++i, y_pos+=y_inc) y[y_pos] = 0.0f;
  }
  else if (beta != 1.0f) {
    unsigned int y_pos = 0;
    for (int i=0; i<rows; ++i, y_pos+=y_inc) y[y_pos] *= beta;
  }
 
  // y += alpha*A*x
  if (alpha != 0.0f) {
    if ( (order == CblasRowMajor && a_transpose == CblasNoTrans) ||
	 (order == CblasColMajor && a_transpose == CblasTrans) ) {
      unsigned int y_pos = 0;
      for (int i=0; i<rows; ++i, y_pos+=y_inc, a += a_inc) {
	float aux = 0.0f;
	unsigned int x_pos = 0;
	for (int j=0; j<cols; ++j, x_pos += x_inc) aux += x[x_pos] * a[j];
	y[y_pos] += alpha * aux;
      }
    }
    else if ( (order == CblasRowMajor && a_transpose == CblasTrans) ||
	      (order == CblasColMajor && a_transpose == CblasNoTrans) ) {
      unsigned int x_pos = 0;
      for (int j=0; j<cols; ++j, x_pos += x_inc) {
	float aux = alpha * x[x_pos];
	if (aux != 0.0f) {
	  unsigned int y_pos = 0;
	  for (int i=0; i<rows; ++i, y_pos+=y_inc)
	    y[y_pos] += aux * a[ j*a_inc + i];
	}
      }
    }
    else ERROR_EXIT(128, "Incorrect BLAS parameters\n");
  }
}

void cblas_scopy(int N,
		 const float *x, unsigned int x_inc,
		 float *y, unsigned int y_inc) {
  for (int i=0; i<N; ++i, x+=x_inc, y+=y_inc) *y = *x;
}

void cblas_axpy(int N, float alpha,
		const float *x, unsigned int x_inc,
		float *y, unsigned int y_inc) {
  for (int i=0; i<N; ++i, x+=x_inc, y+=y_inc) *y += alpha * (*x);
}

void cblas_sgemm(CBLAS_ORDER order,
		 CBLAS_TRANSPOSE a_transpose, CBLAS_TRANSPOSE b_transpose,
		 int m, int n, int k,
		 float alpha, const float *a, unsigned int a_inc,
		 const float *b, unsigned int b_inc,
		 float beta, float *c, unsigned int c_inc) {
  ERROR_EXIT(128, "NOT IMPLEMENTED\n");
}

void cblas_sscal(unsigned int N, float alpha, float *x, unsigned int inc) {
  for (unsigned int i=0; i<N; ++i, x+=inc) *x *= alpha;
}

void cblas_sger(CBLAS_ORDER order,
		int m, int n,
		float alpha, const float *x, unsigned int x_inc,
		const float *y, unsigned int y_inc,
		float *a, unsigned int a_inc) {
  ERROR_EXIT(128, "NOT IMPLEMENTED\n");
}

float cblas_sdot(unsigned int N,
		 const float *x, unsigned int x_inc,
		 const float *y, unsigned int y_inc) {
  float sum=0.0f;
  for (unsigned int i=0; i<N; ++i, x+=x_inc, y+=y_inc) sum += (*x) * (*y);
}

float cblas_snrm2(unsigned int N, const float *x, unsigned int inc) {
  float sum2=0.0f;
  for (unsigned int i=0; i<N; ++i, x+=inc) sum2 += (*x) * (*x);
  return sqrtf(sum2);
}

void cblas_ssbmv(CBLAS_ORDER order,
		 CBLAS_UPLO uplo,
		 int n, int k,
		 float alpha, const float *a, unsigned int a_lda,
		 const float *x, unsigned int x_inc,
		 float beta, float *y, unsigned int y_inc) {
  ERROR_EXIT(128, "NOT IMPLEMENTED\n");
}

#endif // ADHOC_BLAS


#ifndef USE_MKL
// sparse BLAS is only available with CUDA or MKL

// generic templates
template<typename T>
void generic_cblas_axpyi(int NNZ, T alpha,
			 const T *x_values_mem,
			 const int *x_indices_mem,
			 T *y_mem) {
  for (int i=0; i<NNZ; ++i) {
    int idx = x_indices_mem[i];
    y_mem[idx] += alpha * x_values_mem[i];
  }
}

template<typename T>
void generic_cblas_sparse_mm(SPARSE_FORMAT sparse_format,
                             CBLAS_TRANSPOSE a_transpose,
                             int m, int n, int k,
                             T alpha,
                             const T *a_values_mem,
                             const int *a_indices_mem,
                             const int *a_first_index_mem,
                             const T *b_mem, int b_inc,
                             T beta, T *c_mem, int c_inc) {
  UNUSED_VARIABLE(k);
  if (a_transpose == CblasTrans) {
    if (sparse_format == CSR_FORMAT) sparse_format = CSC_FORMAT;
    else sparse_format = CSR_FORMAT;
  }
  if (sparse_format == CSR_FORMAT) {
    // for each destination row
    for (int i=0; i<m; ++i) {
      int first  = a_first_index_mem[i];
      int lastp1 = a_first_index_mem[i+1]; // last plus 1
      int c_pos  = i*c_inc;
      printf("%d\n", c_pos);
      // for each destination col
      for (int j=0; j<n; ++j, ++c_pos) {
        T aux = T();
        for (int x=first; x<lastp1; ++x) {
          int c1    = a_indices_mem[x];
          int b_pos = c1*b_inc + j;
          april_assert(0 <= c1 && c1 < k);
          aux = aux + a_values_mem[x] * b_mem[b_pos];
        }
        if (beta == T()) c_mem[c_pos] = alpha * aux;
        else c_mem[c_pos] = beta*c_mem[c_pos] + alpha*aux;
      }
    }
  }
  else if (sparse_format == CSC_FORMAT) {
    // first C matrix needs to be initialized
    if (beta == T()) {
      for (int i=0; i<m; ++i) {
        int c_pos  = i*c_inc;
        for (int j=0; j<n; ++j, ++c_pos) {
          c_mem[c_pos] = T();
        }
      }
    }
    else {
      for (int i=0; i<m; ++i) {
        int c_pos  = i*c_inc;
        for (int j=0; j<n; ++j, ++c_pos) {
          c_mem[c_pos] = c_mem[c_pos] * beta;
        }
      }
    }
    // for each destination col
    for (int j=0; j<n; ++j) {
      // for each A col
      for (int i=0; i<k; ++i) {
        int first  = a_first_index_mem[i];
        int lastp1 = a_first_index_mem[i+1]; // last plus 1
        // for each sparse destination row
        for (int x=first; x<lastp1; ++x) {
          int c0    = a_indices_mem[x];
          int b_pos = c0*b_inc + j;
          int c_pos = x*c_inc  + j;
          c_mem[c_pos] = c_mem[c_pos] + alpha * a_values_mem[x] * b_mem[b_pos];
        }
      }
    }
  }
}

// cblas function implementations
void cblas_saxpyi(int NNZ, float alpha,
		  const float *x_values_mem,
		  const int *x_indices_mem,
		  float *y_mem) {
  generic_cblas_axpyi(NNZ, alpha, x_values_mem, x_indices_mem, y_mem);
}
void cblas_daxpyi(int NNZ, double alpha,
		  const double *x_values_mem,
		  const int *x_indices_mem,
		  double *y_mem) {
  generic_cblas_axpyi(NNZ, alpha, x_values_mem, x_indices_mem, y_mem);
}
void cblas_caxpyi(int NNZ, const ComplexF *alpha,
		  const ComplexF *x_values_mem,
		  const int *x_indices_mem,
		  ComplexF *y_mem) {
  generic_cblas_axpyi(NNZ, *alpha, x_values_mem, x_indices_mem, y_mem);
}
void cblas_sparse_mm(SPARSE_FORMAT sparse_format,
		     CBLAS_TRANSPOSE a_transpose,
		     int m, int n, int k,
		     float alpha,
		     const float *a_values_mem,
		     const int *a_indices_mem,
		     const int *a_first_index_mem,
		     const float *b_mem, int b_inc,
		     float beta, float *c_mem, int c_inc) {
  generic_cblas_sparse_mm(sparse_format,
                          a_transpose,
                          m,n,k,
                          alpha,
                          a_values_mem, a_indices_mem, a_first_index_mem,
                          b_mem, b_inc,
                          beta,
                          c_mem, c_inc);
}
void cblas_sparse_mm(SPARSE_FORMAT sparse_format,
		     CBLAS_TRANSPOSE a_transpose,
		     int m, int n, int k,
		     double alpha,
		     const double *a_values_mem,
		     const int *a_indices_mem,
		     const int *a_first_index_mem,
		     const double *b_mem, int b_inc,
		     double beta, double *c_mem, int c_inc) {
  generic_cblas_sparse_mm(sparse_format,
                          a_transpose,
                          m,n,k,
                          alpha,
                          a_values_mem, a_indices_mem, a_first_index_mem,
                          b_mem, b_inc,
                          beta,
                          c_mem, c_inc);
}
void cblas_sparse_mm(SPARSE_FORMAT sparse_format,
		     CBLAS_TRANSPOSE a_transpose,
		     int m, int n, int k,
		     ComplexF alpha,
		     const ComplexF *a_values_mem,
		     const int *a_indices_mem,
		     const int *a_first_index_mem,
		     const ComplexF *b_mem, int b_inc,
		     ComplexF beta, ComplexF *c_mem, int c_inc) {
  generic_cblas_sparse_mm(sparse_format,
                          a_transpose,
                          m,n,k,
                          alpha,
                          a_values_mem, a_indices_mem, a_first_index_mem,
                          b_mem, b_inc,
                          beta,
                          c_mem, c_inc);
}
#else
#include <mkl_spblas.h>
void cblas_sparse_mm(SPARSE_FORMAT sparse_format,
		     CBLAS_TRANSPOSE a_transpose,
		     int m, int n, int k,
		     float alpha,
		     const float *a_values_mem,
		     const int *a_indices_mem,
		     const int *a_first_index_mem,
		     const float *b_mem, int b_inc,
		     float beta, float *c_mem, int c_inc) {
  char descrA[6]; descrA[0] = 'g'; descrA[3]='c';
  char trans = (a_transpose == CblasTrans) ? 't' : 'n';
  switch(sparse_format) {
  case CSR_FORMAT:
    mkl_scsrmm(&trans,
	       &m, &n, &k,
	       &alpha,
	       descrA,
	       const_cast<float*>(a_values_mem),
               const_cast<int*>(a_indices_mem),
	       const_cast<int*>(a_first_index_mem),
               const_cast<int*>(a_first_index_mem+1),
	       const_cast<float*>(b_mem), &b_inc,
	       &beta, const_cast<float*>(c_mem), &c_inc);
    break;
  case CSC_FORMAT:
    mkl_scscmm(&trans,
	       &m, &n, &k,
	       &alpha,
	       descrA,
	       const_cast<float*>(a_values_mem),
               const_cast<int*>(a_indices_mem),
	       const_cast<int*>(a_first_index_mem),
               const_cast<int*>(a_first_index_mem+1),
	       const_cast<float*>(b_mem), &b_inc,
	       &beta, const_cast<float*>(c_mem), &c_inc);
    break;
  default:
    ERROR_EXIT(128, "Incorrect sparse format\n");
  }
}
void cblas_sparse_mm(SPARSE_FORMAT sparse_format,
		     CBLAS_TRANSPOSE a_transpose,
		     int m, int n, int k,
		     double alpha,
		     const double *a_values_mem,
		     const int *a_indices_mem,
		     const int *a_first_index_mem,
		     const double *b_mem, int b_inc,
		     double beta, double *c_mem, int c_inc) {
  char descrA[6]; descrA[0] = 'g'; descrA[3]='c';
  char trans = (a_transpose == CblasTrans) ? 't' : 'n';
  switch(sparse_format) {
  case CSR_FORMAT:
    mkl_dcsrmm(&trans,
	       &m, &n, &k,
	       &alpha,
	       descrA,
	       const_cast<double*>(a_values_mem),
               const_cast<int*>(a_indices_mem),
	       const_cast<int*>(a_first_index_mem),
               const_cast<int*>(a_first_index_mem+1),
	       const_cast<double*>(b_mem), &b_inc,
	       &beta, const_cast<double*>(c_mem), &c_inc);
    break;
  case CSC_FORMAT:
    mkl_dcscmm(&trans,
	       &m, &n, &k,
	       &alpha,
	       descrA,
	       const_cast<double*>(a_values_mem),
               const_cast<int*>(a_indices_mem),
	       const_cast<int*>(a_first_index_mem),
               const_cast<int*>(a_first_index_mem+1),
	       const_cast<double*>(b_mem), &b_inc,
	       &beta, const_cast<double*>(c_mem), &c_inc);
    break;
  default:
    ERROR_EXIT(128, "Incorrect sparse format\n");
  }
}
void cblas_sparse_mm(SPARSE_FORMAT sparse_format,
		     CBLAS_TRANSPOSE a_transpose,
		     int m, int n, int k,
		     ComplexF alpha,
		     const ComplexF *a_values_mem,
		     const int *a_indices_mem,
		     const int *a_first_index_mem,
		     const ComplexF *b_mem, int b_inc,
		     ComplexF beta, ComplexF *c_mem, int c_inc) {
  char descrA[6]; descrA[0] = 'g'; descrA[3]='c';
  char trans = (a_transpose == CblasTrans) ? 't' : 'n';
  switch(sparse_format) {
  case CSR_FORMAT:
    mkl_ccsrmm(&trans,
	       &m, &n, &k,
	       reinterpret_cast<MKL_Complex8*>(&alpha),
	       descrA,
	       reinterpret_cast<MKL_Complex8*>(const_cast<ComplexF*>(a_values_mem)),
               const_cast<int*>(a_indices_mem),
	       const_cast<int*>(a_first_index_mem),
               const_cast<int*>(a_first_index_mem+1),
	       reinterpret_cast<MKL_Complex8*>(const_cast<ComplexF*>(b_mem)),
               &b_inc,
               reinterpret_cast<MKL_Complex8*>(&beta),
               reinterpret_cast<MKL_Complex8*>(const_cast<ComplexF*>(c_mem)),
               &c_inc);
    break;
  case CSC_FORMAT:
    mkl_ccscmm(&trans,
	       &m, &n, &k,
	       reinterpret_cast<MKL_Complex8*>(&alpha),
	       descrA,
	       reinterpret_cast<MKL_Complex8*>(const_cast<ComplexF*>(a_values_mem)),
               const_cast<int*>(a_indices_mem),
	       const_cast<int*>(a_first_index_mem),
               const_cast<int*>(a_first_index_mem+1),
	       reinterpret_cast<MKL_Complex8*>(const_cast<ComplexF*>(b_mem)),
               &b_inc,
	       reinterpret_cast<MKL_Complex8*>(&beta),
               reinterpret_cast<MKL_Complex8*>(const_cast<ComplexF*>(c_mem)),
               &c_inc);
    break;
  default:
    ERROR_EXIT(128, "Incorrect sparse format\n");
  }
}
#endif
