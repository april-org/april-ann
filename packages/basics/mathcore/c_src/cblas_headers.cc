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

#include "cblas_headers.h"

#ifdef ADHOC_BLAS

#include "error_print.h"

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
#endif
