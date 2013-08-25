 /*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Francisco Zamora-Martinez
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
#include "lapack_headers.h"
#include "error_print.h"

#if defined(USE_MKL) || defined(USE_XCODE)
#include "cblas_headers.h"
int clapack_sgetrf(const int Order, const int M, const int N,
                   float *A, const int lda, int *ipiv) {
  if (Order != CblasColMajor)
    ERROR_EXIT(256, "Only col_major order is allowed\n");
  int INFO;
  sgetrf_(&M,&N,A,&lda,ipiv,&INFO);
  return INFO;
}
int clapack_sgetri(const int Order, const int N,
                   float *A, const int lda, int *ipiv) {
  if (Order != CblasColMajor)
    ERROR_EXIT(256, "Only col_major order is allowed\n");
  int INFO;
  int LWORK = N*N;
  float *WORK = new float[LWORK];
  sgetri_(&N,A,&lda,ipiv,WORK,&LWORK,&INFO);
  delete[] WORK;
  return INFO;
}
#endif

void checkLapackInfo(int info) {
  if (info < 0)
    ERROR_EXIT1(128, "The %d argument had an ilegal value\n", -info);
  else if (info > 0)
    ERROR_EXIT(128, "The matrix is singular and inverse can't be computed\n");
}
