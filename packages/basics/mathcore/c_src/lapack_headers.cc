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
#include "unused_variable.h"
#include "lapack_headers.h"
#include "error_print.h"

#if defined(USE_MKL)
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
int clapack_sgesdd(const int Order, const int M, const int N, const int LDA,
		   float *A, float *U, float *S, float *VT) {
  if (Order != CblasColMajor)
    ERROR_EXIT(256, "Only col_major order is allowed\n");
  const int numSV = (M<N) ? M : N;
  // workspace
  float workSize;
  float *work = &workSize;
  int lwork = -1;
  int *iwork = new int[2*numSV];
  int info = 0;
  // call sgesdd_ for workspace size computation
  sgesdd_("A", &M, &N, A, &LDA, S, U, &M, VT, &N, work, &lwork, iwork, &info);
  // optimal workspace size is in work[0]
  lwork = workSize;
  work = new float[lwork];
  // computation
  sgesdd_("A", &M, &N, A, &LDA, S, U, &M, VT, &N, work, &lwork, iwork, &info);
  // free auxiliary data
  delete[] work;
  delete[] iwork;
  return info;
}
#elif defined(USE_XCODE)
#include "cblas_headers.h"
int clapack_sgetrf(int Order, int M, int N,
                   float *A, int lda, int *ipiv) {
  if (Order != CblasColMajor)
    ERROR_EXIT(256, "Only col_major order is allowed\n");
  int INFO;
  sgetrf_(&M,&N,A,&lda,ipiv,&INFO);
  return INFO;
}
int clapack_sgetri(int Order, int N,
                   float *A, int lda, int *ipiv) {
  if (Order != CblasColMajor)
    ERROR_EXIT(256, "Only col_major order is allowed\n");
  int INFO;
  int LWORK = N*N;
  float *WORK = new float[LWORK];
  sgetri_(&N,A,&lda,ipiv,WORK,&LWORK,&INFO);
  delete[] WORK;
  return INFO;
}
int clapack_sgesdd(int Order, int M, int N, int LDA,
                   float *A, float *U, float *S, float *VT) {
  if (Order != CblasColMajor)
    ERROR_EXIT(256, "Only col_major order is allowed\n");
  const int numSV = (M<N) ? M : N;
  // workspace
  float workSize;
  float *work = &workSize;
  int lwork = -1;
  int *iwork = new int[2*numSV];
  int info = 0;
  // call sgesdd_ for workspace size computation
  sgesdd_("A", &M, &N, A, &LDA, S, U, &M, VT, &N, work, &lwork, iwork, &info);
  // optimal workspace size is in work[0]
  lwork = workSize;
  work = new float[lwork];
  // computation
  sgesdd_("A", &M, &N, A, &LDA, S, U, &M, VT, &N, work, &lwork, iwork, &info);
  // free auxiliary data
  delete[] work;
  delete[] iwork;
  return info;
}
#else
int clapack_sgesdd(const int Order, const int M, const int N, const int LDA,
		   float *A, float *U, float *S, float *VT) {
  UNUSED_VARIABLE(Order);
  UNUSED_VARIABLE(M);
  UNUSED_VARIABLE(N);
  UNUSED_VARIABLE(LDA);
  UNUSED_VARIABLE(A);
  UNUSED_VARIABLE(U);
  UNUSED_VARIABLE(S);
  UNUSED_VARIABLE(VT);
  ERROR_EXIT("SGESDD FUNCTION NOT IMPLEMENTED IN ATLAS CLAPACK\n");
  return 0;
}
#endif

void checkLapackInfo(int info) {
  if (info < 0)
    ERROR_EXIT1(128, "The %d argument had an ilegal value\n", -info);
  else if (info > 0)
    ERROR_EXIT(128, "The matrix is singular and inverse can't be computed\n");
}
