/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2014, Francisco Zamora-Martinez
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
#ifndef MATRIX_NOT_IMPLEMENTED_H
#define MATRIX_NOT_IMPLEMENTED_H

#include "unused_variable.h"
#include "error_print.h"

#define NOT_IMPLEMENT_AXPY(TYPE)					\
  template <>								\
  void Matrix<TYPE>::axpy(TYPE alpha, const Matrix<TYPE> *other) {	\
    UNUSED_VARIABLE(alpha);						\
    UNUSED_VARIABLE(other);						\
    ERROR_EXIT(128, "NOT IMPLEMENTED\n");				\
  }									\
  template <>								\
  void Matrix<TYPE>::axpy(TYPE alpha, const SparseMatrix<TYPE> *other) { \
    UNUSED_VARIABLE(alpha);						\
    UNUSED_VARIABLE(other);						\
    ERROR_EXIT(128, "NOT IMPLEMENTED\n");				\
  }

#define NOT_IMPLEMENT_AXPY_HEADER(TYPE)					\
  template <>								\
  void Matrix<TYPE>::axpy(TYPE alpha, const Matrix<TYPE> *other);	\
  template <>								\
  void Matrix<TYPE>::axpy(TYPE alpha, const SparseMatrix<TYPE> *other);

//////////////////////////////////////////////////////////////////////////////

#define NOT_IMPLEMENT_GEMM(TYPE)				\
  template <>							\
  void Matrix<TYPE>::gemm(CBLAS_TRANSPOSE trans_A,		\
			  CBLAS_TRANSPOSE trans_B,		\
			  TYPE alpha,				\
			  const Matrix<TYPE> *otherA,		\
			  const Matrix<TYPE> *otherB,		\
			  TYPE beta) {				\
    UNUSED_VARIABLE(trans_A);					\
    UNUSED_VARIABLE(trans_B);					\
    UNUSED_VARIABLE(alpha);					\
    UNUSED_VARIABLE(otherA);					\
    UNUSED_VARIABLE(otherB);					\
    UNUSED_VARIABLE(beta);					\
    ERROR_EXIT(128, "NOT IMPLEMENTED\n");			\
  }								\
  template <>							\
  void Matrix<TYPE>::gemm(CBLAS_TRANSPOSE trans_A,		\
			  CBLAS_TRANSPOSE trans_B,		\
			  TYPE alpha,				\
			  const SparseMatrix<TYPE> *otherA,	\
			  const Matrix<TYPE> *otherB,		\
			  TYPE beta) {				\
    UNUSED_VARIABLE(trans_A);					\
    UNUSED_VARIABLE(trans_B);					\
    UNUSED_VARIABLE(alpha);					\
    UNUSED_VARIABLE(otherA);					\
    UNUSED_VARIABLE(otherB);					\
    UNUSED_VARIABLE(beta);					\
    ERROR_EXIT(128, "NOT IMPLEMENTED\n");			\
  }

#define NOT_IMPLEMENT_GEMM_HEADER(TYPE)				\
  template <>							\
  void Matrix<TYPE>::gemm(CBLAS_TRANSPOSE trans_A,		\
			  CBLAS_TRANSPOSE trans_B,		\
			  TYPE alpha,				\
			  const Matrix<TYPE> *otherA,		\
			  const Matrix<TYPE> *otherB,		\
			  TYPE beta);				\
  template <>							\
  void Matrix<TYPE>::gemm(CBLAS_TRANSPOSE trans_A,		\
			  CBLAS_TRANSPOSE trans_B,		\
			  TYPE alpha,				\
			  const SparseMatrix<TYPE> *otherA,	\
			  const Matrix<TYPE> *otherB,		\
			  TYPE beta);

/////////////////////////////////////////////////////////////////////////////

#define NOT_IMPLEMENT_GEMV(TYPE)				\
  template <>							\
    void Matrix<TYPE>::gemv(CBLAS_TRANSPOSE trans_A,		\
			    TYPE alpha,				\
			    const Matrix<TYPE> *otherA,		\
			    const Matrix<TYPE> *otherX,		\
			    TYPE beta) {			\
    UNUSED_VARIABLE(trans_A);					\
    UNUSED_VARIABLE(alpha);					\
    UNUSED_VARIABLE(otherA);					\
    UNUSED_VARIABLE(otherX);					\
    UNUSED_VARIABLE(beta);					\
    ERROR_EXIT(128, "NOT IMPLEMENTED\n");			\
  }								\
  template <>							\
  void Matrix<TYPE>::gemv(CBLAS_TRANSPOSE trans_A,		\
			  TYPE alpha,				\
			  const SparseMatrix<TYPE> *otherA,	\
			  const Matrix<TYPE> *otherX,		\
			  TYPE beta) {				\
    UNUSED_VARIABLE(trans_A);					\
    UNUSED_VARIABLE(alpha);					\
    UNUSED_VARIABLE(otherA);					\
    UNUSED_VARIABLE(otherX);					\
    UNUSED_VARIABLE(beta);					\
    ERROR_EXIT(128, "NOT IMPLEMENTED\n");			\
  }

#define NOT_IMPLEMENT_GEMV_HEEADER(TYPE)			\
  template <>							\
  void Matrix<TYPE>::gemv(CBLAS_TRANSPOSE trans_A,		\
			  TYPE alpha,				\
			  const Matrix<TYPE> *otherA,		\
			  const Matrix<TYPE> *otherX,		\
			  TYPE beta);				\
  template <>							\
  void Matrix<TYPE>::gemv(CBLAS_TRANSPOSE trans_A,		\
			  TYPE alpha,				\
			  const SparseMatrix<TYPE> *otherA,	\
			  const Matrix<TYPE> *otherX,		\
			  TYPE beta);

/////////////////////////////////////////////////////////////////////////////

#define NOT_IMPLEMENT_GER(TYPE)				\
  template <>						\
  void Matrix<TYPE>::ger(TYPE alpha,			\
			   const Matrix<TYPE> *otherX,	\
			 const Matrix<TYPE> *otherY) {	\
    UNUSED_VARIABLE(alpha);				\
    UNUSED_VARIABLE(otherX);				\
    UNUSED_VARIABLE(otherY);				\
    ERROR_EXIT(128, "NOT IMPLEMENTED\n");		\
  }

#define NOT_IMPLEMENT_GER_HEADER(TYPE)			\
  template <>						\
  void Matrix<TYPE>::ger(TYPE alpha,			\
			 const Matrix<TYPE> *otherX,	\
			 const Matrix<TYPE> *otherY);


#endif // MATRIX_NOT_IMPLEMENTED_H
