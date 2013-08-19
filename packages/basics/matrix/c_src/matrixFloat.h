/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera
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

#ifndef MATRIXFLOAT_H
#define MATRIXFLOAT_H

#include "matrix.h"

template<>
void Matrix<float>::fill(float value);

template<>
void Matrix<float>::clamp(float lower, float upper);

template<>
void Matrix<float>::zeros();

template<>
void Matrix<float>::ones();

template<>
Matrix<float> *Matrix<float>::addition(const Matrix<float> *other);

template<>
Matrix<float> *Matrix<float>::substraction(const Matrix<float> *other);

template <>
Matrix<float>* Matrix<float>::multiply(const Matrix<float> *other) const;

template<>
float Matrix<float>::sum() const;

template<>
void Matrix<float>::scalarAdd(float s);

template<>
bool Matrix<float>::equals(const Matrix<float> *other, float epsilon) const;

template<>
void Matrix<float>::plogp();

template<>
void Matrix<float>::log();

template<>
void Matrix<float>::log1p();

template<>
void Matrix<float>::exp();

template<>
void Matrix<float>::sqrt();

template<>
void Matrix<float>::pow(float value);

template<>
void Matrix<float>::tanh();

template<>
Matrix<float> *Matrix<float>::cmul(const Matrix<float> *other);

/**** BLAS OPERATIONS ****/

template<>
void Matrix<float>::copy(const Matrix<float> *other);

template<>
void Matrix<float>::axpy(float alpha, const Matrix<float> *other);

template<>
void Matrix<float>::gemm(CBLAS_TRANSPOSE trans_A,
			 CBLAS_TRANSPOSE trans_B,
			 float alpha,
			 const Matrix<float> *otherA,
			 const Matrix<float> *otherB,
			 float beta);

template<>
void Matrix<float>::gemv(CBLAS_TRANSPOSE trans_A,
			 float alpha,
			 const Matrix<float> *otherA,
			 const Matrix<float> *otherX,
			 float beta);

template<>
void Matrix<float>::ger(float alpha,
			const Matrix<float> *otherX,
			const Matrix<float> *otherY);

template<>
float Matrix<float>::dot(const Matrix<float> *other) const;

template<>
void Matrix<float>::scal(float value);

template<>
float Matrix<float>::norm2() const;

template<>
float Matrix<float>::min(int &arg_min, int &arg_min_raw_pos) const;

template<>
float Matrix<float>::max(int &arg_max, int &arg_max_raw_pos) const;

template<>
void Matrix<float>::minAndMax(float &min, float &max) const;

template <>
Matrix<float> *Matrix<float>::maxSelDim(const int dim,
					IntGPUMirroredMemoryBlock *raw_positions,
					int shift) const;

template<>
void Matrix<float>::adjustRange(float rmin, float rmax);

////////////////////////////////////////////////////////////////////////////

typedef Matrix<float> MatrixFloat;

#endif // MATRIXFLOAT_H
