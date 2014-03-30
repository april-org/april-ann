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
void Matrix<float>::tan();

template<>
void Matrix<float>::tanh();

template<>
void Matrix<float>::atan();

template<>
void Matrix<float>::atanh();

template <>
void Matrix<float>::cos();

template <>
void Matrix<float>::cosh();

template <>
void Matrix<float>::acos();

template <>
void Matrix<float>::acosh();

template <>
void Matrix<float>::sin();

template <>
void Matrix<float>::sinh();

template <>
void Matrix<float>::asin();

template <>
void Matrix<float>::asinh();

template <>
void Matrix<float>::abs();

template <>
void Matrix<float>::complement();

template <>
void Matrix<float>::sign();

template<>
void Matrix<float>::cmul(const Matrix<float> *other);

/**** BLAS OPERATIONS ****/

template<>
void Matrix<float>::copy(const Matrix<float> *other);

template<>
void Matrix<float>::ger(float alpha,
			const Matrix<float> *otherX,
			const Matrix<float> *otherY);

template<>
float Matrix<float>::dot(const Matrix<float> *other) const;

template<>
float Matrix<float>::dot(const SparseMatrix<float> *other) const;

template<>
void Matrix<float>::scal(float value);

template<>
void Matrix<float>::div(float value);

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
					Int32GPUMirroredMemoryBlock *raw_positions,
					int shift) const;

template<>
void Matrix<float>::adjustRange(float rmin, float rmax);

template<>
Matrix<float> *Matrix<float>::inv();

template <>
void Matrix<float>::svd(Matrix<float> **U, SparseMatrix<float> **S,
			Matrix<float> **V);

template <>
void Matrix<float>::pruneSubnormalAndCheckNormal();

/* BOOLEAN CONDITIONS: this methods transforms the given matrix in a ZERO/ONE
   matrix, depending in the truth of the given condition */
// less than
template <>
void Matrix<float>::LTCondition(float value);
template <>
void Matrix<float>::LTCondition(Matrix<float> *value);
// greater than
template <>
void Matrix<float>::GTCondition(float value);
template <>
void Matrix<float>::GTCondition(Matrix<float> *value);
//

////////////////////////////////////////////////////////////////////////////

typedef Matrix<float> MatrixFloat;

#endif // MATRIXFLOAT_H
