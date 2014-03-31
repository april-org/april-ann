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

#ifndef MATRIXCOMPLEXF_H
#define MATRIXCOMPLEXF_H

#include "complex_number.h"
#include "matrix.h"

template<>
void Matrix<ComplexF>::fill(ComplexF value);

template<>
void Matrix<ComplexF>::zeros();

template<>
void Matrix<ComplexF>::ones();

template<>
Matrix<ComplexF> *Matrix<ComplexF>::addition(const Matrix<ComplexF> *other);

template<>
Matrix<ComplexF> *Matrix<ComplexF>::substraction(const Matrix<ComplexF> *other);

template <>
Matrix<ComplexF>* Matrix<ComplexF>::multiply(const Matrix<ComplexF> *other) const;

template<>
ComplexF Matrix<ComplexF>::sum() const;

template<>
void Matrix<ComplexF>::scalarAdd(ComplexF s);

template<>
bool Matrix<ComplexF>::equals(const Matrix<ComplexF> *other,
			      float epsilon) const;

template<>
void Matrix<ComplexF>::cmul(const Matrix<ComplexF> *other);

/**** BLAS OPERATIONS ****/

template<>
void Matrix<ComplexF>::copy(const Matrix<ComplexF> *other);

template<>
void Matrix<ComplexF>::ger(ComplexF alpha,
			   const Matrix<ComplexF> *otherX,
			   const Matrix<ComplexF> *otherY);

template<>
ComplexF Matrix<ComplexF>::dot(const Matrix<ComplexF> *other) const;

template<>
void Matrix<ComplexF>::scal(ComplexF value);

template<>
float Matrix<ComplexF>::norm2() const;

//////////////////////////////////////////////////////////////////////////////

typedef Matrix<ComplexF> MatrixComplexF;

#endif // MATRIXCOMPLEXF_H
