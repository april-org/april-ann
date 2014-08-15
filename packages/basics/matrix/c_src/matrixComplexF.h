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

namespace basics {

  template<>
  void Matrix<april_math::ComplexF>::fill(april_math::ComplexF value);

  template<>
  april_math::ComplexF Matrix<april_math::ComplexF>::sum() const;

  template<>
  void Matrix<april_math::ComplexF>::scalarAdd(april_math::ComplexF s);

  template<>
  bool Matrix<april_math::ComplexF>::equals(const Matrix<april_math::ComplexF> *other,
                                            float epsilon) const;

  template<>
  void Matrix<april_math::ComplexF>::cmul(const Matrix<april_math::ComplexF> *other);

  /**** BLAS OPERATIONS ****/

  template<>
  void Matrix<april_math::ComplexF>::copy(const Matrix<april_math::ComplexF> *other);

  template<>
  void Matrix<april_math::ComplexF>::scal(april_math::ComplexF value);

  template<>
  float Matrix<april_math::ComplexF>::norm2() const;

  //////////////////////////////////////////////////////////////////////////////

  typedef Matrix<april_math::ComplexF> MatrixComplexF;

} // namespace basics

#endif // MATRIXCOMPLEXF_H
