/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Salvador España-Boquera, Francisco Zamora-Martinez
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
#ifndef MATRIX_H
#include "matrix.h"
#include "sparse_matrix.h"
#endif
#ifndef MATRIX_EXT_H
#define MATRIX_EXT_H
#include "matrix_ext_blas.h"
#include "matrix_ext_boolean.h"
#include "matrix_ext_initializers.h"
#include "matrix_ext_lapack.h"
#include "matrix_ext_misc.h"
#include "matrix_ext_operations.h"
#include "matrix_ext_reductions.h"
namespace AprilMath {

  /**
   * @brief Linear algebra routines and other math operations for matrices.
   *
   * By default, the zero value must be T(). Additionally, T(0.0f) and T(1.0f)
   * and T(-1.0f) and T(-nan) constructors must be available with correct math
   * values. In case of char buffer or integer matrices these constructors are
   * needed but not operational because math methods are forbidden for these
   * data types.
   */
  namespace MatrixExt {
    
    // Just for documentation
    
  }
}
#endif // MATRIX_EXT_H
