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
#include <stdint.h>
#include "matrixInt32.h"
#include "matrix_not_implemented.h"

NOT_IMPLEMENT_AXPY(int32_t)
NOT_IMPLEMENT_GEMM(int32_t)
NOT_IMPLEMENT_GEMV(int32_t)
NOT_IMPLEMENT_GER(int32_t)
NOT_IMPLEMENT_DOT(int32_t)

/************* ZEROS FUNCTION **************/
template<>
void Matrix<int32_t>::zeros() {
  fill(0);
}

/************* ONES FUNCTION **************/
template<>
void Matrix<int32_t>::ones() {
  fill(1);
}

///////////////////////////////////////////////////////////////////////////////

template class Matrix<int32_t>;
