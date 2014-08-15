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
#ifndef MATRIX_CHAR_H
#define MATRIX_CHAR_H
#include "matrix.h"
#include "matrix_not_implemented.h"

namespace basics {

  NOT_IMPLEMENT_AXPY_HEADER(char)
  NOT_IMPLEMENT_GEMM_HEADER(char)
  NOT_IMPLEMENT_GEMV_HEEADER(char)
  NOT_IMPLEMENT_GER_HEADER(char)
  NOT_IMPLEMENT_DOT_HEADER(char)

  ///////////////////////////////////////////////////////////////////////////////
  typedef Matrix<char> MatrixChar;

} // namespace basics

#endif // MATRIX_CHAR_H
