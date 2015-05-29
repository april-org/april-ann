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
#ifndef MATRIX_INT32_H
#define MATRIX_INT32_H
#include "matrix.h"

namespace Basics {


  namespace MatrixIO {

    /* Especialization of MatrixInt32 ascii and binary extractors, sizers and
       coders */
    template<>
    bool AsciiExtractor<int32_t>::operator()(AprilUtils::constString &line,
                                             int32_t &destination);
  
    template<>
    bool BinaryExtractor<int32_t>::operator()(AprilUtils::constString &line,
                                              int32_t &destination);
  
    template<>
    int AsciiSizer<int32_t>::operator()(const Matrix<int32_t> *mat);

    template<>
    int BinarySizer<int32_t>::operator()(const Matrix<int32_t> *mat);

    template<>
    void AsciiCoder<int32_t>::operator()(const int32_t &value,
                                         AprilIO::StreamInterface *stream);
  
    template<>
    void BinaryCoder<int32_t>::operator()(const int32_t &value,
                                          AprilIO::StreamInterface *stream);

  } // namespace MatrixIO
  
  //////////////////////////////////////////////////////////////////////////////

  template<>
  const char *Matrix<int32_t>::luaCtorName() const;
  
  typedef Matrix<int32_t> MatrixInt32;

} // namespace Basics

////////////////////////////////////////////////////////////////////////////

namespace AprilUtils {

  template<> Basics::MatrixInt32 *LuaTable::
  convertTo<Basics::MatrixInt32 *>(lua_State *L, int idx);
  
  template<> void LuaTable::
  pushInto<Basics::MatrixInt32 *>(lua_State *L, Basics::MatrixInt32 *value);

  template<> bool LuaTable::
  checkType<Basics::MatrixInt32 *>(lua_State *L, int idx);
}

#endif // MATRIX_INT_H
