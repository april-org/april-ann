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

namespace Basics {


  namespace MatrixIO {

    /* Especialization of MatrixChar ascii and binary extractors, sizers and
       coders */
    template<>
    bool AsciiExtractor<char>::operator()(AprilUtils::constString &line,
                                          char &destination);
  
    template<>
    bool BinaryExtractor<char>::operator()(AprilUtils::constString &line,
                                           char &destination);
  
    template<>
    int AsciiSizer<char>::operator()(const Matrix<char> *mat);

    template<>
    int BinarySizer<char>::operator()(const Matrix<char> *mat);

    template<>
    void AsciiCoder<char>::operator()(const char &value,
                                      AprilIO::StreamInterface *stream);
  
    template<>
    void BinaryCoder<char>::operator()(const char &value,
                                       AprilIO::StreamInterface *stream);

  } // namespace MatrixIO

  ///////////////////////////////////////////////////////////////////////////////
  typedef Matrix<char> MatrixChar;

} // namespace Basics

////////////////////////////////////////////////////////////////////////////

namespace AprilUtils {

  template<> Basics::MatrixChar *LuaTable::
  convertTo<Basics::MatrixChar *>(lua_State *L, int idx);
  
  template<> void LuaTable::
  pushInto<Basics::MatrixChar *>(lua_State *L, Basics::MatrixChar *value);

  template<> bool LuaTable::
  checkType<Basics::MatrixChar *>(lua_State *L, int idx);
}

#endif // MATRIX_CHAR_H
