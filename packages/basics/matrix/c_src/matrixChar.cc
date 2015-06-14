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
#include "matrixChar.h"

namespace Basics {

  namespace MatrixIO {
    /////////////////////////////////////////////////////////////////////////
  
    template<>
    bool AsciiExtractor<char>::operator()(AprilUtils::constString &line,
                                          char &destination) {
      if (!line.extract_char(&destination)) return false;
      return true;
    }
  
    template<>
    bool BinaryExtractor<char>::operator()(AprilUtils::constString &line,
                                           char &destination) {
      UNUSED_VARIABLE(line);
      UNUSED_VARIABLE(destination);
      ERROR_EXIT(128, "Char type has not binary option\n");
      return false;

    }
  
    template<>
    int AsciiSizer<char>::operator()(const Matrix<char> *mat) {
      return mat->size()*2;
    }

    template<>
    int BinarySizer<char>::operator()(const Matrix<char> *mat) {
      UNUSED_VARIABLE(mat);
      ERROR_EXIT(128, "Char type has not binary option\n");
      return 0;
    }

    template<>
    void AsciiCoder<char>::operator()(const char &value,
                                      AprilIO::StreamInterface *stream) {
      stream->printf("%c", value);
    }
  
    template<>
    void BinaryCoder<char>::operator()(const char &value,
                                       AprilIO::StreamInterface *stream) {
      UNUSED_VARIABLE(value);
      UNUSED_VARIABLE(stream);
      ERROR_EXIT(128, "Char type has not binary option\n");

    }

    /////////////////////////////////////////////////////////////////////////////

  } // namespace MatrixIO

  template<>
  int Matrix<char>::exportParamsToLua(lua_State *L) {
    UNUSED_VARIABLE(L);
    ERROR_EXIT(128, "Serialization of matrixChar not implemented\n");
    return 0;
  }
  
  ///////////////////////////////////////////////////////////////////////////////
  template class Matrix<char>;

} // namespace Basics
