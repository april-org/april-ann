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

namespace Basics {

  namespace MatrixIO {

    /////////////////////////////////////////////////////////////////////////
  
    template<>
    bool AsciiExtractor<int32_t>::operator()(AprilUtils::constString &line,
                                             int32_t &destination) {
      if (!line.extract_int(&destination)) return false;
      return true;
    }
  
    template<>
    bool BinaryExtractor<int32_t>::operator()(AprilUtils::constString &line,
                                              int32_t &destination) {
      if (!line.extract_int32_binary(&destination)) return false;
      return true;
    }
  
    template<>
    int AsciiSizer<int32_t>::operator()(const Matrix<int32_t> *mat) {
      return mat->size()*12;
    }

    template<>
    int BinarySizer<int32_t>::operator()(const Matrix<int32_t> *mat) {
      return AprilUtils::binarizer::buffer_size_32(mat->size());
    }

    template<>
    void AsciiCoder<int32_t>::operator()(const int32_t &value,
                                         AprilIO::StreamInterface *stream) {
      stream->printf("%d", value);
    }
  
    template<>
    void BinaryCoder<int32_t>::operator()(const int32_t &value,
                                          AprilIO::StreamInterface *stream) {
      char b[5];
      AprilUtils::binarizer::code_int32(value, b);
      stream->put(b, sizeof(char)*5);
    }

    /////////////////////////////////////////////////////////////////////////
    
  } // namespace MatrixIO

  template<>
  const char *Matrix<int32_t>::luaCtorName() const {
    return "matrixInt32.deserialize";
  }
  
  /////////////////////////////////////////////////////////////////////////////
  
  template class Matrix<int32_t>;

} // namespace Basics
