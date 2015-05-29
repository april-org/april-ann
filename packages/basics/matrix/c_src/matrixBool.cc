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
#include "matrixBool.h"

namespace Basics {

  namespace MatrixIO {
    /////////////////////////////////////////////////////////////////////////
  
    template<>
    bool AsciiExtractor<bool>::operator()(AprilUtils::constString &line,
                                          bool &destination) {
      char aux;
      if (!line.extract_char(&aux)) return false;
      destination = (aux == '1');
      return true;
    }
  
    template<>
    bool BinaryExtractor<bool>::operator()(AprilUtils::constString &line,
                                           bool &destination) {
      UNUSED_VARIABLE(line);
      UNUSED_VARIABLE(destination);
      ERROR_EXIT(128, "Bool type has not binary option\n");
      return false;

    }
  
    template<>
    int AsciiSizer<bool>::operator()(const Matrix<bool> *mat) {
      return mat->size()*2;
    }

    template<>
    int BinarySizer<bool>::operator()(const Matrix<bool> *mat) {
      UNUSED_VARIABLE(mat);
      ERROR_EXIT(128, "Bool type has not binary option\n");
      return 0;
    }

    template<>
    void AsciiCoder<bool>::operator()(const bool &value,
                                      AprilIO::StreamInterface *stream) {
      stream->printf("%c", (value) ? ('1') : ('0'));
    }
  
    template<>
    void BinaryCoder<bool>::operator()(const bool &value,
                                       AprilIO::StreamInterface *stream) {
      UNUSED_VARIABLE(value);
      UNUSED_VARIABLE(stream);
      ERROR_EXIT(128, "Bool type has not binary option\n");

    }
    
    /////////////////////////////////////////////////////////////////////////////
    
  } // namespace MatrixIO

  template<>
  const char *Matrix<float>::ctorName() const {
    return "matrixBool.deserialize";
  }


  ///////////////////////////////////////////////////////////////////////////////
  template class Matrix<bool>;

} // namespace Basics
