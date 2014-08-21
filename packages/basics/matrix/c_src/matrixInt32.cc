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

namespace basics {

  namespace MatrixIO {

    /////////////////////////////////////////////////////////////////////////
  
    template<>
    bool AsciiExtractor<int32_t>::operator()(april_utils::constString &line,
                                             int32_t &destination) {
      if (!line.extract_int(&destination)) return false;
      return true;
    }
  
    template<>
    bool BinaryExtractor<int32_t>::operator()(april_utils::constString &line,
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
      return april_utils::binarizer::buffer_size_32(mat->size());
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
      april_utils::binarizer::code_int32(value, b);
      stream->put(b, sizeof(char)*5);
    }

    /////////////////////////////////////////////////////////////////////////

  } // namespace MatrixIO
  
  template<>
  int32_t Matrix<int32_t>::
  getTemplateOption(const april_utils::GenericOptions *options,
                    const char *name, int32_t default_value) {
    return options->getOptionalInt32(name, default_value);
  }

  /////////////////////////////////////////////////////////////////////////////
  
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

} // namespace basics
