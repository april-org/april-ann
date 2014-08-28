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
#include "matrixDouble.h"

namespace basics {

  namespace MatrixIO {

    /////////////////////////////////////////////////////////////////////////
  
    template<>
    bool AsciiExtractor<double>::operator()(april_utils::constString &line,
                                            double &destination) {
      if (!line.extract_double(&destination)) return false;
      return true;
    }
  
    template<>
    bool BinaryExtractor<double>::operator()(april_utils::constString &line,
                                             double &destination) {
      if (!line.extract_double_binary(&destination)) return false;
      return true;
    }
  
    template<>
    int AsciiSizer<double>::operator()(const Matrix<double> *mat) {
      return mat->size()*12;
    }

    template<>
    int BinarySizer<double>::operator()(const Matrix<double> *mat) {
      return april_utils::binarizer::buffer_size_64(mat->size());
    }

    template<>
    void AsciiCoder<double>::operator()(const double &value,
                                        AprilIO::StreamInterface *stream) {
      stream->printf("%.5g", value);
    }
  
    template<>
    void BinaryCoder<double>::operator()(const double &value,
                                         AprilIO::StreamInterface *stream) {
      char b[10];
      april_utils::binarizer::code_double(value, b);
      stream->put(b, sizeof(char)*10);
    }

    /////////////////////////////////////////////////////////////////////////////

  } // namespace MatrixIO
  
  template<>
  double Matrix<double>::
  getTemplateOption(const april_utils::GenericOptions *options,
                    const char *name, double default_value) {
    return options->getOptionalDouble(name, default_value);
  }

  /////////////////////////////////////////////////////////////////////////
  
  template class Matrix<double>;

}
