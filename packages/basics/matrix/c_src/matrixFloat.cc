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
#include "binarizer.h"
#include "check_floats.h"
#include "constString.h"
#include "matrix.h"
#include "matrixFloat.h"

// WARNING: ALL THE METHODS IMPLEMENTED HERE ARE SPECIALIZED TO FLOAT VERSION

using AprilUtils::constString;

namespace Basics {

  namespace MatrixIO {
  
    /////////////////////////////////////////////////////////////////////////
  
    template<>
    bool AsciiExtractor<float>::operator()(constString &line,
                                           float &destination) {
      bool result = line.extract_float(&destination);
      if (!result) return false;
      return true;
    }
  
    template<>
    bool BinaryExtractor<float>::operator()(constString &line,
                                            float &destination) {
      if (!line.extract_float_binary(&destination)) return false;
      return true;
    }
  
    template<>
    int AsciiSizer<float>::operator()(const Matrix<float> *mat) {
      return mat->size()*12;
    }

    template<>
    int BinarySizer<float>::operator()(const Matrix<float> *mat) {
      return AprilUtils::binarizer::buffer_size_32(mat->size());
    }

    template<>
    void AsciiCoder<float>::operator()(const float &value,
                                       AprilIO::StreamInterface *stream) {
      stream->printf("%.5g", value);
    }
  
    template<>
    void BinaryCoder<float>::operator()(const float &value,
                                        AprilIO::StreamInterface *stream) {
      char b[5];
      AprilUtils::binarizer::code_float(value, b);
      stream->put(b, sizeof(char)*5);
    }

    /////////////////////////////////////////////////////////////////////////////
  

  } // namespace MatrixIO

  template<>
  float Matrix<float>::getTemplateOption(const AprilUtils::GenericOptions *options,
                                         const char *name, float default_value) {
    return options->getOptionalFloat(name, default_value);
  }
  
  template <>
  void Matrix<float>::pruneSubnormalAndCheckNormal() {
    float *data = getRawDataAccess()->getPPALForReadAndWrite();
    if (!AprilUtils::check_floats(data, size()))
      ERROR_EXIT(128, "No finite numbers at weights matrix!!!\n");
  }
  
  ///////////////////////////////////////////////////////////////////////////////

  template class Matrix<float>;

} // namespace Basics
