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

#include "matrix.h"
#include "matrixComplexF.h"

// WARNING: ALL THE METHODS IMPLEMENTED HERE ARE SPECIALIZED TO COMPLEXF VERSION

namespace Basics {
  typedef AprilMath::ComplexF ComplexF;

  namespace MatrixIO {  
    /////////////////////////////////////////////////////////////////////////
  
    template<>
    bool AsciiExtractor<ComplexF>::operator()(AprilUtils::constString &line,
                                              ComplexF &destination) {
      if (!line.extract_float(&destination.real())) return false;
      if (!line.extract_float(&destination.img())) return false;
      char ch;
      if (!line.extract_char(&ch)) return false;
      if (ch != 'i') return false;
      return true;
    }
  
    template<>
    bool BinaryExtractor<ComplexF>::operator()(AprilUtils::constString &line,
                                               ComplexF &destination) {
      if (!line.extract_float_binary(&destination.real())) return false;
      if (!line.extract_float_binary(&destination.img())) return false;
      return true;
    }
  
    template<>
    int AsciiSizer<ComplexF>::operator()(const Matrix<ComplexF> *mat) {
      return mat->size()*26; // 12*2+2
    }

    template<>
    int BinarySizer<ComplexF>::operator()(const Matrix<ComplexF> *mat) {
      return AprilUtils::binarizer::buffer_size_32(mat->size()<<1); // mat->size() * 2

    }

    template<>
    void AsciiCoder<ComplexF>::operator()(const ComplexF &value,
                                          AprilIO::StreamInterface *stream) {
      stream->printf("%.5g%+.5gi", value.real(), value.img());
    }
  
    template<>
    void BinaryCoder<ComplexF>::operator()(const ComplexF &value,
                                           AprilIO::StreamInterface *stream) {
      char b[10];
      AprilUtils::binarizer::code_float(value.real(), b);
      AprilUtils::binarizer::code_float(value.img(),  b+5);
      stream->put(b, sizeof(char)*10);
    }

    /////////////////////////////////////////////////////////////////////////////

  } // namespace MatrixIO

  template<>
  ComplexF Matrix<ComplexF>::
  getTemplateOption(const AprilUtils::GenericOptions *options,
                    const char *name,
                    ComplexF default_value) {
    AprilMath::LuaComplexFNumber *c =
      options->getOptionalReferenced<AprilMath::LuaComplexFNumber>(name, 0);
    if (c == 0) return default_value;
    else return c->getValue();
  }

  /////////////////////////////////////////////////////////////////////////////
  
  template class Matrix<ComplexF>;

} // namespace Basics
