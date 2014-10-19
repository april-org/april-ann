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
#ifndef MATRIX_BOOL_H
#define MATRIX_BOOL_H
#include "matrix.h"

namespace Basics {


  namespace MatrixIO {

    /* Especialization of MatrixBool ascii and binary extractors, sizers and
       coders */
    template<>
    bool AsciiExtractor<bool>::operator()(AprilUtils::constString &line,
                                          bool &destination);
  
    template<>
    bool BinaryExtractor<bool>::operator()(AprilUtils::constString &line,
                                           bool &destination);
  
    template<>
    int AsciiSizer<bool>::operator()(const Matrix<bool> *mat);

    template<>
    int BinarySizer<bool>::operator()(const Matrix<bool> *mat);

    template<>
    void AsciiCoder<bool>::operator()(const bool &value,
                                      AprilIO::StreamInterface *stream);
  
    template<>
    void BinaryCoder<bool>::operator()(const bool &value,
                                       AprilIO::StreamInterface *stream);

  } // namespace MatrixIO

  template<>
  bool Matrix<bool>::getTemplateOption(const AprilUtils::GenericOptions *options,
                                       const char *name, bool default_value);
  
  ///////////////////////////////////////////////////////////////////////////////
  typedef Matrix<bool> MatrixBool;

} // namespace Basics

#endif // MATRIX_BOOL_H
