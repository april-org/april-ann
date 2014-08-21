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
#ifndef MATRIX_DOUBLE_H
#define MATRIX_DOUBLE_H
#include "matrix.h"

namespace basics {

  typedef Matrix<double> MatrixDouble;


  namespace MatrixIO {

    /* Especialization of MatrixDouble ascii and binary extractors, sizers and
       coders */
    template<>
    bool AsciiExtractor<double>::operator()(april_utils::constString &line,
                                            double &destination);
  
    template<>
    bool BinaryExtractor<double>::operator()(april_utils::constString &line,
                                             double &destination);
  
    template<>
    int AsciiSizer<double>::operator()(const Matrix<double> *mat);

    template<>
    int BinarySizer<double>::operator()(const Matrix<double> *mat);

    template<>
    void AsciiCoder<double>::operator()(const double &value,
                                        AprilIO::StreamInterface *stream);
  
    template<>
    void BinaryCoder<double>::operator()(const double &value,
                                         AprilIO::StreamInterface *stream);
    
  } // namespace MatrixIO

  template<>
  double Matrix<double>::getTemplateOption(const april_utils::GenericOptions *options,
                                           const char *name, double default_value);


  template<>
  void Matrix<double>::zeros();

  template<>
  void Matrix<double>::ones();

}

#endif // MATRIX_DOUBLE_H
