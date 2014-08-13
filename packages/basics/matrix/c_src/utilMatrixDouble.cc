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
#include <cmath>
#include <cstdio>

#include "binarizer.h"
#include "clamp.h"
#include "matrixDouble.h"
#include "ignore_result.h"
#include "stream.h"
#include "utilMatrixDouble.h"

namespace basics {

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
                                      april_io::StreamInterface *stream) {
    stream->printf("%.5g", value);
  }
  
  template<>
  void BinaryCoder<double>::operator()(const double &value,
                                       april_io::StreamInterface *stream) {
    char b[10];
    april_utils::binarizer::code_double(value, b);
    stream->put(b, sizeof(char)*10);
  }

  /////////////////////////////////////////////////////////////////////////////

  MatrixFloat *convertFromMatrixDoubleToMatrixFloat(MatrixDouble *mat,
                                                    bool col_major) {
    MatrixFloat *new_mat=new MatrixFloat(mat->getNumDim(),
                                         mat->getDimPtr(),
                                         (col_major)?CblasColMajor:CblasRowMajor);
#ifdef USE_CUDA
    new_mat->setUseCuda(mat->getCudaFlag());
#endif
    MatrixDouble::const_iterator orig_it(mat->begin());
    MatrixFloat::iterator dest_it(new_mat->begin());
    while(orig_it != mat->end()) {
      if (fabs(*orig_it) >= 16777216.0)
        ERROR_PRINT("The integer part can't be represented "
                    "using float precision\n");
      *dest_it = static_cast<float>(*orig_it);
      ++orig_it;
      ++dest_it;
    }
    return new_mat;
  }

} // namespace basics
