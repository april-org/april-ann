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
#ifndef UTILMATRIXINT32_H
#define UTILMATRIXINT32_H

#include "constString.h"
#include "matrixInt32.h"
#include "stream.h"
#include "utilMatrixIO.h"
#include "utilMatrixFloat.h"

namespace basics {

  /* Especialization of MatrixInt32 ascii and binary extractors, sizers and
     coders */
  template<>
  bool AsciiExtractor<int32_t>::operator()(april_utils::constString &line,
                                           int32_t &destination);
  
  template<>
  bool BinaryExtractor<int32_t>::operator()(april_utils::constString &line,
                                            int32_t &destination);
  
  template<>
  int AsciiSizer<int32_t>::operator()(const Matrix<int32_t> *mat);

  template<>
  int BinarySizer<int32_t>::operator()(const Matrix<int32_t> *mat);

  template<>
  void AsciiCoder<int32_t>::operator()(const int32_t &value,
                                       april_io::StreamInterface *stream);
  
  template<>
  void BinaryCoder<int32_t>::operator()(const int32_t &value,
                                        april_io::StreamInterface *stream);

  MatrixFloat *convertFromMatrixInt32ToMatrixFloat(MatrixInt32 *mat,
                                                   bool col_major);

} // namespace basics

#endif // UTILMATRIXINT32_H
