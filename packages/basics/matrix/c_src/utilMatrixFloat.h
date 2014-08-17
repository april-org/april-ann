/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera
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
#ifndef UTILMATRIXFLOAT_H
#define UTILMATRIXFLOAT_H

#include "constString.h"
#include "matrixFloat.h"
#include "sparse_matrixFloat.h"
#include "utilMatrixIO.h"

namespace basics {

  /* Especialization of MatrixFloat ascii and binary extractors, sizers and
     coders */
  template<>
  bool AsciiExtractor<float>::operator()(april_utils::constString &line,
                                         float &destination);
  
  template<>
  bool BinaryExtractor<float>::operator()(april_utils::constString &line,
                                          float &destination);
  
  template<>
  int AsciiSizer<float>::operator()(const Matrix<float> *mat);

  template<>
  int BinarySizer<float>::operator()(const Matrix<float> *mat);

  template<>
  void AsciiCoder<float>::operator()(const float &value,
                                     AprilIO::StreamInterface *stream);
  
  template<>
  void BinaryCoder<float>::operator()(const float &value,
                                      AprilIO::StreamInterface *stream);
  
  /* Especialization of SparseMatrixFloat ascii and binary sizers */
  
  template<>
  int SparseAsciiSizer<float>::operator()(const SparseMatrix<float> *mat);
  
  template<>
  int SparseBinarySizer<float>::operator()(const SparseMatrix<float> *mat);
  
  /**************************************************************************/

  MatrixFloat* readMatrixFloatHEX(int width, int height,
                                  april_utils::constString cs);

  const float CTENEGRO  = 1.0f;
  const float CTEBLANCO = 0.0f;
  MatrixFloat* readMatrixFloatPNM(april_utils::constString cs,
                                  bool forcecolor=false, 
                                  bool forcegray=false);

  int saveMatrixFloatPNM(MatrixFloat *mat,
                         char **buffer);
  int saveMatrixFloatHEX(MatrixFloat *mat,
                         char **buffer,
                         int *width,
                         int *height);

  //////////////////////////////////////////////////////////////////////////////
}

#endif // UTILMATRIXFLOAT_H
