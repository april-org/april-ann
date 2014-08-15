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
#ifndef UTILMATRIXCOMPLEXF_H
#define UTILMATRIXCOMPLEXF_H

#include "constString.h"
#include "complex_number.h"
#include "matrixComplexF.h"
#include "stream.h"
#include "utilMatrixIO.h"
#include "utilMatrixFloat.h"

namespace basics {

  /* Especialization of MatrixComplexF ascii and binary extractors, sizers and
     coders */
  template<>
  bool AsciiExtractor<april_math::ComplexF>::operator()(april_utils::constString &line,
                                                        april_math::ComplexF &destination);
  
  template<>
  bool BinaryExtractor<april_math::ComplexF>::operator()(april_utils::constString &line,
                                                         april_math::ComplexF &destination);
  
  template<>
  int AsciiSizer<april_math::ComplexF>::operator()(const Matrix<april_math::ComplexF> *mat);

  template<>
  int BinarySizer<april_math::ComplexF>::operator()(const Matrix<april_math::ComplexF> *mat);

  template<>
  void AsciiCoder<april_math::ComplexF>::operator()(const april_math::ComplexF &value,
                                                    april_io::StreamInterface *stream);
  
  template<>
  void BinaryCoder<april_math::ComplexF>::operator()(const april_math::ComplexF &value,
                                                     april_io::StreamInterface *stream);
  
  MatrixFloat *convertFromMatrixComplexFToMatrixFloat(MatrixComplexF *mat);
  void applyConjugateInPlace(MatrixComplexF *mat);
  MatrixFloat *realPartFromMatrixComplexFToMatrixFloat(MatrixComplexF *mat);
  MatrixFloat *imgPartFromMatrixComplexFToMatrixFloat(MatrixComplexF *mat);
  MatrixFloat *absFromMatrixComplexFToMatrixFloat(MatrixComplexF *mat);
  MatrixFloat *angleFromMatrixComplexFToMatrixFloat(MatrixComplexF *mat);

} // namespace basics

#endif // UTILMATRIXCOMPLEXF_H
