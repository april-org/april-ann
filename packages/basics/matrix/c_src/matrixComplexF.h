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

#ifndef MATRIXCOMPLEXF_H
#define MATRIXCOMPLEXF_H

#include "complex_number.h"
#include "matrix.h"

namespace Basics {

  namespace MatrixIO {

    /* Especialization of MatrixComplexF ascii and binary extractors, sizers and
       coders */
    template<>
    bool AsciiExtractor<AprilMath::ComplexF>::operator()(AprilUtils::constString &line,
                                                          AprilMath::ComplexF &destination);
  
    template<>
    bool BinaryExtractor<AprilMath::ComplexF>::operator()(AprilUtils::constString &line,
                                                           AprilMath::ComplexF &destination);
  
    template<>
    int AsciiSizer<AprilMath::ComplexF>::operator()(const Matrix<AprilMath::ComplexF> *mat);

    template<>
    int BinarySizer<AprilMath::ComplexF>::operator()(const Matrix<AprilMath::ComplexF> *mat);

    template<>
    void AsciiCoder<AprilMath::ComplexF>::operator()(const AprilMath::ComplexF &value,
                                                      AprilIO::StreamInterface *stream);
  
    template<>
    void BinaryCoder<AprilMath::ComplexF>::operator()(const AprilMath::ComplexF &value,
                                                       AprilIO::StreamInterface *stream);
  
  } // namespace MatrixIO

  //////////////////////////////////////////////////////////////////////////////

  template<>
  const char *Matrix<AprilMath::ComplexF>::luaCtorName() const;
  
  typedef Matrix<AprilMath::ComplexF> MatrixComplexF;

} // namespace Basics

////////////////////////////////////////////////////////////////////////////

DECLARE_LUA_TABLE_BIND_SPECIALIZATION(Basics::MatrixComplexF);

#endif // MATRIXCOMPLEXF_H
