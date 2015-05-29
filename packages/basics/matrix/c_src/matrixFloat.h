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

#ifndef MATRIXFLOAT_H
#define MATRIXFLOAT_H

#include "lua_table.h"
#include "matrix.h"

namespace Basics {

  namespace MatrixIO {
    
    /* Especialization of MatrixFloat ascii and binary extractors, sizers and
       coders */
    template<>
    bool AsciiExtractor<float>::operator()(AprilUtils::constString &line,
                                           float &destination);
  
    template<>
    bool BinaryExtractor<float>::operator()(AprilUtils::constString &line,
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
  
    /**************************************************************************/
 
  } // namespace MatrixIO
    
  //////////////////////////////////////////////////////////////////////////

  template<>
  const char *Matrix<float>::luaCtorName() const;
  
  template <>
  void Matrix<float>::pruneSubnormalAndCheckNormal();

  ////////////////////////////////////////////////////////////////////////////

  typedef Matrix<float> MatrixFloat;

}

////////////////////////////////////////////////////////////////////////////

namespace AprilUtils {

  template<> Basics::MatrixFloat *LuaTable::
  convertTo<Basics::MatrixFloat *>(lua_State *L, int idx);
  
  template<> void LuaTable::
  pushInto<Basics::MatrixFloat *>(lua_State *L, Basics::MatrixFloat *value);

  template<> bool LuaTable::
  checkType<Basics::MatrixFloat *>(lua_State *L, int idx);
}

#endif // MATRIXFLOAT_H
