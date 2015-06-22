/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2014, Francisco Zamora-Martinez
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

#ifndef SPARSEMATRIXDOUBLE_H
#define SPARSEMATRIXDOUBLE_H

#include "matrixDouble.h"

///////////////////////////////////////////////////////////////////////////////

namespace Basics {
  
  namespace MatrixIO {
    
    /* Especialization of SparseMatrixDouble ascii and binary sizers */
    
    template<>
    int SparseAsciiSizer<double>::operator()(const SparseMatrix<double> *mat);
    
    template<>
    int SparseBinarySizer<double>::operator()(const SparseMatrix<double> *mat);
    
  }
  
  /////////////////////////////////////////////////////////////////////////
  
  typedef SparseMatrix<double> SparseMatrixDouble;

  typedef SparseMatrix<double>::DOKBuilder DOKBuilderSparseMatrixDouble;

}

////////////////////////////////////////////////////////////////////////////

DECLARE_LUA_TABLE_BIND_SPECIALIZATION(Basics::SparseMatrixDouble);

#endif // SPARSEMATRIXDOUBLE_H
