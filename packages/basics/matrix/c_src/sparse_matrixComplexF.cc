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

#include <cmath>
#include "sparse_matrixComplexF.h"

using AprilMath::ComplexF;

///////////////////////////////////////////////////////////////////////////////

namespace Basics {
  
  namespace MatrixIO {
    
    template<>
    int SparseAsciiSizer<ComplexF>::operator()(const SparseMatrix<ComplexF> *mat) {
      return mat->nonZeroSize()*26;
    }
    
    template<>
    int SparseBinarySizer<ComplexF>::operator()(const SparseMatrix<ComplexF> *mat) {
      return AprilUtils::binarizer::buffer_size_32(mat->nonZeroSize()<<1);
    }
    
  }
  
  //////////////////////////////////////////////////////////////////////////
  
  template class SparseMatrix<ComplexF>;
}
