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
#include "utilMatrixInt32.h"

namespace Basics {

  MatrixFloat *convertFromMatrixInt32ToMatrixFloat(MatrixInt32 *mat) {
    MatrixFloat *new_mat=new MatrixFloat(mat->getNumDim(),
                                         mat->getDimPtr());
#ifdef USE_CUDA
    new_mat->setUseCuda(mat->getCudaFlag());
#endif
    MatrixInt32::const_iterator orig_it(mat->begin());
    MatrixFloat::iterator dest_it(new_mat->begin());
    while(orig_it != mat->end()) {
      if (abs(*orig_it) >= 16777216)
        ERROR_PRINT("The integer part can't be represented "
                    "using float precision\n");
      *dest_it = static_cast<float>(*orig_it);
      ++orig_it;
      ++dest_it;
    }
    return new_mat;
  }

} // namespace Basics
