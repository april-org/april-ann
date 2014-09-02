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

#include "cmath_overloads.h"
#include "reduce_matrix.impl.cu"

// Must be defined here.
#include "matrix_operations.h"

#define INSTANTIATE_MATRIX_SCALAR_REDUCE1(TYPE,OP1,OP2)                 \
  template TYPE                                                         \
  MatrixScalarReduce1< TYPE, OP1, OP2 >(const Basics::Matrix< TYPE > *, \
                                        const OP1 &,                    \
                                        const OP2 &,                    \
                                        const TYPE &,                   \
                                        Basics::Matrix< TYPE > *,       \
                                        unsigned int)

#define INSTANTIATE_MATRIX_SPAN_REDUCE1(TYPE,OUTPUT,OP1,OP2)            \
  template OUTPUT                                                       \
  MatrixSpanReduce1< TYPE, OUTPUT, OP1, OP2 >(const Basics::Matrix< TYPE > *, \
                                              const OP1 &,              \
                                              const OP2 &,              \
                                              const OUTPUT &,           \
                                              Basics::Matrix< OUTPUT > *, \
                                              unsigned int)

#define INSTANTIATE_MATRIX_SPAN_REDUCE2(TYPE,OP1,OP2)                   \
  template TYPE                                                         \
  MatrixSpanReduce2< TYPE, OP1, OP2 >(const Basics::Matrix< TYPE > *,   \
                                      const Basics::Matrix< TYPE > *,   \
                                      const OP1 &,                      \
                                      const OP2 &,                      \
                                      const TYPE &,                     \
                                      Basics::Matrix< TYPE > *,         \
                                      unsigned int)

namespace AprilMath {  
  namespace MatrixExt {
    
    INSTANTIATE_MATRIX_SPAN_REDUCE1(float, float, float_float_span_reduce1_t,
                                    AprilMath::MatrixExt::Functors::MatrixNorm2Reductor<float>);
    
  } // namespace MatrixExt
} // namespace AprilMath
