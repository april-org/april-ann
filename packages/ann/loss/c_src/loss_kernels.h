/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2014, Salvador España-Boquera, Francisco Zamora-Martinez, Adrián
 * Palacios-Corella
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
#ifndef MATRIX_LOSS_OPERATIONS_H
#define MATRIX_LOSS_OPERATIONS_H

#include "matrixFloat.h"

namespace AprilMath {
  namespace MatrixExt {
    namespace LossOperations {

      void matMSE(Basics::MatrixFloat *output,
                  const Basics::MatrixFloat *input,
                  const Basics::MatrixFloat *target);

      void matCrossEntropy(Basics::MatrixFloat *output,
                           const Basics::MatrixFloat *input,
                           const Basics::MatrixFloat *target,
                           float near_zero);

      void matCrossEntropyGradient(Basics::MatrixFloat *output,
                                   const Basics::MatrixFloat *input,
                                   const Basics::MatrixFloat *target,
                                   float near_zero);
      
      void matMAEGradient(Basics::MatrixFloat *output,
                          const Basics::MatrixFloat *input,
                          const Basics::MatrixFloat *target,
                          float near_zero,
                          float inv_n);
      
      void matMultiClassCrossEntropy(Basics::MatrixFloat *output,
                                     const Basics::MatrixFloat *input,
                                     const Basics::MatrixFloat *target,
                                     float near_zero);
      
    }
  }
}

#endif // MATRIX_LOSS_OPERATIONS_H
