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
#ifndef ACTF_KERNELS_H
#define ACTF_KERNELS_H
#include "activation_function_kernels.h"
#include "matrixFloat.h"

namespace ANN {
  namespace Kernels {
    
    void applyHardTanhDerivative(Basics::MatrixFloat *output_errors,
                                 const Basics::MatrixFloat *input_units,
                                 float inf, float sup);

    void applyTanh(Basics::MatrixFloat *output,
                   const Basics::MatrixFloat *input);

    void applyTanhDerivative(Basics::MatrixFloat *output_errors,
                             const Basics::MatrixFloat *output_units);

    void applyLogistic(Basics::MatrixFloat *output,
                       const Basics::MatrixFloat *input);
    
    void applyLogisticDerivative(Basics::MatrixFloat *output_errors,
                                 const Basics::MatrixFloat *output_units);

    void applySoftsign(Basics::MatrixFloat *output,
                       const Basics::MatrixFloat *input);
    
    void applySoftsignDerivative(Basics::MatrixFloat *output_errors,
                                 const Basics::MatrixFloat *output_units);

    void applySoftplus(Basics::MatrixFloat *output,
                       const Basics::MatrixFloat *input);
    
    void applySoftplusDerivative(Basics::MatrixFloat *output_errors,
                                 const Basics::MatrixFloat *input_units);

    void applyReLU(Basics::MatrixFloat *output,
                   const Basics::MatrixFloat *input);
    
    void applyReLUDerivative(Basics::MatrixFloat *output_errors,
                             const Basics::MatrixFloat *input_units);

    void applyLeakyReLU(Basics::MatrixFloat *output,
                        const Basics::MatrixFloat *input,
                        float leak);
    
    void applyLeakyReLUDerivative(Basics::MatrixFloat *output_errors,
                                  const Basics::MatrixFloat *input_units,
                                  float leak);

    void applyPReLU(Basics::MatrixFloat *output,
                    Basics::MatrixFloat *input,
                    const Basics::MatrixFloat *w);
    
    void applyPReLUDerivative(Basics::MatrixFloat *output_errors,
                              Basics::MatrixFloat *input_units,
                              const Basics::MatrixFloat *w);

    void applyLogLogistic(Basics::MatrixFloat *output,
                          const Basics::MatrixFloat *input);
    
    void applySoftmax(Basics::MatrixFloat *output,
                      const Basics::MatrixFloat *input);

    void applyLogSoftmax(Basics::MatrixFloat *output,
                         const Basics::MatrixFloat *input);
    
    void applySoftmaxDerivative(Basics::MatrixFloat *output_errors,
                                const Basics::MatrixFloat *input_errors,
                                const Basics::MatrixFloat *output_units);
  }
}
#endif // ACTF_KERNELS_H
