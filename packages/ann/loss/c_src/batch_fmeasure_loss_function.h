/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera, Adrian Palacios, Francisco Zamora-Martinez
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
#ifndef BATCHFMEASURELOSSFUNCTION_H
#define BATCHFMEASURELOSSFUNCTION_H

#include "referenced.h"
#include "token_base.h"
#include "loss_function.h"

namespace ANN {
  /// A multi-class version of the F-Measure loss function as described in: Joan
  /// Pastor-Pellicer, Francisco Zamora-Martinez, Salvador España-Boquera, and
  /// M.J. Castro-Bleda.  F-Measure as the error function to train Neural
  /// Networks.  In Advances in Computational Intelligence, IWANN, part I, LNCS,
  /// pages 376-384. Springer, 2013.
  class BatchFMeasureLossFunction : public LossFunction {
    float beta, beta2;
    // auxiliary data for gradient computation speed-up
    float G1, G2, H;
    /// The dot product for each class => input(:,j) * target(:,j)
    MatrixFloat *dot_products;
    /// The sums per each class => input(:,j)->sum()
    MatrixFloat *input_sums, *target_sums;
    bool complement_output;
    
    BatchFMeasureLossFunction(BatchFMeasureLossFunction *other) :
    LossFunction(other), beta(other->beta), beta2(other->beta2),
    G1(other->G1), G2(other->G2), H(other->H),
    dot_products(0), input_sums(0), target_sums(0),
    complement_output(other->complement_output) {
      if (other->dot_products) dot_products = other->dot_products->clone();
      if (other->input_sums)   input_sums   = other->input_sums->clone();
      if (other->target_sums)  target_sums  = other->target_sums->clone();
    }
    
  protected:
    virtual MatrixFloat *computeLossBunch(Token *input, Token *target);
  public:
    BatchFMeasureLossFunction(unsigned int size, float beta=1.0f,
			      bool complement_output=false);
    virtual ~BatchFMeasureLossFunction();
    virtual Token *computeGradient(Token *input, Token *target);
    virtual LossFunction *clone() {
      return new BatchFMeasureLossFunction(this);
    }
  };
}

#endif // BATCHFMEASURELOSSFUNCTION_H
