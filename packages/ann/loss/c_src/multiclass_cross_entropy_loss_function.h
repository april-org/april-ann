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
#ifndef MULTICLASSCROSSENTROPYLOSSFUNCTION_H
#define MULTICLASSCROSSENTROPYLOSSFUNCTION_H

#include "referenced.h"
#include "token_base.h"
#include "loss_function.h"

namespace ANN {
  class MultiClassCrossEntropyLossFunction : public LossFunction {
    MultiClassCrossEntropyLossFunction(MultiClassCrossEntropyLossFunction *other) :
    LossFunction(other) { }
  protected:
    virtual MatrixFloat *computeLossBunch(Token *input, Token *target);
  public:
    MultiClassCrossEntropyLossFunction(unsigned int size);
    virtual ~MultiClassCrossEntropyLossFunction();
    virtual Token *computeGradient(Token *input, Token *target);
    virtual LossFunction *clone() {
      return new MultiClassCrossEntropyLossFunction(this);
    }
  };
}

#endif // MULTICLASSCROSSENTROPYLOSSFUNCTION_H
