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
#ifndef ZEROONELOSSFUNCTION_H
#define ZEROONELOSSFUNCTION_H

#include "referenced.h"
#include "token_base.h"
#include "loss_function.h"

namespace ANN {
  class ZeroOneLossFunction : public LossFunction {
    float accumulated_loss;
    unsigned int N;
    ZeroOneLossFunction(unsigned int size, float accumulated_loss, unsigned int N) :
      LossFunction(size), accumulated_loss(accumulated_loss), N(N) { }
  public:
    ZeroOneLossFunction(unsigned int size);
    virtual ~ZeroOneLossFunction();
    virtual float  addLoss(Token *input, Token *target);
    virtual Token *computeGradient(Token *input, Token *target);
    virtual float  getAccumLoss();
    virtual void reset();
    virtual LossFunction *clone() {
      return new ZeroOneLossFunction(size, accumulated_loss, N);
    }
  };
}

#endif // ZEROONELOSSFUNCTION_H
