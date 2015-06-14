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
#ifndef MAELOSSFUNCTION_H
#define MAELOSSFUNCTION_H

#include "referenced.h"
#include "token_base.h"
#include "loss_function.h"

namespace ANN {
  class MAELossFunction : public LossFunction {
    MAELossFunction(MAELossFunction *other) : LossFunction(other) { }
  protected:
    virtual Basics::MatrixFloat *computeLossBunch(Basics::Token *input,
                                                  Basics::Token *target);
  public:
    MAELossFunction(unsigned int size);
    virtual ~MAELossFunction();
    virtual Basics::Token *computeGradient(Basics::Token *input,
                                           Basics::Token *target);
    virtual LossFunction *clone() {
      return new MAELossFunction(this);
    }
    virtual const char *luaCtorName() const {
      return "ann.loss.mae";
    }
  };
}

#endif // MAELOSSFUNCTION_H
