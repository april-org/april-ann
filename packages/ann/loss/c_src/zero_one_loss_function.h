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
    /// Threshold for take the neuron as activated. It is only necessary when
    /// the neural network has one logistic output neuron, so it is solving a
    /// two-class problem.
    float TH;
    ZeroOneLossFunction(ZeroOneLossFunction *other) : LossFunction(other),
						      TH(other->TH) { }
    virtual Basics::MatrixFloat *computeLossBunch(Basics::Token *input,
                                                  Basics::Token *target);
  public:
    ZeroOneLossFunction(unsigned int size, float TH=0.5f);
    virtual ~ZeroOneLossFunction();
    virtual Basics::Token *computeGradient(Basics::Token *input,
                                           Basics::Token *target);
    virtual LossFunction *clone() {
      return new ZeroOneLossFunction(this);
    }
    virtual char *toLuaString();
    virtual const char *luaCtorName() const {
      return "ann.loss.zero_one";
    }
    virtual int exportParamsToLua(lua_State *L) {
      lua_pushinteger(L, size);
      lua_pushnumber(L, TH);
      return 2;
    }
  };
}

#endif // ZEROONELOSSFUNCTION_H
