/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
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
#ifndef MSELOSSFUNCTION_H
#define MSELOSSFUNCTION_H

#include "referenced.h"
#include "token_base.h"
#include "loss_function.h"

namespace ANN {
  class MSELossFunction : public LossFunction {
    TokenMemoryBlock *error_mem_block;
    float accumulated_loss;
  public:
    MSELossFunction(unsigned int size);
    virtual ~MSELossFunction();
    virtual float  addLoss(Token *input, Token *target);
    virtual Token *computeGrandient(Token *input, Token *target);
    virtual float  getAccumLoss();
    virtual void reset();
  };
}

#endif // MSELOSSFUNCTION_H
