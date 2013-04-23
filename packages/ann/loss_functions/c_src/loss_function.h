/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
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
#ifndef LOSSFUNCTION_H
#define LOSSFUNCTION_H

#include "referenced.h"
#include "token_base.h"
#include "token_memory_block.h"

namespace ANN {
  /// An abstract class that defines the basic interface that
  /// the loss_functions must complain.
  class LossFunction : public Referenced {
  protected:
    Token *input, *error_output;
    unsigned int size;
  public:
    LossFunction(unsigned int size) :
    Referenced(), input(0), error_output(0), size(size) {
    }
    virtual ~LossFunction() {
      if (error_output) DecRef(error_output);
      if (input) DecRef(input);
    }
    virtual float  addLoss(Token *input, Token *target) = 0;
    virtual Token *computeGrandient(Token *input, Token *target) = 0;
    virtual float  getAccumLoss() = 0;
    virtual void reset() = 0;
  };
}

#endif // LOSSFUNCTION_H
