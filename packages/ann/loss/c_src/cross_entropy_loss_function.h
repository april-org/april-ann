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
#ifndef CROSSENTROPYLOSSFUNCTION_H
#define CROSSENTROPYLOSSFUNCTION_H

#include "referenced.h"
#include "token_base.h"
#include "loss_function.h"

namespace ANN {
  /// Cross-entropy paired with log_logistic outputs
  class CrossEntropyLossFunction : public LossFunction {
    CrossEntropyLossFunction(CrossEntropyLossFunction *other) :
    LossFunction(other) { }
  protected:
    virtual Basics::MatrixFloat *computeLossBunch(Basics::Token *input,
                                                  Basics::Token *target);
  public:
    CrossEntropyLossFunction(unsigned int size);
    virtual ~CrossEntropyLossFunction();
    virtual Basics::Token *computeGradient(Basics::Token *input,
                                           Basics::Token *target);
    virtual float getAccumLoss();
    virtual LossFunction *clone() {
      return new CrossEntropyLossFunction(this);
    }
    virtual const char *luaCtorName() const {
      return "ann.loss.cross_entropy";
    }
  };

  //////////////////////////////////////////////////////////

  /// Cross-entropy non paired with log_logistic outputs
  class NonPairedCrossEntropyLossFunction : public LossFunction {
    NonPairedCrossEntropyLossFunction(NonPairedCrossEntropyLossFunction *other) :
    LossFunction(other) { }
  protected:
    virtual Basics::MatrixFloat *computeLossBunch(Basics::Token *input,
                                                  Basics::Token *target);
  public:
    NonPairedCrossEntropyLossFunction(unsigned int size);
    virtual ~NonPairedCrossEntropyLossFunction();
    virtual Basics::Token *computeGradient(Basics::Token *input,
                                           Basics::Token *target);
    virtual float getAccumLoss();
    virtual LossFunction *clone() {
      return new NonPairedCrossEntropyLossFunction(this);
    }
    virtual const char *luaCtorName() const {
      return "ann.loss.non_paired_cross_entropy";
    }
  };

}

#endif // CROSSENTROPYLOSSFUNCTION_H
