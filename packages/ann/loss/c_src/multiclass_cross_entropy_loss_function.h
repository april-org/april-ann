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

  /// Multi-class cross-entropy paired with log_softmax outputs
  class MultiClassCrossEntropyLossFunction : public LossFunction {
    MultiClassCrossEntropyLossFunction(MultiClassCrossEntropyLossFunction *other) :
    LossFunction(other) { }
  protected:
    virtual Basics::MatrixFloat *computeLossBunch(Basics::Token *input,
                                                  Basics::Token *target);
  public:
    MultiClassCrossEntropyLossFunction(unsigned int size);
    virtual ~MultiClassCrossEntropyLossFunction();
    virtual Basics::Token *computeGradient(Basics::Token *input,
                                   Basics::Token *target);
    virtual LossFunction *clone() {
      return new MultiClassCrossEntropyLossFunction(this);
    }
    virtual const char *luaCtorName() const {
      return "ann.loss.multi_class_cross_entropy";
    }
  };
  
  ///////////////////////////////////
  
  /// Multi-class cross-entropy non paired with log_softmax outputs
  class NonPairedMultiClassCrossEntropyLossFunction : public LossFunction {
    NonPairedMultiClassCrossEntropyLossFunction(NonPairedMultiClassCrossEntropyLossFunction *other) :
    LossFunction(other) { }
  protected:
    virtual Basics::MatrixFloat *computeLossBunch(Basics::Token *input,
                                                  Basics::Token *target);
  public:
    NonPairedMultiClassCrossEntropyLossFunction(unsigned int size);
    virtual ~NonPairedMultiClassCrossEntropyLossFunction();
    virtual Basics::Token *computeGradient(Basics::Token *input,
                                           Basics::Token *target);
    virtual LossFunction *clone() {
      return new NonPairedMultiClassCrossEntropyLossFunction(this);
    }
    virtual const char *luaCtorName() const {
      return "ann.loss.non_paired_multi_class_cross_entropy";
    }
  };
}

#endif // MULTICLASSCROSSENTROPYLOSSFUNCTION_H
