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
#ifndef BATCHFMEASUREMACROAVGLOSSFUNCTION_H
#define BATCHFMEASUREMACROAVGLOSSFUNCTION_H

#include "referenced.h"
#include "token_base.h"
#include "loss_function.h"
#include "smart_ptr.h"

namespace ANN {
  /// A multi-class version of the F-Measure loss function as described in: Joan
  /// Pastor-Pellicer, Francisco Zamora-Martinez, Salvador España-Boquera, and
  /// M.J. Castro-Bleda.  F-Measure as the error function to train Neural
  /// Networks.  In Advances in Computational Intelligence, IWANN, part I, LNCS,
  /// pages 376-384. Springer, 2013.
  ///
  /// FMeasure macro averaging computes an average of independent FM per class.
  class BatchFMeasureMacroAvgLossFunction : public LossFunction {
    float beta, beta2;
    // auxiliary data for gradient computation speed-up
    AprilUtils::SharedPtr<Basics::MatrixFloat> Gs, Hs;
    bool complement_output;
    
    BatchFMeasureMacroAvgLossFunction(BatchFMeasureMacroAvgLossFunction *other) :
    LossFunction(other), beta(other->beta), beta2(other->beta2),
    complement_output(other->complement_output) {
      if (!other->Gs.empty()) Gs = other->Gs->clone();
      if (!other->Hs.empty()) Hs = other->Hs->clone();
    }
    
  protected:
    virtual Basics::MatrixFloat *computeLossBunch(Basics::Token *input,
                                                  Basics::Token *target);
  public:
    BatchFMeasureMacroAvgLossFunction(unsigned int size, float beta=1.0f,
				      bool complement_output=false);
    virtual ~BatchFMeasureMacroAvgLossFunction();
    virtual Basics::Token *computeGradient(Basics::Token *input,
                                           Basics::Token *target);
    virtual LossFunction *clone() {
      return new BatchFMeasureMacroAvgLossFunction(this);
    }
    virtual const char *luaCtorName() const {
      return "ann.loss.batch_fmeasure_macro_avg";
    }
    virtual int exportParamsToLua(lua_State *L) {
      AprilUtils::LuaTable t(L);
      t["size"] = size;
      t["beta"] = beta;
      t["complement"] = complement_output;
      t.pushTable(L);
      return 1;
    }
  };
}

#endif // BATCHFMEASUREMACROAVGLOSSFUNCTION_H
