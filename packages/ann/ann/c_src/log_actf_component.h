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
#ifndef LOGACTFCOMPONENT_H
#define LOGACTFCOMPONENT_H

#include "activation_function_component.h"
#include "ann_component.h"
#include "gpu_mirrored_memory_block.h"

namespace ANN {

  /// A component which computes the Log activation function.
  class LogActfANNComponent : public ActivationFunctionANNComponent {
    APRIL_DISALLOW_COPY_AND_ASSIGN(LogActfANNComponent);
    
  protected:
    virtual void applyActivation(Basics::MatrixFloat *input_units,
				 Basics::MatrixFloat *output_units);
    virtual void multiplyDerivatives(Basics::MatrixFloat *input_units,
				     Basics::MatrixFloat *output_units,
				     Basics::MatrixFloat *input_errors,
				     Basics::MatrixFloat *output_errors);
  public:
    LogActfANNComponent(const char *name);
    virtual ~LogActfANNComponent();
    virtual ANNComponent *clone(AprilUtils::LuaTable &copies);

    virtual const char *luaCtorName() const;
    // virtual int exportParamsToLua(lua_State *L);
  };
}

#endif // LOGACTFCOMPONENT_H
