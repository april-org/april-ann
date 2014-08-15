/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera, Adrian Palacios Corella, Francisco
 * Zamora-Martinez
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

#ifndef SKIP_FUNCTION_H
#define SKIP_FUNCTION_H

#include "function_interface.h"

namespace Functions {
  
  class SkipFunction : public FunctionInterface {
  public:
    SkipFunction() : FunctionInterface() { }
    virtual ~SkipFunction() {
    }
    /// It returns the input (or domain) size of the function.
    virtual unsigned int getInputSize() const {
      return 0;
    }
    /// It returns the output (or range) size of the function.
    virtual unsigned int getOutputSize() const {
      return 0;
    }
    virtual basics::Token *calculate(basics::Token *input) {
      return input;
    }
  };
}

#endif //SKIP_FUNCTION_H
