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

#ifndef IDENTITY_FUNCTION_H
#define IDENTITY_FUNCTION_H

#include "function_interface.h"

namespace Functions {
  
  class IdentityFunction : public FunctionInterface {
  public:
    IdentityFunction() : FunctionInterface() { }
    virtual ~IdentityFunction() {
    }
    /// It returns the input (or domain) size of the function.
    virtual unsigned int getInputSize() const {
      return 0;
    }
    /// It returns the output (or range) size of the function.
    virtual unsigned int getOutputSize() const {
      return 0;
    }
    virtual Token *calculate(Token *input) {
      return input;
    }
  };
}

#endif // IDENTITY_FUNCTION_H
