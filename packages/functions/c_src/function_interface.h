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

#ifndef FUNCTION_INTERFACE_H
#define FUNCTION_INTERFACE_H

#include "error_print.h"
#include "referenced.h"
#include "token_base.h"

/// Generic functions that receive and produce Basics::Token pointers.
namespace Functions {
  
  /// A virtual class that serves as high level interface.
  /**
     A FunctionInterface is an abstraction of an object which represents a
     mathematical function. It adds to the interface abstract methods which
     calculates output vector given input vector. Every function is feeded with
     an input Token and produces an output Token.
   */
  class FunctionInterface : public Referenced {
  public:
    FunctionInterface() : Referenced() { }
    virtual ~FunctionInterface() {
    }
    /// It returns the input (or domain) size of the function.
    virtual unsigned int getInputSize()  const = 0;
    /// It returns the output (or range) size of the function.
    virtual unsigned int getOutputSize() const = 0;
    
    /**
     * @brief A new abstract method that computes output vector given input
     * vector.
     *
     * The function doesn't receive the ownership of the given Basics::Token.
     *
     * @note ANN based filters would IncRef and DecRef the given input
     * Basics::Token, so be careful with this.
     *
     * @note Some functions work in-place, so take it into account.
     *
     * @todo const Token *input
     */
    virtual Basics::Token *calculate(Basics::Token *input) = 0;
  };
}

#endif //FUNCTION_INTERFACE_H
