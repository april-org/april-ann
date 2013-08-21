/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Salvador España-Boquera, Francisco Zamora-Martinez
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
#ifndef TOKEN_BASE_H
#define TOKEN_BASE_H

#include "april_assert.h"
#include <typeinfo>
#include <stdint.h>
#include "table_of_token_codes.h"
#include "buffer_list.h"
#include "constString.h"
#include "referenced.h"

/// A pure abstract class with interface of Token. A Token represents the
/// concept of a datatype throught Dataflow architectures (even Artificial
/// Neural Network components).
class Token : public Referenced {
 public:
  Token();
  virtual ~Token();
  /// This method implements a deep copy
  virtual Token *clone() const = 0;
  /// FOR DEBUG PURPOSES
  virtual buffer_list* debugString(const char *prefix, int debugLevel)=0;
  /// Abstract method which indicates the TokenCode of child Token classes
  virtual TokenCode    getTokenCode() const = 0;
  /// Abstract method for serialization purposes
  virtual buffer_list* toString()=0;

  // ALL TOKENS MUST IMPLEMENT THIS STATIC METHOD FOR SERIALIZATION PURPOSES
  // Token *fromString(constString &cs);
  
  /// Templatized method which returns true if a Token pointer could be
  /// transformed to a given Token child class
  template <typename T> bool isA() const {
    return typeid(T) == typeid(*this);
  }

  /// Templatized method which converts the given Token pointer to a Token child
  /// class
  template <typename T> T convertTo() {
    T result = dynamic_cast<T>(this);
    april_assert(result != 0);
    return result;
  }
  
  /// Templatized method which converts the given const Token pointer to a const
  /// Token child class
  template <typename T> const T convertTo() const {
    T result = dynamic_cast<const T>(this);
    april_assert(result != 0);
    return result;
  }
};
#endif // TOKEN_BASE_H
