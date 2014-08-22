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
#ifndef TOKEN_NULL_H
#define TOKEN_NULL_H

#include "april_assert.h"
#include "buffer_list.h"
#include "constString.h"
#include "disallow_class_methods.h"
#include "referenced.h"
#include "table_of_token_codes.h"
#include "token_base.h"
#include "unused_variable.h"

namespace basics {

  class TokenNull : public Token {
    APRIL_DISALLOW_COPY_AND_ASSIGN(TokenNull);
  public:
    TokenNull() { }
    virtual ~TokenNull() { }
    
    virtual Token *clone() const { return new TokenNull(); }
    
    // FOR DEBUG PURPOSES
    // TODO:
    virtual april_utils::buffer_list* debugString(const char *prefix,
                                                  int debugLevel) {
      UNUSED_VARIABLE(prefix);
      UNUSED_VARIABLE(debugLevel);
      return 0;
    }
    
    virtual TokenCode getTokenCode() const {
      return table_of_token_codes::token_null;
    }
    
    // TODO:
    virtual april_utils::buffer_list* toString() { return 0; }

    // ALL TOKENS MUST IMPLEMENT THIS STATIC METHOD FOR SERIALIZATION PURPOSES
    // TODO:
    // Token *fromString(april_utils::constString &cs);
  };

} // namespace basics

#endif // TOKEN_NULL_H
