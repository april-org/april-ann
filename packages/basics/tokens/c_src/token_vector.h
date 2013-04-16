/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
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
#ifndef TOKEN_VECTOR_H
#define TOKEN_VECTOR_H

#include "vector.h"
#include "token_base.h"

using namespace april_utils::vector;

class TokenVectorGeneric : public TokenBase {
public:
  unsigned int size() const = 0;
};

template <typename T>
class TokenVector : public TokenVectorGeneric {
  vector<T> vec;
public:
  TokenVector();
  TokenVector(unsigned int vlength);
  // always copy the vector
  TokenVector(const T *vec, unsigned int vlength);
  TokenVector(const vector<T> &vec);
  ~TokenVector();
  
  T& operator[] (unsigned int i) { return vec[i]; }
  const T& operator[] (unsigned int i) const { return vec[i]; }
  vector<T> &data() { return vec; }
  const vector<T> &data() const { return vec; }
  Token *clone() const;
  buffer_list* toString();
  buffer_list* debugString(const char *prefix, int debugLevel);
  TokenCode getTokenCode() const;
  static Token *fromString(constString &cs);
};

typedef TokenVector<float>    TokenVectorFloat;
typedef TokenVector<double>   TokenVectorDouble;
typedef TokenVector<int32_t>  TokenVectorInt32;
typedef TokenVector<uint32_t> TokenVectorUint32;
typedef TokenVector<char>     TokenVectorChar;
typedef TokenVector<Token*>   TokenBunchVector;

#endif // TOKEN_VECTOR_H
