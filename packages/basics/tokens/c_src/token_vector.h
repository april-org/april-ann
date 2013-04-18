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
#include "pair.h"

using april_utils::vector;

class TokenVectorGeneric : public Token {
public:
  virtual unsigned int size() const = 0;
};

template <typename T>
class TokenVector : public TokenVectorGeneric {
  april_utils::vector<T> vec;
public:
  TokenVector();
  TokenVector(unsigned int vlength);
  // always copy the vector
  TokenVector(const T *vec, unsigned int vlength);
  TokenVector(const april_utils::vector<T> &vec);
  ~TokenVector();
  
  T& operator[] (unsigned int i) { return vec[i]; }
  const T& operator[] (unsigned int i) const { return vec[i]; }
  april_utils::vector<T> &getContainer() { return vec; }
  const april_utils::vector<T> &getContainer() const { return vec; }
  void push_back(const T &data) { vec.push_back(data); }
  void clear() { vec.clear(); }
  T *data() { return vec.begin(); }
  const T *data() const { return vec.begin(); }
  Token *clone() const;
  buffer_list* toString();
  buffer_list* debugString(const char *prefix, int debugLevel);
  TokenCode getTokenCode() const;
  static Token *fromString(constString &cs);
  virtual unsigned int size() const { return vec.size(); }
};

typedef TokenVector<float>    TokenVectorFloat;
typedef TokenVector<double>   TokenVectorDouble;
typedef TokenVector<int32_t>  TokenVectorInt32;
typedef TokenVector<uint32_t> TokenVectorUint32;
typedef TokenVector<char>     TokenVectorChar;
typedef TokenVector<Token*>   TokenBunchVector;
typedef TokenVector<april_utils::pair<unsigned int, float> > TokenSparseVectorFloat;

#endif // TOKEN_VECTOR_H
