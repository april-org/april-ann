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
