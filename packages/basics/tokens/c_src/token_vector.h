#ifndef TOKEN_VECTOR_H
#define TOKEN_VECTOR_H

#include "token_base.h"

class TokenVectorGeneric : public TokenBase {
 protected:
  unsigned int vector_length;
 public:
  TokenVectorGeneric(unsigned int vector_length) :
  vector_length(vector_length) { }
  unsigned int length() const { return vector_length; }
};

template <typename T>
class TokenVector : public TokenVectorGeneric {
  bool vector_owner;
  T *vec;
 public:
  T& operator[] (unsigned int i) { return vec[i]; }
  const T& operator[] (unsigned int i) const { return vec[i]; }
  T* data() { return vec; }
  const T* data() const { return vec; }
  TokenVector(unsigned int vlength);
  TokenVector(T *vec, unsigned int vlength, bool vector_owner=true);
  TokenVector(const T *vec, unsigned int vlength);
  ~TokenVector();
  token copyToken() const;
  buffer_list* toString();
  buffer_list* debugString(const char *prefix, int debugLevel);
  uint32_t codeTypeOfToken() const;
  static token fromString(constString cs);
};

typedef TokenVector<float>    TokenVectorFloat;
typedef TokenVector<double>   TokenVectorDouble;
typedef TokenVector<int32_t>  TokenVectorInt32;
typedef TokenVector<uint32_t> TokenVectorUint32;
typedef TokenVector<char>     TokenVectorChar;

#endif // TOKEN_VECTOR_H
