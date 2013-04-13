#ifndef TOKEN_BASE_H
#define TOKEN_BASE_H

#include <cassert>
#include <typeinfo>
#include <stdint.h>
#include "buffer_list.h"
#include "constString.h"
#include "referenced.h"

class Token : public Referenced {
 public:
  Token();
  virtual ~Token();
  virtual Token *copyToken() const = 0;
  // FOR DEBUG PURPOSES
  virtual buffer_list* debugString(const char *prefix, int debugLevel)=0;
  virtual uint32_t codeTypeOfToken() const = 0;
  virtual buffer_list* toString()=0;

  // ALL TOKENS MUST IMPLEMENT THIS STATIC METHOD
  // Token *fromString(constString cs);

  template <typename T> bool isA() const {
    return typeid(T) == typeid(*this);
  }

  template <typename T> T convertTo() {
    T result = dynamic_cast<T>(this);
    assert(result != 0);
    return result;
  }
  
  template <typename T> const T convertTo() const {
    T result = dynamic_cast<const T>(this);
    assert(result != 0);
    return result;
  }
};
#endif // TOKEN_BASE_H
