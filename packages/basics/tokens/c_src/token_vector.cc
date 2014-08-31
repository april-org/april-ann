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
#include <cstdlib>
#include "unused_variable.h"
#include "token_vector.h"
#include "binarizer.h"
#include "table_of_token_codes.h"

using AprilUtils::buffer_list;
using AprilUtils::constString;
using AprilUtils::vector;
using AprilUtils::pair;

namespace Basics {

  template <typename T>
  TokenVector<T>::TokenVector() : TokenVectorGeneric() {
  }

  template <typename T>
  TokenVector<T>::TokenVector(unsigned int vlength) :
    TokenVectorGeneric(),
    vec(vlength) {
  }

  template <typename T>
  TokenVector<T>::TokenVector(const T *vec, unsigned int vlength) :
    TokenVectorGeneric(),
    vec(vec, vec+vlength) {
  }

  template <typename T>
  TokenVector<T>::TokenVector(const vector<T> &vec) :
    TokenVectorGeneric(),
    vec(vec) {
  }

  template <typename T>
  TokenVector<T>::~TokenVector() {
  }

  template <typename T>
  Token *TokenVector<T>::clone() const {
    return new TokenVector<T>(vec);
  }

  template <typename T>
  buffer_list* TokenVector<T>::toString() {
    // ERROR, no se puede serializar de forma general
    return 0; // para indicar error
  }

  template <typename T>
  buffer_list* TokenVector<T>::debugString(const char *prefix,
                                           int debugLevel) {
    UNUSED_VARIABLE(debugLevel);
    buffer_list *resul = new buffer_list;
    resul->add_formatted_string_right("%s TokenVector of generic type\n",prefix);
    return resul;
  }

  template <typename T>
  TokenCode TokenVector<T>::getTokenCode() const {
    return table_of_token_codes::error;
  }

  template <typename T>
  Token *TokenVector<T>::fromString(constString &cs) {
    UNUSED_VARIABLE(cs);
    return 0; // no hay un fromString generico
  }

  // ------------------------- vector float -------------------------

  template<>
  buffer_list* TokenVector<float>::toString() {
    buffer_list *resul = new buffer_list;
    resul->add_binarized_float_left(vec.begin(),vec.size());
    return resul;
  }

  template<>
  buffer_list* TokenVector<float>::debugString(const char *prefix,
                                               int debugLevel) {
    // TODO: de momento ignoramos debugLevel
    UNUSED_VARIABLE(debugLevel);
    buffer_list *resul = new buffer_list;
    resul->add_formatted_string_right("%s TokenVector_float with %d values\n",
                                      prefix,vec.size());
    return resul;
  }

  template <>
  TokenCode TokenVector<float>::getTokenCode() const {
    return table_of_token_codes::vector_float;
  }

  template <>
  Token *TokenVector<float>::fromString(constString &cs) {
    if (cs.len() % 5 != 0)
      return 0; // talla incorrecta
    int vec_len = cs.len() / 5;
    TokenVector<float> *resul = new TokenVector<float>(vec_len);
    bool all_is_ok = true;
    for (int i=0; all_is_ok && i<vec_len; i++) {
      all_is_ok = cs.extract_float_binary(&(resul->vec[i]));
    }
    if (!all_is_ok) {
      delete resul;
      resul = 0;
    }
    return resul;
  }

  // ------------------------- vector double -------------------------

  template <>
  buffer_list* TokenVector<double>::toString() {
    buffer_list *resul = new buffer_list;
    resul->add_binarized_double_left(vec.begin(),vec.size());
    return resul;
  }

  template <>
  buffer_list* TokenVector<double>::debugString(const char *prefix,
                                                int debugLevel) {
    // TODO: de momento ignoramos debugLevel
    UNUSED_VARIABLE(debugLevel);
    buffer_list *resul = new buffer_list;
    resul->add_formatted_string_right("%s TokenVector_double with %u values\n",
                                      prefix,vec.size());
    return resul;
  }

  template <>
  TokenCode TokenVector<double>::getTokenCode() const {
    return table_of_token_codes::vector_double;
  }

  template <>
  Token *TokenVector<double>::fromString(constString &cs) {
    if (cs.len() % 10 != 0)
      return 0; // talla incorrecta
    int vec_len = cs.len() / 10;
    TokenVector<double> *resul = new TokenVector<double>(vec_len);
    bool all_is_ok = true;
    for (int i=0; all_is_ok && i<vec_len; i++) {
      all_is_ok = cs.extract_double_binary(&(resul->vec[i]));
    }
    if (!all_is_ok) {
      delete resul;
      resul = 0;
    }
    return resul;
  }

  // ------------------------- vector int32 -------------------------

  template <>
  buffer_list* TokenVector<int32_t>::toString() {
    buffer_list *resul = new buffer_list;
    resul->add_binarized_int32_left(vec.begin(),vec.size());
    return resul;
  }

  template <>
  buffer_list* TokenVector<int32_t>::debugString(const char *prefix,
                                                 int debugLevel) {
    // TODO: de momento ignoramos debugLevel
    UNUSED_VARIABLE(debugLevel);
    buffer_list *resul = new buffer_list;
    resul->add_formatted_string_right("%s TokenVector_int32 with %d values\n",
                                      prefix,vec.size());
    return resul;
  }

  template <>
  TokenCode TokenVector<int32_t>::getTokenCode() const {
    return table_of_token_codes::vector_int32;
  }

  template <>
  Token *TokenVector<int32_t>::fromString(constString &cs) {
    if (cs.len() % 5 != 0)
      return 0; // talla incorrecta
    int vec_len = cs.len() / 5;
    TokenVector<int32_t> *resul = new TokenVector<int32_t>(vec_len);
    bool all_is_ok = true;
    for (int i=0; all_is_ok && i<vec_len; i++) {
      all_is_ok = cs.extract_int32_binary(&(resul->vec[i]));
    }
    if (!all_is_ok) {
      delete resul;
      resul = 0;
    }
    return resul;
  }

  // ------------------------- vector uint32_t -------------------------

  template <>
  buffer_list* TokenVector<uint32_t>::toString() {
    buffer_list *resul = new buffer_list;
    resul->add_binarized_uint32_left(vec.begin(),vec.size());
    return resul;
  }

  template <>
  buffer_list* TokenVector<uint32_t>::debugString(const char *prefix,
                                                  int debugLevel) {
    // TODO: de momento ignoramos debugLevel
    UNUSED_VARIABLE(debugLevel);
    buffer_list *resul = new buffer_list;
    resul->add_formatted_string_right("%s TokenVector_uint32 with %d values\n",
                                      prefix,vec.size());
    return resul;
  }

  template <>
  TokenCode TokenVector<uint32_t>::getTokenCode() const {
    return table_of_token_codes::vector_uint32;
  }

  template <>
  Token *TokenVector<uint32_t>::fromString(constString &cs) {
    if (cs.len() % 5 != 0)
      return 0; // talla incorrecta
    int vec_len = cs.len() / 5;
    TokenVector<uint32_t> *resul = new TokenVector<uint32_t>(vec_len);
    bool all_is_ok = true;
    for (int i=0; all_is_ok && i<vec_len; i++) {
      all_is_ok = cs.extract_uint32_binary(&(resul->vec[i]));
    }
    if (!all_is_ok) {
      delete resul;
      resul = 0;
    }
    return resul;
  }

  // ------------------------- vector char -------------------------

  template <>
  buffer_list* TokenVector<char>::toString() {
    buffer_list *resul = new buffer_list;
    resul->add_constString_left(constString(vec.begin(),vec.size()));
    return resul;
  }

  template <>
  buffer_list* TokenVector<char>::debugString(const char *prefix,
                                              int debugLevel) {
    // TODO: de momento ignoramos debugLevel
    UNUSED_VARIABLE(debugLevel);
    buffer_list *resul = new buffer_list;
    resul->add_formatted_string_right("%s TokenVector_char with %d values\n",
                                      prefix,vec.size());
    return resul;
  }

  template <>
  TokenCode TokenVector<char>::getTokenCode() const {
    return table_of_token_codes::vector_char;
  }

  template <>
  Token *TokenVector<char>::fromString(constString &cs) {
    TokenVector<char> *resul = new TokenVector<char>((const char*)cs,cs.len());
    return resul;
  }


  // ------------------------- vector token -------------------------

  template <>
  TokenVector<Token*>::TokenVector(Token * const *vec, unsigned int vlength) :
    TokenVectorGeneric(),
    vec(vec, vec+vlength) {
    for (unsigned int i=0; i<vlength; ++i) IncRef(vec[i]);
  }

  template <>
  TokenVector<Token*>::TokenVector(const vector<Token*> &vec) :
    TokenVectorGeneric(),
    vec(vec) {
    for (unsigned int i=0; i<vec.size(); ++i) IncRef(vec[i]);
  }

  template<>
  void TokenVector<Token*>::clear() {
    for (unsigned int i=0; i<vec.size(); ++i) {
      if (vec[i]) {
        DecRef(vec[i]);
        vec[i] = 0;
      }
    }
    vec.clear();
  }

  template<>
  void TokenVector<Token*>::push_back(Token *&data) {
    if (data) IncRef(data);
    vec.push_back(data);
  }

  template <>
  TokenVector<Token*>::~TokenVector() {
    for (unsigned int i=0; i<vec.size(); ++i) if (vec[i]) DecRef(vec[i]);
  }

  template <>
  buffer_list* TokenVector<Token*>::toString() {
    /*
      buffer_list *resul = new buffer_list;
      char b[5];
      constString cs(binarizer::code_uint32(vec.size()));
      resul->add_constString_left(cs);
      for (unsigned int i=0; i<vec.size(); ++i)
      resul->add_buffer_right(vec[i].toString());
      return resul;
    */
    // NOT IMPLEMENTED
    return 0;
  }

  template <>
  buffer_list* TokenVector<Token*>::debugString(const char *prefix,
                                                int debugLevel) {
    // TODO: de momento ignoramos debugLevel
    UNUSED_VARIABLE(debugLevel);
    buffer_list *resul = new buffer_list;
    resul->add_formatted_string_right("%s TokenVector_Token with %d values\n",
                                      prefix,vec.size());
    return resul;
  }

  template <>
  TokenCode TokenVector<Token*>::getTokenCode() const {
    return table_of_token_codes::vector_Tokens;
  }

  template <>
  Token *TokenVector<Token*>::fromString(constString &cs) {
    UNUSED_VARIABLE(cs);
    /*
      uint32_t size;
      cs.extract_uint32_binary(&size);
      TokenVector<Token> *resul = new TokenVector<Token>();
      resul->vec.resize(size);
      for (unsigned int i=0; i<size; ++i) {
      if (resul->
      resul->vec.push_back(
      }
      return resul;
    */
    // NOT IMPLEMENTED
    return 0;
  }

  // ----------------------------------------------------------------

  template class TokenVector<float>;
  template class TokenVector<double>;
  template class TokenVector<int32_t>;
  template class TokenVector<uint32_t>;
  template class TokenVector<char>;
  template class TokenVector<Token*>;

} // namespace Basics
