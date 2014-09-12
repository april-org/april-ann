/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Francisco Zamora-Martinez
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
#ifndef TOKEN_MATRIX_H
#define TOKEN_MATRIX_H

#include "disallow_class_methods.h"
#include "matrix.h"
#include "token_base.h"
#include "unused_variable.h"

namespace Basics {

  template<typename T>
  class TokenMatrix : public Token {
    APRIL_DISALLOW_COPY_AND_ASSIGN(TokenMatrix);
    Matrix<T> *mat;
  public:
    TokenMatrix(Matrix<T> *mat);
    ~TokenMatrix();
    Matrix<T> *getMatrix() { return mat; }
    const Matrix<T> *getMatrix() const { return mat; }
    unsigned int size() const { return mat->size(); }
    Token *clone() const;
    AprilUtils::buffer_list* toString();
    AprilUtils::buffer_list* debugString(const char *prefix, int debugLevel);
    TokenCode getTokenCode() const;
    static Token *fromString(AprilUtils::constString &cs) {
      // NOT IMPLEMENTED
      UNUSED_VARIABLE(cs);
      return 0;
    }
    bool getCudaFlag() { return mat->getCudaFlag(); }
    void printDebug() {
      /*
        const float *data = mem_block->getPPALForRead();
        for (unsigned int i=0; i<used_size; ++i)
        printf ("%f ", data[i]);
        printf("\n");
      */
    }
  };

  template<typename T>
  TokenMatrix<T>::TokenMatrix(Matrix<T> *mat) :
    Token(),
    mat(mat) {
    IncRef(mat);
  }

  template<typename T>
  TokenMatrix<T>::~TokenMatrix() {
    DecRef(mat);
  }

  template<typename T>
  Token *TokenMatrix<T>::clone() const {
    TokenMatrix *token = new TokenMatrix( mat->clone() );
    return token;
  }

  template<typename T>
  AprilUtils::buffer_list* TokenMatrix<T>::toString() {
    // NOT IMPLEMENTED
    return 0;
  }
  
  template<typename T>
  AprilUtils::buffer_list* TokenMatrix<T>::debugString(const char *prefix,
                                                       int debugLevel) {
    // NOT IMPLEMENTED
    UNUSED_VARIABLE(prefix);
    UNUSED_VARIABLE(debugLevel);
    return 0;
  }

  template<typename T>
  TokenCode TokenMatrix<T>::getTokenCode() const {
    return table_of_token_codes::token_matrix;
  }

  ///////////////////////////////////////////////////////////////////////////

  typedef TokenMatrix<float> TokenMatrixFloat;

} // namespace Basics

#endif // TOKEN_MATRIX_H
