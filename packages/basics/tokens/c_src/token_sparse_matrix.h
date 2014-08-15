/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2014, Francisco Zamora-Martinez
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
#ifndef TOKEN_SPARSE_MATRIX_H
#define TOKEN_SPARSE_MATRIX_H

#include "disallow_class_methods.h"
#include "matrix.h"
#include "token_base.h"
#include "unused_variable.h"

namespace basics {

  template<typename T>
  class TokenSparseMatrix : public Token {
    APRIL_DISALLOW_COPY_AND_ASSIGN(TokenSparseMatrix);
    SparseMatrix<T> *mat;
  public:
    TokenSparseMatrix(SparseMatrix<T> *mat);
    ~TokenSparseMatrix();
    SparseMatrix<T> *getMatrix() { return mat; }
    const SparseMatrix<T> *getMatrix() const { return mat; }
    unsigned int size() const { return mat->size(); }
    Token *clone() const;
    april_utils::buffer_list* toString();
    april_utils::buffer_list* debugString(const char *prefix, int debugLevel);
    TokenCode getTokenCode() const;
    static Token *fromString(april_utils::constString &cs) {
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
  TokenSparseMatrix<T>::TokenSparseMatrix(SparseMatrix<T> *mat) :
    Token(),
    mat(mat) {
    IncRef(mat);
  }

  template<typename T>
  TokenSparseMatrix<T>::~TokenSparseMatrix() {
    DecRef(mat);
  }

  template<typename T>
  Token *TokenSparseMatrix<T>::clone() const {
    TokenSparseMatrix *token = new TokenSparseMatrix( mat->clone() );
    return token;
  }

  template<typename T>
  april_utils::buffer_list* TokenSparseMatrix<T>::toString() {
    // NOT IMPLEMENTED
    return 0;
  }

  template<typename T>
  april_utils::buffer_list* TokenSparseMatrix<T>::debugString(const char *prefix,
                                                              int debugLevel) {
    // NOT IMPLEMENTED
    UNUSED_VARIABLE(prefix);
    UNUSED_VARIABLE(debugLevel);
    return 0;
  }

  template<typename T>
  TokenCode TokenSparseMatrix<T>::getTokenCode() const {
    return table_of_token_codes::token_sparse_matrix;
  }

  ///////////////////////////////////////////////////////////////////////////

  typedef TokenSparseMatrix<float> TokenSparseMatrixFloat;

} // namespace basics

#endif // TOKEN_SPARSE_MATRIX_H
