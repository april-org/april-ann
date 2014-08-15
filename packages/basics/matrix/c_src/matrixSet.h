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

#ifndef MATRIXSET_H
#define MATRIXSET_H

#include <cmath>
#include "referenced.h"
#include "matrix.h"
#include "matrixFloat.h"
#include "aux_hash_table.h"
#include "hash_table.h"
#include "mystring.h"
#include "maxmin.h"
#include "smart_ptr.h"

namespace basics {

  /**
   * The class MatrixSet is a hash map of string=>Matrix<T>, allowing to compute
   * math operations to all the underlying matrices, or between two MatrixSet.
   */
  template<typename T>
  class MatrixSet : public Referenced {
    typedef april_utils::hash< april_utils::string,
                               april_utils::SharedPtr< Matrix<T> > > DictType;
    DictType matrix_dict;
  public:
    typedef typename DictType::iterator       iterator;
    typedef typename DictType::const_iterator const_iterator;

    //

    MatrixSet() : matrix_dict(32, 2.0f) { }
    virtual ~MatrixSet() { }

    iterator begin() { return matrix_dict.begin(); }

    iterator end()   { return matrix_dict.end(); }

    MatrixSet<T> *clone() {
      MatrixSet<T> *cloned = new MatrixSet<T>();
      for (iterator it = matrix_dict.begin(); it!=matrix_dict.end(); ++it) {
        cloned->insert(it->first, it->second->clone());
      }
      return cloned;
    }

    MatrixSet<T> *cloneOnlyDims() {
      MatrixSet<T> *cloned = new MatrixSet<T>();
      for (iterator it = matrix_dict.begin(); it!=matrix_dict.end(); ++it) {
        cloned->insert(it->first, it->second->cloneOnlyDims());
      }
      return cloned;
    }

    // operator[]
    april_utils::SharedPtr< Matrix<T> > &operator[](const char *k) {
      return matrix_dict[april_utils::string(k)];
    }
    april_utils::SharedPtr< Matrix<T> > &operator[](const april_utils::string &k) {
      return matrix_dict[k];
    }
    // insert operation
    void insert(const char *k, Matrix<T> *v) {
      return insert(april_utils::string(k), v);
    }
    void insert(const april_utils::string &k, Matrix<T> *v) {
      matrix_dict[k] = v;
    }
    // find operation
    Matrix<T> *find(const char *k) const {
      return find(april_utils::string(k));
    }
    Matrix<T> *find(const april_utils::string &k) const {
      april_utils::SharedPtr< Matrix<T> > *ptr = matrix_dict.find(k);
      return (ptr!=0) ? (ptr->get()) : 0;
    }
    // matrix component-wise operators macros
#define MAKE_N0_OPERATOR(NAME)			\
    void NAME() {                               \
      for (iterator it = matrix_dict.begin();	\
           it!=matrix_dict.end(); ++it) {       \
        it->second->NAME();			\
      }						\
    }
#define MAKE_N1_OPERATOR(NAME,TYPE1)		\
    void NAME(const TYPE1 &v1) {                \
      for (iterator it = matrix_dict.begin();	\
           it!=matrix_dict.end(); ++it) {       \
        it->second->NAME(v1);			\
      }						\
    }
#define MAKE_N2_OPERATOR(NAME,TYPE1,TYPE2)              \
    void NAME(const TYPE1 &v1, const TYPE2 &v2) {	\
      for (iterator it = matrix_dict.begin();           \
           it!=matrix_dict.end(); ++it) {		\
        it->second->NAME(v1,v2);			\
      }                                                 \
    }
    // matrix component-wise operators declaration
    MAKE_N1_OPERATOR(fill,T);
    MAKE_N2_OPERATOR(clamp,T,T);
    MAKE_N0_OPERATOR(zeros);
    MAKE_N0_OPERATOR(ones);
    MAKE_N1_OPERATOR(scalarAdd,T);
    MAKE_N0_OPERATOR(plogp);
    MAKE_N0_OPERATOR(log);
    MAKE_N0_OPERATOR(log1p);
    MAKE_N0_OPERATOR(exp);
    MAKE_N0_OPERATOR(sqrt);
    MAKE_N1_OPERATOR(pow,T);
    MAKE_N0_OPERATOR(tan);
    MAKE_N0_OPERATOR(tanh);
    MAKE_N0_OPERATOR(atan);
    MAKE_N0_OPERATOR(atanh);
    MAKE_N0_OPERATOR(cos);
    MAKE_N0_OPERATOR(cosh);
    MAKE_N0_OPERATOR(acos);
    MAKE_N0_OPERATOR(acosh);
    MAKE_N0_OPERATOR(sin);
    MAKE_N0_OPERATOR(sinh);
    MAKE_N0_OPERATOR(asin);
    MAKE_N0_OPERATOR(asinh);
    MAKE_N0_OPERATOR(abs);
    MAKE_N0_OPERATOR(complement);
    MAKE_N0_OPERATOR(sign);
    MAKE_N1_OPERATOR(scal,T);
    MAKE_N2_OPERATOR(adjustRange,T,T);
    MAKE_N0_OPERATOR(inv);
    MAKE_N0_OPERATOR(pruneSubnormalAndCheckNormal);
#undef MAKE_N1_OPERATOR
#undef MAKE_N2_OPERATOR

    // two matrix basic math operator macros
#define MAKE_OPERATOR(NAME)					\
    void NAME(const MatrixSet<T> *other) {			\
      for (iterator it = matrix_dict.begin();			\
           it!=matrix_dict.end(); ++it) {                       \
        Matrix<T> *a = it->second.get();                        \
        const Matrix<T> *b = other->find(it->first);		\
        if (b == 0)						\
          ERROR_EXIT1(128, "Matrix with name %s not found\n",	\
                      it->first.c_str());                       \
        a->NAME(b);						\
      }								\
    }
    // two matrix basic math operator declarations
    MAKE_OPERATOR(cmul);
    MAKE_OPERATOR(copy);
#undef MAKE_OPERATOR

    // AXPY
    void axpy(T alpha, const MatrixSet<T> *other) {
      for (iterator it = matrix_dict.begin(); it!=matrix_dict.end(); ++it) {
        Matrix<T> *a = it->second.get();
        const Matrix<T> *b = other->find(it->first);
        if (b == 0)
          ERROR_EXIT1(128, "Matrix with name %s not found\n",
                      it->first.c_str());
        a->axpy(alpha, b);
      }
    }

    // EQUALS
    void equals(const MatrixSet<T> *other, T epsilon) {
      for (iterator it = matrix_dict.begin(); it!=matrix_dict.end(); ++it) {
        const Matrix<T> *a = it->second.get();
        const Matrix<T> *b = other->find(it->first);
        if (b == 0)
          ERROR_EXIT1(128, "Matrix with name %s not found\n",
                      it->first.c_str());
        a->equals(b, epsilon);
      }
    }

    // matrix math reductions
    T norm2() {
      T result_norm2 = T();
      for (iterator it = matrix_dict.begin(); it!=matrix_dict.end(); ++it) {
        T current_norm2 = it->second->norm2();
        result_norm2 = result_norm2 + current_norm2*current_norm2;
      }
      // FIXME: this call only work with float
      return sqrtf(result_norm2);
    }

    // matrix math reductions
    int size() {
      int total_size = 0;
      for (iterator it = matrix_dict.begin(); it!=matrix_dict.end(); ++it)
        total_size += it->second->size();
      return total_size;
    }

    // dot reduction
    T dot(MatrixSet<T> *other) {
      T result = T();
      for (iterator it = matrix_dict.begin(); it!=matrix_dict.end(); ++it) {
        april_utils::SharedPtr< Matrix<T> > a( it->second.get() );
        april_utils::SharedPtr< Matrix<T> > b( other->find(it->first) );
        if (b.empty()) {
          ERROR_EXIT1(128, "Matrix with name %s not found\n",
                      it->first.c_str());
        }
        if (!a->getIsContiguous()) a = a->clone();
        if (!b->getIsContiguous()) b = b->clone();
        int a_size = a->size();
        int b_size = b->size();
        a = a->rewrap(&a_size, 1);
        b = b->rewrap(&b_size, 1);
        result = result + a->dot(b.get());
      }
      return result;
    }

  };

} // namespace basics

#endif // MATRIXSET_H
