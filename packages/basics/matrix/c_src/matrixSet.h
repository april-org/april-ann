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

#include <cmath>
#include "referenced.h"
#include "matrix.h"
#include "aux_hash_table.h"
#include "hash_table.h"
#include "mystring.h"
#include "maxmin.h"

template<typename T>
class MatrixSet : public Referenced {
  typedef april_utils::hash<april_utils::string, Matrix<T> *> DictType;
  DictType matrix_dict;
public:
  MatrixSet() { }
  ~MatrixSet() {
    for (typename DictType::iterator it = matrix_dict.begin();
	 it!=matrix_dict.end(); ++it) {
      DecRef(it->second);
    }
  }
  Matrix<T> *&operator[](const april_utils::string &k) {
    return matrix_dict[k];
  }
  Matrix<T> *find(const april_utils::string &k) const {
    Matrix<T> **ptr = matrix_dict.find(k);
    return (ptr!=0) ? (*ptr) : 0;
  }
  // matrix component-wise operators macros
#define MAKE_N0_OPERATOR(NAME)					\
  void NAME() {							\
    for (typename DictType::iterator it = matrix_dict.begin();	\
	 it!=matrix_dict.end(); ++it) {				\
      it->second->NAME();					\
    }								\
  }
#define MAKE_N1_OPERATOR(NAME,TYPE1)				\
  void NAME(const TYPE1 &v1) {					\
    for (typename DictType::iterator it = matrix_dict.begin();	\
	 it!=matrix_dict.end(); ++it) {				\
      it->second->NAME(v1);					\
    }								\
  }
#define MAKE_N2_OPERATOR(NAME,TYPE1,TYPE2)			\
  void NAME(const TYPE1 &v1, const TYPE2 &v2) {			\
    for (typename DictType::iterator it = matrix_dict.begin();	\
	 it!=matrix_dict.end(); ++it) {				\
      it->second->NAME(v1,v2);					\
    }								\
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
#define MAKE_OPERATOR(NAME)						\
  void NAME(const MatrixSet<T> *other) {				\
    for (typename DictType::const_iterator it = matrix_dict.begin();	\
	 it!=matrix_dict.end(); ++it) {					\
      Matrix<T> *a = it->second;					\
      const Matrix<T> *b = other->find(it->first);			\
      if (b == 0)							\
	ERROR_EXIT1(128, "Matrix with name %s not found\n",		\
		    it->first.c_str());					\
      a->NAME(b);							\
    }									\
  }
  // two matrix basic math operator declarations
  MAKE_OPERATOR(cmul);
  MAKE_OPERATOR(copy);
#undef MAKE_OPERATOR

  // AXPY
  void axpy(T alpha, const MatrixSet<T> *other) {
    for (typename DictType::const_iterator it = matrix_dict.begin();
         it!=matrix_dict.end(); ++it) {
      Matrix<T> *a = it->second;
      const Matrix<T> *b = other->find(it->first);
      if (b == 0)
        ERROR_EXIT1(128, "Matrix with name %s not found\n",
                    it->first.c_str());
      a->axpy(alpha, b);
    }
  }

  // EQUALS
  void equals(const MatrixSet<T> *other, T epsilon) {
    for (typename DictType::const_iterator it = matrix_dict.begin();
         it!=matrix_dict.end(); ++it) {
      Matrix<T> *a = it->second;
      const Matrix<T> *b = other->find(it->first);
      if (b == 0)
        ERROR_EXIT1(128, "Matrix with name %s not found\n",
                    it->first.c_str());
      a->equals(b, epsilon);
    }
  }
  
  // matrix math reductions
  T norm2() {
    T result_norm2 = 0.0f;
    for (typename DictType::iterator it = matrix_dict.begin();
         it!=matrix_dict.end(); ++it) {
      T current_norm2 = it->second->norm2();
      result_norm2 = result_norm2 + current_norm2*current_norm2;
    }
    // FIXME: this call only work with float
    return sqrtf(result_norm2);
  }
};

typedef MatrixSet<float> MatrixFloatSet;
