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

#include "aux_hash_table.h"
#include "hash_table.h"
#include "matrix.h"
#include "matrixFloat.h"
#include "maxmin.h"
#include "mystring.h"
#include "referenced.h"
#include "smart_ptr.h"
#include "sparse_matrix.h"
#include "sparse_matrixFloat.h"

namespace Basics {

  /**
   * The class MatrixSet is a hash map of string=>Matrix<T>, allowing to compute
   * math operations to all the underlying matrices, or between two MatrixSet.
   */
  template<typename T>
    class MatrixSet : public Referenced {
  public:

    ///////////////////////////////////////////////////////////////////////
    
    class Value {
    public:
      Value() {}
      Value(Matrix<T> *m) : dense(m) {}
      Value(SparseMatrix<T> *m) : sparse(m) {}
      bool isSparse() const { return !sparse.empty(); }
      bool empty() const { return sparse.empty() && dense.empty(); }
      void assign(Matrix<T> *m) {
        dense.reset(m);
        sparse.reset();
      }
      void assign(SparseMatrix<T> *m) {
        sparse.reset(m);
        dense.reset();
      }
      AprilUtils::SharedPtr< Matrix<T> > &checkDense() {
        if (dense.empty()) ERROR_EXIT(128, "Impossible to retrive a dense matrix\n");
        return dense;
      }
      AprilUtils::SharedPtr< SparseMatrix<T> > &checkSparse() {
        if (sparse.empty()) ERROR_EXIT(128, "Impossible to retrive a sparse matrix\n");
        return sparse;
      }
      AprilUtils::SharedPtr< Matrix<T> > &getDense() {
        return dense;
      }
      AprilUtils::SharedPtr< SparseMatrix<T> > &getSparse() {
        return sparse;
      }
      const AprilUtils::SharedPtr< Matrix<T> > &getDense() const {
        return dense;
      }
      const AprilUtils::SharedPtr< SparseMatrix<T> > &getSparse() const {
        return sparse;
      }
    private:
      AprilUtils::SharedPtr< Matrix<T> > dense;
      AprilUtils::SharedPtr< SparseMatrix<T> > sparse;
    };
    
    ///////////////////////////////////////////////////////////////////////
    
  private:
    typedef AprilUtils::hash<AprilUtils::string, Value> DictType;
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
        if (it->second.isSparse()) {
          cloned->insert(it->first, it->second.getSparse()->clone());
        }
        else {
          cloned->insert(it->first, it->second.checkDense()->clone());
        }
      }
      return cloned;
    }

    MatrixSet<T> *cloneOnlyDims() {
      MatrixSet<T> *cloned = new MatrixSet<T>();
      for (iterator it = matrix_dict.begin(); it!=matrix_dict.end(); ++it) {
        if (it->second.isSparse())
          ERROR_EXIT(256, "Impossible to cloneOnlyDims of MatrixSet objects "
                     "with sparse matrices\n");
        cloned->insert(it->first, it->second.checkDense()->cloneOnlyDims());
      }
      return cloned;
    }

    // operator[]
    Value &operator[](const char *k) {
      return matrix_dict[AprilUtils::string(k)];
    }
    Value &operator[](const AprilUtils::string &k) {
      return matrix_dict[k];
    }
    // insert operation
    void insert(const char *k, Matrix<T> *v) {
      return insert(AprilUtils::string(k), v);
    }
    void insert(const AprilUtils::string &k, Matrix<T> *v) {
      Value &old = matrix_dict[k];
      old.assign(v);
    }
    void insert(const char *k, SparseMatrix<T> *v) {
      return insert(AprilUtils::string(k), v);
    }
    void insert(const AprilUtils::string &k, SparseMatrix<T> *v) {
      Value &old = matrix_dict[k];
      old.assign(v);
    }
    // find operation
    Value *find(const char *k) const {
      return find(AprilUtils::string(k));
    }
    Value *find(const AprilUtils::string &k) const {
      Value *ptr = matrix_dict.find(k);
      return (ptr!=0) ? ptr : 0;
    }
    // matrix component-wise operators macros
#define MAKE_N0_OPERATOR(NAME,FUNC)                                         \
    void NAME() {                                                       \
      for (iterator it = matrix_dict.begin();                           \
           it!=matrix_dict.end(); ++it) {                               \
        if (it->second.isSparse())                                      \
          ERROR_EXIT(256, "Impossible to execute operators with sparse matrices\n"); \
        AprilMath::MatrixExt::Operations::FUNC(it->second.checkDense().get()); \
      }                                                                 \
    }
#define MAKE_N1_OPERATOR(NAME,FUNC,TYPE1)                               \
    void NAME(const TYPE1 &v1) {                                        \
      for (iterator it = matrix_dict.begin();                           \
           it!=matrix_dict.end(); ++it) {                               \
        if (it->second.isSparse())                                      \
          ERROR_EXIT(256, "Impossible to execute operators with sparse matrices\n"); \
        AprilMath::MatrixExt::Operations::FUNC(it->second.checkDense().get(), v1); \
      }                                                                 \
    }
#define MAKE_N2_OPERATOR(NAME,FUNC,TYPE1,TYPE2)                         \
    void NAME(const TYPE1 &v1, const TYPE2 &v2) {                       \
      for (iterator it = matrix_dict.begin();                           \
           it!=matrix_dict.end(); ++it) {                               \
        if (it->second.isSparse())                                      \
          ERROR_EXIT(256, "Impossible to execute operators with sparse matrices\n"); \
        AprilMath::MatrixExt::Operations::FUNC(it->second.checkDense().get(),v1,v2); \
      }                                                                 \
    }
    // matrix component-wise operators declaration
    MAKE_N1_OPERATOR(fill,matFill,T);
    MAKE_N2_OPERATOR(clamp,matClamp,T,T);
    MAKE_N0_OPERATOR(zeros,matZeros);
    MAKE_N0_OPERATOR(ones,matOnes);
    MAKE_N1_OPERATOR(scalarAdd,matScalarAdd,T);
    MAKE_N0_OPERATOR(plogp,matPlogp);
    MAKE_N0_OPERATOR(log,matLog);
    MAKE_N0_OPERATOR(log1p,matLog1p);
    MAKE_N0_OPERATOR(exp,matExp);
    MAKE_N0_OPERATOR(sqrt,matSqrt);
    MAKE_N1_OPERATOR(pow,matPow,T);
    MAKE_N0_OPERATOR(tan,matTan);
    MAKE_N0_OPERATOR(tanh,matTanh);
    MAKE_N0_OPERATOR(atan,matAtan);
    MAKE_N0_OPERATOR(atanh,matAtanh);
    MAKE_N0_OPERATOR(cos,matCos);
    MAKE_N0_OPERATOR(cosh,matCosh);
    MAKE_N0_OPERATOR(acos,matAcos);
    MAKE_N0_OPERATOR(acosh,matAcosh);
    MAKE_N0_OPERATOR(sin,matSin);
    MAKE_N0_OPERATOR(sinh,matSinh);
    MAKE_N0_OPERATOR(asin,matAsin);
    MAKE_N0_OPERATOR(asinh,matAsinh);
    MAKE_N0_OPERATOR(abs,matAbs);
    MAKE_N0_OPERATOR(complement,matComplement);
    MAKE_N0_OPERATOR(sign,matSign);
    MAKE_N1_OPERATOR(scal,matScal,T);
    MAKE_N2_OPERATOR(adjustRange,matAdjustRange,T,T);
    MAKE_N0_OPERATOR(inv,matInv);
#undef MAKE_N1_OPERATOR
#undef MAKE_N2_OPERATOR

    void pruneSubnormalAndCheckNormal() {
      for (iterator it = matrix_dict.begin();
           it!=matrix_dict.end(); ++it) {
        if (it->second.isSparse())
          ERROR_EXIT(256, "Impossible to execute operators with sparse matrices\n");
        it->second.checkDense()->pruneSubnormalAndCheckNormal();
      }
    }
    
    // two matrix basic math operator macros
#define MAKE_OPERATOR(NAME,FUNC)                                            \
    void NAME(const MatrixSet<T> *other) {                              \
      for (iterator it = matrix_dict.begin();                           \
           it!=matrix_dict.end(); ++it) {                               \
        Value &a = it->second;                                          \
        const Value *b = other->find(it->first);                        \
        if (a.isSparse() || b->isSparse())                              \
          ERROR_EXIT(256, "Impossible to execute operators with sparse matrices\n"); \
        if (b->empty())                                                 \
          ERROR_EXIT1(128, "Matrix with name %s not found\n",           \
                      it->first.c_str());                               \
        AprilMath::MatrixExt::Operations::FUNC(a.getDense().get(),      \
                                               b->getDense().get());    \
      }                                                                 \
    }
    // two matrix basic math operator declarations
    MAKE_OPERATOR(cmul,matCmul);
    MAKE_OPERATOR(copy,matCopy);
#undef MAKE_OPERATOR

    // AXPY
    void axpy(T alpha, const MatrixSet<T> *other) {
      for (iterator it = matrix_dict.begin(); it!=matrix_dict.end(); ++it) {
        Value &a = it->second;
        const Value *b = other->find(it->first);
        if (b->empty())
          ERROR_EXIT1(128, "Matrix with name %s not found\n",
                      it->first.c_str());
        if (a.isSparse() || b->isSparse())
          ERROR_EXIT(256, "Impossible to execute operators with sparse matrices\n");
        AprilMath::MatrixExt::Operations::matAxpy(a.getDense().get(), alpha,
                                                  b->getDense().get());
      }
    }

    // EQUALS
    void equals(const MatrixSet<T> *other, T epsilon) {
      for (iterator it = matrix_dict.begin(); it!=matrix_dict.end(); ++it) {
        const Value &a = it->second;
        const Value *b = other->find(it->first);
        if (b->empty())
          ERROR_EXIT1(128, "Matrix with name %s not found\n",
                      it->first.c_str());
        if (a.isSparse() != b->isSparse())
          ERROR_EXIT(256, "Impossible to execute operators with different matrix types\n");
        if (a.isSparse())
          AprilMath::MatrixExt::Operations::matEquals(a.getSparse().get(),
                                                      b->getSparse().get(),
                                                      epsilon);
        else
          AprilMath::MatrixExt::Operations::matEquals(a.getDense().get(),
                                                      b->getDense().get(),
                                                      epsilon);
      }
    }

    // matrix math reductions
    T norm2() {
      T result_norm2 = T();
      for (iterator it = matrix_dict.begin(); it!=matrix_dict.end(); ++it) {
        T current_norm2;
        Value &a = it->second;
        if (a.isSparse()) {
          current_norm2 = AprilMath::MatrixExt::Operations::
            matNorm2(it->second.getSparse().get());
        }
        else {
          current_norm2 = AprilMath::MatrixExt::Operations::
            matNorm2(it->second.checkDense().get());
        }
        result_norm2 = result_norm2 + current_norm2*current_norm2;
      }
      return result_norm2;
    }

    // matrix math reductions
    int size() {
      int total_size = 0;
      for (iterator it = matrix_dict.begin(); it!=matrix_dict.end(); ++it) {
        Value &a = it->second;
        if (a.isSparse())
          total_size += it->second.getSparse()->size();
        else
          total_size += it->second.checkDense()->size();
      }
      return total_size;
    }

    // dot reduction
    T dot(MatrixSet<T> *other) {
      T result = T();
      for (iterator it = matrix_dict.begin(); it!=matrix_dict.end(); ++it) {
        Value &va = it->second;
        Value *vb = other->find(it->first);
        if (vb->empty())
          ERROR_EXIT1(128, "Matrix with name %s not found\n",
                      it->first.c_str());
        if (va.isSparse() || vb->isSparse())
          ERROR_EXIT(256, "Impossible to execute operators with sparse matrices\n");
        AprilUtils::SharedPtr< Matrix<T> > a = va.checkDense(), b = vb->checkDense();
        if (!a->getIsContiguous()) a = a->clone();
        if (!b->getIsContiguous()) b = b->clone();
        int a_size = a->size();
        int b_size = b->size();
        a = a->rewrap(&a_size, 1);
        b = b->rewrap(&b_size, 1);
        result = result + AprilMath::MatrixExt::Operations::matDot(a.get(),
                                                                   b.get());
      }
      return result;
    }

  };

} // namespace Basics

#endif // MATRIXSET_H
