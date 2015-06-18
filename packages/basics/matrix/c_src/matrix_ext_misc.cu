/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Salvador España-Boquera, Francisco Zamora-Martinez
 * Copyright 2012, Salvador España-Boquera
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
#include "cmath_overloads.h"
#include "mathcore.h"
#include "matrix.h"
#include "maxmin.h"
#include "realfftwithhamming.h"
#include "smart_ptr.h"
#include "sparse_matrix.h"

// Must be defined in this order.
#include "matrix_ext_misc.h"

// Must to be defined here.
#include "map_matrix.h"
#include "map_sparse_matrix.h"

// Must to be defined here.
#include "reduce_matrix.h"
#include "reduce_sparse_matrix.h"

using Basics::Matrix;
using Basics::SparseMatrix;

namespace AprilMath {
  namespace MatrixExt {
    
    namespace Misc {
      //////////////////// OTHER MATH OPERATIONS ////////////////////

      /// For the implementation of matOrder() function.
      template <typename T>
      struct MatOrderCompare {
        typename Matrix<T>::const_random_access_iterator data;
        MatOrderCompare(Matrix<T> *m) : data(m) {}
        bool operator()(const int32_t &a, const int32_t &b) const {
          return data(a-1) < data(b-1);
        }
      };
      
      template <typename T>
      Matrix<int32_t> *matOrder(const Matrix<T> *m, Matrix<int32_t> *dest) {
        AprilUtils::SharedPtr< Matrix<T> > squeezed(m->constSqueeze());
        if (squeezed->getNumDim() > 1) {
          ERROR_EXIT(128, "Needs a rank 1 matrix object\n");
        }
        if (dest == 0) {
          dest = new Matrix<int32_t>(1, squeezed->size());
        }
        else {
          if (squeezed->size() != dest->size()) {
            ERROR_EXIT(128, "Incorrect destination size\n");
          }
          if (!squeezed->getIsContiguous()) {
            ERROR_EXIT(128, "Destination matrix should be contiguous\n");
          }
        }
        int i=1; // WARNING: we start counting at 1, instead of 0
        for (Matrix<int32_t>::iterator it = dest->begin();
             it!=dest->end(); ++it) {
          *it = i++;
        }
        int32_t *ptr = dest->getRawDataAccess()->getPPALForReadAndWrite();
        MatOrderCompare<T> cmp(squeezed.get());
        AprilUtils::Sort(ptr, 0, dest->size()-1, cmp);
        return dest;
      }

      template <typename T>
      Matrix<int32_t> *matOrderRank(const Matrix<T> *m, Matrix<int32_t> *dest) {
        AprilUtils::SharedPtr< Matrix<int32_t> > aux = matOrder(m);
        if (dest == 0) {
          dest = new Matrix<int32_t>(1, aux->size());
        }
        else {
          if (aux->size() != dest->size()) {
            ERROR_EXIT(128, "Incorrect destination size\n");
          }
        }

        Matrix<int32_t>::random_access_iterator dst_it(dest);
        int i=1; // WARNING: we start counting at 1, instead of 0
        for (Matrix<int32_t>::const_iterator src_it = aux->begin();
             src_it != aux->end(); ++src_it) {
          dst_it( *src_it - 1 ) = i++;
        }
        return dest;
      }
      
      template <typename T>
      Matrix<T> *matAddition(const Matrix<T> *a,
                             const Matrix<T> *b,
                             Matrix<T> *c) {
        if (c == 0) c = a->clone();
        return AprilMath::MatrixExt::BLAS::matAxpy(c, T(1.0f), b);
      }

      template <typename T>
      Matrix<T> *matSubstraction(const Matrix<T> *a,
                                 const Matrix<T> *b,
                                 Matrix<T> *c) {
        if (c == 0) c = a->clone();
        return AprilMath::MatrixExt::BLAS::matAxpy(c, T(-1.0f), b);
      }
    
      template <typename T>
      Matrix<T> *matMultiply(const Matrix<T> *a,
                             const Matrix<T> *b,
                             Matrix<T> *c) {
        if (b->isVector()) {
          if (a->isColVector()) {
            // OUTER product
            int dim[2] = {a->size(),b->size()};
            if (c == 0) {
              c = new Matrix<T>(2, dim);
#ifdef USE_CUDA
              c->setUseCuda(a->getCudaFlag() || b->getCudaFlag());
#endif
            }
            else if (!c->sameDim(dim, 2)) {
              ERROR_EXIT2(128, "Incorrect matrix sizes, expected %dx%d\n",
                          dim[0], dim[1]);
            }
            AprilMath::MatrixExt::BLAS::
              matGer(AprilMath::MatrixExt::Initializers::matZeros(c),
                     T(1.0f), a, b);
          }
          else if (!a->isVector()) {
            // Matrix-Vector product
            int dim[2] = {a->getDimSize(0),1};
            if (c == 0) {
              c = new Matrix<T>(b->getNumDim(), dim);
#ifdef USE_CUDA
              c->setUseCuda(a->getCudaFlag() || b->getCudaFlag());
#endif
            }
            else if (!c->sameDim(dim, b->getNumDim())) {
              ERROR_EXIT2(128, "Incorrect matrix sizes, expected %dx%d\n",
                          dim[0], dim[1]);
            }
            AprilMath::MatrixExt::BLAS::
              matGemv(AprilMath::MatrixExt::Initializers::matZeros(c),
                      CblasNoTrans, T(1.0f), a, b, T());
          }
          else {
            // DOT product
            int dim[2] = {1,1};
            if (c == 0) {
              c = new Matrix<T>(a->getNumDim(), dim);
#ifdef USE_CUDA
              c->setUseCuda(a->getCudaFlag() || b->getCudaFlag());
#endif
            }
            else if (!c->sameDim(dim, a->getNumDim())) {
              ERROR_EXIT2(128, "Incorrect matrix sizes, expected %dx%d\n",
                          dim[0], dim[1]);
            }
            c->getRawDataAccess()->putValue( c->getOffset(),
                                             AprilMath::MatrixExt::BLAS::
                                             matDot(a, b) );
          }
        }
        else if (a->getNumDim() == 2 && b->getNumDim() == 2 &&
                 a->getDimSize(1) == b->getDimSize(0)) {
          // Matrix-Matrix product
          int dim[2] = {a->getDimSize(0), b->getDimSize(1)};
          if (c == 0) {
            c = new Matrix<T>(2,dim);
#ifdef USE_CUDA
              c->setUseCuda(a->getCudaFlag() || b->getCudaFlag());
#endif
          }
          else if (!c->sameDim(dim,2)) {
            ERROR_EXIT2(128, "Incorrect matrix sizes, expected %dx%d\n",
                        dim[0], dim[1]);
          }
          AprilMath::MatrixExt::BLAS::
            matGemm(AprilMath::MatrixExt::Initializers::matZeros(c),
                    CblasNoTrans, CblasNoTrans,
                    T(1.0f), a, b, T());
        }
        else {
          ERROR_EXIT(128, "Incompatible matrix sizes\n");
        }
        return c;
      }
      
      
      Basics::Matrix<float> *matRealFFTwithHamming(Basics::Matrix<float> *obj,
						   int wsize,
						   int wadvance,
						   Basics::Matrix<float> *dest) {
	const int N = obj->getNumDim();
	if (N != 1) ERROR_EXIT(128, "Only valid for numDim=1\n");
	if (wsize > obj->size() || wadvance > obj->size()) {
	  ERROR_EXIT(128, "Incompatible wsize or wadvance value\n");
	}
	AprilMath::RealFFTwithHamming real_fft(wsize);
	const int M = real_fft.getOutputSize();
	AprilUtils::UniquePtr<int []> dest_size(new int[N+1]);
	dest_size[0] = (obj->getDimSize(0) - wsize)/wadvance + 1;
	dest_size[1] = M;
	if (dest != 0) {
	  if (!dest->sameDim(dest_size.get(), N+1)) {
	    ERROR_EXIT(128, "Incompatible dest matrix\n");
	  }
	}
	else {
	  dest = new Matrix<float>(N+1, dest_size.get());
#ifdef USE_CUDA
	  dest->setUseCuda(obj->getCudaFlag());
#endif
	}
	AprilUtils::UniquePtr<double []> input(new double[wsize]);
	AprilUtils::UniquePtr<double []> output(new double[M]);
	//
	Basics::Matrix<float>::sliding_window swindow(obj,
                                                      &wsize,
                                                      0, // offset
                                                      &wadvance);
	AprilUtils::SharedPtr< Matrix<float> > input_slice;
	AprilUtils::SharedPtr< Matrix<float> > output_slice;
	int i=0, j;
	while(!swindow.isEnd()) {
	  april_assert(i < dest_size[0]);
	  input_slice = swindow.getMatrix(input_slice.get());
	  output_slice = dest->select(0, i);
	  j=0;
	  for (Basics::Matrix<float>::const_iterator it(input_slice->begin());
	       it != input_slice->end(); ++it, ++j) {
	    april_assert(j<wsize);
	    input[j] = static_cast<double>(*it);
	  }
	  april_assert(j==wsize);
	  real_fft(input.get(), output.get());
	  j=0;
	  for (Basics::Matrix<float>::iterator it(output_slice->begin());
	       it != output_slice->end(); ++it, ++j) {
	    april_assert(j<M);
	    *it = static_cast<float>(output[j]);
	  }
	  april_assert(j==M);
	  ++i;
	  swindow.next();
	}
	april_assert(i == dest_size[0]);
	return dest;
      }

      template <typename T, typename O>
      Basics::Matrix<O> *matConvertTo(const Basics::Matrix<T> *input,
                                      Basics::Matrix<O> *dest) {
        if (dest != 0) {
          if (!dest->sameDim(input)) {
            ERROR_EXIT(256, "Incompatible matrix sizes\n");
          }
        }
        else {
          dest = new Basics::Matrix<O>(input->getNumDim(), input->getDimPtr());
        }
        return MatrixScalarMap1<T,O>(input, AprilMath::Functors::m_cast<T,O>(),
                                     dest);
      }

      // TODO: Use this function in matrixBool count_zeros, count_ones, ...
      template <typename T>
      int matCount(const Basics::Matrix<T> *input, T value) {
        return MatrixScalarReduce1(input,
                                   AprilMath::make_r_map1<T,int>
                                   (AprilMath::m_curried_eq<T>(value),
                                    AprilMath::Functors::r_add<bool,int>()),
                                   AprilMath::Functors::r_add<int,int>(),
                                   0);
      }
      
      template <typename T>
      Basics::Matrix<int32_t> *matNonZeroIndices(const Basics::Matrix<T> *input,
                                                 Basics::Matrix<int32_t> *dest) {
        AprilUtils::SharedPtr<Matrix<T> > sq_input = input->constSqueeze();
        if (input->getNumDim() != 1) {
          ERROR_EXIT(128, "Needs a rank 1 matrix\n");
        }
        int non_zeros = input->size() - matCount(input, T(0.0));
        if (non_zeros == 0) return 0;
        if (dest != 0) {
          if (dest->size() != non_zeros) {
            ERROR_EXIT(256, "Incompatible matrix sizes\n");
          }
        }
        else {
          dest = new Basics::Matrix<int32_t>(1, &non_zeros);
        }
        int k=0;
        Basics::Matrix<int32_t>::iterator dest_it = dest->begin();
        for (typename Basics::Matrix<T>::const_iterator it = input->begin();
             it != input->end(); ++it) {
          ++k; // first index is 1, to be compatible with Lua
          if (*it != T(0.0)) {
            april_assert(dest_it != dest->end());
            *dest_it = k;
            ++dest_it;
          }
        }
        april_assert(dest_it == dest->end());
        return dest;
      }
      

      template Matrix<int32_t> *matOrder(const Matrix<float> *,
                                         Matrix<int32_t> *);

      template Matrix<int32_t> *matOrderRank(const Matrix<float> *,
                                             Matrix<int32_t> *);
      
      template Matrix<float> *matAddition(const Matrix<float> *,
                                          const Matrix<float> *,
                                          Matrix<float> *);

      template Matrix<float> *matSubstraction(const Matrix<float> *,
                                              const Matrix<float> *,
                                              Matrix<float> *);
      template Matrix<float> *matMultiply(const Matrix<float> *,
                                          const Matrix<float> *,
                                          Matrix<float> *);
      template Matrix<float> *matConvertTo(const Matrix<float> *,
                                           Matrix<float> *);
      template Matrix<float> *matConvertTo(const Matrix<bool> *,
                                           Matrix<float> *);
      template Matrix<float> *matConvertTo(const Matrix<double> *,
                                           Matrix<float> *);
      template Matrix<float> *matConvertTo(const Matrix<char> *,
                                           Matrix<float> *);
      template Matrix<float> *matConvertTo(const Matrix<int32_t> *,
                                           Matrix<float> *);
      template Matrix<int32_t> *matNonZeroIndices(const Matrix<float> *input,
                                                  Basics::Matrix<int32_t> *dest);


      template Matrix<int32_t> *matOrder(const Matrix<double> *,
                                         Matrix<int32_t> *);

      template Matrix<int32_t> *matOrderRank(const Matrix<double> *,
                                             Matrix<int32_t> *);
      
      template Matrix<double> *matAddition(const Matrix<double> *,
                                          const Matrix<double> *,
                                          Matrix<double> *);

      template Matrix<double> *matSubstraction(const Matrix<double> *,
                                              const Matrix<double> *,
                                              Matrix<double> *);
      template Matrix<double> *matMultiply(const Matrix<double> *,
                                          const Matrix<double> *,
                                          Matrix<double> *);
      template Matrix<double> *matConvertTo(const Matrix<double> *,
                                            Matrix<double> *);
      template Matrix<double> *matConvertTo(const Matrix<bool> *,
                                            Matrix<double> *);
      template Matrix<double> *matConvertTo(const Matrix<float> *,
                                            Matrix<double> *);
      template Matrix<double> *matConvertTo(const Matrix<char> *,
                                            Matrix<double> *);
      template Matrix<double> *matConvertTo(const Matrix<int32_t> *,
                                            Matrix<double> *);
      template Matrix<int32_t> *matNonZeroIndices(const Matrix<double> *input,
                                                  Basics::Matrix<int32_t> *dest);
      

      template Matrix<ComplexF> *matAddition(const Matrix<ComplexF> *,
                                          const Matrix<ComplexF> *,
                                          Matrix<ComplexF> *);

      template Matrix<ComplexF> *matSubstraction(const Matrix<ComplexF> *,
                                                 const Matrix<ComplexF> *,
                                                 Matrix<ComplexF> *);
      template Matrix<ComplexF> *matMultiply(const Matrix<ComplexF> *,
                                             const Matrix<ComplexF> *,
                                             Matrix<ComplexF> *);
      template Matrix<int32_t> *matNonZeroIndices(const Matrix<ComplexF> *input,
                                                  Basics::Matrix<int32_t> *dest);

      
      template Matrix<char> *matConvertTo(const Matrix<bool> *,
                                          Matrix<char> *);
      template Matrix<char> *matConvertTo(const Matrix<float> *,
                                          Matrix<char> *);
      template Matrix<char> *matConvertTo(const Matrix<double> *,
                                          Matrix<char> *);
      template Matrix<char> *matConvertTo(const Matrix<int32_t> *,
                                          Matrix<char> *);
      template Matrix<char> *matConvertTo(const Matrix<char> *,
                                          Matrix<char> *);
      template Matrix<int32_t> *matNonZeroIndices(const Matrix<char> *input,
                                                  Basics::Matrix<int32_t> *dest);

      
      template Matrix<int32_t> *matOrder(const Matrix<int32_t> *,
                                         Matrix<int32_t> *);
      template Matrix<int32_t> *matOrderRank(const Matrix<int32_t> *,
                                             Matrix<int32_t> *);
      template Matrix<int32_t> *matConvertTo(const Matrix<bool> *,
                                             Matrix<int32_t> *);
      template Matrix<int32_t> *matConvertTo(const Matrix<float> *,
                                             Matrix<int32_t> *);
      template Matrix<int32_t> *matConvertTo(const Matrix<double> *,
                                             Matrix<int32_t> *);
      template Matrix<int32_t> *matConvertTo(const Matrix<char> *,
                                             Matrix<int32_t> *);
      template Matrix<int32_t> *matConvertTo(const Matrix<int32_t> *,
                                             Matrix<int32_t> *);
      template Matrix<int32_t> *matNonZeroIndices(const Matrix<int32_t> *input,
                                                  Basics::Matrix<int32_t> *dest);

      
      template Matrix<bool> *matConvertTo(const Matrix<float> *,
                                          Matrix<bool> *);
      template Matrix<bool> *matConvertTo(const Matrix<double> *,
                                          Matrix<bool> *);
      template Matrix<bool> *matConvertTo(const Matrix<int32_t> *,
                                          Matrix<bool> *);
      template Matrix<bool> *matConvertTo(const Matrix<char> *,
                                          Matrix<bool> *);
      template Matrix<bool> *matConvertTo(const Matrix<bool> *,
                                          Matrix<bool> *);
      template Matrix<int32_t> *matNonZeroIndices(const Matrix<bool> *input,
                                                  Basics::Matrix<int32_t> *dest);
      
    } // namespace Misc
    
  } // namespace MatrixExt
} // namespace AprilMath
