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

// Must to be defined here.
#include "cuda_utils.h"
#include "matrix_ext_blas.h"
#include "omp_utils.h"

using Basics::Matrix;
using Basics::SparseMatrix;

#ifdef USE_CUDA
using namespace AprilMath::CUDA;
#endif

namespace AprilMath {
  namespace MatrixExt {
    
    namespace Misc {
      //////////////////// OTHER MATH OPERATIONS ////////////////////

#ifdef USE_CUDA
      template<typename T>
      __global__ void indexVectorKernel(const T *m_ptr,
                                        const int m_stride,
                                        const int32_t *idx_ptr,
                                        const int idx_stride,
                                        T *dest_ptr,
                                        const int dest_stride,
                                        const int N) {
        for ( int i = blockIdx.x*blockDim.x + threadIdx.x;
              i < N;
              i += blockDim.x*gridDim.x ) {
          // WARNING: idx counts from 1, instead of 0
          dest_ptr[ i*dest_stride ] = m_ptr[ m_stride * (idx_ptr[i*idx_stride]-1) ];
        }
      }

      template<typename T>
      __global__ void indexedFillVectorKernel(T *m_ptr,
                                              const int m_stride,
                                              const int32_t *idx_ptr,
                                              const int idx_stride,
                                              const int N,
                                              const T val) {
        for ( int i = blockIdx.x*blockDim.x + threadIdx.x;
              i < N;
              i += blockDim.x*gridDim.x ) {
          // WARNING: idx counts from 1, instead of 0
          m_ptr[ m_stride * (idx_ptr[i*idx_stride]-1) ] = val;
        }
      }

      template<typename T>
      __global__ void indexedCopyVectorKernel(T *m_ptr,
                                              const int m_stride,
                                              const int32_t *idx_ptr,
                                              const int idx_stride,
                                              const T *other_ptr,
                                              const int other_stride,
                                              const int N) {
        for ( int i = blockIdx.x*blockDim.x + threadIdx.x;
              i < N;
              i += blockDim.x*gridDim.x ) {
          // WARNING: idx counts from 1, instead of 0
          m_ptr[ m_stride * (idx_ptr[i*idx_stride]-1) ] = other_ptr[ i*other_stride ];
        }
      }
#endif
      
      template <typename T>
      void indexVectorWrapper(Matrix<T> *m, Matrix<int32_t> *idx,
                              Matrix<T> *dest) {
        const int N = idx->size();
        const int m_offset = m->getOffset();
        const int idx_offset = idx->getOffset();
        const int dest_offset = dest->getOffset();
        const int m_stride = m->getStrideSize(0);
        const int idx_stride = idx->getStrideSize(0);
        const int dest_stride = dest->getStrideSize(0);
#ifdef USE_CUDA
        bool cuda_flag = m->getCudaFlag() || idx->getCudaFlag() || dest->getCudaFlag();
        if (cuda_flag) {
          const T *m_ptr = m->getRawDataAccess()->getGPUForRead() + m_offset;
          const int32_t *idx_ptr = idx->getRawDataAccess()->getGPUForRead() + idx_offset;
          T *dest_ptr = dest->getRawDataAccess()->getGPUForWrite() + dest_offset;
          // Number of threads on each block dimension
          int num_threads, num_blocks;
          computeBlockAndGridSizesForArray(N, num_threads, num_blocks);
          indexVectorKernel<<<num_blocks, num_threads, 0, GPUHelper::getCurrentStream()>>>
            (m_ptr, m_stride, idx_ptr, idx_stride, dest_ptr, dest_stride, N);
        }
        else {
#endif
          const T *m_ptr = m->getRawDataAccess()->getPPALForRead() + m_offset;
          const int32_t *idx_ptr = idx->getRawDataAccess()->getPPALForRead() + idx_offset;
          T *dest_ptr = dest->getRawDataAccess()->getPPALForWrite() + dest_offset;
          for(int i=0; i<N; ++i) {
            // WARNING: idx counts from 1, instead of 0
            dest_ptr[ i*dest_stride ] = m_ptr[ m_stride * (idx_ptr[idx_stride*i]-1) ];
          }
#ifdef USE_CUDA
        }
#endif
      }

      template <typename T>
      void indexedFillVectorWrapper(Matrix<T> *m, Matrix<int32_t> *idx,
                                    T val) {
        const int N = idx->size();
        const int m_offset = m->getOffset();
        const int idx_offset = idx->getOffset();
        const int m_stride = m->getStrideSize(0);
        const int idx_stride = idx->getStrideSize(0);
#ifdef USE_CUDA
        bool cuda_flag = m->getCudaFlag() || idx->getCudaFlag();
        if (cuda_flag) {
          T *m_ptr = m->getRawDataAccess()->getGPUForWrite() + m_offset;
          const int32_t *idx_ptr = idx->getRawDataAccess()->getGPUForRead() + idx_offset;
          // Number of threads on each block dimension
          int num_threads, num_blocks;
          computeBlockAndGridSizesForArray(N, num_threads, num_blocks);
          indexedFillVectorKernel<<<num_blocks, num_threads, 0, GPUHelper::getCurrentStream()>>>
            (m_ptr, m_stride, idx_ptr, idx_stride, N, val);
        }
        else {
#endif
          T *m_ptr = m->getRawDataAccess()->getPPALForWrite() + m_offset;
          const int32_t *idx_ptr = idx->getRawDataAccess()->getPPALForRead() + idx_offset;
          for(int i=0; i<N; ++i) {
            // WARNING: idx counts from 1, instead of 0
            m_ptr[ m_stride * (idx_ptr[idx_stride*i]-1) ] = val;
          }
#ifdef USE_CUDA
        }
#endif
      }

      template <typename T>
      void indexedCopyVectorWrapper(Matrix<T> *m, Matrix<int32_t> *idx,
                                    Matrix<T> *other) {
        const int N = idx->size();
        const int m_offset = m->getOffset();
        const int idx_offset = idx->getOffset();
        const int other_offset = other->getOffset();
        const int m_stride = m->getStrideSize(0);
        const int idx_stride = idx->getStrideSize(0);
        const int other_stride = other->getStrideSize(0);
#ifdef USE_CUDA
        bool cuda_flag = m->getCudaFlag() || idx->getCudaFlag() || other->getCudaFlag();
        if (cuda_flag) {
          T *m_ptr = m->getRawDataAccess()->getGPUForWrite() + m_offset;
          const int32_t *idx_ptr = idx->getRawDataAccess()->getGPUForRead() + idx_offset;
          const T *other_ptr = other->getRawDataAccess()->getGPUForRead() + other_offset;
          // Number of threads on each block dimension
          int num_threads, num_blocks;
          computeBlockAndGridSizesForArray(N, num_threads, num_blocks);
          indexedCopyVectorKernel<<<num_blocks, num_threads, 0, GPUHelper::getCurrentStream()>>>
            (m_ptr, m_stride, idx_ptr, idx_stride, other_ptr, other_stride, N);
        }
        else {
#endif
          T *m_ptr = m->getRawDataAccess()->getPPALForWrite() + m_offset;
          const int32_t *idx_ptr = idx->getRawDataAccess()->getPPALForRead() + idx_offset;
          const T *other_ptr = other->getRawDataAccess()->getPPALForRead() + other_offset;
          for(int i=0; i<N; ++i) {
            // WARNING: idx counts from 1, instead of 0
            m_ptr[ m_stride * (idx_ptr[idx_stride*i]-1) ] = other_ptr[ i*other_stride ];
          }
#ifdef USE_CUDA
        }
#endif
      }
      
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
      Matrix<T> *matIndex(const Matrix<T> *m, int dim,
                          const Matrix<int32_t> *idx) {
        AprilUtils::SharedPtr< Matrix<int32_t> > sq_idx = idx->constSqueeze();
        if (sq_idx->getNumDim() != 1) {
          ERROR_EXIT(128, "Needs a rank 1 tensor as second argument (index)\n");
        }
        if (dim < 0 || dim >= m->getNumDim()) {
          ERROR_EXIT(128, "Dimension argument out-of-bounds");
        }
        const int D = m->getNumDim();
        // copy m->dims into a new array
        AprilUtils::UniquePtr<int[]> dims = new int[D];
        for (int i=0; i<D; ++i) dims[i] = m->getDimSize(i);
        // take the given dimension size and change it to fit the result matrix
        int dim_limit = dims[dim]; dims[dim] = idx->size();
        // allocate memory for the resulting matrix
        AprilUtils::SharedPtr< Matrix<T> > result;
        result = new Matrix<T>(m->getNumDim(), dims.get());
#ifdef USE_CUDA
        result->setUseCuda(m->getCudaFlag());
#endif
        if (m->size() == dim_limit) {
          // vector version, ad-hoc implementation
          AprilUtils::SharedPtr< Matrix<T> > sq_m = m->constSqueeze();
          AprilUtils::SharedPtr< Matrix<T> > sq_r = result->squeeze();
          indexVectorWrapper(sq_m.get(), sq_idx.get(), sq_r.get());
        }
        else {
          // matrix version, ad-hoc general implementation using select
          AprilUtils::SharedPtr< Matrix<T> > m_slice, r_slice;
          typename Matrix<int32_t>::const_iterator it = idx->begin();
          // traverse all given indices copying submatrices
          for (int i=0; i<idx->size(); ++i, ++it) {
            april_assert(it != idx->end());
            april_assert(i < result->getDimSize(dim));
            // WARNING: idx counts from 1, instead of 0
            const int p = *it - 1;
            april_assert(p < m->getDimSize(dim));
            // take submatrices
            m_slice = m->select(dim, p, m_slice.get());
            r_slice = result->select(dim, i, r_slice.get());
            // copy them
            AprilMath::MatrixExt::BLAS::matCopy(r_slice.get(), m_slice.get());
          }
        }
        return result.weakRelease();
      }

      template <typename T>
      Matrix<T> *matIndexedFill(Matrix<T> *m, int dim,
                                const Matrix<int32_t> *idx, T val) {
        AprilUtils::SharedPtr< Matrix<T> > m_ref = m; // IncRef m
        AprilUtils::SharedPtr< Matrix<int32_t> > sq_idx = idx->constSqueeze();
        if (sq_idx->getNumDim() != 1) {
          ERROR_EXIT(128, "Needs a rank 1 tensor as second argument (index)\n");
        }
        if (dim < 0 || dim >= m->getNumDim()) {
          ERROR_EXIT(128, "Dimension argument out-of-bounds");
        }
        const int dim_limit = m->getDimSize(dim);
        if (m->size() == dim_limit) {
          // vector version, ad-hoc implementation
          AprilUtils::SharedPtr< Matrix<T> > sq_m = m->squeeze();
          indexedFillVectorWrapper(sq_m.get(), sq_idx.get(), val);
        }
        else {
          // matrix version, ad-hoc general implementation using select
          AprilUtils::SharedPtr< Matrix<T> > m_slice;
          typename Matrix<int32_t>::const_iterator it = idx->begin();
          // traverse all given indices copying submatrices
          for (int i=0; i<idx->size(); ++i, ++it) {
            april_assert(it != idx->end());
            // WARNING: idx counts from 1, instead of 0
            const int p = *it - 1;
            april_assert(p < m->getDimSize(dim));
            // take submatrix
            m_slice = m->select(dim, p, m_slice.get());
            // fill it with val constant
            AprilMath::MatrixExt::Initializers::matFill(m_slice.get(), val);
          }
        }
        return m;
      }

      template <typename T>
      Matrix<T> *matIndexedCopy(Matrix<T> *m, int dim,
                                const Matrix<int32_t> *idx,
                                const Matrix<T> *other) {
        AprilUtils::SharedPtr< Matrix<int32_t> > sq_idx = idx->constSqueeze();
        if (sq_idx->getNumDim() != 1) {
          ERROR_EXIT(128, "Needs a rank 1 tensor as second argument (index)\n");
        }
        if (dim < 0 || dim >= m->getNumDim()) {
          ERROR_EXIT(128, "Dimension argument out-of-bounds\n");
        }
        const int D = m->getNumDim();
        // copy m->dims into a new array
        AprilUtils::UniquePtr<int[]> dims = new int[D];
        for (int i=0; i<D; ++i) dims[i] = m->getDimSize(i);
        const int dim_limit = dims[dim];
        dims[dim] = idx->size();
        if (!other->sameDim(dims.get(), D)) {
          ERROR_EXIT(128, "Incompatible matrix sizes\n");
        }
        if (m->size() == dim_limit) {
          // vector version, ad-hoc implementation
          AprilUtils::SharedPtr< Matrix<T> > sq_m = m->squeeze();
          AprilUtils::SharedPtr< Matrix<T> > sq_o = other->constSqueeze();
          indexedCopyVectorWrapper(sq_m.get(), sq_idx.get(), sq_o.get());
        }
        else {
          // matrix version, ad-hoc general implementation using select          
          AprilUtils::SharedPtr< Matrix<T> > m_slice, o_slice;
          typename Matrix<int32_t>::const_iterator it = idx->begin();
          // traverse all given indices copying submatrices
          for (int i=0; i<idx->size(); ++i, ++it) {
            april_assert(it != idx->end());
            april_assert(i < other->getDimSize(dim));
            // WARNING: idx counts from 1, instead of 0
            const int p = *it - 1;
            april_assert(p < m->getDimSize(dim));
            // take submatrices
            m_slice = m->select(dim, p, m_slice.get());
            o_slice = other->select(dim, i, o_slice.get());
            // fill it with val constant
            AprilMath::MatrixExt::BLAS::matCopy(m_slice.get(), o_slice.get());
          }
        }
        return m;
      }
      
      template <typename T>
      Matrix<T> *matIndex(const Matrix<T> *m, int dim,
                          const Matrix<bool> *mask) {
        AprilUtils::SharedPtr< Matrix<int32_t> > idx;
        idx = matNonZeroIndices(mask);
        return matIndex(m, dim, idx.get());
      }

      template <typename T>
      Matrix<T> *matIndexedFill(Matrix<T> *m, int dim,
                                const Matrix<bool> *mask,
                                T val) {
        AprilUtils::SharedPtr< Matrix<int32_t> > idx;
        idx = matNonZeroIndices(mask);
        return matIndexedFill(m, dim, idx.get(), val);
      }

      template <typename T>
      Matrix<T> *matIndexedCopy(Matrix<T> *m, int dim,
                                const Matrix<bool> *mask,
                                const Matrix<T> *other) {
        AprilUtils::SharedPtr< Matrix<int32_t> > idx;
        idx = matNonZeroIndices(mask);
        return matIndexedCopy(m, dim, idx.get(), other);
      }
      
      template <typename T>
      Matrix<T> *matAddition(const Matrix<T> *a,
                             const Matrix<T> *b,
                             Matrix<T> *c) {
        if (c == 0) c = a->clone();
        return AprilMath::MatrixExt::BLAS::
          matAxpy(c, AprilMath::Limits<T>::one(), b);
      }

      template <typename T>
      Matrix<T> *matSubstraction(const Matrix<T> *a,
                                 const Matrix<T> *b,
                                 Matrix<T> *c) {
        if (c == 0) c = a->clone();
        return AprilMath::MatrixExt::BLAS::
          matAxpy(c, -AprilMath::Limits<T>::one(), b);
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
                     AprilMath::Limits<T>::one(), a, b);
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
                      CblasNoTrans,
                      AprilMath::Limits<T>::one(),
                      a, b,
                      AprilMath::Limits<T>::zero());
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
                    AprilMath::Limits<T>::one(),
                    a, b,
                    AprilMath::Limits<T>::zero());
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
        if (sq_input->getNumDim() != 1) {
          ERROR_EXIT(128, "Needs a rank 1 matrix\n");
        }
        int non_zeros = sq_input->size() - matCount(sq_input.get(),
                                                    AprilMath::Limits<T>::zero());
        if (dest != 0) {
          if (dest->size() != non_zeros) {
            ERROR_EXIT(256, "Incompatible matrix sizes\n");
          }
        }
        else {
          dest = new Basics::Matrix<int32_t>(1, &non_zeros);
        }
        if (non_zeros == 0) return dest;
        int k=0;
        Basics::Matrix<int32_t>::iterator dest_it = dest->begin();
        for (typename Basics::Matrix<T>::const_iterator it = sq_input->begin();
             it != sq_input->end(); ++it) {
          ++k; // first index is 1, to be compatible with Lua
          if (*it != AprilMath::Limits<T>::zero()) {
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

      template Matrix<float> *matIndex(const Matrix<float> *, int, const Matrix<int32_t> *);

      template Matrix<float> *matIndex(const Matrix<float> *, int, const Matrix<bool> *);
      
      template Matrix<float> *matIndexedFill(Matrix<float> *, int,
                                             const Matrix<int32_t> *, float);

      template Matrix<float> *matIndexedFill(Matrix<float> *, int,
                                             const Matrix<bool> *, float);

      template Matrix<float> *matIndexedCopy(Matrix<float> *, int,
                                             const Matrix<int32_t> *,
                                             const Matrix<float> *);

      template Matrix<float> *matIndexedCopy(Matrix<float> *, int,
                                             const Matrix<bool> *,
                                             const Matrix<float> *);
      
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
      template Matrix<float> *matConvertTo(const Matrix<ComplexF> *,
                                           Matrix<float> *);
      template Matrix<int32_t> *matNonZeroIndices(const Matrix<float> *input,
                                                  Basics::Matrix<int32_t> *dest);


      template Matrix<int32_t> *matOrder(const Matrix<double> *,
                                         Matrix<int32_t> *);

      template Matrix<int32_t> *matOrderRank(const Matrix<double> *,
                                             Matrix<int32_t> *);

      template Matrix<double> *matIndex(const Matrix<double> *, int, const Matrix<int32_t> *);
      
      template Matrix<double> *matIndex(const Matrix<double> *, int, const Matrix<bool> *);
            
      template Matrix<double> *matIndexedFill(Matrix<double> *, int,
                                              const Matrix<int32_t> *, double);

      template Matrix<double> *matIndexedFill(Matrix<double> *, int,
                                              const Matrix<bool> *, double);

      template Matrix<double> *matIndexedCopy(Matrix<double> *, int,
                                              const Matrix<int32_t> *,
                                              const Matrix<double> *);

      template Matrix<double> *matIndexedCopy(Matrix<double> *, int,
                                              const Matrix<bool> *,
                                              const Matrix<double> *);
      
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
      template Matrix<double> *matConvertTo(const Matrix<ComplexF> *,
                                            Matrix<double> *);
      template Matrix<int32_t> *matNonZeroIndices(const Matrix<double> *input,
                                                  Basics::Matrix<int32_t> *dest);
      


      template Matrix<ComplexF> *matIndex(const Matrix<ComplexF> *, int, const Matrix<int32_t> *);

      template Matrix<ComplexF> *matIndex(const Matrix<ComplexF> *, int, const Matrix<bool> *);
      
      template Matrix<ComplexF> *matIndexedFill(Matrix<ComplexF> *, int,
                                                const Matrix<int32_t> *, ComplexF);

      template Matrix<ComplexF> *matIndexedFill(Matrix<ComplexF> *, int,
                                                const Matrix<bool> *, ComplexF);

      template Matrix<ComplexF> *matIndexedCopy(Matrix<ComplexF> *, int,
                                                const Matrix<int32_t> *,
                                                const Matrix<ComplexF> *);

      template Matrix<ComplexF> *matIndexedCopy(Matrix<ComplexF> *, int,
                                                const Matrix<bool> *,
                                                const Matrix<ComplexF> *);
      
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


      template Matrix<char> *matIndex(const Matrix<char> *, int, const Matrix<int32_t> *);

      template Matrix<char> *matIndex(const Matrix<char> *, int, const Matrix<bool> *);
      
      template Matrix<char> *matIndexedFill(Matrix<char> *, int,
                                            const Matrix<int32_t> *, char);

      template Matrix<char> *matIndexedFill(Matrix<char> *, int,
                                            const Matrix<bool> *, char);

      template Matrix<char> *matIndexedCopy(Matrix<char> *, int,
                                            const Matrix<int32_t> *,
                                            const Matrix<char> *);

      template Matrix<char> *matIndexedCopy(Matrix<char> *, int,
                                            const Matrix<bool> *,
                                            const Matrix<char> *);
      
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
      template Matrix<char> *matConvertTo(const Matrix<ComplexF> *,
                                          Matrix<char> *);
      template Matrix<int32_t> *matNonZeroIndices(const Matrix<char> *input,
                                                  Basics::Matrix<int32_t> *dest);


      template Matrix<int32_t> *matIndex(const Matrix<int32_t> *, int, const Matrix<int32_t> *);

      template Matrix<int32_t> *matIndex(const Matrix<int32_t> *, int, const Matrix<bool> *);

      template Matrix<int32_t> *matIndexedFill(Matrix<int32_t> *, int,
                                               const Matrix<int32_t> *, int32_t);

      template Matrix<int32_t> *matIndexedFill(Matrix<int32_t> *, int,
                                               const Matrix<bool> *, int32_t);

      template Matrix<int32_t> *matIndexedCopy(Matrix<int32_t> *, int,
                                               const Matrix<int32_t> *,
                                               const Matrix<int32_t> *);

      template Matrix<int32_t> *matIndexedCopy(Matrix<int32_t> *, int,
                                               const Matrix<bool> *,
                                               const Matrix<int32_t> *);
      
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
      template Matrix<int32_t> *matConvertTo(const Matrix<ComplexF> *,
                                             Matrix<int32_t> *);
      template Matrix<int32_t> *matNonZeroIndices(const Matrix<int32_t> *input,
                                                  Basics::Matrix<int32_t> *dest);
      template Matrix<int32_t> *matAddition(const Matrix<int32_t> *,
                                            const Matrix<int32_t> *,
                                            Matrix<int32_t> *);
      template Matrix<int32_t> *matSubstraction(const Matrix<int32_t> *,
                                                const Matrix<int32_t> *,
                                                Matrix<int32_t> *);


      template Matrix<bool> *matIndex(const Matrix<bool> *, int, const Matrix<int32_t> *);

      template Matrix<bool> *matIndex(const Matrix<bool> *, int, const Matrix<bool> *);

      template Matrix<bool> *matIndexedFill(Matrix<bool> *, int,
                                            const Matrix<int32_t> *, bool);

      template Matrix<bool> *matIndexedFill(Matrix<bool> *, int,
                                            const Matrix<bool> *, bool);

      template Matrix<bool> *matIndexedCopy(Matrix<bool> *, int,
                                            const Matrix<int32_t> *,
                                            const Matrix<bool> *);
      
      template Matrix<bool> *matIndexedCopy(Matrix<bool> *, int,
                                            const Matrix<bool> *,
                                            const Matrix<bool> *);
      
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
      template Matrix<bool> *matConvertTo(const Matrix<ComplexF> *,
                                          Matrix<bool> *);
      template Matrix<int32_t> *matNonZeroIndices(const Matrix<bool> *input,
                                                  Basics::Matrix<int32_t> *dest);

      
      template Matrix<ComplexF> *matConvertTo(const Matrix<float> *,
                                              Matrix<ComplexF> *);
      template Matrix<ComplexF> *matConvertTo(const Matrix<double> *,
                                              Matrix<ComplexF> *);
      template Matrix<ComplexF> *matConvertTo(const Matrix<int32_t> *,
                                              Matrix<ComplexF> *);
      template Matrix<ComplexF> *matConvertTo(const Matrix<char> *,
                                              Matrix<ComplexF> *);
      template Matrix<ComplexF> *matConvertTo(const Matrix<bool> *,
                                              Matrix<ComplexF> *);
      template Matrix<ComplexF> *matConvertTo(const Matrix<ComplexF> *,
                                              Matrix<ComplexF> *);
      
    } // namespace Misc
    
  } // namespace MatrixExt
} // namespace AprilMath
