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
#include "matrix_ext_reductions.h"

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
    
    namespace Reductions {
      
      /////////////////// MAX MIN REDUCTIONS ///////////////////

      // Min and max over given dimension, be careful, argmin and argmax matrices
      // contains the min/max index at the given dimension, but starting in 1 (not
      // in 0)

      template <typename T>
      Matrix<T> *matMin(const Matrix<T> *obj,
                        int dim,
                        Matrix<T> *dest,
                        Matrix<int32_t> *argmin) {
        if (argmin == 0) {
          return MatrixScalarReduce1OverDimension(obj, dim,
                                                  AprilMath::Functors::r_min<T>(),
                                                  AprilMath::Functors::r_min<T>(),
                                                  Limits<T>::infinity(), dest);
        }
        else {
          return MatrixScalarReduceMinMaxOverDimension(obj, dim,
                                                       AprilMath::Functors::r_min2<T>(),
                                                       Limits<T>::infinity(), argmin, dest);
        }
      }

      // TODO: use a wrapper for GPU/CPU
      template <typename T>
      Matrix<T> *matMin(const SparseMatrix<T> *obj, int dim,
                        Matrix<T> *dest,
                        Matrix<int32_t> *argmin) {
        if (dim != 0 && dim != 1) {
          ERROR_EXIT1(128, "Incorrect given dimension %d\n", dim);
        }
        int ndim = (dim==0)?(1):(0);
        if (dest) {
          if (dest->getDimSize(dim) != 1 ||
              dest->getDimSize(ndim) != obj->getDimSize(ndim)) {
            ERROR_EXIT(128, "Incorrect matrix sizes\n");
          }
        }
        else {
          int result_dims[2] = { obj->getDimSize(0), obj->getDimSize(1) };
          result_dims[dim] = 1;
          dest = new Matrix<T>(1, result_dims);
        }
        if (argmin) {
          if (argmin->getDimSize(dim) != 1 ||
              argmin->getDimSize(ndim) != obj->getDimSize(ndim)) {
            ERROR_EXIT(128, "Incorrect matrix sizes\n");
          }
          AprilMath::MatrixExt::Initializers::matZeros(argmin);
        }
        AprilMath::MatrixExt::Initializers::matZeros(dest);
        typename Matrix<T>::random_access_iterator dest_it(dest);
        int aux_dims[2] = { 0, 0 };
        if (argmin == 0) {
          for (typename SparseMatrix<T>::const_iterator it(obj->begin());
               it!=obj->end(); ++it) {
            int coords[2];
            it.getCoords(coords[0],coords[1]);
            aux_dims[ndim] = coords[ndim];
            dest_it(aux_dims[0],aux_dims[1]) =
              AprilUtils::min(dest_it(aux_dims[0],aux_dims[1]),(*it));
          }
        }
        else {
          typename Matrix<int32_t>::random_access_iterator argmin_it(argmin);
          for (typename SparseMatrix<T>::const_iterator it(obj->begin());
               it!=obj->end(); ++it) {
            int coords[2];
            it.getCoords(coords[0],coords[1]);
            aux_dims[ndim] = coords[ndim];
            if (*it < dest_it(aux_dims[0],aux_dims[1])) {
              dest_it(aux_dims[0],aux_dims[1]) = *it;
              argmin_it(aux_dims[0],aux_dims[1]) = aux_dims[ndim];
            }
          }
        }
        return dest;
      }
    
      template <typename T>
      Matrix<T> *matMax(const Matrix<T> *obj,
                        int dim,
                        Matrix<T> *dest,
                        Matrix<int32_t> *argmax) {
        if (argmax == 0) {
          return MatrixScalarReduce1OverDimension(obj, dim,
                                                  AprilMath::Functors::r_max<T>(),
                                                  AprilMath::Functors::r_max<T>(),
                                                  -Limits<T>::infinity(), dest);
        }
        else {
          return MatrixScalarReduceMinMaxOverDimension(obj, dim,
                                                       AprilMath::Functors::r_max2<T>(),
                                                       -Limits<T>::infinity(), argmax, dest);
        }
      }

      // TODO: use a wrapper for GPU/CPU
      template <typename T>
      Matrix<T> *matMax(const SparseMatrix<T> *obj,
                        int dim, Matrix<T> *dest,
                        Matrix<int32_t> *argmax) {
        if (dim != 0 && dim != 1) {
          ERROR_EXIT1(128, "Incorrect given dimension %d\n", dim);
        }
        int ndim = (dim==0)?(1):(0);
        if (dest) {
          if (dest->getDimSize(dim) != 1 ||
              dest->getDimSize(ndim) != obj->getDimSize(ndim)) {
            ERROR_EXIT(128, "Incorrect matrix sizes\n");
          }
        }
        else {
          int result_dims[2] = { obj->getDimSize(0), obj->getDimSize(1) };
          result_dims[dim] = 1;
          dest = new Matrix<T>(1, result_dims);
        }
        if (argmax) {
          if (argmax->getDimSize(dim) != 1 ||
              argmax->getDimSize(ndim) != obj->getDimSize(ndim)) {
            ERROR_EXIT(128, "Incorrect matrix sizes\n");
          }
          AprilMath::MatrixExt::Initializers::matZeros(argmax);
        }
        AprilMath::MatrixExt::Initializers::matZeros(dest);
        typename Matrix<T>::random_access_iterator dest_it(dest);
        int aux_dims[2] = { 0, 0 };
        if (argmax == 0) {
          for (typename SparseMatrix<T>::const_iterator it(obj->begin());
               it!=obj->end(); ++it) {
            int coords[2];
            it.getCoords(coords[0],coords[1]);
            aux_dims[ndim] = coords[ndim];
            dest_it(aux_dims[0],aux_dims[1]) =
              AprilUtils::max(dest_it(aux_dims[0],aux_dims[1]),(*it));
          }
        }
        else {
          typename Matrix<int32_t>::random_access_iterator argmax_it(argmax);
          for (typename SparseMatrix<T>::const_iterator it(obj->begin());
               it!=obj->end(); ++it) {
            int coords[2];
            it.getCoords(coords[0],coords[1]);
            aux_dims[ndim] = coords[ndim];
            if (dest_it(aux_dims[0],aux_dims[1]) < *it) {
              dest_it(aux_dims[0],aux_dims[1]) = *it;
              argmax_it(aux_dims[0],aux_dims[1]) = aux_dims[ndim];
            }
          }
        }
        return dest;
      }
    
      // FIXME: using WRAPPER
      template <typename T>
      T matMin(const Matrix<T> *obj, int &arg_min, int &arg_min_raw_pos) {
        typename Matrix<T>::const_iterator it(obj->begin());
        typename Matrix<T>::const_iterator result =
          AprilUtils::argmin(it, typename Matrix<T>::const_iterator(obj->end()));
        arg_min = result.getIdx();
        arg_min_raw_pos = result.getRawPos();
        return *result;
      }
    
      // FIXME: using WRAPPER
      template <typename T>
      T matMin(const SparseMatrix<T> *obj, int &c0, int &c1) {
        typename SparseMatrix<T>::const_iterator it =
          AprilUtils::argmin(obj->begin(),obj->end());
        it.getCoords(c0,c1);
        return *it;
      }

      // FIXME: using WRAPPER
      template<typename T>
      T matMax(const Matrix<T> *obj, int &arg_max, int &arg_max_raw_pos) {
        typename Matrix<T>::const_iterator it(obj->begin());
        typename Matrix<T>::const_iterator result =
          AprilUtils::argmax(it, typename Matrix<T>::const_iterator(obj->end()));
        arg_max = result.getIdx();
        arg_max_raw_pos = result.getRawPos();
        return *result;
      }
    
      // FIXME: using WRAPPER
      template<typename T>
      T matMax(const SparseMatrix<T> *obj, int &c0, int &c1) {
        typename SparseMatrix<T>::const_iterator it =
          AprilUtils::argmax(obj->begin(),obj->end());
        it.getCoords(c0,c1);
        return *it;
      }

      // FIXME: using WRAPPER
      template<typename T>
      void matMinAndMax(const Matrix<T> *obj, T &min, T &max) {
        typename Matrix<T>::const_iterator it(obj->begin());
        min = *it;
        max = *it;
        for (; it!=obj->end(); ++it) {
          if (*it < min) min = *it;
          if (*it > max) max = *it;
        }
      }
      
      template<typename T>
      void matMinAndMax(const SparseMatrix<T> *obj, T &min, T &max) {
        typename SparseMatrix<T>::const_iterator it(obj->begin());
        min = max = *it;
        ++it;
        for (; it != obj->end(); ++it) {
          if ( max < (*it) ) max = *it;
          else if ( (*it) < min ) min = *it;
        }
      }
    
      template <typename T>
      Matrix<T> *matMaxSelDim(const Matrix<T> *obj,
                              const int dim,
                              Int32GPUMirroredMemoryBlock *raw_positions,
                              const int shift,
                              Matrix<T> *result) {
        if (dim < 0 || dim > obj->getNumDim()) {
          ERROR_EXIT2(128, "Incorrect dimension %d, numDim=%d\n",
                      dim, obj->getNumDim());
        }
        if (result == 0) {
          result = new Matrix<T>(1, obj->getDimSize(dim));
        }
        else {
          if (result->size()!=obj->getDimSize(dim) || result->getNumDim()!=1) {
            ERROR_EXIT1(128, "Incorrect result matrix size, "
                        "expected unidimensional matrix with size %d\n",
                        obj->getDimSize(dim));
          }
        }
#ifdef USE_CUDA
        result->setUseCuda(obj->getCudaFlag());
#endif
        int *argmax = 0;
        if (raw_positions != 0) {
          argmax = raw_positions->getPPALForWrite() + shift;
        }
        switch(obj->getNumDim()) {
        case 1:
          ERROR_EXIT(128, "Impossible to compute maxSelDim when numDim=1\n");
          break;
        case 2:
          {
            const int other_dim = 1 - dim;
            T *res_ptr = result->getRawDataAccess()->getPPALForWrite();
            const T *src_ptr = obj->getRawDataAccess()->getPPALForRead();
            for (int i=0; i<obj->getDimSize(dim); ++i, ++res_ptr) {
              int current_raw_pos = obj->getOffset() + i*obj->getStrideSize(dim);
              int raw_pos_max = current_raw_pos;
              *res_ptr = src_ptr[current_raw_pos];
              current_raw_pos += obj->getStrideSize(other_dim);
              for (int j=1; j<obj->getDimSize(other_dim);
                   ++j, current_raw_pos += obj->getStrideSize(other_dim)) {
                if (src_ptr[current_raw_pos] > *res_ptr) {
                  *res_ptr    = src_ptr[current_raw_pos];
                  raw_pos_max = current_raw_pos;
                }
              }
              if (argmax) argmax[i] = raw_pos_max;
            }
            break;
          }
        case 3:
          {
            int other_dim1 = (dim+1)%3;
            int other_dim2 = (dim+2)%3;
            if (other_dim2 < other_dim1) {
              AprilUtils::swap(other_dim1, other_dim2);
            }
            T *res_ptr = result->getRawDataAccess()->getPPALForWrite();
            const T *src_ptr = obj->getRawDataAccess()->getPPALForRead();
            for (int i=0; i<obj->getDimSize(dim); ++i, ++res_ptr) {
              int raw_pos_max = i*obj->getStrideSize(dim) + obj->getOffset();
              *res_ptr = src_ptr[raw_pos_max];
              for (int j=0; j<obj->getDimSize(other_dim1); ++j) {
                int current_raw_pos = obj->getOffset() + i*obj->getStrideSize(dim) + j*obj->getStrideSize(other_dim1);
                for (int k=0; k<obj->getDimSize(other_dim2);
                     ++k, current_raw_pos += obj->getStrideSize(other_dim2)) {
                  if (src_ptr[current_raw_pos] > *res_ptr) {
                    *res_ptr    = src_ptr[current_raw_pos];
                    raw_pos_max = current_raw_pos;
                  }
                }
              }
              if (argmax) argmax[i] = raw_pos_max;
            }
            break;
          }
        case 4:
          {
            int other_dim1 = (dim+1)%4;
            int other_dim2 = (dim+2)%4;
            int other_dim3 = (dim+3)%4;
            if (other_dim1 > other_dim2)
              AprilUtils::swap(other_dim1, other_dim2);
            if (other_dim2 > other_dim3) {
              AprilUtils::swap(other_dim2, other_dim3);
              if (other_dim1 > other_dim2)
                AprilUtils::swap(other_dim1, other_dim2);
            }
            T *res_ptr = result->getRawDataAccess()->getPPALForWrite();
            const T *src_ptr = obj->getRawDataAccess()->getPPALForRead();
            for (int i=0; i<obj->getDimSize(dim); ++i, ++res_ptr) {
              int raw_pos_max = i*obj->getStrideSize(dim) + obj->getOffset();
              *res_ptr = src_ptr[raw_pos_max];
              for (int j=0; j<obj->getDimSize(other_dim1); ++j) {
                for (int k=0; k<obj->getDimSize(other_dim2); ++k) {
                  int current_raw_pos=obj->getOffset()+i*obj->getStrideSize(dim)+j*obj->getStrideSize(other_dim1)+k*obj->getStrideSize(other_dim2);
                  for (int k2=0; k2<obj->getDimSize(other_dim3);
                       ++k2, current_raw_pos += obj->getStrideSize(other_dim3)) {
                    if (src_ptr[current_raw_pos] > *res_ptr) {
                      *res_ptr    = src_ptr[current_raw_pos];
                      raw_pos_max = current_raw_pos;
                    }
                  }
                }
              }
              if (argmax) argmax[i] = raw_pos_max;
            }
            break;
          }
        default:
          {
            T *res_ptr = result->getRawDataAccess()->getPPALForWrite();
            for (int i=0; i<obj->getDimSize(dim); ++i, ++res_ptr) {
              int aux, argmax_raw_pos;
              AprilUtils::SharedPtr< Matrix<T> >
                current( const_cast<Matrix<T>*>(obj)->select(dim, i) );
              matMax(current.get(), aux, argmax_raw_pos);
              if (argmax) argmax[i] = argmax_raw_pos;
            }
          }
        }
        return result;
      }
      
            template <typename T>
      T matSum(const Matrix<T> *obj) {
        return MatrixSpanSumReduce1(obj,
                                    ScalarToSpanReduce1< T, T, AprilMath::Functors::r_add<T,T> >
                                    (AprilMath::Functors::r_add<T,T>()));
      }
      
      template <>
      ComplexF matSum(const Matrix<ComplexF> *obj) {
        return MatrixScalarReduce1(obj,
                                   AprilMath::Functors::r_add<ComplexF,ComplexF>(),
                                   AprilMath::Functors::r_add<ComplexF,ComplexF>(),
                                   ComplexF(0.0f,0.0f));
      }
      
      template <typename T>
      T matSum(const SparseMatrix<T> *obj) {
        return SparseMatrixScalarReduce1<T>(obj,
                                            AprilMath::Functors::r_add<T,T>(),
                                            AprilMath::Limits<T>::zero());
      }
    
      template <typename T>
      Matrix<T> *matSum(Matrix<T> *obj,
                        int dim,
                        Matrix<T> *dest,
                        bool accumulated) {
        return MatrixScalarReduce1OverDimension(obj, dim,
                                                AprilMath::Functors::r_add<T,T>(),
                                                AprilMath::Functors::r_add<T,T>(),
                                                AprilMath::Limits<T>::zero(),
                                                dest, !accumulated);
      }

      // TODO: Implement using a wrapper for GPU/CPU computation.
      template <typename T>
      Matrix<T> *matSum(const SparseMatrix<T> *obj, int dim,
                        Matrix<T> *dest, bool accumulated) {
        if (dim != 0 && dim != 1) {
          ERROR_EXIT1(128, "Incorrect given dimension %d\n", dim);
        }
        int ndim = (dim==0)?(1):(0);
        if (dest) {
          if (dest->getDimSize(dim) != 1 ||
              dest->getDimSize(ndim) != obj->getDimSize(ndim)) {
            ERROR_EXIT(128, "Incorrect matrix sizes\n");
          }
          if (!accumulated) {
            AprilMath::MatrixExt::Initializers::matZeros(dest);
          }
        }
        else {
          int result_dims[2] = { obj->getDimSize(0), obj->getDimSize(1) };
          result_dims[dim] = 1;
          dest = new Matrix<T>(1, result_dims);
          AprilMath::MatrixExt::Initializers::matZeros(dest);
        }
        typename Matrix<T>::random_access_iterator dest_it(dest);
        int aux_dims[2] = { 0, 0 };
        for (typename SparseMatrix<T>::const_iterator it(obj->begin());
             it!=obj->end(); ++it) {
          int coords[2];
          it.getCoords(coords[0],coords[1]);
          aux_dims[ndim] = coords[ndim];
          dest_it(aux_dims[0],aux_dims[1]) += (*it);
        }
        return dest;
      }

      /**** COMPONENT WISE OPERATIONS ****/
    
      template <typename T>
      bool matEquals(const Matrix<T> *a, const Matrix<T> *b,
                     float epsilon) {
        if (!a->sameDim(b)) return false;
        typename Matrix<T>::const_iterator a_it(a->begin());
        typename Matrix<T>::const_iterator b_it(b->begin());
        while(a_it != a->end() && b_it != b->end()) {
          if (!m_relative_equals(*a_it, *b_it, epsilon)) {
            return false;
          }
          ++a_it;
          ++b_it;
        }
        if (a_it == a->end() && b_it == b->end()) {
          return true;
        }
        else {
          return false;
        }
      }

      template <typename T>
      bool matEquals(const SparseMatrix<T> *a,
                     const SparseMatrix<T> *b,
                     float epsilon) {
        if (!a->sameDim(b)) return false;
        typename SparseMatrix<T>::const_iterator a_it(a->begin());
        typename SparseMatrix<T>::const_iterator b_it(b->begin());
        while(a_it != a->end() && b_it != b->end()) {
          int a_c0, a_c1, b_c0, b_c1;
          a_it.getCoords(a_c0, a_c1);
          b_it.getCoords(b_c0, b_c1);
          if (a_c0 != b_c0 || a_c1 != b_c1 ||
              !m_relative_equals(*a_it, *b_it, epsilon)) return false;
          ++a_it;
          ++b_it;
        }
        if (a_it != a->end() || b_it != b->end()) return false;
        return true;
      }

      template <typename T>
      bool matIsFinite(const Matrix<T> *obj) {
        return MatrixScalarReduce1(obj,
                                   AprilMath::make_r_map1<T,bool>
                                   (AprilMath::Functors::m_is_finite<T>(),
                                    AprilMath::Functors::r_and<bool>()),
                                   AprilMath::Functors::r_and<bool>(),
                                   true);
      }
      
      template Matrix<float> *matMin(const Matrix<float> *,
                                     int,
                                     Matrix<float> *,
                                     Matrix<int32_t> *);
      template Matrix<float> *matMax(const Matrix<float> *,
                                     int,
                                     Matrix<float> *,
                                     Matrix<int32_t> *);
      template float matMin(const Matrix<float> *, int &, int &);
      template float matMax(const Matrix<float> *, int &, int &);
      template void matMinAndMax(const Matrix<float> *, float &, float &);
      template Matrix<float> *matMaxSelDim(const Matrix<float> *,
                                           const int,
                                           Int32GPUMirroredMemoryBlock *,
                                           const int,
                                           Basics::Matrix<float> *);
      template float matSum(const Matrix<float> *);
      template Matrix<float> *matSum(Matrix<float> *,
                                     int,
                                     Matrix<float> *,
                                     bool);
      template bool matEquals(const Matrix<float> *, const Matrix<float> *,
                              float);
      template bool matIsFinite(const Matrix<float> *);

      
      template Matrix<double> *matMin(const Matrix<double> *,
                                      int,
                                      Matrix<double> *,
                                      Matrix<int32_t> *);
      template Matrix<double> *matMax(const Matrix<double> *,
                                      int,
                                      Matrix<double> *,
                                      Matrix<int32_t> *);
      template double matMin(const Matrix<double> *, int &, int &);
      template double matMax(const Matrix<double> *, int &, int &);
      template void matMinAndMax(const Matrix<double> *, double &, double &);
      template Matrix<double> *matMaxSelDim(const Matrix<double> *,
                                            const int,
                                            Int32GPUMirroredMemoryBlock *,
                                            const int,
                                            Basics::Matrix<double> *);
      template double matSum(const Matrix<double> *);
      template Matrix<double> *matSum(Matrix<double> *,
                                      int,
                                      Matrix<double> *,
                                      bool);
      template bool matEquals(const Matrix<double> *, const Matrix<double> *,
                              float);
      template bool matIsFinite(const Matrix<double> *);

      
      template Matrix<float> *matMin(const SparseMatrix<float> *, int ,
                                     Matrix<float> *,
                                     Matrix<int32_t> *);
      template Matrix<float> *matMax(const SparseMatrix<float> *,
                                     int, Matrix<float> *,
                                     Matrix<int32_t> *);
      template float matMin(const SparseMatrix<float> *, int &, int &);
      template float matMax(const SparseMatrix<float> *, int &, int &);
      template void matMinAndMax(const SparseMatrix<float> *, float &, float &);
      template float matSum(const SparseMatrix<float> *);
      template Matrix<float> *matSum(const SparseMatrix<float> *, int,
                                     Matrix<float> *,
                                     bool);
      template bool matEquals(const SparseMatrix<float> *,
                              const SparseMatrix<float> *,
                              float);


      template bool matEquals(const SparseMatrix<ComplexF> *,
                              const SparseMatrix<ComplexF> *,
                              float);


      template Matrix<double> *matMin(const SparseMatrix<double> *, int ,
                                      Matrix<double> *,
                                      Matrix<int32_t> *);
      template Matrix<double> *matMax(const SparseMatrix<double> *,
                                      int, Matrix<double> *,
                                      Matrix<int32_t> *);
      template double matMin(const SparseMatrix<double> *, int &, int &);
      template double matMax(const SparseMatrix<double> *, int &, int &);
      template void matMinAndMax(const SparseMatrix<double> *, double &, double &);
      template double matSum(const SparseMatrix<double> *);
      template Matrix<double> *matSum(const SparseMatrix<double> *, int,
                                      Matrix<double> *,
                                      bool);
      template bool matEquals(const SparseMatrix<double> *,
                              const SparseMatrix<double> *,
                              float);
      
      
      template ComplexF matSum(const Matrix<ComplexF> *);
      template Matrix<ComplexF> *matSum(Matrix<ComplexF> *,
                                        int,
                                        Matrix<ComplexF> *,
                                        bool);
      template bool matEquals(const Matrix<ComplexF> *, const Matrix<ComplexF> *,
                              float);
      template bool matIsFinite(const Matrix<ComplexF> *);
      

      template Matrix<int32_t> *matMin(const SparseMatrix<int32_t> *, int ,
                                       Matrix<int32_t> *,
                                       Matrix<int32_t> *);
      template Matrix<int32_t> *matMax(const SparseMatrix<int32_t> *,
                                       int, Matrix<int32_t> *,
                                       Matrix<int32_t> *);
      template int32_t matMin(const SparseMatrix<int32_t> *, int &, int &);
      template int32_t matMax(const SparseMatrix<int32_t> *, int &, int &);
      
      
      template Matrix<int32_t> *matMin(const Matrix<int32_t> *,
                                       int,
                                       Matrix<int32_t> *,
                                       Matrix<int32_t> *);
      template Matrix<int32_t> *matMax(const Matrix<int32_t> *,
                                       int,
                                       Matrix<int32_t> *,
                                       Matrix<int32_t> *);
      template int32_t matMin(const Matrix<int32_t> *, int &, int &);
      template int32_t matMax(const Matrix<int32_t> *, int &, int &);
      template void matMinAndMax(const Matrix<int32_t> *, int32_t &, int32_t &);
      template Matrix<int32_t> *matMaxSelDim(const Matrix<int32_t> *,
                                             const int,
                                             Int32GPUMirroredMemoryBlock *,
                                             const int,
                                             Basics::Matrix<int32_t> *);
      template int32_t matSum(const Matrix<int32_t> *);
      template Matrix<int32_t> *matSum(Matrix<int32_t> *,
                                       int,
                                       Matrix<int32_t> *,
                                       bool);
      template bool matEquals(const Matrix<int32_t> *, const Matrix<int32_t> *,
                              float);

      
      template bool matEquals(const Matrix<char> *, const Matrix<char> *,
                              float);

      template bool matEquals(const Matrix<bool> *, const Matrix<bool> *,
                              float);
      
    } // namespace Reductions
    
  } // namespace MatrixExt
} // namespace AprilMath
