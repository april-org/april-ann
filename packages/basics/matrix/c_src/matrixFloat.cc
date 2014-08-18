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
#include <cmath>

#include "binarizer.h"
#include "cblas_headers.h"
#include "check_floats.h"
#include "constString.h"
#include "swap.h"
#include "matrix.h"
#include "matrixFloat.h"
#include "matrix_generic_math_templates.h" // functions which apply functors
#include "matrix_generic_math_functors.h"  // standard functors
#include "wrapper.h" // wrappers of mathematical function (for CPU/GPU)

// WARNING: ALL THE METHODS IMPLEMENTED HERE ARE SPECIALIZED TO FLOAT VERSION

using april_utils::constString;

namespace basics {

  namespace MatrixIO {
  
    /////////////////////////////////////////////////////////////////////////
  
    template<>
    bool AsciiExtractor<float>::operator()(constString &line,
                                           float &destination) {
      bool result = line.extract_float(&destination);
      if (!result) return false;
      return true;
    }
  
    template<>
    bool BinaryExtractor<float>::operator()(constString &line,
                                            float &destination) {
      if (!line.extract_float_binary(&destination)) return false;
      return true;
    }
  
    template<>
    int AsciiSizer<float>::operator()(const Matrix<float> *mat) {
      return mat->size()*12;
    }

    template<>
    int BinarySizer<float>::operator()(const Matrix<float> *mat) {
      return april_utils::binarizer::buffer_size_32(mat->size());
    }

    template<>
    void AsciiCoder<float>::operator()(const float &value,
                                       AprilIO::StreamInterface *stream) {
      stream->printf("%.5g", value);
    }
  
    template<>
    void BinaryCoder<float>::operator()(const float &value,
                                        AprilIO::StreamInterface *stream) {
      char b[5];
      april_utils::binarizer::code_float(value, b);
      stream->put(b, sizeof(char)*5);
    }

    /////////////////////////////////////////////////////////////////////////////
  

  } // namespace MatrixIO

  /************* FILL FUNCTION **************/
  DEF_CWISE_FUNCTOR_1(doFill,float);
  template<>
  void Matrix<float>::fill(float value) {
    applyFunctionWithSpanIterator<float>(this,
                                         MAKE_CWISE_FUNCTOR_1(doFill,float,
                                                              value));
  }

  /************* CLAMP FUNCTION **************/
  struct clamp_functor {
    float lower, upper;
    clamp_functor(float lower, float upper) :
      lower(lower), upper(upper) { }
    void operator()(MatrixFloat *m,
                    unsigned int size, unsigned int stride,
                    unsigned int offset) const {
      april_math::doClamp(size, m->getRawDataAccess(), stride, offset,
                          lower, upper, m->getCudaFlag());
    }
  };
  template<>
  void Matrix<float>::clamp(float lower, float upper) {
    clamp_functor functor(lower, upper);
    applyFunctionWithSpanIterator<float>(this, functor);
  }

  /************* SUM FUNCTION **************/
  struct sum_functor {
    float operator()(const MatrixFloat *m,
                     unsigned int size, unsigned int stride,
                     unsigned int offset) const {
      return april_math::doSum(size, m->getRawDataAccess(), stride, offset,
                               m->getCudaFlag(), 0.0f);
    }
  };
  template<>
  float Matrix<float>::sum() const {
    sum_functor functor;
    return applySumReductionWithSpanIterator<float>(this, functor);
  }

  /**** COMPONENT WISE OPERATIONS ****/


  /************* scalarAdd FUNCTION **************/
  DEF_CWISE_FUNCTOR_1(doScalarAdd,float);
  template<>
  void Matrix<float>::scalarAdd(float s) {
    applyFunctionWithSpanIterator<float>(this,
                                         MAKE_CWISE_FUNCTOR_1(doScalarAdd,
                                                              float,s));
  }

  /************* equals FUNCTION **************/
  struct equals_functor {
    float epsilon;
    equals_functor(float epsilon) : epsilon(epsilon) { }
    bool operator()(const MatrixFloat *m1,
                    const MatrixFloat *m2,
                    unsigned int size,
                    unsigned int stride1,
                    unsigned int stride2,
                    unsigned int offset1,
                    unsigned int offset2) const {
      return april_math::doEquals(size, m1->getRawDataAccess(), m2->getRawDataAccess(),
                                  stride1, stride2, offset1, offset2, epsilon,
                                  m1->getCudaFlag() && m2->getCudaFlag());
    }
  };
  template<>
  bool Matrix<float>::equals(const Matrix<float> *other, float epsilon) const {
    if (!sameDim(other)) return false;
    equals_functor functor(epsilon);
    return applyBinaryAndReductionWithSpanIterator<float>(this, other, functor);
  }

  /************* LOG FUNCTION **************/
  DEF_CWISE_FUNCTOR_0(doPLogP,float);
  template<>
  void Matrix<float>::plogp() {
    applyFunctionWithSpanIterator(this, MAKE_CWISE_FUNCTOR_0(doPLogP,float));
  }

  /************* LOG FUNCTION **************/
  DEF_CWISE_FUNCTOR_0(doLog,float);
  template<>
  void Matrix<float>::log() {
    applyFunctionWithSpanIterator<float>(this,
                                         MAKE_CWISE_FUNCTOR_0(doLog,float));
  }

  /************* LOG1P FUNCTION **************/
  DEF_CWISE_FUNCTOR_0(doLog1p,float);
  template<>
  void Matrix<float>::log1p() {
    applyFunctionWithSpanIterator<float>(this,
                                         MAKE_CWISE_FUNCTOR_0(doLog1p,float));

  }

  /************* EXP FUNCTION **************/
  DEF_CWISE_FUNCTOR_0(doExp,float);
  template<>
  void Matrix<float>::exp() {
    applyFunctionWithSpanIterator<float>(this,
                                         MAKE_CWISE_FUNCTOR_0(doExp,float));
  }

  /************* SQRT FUNCTION **************/
  DEF_CWISE_FUNCTOR_0(doSqrt,float);
  template<>
  void Matrix<float>::sqrt() {
    applyFunctionWithSpanIterator<float>(this,
                                         MAKE_CWISE_FUNCTOR_0(doSqrt,float));
  }

  /************* POW FUNCTION **************/
  DEF_CWISE_FUNCTOR_1(doPow,float);
  template<>
  void Matrix<float>::pow(float value) {
    applyFunctionWithSpanIterator<float>(this,
                                         MAKE_CWISE_FUNCTOR_1(doPow,float,value));
  }

  /************* TAN FUNCTION **************/
  DEF_CWISE_FUNCTOR_0(doTan,float);
  template<>
  void Matrix<float>::tan() {
    applyFunctionWithSpanIterator<float>(this,
                                         MAKE_CWISE_FUNCTOR_0(doTan,float));
  }

  /************* TANH FUNCTION **************/
  DEF_CWISE_FUNCTOR_0(doTanh,float);
  template<>
  void Matrix<float>::tanh() {
    applyFunctionWithSpanIterator<float>(this,
                                         MAKE_CWISE_FUNCTOR_0(doTanh,float));
  }

  /************* ATAN FUNCTION **************/
  DEF_CWISE_FUNCTOR_0(doAtan,float);
  template<>
  void Matrix<float>::atan() {
    applyFunctionWithSpanIterator<float>(this,
                                         MAKE_CWISE_FUNCTOR_0(doAtan,float));
  }

  /************* ATANH FUNCTION **************/
  DEF_CWISE_FUNCTOR_0(doAtanh,float);
  template<>
  void Matrix<float>::atanh() {
    applyFunctionWithSpanIterator<float>(this,
                                         MAKE_CWISE_FUNCTOR_0(doAtanh,float));
  }

  /************* SIN FUNCTION **************/
  DEF_CWISE_FUNCTOR_0(doSin,float);
  template<>
  void Matrix<float>::sin() {
    applyFunctionWithSpanIterator<float>(this,
                                         MAKE_CWISE_FUNCTOR_0(doSin,float));
  }

  /************* SINH FUNCTION **************/
  DEF_CWISE_FUNCTOR_0(doSinh,float);
  template<>
  void Matrix<float>::sinh() {
    applyFunctionWithSpanIterator<float>(this,
                                         MAKE_CWISE_FUNCTOR_0(doSinh,float));
  }

  /************* ASIN FUNCTION **************/
  DEF_CWISE_FUNCTOR_0(doAsin,float);
  template<>
  void Matrix<float>::asin() {
    applyFunctionWithSpanIterator<float>(this,
                                         MAKE_CWISE_FUNCTOR_0(doAsin,float));
  }

  /************* ASINH FUNCTION **************/
  DEF_CWISE_FUNCTOR_0(doAsinh,float);
  template<>
  void Matrix<float>::asinh() {
    applyFunctionWithSpanIterator<float>(this,
                                         MAKE_CWISE_FUNCTOR_0(doAsinh,float));
  }

  /************* COS FUNCTION **************/
  DEF_CWISE_FUNCTOR_0(doCos,float);
  template<>
  void Matrix<float>::cos() {
    applyFunctionWithSpanIterator<float>(this,
                                         MAKE_CWISE_FUNCTOR_0(doCos,float));
  }

  /************* COSH FUNCTION **************/
  DEF_CWISE_FUNCTOR_0(doCosh,float);
  template<>
  void Matrix<float>::cosh() {
    applyFunctionWithSpanIterator<float>(this,
                                         MAKE_CWISE_FUNCTOR_0(doCosh,float));
  }

  /************* ACOS FUNCTION **************/
  DEF_CWISE_FUNCTOR_0(doAcos,float);
  template<>
  void Matrix<float>::acos() {
    applyFunctionWithSpanIterator<float>(this,
                                         MAKE_CWISE_FUNCTOR_0(doAcos,float));
  }

  /************* ACOSH FUNCTION **************/
  DEF_CWISE_FUNCTOR_0(doAcosh,float);
  template<>
  void Matrix<float>::acosh() {
    applyFunctionWithSpanIterator<float>(this,
                                         MAKE_CWISE_FUNCTOR_0(doAcosh,float));
  }

  /************* ABS FUNCTION **************/
  DEF_CWISE_FUNCTOR_0(doAbs,float);
  template<>
  void Matrix<float>::abs() {
    applyFunctionWithSpanIterator<float>(this,
                                         MAKE_CWISE_FUNCTOR_0(doAbs,float));
  }

  /************* ABS FUNCTION **************/
  DEF_CWISE_FUNCTOR_0(doComplement,float);
  template<>
  void Matrix<float>::complement() {
    applyFunctionWithSpanIterator<float>(this,
                                         MAKE_CWISE_FUNCTOR_0(doComplement,
                                                              float));
  }


  /************* SIGN FUNCTION **************/
  DEF_CWISE_FUNCTOR_0(doSign,float);
  template<>
  void Matrix<float>::sign() {
    applyFunctionWithSpanIterator<float>(this,
                                         MAKE_CWISE_FUNCTOR_0(doSign,float));;
  }

  /****************************************************************************/

  // COMPONENT-WISE MULTIPLICATION
  struct cmul_functor {
    cmul_functor() { }
    void operator()(MatrixFloat *one, const MatrixFloat *other,
                    unsigned int size,
                    unsigned int stride_one,
                    unsigned int stride_other,
                    unsigned int offset_one,
                    unsigned int offset_other) const {
      april_math::doCmul(size,
                         other->getRawDataAccess(),
                         offset_other, stride_other,
                         one->getRawDataAccess(),
                         offset_one, stride_one,
                         one->getCudaFlag());
    }
  };
  template<>
  void Matrix<float>::cmul(const Matrix<float> *other) {
    if (size() != other->size())
      ERROR_EXIT2(128, "Incorrect matrices sizes: %d != %d\n",
                  size(), other->size());
    if (major_order != other->major_order)
      ERROR_EXIT(128, "Matrices with different major orders\n");
    cmul_functor functor;
    applyBinaryFunctionWithSpanIterator<float>(this, other, functor);
  }

  /**** BLAS OPERATIONS ****/
  struct copy_functor {
    void operator()(MatrixFloat *dest, const MatrixFloat *orig,
                    unsigned int size,
                    unsigned int stride_dest,
                    unsigned int stride_orig,
                    unsigned int offset_dest,
                    unsigned int offset_orig) const {
      april_math::doCopy(size,
                         orig->getRawDataAccess(),
                         offset_orig, stride_orig,
                         dest->getRawDataAccess(),
                         offset_dest, stride_dest,
                         orig->getCudaFlag());
    }
  };
  template<>
  void Matrix<float>::copy(const Matrix<float> *other) {
    if (this != other) {
      if (size() != other->size())
        ERROR_EXIT2(128, "Incorrect matrices sizes: %d != %d\n",
                    size(), other->size());
      if (major_order != other->major_order)
        ERROR_EXIT(128, "Matrices with different major orders\n");
      if (! sameDim(other) )
        ERROR_EXIT(128, "Matrices with different dimension sizes\n");
      use_cuda = other->use_cuda;
      copy_functor functor;
      applyBinaryFunctionWithSpanIterator<float>(this, other, functor);
    }
  }

  /********** SCAL FUNCTION ***************/
  DEF_CWISE_FUNCTOR_1(doScal,float);
  template<>
  void Matrix<float>::scal(float value) {
#ifdef USE_MKL
    applyFunctionWithSpanIteratorNOPARALLEL<float>(this,
                                                   MAKE_CWISE_FUNCTOR_1(doScal,
                                                                        float,
                                                                        value));
#else
    applyFunctionWithSpanIterator<float>(this,
                                         MAKE_CWISE_FUNCTOR_1(doScal,float,
                                                              value));
#endif
  }

  /********** DIV FUNCTION ***************/
  DEF_CWISE_FUNCTOR_1(doDiv,float);
  template<>
  void Matrix<float>::div(float value) {
    applyFunctionWithSpanIterator<float>(this,
                                         MAKE_CWISE_FUNCTOR_1(doDiv,float,
                                                              value));
  }

  /********** NORM2 FUNCTION ***************/
  struct norm2_functor {
    float operator()(const MatrixFloat *m, unsigned int size, unsigned int stride,
                     unsigned int offset) const {
      return april_math::doNrm2(size, m->getRawDataAccess(), stride, offset,
                                m->getCudaFlag());
    }
  };
  struct norm2_reductor {
    float operator()(float accum, float other) const {
      return accum + other*other;
    }
  };
  // In this method we do ad-hoc specialization of BASIC cases because
  // we avoid the SQUARE and SQRT functions
  template<>
  float Matrix<float>::norm2() const {
    float v;
    // Contiguous memory block
    if (getIsContiguous()) {
      v=april_math::doNrm2(total_size, data.get(), 1, offset, use_cuda);
    }
    // One dimension
    else if (numDim == 1) {
      v=april_math::doNrm2(total_size, data.get(), stride[0], offset, use_cuda);
    }
    // General case
    else {
      norm2_functor  functor;
      norm2_reductor reductor;
      v = applyReductionWithSpanIteratorNOPARALLEL<float,float>(this,
                                                                functor,
                                                                reductor,
                                                                0.0f);
      v = sqrtf(v);
    }
    return v;
  }

  // FIXME: using WRAPPER
  template<>
  float Matrix<float>::min(int &arg_min, int &arg_min_raw_pos) const {
    const_iterator it(begin());
    const_iterator result = april_utils::argmin(it, const_iterator(end()));
    arg_min = result.getIdx();
    arg_min_raw_pos = result.getRawPos();
    return *result;
  }

  // FIXME: using WRAPPER
  template<>
  float Matrix<float>::max(int &arg_max, int &arg_max_raw_pos) const {
    const_iterator it(begin());
    const_iterator result = april_utils::argmax(it, const_iterator(end()));
    arg_max = result.getIdx();
    arg_max_raw_pos = result.getRawPos();
    return *result;
  }

  // FIXME: using WRAPPER
  template<>
  void Matrix<float>::minAndMax(float &min, float &max) const {
    if (major_order == CblasRowMajor) {
      const_iterator it(begin());
      min = *it;
      max = *it;
      for (; it!=end(); ++it) {
        if (*it < min) min = *it;
        if (*it > max) max = *it;
      }
    }
    else {
      const_col_major_iterator it(begin());
      min = *it;
      max = *it;
      for (; it!=end(); ++it) {
        if (*it < min) min = *it;
        if (*it > max) max = *it;
      }
    }
  }

  template <>
  Matrix<float> *Matrix<float>::maxSelDim(const int dim,
                                          april_math::Int32GPUMirroredMemoryBlock *raw_positions,
                                          int shift) const {
    if (dim < 0 || dim > numDim)
      ERROR_EXIT2(128, "Incorrect dimension %d, numDim=%d\n", dim, numDim);
    MatrixFloat *result = new MatrixFloat(1, &matrixSize[dim], major_order);
#ifdef USE_CUDA
    result->setUseCuda(use_cuda);
#endif
    int *argmax = 0;
    if (raw_positions != 0) {
      argmax = raw_positions->getPPALForWrite() + shift;
    }
    switch(numDim) {
    case 1:
      ERROR_EXIT(128, "Impossible to compute maxSelDim when numDim=1\n");
      break;
    case 2:
      {
        const int other_dim = 1 - dim;
        float *res_ptr = result->getRawDataAccess()->getPPALForWrite();
        const float *src_ptr = data->getPPALForRead();
        for (int i=0; i<matrixSize[dim]; ++i, ++res_ptr) {
          int current_raw_pos = offset + i*stride[dim];
          int raw_pos_max = current_raw_pos;
          *res_ptr = src_ptr[current_raw_pos];
          current_raw_pos += stride[other_dim];
          for (int j=1; j<matrixSize[other_dim]; ++j,current_raw_pos+=stride[other_dim]) {
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
        if (other_dim2 < other_dim1)
          april_utils::swap(other_dim1, other_dim2);
#ifdef USE_CUDA
        result->setUseCuda(use_cuda);
#endif
        float *res_ptr = result->getRawDataAccess()->getPPALForWrite();
        const float *src_ptr = data->getPPALForRead();
        for (int i=0; i<matrixSize[dim]; ++i, ++res_ptr) {
          int raw_pos_max = i*stride[dim] + offset;
          *res_ptr = src_ptr[raw_pos_max];
          for (int j=0; j<matrixSize[other_dim1]; ++j) {
            int current_raw_pos = offset + i*stride[dim] + j*stride[other_dim1];
            for (int k=0; k<matrixSize[other_dim2];
                 ++k, current_raw_pos += stride[other_dim2]) {
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
          april_utils::swap(other_dim1, other_dim2);
        if (other_dim2 > other_dim3) {
          april_utils::swap(other_dim2, other_dim3);
          if (other_dim1 > other_dim2)
            april_utils::swap(other_dim1, other_dim2);
        }
#ifdef USE_CUDA
        result->setUseCuda(use_cuda);
#endif
        float *res_ptr = result->getRawDataAccess()->getPPALForWrite();
        const float *src_ptr = data->getPPALForRead();
        for (int i=0; i<matrixSize[dim]; ++i, ++res_ptr) {
          int raw_pos_max = i*stride[dim] + offset;
          *res_ptr = src_ptr[raw_pos_max];
          for (int j=0; j<matrixSize[other_dim1]; ++j) {
            for (int k=0; k<matrixSize[other_dim2]; ++k) {
              int current_raw_pos=offset+i*stride[dim]+j*stride[other_dim1]+k*stride[other_dim2];
              for (int k2=0; k2<matrixSize[other_dim3];
                   ++k2, current_raw_pos += stride[other_dim3]) {
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
        float *res_ptr = result->getRawDataAccess()->getPPALForWrite();
        for (int i=0; i<matrixSize[dim]; ++i, ++res_ptr) {
          int aux, argmax_raw_pos;
          MatrixFloat *current = const_cast<MatrixFloat*>(this)->select(dim, i);
          current->max(aux, argmax_raw_pos);
          if (argmax) argmax[i] = argmax_raw_pos;
          delete current;
        }
      }
    }
    return result;
  }

  // FIXME: using WRAPPER
  template<>
  void Matrix<float>::adjustRange(float rmin, float rmax) {
    float mmin, mmax;
    minAndMax(mmin, mmax);
    // especial case, set all values to rmin
    if (mmax - mmin == 0) fill(rmin);
    else {
      float ratio = (rmax-rmin)/(mmax-mmin);
      if (mmin > 0.0f || mmin < 0.0f) scalarAdd(-mmin);
      scal(ratio);
      if (rmin > 0.0f || rmin < 0.0f) scalarAdd(rmin);
    }
  }

  // FIXME: using WRAPPER for generalized CULA, LAPACK, float and complex numbers
  template<>
  Matrix<float> *Matrix<float>::inv() {
    if (numDim != 2)
      ERROR_EXIT(128, "Only bi-dimensional matrices are allowed\n");
    if (matrixSize[0] != matrixSize[1])
      ERROR_EXIT(128, "Only square matrices are allowed\n");
    MatrixFloat *A = this->clone(CblasColMajor);
    int *IPIV = new int[numDim+1];
    int INFO;
    INFO = clapack_sgetrf(CblasColMajor,
                          A->numDim,A->numDim,A->getData(),A->stride[1],IPIV);
    checkLapackInfo(INFO);
    INFO = clapack_sgetri(CblasColMajor,
                          A->numDim,A->getData(),A->stride[1],IPIV);
    checkLapackInfo(INFO);
    delete[] IPIV;
    return A;
  }

  // FIXME: using WRAPPER for generalized CULA, LAPACK, float and complex numbers
  // WARNING: the V matrix is returned transposed
  template <>
  void Matrix<float>::svd(Matrix<float> **U, SparseMatrix<float> **S, Matrix<float> **VT) {
    if (numDim != 2)
      ERROR_EXIT(128, "Only bi-dimensional matrices are allowed\n");
    april_utils::SharedPtr< Matrix<float> > A( this->clone(CblasColMajor) );
    int INFO;
    const int m = A->matrixSize[0]; // cols
    const int n = A->matrixSize[1]; // rows
    const int lda = A->stride[1];
    const int numSV = m<n ? m : n;
    const int dimsU[2]  = {m, m};
    const int dimsVT[2] = {n, n};
    *U  = new Matrix<float>(2, dimsU,  CblasColMajor);
    *S  = SparseMatrix<float>::diag(numSV, 0.0f, CSR_FORMAT);
    *VT = new Matrix<float>(2, dimsVT, CblasColMajor);
    INFO = clapack_sgesdd(CblasColMajor, m, n, lda, A->getData(),
                          (*U)->getData(),
                          (*S)->getRawValuesAccess()->getPPALForWrite(),
                          (*VT)->getData());
    checkLapackInfo(INFO);
  }

  template <>
  void Matrix<float>::pruneSubnormalAndCheckNormal() {
    float *data = getRawDataAccess()->getPPALForReadAndWrite();
    if (!april_utils::check_floats(data, size()))
      ERROR_EXIT(128, "No finite numbers at weights matrix!!!\n");
  }


  // FIXME: IMPLEMENT THE BOOLEAN CONDITIONS USING CUDA WRAPPERS

  /* BOOLEAN CONDITIONS: this methods transforms the given matrix in a ZERO/ONE
     matrix, depending in the truth of the given condition */
  // less than
  template <>
  void Matrix<float>::LTCondition(float value) {
    iterator it(begin());
    while(it != end()) {
      if ( (*it) < value ) *it = 1.0f;
      else *it = 0.0f;
      ++it;
    }
  }

  template <>
  void Matrix<float>::LTCondition(Matrix<float> *value) {
    if (!sameDim(value))
      ERROR_EXIT(128, "Incompatible matrix sizes\n");
    const_iterator it_value(value->begin());
    iterator it(begin());
    while(it != end()) {
      if ( (*it) < (*it_value) ) *it = 1.0f;
      else *it = 0.0f;
      ++it;
      ++it_value;
    }
  }

  // greater than
  template <>
  void Matrix<float>::GTCondition(float value) {
    iterator it(begin());
    while(it != end()) {
      if ( (*it) > value ) *it = 1.0f;
      else *it = 0.0f;
      ++it;
    }
  }

  template <>
  void Matrix<float>::GTCondition(Matrix<float> *value) {
    if (!sameDim(value))
      ERROR_EXIT(128, "Incompatible matrix sizes\n");
    const_iterator it_value(value->begin());
    iterator it(begin());
    while(it != end()) {
      if ( (*it) > (*it_value) ) *it = 1.0f;
      else *it = 0.0f;
      ++it;
      ++it_value;
    }
  }
  // equals
  template <>
  void Matrix<float>::EQCondition(float value) {
    iterator it(begin());
    if (std::isnan(value)) {
      while(it != end()) {
        if ( std::isnan(*it) ) *it = 1.0f;
        else *it = 0.0f;
        ++it;
      }
    }
    else {
      while(it != end()) {
        if ( (*it) == value ) *it = 1.0f;
        else *it = 0.0f;
        ++it;
      }
    }
  }

  template <>
  void Matrix<float>::EQCondition(Matrix<float> *value) {
    if (!sameDim(value))
      ERROR_EXIT(128, "Incompatible matrix sizes\n");
    const_iterator it_value(value->begin());
    iterator it(begin());
    while(it != end()) {
      if ( (*it) == (*it_value) ||
           (std::isnan(*it) && std::isnan(*it_value)) ) {
        *it = 1.0f;
      }
      else {
        *it = 0.0f;
      }
      ++it;
      ++it_value;
    }
  }
  // not equals
  template <>
  void Matrix<float>::NEQCondition(float value) {
    iterator it(begin());
    if (std::isnan(value)) {
      while(it != end()) {
        if ( !std::isnan(*it) ) *it = 1.0f;
        else *it = 0.0f;
        ++it;
      }
    }
    else {
      while(it != end()) {
        if ( (*it) != value ) *it = 1.0f;
        else *it = 0.0f;
        ++it;
      }
    }
  }

  template <>
  void Matrix<float>::NEQCondition(Matrix<float> *value) {
    if (!sameDim(value))
      ERROR_EXIT(128, "Incompatible matrix sizes\n");
    const_iterator it_value(value->begin());
    iterator it(begin());
    while(it != end()) {
      if ( (*it) != (*it_value) &&
           std::isnan(*it) != std::isnan(*it_value) ) {
        *it = 1.0f;
      }
      else {
        *it = 0.0f;
      }
      ++it;
      ++it_value;
    }
  }
  //


  ///////////////////////////////////////////////////////////////////////////////

  template class Matrix<float>;

} // namespace basics
