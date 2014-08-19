/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
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

#ifndef MATRIXFLOAT_H
#define MATRIXFLOAT_H

#include "matrix.h"

namespace basics {

  namespace MatrixIO {
      
    /* Especialization of MatrixFloat ascii and binary extractors, sizers and
       coders */
    template<>
    bool AsciiExtractor<float>::operator()(april_utils::constString &line,
                                           float &destination);
  
    template<>
    bool BinaryExtractor<float>::operator()(april_utils::constString &line,
                                            float &destination);
  
    template<>
    int AsciiSizer<float>::operator()(const Matrix<float> *mat);

    template<>
    int BinarySizer<float>::operator()(const Matrix<float> *mat);

    template<>
    void AsciiCoder<float>::operator()(const float &value,
                                       AprilIO::StreamInterface *stream);
  
    template<>
    void BinaryCoder<float>::operator()(const float &value,
                                        AprilIO::StreamInterface *stream);
  
    /**************************************************************************/
 
  } // namespace AprilIO
  
  template<>
  float Matrix<float>::getTemplateOption(const april_utils::GenericOptions *options,
                                         const char *name, float default_value);
  
  //////////////////////////////////////////////////////////////////////////
  
  template<>
  void Matrix<float>::fill(float value);

  template<>
  void Matrix<float>::clamp(float lower, float upper);

  template<>
  float Matrix<float>::sum() const;

  template<>
  void Matrix<float>::scalarAdd(float s);

  template<>
  bool Matrix<float>::equals(const Matrix<float> *other, float epsilon) const;

  template<>
  void Matrix<float>::plogp();

  template<>
  void Matrix<float>::log();

  template<>
  void Matrix<float>::log1p();

  template<>
  void Matrix<float>::exp();

  template<>
  void Matrix<float>::sqrt();

  template<>
  void Matrix<float>::pow(float value);

  template<>
  void Matrix<float>::tan();

  template<>
  void Matrix<float>::tanh();

  template<>
  void Matrix<float>::atan();

  template<>
  void Matrix<float>::atanh();

  template <>
  void Matrix<float>::cos();

  template <>
  void Matrix<float>::cosh();

  template <>
  void Matrix<float>::acos();

  template <>
  void Matrix<float>::acosh();

  template <>
  void Matrix<float>::sin();

  template <>
  void Matrix<float>::sinh();

  template <>
  void Matrix<float>::asin();

  template <>
  void Matrix<float>::asinh();

  template <>
  void Matrix<float>::abs();

  template <>
  void Matrix<float>::complement();

  template <>
  void Matrix<float>::sign();

  template<>
  void Matrix<float>::cmul(const Matrix<float> *other);

  /**** BLAS OPERATIONS ****/

  template<>
  void Matrix<float>::copy(const Matrix<float> *other);

  template<>
  void Matrix<float>::scal(float value);

  template<>
  void Matrix<float>::div(float value);

  template<>
  float Matrix<float>::norm2() const;

  template<>
  float Matrix<float>::min(int &arg_min, int &arg_min_raw_pos) const;

  template<>
  float Matrix<float>::max(int &arg_max, int &arg_max_raw_pos) const;

  template<>
  void Matrix<float>::minAndMax(float &min, float &max) const;

  template <>
  Matrix<float> *Matrix<float>::maxSelDim(const int dim,
                                          april_math::Int32GPUMirroredMemoryBlock *raw_positions,
                                          int shift) const;

  template <>
  april_utils::log_float Matrix<float>::logDeterminant(float &sign);

  template <>
  double Matrix<float>::determinant();

  template <>
  Matrix<float> *Matrix<float>::cholesky(char uplo);

  template<>
  void Matrix<float>::adjustRange(float rmin, float rmax);

  template<>
  Matrix<float> *Matrix<float>::inv();

  template <>
  void Matrix<float>::svd(Matrix<float> **U, SparseMatrix<float> **S,
                          Matrix<float> **V);

  template <>
  void Matrix<float>::pruneSubnormalAndCheckNormal();

  /* BOOLEAN CONDITIONS: this methods transforms the given matrix in a ZERO/ONE
     matrix, depending in the truth of the given condition */
  // less than
  template <>
  void Matrix<float>::LTCondition(float value);
  template <>
  void Matrix<float>::LTCondition(Matrix<float> *value);
  // greater than
  template <>
  void Matrix<float>::GTCondition(float value);
  template <>
  void Matrix<float>::GTCondition(Matrix<float> *value);
  // equals
  template <>
  void Matrix<float>::EQCondition(float value);
  template <>
  void Matrix<float>::EQCondition(Matrix<float> *value);
  // not equals
  template <>
  void Matrix<float>::NEQCondition(float value);
  template <>
  void Matrix<float>::NEQCondition(Matrix<float> *value);
  //

  ////////////////////////////////////////////////////////////////////////////

  typedef Matrix<float> MatrixFloat;

}

#endif // MATRIXFLOAT_H
