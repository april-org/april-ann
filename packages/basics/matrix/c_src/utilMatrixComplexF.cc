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
#include <cstdio>

#include "binarizer.h"
#include "clamp.h"
#include "complex_number.h"
#include "ignore_result.h"
#include "matrixFloat.h"
#include "utilMatrixComplexF.h"

using april_math::ComplexF;

namespace basics {
  
  /////////////////////////////////////////////////////////////////////////
  
  template<>
  bool AsciiExtractor<ComplexF>::operator()(april_utils::constString &line,
                                            ComplexF &destination) {
    if (!line.extract_float(&destination.real())) return false;
    if (!line.extract_float(&destination.img())) return false;
    char ch;
    if (!line.extract_char(&ch)) return false;
    if (ch != 'i') return false;
    return true;
  }
  
  template<>
  bool BinaryExtractor<ComplexF>::operator()(april_utils::constString &line,
                                             ComplexF &destination) {
    if (!line.extract_float_binary(&destination.real())) return false;
    if (!line.extract_float_binary(&destination.img())) return false;
    return true;
  }
  
  template<>
  int AsciiSizer<ComplexF>::operator()(const Matrix<ComplexF> *mat) {
    return mat->size()*26; // 12*2+2
  }

  template<>
  int BinarySizer<ComplexF>::operator()(const Matrix<ComplexF> *mat) {
    return april_utils::binarizer::buffer_size_32(mat->size()<<1); // mat->size() * 2

  }

  template<>
  void AsciiCoder<ComplexF>::operator()(const ComplexF &value,
                                        april_io::StreamInterface *stream) {
    stream->printf("%.5g%+.5gi", value.real(), value.img());
  }
  
  template<>
  void BinaryCoder<ComplexF>::operator()(const ComplexF &value,
                                         april_io::StreamInterface *stream) {
    char b[10];
    april_utils::binarizer::code_float(value.real(), b);
    april_utils::binarizer::code_float(value.img(),  b+5);
    stream->put(b, sizeof(char)*10);
  }

  /////////////////////////////////////////////////////////////////////////////

  
  MatrixFloat *convertFromMatrixComplexFToMatrixFloat(MatrixComplexF *mat) {
    MatrixFloat *new_mat;
    int N     = mat->getNumDim();
    int *dims = new int[N+1];
    if (mat->getMajorOrder() == CblasRowMajor) {
      // the real and imaginary part are the last dimension (they are stored
      // together in row major)
      for (int i=0; i<N; ++i) dims[i] = mat->getDimPtr()[i];
      dims[N] = 2;
    }
    else {
      // the real and imaginary part are the first dimension (they are stored
      // together in col major)
      dims[0] = 2;
      for (int i=0; i<N; ++i) dims[i+1] = mat->getDimPtr()[i];
    }
    april_math::FloatGPUMirroredMemoryBlock *new_mat_memory;
    new_mat_memory = mat->getRawDataAccess()->reinterpretAs<float>();
    new_mat=new MatrixFloat(N+1, dims, mat->getMajorOrder(), new_mat_memory);
#ifdef USE_CUDA
    new_mat->setUseCuda(mat->getCudaFlag());
#endif
    delete[] dims;
    return new_mat;
  }

  void applyConjugateInPlace(MatrixComplexF *mat) {
    for (MatrixComplexF::iterator it(mat->begin());
         it != mat->end(); ++it) {
      it->conj();
    }
  }

  MatrixFloat *realPartFromMatrixComplexFToMatrixFloat(MatrixComplexF *mat) {
    MatrixFloat *new_mat = new MatrixFloat(mat->getNumDim(),
                                           mat->getDimPtr(),
                                           mat->getMajorOrder());
#ifdef USE_CUDA
    new_mat->setUseCuda(mat->getCudaFlag());
#endif
    MatrixComplexF::const_iterator orig_it(mat->begin());
    MatrixFloat::iterator dest_it(new_mat->begin());
    while(orig_it != mat->end()) {
      *dest_it = orig_it->real();
      ++orig_it;
      ++dest_it;
    }
    return new_mat;
  }

  MatrixFloat *imgPartFromMatrixComplexFToMatrixFloat(MatrixComplexF *mat) {
    MatrixFloat *new_mat = new MatrixFloat(mat->getNumDim(),
                                           mat->getDimPtr(),
                                           mat->getMajorOrder());
#ifdef USE_CUDA
    new_mat->setUseCuda(mat->getCudaFlag());
#endif
    MatrixComplexF::const_iterator orig_it(mat->begin());
    MatrixFloat::iterator dest_it(new_mat->begin());
    while(orig_it != mat->end()) {
      *dest_it = orig_it->img();
      ++orig_it;
      ++dest_it;
    }
    return new_mat;
  }

  MatrixFloat *absFromMatrixComplexFToMatrixFloat(MatrixComplexF *mat) {
    MatrixFloat *new_mat = new MatrixFloat(mat->getNumDim(),
                                           mat->getDimPtr(),
                                           mat->getMajorOrder());
#ifdef USE_CUDA
    new_mat->setUseCuda(mat->getCudaFlag());
#endif
    MatrixComplexF::const_iterator orig_it(mat->begin());
    MatrixFloat::iterator dest_it(new_mat->begin());
    while(orig_it != mat->end()) {
      *dest_it = orig_it->abs();
      ++orig_it;
      ++dest_it;
    }
    return new_mat;
  }

  MatrixFloat *angleFromMatrixComplexFToMatrixFloat(MatrixComplexF *mat) {
    MatrixFloat *new_mat = new MatrixFloat(mat->getNumDim(),
                                           mat->getDimPtr(),
                                           mat->getMajorOrder());
#ifdef USE_CUDA
    new_mat->setUseCuda(mat->getCudaFlag());
#endif
    MatrixComplexF::const_iterator orig_it(mat->begin());
    MatrixFloat::iterator dest_it(new_mat->begin());
    while(orig_it != mat->end()) {
      *dest_it = orig_it->angle();
      ++orig_it;
      ++dest_it;
    }
    return new_mat;
  }

} // namespace basics
