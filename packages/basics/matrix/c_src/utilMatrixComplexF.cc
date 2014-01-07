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

#include "utilMatrixComplexF.h"
#include "binarizer.h"
#include "clamp.h"
#include "matrixFloat.h"
#include "buffered_gzfile.h"
#include "buffered_file.h"
#include "ignore_result.h"
#include <cmath>
#include <cstdio>

void writeMatrixComplexFToFile(MatrixComplexF *mat,
			       const char *filename,
			       bool is_ascii) {
  if (GZFileWrapper::isGZ(filename)) {
    BufferedGZFile f(filename, "w");
    writeMatrixToStream(mat, f, ComplexFAsciiSizer(), ComplexFBinarySizer(),
			ComplexFAsciiCoder<BufferedGZFile>(),
			ComplexFBinaryCoder<BufferedGZFile>(),
			is_ascii);
  }
  else {
    BufferedFile f(filename, "w");
    writeMatrixToStream(mat, f, ComplexFAsciiSizer(), ComplexFBinarySizer(),
			ComplexFAsciiCoder<BufferedFile>(),
			ComplexFBinaryCoder<BufferedFile>(),
			is_ascii);
  }
}

char *writeMatrixComplexFToString(MatrixComplexF *mat,
				  bool is_ascii,
				  int &len) {
  WriteBufferWrapper wrapper;
  len = writeMatrixToStream(mat, wrapper,
			    ComplexFAsciiSizer(),
			    ComplexFBinarySizer(),
			    ComplexFAsciiCoder<WriteBufferWrapper>(),
			    ComplexFBinaryCoder<WriteBufferWrapper>(),
			    is_ascii);
  return wrapper.getBufferProperty();
}

void writeMatrixComplexFToLuaString(MatrixComplexF *mat,
				    lua_State *L,
				    bool is_ascii) {
  WriteLuaBufferWrapper wrapper(L);
  IGNORE_RESULT(writeMatrixToStream(mat, wrapper,
				    ComplexFAsciiSizer(),
				    ComplexFBinarySizer(),
				    ComplexFAsciiCoder<WriteLuaBufferWrapper>(),
				    ComplexFBinaryCoder<WriteLuaBufferWrapper>(),
				    is_ascii));
  wrapper.finish();
}

MatrixComplexF *readMatrixComplexFFromFile(const char *filename) {
  if (GZFileWrapper::isGZ(filename)) {
    BufferedGZFile f(filename, "r");
    return readMatrixFromStream<BufferedGZFile,
      ComplexF>(f, ComplexFAsciiExtractor(),
		ComplexFBinaryExtractor());
  }
  else {
    BufferedFile f(filename, "r");
    return readMatrixFromStream<BufferedFile,
      ComplexF>(f, ComplexFAsciiExtractor(),
		ComplexFBinaryExtractor());
  }
}

MatrixComplexF *readMatrixComplexFFromString(constString &cs) {
  return readMatrixFromStream<constString,
    ComplexF>(cs,
	      ComplexFAsciiExtractor(),
	      ComplexFBinaryExtractor());
}

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
  FloatGPUMirroredMemoryBlock *new_mat_memory;
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
