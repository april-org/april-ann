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
#include <cmath>
#include <cstdio>

template class ComplexFAsciiCoder<WriteBufferWrapper>;
template class ComplexFBinaryCoder<WriteBufferWrapper>;
template class ComplexFAsciiCoder<WriteFileWrapper>;
template class ComplexFBinaryCoder<WriteFileWrapper>;

template MatrixComplexF *readMatrixFromStream(constString &,
					      ComplexFAsciiExtractor,
					      ComplexFBinaryExtractor);
template MatrixComplexF *readMatrixFromStream(ReadFileStream &,
					      ComplexFAsciiExtractor,
					      ComplexFBinaryExtractor);
template int writeMatrixToStream(MatrixComplexF *,
				 WriteBufferWrapper &,
				 ComplexFAsciiSizer,
				 ComplexFBinarySizer,
				 ComplexFAsciiCoder<WriteBufferWrapper>,
				 ComplexFBinaryCoder<WriteBufferWrapper>,
				 bool is_ascii);
template int writeMatrixToStream(MatrixComplexF *,
				 WriteFileWrapper &,
				 ComplexFAsciiSizer,
				 ComplexFBinarySizer,
				 ComplexFAsciiCoder<WriteFileWrapper>,
				 ComplexFBinaryCoder<WriteFileWrapper>,
				 bool is_ascii);

void writeMatrixComplexFToFile(MatrixComplexF *mat,
			       const char *filename,
			       bool is_ascii) {
  WriteFileWrapper wrapper(filename);
  writeMatrixToStream(mat, wrapper, ComplexFAsciiSizer(), ComplexFBinarySizer(),
		      ComplexFAsciiCoder<WriteFileWrapper>(),
		      ComplexFBinaryCoder<WriteFileWrapper>(),
		      is_ascii);
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

MatrixComplexF *readMatrixComplexFFromFile(const char *filename) {
  ReadFileStream f(filename);
  return readMatrixFromStream<ReadFileStream,
			      ComplexF>(f, ComplexFAsciiExtractor(),
					ComplexFBinaryExtractor());
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
  new_mat=new MatrixFloat(N+1, dims, mat->getMajorOrder());
  MatrixComplexF::const_iterator orig_it(mat->begin());
  MatrixFloat::iterator dest_it(new_mat->begin());
  while(orig_it != mat->end()) {
    *dest_it = orig_it->real();
    ++dest_it;
    *dest_it = orig_it->img();
    ++dest_it;
    ++orig_it;
  }
  delete[] dims;
  return new_mat;
}

void applyConjugateInPlace(MatrixComplexF *mat) {
  for (MatrixComplexF::iterator it(mat->begin());
       it != mat->end(); ++it) {
    it->conj();
  }
}
