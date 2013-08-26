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

#include "utilMatrixDouble.h"
#include "binarizer.h"
#include "clamp.h"
#include "matrixFloat.h"
#include "gzfile_wrapper.h"
#include <cmath>
#include <cstdio>

void writeMatrixDoubleToFile(MatrixDouble *mat,
			     const char *filename,
			     bool is_ascii) {
  if (GZFileWrapper::isGZ(filename)) {
    GZFileWrapper f(filename, "r");
    writeMatrixToStream(mat, f, DoubleAsciiSizer(), DoubleBinarySizer(),
			DoubleAsciiCoder<GZFileWrapper>(),
			DoubleBinaryCoder<GZFileWrapper>(),
			is_ascii);
  }
  else {
    WriteFileWrapper wrapper(filename);
    writeMatrixToStream(mat, wrapper, DoubleAsciiSizer(), DoubleBinarySizer(),
			DoubleAsciiCoder<WriteFileWrapper>(),
			DoubleBinaryCoder<WriteFileWrapper>(),
			is_ascii);
  }
}

char *writeMatrixDoubleToString(MatrixDouble *mat,
				bool is_ascii,
				int &len) {
  WriteBufferWrapper wrapper;
  len = writeMatrixToStream(mat, wrapper,
			    DoubleAsciiSizer(),
			    DoubleBinarySizer(),
			    DoubleAsciiCoder<WriteBufferWrapper>(),
			    DoubleBinaryCoder<WriteBufferWrapper>(),
			    is_ascii);
  return wrapper.getBufferProperty();
}

MatrixDouble *readMatrixDoubleFromFile(const char *filename) {
  if (GZFileWrapper::isGZ(filename)) {
    GZFileWrapper f(filename, "r");
    return readMatrixFromStream<GZFileWrapper,
				double>(f, DoubleAsciiExtractor(),
					DoubleBinaryExtractor());
  }
  else {
    ReadFileStream f(filename);
    return readMatrixFromStream<ReadFileStream,
				double>(f, DoubleAsciiExtractor(),
					DoubleBinaryExtractor());
  }
}

MatrixDouble *readMatrixDoubleFromString(constString &cs) {
  return readMatrixFromStream<constString,
			      double>(cs,
				      DoubleAsciiExtractor(),
				      DoubleBinaryExtractor());
}

MatrixFloat *convertFromMatrixDoubleToMatrixFloat(MatrixDouble *mat,
						  bool col_major) {
  MatrixFloat *new_mat=new MatrixFloat(mat->getNumDim(),
				       mat->getDimPtr(),
				       (col_major)?CblasColMajor:CblasRowMajor);
#ifdef USE_CUDA
  new_mat->setUseCuda(mat->getCudaFlag());
#endif
  MatrixDouble::const_iterator orig_it(mat->begin());
  MatrixFloat::iterator dest_it(new_mat->begin());
  while(orig_it != mat->end()) {
    if (fabs(*orig_it) >= 16777216.0)
      ERROR_PRINT("The integer part can't be represented "
		  "using float precision\n");
    *dest_it = static_cast<float>(*orig_it);
    ++orig_it;
    ++dest_it;
  }
return new_mat;
}
