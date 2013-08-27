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

#include "utilMatrixInt32.h"
#include "binarizer.h"
#include "clamp.h"
#include "matrixFloat.h"
#include "buffered_gzfile.h"
#include "buffered_file.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

void writeMatrixInt32ToFile(MatrixInt32 *mat,
			    const char *filename,
			    bool is_ascii) {
  if (GZFileWrapper::isGZ(filename)) {
    BufferedGZFile f(filename, "w");
    writeMatrixToStream(mat, f, Int32AsciiSizer(), Int32BinarySizer(),
			Int32AsciiCoder<BufferedGZFile>(),
			Int32BinaryCoder<BufferedGZFile>(),
			is_ascii);
  }
  else {
    BufferedFile f(filename, "w");
    writeMatrixToStream(mat, f, Int32AsciiSizer(), Int32BinarySizer(),
			Int32AsciiCoder<BufferedFile>(),
			Int32BinaryCoder<BufferedFile>(),
			is_ascii);
  }
}

char *writeMatrixInt32ToString(MatrixInt32 *mat,
			       bool is_ascii,
			       int &len) {
  WriteBufferWrapper wrapper;
  len = writeMatrixToStream(mat, wrapper,
			    Int32AsciiSizer(),
			    Int32BinarySizer(),
			    Int32AsciiCoder<WriteBufferWrapper>(),
			    Int32BinaryCoder<WriteBufferWrapper>(),
			    is_ascii);
  return wrapper.getBufferProperty();
}

MatrixInt32 *readMatrixInt32FromFile(const char *filename) {
  if (GZFileWrapper::isGZ(filename)) {
    BufferedGZFile f(filename, "r");
    return readMatrixFromStream<BufferedGZFile,
				int32_t>(f, Int32AsciiExtractor(),
					 Int32BinaryExtractor());
  }
  else {
    BufferedFile f(filename, "r");
    return readMatrixFromStream<BufferedFile,
				int32_t>(f, Int32AsciiExtractor(),
					 Int32BinaryExtractor());
  }
}

MatrixInt32 *readMatrixInt32FromString(constString &cs) {
  return readMatrixFromStream<constString,
			      int32_t>(cs,
				       Int32AsciiExtractor(),
				       Int32BinaryExtractor());
}

MatrixFloat *convertFromMatrixInt32ToMatrixFloat(MatrixInt32 *mat,
						 bool col_major) {
  MatrixFloat *new_mat=new MatrixFloat(mat->getNumDim(),
				       mat->getDimPtr(),
				       (col_major)?CblasColMajor:CblasRowMajor);
#ifdef USE_CUDA
  new_mat->setUseCuda(mat->getCudaFlag());
#endif
  MatrixInt32::const_iterator orig_it(mat->begin());
  MatrixFloat::iterator dest_it(new_mat->begin());
  while(orig_it != mat->end()) {
    if (abs(*orig_it) >= 16777216)
      ERROR_PRINT("The integer part can't be represented "
		  "using float precision\n");
    *dest_it = static_cast<float>(*orig_it);
    ++orig_it;
    ++dest_it;
  }
  return new_mat;
}
