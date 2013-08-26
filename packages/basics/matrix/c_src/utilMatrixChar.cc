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

#include "utilMatrixChar.h"
#include "binarizer.h"
#include "clamp.h"
#include "matrixFloat.h"
#include "gzfile_wrapper.h"
#include <cmath>
#include <cstdio>

void writeMatrixCharToFile(MatrixChar *mat,
			   const char *filename) {
  if (GZFileWrapper::isGZ(filename)) {
    GZFileWrapper f(filename, "w");
    writeMatrixToStream(mat, f, CharAsciiSizer(), CharBinarySizer(),
			CharAsciiCoder<GZFileWrapper>(),
			CharBinaryCoder<GZFileWrapper>(),
			true);
  }
  else {
    WriteFileWrapper wrapper(filename);
    writeMatrixToStream(mat, wrapper, CharAsciiSizer(), CharBinarySizer(),
			CharAsciiCoder<WriteFileWrapper>(),
			CharBinaryCoder<WriteFileWrapper>(),
			true);
  }
}

char *writeMatrixCharToString(MatrixChar *mat,
			      int &len) {
  WriteBufferWrapper wrapper;
  len = writeMatrixToStream(mat, wrapper,
			    CharAsciiSizer(),
			    CharBinarySizer(),
			    CharAsciiCoder<WriteBufferWrapper>(),
			    CharBinaryCoder<WriteBufferWrapper>(),
			    true);
  return wrapper.getBufferProperty();
}

MatrixChar *readMatrixCharFromFile(const char *filename) {
  if (GZFileWrapper::isGZ(filename)) {
    GZFileWrapper f(filename, "r");
    return readMatrixFromStream<GZFileWrapper,
				char>(f, CharAsciiExtractor(),
				      CharBinaryExtractor());
  }
  else {
    ReadFileStream f(filename);
    return readMatrixFromStream<ReadFileStream,
				char>(f, CharAsciiExtractor(),
				      CharBinaryExtractor());
  }
}

MatrixChar *readMatrixCharFromString(constString &cs) {
  return readMatrixFromStream<constString,
			      char>(cs,
				    CharAsciiExtractor(),
				    CharBinaryExtractor());
}
