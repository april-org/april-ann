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
#include <cmath>
#include <cstdio>

template class CharAsciiCoder<WriteBufferWrapper>;
template class CharBinaryCoder<WriteBufferWrapper>;
template class CharAsciiCoder<WriteFileWrapper>;
template class CharBinaryCoder<WriteFileWrapper>;

template MatrixChar *readMatrixFromStream(constString &,
					  CharAsciiExtractor,
					  CharBinaryExtractor);
template MatrixChar *readMatrixFromStream(ReadFileStream &,
					  CharAsciiExtractor,
					  CharBinaryExtractor);
template int writeMatrixToStream(MatrixChar *,
				 WriteBufferWrapper &,
				 CharAsciiSizer,
				 CharBinarySizer,
				 CharAsciiCoder<WriteBufferWrapper>,
				 CharBinaryCoder<WriteBufferWrapper>,
				 bool is_ascii);
template int writeMatrixToStream(MatrixChar *,
				 WriteFileWrapper &,
				 CharAsciiSizer,
				 CharBinarySizer,
				 CharAsciiCoder<WriteFileWrapper>,
				 CharBinaryCoder<WriteFileWrapper>,
				 bool is_ascii);

void writeMatrixCharToFile(MatrixChar *mat,
			   const char *filename) {
  WriteFileWrapper wrapper(filename);
  writeMatrixToStream(mat, wrapper, CharAsciiSizer(), CharBinarySizer(),
		      CharAsciiCoder<WriteFileWrapper>(),
		      CharBinaryCoder<WriteFileWrapper>(),
		      true);
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
  ReadFileStream f(filename);
  return readMatrixFromStream<ReadFileStream,
			      char>(f, CharAsciiExtractor(),
				    CharBinaryExtractor());
}

MatrixChar *readMatrixCharFromString(constString &cs) {
  return readMatrixFromStream<constString,
			      char>(cs,
				    CharAsciiExtractor(),
				    CharBinaryExtractor());
}
