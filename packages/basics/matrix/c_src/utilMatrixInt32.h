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
#ifndef UTILMATRIXINT32_H
#define UTILMATRIXINT32_H

#include "constString.h"
#include "matrixInt32.h"
#include "utilMatrixIO.h"
#include "utilMatrixFloat.h"

/// A functor which reads from a constString (in ascii format) and extracts a
/// Int32
struct Int32AsciiExtractor {
  // returns true if success, false otherwise
  bool operator()(constString &line, int32_t &destination) {
    if (!line.extract_int(&destination)) return false;
    return true;
  }
};

/// A functor which reads from a constString (in binary format) and extracts a
/// Int32
struct Int32BinaryExtractor {
  // returns true if success, false otherwise
  bool operator()(constString &line, int32_t &destination) {
    if (!line.extract_int32_binary(&destination)) return false;
    return true;
  }
};

/// A functor which receives a MatrixInt32 and computes the number of bytes
/// needed to store it using ascii format
struct Int32AsciiSizer {
  // returns the number of bytes needed for all matrix data (plus spaces)
  int operator()(const Matrix<int32_t> *mat) {
    return mat->size()*12;
  }
};

/// A functor which receives a MatrixInt32 and computes the number of bytes
/// needed to store it using binary format
struct Int32BinarySizer {
  // returns the number of bytes needed for all matrix data (plus spaces)
  int operator()(const Matrix<int32_t> *mat) {
    return binarizer::buffer_size_32(mat->size());
  }
};

/// A functor which receives a Int32 and a STREAM and stores the complex
/// number in the given stream (in ascii format)
template<typename StreamType>
struct Int32AsciiCoder {
  // puts to the stream the given value
  void operator()(const int32_t &value, StreamType &stream) {
    stream.printf("%d", value);
  }
};

/// A functor which receives a Int32 and a STREAM and stores the complex
/// number in the given stream (in binary format)
template<typename StreamType>
struct Int32BinaryCoder {
  // puts to the stream the given value
  void operator()(const int32_t &value, StreamType &stream) {
    char b[5];
    binarizer::code_int32(value, b);
    stream.printf("%c%c%c%c%c", b[0],b[1],b[2],b[3],b[4]);
  }
};

// Auxiliary functions to read and write from strings and files. They
// instantiate the templates (readMatrixFromStream, writeMatrixToStream) using
// the correct functors.

void writeMatrixInt32ToFile(MatrixInt32 *mat, const char *filename,
			    bool is_ascii);
char *writeMatrixInt32ToString(MatrixInt32 *mat, bool is_ascii, int &len);
void writeMatrixInt32ToLuaString(MatrixInt32 *mat, lua_State *L, bool is_ascii);
MatrixInt32 *readMatrixInt32FromFile(const char *filename);
MatrixInt32 *readMatrixInt32FromString(constString &cs);

MatrixFloat *convertFromMatrixInt32ToMatrixFloat(MatrixInt32 *mat,
						 bool col_major);

#endif // UTILMATRIXINT32_H
