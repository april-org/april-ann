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
#ifndef UTILMATRIXCHAR_H
#define UTILMATRIXCHAR_H

#include "constString.h"
#include "matrixChar.h"
#include "utilMatrixIO.h"
#include "utilMatrixFloat.h"

/// A functor which reads from a constString (in ascii format) and extracts a
/// Char
struct CharAsciiExtractor {
  // returns true if success, false otherwise
  bool operator()(constString &line, char &destination) {
    if (!line.extract_char(&destination)) return false;
    return true;
  }
};

/// A functor which reads from a constString (in binary format) and extracts a
/// Char
struct CharBinaryExtractor {
  // returns true if success, false otherwise
  bool operator()(constString &line, char &destination) {
    ERROR_EXIT(128, "Char type has not binary option\n");
    return false;
  }
};

/// A functor which receives a MatrixChar and computes the number of bytes
/// needed to store it using ascii format
struct CharAsciiSizer {
  // returns the number of bytes needed for all matrix data (plus spaces)
  int operator()(const Matrix<char> *mat) {
    return mat->size()*2;
  }
};

/// A functor which receives a MatrixChar and computes the number of bytes
/// needed to store it using binary format
struct CharBinarySizer {
  // returns the number of bytes needed for all matrix data (plus spaces)
  int operator()(const Matrix<char> *mat) {
    ERROR_EXIT(128, "Char type has not binary option\n");
    return 0;
  }
};

/// A functor which receives a Char and a STREAM and stores the complex
/// number in the given stream (in ascii format)
template<typename StreamType>
struct CharAsciiCoder {
  // puts to the stream the given value
  void operator()(const char &value, StreamType &stream) {
    stream.printf("%c", value);
  }
};

/// A functor which receives a Char and a STREAM and stores the complex
/// number in the given stream (in binary format)
template<typename StreamType>
struct CharBinaryCoder {
  // puts to the stream the given value
  void operator()(const char &value, StreamType &stream) {
    ERROR_EXIT(128, "Char type has not binary option\n");
  }
};

// Auxiliary functions to read and write from strings and files. They
// instantiate the templates (readMatrixFromStream, writeMatrixToStream) using
// the correct functors.

void writeMatrixCharToFile(MatrixChar *mat, const char *filename);
char *writeMatrixCharToString(MatrixChar *mat, int &len);
void writeMatrixCharToLuaString(MatrixChar *mat, lua_State *L, bool is_ascii);
MatrixChar *readMatrixCharFromFile(const char *filename);
MatrixChar *readMatrixCharFromString(constString &cs);

#endif // UTILMATRIXCHAR_H
