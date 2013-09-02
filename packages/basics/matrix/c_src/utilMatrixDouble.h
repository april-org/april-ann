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
#ifndef UTILMATRIXDOUBLE_H
#define UTILMATRIXDOUBLE_H

#include "constString.h"
#include "matrixDouble.h"
#include "utilMatrixIO.h"
#include "utilMatrixFloat.h"

/// A functor which reads from a constString (in ascii format) and extracts a
/// Double
struct DoubleAsciiExtractor {
  // returns true if success, false otherwise
  bool operator()(constString &line, double &destination) {
    if (!line.extract_double(&destination)) return false;
    return true;
  }
};

/// A functor which reads from a constString (in binary format) and extracts a
/// Double
struct DoubleBinaryExtractor {
  // returns true if success, false otherwise
  bool operator()(constString &line, double &destination) {
    if (!line.extract_double_binary(&destination)) return false;
    return true;
  }
};

/// A functor which receives a MatrixDouble and computes the number of bytes
/// needed to store it using ascii format
struct DoubleAsciiSizer {
  // returns the number of bytes needed for all matrix data (plus spaces)
  int operator()(const Matrix<double> *mat) {
    return mat->size()*12;
  }
};

/// A functor which receives a MatrixDouble and computes the number of bytes
/// needed to store it using binary format
struct DoubleBinarySizer {
  // returns the number of bytes needed for all matrix data (plus spaces)
  int operator()(const Matrix<double> *mat) {
    return binarizer::buffer_size_64(mat->size());
  }
};

/// A functor which receives a Double and a STREAM and stores the complex
/// number in the given stream (in ascii format)
template<typename StreamType>
struct DoubleAsciiCoder {
  // puts to the stream the given value
  void operator()(const double &value, StreamType &stream) {
    stream.printf("%.5g", value);
  }
};

/// A functor which receives a Double and a STREAM and stores the complex
/// number in the given stream (in binary format)
template<typename StreamType>
struct DoubleBinaryCoder {
  // puts to the stream the given value
  void operator()(const double &value, StreamType &stream) {
    char b[10];
    binarizer::code_double(value, b);
    stream.write(b, sizeof(char)*10);
  }
};

// Auxiliary functions to read and write from strings and files. They
// instantiate the templates (readMatrixFromStream, writeMatrixToStream) using
// the correct functors.

void writeMatrixDoubleToFile(MatrixDouble *mat, const char *filename,
			     bool is_ascii);
char *writeMatrixDoubleToString(MatrixDouble *mat, bool is_ascii, int &len);
void writeMatrixDoubleToLuaString(MatrixDouble *mat, lua_State *L, bool is_ascii);
MatrixDouble *readMatrixDoubleFromFile(const char *filename);
MatrixDouble *readMatrixDoubleFromString(constString &cs);

MatrixFloat *convertFromMatrixDoubleToMatrixFloat(MatrixDouble *mat,
						  bool col_major);

#endif // UTILMATRIXDOUBLE_H
