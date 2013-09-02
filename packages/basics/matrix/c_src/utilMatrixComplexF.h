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
#ifndef UTILMATRIXCOMPLEXF_H
#define UTILMATRIXCOMPLEXF_H

#include "constString.h"
#include "matrixComplexF.h"
#include "utilMatrixIO.h"
#include "utilMatrixFloat.h"

/// A functor which reads from a constString (in ascii format) and extracts a
/// ComplexF
struct ComplexFAsciiExtractor {
  // returns true if success, false otherwise
  bool operator()(constString &line, ComplexF &destination) {
    if (!line.extract_float(&destination.real())) return false;
    if (!line.extract_float(&destination.img())) return false;
    char ch;
    if (!line.extract_char(&ch)) return false;
    if (ch != 'i') return false;
    return true;
  }
};

/// A functor which reads from a constString (in binary format) and extracts a
/// ComplexF
struct ComplexFBinaryExtractor {
  // returns true if success, false otherwise
  bool operator()(constString &line, ComplexF &destination) {
    if (!line.extract_float_binary(&destination.real())) return false;
    if (!line.extract_float_binary(&destination.img())) return false;
    return true;
  }
};

/// A functor which receives a MatrixComplexF and computes the number of bytes
/// needed to store it using ascii format
struct ComplexFAsciiSizer {
  // returns the number of bytes needed for all matrix data (plus spaces)
  int operator()(const Matrix<ComplexF> *mat) {
    return mat->size()*26; // 12*2+2
  }
};

/// A functor which receives a MatrixComplexF and computes the number of bytes
/// needed to store it using binary format
struct ComplexFBinarySizer {
  // returns the number of bytes needed for all matrix data (plus spaces)
  int operator()(const Matrix<ComplexF> *mat) {
    return binarizer::buffer_size_32(mat->size()<<1); // mat->size() * 2
  }
};

/// A functor which receives a ComplexF and a STREAM and stores the complex
/// number in the given stream (in ascii format)
template<typename StreamType>
struct ComplexFAsciiCoder {
  // puts to the stream the given value
  void operator()(const ComplexF &value, StreamType &stream) {
    stream.printf("%.5g%+.5gi", value.real(), value.img());
  }
};

/// A functor which receives a ComplexF and a STREAM and stores the complex
/// number in the given stream (in binary format)
template<typename StreamType>
struct ComplexFBinaryCoder {
  // puts to the stream the given value
  void operator()(const ComplexF &value, StreamType &stream) {
    char b[10];
    binarizer::code_float(value.real(), b);
    binarizer::code_float(value.img(),  b+5);
    stream.write(b, sizeof(char)*10);
  }
};

// Auxiliary functions to read and write from strings and files. They
// instantiate the templates (readMatrixFromStream, writeMatrixToStream) using
// the correct functors.

void writeMatrixComplexFToFile(MatrixComplexF *mat, const char *filename,
			       bool is_ascii);
char *writeMatrixComplexFToString(MatrixComplexF *mat, bool is_ascii, int &len);
void writeMatrixComplexFToLuaString(MatrixComplexF *mat, lua_State *L, bool is_ascii);
MatrixComplexF *readMatrixComplexFFromFile(const char *filename);
MatrixComplexF *readMatrixComplexFFromString(constString &cs);

MatrixFloat *convertFromMatrixComplexFToMatrixFloat(MatrixComplexF *mat);
void applyConjugateInPlace(MatrixComplexF *mat);
MatrixFloat *realPartFromMatrixComplexFToMatrixFloat(MatrixComplexF *mat);
MatrixFloat *imgPartFromMatrixComplexFToMatrixFloat(MatrixComplexF *mat);
MatrixFloat *absFromMatrixComplexFToMatrixFloat(MatrixComplexF *mat);
MatrixFloat *angleFromMatrixComplexFToMatrixFloat(MatrixComplexF *mat);

#endif // UTILMATRIXCOMPLEXF_H
