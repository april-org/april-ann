/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera
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
#ifndef UTILMATRIXFLOAT_H
#define UTILMATRIXFLOAT_H

#include "buffered_gzfile.h"
#include "constString.h"
#include "matrixFloat.h"
#include "sparse_matrixFloat.h"
#include "utilMatrixIO.h"

struct FloatAsciiExtractor {
  // returns true if success, false otherwise
  bool operator()(constString &line, float &destination) {
    if (!line.extract_float(&destination)) return false;
    return true;
  }
};

struct FloatBinaryExtractor {
  // returns true if success, false otherwise
  bool operator()(constString &line, float &destination) {
    if (!line.extract_float_binary(&destination)) return false;
    return true;
  }
};

struct FloatAsciiSizer {
  // returns the number of bytes needed for all matrix data (plus spaces)
  int operator()(const Matrix<float> *mat) {
    return mat->size()*12;
  }
};
struct FloatBinarySizer {
  // returns the number of bytes needed for all matrix data (plus spaces)
  int operator()(const Matrix<float> *mat) {
    return binarizer::buffer_size_32(mat->size());
  }
};
struct SparseFloatAsciiSizer {
  // returns the number of bytes needed for all matrix data (plus spaces)
  int operator()(const SparseMatrix<float> *mat) {
    return mat->nonZeroSize()*12;
  }
};
struct SparseFloatBinarySizer {
  // returns the number of bytes needed for all matrix data (plus spaces)
  int operator()(const SparseMatrix<float> *mat) {
    return binarizer::buffer_size_32(mat->nonZeroSize());
  }
};
template<typename StreamType>
struct FloatAsciiCoder {
  // puts to the stream the given value
  void operator()(const float &value, StreamType &stream) {
    stream.printf("%.5g", value);
  }
};
template<typename StreamType>
struct FloatBinaryCoder {
  // puts to the stream the given value
  void operator()(const float &value, StreamType &stream) {
    char b[5];
    binarizer::code_float(value, b);
    stream.printf("%c%c%c%c%c", b[0],b[1],b[2],b[3],b[4]);
  }
};

MatrixFloat* readMatrixFloatHEX(int width, int height, constString cs);

const float CTENEGRO  = 1.0f;
const float CTEBLANCO = 0.0f;
MatrixFloat* readMatrixFloatPNM(constString cs,
				bool forcecolor=false, 
				bool forcegray=false);

int saveMatrixFloatPNM(MatrixFloat *mat,
		       char **buffer);
int saveMatrixFloatHEX(MatrixFloat *mat,
		       char **buffer,
		       int *width,
		       int *height);

void writeMatrixFloatToFile(MatrixFloat *mat, const char *filename,
			    bool is_ascii);
char *writeMatrixFloatToString(MatrixFloat *mat, bool is_ascii, int &len);
void writeMatrixFloatToLuaString(MatrixFloat *mat, lua_State *L, bool is_ascii);
MatrixFloat *readMatrixFloatFromFile(const char *filename, const char *order);
MatrixFloat *readMatrixFloatFromString(constString &cs);

void writeMatrixFloatToTabFile(MatrixFloat *mat, const char *filename);
MatrixFloat *readMatrixFloatFromTabFile(const char *filename,
					const char *order = "row_major");

void writeMatrixFloatToTabGZStream(MatrixFloat *mat, BufferedGZFile *stream);
void writeMatrixFloatToTabStream(MatrixFloat *mat, FILE *f);

// SPARSE MATRIX FLOAT

char *writeSparseMatrixFloatToString(SparseMatrixFloat *mat, bool is_ascii,
				     int &len);
void writeSparseMatrixFloatToLuaString(SparseMatrixFloat *mat, lua_State *L,
				       bool is_ascii);
SparseMatrixFloat *readSparseMatrixFloatFromString(constString &cs);
#endif // UTILMATRIXFLOAT_H
