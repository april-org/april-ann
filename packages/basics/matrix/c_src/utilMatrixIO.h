/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Salvador España-Boquera, Francisco Zamora-Martinez
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
#ifndef UTILMATRIXIO_H
#define UTILMATRIXIO_H
#include "read_file_stream.h"
#include "binarizer.h"
#include "error_print.h"
#include <cmath>
#include <cstdio>

class WriteFileWrapper {
  int total_bytes;
  FILE *f;
public:
  WriteFileWrapper(const char *path) :
  total_bytes(0),
  f(fopen(path, "w")) {
    if (f == 0) ERROR_EXIT1(256, "Unable to open file %s\n", path);
  }
  WriteFileWrapper(FILE *f) : f(f) { }
  ~WriteFileWrapper() { fclose(f); }
  void printf(const char *format, ...) {
    va_list args;
    va_start(args, format);
    total_bytes += vfprintf(f, format, args);
    va_end(args);
  }
  void setExpectedSize(int sz) { }
  int getTotalBytes() const { return total_bytes; }
};

class WriteBufferWrapper {
  char *buffer;
  char *pos;
public:
  WriteBufferWrapper() : buffer(0) { }
  ~WriteBufferWrapper() { delete[] buffer; }
  void printf(const char *format, ...) {
    va_list args;
    va_start(args, format);
    pos += vsprintf(pos, format, args);
    va_end(args);
  }
  void setExpectedSize(int sz) {
    buffer = new char[sz];
    pos = buffer;
  }
  int getTotalBytes() const { return pos - buffer; }
  char *getBufferProperty() { char *aux = buffer; buffer = 0; return aux; }
};


/*** The ASCII or BINARY extractor are like this functor struct:
struct DummyAsciiExtractor {
  // returns true if success, false otherwise
  bool operator()(constString &line, MatrixType &destination) {
    READ FROM LINE OR RETURN FALSE;
    WRITE AT DESTINATION OR RETURN FALSE;
    RETURN TRUE;
  }
};
struct DummyBinaryExtractor {
  // returns true if success, false otherwise
  bool operator()(constString &line, MatrixType &destination) {
    READ FROM LINE OR RETURN FALSE;
    WRITE AT DESTINATION OR RETURN FALSE;
    RETURN TRUE;
  }
};
****************************************************************/

template <typename StreamType, typename MatrixType,
	  typename AsciiExtractFunctor,  typename BinaryExtractorFunctor>
Matrix<MatrixType>*
readMatrixFromStream(StreamType &stream,
		     AsciiExtractFunctor ascii_extractor,
		     BinaryExtractorFunctor bin_extractor) {
  
  constString line,format,order,token;
  // First we read the matrix dimensions
  line = stream.extract_u_line();
  if (!line) {
    fprintf(stderr, "empty file!!!\n");
    return 0;
  }
  static const int maxdim=100;
  int dims[maxdim];
  int n=0, pos_comodin=-1;
  while (n<maxdim && (token = line.extract_token())) {
    if (token == "*") {
      if (pos_comodin != -1) {
	// Error, more than one comodin
	fprintf(stderr,"more than one '*' reading a matrix\n");
	return 0;
      }
      pos_comodin = n;
    } else if (!token.extract_int(&dims[n])) {
      fprintf(stderr,"incorrect dimension %d type, expected a integer\n", n);
      return 0;
    }
    n++;
  }
  if (n==maxdim) {
    fprintf(stderr,"number of dimensions overflow\n");
    return 0; // Maximum allocation problem
  }
  Matrix<MatrixType> *mat = 0;
  // Now we read the type of the format
  line = stream.extract_u_line();
  format = line.extract_token();
  if (!format) {
    fprintf(stderr,"impossible to read format token\n");
    return 0;
  }
  order = line.extract_token();
  if (pos_comodin == -1) { // Normal version
    if (!order || order=="row_major")
      mat = new Matrix<MatrixType>(n,dims);
    else if (order == "col_major")
      mat = new Matrix<MatrixType>(n,dims,CblasColMajor);
    int i=0;
    typename Matrix<MatrixType>::iterator data_it(mat->begin());
    if (format == "ascii") {
      while (data_it!=mat->end() && (line=stream.extract_u_line()))
	while (data_it!=mat->end() &&
	       ascii_extractor(line, *data_it)) { ++data_it; }
    } else { // binary
      while (data_it!=mat->end() && (line=stream.extract_u_line()))
	while (data_it!=mat->end() &&
	       bin_extractor(line, *data_it)) { ++data_it; }
    }
    if (data_it!=mat->end()) {
      ERROR_PRINT("Impossible to fill all the matrix components\n");
      delete mat; mat = 0;
    }
  } else { // version with comodin
    int size=0,maxsize=4096;
    MatrixType *data = new MatrixType[maxsize];
    if (format == "ascii") {
      while ( (line=stream.extract_u_line()) )
	while (ascii_extractor(line, data[size])) { 
	  size++; 
	  if (size == maxsize) { // resize data vector
	    MatrixType *aux = new MatrixType[2*maxsize];
	    for (int a=0;a<maxsize;a++)
	      aux[a] = data[a];
	    maxsize *= 2;
	    delete[] data; data = aux;
	  }
	}
    } else { // binary
      while ( (line=stream.extract_u_line()) )
	while (bin_extractor(line, data[size])) {
	  size++; 
	  if (size == maxsize) { // resize data vector
	    MatrixType *aux = new MatrixType[2*maxsize];
	    for (int a=0;a<maxsize;a++)
	      aux[a] = data[a];
	    maxsize *= 2;
	    delete[] data; data = aux;
	  }
	}
    }
    int sizesincomodin = 1;
    for (int i=0;i<n;i++)
      if (i != pos_comodin)
	sizesincomodin *= dims[i];
    if ((size % sizesincomodin) != 0) {
      // Error: The size of the data does not coincide
      fprintf(stderr,"data size is not valid reading a matrix with '*'\n");
      delete[] data; return 0;
    }
    dims[pos_comodin] = size / sizesincomodin;
    if (!order || order == "row_major")
      mat = new Matrix<MatrixType>(n,dims);
    else if (order == "col_major")
      mat = new Matrix<MatrixType>(n,dims,CblasColMajor);
    int i=0;
    for (typename Matrix<MatrixType>::iterator it(mat->begin());
	 it!=mat->end();
	 ++it,++i)
      *it = data[i];
    delete[] data;
  }
  return mat;
}

/*** Functor examples

struct DummyAsciiSizer {
  // returns the number of bytes needed for all matrix data (plus spaces)
  int operator()(const Matrix<MatrixType> *mat) {
    RETURN 12 * mat->size();
  }
};
struct DummyBinarySizer {
  // returns the number of bytes needed for all matrix data (plus spaces)
  int operator()(const Matrix<MatrixType> *mat) {
    RETURN binarizer::buffer_size_32(mat->size());
  }
};
template<typename StreamType>
struct DummyAsciiCoder {
  // puts to the stream the given value
  void operator()(const MatrixType &value, StreamType &stream) {
    stream.printf("%.5g", value);
  }
};
template<typename StreamType>
struct DummyBinaryCoder {
  // puts to the stream the given value
  void operator()(const MatrixType &value, StreamType &stream) {
    char b[5];
    binarizer::code_BLAH(value, b);
    stream.printf("%s", b);
  }
};
****************************************************************/

// Returns the number of chars written (there is a '\0' that is not counted)
template <typename StreamType, typename MatrixType,
	  typename AsciiSizeFunctor,  typename BinarySizeFunctor,
	  typename AsciiCodeFunctor,  typename BinaryCodeFunctor>
int writeMatrixToStream(Matrix<MatrixType> *mat,
			StreamType &stream,
			AsciiSizeFunctor ascii_sizer,
			BinarySizeFunctor bin_sizer,
			AsciiCodeFunctor ascii_coder,
			BinaryCodeFunctor bin_coder,
			bool is_ascii) {
  int sizedata,sizeheader;
  sizeheader = mat->getNumDim()*10+10+10; // FIXME: To put adequate values
  // sizedata contains the memory used by MatrixType in ascii including spaces,
  // new lines, etc...
  if (is_ascii) sizedata = ascii_sizer(mat);
  else sizedata = bin_sizer(mat);
  stream.setExpectedSize(sizedata+sizeheader+1);
  for (int i=0;i<mat->getNumDim()-1;i++) stream.printf("%d ",mat->getDimSize(i));
  stream.printf("%d\n",mat->getDimSize(mat->getNumDim()-1));
  if (is_ascii) {
    const int columns = 9;
    stream.printf("ascii");
    if (mat->getMajorOrder() == CblasColMajor)
      stream.printf(" col_major");
    else
      stream.printf(" row_major");
    stream.printf("\n");
    int i=0;
    for(typename Matrix<MatrixType>::const_iterator it(mat->begin());
	it!=mat->end();++it,++i) {
      ascii_coder(*it, stream);
      stream.printf("%c", ((((i+1) % columns) == 0) ? '\n' : ' '));
    }
    if ((i % columns) != 0) {
      stream.printf("\n"); 
    }
  } else { // binary
    const int columns = 16;
    stream.printf("binary");
    if (mat->getMajorOrder() == CblasColMajor)
      stream.printf(" col_major");
    else
      stream.printf(" row_major");
    stream.printf("\n");
    // We substract 1 so the final '\0' is not considered
    int i=0;
    for(typename Matrix<MatrixType>::const_iterator it(mat->begin());
	it!=mat->end();
	++it,++i) {
      bin_coder(*it, stream);
      /*
	char b[5];
	binarizer::code_float(*it, b);
	fprintf(f, "%c%c%c%c%c", b[0], b[1], b[2], b[3], b[4]);
      */
      if ((i+1) % columns == 0) stream.printf("\n");
    }
    if ((i % columns) != 0) stream.printf("\n"); 
  }
  return stream.getTotalBytes();
}

#endif // UTILMATRIXIO_H
