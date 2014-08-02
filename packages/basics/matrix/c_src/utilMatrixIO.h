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
#include <cmath>
#include <cstdio>
#include <cstring>

#include "binarizer.h"
#include "constString.h"
#include "error_print.h"
#include "file.h"
#include "matrix.h"

extern "C" {
#include "lauxlib.h"
#include "lualib.h"
#include "lua.h"
}

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

template <typename MatrixType,
	  typename AsciiExtractFunctor,  typename BinaryExtractorFunctor>
Matrix<MatrixType>*
readMatrixFromFileStream(april_io::File *file,
                         AsciiExtractFunctor ascii_extractor,
                         BinaryExtractorFunctor bin_extractor,
                         const char *given_order=0) {
  if (!file->good()) {
    ERROR_PRINT("The stream is not prepared, it is empty, or EOF\n");
    return 0;
  }
  constString line,format,order,token;
  // First we read the matrix dimensions
  line = file->extract_u_line();
  if (!line) {
    ERROR_PRINT("empty file!!!\n");
    return 0;
  }
  static const int maxdim=100;
  int dims[maxdim];
  int n=0, pos_comodin=-1;
  while (n<maxdim && (token = line.extract_token())) {
    if (token == "*") {
      if (pos_comodin != -1) {
	// Error, more than one comodin
	ERROR_PRINT("more than one '*' reading a matrix\n");
	return 0;
      }
      pos_comodin = n;
    } else if (!token.extract_int(&dims[n])) {
      ERROR_PRINT1("incorrect dimension %d type, expected a integer\n", n);
      return 0;
    }
    n++;
  }
  if (n==maxdim) {
    ERROR_PRINT("number of dimensions overflow\n");
    return 0; // Maximum allocation problem
  }
  Matrix<MatrixType> *mat = 0;
  // Now we read the type of the format
  line = file->extract_u_line();
  format = line.extract_token();
  if (!format) {
    ERROR_PRINT("impossible to read format token\n");
    return 0;
  }
  order = line.extract_token();
  if (given_order != 0) order = given_order;
  if (pos_comodin == -1) { // Normal version
    if (!order || order=="row_major")
      mat = new Matrix<MatrixType>(n,dims);
    else if (order == "col_major")
      mat = new Matrix<MatrixType>(n,dims,CblasColMajor);
    else {
      ERROR_PRINT("Impossible to determine the order\n");
      return 0;
    }
    typename Matrix<MatrixType>::iterator data_it(mat->begin());
    if (format == "ascii") {
      while (data_it!=mat->end() && (line=file->extract_u_line()))
	while (data_it!=mat->end() &&
	       ascii_extractor(line, *data_it)) { ++data_it; }
    } else { // binary
      while (data_it!=mat->end() && (line=file->extract_u_line()))
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
      while ( (line=file->extract_u_line()) )
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
      while ( (line=file->extract_u_line()) )
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
      ERROR_PRINT("data size is not valid reading a matrix with '*'\n");
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

     struct DummyAsciiCoder {
     // puts to the stream the given value
     void operator()(const MatrixType &value, april_io::File *file) {
     file->printf("%.5g", value);
     }
     };

     struct DummyBinaryCoder {
     // puts to the stream the given value
     void operator()(const MatrixType &value, april_io::File *file) {
     char b[5];
     binarizer::code_BLAH(value, b);
     file->printf("%s", b);
     }
     };
****************************************************************/

// Returns the number of chars written (there is a '\0' that is not counted)
template <typename MatrixType,
	  typename AsciiSizeFunctor,  typename BinarySizeFunctor,
	  typename AsciiCodeFunctor,  typename BinaryCodeFunctor>
int writeMatrixToFileStream(Matrix<MatrixType> *mat,
                            april_io::File *file,
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
  file->setExpectedSize(sizedata+sizeheader+1);
  if (!file->good()) {
    ERROR_EXIT(256, "The stream is not prepared, it is empty, or EOF\n");
  }
  for (int i=0;i<mat->getNumDim()-1;i++) file->printf("%d ",mat->getDimSize(i));
  file->printf("%d\n",mat->getDimSize(mat->getNumDim()-1));
  if (is_ascii) {
    const int columns = 9;
    file->printf("ascii");
    if (mat->getMajorOrder() == CblasColMajor)
      file->printf(" col_major");
    else
      file->printf(" row_major");
    file->printf("\n");
    int i=0;
    for(typename Matrix<MatrixType>::const_iterator it(mat->begin());
	it!=mat->end();++it,++i) {
      ascii_coder(*it, file);
      file->printf("%c", ((((i+1) % columns) == 0) ? '\n' : ' '));
    }
    if ((i % columns) != 0) {
      file->printf("\n"); 
    }
  } else { // binary
    const int columns = 16;
    file->printf("binary");
    if (mat->getMajorOrder() == CblasColMajor)
      file->printf(" col_major");
    else
      file->printf(" row_major");
    file->printf("\n");
    // We substract 1 so the final '\0' is not considered
    int i=0;
    for(typename Matrix<MatrixType>::const_iterator it(mat->begin());
	it!=mat->end();
	++it,++i) {
      bin_coder(*it, file);
      /*
	char b[5];
	binarizer::code_float(*it, b);
	fprintf(f, "%c%c%c%c%c", b[0], b[1], b[2], b[3], b[4]);
      */
      if ((i+1) % columns == 0) file->printf("\n");
    }
    if ((i % columns) != 0) file->printf("\n"); 
  }
  return file->getTotalBytes();
}

/*** The ASCII extractor are like this functor struct:
     struct DummyAsciiExtractor {
     // returns true if success, false otherwise
     bool operator()(constString &line, MatrixType &destination) {
     READ FROM LINE OR RETURN FALSE;
     WRITE AT DESTINATION OR RETURN FALSE;
     RETURN TRUE;
     }
     };
****************************************************************/

template <typename MatrixType,
	  typename AsciiExtractFunctor>
Matrix<MatrixType>*
readMatrixFromTabFileStream(const int rows, const int cols,
                            april_io::File *file,
                            AsciiExtractFunctor ascii_extractor,
                            const char *given_order) {
  if (!file->good()) {
    ERROR_PRINT("The stream is not prepared, it is empty, or EOF\n");
    return 0;
  }
  constString line,order(given_order),token;
  int dims[2] = { rows, cols };
  Matrix<MatrixType> *mat = 0;
  if (order=="row_major")
    mat = new Matrix<MatrixType>(2,dims);
  else if (order == "col_major")
    mat = new Matrix<MatrixType>(2,dims,CblasColMajor);
  else {
    ERROR_PRINT("Impossible to determine the order\n");
    return 0;
  }
  int i=0;
  typename Matrix<MatrixType>::iterator data_it(mat->begin());
  while (data_it!=mat->end() && (line=file->extract_u_line())) {
    int num_cols_size_count = 0;
    while (data_it!=mat->end() &&
	   ascii_extractor(line, *data_it)) { ++data_it; ++num_cols_size_count; }
    if (num_cols_size_count != cols)
      ERROR_EXIT3(128, "Incorrect number of elements at line %d, "
		  "expected %d, found %d\n", i, cols, num_cols_size_count);
    ++i;
  }
  if (data_it!=mat->end()) {
    ERROR_PRINT("Impossible to fill all the matrix components\n");
    delete mat; mat = 0;
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

     struct DummyAsciiCoder {
     // puts to the stream the given value
     void operator()(const MatrixType &value, april_io::File *file) {
     file->printf("%.5g", value);
     }
     };
****************************************************************/

// Returns the number of chars written (there is a '\0' that is not counted)
template <typename MatrixType,
	  typename AsciiSizeFunctor, typename AsciiCodeFunctor>
int writeMatrixToTabFileStream(Matrix<MatrixType> *mat,
                               april_io::File *file,
                               AsciiSizeFunctor ascii_sizer,
                               AsciiCodeFunctor ascii_coder) {
  if (mat->getNumDim() != 2)
    ERROR_EXIT(128, "Needs a matrix with 2 dimensions");
  
  int sizedata;
  sizedata = ascii_sizer(mat);
  file->setExpectedSize(sizedata+1);
  if (!file->good()) {
    ERROR_EXIT(256, "The stream is not prepared, it is empty, or EOF\n");
  }
  const int columns = mat->getDimSize(1);
  int i=0;
  for(typename Matrix<MatrixType>::const_iterator it(mat->begin());
      it!=mat->end();++it,++i) {
    ascii_coder(*it, file);
    file->printf("%c", ((((i+1) % columns) == 0) ? '\n' : ' '));
  }
  if ((i % columns) != 0) {
    file->printf("\n"); 
  }
  return file->getTotalBytes();
}

/////////////////////////////////////////////////////////////////////////////

// SPARSE

template <typename MatrixType,
	  typename AsciiExtractFunctor,  typename BinaryExtractorFunctor>
SparseMatrix<MatrixType>*
readSparseMatrixFromFileStream(april_io::File *file,
                               AsciiExtractFunctor ascii_extractor,
                               BinaryExtractorFunctor bin_extractor) {
  if (!file->good()) {
    ERROR_PRINT("The stream is not prepared, it is empty, or EOF\n");
    return 0;
  }
  constString line,format,sparse,token;
  // First we read the matrix dimensions
  line = file->extract_u_line();
  if (!line) {
    ERROR_PRINT("empty file!!!\n");
    return 0;
  }
  int dims[2], n=0, NZ;
  for (int i=0; i<2; ++i, ++n) {
    if (!line.extract_int(&dims[i])) {
      ERROR_PRINT1("incorrect dimension %d type, expected a integer\n", n);
      return 0;
    }
    n++;
  }
  if (!line.extract_int(&NZ)) {
    ERROR_PRINT("impossible to read the number of non-zero elements\n");
    return 0;
  }
  SparseMatrix<MatrixType> *mat = 0;
  // Now we read the type of the format
  line = file->extract_u_line();
  format = line.extract_token();
  if (!format) {
    ERROR_PRINT("impossible to read format token\n");
    return 0;
  }
  sparse = line.extract_token();
  GPUMirroredMemoryBlock<MatrixType> *values = new GPUMirroredMemoryBlock<MatrixType>(NZ);
  Int32GPUMirroredMemoryBlock *indices = new Int32GPUMirroredMemoryBlock(NZ);
  Int32GPUMirroredMemoryBlock *first_index = 0;
  if (!sparse || sparse=="csr") {
    first_index = new Int32GPUMirroredMemoryBlock(dims[0]+1);
  }
  else if (sparse=="csc") {
    first_index = new Int32GPUMirroredMemoryBlock(dims[1]+1);
  }
  else {
    ERROR_PRINT("Impossible to determine the sparse format\n");
    return 0;
  }
  float *values_ptr = values->getPPALForWrite();
  int32_t *indices_ptr = indices->getPPALForWrite();
  int32_t *first_index_ptr = first_index->getPPALForWrite();
  if (format == "ascii") {
    int i=0;
    while(i<NZ) {
      if (! (line=file->extract_u_line()) )
        ERROR_EXIT(128, "Incorrect sparse matrix format\n");
      while(i<NZ &&
            ascii_extractor(line, values_ptr[i])) {
        ++i;
      }
    }
    i=0;
    while(i<NZ) {
      if (! (line=file->extract_u_line()) )
        ERROR_EXIT(128, "Incorrect sparse matrix format\n");
      while(i<NZ &&
            line.extract_int(&indices_ptr[i])) {
        ++i;
      }
    }
    i=0;
    while(i<static_cast<int>(first_index->getSize())) {
      if (! (line=file->extract_u_line()) )
        ERROR_EXIT(128, "Incorrect sparse matrix format\n");
      while(i<static_cast<int>(first_index->getSize()) &&
            line.extract_int(&first_index_ptr[i])) {
        ++i;
      }
    }
  } else { // binary
    int i=0;
    while(i<NZ) {
      if (! (line=file->extract_u_line()) )
        ERROR_EXIT(128, "Incorrect sparse matrix format\n");
      while(i<NZ &&
            bin_extractor(line, values_ptr[i])) {
        ++i;
      }
    }
    i=0;
    while(i<NZ) {
      if (! (line=file->extract_u_line()) )
        ERROR_EXIT(128, "Incorrect sparse matrix format\n");
      while(i<NZ &&
            line.extract_int32_binary(&indices_ptr[i])) {
        ++i;
      }
    }
    i=0;
    while(i<static_cast<int>(first_index->getSize())) {
      if (! (line=file->extract_u_line()) )
        ERROR_EXIT(128, "Incorrect sparse matrix format\n");
      while(i<static_cast<int>(first_index->getSize()) &&
            line.extract_int32_binary(&first_index_ptr[i])) {
        ++i;
      }
    }
  }
  if (sparse=="csr") {
    mat = new SparseMatrix<MatrixType>(dims[0],dims[1],
				       values,indices,first_index,
				       CSR_FORMAT);
  }
  else {
    // This was checked before: else if (sparse=="csc") {
    mat = new SparseMatrix<MatrixType>(dims[0],dims[1],
				       values,indices,first_index,
				       CSC_FORMAT);
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

     struct DummyAsciiCoder {
     // puts to the stream the given value
     void operator()(const MatrixType &value, april_io::File *file) {
     file->printf("%.5g", value);
     }
     };

     struct DummyBinaryCoder {
     // puts to the stream the given value
     void operator()(const MatrixType &value, april_io::File *file) {
     char b[5];
     binarizer::code_BLAH(value, b);
     file->printf("%s", b);
     }
     };
****************************************************************/

// Returns the number of chars written (there is a '\0' that is not counted)
template <typename MatrixType,
	  typename AsciiSizeFunctor,  typename BinarySizeFunctor,
	  typename AsciiCodeFunctor,  typename BinaryCodeFunctor>
int writeSparseMatrixToFileStream(SparseMatrix<MatrixType> *mat,
                                  april_io::File *file,
                                  AsciiSizeFunctor ascii_sizer,
                                  BinarySizeFunctor bin_sizer,
                                  AsciiCodeFunctor ascii_coder,
                                  BinaryCodeFunctor bin_coder,
                                  bool is_ascii) {
  int sizedata,sizeheader;
  sizeheader = (mat->getNumDim()+1)*10+10+10; // FIXME: To put adequate values
  // sizedata contains the memory used by MatrixType in ascii including spaces,
  // new lines, etc...
  if (is_ascii) sizedata = ascii_sizer(mat) + mat->nonZeroSize()*12 + mat->getDenseCoordinateSize()*12 + 3;
  else sizedata = bin_sizer(mat) + binarizer::buffer_size_32(mat->nonZeroSize()) + binarizer::buffer_size_32(mat->getDenseCoordinateSize()) + 3;
  file->setExpectedSize(sizedata+sizeheader+1);
  if (!file->good()) {
    ERROR_EXIT(256, "The stream is not prepared, it is empty, or EOF\n");
  }
  file->printf("%d ",mat->getDimSize(0));
  file->printf("%d ",mat->getDimSize(1));
  file->printf("%d\n",mat->nonZeroSize());
  const float *values_ptr = mat->getRawValuesAccess()->getPPALForRead();
  const int32_t *indices_ptr = mat->getRawIndicesAccess()->getPPALForRead();
  const int32_t *first_index_ptr = mat->getRawFirstIndexAccess()->getPPALForRead();
  if (is_ascii) {
    const int columns = 9;
    file->printf("ascii");
    if (mat->getSparseFormat() == CSR_FORMAT)
      file->printf(" csr");
    else
      file->printf(" csc");
    file->printf("\n");
    int i;
    for (i=0; i<mat->nonZeroSize(); ++i) {
      ascii_coder(values_ptr[i], file);
      file->printf("%c", ((((i+1) % columns) == 0) ? '\n' : ' '));
    }
    if ((i % columns) != 0) {
      file->printf("\n"); 
    }
    for (i=0; i<mat->nonZeroSize(); ++i) {
      file->printf("%d", indices_ptr[i]);
      file->printf("%c", ((((i+1) % columns) == 0) ? '\n' : ' '));
    }
    if ((i % columns) != 0) {
      file->printf("\n"); 
    }
    for (i=0; i<=mat->getDenseCoordinateSize(); ++i) {
      file->printf("%d", first_index_ptr[i]);
      file->printf("%c", ((((i+1) % columns) == 0) ? '\n' : ' '));
    }
    if ((i % columns) != 0) {
      file->printf("\n"); 
    }
  } else { // binary
    const int columns = 16;
    file->printf("binary");
    if (mat->getSparseFormat() == CSR_FORMAT)
      file->printf(" csr");
    else
      file->printf(" csc");
    file->printf("\n");
    int i=0;
    char b[5];
    for (i=0; i<mat->nonZeroSize(); ++i) {
      bin_coder(values_ptr[i], file);
      if ((i+1) % columns == 0) file->printf("\n");
    }
    if ((i % columns) != 0) file->printf("\n"); 
    for (i=0; i<mat->nonZeroSize(); ++i) {
      binarizer::code_int32(indices_ptr[i], b);
      file->printf("%c%c%c%c%c", b[0],b[1],b[2],b[3],b[4]);
      if ((i+1) % columns == 0) file->printf("\n");
    }
    if ((i % columns) != 0) file->printf("\n"); 
    for (i=0; i<=mat->getDenseCoordinateSize(); ++i) {
      binarizer::code_int32(first_index_ptr[i], b);
      file->printf("%c%c%c%c%c", b[0],b[1],b[2],b[3],b[4]);
      if ((i+1) % columns == 0) file->printf("\n");
    }
    if ((i % columns) != 0) file->printf("\n"); 
  }
  return file->getTotalBytes();
}


#endif // UTILMATRIXIO_H
