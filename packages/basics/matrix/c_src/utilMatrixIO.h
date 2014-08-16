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
#include "c_string.h"
#include "error_print.h"
#include "matrix.h"
#include "smart_ptr.h"
#include "stream.h"
#include "stream_memory.h"

extern "C" {
#include "lauxlib.h"
#include "lualib.h"
#include "lua.h"
}

namespace basics {

  april_utils::constString readULine(AprilIO::StreamInterface *stream,
                                     AprilIO::CStringStream *dest);

  /* Templates for ascii and binary extractors, sizers and coders */
  
  /**
   * Template for AsciiExtractor implementations, it needs to be specialized
   * depending in the Matrix<T> type T.
   */
  template<typename T>
  struct AsciiExtractor {
    // returns true if success, false otherwise
    bool operator()(april_utils::constString &line, T &destination) {
      UNUSED_VARIABLE(line);
      UNUSED_VARIABLE(destination);
      ERROR_EXIT(128, "NOT IMPLEMENTED\n");
      return false;
    }
  };

  /**
   * Template for BinaryExtractor implementations, it needs to be specialized
   * depending in the Matrix<T> type T.
   */
  template<typename T>
  struct BinaryExtractor {
    // returns true if success, false otherwise
    bool operator()(april_utils::constString &line, T &destination) {
      UNUSED_VARIABLE(line);
      UNUSED_VARIABLE(destination);
      ERROR_EXIT(128, "NOT IMPLEMENTED\n");
      return false;
    }
  };

  /**
   * Template for AsciiSizer implementations, it needs to be specialized
   * depending in the Matrix<T> type T.
   */
  template <typename T>
  struct AsciiSizer {
    // returns the number of bytes needed for all matrix data (plus spaces)
    int operator()(const Matrix<T> *mat) {
      UNUSED_VARIABLE(mat);
      ERROR_EXIT(128, "NOT IMPLEMENTED\n");
      return -1;
    }
  };

  /**
   * Template for BinarySizer implementations, it needs to be specialized
   * depending in the Matrix<T> type T.
   */
  template <typename T>
  struct BinarySizer {
    // returns the number of bytes needed for all matrix data (plus spaces)
    int operator()(const Matrix<T> *mat) {
      UNUSED_VARIABLE(mat);
      ERROR_EXIT(128, "NOT IMPLEMENTED\n");
      return -1;
    }
  };

  /**
   * Template for AsciiCoder implementations, it needs to be specialized
   * depending in the Matrix<T> type T.
   */
  template <typename T>
  struct AsciiCoder {
    // puts to the stream the given value
    void operator()(const T &value, AprilIO::StreamInterface *stream) {
      UNUSED_VARIABLE(value);
      UNUSED_VARIABLE(stream);
      ERROR_EXIT(128, "NOT IMPLEMENTED\n");
    }
  };

  /**
   * Template for BinaryCoder implementations, it needs to be specialized
   * depending in the Matrix<T> type T.
   */
  template <typename T>
  struct BinaryCoder {
    // puts to the stream the given value
    void operator()(const T &value, AprilIO::StreamInterface *stream) {
      UNUSED_VARIABLE(value);
      UNUSED_VARIABLE(stream);
      ERROR_EXIT(128, "NOT IMPLEMENTED\n");
    }
  };

  /****************************************************************************/
  
  /**
   * Template for SparseAsciiSizer implementations, it needs to be specialized
   * depending in the SparseMatrix<T> type T.
   */
  template <typename T>
  struct SparseAsciiSizer {
    // returns the number of bytes needed for all matrix data (plus spaces)
    int operator()(const SparseMatrix<T> *mat) {
      UNUSED_VARIABLE(mat);
      ERROR_EXIT(128, "NOT IMPLEMENTED\n");
      return -1;
    }
  };

  /**
   * Template for SparseBinarySizer implementations, it needs to be specialized
   * depending in the SparseMatrix<T> type T.
   */
  template <typename T>
  struct SparseBinarySizer {
    // returns the number of bytes needed for all matrix data (plus spaces)
    int operator()(const SparseMatrix<T> *mat) {
      UNUSED_VARIABLE(mat);
      ERROR_EXIT(128, "NOT IMPLEMENTED\n");
      return -1;
    }
  };
  
  /****************************************************************************/

  template <typename T>
  Matrix<T>*
  readMatrixFromStream(AprilIO::StreamInterface *stream,
                       const char *given_order=0) {
    AsciiExtractor<T> ascii_extractor;
    BinaryExtractor<T> bin_extractor;
    if (!stream->good()) {
      ERROR_PRINT("The stream is not prepared, it is empty, or EOF\n");
      return 0;
    }
    april_utils::SharedPtr<AprilIO::CStringStream>
      c_str(new AprilIO::CStringStream());;
    april_assert(!c_str.empty());
    april_utils::constString line,format,order,token;
    // First we read the matrix dimensions
    line = readULine(stream, c_str.get());
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
    Matrix<T> *mat = 0;
    // Now we read the type of the format
    line = readULine(stream, c_str.get());
    format = line.extract_token();
    if (!format) {
      ERROR_PRINT("impossible to read format token\n");
      return 0;
    }
    order = line.extract_token();
    if (given_order != 0) order = given_order;
    if (pos_comodin == -1) { // Normal version
      if (!order || order=="row_major")
        mat = new Matrix<T>(n,dims);
      else if (order == "col_major")
        mat = new Matrix<T>(n,dims,CblasColMajor);
      else {
        ERROR_PRINT("Impossible to determine the order\n");
        return 0;
      }
      typename Matrix<T>::iterator data_it(mat->begin());
      if (format == "ascii") {
        while (data_it!=mat->end() && (line=readULine(stream, c_str.get()))) {
          while (data_it!=mat->end() && ascii_extractor(line, *data_it)) {
            ++data_it;
          }
        }
      } else { // binary
        while (data_it!=mat->end() && (line=readULine(stream, c_str.get()))) {
          while (data_it!=mat->end() && bin_extractor(line, *data_it)) {
            ++data_it;
          }
        }
      }
      if (data_it!=mat->end()) {
        ERROR_PRINT("Impossible to fill all the matrix components\n");
        delete mat; mat = 0;
      }
    } else { // version with comodin
      int size=0,maxsize=4096;
      T *data = new T[maxsize];
      if (format == "ascii") {
        while ( (line=readULine(stream, c_str.get())) ) {
          while (ascii_extractor(line, data[size])) { 
            size++; 
            if (size == maxsize) { // resize data vector
              T *aux = new T[2*maxsize];
              for (int a=0;a<maxsize;a++) {
                aux[a] = data[a];
              }
              maxsize *= 2;
              delete[] data; data = aux;
            }
          }
        }
      } else { // binary
        while ( (line=readULine(stream, c_str.get())) ) {
          while (bin_extractor(line, data[size])) {
            size++;
            if (size == maxsize) { // resize data vector
              T *aux = new T[2*maxsize];
              for (int a=0;a<maxsize;a++) {
                aux[a] = data[a];
              }
              maxsize *= 2;
              delete[] data; data = aux;
            }
          }
        }
      }
      int sizesincomodin = 1;
      for (int i=0;i<n;i++) {
        if (i != pos_comodin) {
          sizesincomodin *= dims[i];
        }
      }
      if ((size % sizesincomodin) != 0) {
        // Error: The size of the data does not coincide
        ERROR_PRINT("data size is not valid reading a matrix with '*'\n");
        delete[] data; return 0;
      }
      dims[pos_comodin] = size / sizesincomodin;
      if (!order || order == "row_major")
        mat = new Matrix<T>(n,dims);
      else if (order == "col_major")
        mat = new Matrix<T>(n,dims,CblasColMajor);
      int i=0;
      for (typename Matrix<T>::iterator it(mat->begin());
           it!=mat->end();
           ++it,++i)
        *it = data[i];
      delete[] data;
    }
    return mat;
  }
  
  // Returns the number of chars written (there is a '\0' that is not counted)
  template <typename T>
  int writeMatrixToStream(Matrix<T> *mat,
                          AprilIO::StreamInterface *stream,
                          bool is_ascii) {
    AsciiSizer<T> ascii_sizer;
    BinarySizer<T> bin_sizer;
    AsciiCoder<T> ascii_coder;
    BinaryCoder<T> bin_coder;
    int sizedata,sizeheader;
    sizeheader = mat->getNumDim()*10+10+10; // FIXME: To put adequate values
    // sizedata contains the memory used by T in ascii including spaces,
    // new lines, etc...
    if (is_ascii) sizedata = ascii_sizer(mat);
    else sizedata = bin_sizer(mat);
    size_t expected_size = static_cast<size_t>(sizedata+sizeheader+1);
    UNUSED_VARIABLE(expected_size);
    if (!stream->isOpened()) {
      ERROR_EXIT(256, "The stream is not prepared\n");
    }
    for (int i=0;i<mat->getNumDim()-1;i++) {
      stream->printf("%d ",mat->getDimSize(i));
    }
    stream->printf("%d\n",mat->getDimSize(mat->getNumDim()-1));
    if (is_ascii) {
      const int columns = 9;
      stream->printf("ascii");
      if (mat->getMajorOrder() == CblasColMajor) {
        stream->printf(" col_major");
      }
      else {
        stream->printf(" row_major");
      }
      stream->printf("\n");
      int i=0;
      for(typename Matrix<T>::const_iterator it(mat->begin());
          it!=mat->end();++it,++i) {
        ascii_coder(*it, stream);
        stream->printf("%c", ((((i+1) % columns) == 0) ? '\n' : ' '));
      }
      if ((i % columns) != 0) {
        stream->printf("\n"); 
      }
    } else { // binary
      const int columns = 16;
      stream->printf("binary");
      if (mat->getMajorOrder() == CblasColMajor) {
        stream->printf(" col_major");
      }
      else {
        stream->printf(" row_major");
      }
      stream->printf("\n");
      // We substract 1 so the final '\0' is not considered
      int i=0;
      for(typename Matrix<T>::const_iterator it(mat->begin());
          it!=mat->end();
          ++it,++i) {
        bin_coder(*it, stream);
        /*
          char b[5];
          binarizer::code_float(*it, b);
          fprintf(f, "%c%c%c%c%c", b[0], b[1], b[2], b[3], b[4]);
        */
        if ((i+1) % columns == 0) stream->printf("\n");
      }
      if ((i % columns) != 0) stream->printf("\n"); 
    }
    return static_cast<int>(stream->seek());
  }

  /****************************************************************/

  template <typename T>
  Matrix<T>*
  readMatrixFromTabStream(AprilIO::StreamInterface *stream,
                          const char *given_order=0) {
    AsciiExtractor<T> ascii_extractor;
    if (!stream->good()) {
      ERROR_PRINT("The stream is not prepared, it is empty, or EOF\n");
      return 0;
    }
    april_utils::SharedPtr<AprilIO::CStringStream> c_str;
    c_str = new AprilIO::CStringStream();
    april_assert(!c_str.empty());
    april_utils::constString line("");
    T value;
    int ncols = 0, nrows = 0;
    while (!stream->eof()) {
      line = readULine(stream, c_str.get());
      if (line.len() > 0) {
        if (ncols == 0) {
          while(ascii_extractor(line,value)) ++ncols;
        }
        ++nrows;
      }
    }
    if (nrows <= 0 || ncols <= 0) ERROR_EXIT(256, "Found 0 rows or 0 cols\n");
    stream->seek(SEEK_SET, 0);
    if (stream->hasError()) {
      ERROR_PRINT1("Impossible to rewind the stream: %s\n",
                   stream->getErrorMsg());
      return 0;
    }
    if (!stream->good()) {
      ERROR_PRINT("The stream is not prepared, it is empty, or EOF\n");
      return 0;
    }
    april_utils::constString order( (given_order) ? given_order : "row_major"),token;
    int dims[2] = { nrows, ncols };
    Matrix<T> *mat = 0;
    if (order=="row_major") {
      mat = new Matrix<T>(2,dims);
    }
    else if (order == "col_major") {
      mat = new Matrix<T>(2,dims,CblasColMajor);
    }
    else {
      ERROR_PRINT("Impossible to determine the order\n");
      return 0;
    }
    int i=0;
    typename Matrix<T>::iterator data_it(mat->begin());
    while (data_it!=mat->end() && (line=readULine(stream, c_str.get()))) {
      int num_cols_size_count = 0;
      while (data_it!=mat->end() &&
             ascii_extractor(line, *data_it)) {
        ++data_it;
        ++num_cols_size_count;
      }
      if (num_cols_size_count != ncols) {
        ERROR_EXIT3(128, "Incorrect number of elements at line %d, "
                    "expected %d, found %d\n", i, ncols, num_cols_size_count);
      }
      ++i;
    }
    if (data_it!=mat->end()) {
      ERROR_PRINT("Impossible to fill all the matrix components\n");
      delete mat; mat = 0;
    }
    return mat;
  }
  
  // Returns the number of chars written (there is a '\0' that is not counted)
  template <typename T>
  int writeMatrixToTabStream(Matrix<T> *mat,
                             AprilIO::StreamInterface *stream) {
    AsciiSizer<T> ascii_sizer;
    AsciiCoder<T> ascii_coder;
    if (mat->getNumDim() != 2) {
      ERROR_EXIT(128, "Needs a matrix with 2 dimensions");
    }
    //
    int sizedata;
    sizedata = ascii_sizer(mat);
    size_t expected_size = static_cast<size_t>(sizedata+1);
    UNUSED_VARIABLE(expected_size);
    if (!stream->isOpened()) {
      ERROR_EXIT(256, "The stream is not prepared\n");
    }
    const int columns = mat->getDimSize(1);
    int i=0;
    for(typename Matrix<T>::const_iterator it(mat->begin());
        it!=mat->end();++it,++i) {
      ascii_coder(*it, stream);
      stream->printf("%c", ((((i+1) % columns) == 0) ? '\n' : ' '));
    }
    if ((i % columns) != 0) {
      stream->printf("\n"); 
    }
    return static_cast<int>(stream->seek());
  }

  /////////////////////////////////////////////////////////////////////////////

  // SPARSE

  template <typename T>
  SparseMatrix<T>*
  readSparseMatrixFromStream(AprilIO::StreamInterface *stream) {
    AsciiExtractor<T> ascii_extractor;
    BinaryExtractor<T> bin_extractor;
    if (!stream->good()) {
      ERROR_PRINT("The stream is not prepared, it is empty, or EOF\n");
      return 0;
    }
    april_utils::SharedPtr<AprilIO::CStringStream>
      c_str(new AprilIO::CStringStream());
    april_assert(!c_str.empty());
    april_utils::constString line,format,sparse,token;
    // First we read the matrix dimensions
    line = readULine(stream, c_str.get());
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
    SparseMatrix<T> *mat = 0;
    // Now we read the type of the format
    line = readULine(stream, c_str.get());
    format = line.extract_token();
    if (!format) {
      ERROR_PRINT("impossible to read format token\n");
      return 0;
    }
    sparse = line.extract_token();
    april_math::GPUMirroredMemoryBlock<T> *values = new april_math::GPUMirroredMemoryBlock<T>(NZ);
    april_math::Int32GPUMirroredMemoryBlock *indices = new april_math::Int32GPUMirroredMemoryBlock(NZ);
    april_math::Int32GPUMirroredMemoryBlock *first_index = 0;
    if (!sparse || sparse=="csr") {
      first_index = new april_math::Int32GPUMirroredMemoryBlock(dims[0]+1);
    }
    else if (sparse=="csc") {
      first_index = new april_math::Int32GPUMirroredMemoryBlock(dims[1]+1);
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
        if (! (line=readULine(stream, c_str.get())) ) {
          ERROR_EXIT(128, "Incorrect sparse matrix format\n");
        }
        while(i<NZ &&
              ascii_extractor(line, values_ptr[i])) {
          ++i;
        }
      }
      i=0;
      while(i<NZ) {
        if (! (line=readULine(stream, c_str.get())) ) {
          ERROR_EXIT(128, "Incorrect sparse matrix format\n");
        }
        while(i<NZ &&
              line.extract_int(&indices_ptr[i])) {
          ++i;
        }
      }
      i=0;
      while(i<static_cast<int>(first_index->getSize())) {
        if (! (line=readULine(stream, c_str.get())) ) {
          ERROR_EXIT(128, "Incorrect sparse matrix format\n");
        }
        while(i<static_cast<int>(first_index->getSize()) &&
              line.extract_int(&first_index_ptr[i])) {
          ++i;
        }
      }
    } else { // binary
      int i=0;
      while(i<NZ) {
        if (! (line=readULine(stream, c_str.get())) ) {
          ERROR_EXIT(128, "Incorrect sparse matrix format\n");
        }
        while(i<NZ &&
              bin_extractor(line, values_ptr[i])) {
          ++i;
        }
      }
      i=0;
      while(i<NZ) {
        if (! (line=readULine(stream, c_str.get())) ) {
          ERROR_EXIT(128, "Incorrect sparse matrix format\n");
        }
        while(i<NZ &&
              line.extract_int32_binary(&indices_ptr[i])) {
          ++i;
        }
      }
      i=0;
      while(i<static_cast<int>(first_index->getSize())) {
        if (! (line=readULine(stream, c_str.get())) ) {
          ERROR_EXIT(128, "Incorrect sparse matrix format\n");
        }
        while(i<static_cast<int>(first_index->getSize()) &&
              line.extract_int32_binary(&first_index_ptr[i])) {
          ++i;
        }
      }
    }
    if (sparse=="csr") {
      mat = new SparseMatrix<T>(dims[0],dims[1],
                                values,indices,first_index,
                                CSR_FORMAT);
    }
    else {
      // This was checked before: else if (sparse=="csc") {
      mat = new SparseMatrix<T>(dims[0],dims[1],
                                values,indices,first_index,
                                CSC_FORMAT);
    }
    return mat;
  }
  
  // Returns the number of chars written (there is a '\0' that is not counted)
  template <typename T>
  int writeSparseMatrixToStream(SparseMatrix<T> *mat,
                                AprilIO::StreamInterface *stream,
                                bool is_ascii) {
    SparseAsciiSizer<T> ascii_sizer;
    SparseBinarySizer<T> bin_sizer;
    AsciiCoder<T> ascii_coder;
    BinaryCoder<T> bin_coder;
    int sizedata,sizeheader;
    sizeheader = (mat->getNumDim()+1)*10+10+10; // FIXME: To put adequate values
    // sizedata contains the memory used by T in ascii including spaces,
    // new lines, etc...
    if (is_ascii) {
      sizedata = ascii_sizer(mat) +
        mat->nonZeroSize()*12 +
        mat->getDenseCoordinateSize()*12 + 3;
    }
    else {
      sizedata = bin_sizer(mat) +
        april_utils::binarizer::buffer_size_32(mat->nonZeroSize()) +
        april_utils::binarizer::buffer_size_32(mat->getDenseCoordinateSize()) +
        3;
    }
    size_t expected_size = static_cast<size_t>(sizedata+sizeheader+1);
    UNUSED_VARIABLE(expected_size);
    if (!stream->isOpened()) {
      ERROR_EXIT(256, "The stream is not prepared\n");
    }
    stream->printf("%d ",mat->getDimSize(0));
    stream->printf("%d ",mat->getDimSize(1));
    stream->printf("%d\n",mat->nonZeroSize());
    const float *values_ptr = mat->getRawValuesAccess()->getPPALForRead();
    const int32_t *indices_ptr = mat->getRawIndicesAccess()->getPPALForRead();
    const int32_t *first_index_ptr = mat->getRawFirstIndexAccess()->getPPALForRead();
    if (is_ascii) {
      const int columns = 9;
      stream->printf("ascii");
      if (mat->getSparseFormat() == CSR_FORMAT) stream->printf(" csr");
      else stream->printf(" csc");
      stream->printf("\n");
      int i;
      for (i=0; i<mat->nonZeroSize(); ++i) {
        ascii_coder(values_ptr[i], stream);
        stream->printf("%c", ((((i+1) % columns) == 0) ? '\n' : ' '));
      }
      if ((i % columns) != 0) {
        stream->printf("\n"); 
      }
      for (i=0; i<mat->nonZeroSize(); ++i) {
        stream->printf("%d", indices_ptr[i]);
        stream->printf("%c", ((((i+1) % columns) == 0) ? '\n' : ' '));
      }
      if ((i % columns) != 0) {
        stream->printf("\n"); 
      }
      for (i=0; i<=mat->getDenseCoordinateSize(); ++i) {
        stream->printf("%d", first_index_ptr[i]);
        stream->printf("%c", ((((i+1) % columns) == 0) ? '\n' : ' '));
      }
      if ((i % columns) != 0) stream->printf("\n"); 
    } else { // binary
      const int columns = 16;
      stream->printf("binary");
      if (mat->getSparseFormat() == CSR_FORMAT)
        stream->printf(" csr");
      else
        stream->printf(" csc");
      stream->printf("\n");
      int i=0;
      char b[5];
      for (i=0; i<mat->nonZeroSize(); ++i) {
        bin_coder(values_ptr[i], stream);
        if ((i+1) % columns == 0) stream->printf("\n");
      }
      if ((i % columns) != 0) stream->printf("\n"); 
      for (i=0; i<mat->nonZeroSize(); ++i) {
        april_utils::binarizer::code_int32(indices_ptr[i], b);
        stream->printf("%c%c%c%c%c", b[0],b[1],b[2],b[3],b[4]);
        if ((i+1) % columns == 0) stream->printf("\n");
      }
      if ((i % columns) != 0) stream->printf("\n"); 
      for (i=0; i<=mat->getDenseCoordinateSize(); ++i) {
        april_utils::binarizer::code_int32(first_index_ptr[i], b);
        stream->printf("%c%c%c%c%c", b[0],b[1],b[2],b[3],b[4]);
        if ((i+1) % columns == 0) stream->printf("\n");
      }
      if ((i % columns) != 0) stream->printf("\n"); 
    }
    return static_cast<int>(stream->seek());
  }

} // namespace basics

#include "utilMatrixFloat.h"
#include "utilMatrixDouble.h"
#include "utilMatrixComplexF.h"
#include "utilMatrixInt32.h"
#include "utilMatrixChar.h"

#endif // UTILMATRIXIO_H
